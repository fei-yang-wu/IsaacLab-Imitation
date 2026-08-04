#!/usr/bin/env python3
"""ICE container entrypoint for the LAFAN1 planner-capacity sweeps.

A submitted ICE job runs exactly one python file via /isaac-sim/python.sh, so this
thin entry wraps the bash orchestrators. Three stages, chained with afterok:

  --stage oracle     : prepare_oracle_baselines.sh (6 Isaac runs, once)
  --stage cell       : run_capacity_point.sh for SLURM_ARRAY_TASK_ID (0-11)
  --stage aggregate  : per-seed + across-seed aggregation (pure python)
  --stage enc380     : walk1 shared-tracker route diagnostic (12-cell grid)
  --stage enc380_h30 : reuse enc380 H10 rows for the H30 temporal diagnostic
  --stage pure_root_planner : train the direct root_qpos-tracker planner
  --stage auxiliary_capacity : train missing H30/pure-root seed-0 size cells

Inside the container pixi is unavailable, so ISAAC_PY/PLAIN_PY are pointed at
/isaac-sim/python.sh. The frozen oracles are Newton-trained on a compute-only GPU,
so ASSERT_KITLESS=1 and the Newton solver args are forwarded by the bash scripts.
Data lives under the container /data bind; walk1_subject1 is restricted via
--motion_name against the full corrected manifest.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# The cluster runner executes this file by path (CLUSTER_PYTHON_EXECUTABLE)
# with an interpreter that may not have the package installed; parents[2] is
# the package's own install root, which is stable by construction.
try:
    import imitation_experiments  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from imitation_experiments.capacity.enc380_capacity_grid import decode_route_task
from imitation_experiments.paths import REPO_ROOT

# The campaign-owned shell stages stay at their frozen campaign path.
CAMPAIGN_DIR = REPO_ROOT / "experiments/campaigns/2026-07-23-lafan1-planner-capacity"

SIZES = ("tiny", "small", "medium", "large")
# Seeds 0-2 are the original grid (array 0-11); 3-5 were added to firm up the
# high-variance cells -- latent/tiny is bimodal at n=3 ([79, 82, 235]), which
# is what the iso-performance claim hinges on. Array 12-23 covers seeds 3-5.
SEEDS = ("0", "1", "2", "3", "4", "5")

# ICE container-side data (see wiki: the corrected 40-motion tree under /data).
ICE_MANIFEST = "/data/lafan1_corrected_8e95d557/manifests/g1_lafan1_manifest.json"
ICE_LATENT_DATASET = "/data/lafan1_corrected_8e95d557/g1_hl_diffsr"

STUDY_ROOT = "logs/interface_baselines/lafan1_planner_capacity_20260723"
ORACLE_ROOT = f"{STUDY_ROOT}/oracle_baselines"


def _inject_cu130_runtime_libs(env: dict[str, str], runtime_root: Path) -> None:
    """Mirror run_singularity.sh's PhysX-path LD setup so the CU130 runtime torch
    finds its bundled NCCL/CUDA libs (else libtorch_cuda.so fails with
    'undefined symbol: ncclDevCommDestroy'). The Newton path skips this, so a
    custom entry that imports torch must do it itself."""
    sites = sorted(runtime_root.glob("lib/python*/site-packages"))
    site = next((s for s in sites if (s / "torch").is_dir()), None)
    if site is None:
        return
    env["ISAACLAB_CU130_SITE_PACKAGES"] = str(site)
    nvidia = site / "nvidia"
    lib_dirs = sorted(
        {str(p) for p in nvidia.glob("*/lib") if p.is_dir()}
        | {str(p) for p in nvidia.glob("*/*/lib") if p.is_dir()}
    )
    if lib_dirs:
        prev = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join(lib_dirs) + (f":{prev}" if prev else "")
    nccl = nvidia / "nccl" / "lib" / "libnccl.so.2"
    if nccl.is_file():
        prev = env.get("LD_PRELOAD", "")
        env["LD_PRELOAD"] = str(nccl) + (f":{prev}" if prev else "")


def _runtime_env() -> dict[str, str]:
    env = dict(os.environ)
    # CU130 split runtime: torch + isaaclab live in the container-runtime pixi env,
    # NOT in /isaac-sim/python.sh. train.py's Newton path execs this python, and it
    # can launch AppLauncher headless (kitless Newton). Mirror that for every eval.
    runtime_root = env.get("ISAACLAB_CU130_RUNTIME_ROOT", "")
    candidates = [
        f"{runtime_root}/bin/python" if runtime_root else "",
        "/opt/isaaclab-imitation-runtime-spec/.pixi/envs/container-runtime/bin/python",
    ]
    runtime_python = next((c for c in candidates if c and Path(c).exists()), None)
    if runtime_python is not None:
        env["ISAAC_PY"] = runtime_python
        env["PLAIN_PY"] = runtime_python
        env["ASSERT_KITLESS"] = "1"  # latent eval validates kitless Newton
        _inject_cu130_runtime_libs(env, Path(runtime_python).parents[1])
    # Central split-runtime fix: /isaac-sim/python.sh exports a PYTHONPATH that
    # carries Kit's stdlib (/isaac-sim/kit/python/lib/python3.12), whose
    # conda-forge-tagged platform.py crashes the CU130 runtime python
    # ("failed to parse CPython sys.version"). Scrub the Kit STDLIB entries from
    # PYTHONPATH for every subprocess (train/merge/aggregate/eval) while keeping
    # Kit site-packages (lazy_loader etc. needed by the Kit eval paths).
    pp = env.get("PYTHONPATH", "")
    if pp:
        kept = [
            p
            for p in pp.split(os.pathsep)
            if not (
                os.path.realpath(p or ".").startswith("/isaac-sim/kit/python")
                and "site-packages" not in os.path.realpath(p or ".")
            )
        ]
        env["PYTHONPATH"] = os.pathsep.join(kept)
    env.setdefault("DEVICE", "cuda:0")
    # Newton-trained oracles + corrected-tree data on the container bind.
    env["MANIFEST"] = env.get("MANIFEST", ICE_MANIFEST)
    env["LATENT_DATASET_PATH"] = env.get("LATENT_DATASET_PATH", ICE_LATENT_DATASET)
    env["DATASET_PATH"] = env.get("DATASET_PATH", ICE_LATENT_DATASET)
    env["STUDY_ROOT"] = STUDY_ROOT
    env["ORACLE_ROOT"] = ORACLE_ROOT
    # EE needs the ee-chunk env adapter; restrict to the two working interfaces
    # (streamed_vanilla is full-body only).
    env["INTERFACES"] = env.get("INTERFACES", "latent_skill full_body_trajectory")
    return env


def _bash(script: str, env: dict[str, str]) -> int:
    path = str(CAMPAIGN_DIR / script)
    print(f"[entry] bash {path}", flush=True)
    return subprocess.run(["bash", path], cwd=str(REPO_ROOT), env=env).returncode


def _aggregate(env: dict[str, str]) -> int:
    py = env.get("PLAIN_PY", "pixi run python").split()
    seed_inputs = []
    for seed in SEEDS:
        seed_root = f"{STUDY_ROOT}/scaling/seed{seed}"
        out = f"{seed_root}/capacity_summary"
        cmd = py + [
            "-m",
            "imitation_experiments.capacity.aggregate_one_motion_capacity_scaling",
            "--scaling_root",
            seed_root,
            "--sizes",
            *SIZES,
            "--output_dir",
            out,
            "--overwrite",
        ]
        for iface in env["INTERFACES"].split():
            cmd += [
                "--oracle",
                f"{iface}={ORACLE_ROOT}/{iface}/oracle_frame0_700/summary.json",
            ]
        rc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env).returncode
        if rc != 0:
            return rc
        seed_inputs.append(f"{out}/capacity_results.json")
    cmd = py + [
        "-m",
        "imitation_experiments.capacity.aggregate_one_motion_capacity_seeds",
    ]
    for path in seed_inputs:
        cmd += ["--input", path]
    cmd += [
        "--min_seeds",
        str(len(SEEDS)),
        "--survival_target",
        "1.0",
        "--normalized_mpjpe_target",
        "1.5",
        "--output_dir",
        f"{STUDY_ROOT}/capacity_seeds_summary",
    ]
    return subprocess.run(cmd, cwd=str(REPO_ROOT), env=env).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "oracle",
            "cell",
            "aggregate",
            "finetune_b",
            "enc380",
            "enc380_h30",
            "pure_root_planner",
            "auxiliary_capacity",
            "fb670_curve",
            "latent_curve",
            "fb670_via_latent_tracker",
        ),
    )
    parser.add_argument("--enc380-low-level-checkpoint", default="")
    parser.add_argument("--enc380-skill-checkpoint", default="")
    parser.add_argument("--enc380-completion-record", default="")
    parser.add_argument(
        "--enc380-mode",
        choices=("qualify", "demo", "cell", "root_recovery", "aggregate"),
        default="qualify",
    )
    parser.add_argument(
        "--enc380-stages",
        default="qualify demo train eval aggregate",
    )
    parser.add_argument(
        "--enc380-output-root",
        default="logs/interface_baselines/lafan1_enc380_route_comparison",
    )
    parser.add_argument("--enc380-low-level-sha256", default="")
    parser.add_argument("--enc380-skill-sha256", default="")
    parser.add_argument(
        "--enc380-source-study-root",
        default="",
        help="Completed H10 study whose paired rows and H10 baseline are reused.",
    )
    parser.add_argument(
        "--enc380-h30-output-root",
        default="logs/interface_baselines/lafan1_enc380_h30_temporal_screen",
    )
    parser.add_argument(
        "--enc380-h30-mode",
        choices=("screen", "aggregate"),
        default="screen",
    )
    parser.add_argument("--pure-root-samples-dir", default="")
    parser.add_argument("--pure-root-output-dir", default="")
    parser.add_argument(
        "--pure-root-output-root",
        default=(
            "logs/interface_baselines/"
            "lafan1_enc380_route_capacity_5b_oracle100_progressive_b1024_20260730/"
            "pure_root_qpos_tracker"
        ),
    )
    parser.add_argument("--pure-root-sample-sha256", default="")
    parser.add_argument(
        "--fb670-mode",
        choices=("collect", "train", "eval", "aggregate"),
        default="collect",
    )
    parser.add_argument(
        "--fb670-output-root",
        default="logs/interface_baselines/lafan1_fb670_budget_curve_20260730",
    )
    parser.add_argument(
        "--fb670-checkpoint",
        default=(
            "logs/downloaded_checkpoints/lafan1_fbchunk_5b_seed0/"
            "model_step_5000085504.pt"
        ),
    )
    parser.add_argument("--fb670-checkpoint-sha256", default="")
    parser.add_argument(
        "--latent-curve-mode",
        choices=("collect", "train", "eval", "aggregate"),
        default="collect",
    )
    parser.add_argument(
        "--latent-curve-output-root",
        default="logs/interface_baselines/lafan1_latent_budget_curve_20260730",
    )
    parser.add_argument(
        "--latent-curve-checkpoint",
        default=(
            "logs/downloaded_checkpoints/lafan1_latent_deterministic_5b_seed0/"
            "model_step_4525129728.pt"
        ),
    )
    parser.add_argument(
        "--latent-curve-skill-checkpoint",
        default=(
            "logs/downloaded_checkpoints/lafan1_latent_deterministic_5b_seed0/"
            "skill_encoder/latest.pt"
        ),
    )
    parser.add_argument("--latent-curve-checkpoint-sha256", default="")
    parser.add_argument("--latent-curve-skill-sha256", default="")
    parser.add_argument(
        "--fb670-route-mode",
        choices=("pin", "eval", "aggregate"),
        default="pin",
    )
    parser.add_argument(
        "--fb670-route-output-root",
        default="logs/interface_baselines/lafan1_fb670_via_latent_tracker_20260730",
    )
    parser.add_argument(
        "--fb670-route-train-root",
        default="logs/interface_baselines/lafan1_fb670_budget_curve_20260730",
    )
    parser.add_argument(
        "--fb670-route-latent-checkpoint",
        default=(
            "logs/downloaded_checkpoints/lafan1_latent_deterministic_5b_seed0/"
            "model_step_4525129728.pt"
        ),
    )
    parser.add_argument(
        "--fb670-route-skill-checkpoint",
        default=(
            "logs/downloaded_checkpoints/lafan1_latent_deterministic_5b_seed0/"
            "skill_encoder/latest.pt"
        ),
    )
    args, _ = parser.parse_known_args()
    env = _runtime_env()

    if args.stage == "fb670_via_latent_tracker":
        env["STUDY_ROOT"] = args.fb670_route_output_root
        env["FB670_STUDY_ROOT"] = args.fb670_route_train_root
        env["LATENT_LOW_LEVEL_CHECKPOINT"] = args.fb670_route_latent_checkpoint
        env["LATENT_SKILL_CHECKPOINT"] = args.fb670_route_skill_checkpoint
        if args.fb670_route_mode in ("pin", "eval"):
            idx = int(
                os.environ.get(
                    "SLURM_ARRAY_TASK_ID", os.environ.get("CELL_INDEX", "0")
                )
            )
            sizes = ("medium", "large")
            if not 0 <= idx < len(sizes):
                parser.error("fb670_via_latent_tracker pin/eval index must be 0-1.")
            env["MODEL_SIZE"] = sizes[idx]
        env["STAGES"] = args.fb670_route_mode
        print(
            f"[entry] fb670_via_latent_tracker mode={args.fb670_route_mode} "
            f"size={env.get('MODEL_SIZE', '-')} "
            f"root={args.fb670_route_output_root}",
            flush=True,
        )
        return _bash("run_fb670_via_latent_tracker_curve.sh", env)

    if args.stage == "latent_curve":
        env["STUDY_ROOT"] = args.latent_curve_output_root
        env["LATENT_LOW_LEVEL_CHECKPOINT"] = args.latent_curve_checkpoint
        env["LATENT_SKILL_CHECKPOINT"] = args.latent_curve_skill_checkpoint
        if args.latent_curve_checkpoint_sha256:
            env["EXPECTED_LATENT_SHA256"] = args.latent_curve_checkpoint_sha256
        if args.latent_curve_skill_sha256:
            env["EXPECTED_SKILL_SHA256"] = args.latent_curve_skill_sha256
        if args.latent_curve_mode in ("train", "eval"):
            idx = int(
                os.environ.get(
                    "SLURM_ARRAY_TASK_ID", os.environ.get("CELL_INDEX", "0")
                )
            )
            sizes = ("medium", "large")
            if not 0 <= idx < len(sizes):
                parser.error("latent_curve train/eval array index must be 0-1.")
            env["MODEL_SIZE"] = sizes[idx]
        env["STAGES"] = args.latent_curve_mode
        print(
            f"[entry] latent_curve mode={args.latent_curve_mode} "
            f"size={env.get('MODEL_SIZE', '-')} "
            f"root={args.latent_curve_output_root}",
            flush=True,
        )
        return _bash("run_latent_budget_curve.sh", env)

    if args.stage == "fb670_curve":
        env["STUDY_ROOT"] = args.fb670_output_root
        env["FB_LOW_LEVEL_CHECKPOINT"] = args.fb670_checkpoint
        if args.fb670_checkpoint_sha256:
            env["EXPECTED_FB_SHA256"] = args.fb670_checkpoint_sha256
        if args.fb670_mode in ("train", "eval"):
            idx = int(
                os.environ.get(
                    "SLURM_ARRAY_TASK_ID", os.environ.get("CELL_INDEX", "0")
                )
            )
            sizes = ("medium", "large")
            if not 0 <= idx < len(sizes):
                parser.error("fb670_curve train/eval array index must be 0-1.")
            env["MODEL_SIZE"] = sizes[idx]
        env["STAGES"] = args.fb670_mode
        print(
            f"[entry] fb670_curve mode={args.fb670_mode} "
            f"size={env.get('MODEL_SIZE', '-')} root={args.fb670_output_root}",
            flush=True,
        )
        return _bash("run_fb670_budget_curve.sh", env)

    if args.stage == "auxiliary_capacity":
        idx = int(
            os.environ.get("SLURM_ARRAY_TASK_ID", os.environ.get("CELL_INDEX", "0"))
        )
        sizes = ("tiny", "small", "large")
        if not 0 <= idx < 2 * len(sizes):
            parser.error("auxiliary_capacity index must be in 0-5.")
        route_index, size_index = divmod(idx, len(sizes))
        size = sizes[size_index]
        updates = {"tiny": 10_000, "small": 20_000, "large": 50_000}[size]
        env["MODEL_SIZE"] = size
        if route_index == 0:
            if (
                not args.enc380_low_level_checkpoint
                or not args.enc380_skill_checkpoint
                or not args.enc380_completion_record
                or not args.enc380_source_study_root
            ):
                parser.error(
                    "H30 auxiliary capacity requires low-level, skill, "
                    "completion-record, and source-study-root paths."
                )
            env.update(
                {
                    "LOW_LEVEL_CHECKPOINT": args.enc380_low_level_checkpoint,
                    "SKILL_CHECKPOINT": args.enc380_skill_checkpoint,
                    "TRACKER_COMPLETION_RECORD": args.enc380_completion_record,
                    "EXPECTED_LOW_LEVEL_SHA256": args.enc380_low_level_sha256,
                    "EXPECTED_SKILL_SHA256": args.enc380_skill_sha256,
                    "SOURCE_STUDY_ROOT": args.enc380_source_study_root,
                    "OUTPUT_ROOT": args.enc380_h30_output_root,
                    "STAGES": "train",
                    "DRY_RUN": "0",
                    "ASSERT_KITLESS": "0",
                    "RENDER_VIDEO": "0",
                }
            )
            print(
                f"[entry] auxiliary_capacity route=h30 size={size} "
                f"updates={updates} output={args.enc380_h30_output_root}",
                flush=True,
            )
            return _bash("run_enc380_h30_temporal_ensemble.sh", env)
        if not args.pure_root_samples_dir:
            parser.error(
                "Pure-root auxiliary capacity requires --pure-root-samples-dir."
            )
        output_dir = (
            f"{args.pure_root_output_root}/{size}/seed0/"
            f"planner_oracle_u{updates}_b1024"
        )
        env.update(
            {
                "SAMPLES_DIR": args.pure_root_samples_dir,
                "OUTPUT_DIR": output_dir,
                "EXPECTED_SAMPLE_SHA256": args.pure_root_sample_sha256,
            }
        )
        print(
            f"[entry] auxiliary_capacity route=pure_root size={size} "
            f"updates={updates} output={output_dir}",
            flush=True,
        )
        return _bash("run_pure_root_qpos_planner.sh", env)

    if args.stage == "oracle":
        return _bash("prepare_oracle_baselines.sh", env)
    if args.stage == "aggregate":
        return _aggregate(env)
    if args.stage == "pure_root_planner":
        if not args.pure_root_samples_dir or not args.pure_root_output_dir:
            parser.error(
                "--stage pure_root_planner requires --pure-root-samples-dir and "
                "--pure-root-output-dir."
            )
        env.update(
            {
                "SAMPLES_DIR": args.pure_root_samples_dir,
                "OUTPUT_DIR": args.pure_root_output_dir,
                "EXPECTED_SAMPLE_SHA256": args.pure_root_sample_sha256,
            }
        )
        return _bash("run_pure_root_qpos_planner.sh", env)
    if args.stage == "enc380_h30":
        if (
            not args.enc380_low_level_checkpoint
            or not args.enc380_skill_checkpoint
            or not args.enc380_completion_record
            or not args.enc380_source_study_root
        ):
            parser.error(
                "--stage enc380_h30 requires low-level, skill, completion-record, "
                "and source-study-root paths."
            )
        env.update(
            {
                "LOW_LEVEL_CHECKPOINT": args.enc380_low_level_checkpoint,
                "SKILL_CHECKPOINT": args.enc380_skill_checkpoint,
                "TRACKER_COMPLETION_RECORD": args.enc380_completion_record,
                "EXPECTED_LOW_LEVEL_SHA256": args.enc380_low_level_sha256,
                "EXPECTED_SKILL_SHA256": args.enc380_skill_sha256,
                "SOURCE_STUDY_ROOT": args.enc380_source_study_root,
                "OUTPUT_ROOT": args.enc380_h30_output_root,
                "STAGES": (
                    "materialize train eval"
                    if args.enc380_h30_mode == "screen"
                    else "aggregate"
                ),
                "DRY_RUN": "0",
                "ASSERT_KITLESS": "0",
                "RENDER_VIDEO": "1",
            }
        )
        print(
            f"[entry] enc380_h30 mode={args.enc380_h30_mode} "
            f"source={args.enc380_source_study_root} "
            f"output={args.enc380_h30_output_root}",
            flush=True,
        )
        return _bash("run_enc380_h30_temporal_ensemble.sh", env)
    if args.stage == "enc380":
        if (
            not args.enc380_low_level_checkpoint
            or not args.enc380_skill_checkpoint
            or not args.enc380_completion_record
        ):
            parser.error(
                "--stage enc380 requires low-level, skill, and completion-record paths."
            )
        mode = args.enc380_mode
        idx = int(
            os.environ.get("SLURM_ARRAY_TASK_ID", os.environ.get("CELL_INDEX", "0"))
        )
        stages_by_mode = {
            "qualify": "qualify",
            "demo": "demo",
            "cell": "train eval",
            "root_recovery": "train eval",
            "aggregate": "aggregate",
        }
        env.update(
            {
                "LOW_LEVEL_CHECKPOINT": args.enc380_low_level_checkpoint,
                "SKILL_CHECKPOINT": args.enc380_skill_checkpoint,
                "TRACKER_COMPLETION_RECORD": args.enc380_completion_record,
                "EXPECTED_LOW_LEVEL_SHA256": args.enc380_low_level_sha256,
                "EXPECTED_SKILL_SHA256": args.enc380_skill_sha256,
                "STAGES": stages_by_mode[mode],
                "OUTPUT_ROOT": args.enc380_output_root,
                "DRY_RUN": "0",
                "ASSERT_KITLESS": "0",
                "RENDER_VIDEO": "1",
            }
        )
        if mode in {"cell", "root_recovery"}:
            route_task_index = idx if mode == "cell" else idx * 2
            try:
                motion, size, seed, route = decode_route_task(route_task_index)
            except ValueError as error:
                parser.error(str(error))
            env.update(
                {
                    "MOTION_NAME": motion,
                    "MODEL_SIZE": size,
                    "SEED": str(seed),
                    "ROUTES": route,
                }
            )
        print(
            f"[entry] enc380 mode={mode} idx={idx} "
            f"motion={env.get('MOTION_NAME', 'all40')} "
            f"size={env.get('MODEL_SIZE', 'all')} seed={env.get('SEED', 'all')} "
            f"route={env.get('ROUTES', 'both')}",
            flush=True,
        )
        return _bash("run_enc380_planner_route_comparison.sh", env)
    if args.stage == "finetune_b":
        # Finetune ablation: oracle-driven aggregation instead of DAgger.
        # One array index per planner seed; size/interface come from the env.
        idx = int(
            os.environ.get("SLURM_ARRAY_TASK_ID", os.environ.get("CELL_INDEX", "0"))
        )
        # Same (size, seed) grid as --stage cell so one array covers the whole
        # capacity table: idx 0-11 -> size = SIZES[idx % 4], seed = SEEDS[idx // 4].
        if not 0 <= idx < len(SIZES) * len(SEEDS):
            print(f"[entry] finetune_b index {idx} out of range", file=sys.stderr)
            return 2
        env["SEEDS"] = SEEDS[idx // len(SIZES)]
        env["MODEL_SIZE"] = SIZES[idx % len(SIZES)]
        # _runtime_env() pre-sets INTERFACES for the main cell stage, so this
        # must OVERRIDE rather than setdefault: method B covers the chunk
        # interfaces only and aborts on latent_skill.
        env["INTERFACES"] = os.environ.get(
            "METHOD_B_INTERFACES", "full_body_trajectory"
        )
        print(
            f"[entry] finetune_b idx={idx} -> seed={env['SEEDS']} "
            f"size={env['MODEL_SIZE']} interfaces={env['INTERFACES']}",
            flush=True,
        )
        return _bash("run_finetune_method_b.sh", env)

    # cell
    idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", os.environ.get("CELL_INDEX", "0")))
    if not 0 <= idx < len(SIZES) * len(SEEDS):
        print(
            f"[entry] cell index {idx} out of range 0-{len(SIZES) * len(SEEDS) - 1}",
            file=sys.stderr,
        )
        return 2
    size = SIZES[idx % len(SIZES)]
    seed = SEEDS[idx // len(SIZES)]
    env["MODEL_SIZE"] = size
    env["PLANNER_SEED"] = seed
    print(f"[entry] cell idx={idx} -> size={size} seed={seed}", flush=True)
    return _bash("run_capacity_point.sh", env)


if __name__ == "__main__":
    raise SystemExit(main())
