"""Driver for the skill-composition probes: one evaluator process per setting.

Each setting pins ``K`` clip pairs to ``2K`` environments (``2i`` target,
``2i+1`` source) and runs the evaluator with a latent blend. The tests, all
on the same pairs:

* ``held_alpha``: the mix ``z_t + a (z_s - z_t)`` held from step 0 at each
  alpha in ``--alphas`` (monotone-in-alpha test);
* ``handover``: walk, then switch to the source code at ``--switch-steps``
  with each ramp in ``--ramps`` (handover cost);
* ``extrapolate``: held mixes at alphas outside ``[0, 1]``.

Every evaluator run is a subprocess (Isaac Sim cannot restart in-process),
launched through ``scripts/rlopt/run_evaluator.py`` so the container's torch
bridge is honoured. Summaries land in ``<out>/<test>/<setting>.json``;
``<out>/<test>/index.json`` records the settings. Score with
``imitation_experiments.evaluation.composition_metrics``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from imitation_experiments.paths import REPO_ROOT

RUN_EVALUATOR = REPO_ROOT / "scripts" / "rlopt" / "run_evaluator.py"

BODY_NAMES = (
    "[pelvis,left_hip_roll_link,left_knee_link,left_ankle_roll_link,"
    "right_hip_roll_link,right_knee_link,right_ankle_roll_link,torso_link,"
    "left_shoulder_roll_link,left_elbow_link,left_wrist_yaw_link,"
    "right_shoulder_roll_link,right_elbow_link,right_wrist_yaw_link]"
)
CELLS = "[2048,2048,1024,1024,512,512]"


@dataclass(frozen=True)
class ArmConfig:
    """What a tracker needs to run: its checkpoint, encoder, and overrides."""

    name: str
    checkpoint: Path
    encoder: Path
    agent_entry_point: str = "rlopt_ipmd_tuned_cfg_entry_point"
    extra_overrides: tuple[str, ...] = ()
    command_dim: int = 66
    code_latent_dim: int = 64


@dataclass(frozen=True)
class Setting:
    test: str
    label: str
    start_step: int
    ramp_steps: int
    final_alpha: float
    steps: int


@dataclass
class Plan:
    settings: list[Setting] = field(default_factory=list)


def resolve_checkpoint(spec: str) -> Path:
    """``newest:<tree>`` picks the deepest ``model_step_*.pt`` under the tree;
    anything else is a file path."""
    if spec.startswith("newest:"):
        tree = Path(spec[len("newest:") :])
        candidates = sorted(
            tree.rglob("model_step_*.pt"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )
        if not candidates:
            raise FileNotFoundError(f"no model_step_*.pt under {tree}")
        return candidates[-1]
    return Path(spec)


def make_plan(
    test: str,
    *,
    steps: int,
    alphas: Sequence[float],
    ramps: Sequence[int],
    switch_steps: Sequence[int],
) -> Plan:
    plan = Plan()
    if test in ("held_alpha", "extrapolate"):
        for a in alphas:
            plan.settings.append(
                Setting(
                    test,
                    f"alpha{a:+.2f}".replace("+", "p").replace("-", "m"),
                    0,
                    0,
                    float(a),
                    steps,
                )
            )
    elif test == "handover":
        for ramp in ramps:
            for switch in switch_steps:
                plan.settings.append(
                    Setting(
                        test,
                        f"switch{switch}_ramp{ramp}",
                        int(switch),
                        int(ramp),
                        1.0,
                        steps,
                    )
                )
    else:
        raise ValueError(f"unknown test {test!r}")
    return plan


def evaluator_args(
    arm: ArmConfig,
    pairs: Sequence[dict[str, Any]],
    setting: Setting,
    *,
    out_json: Path,
    reference_arrays: str,
    persist_id: str,
    physics: str,
    seed: int,
) -> list[str]:
    ranks: list[str] = []
    for pair in pairs:
        ranks += [str(int(pair["a"])), str(int(pair["b"]))]
    args = [
        "--task",
        "Isaac-Imitation-G1-v2",
        "--algo",
        "IPMD",
        "--agent_entry_point",
        arm.agent_entry_point,
        "--checkpoint",
        str(arm.checkpoint),
        "--output_json",
        str(out_json),
        "--label",
        f"{arm.name}_{setting.test}_{setting.label}",
        "--num_envs",
        str(2 * len(pairs)),
        "--trajectory_ranks",
        *ranks,
        "--steps",
        str(setting.steps),
        "--randomization",
        "none",
        "--action_sampling",
        "mode",
        "--seed",
        str(seed),
        "--reference_start_frame",
        "0",
        "--reset_schedule",
        "sequential",
        "--skill_encoder_source",
        "pretrained",
        "--latent_blend_layout",
        "pairs",
        "--latent_blend_start_step",
        str(setting.start_step),
        "--latent_blend_ramp_steps",
        str(setting.ramp_steps),
        "--latent_blend_final_alpha",
        str(setting.final_alpha),
        "--headless",
        f"physics={physics}",
        "env.data.manifest=null",
        "env.data.cache_dir=null",
        f"env.data.reference_arrays_dir={reference_arrays}",
        f"env.data.persist_id={persist_id}",
        "env.data.persist_dir=null",
        "env.data.macro_cache_device=cuda:0",
        "env.data.wrap_steps=false",
        f"env.data.runtime_cache_body_names={BODY_NAMES}",
        "env.events.push_robot=null",
        "env.command_interface.actor=latent",
        f"env.command_interface.actor.dim={arm.command_dim}",
        "env.command_interface.encoder=single",
        "env.expert_macro_state_terms=[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b]",
        "env.expert_macro_frame_stride=1",
        "env.expert_macro_anchor_mode=robot_heading",
        # Reference-relative terminations off: they fire on any gait change.
        "env.terminations.anchor_pos=null",
        "env.terminations.anchor_ori=null",
        "env.terminations.ee_body_pos=null",
        "env.terminations.foot_pos_xyz=null",
        "agent.logger.backend=",
        "agent.ipmd.command_source=hl_skill",
        f"agent.ipmd.hl_skill_checkpoint_path={arm.encoder}",
        "agent.ipmd.hl_skill_finetune_enabled=false",
        f"agent.ipmd.latent_dim={arm.command_dim}",
        "agent.ipmd.latent_steps_min=1",
        "agent.ipmd.latent_steps_max=1",
        "agent.ipmd.hl_skill_horizon_steps=10",
        "agent.ipmd.hl_skill_command_mode=z",
        "agent.ipmd.latent_learning.command_phase_mode=sin_cos",
        f"agent.ipmd.latent_learning.code_latent_dim={arm.code_latent_dim}",
        "agent.ipmd.latent_learning.code_period=1",
        f"agent.policy.num_cells={CELLS}",
        f"agent.value_function.num_cells={CELLS}",
        "agent.policy.activation_fn=silu",
        "agent.value_function.activation_fn=silu",
        *arm.extra_overrides,
    ]
    return args


def chunk_unique(
    pairs: Sequence[dict[str, Any]], max_per_chunk: int
) -> list[list[dict[str, Any]]]:
    """Pack pairs into processes so that no clip rank repeats inside one
    process (the evaluator pins one rank per environment and refuses
    duplicates), each chunk holding at most ``max_per_chunk`` pairs."""
    chunks: list[list[dict[str, Any]]] = []
    used: list[set[int]] = []
    for pair in pairs:
        a, b = int(pair["a"]), int(pair["b"])
        placed = False
        for chunk, ranks in zip(chunks, used):
            if (
                len(chunk) < max_per_chunk
                and a not in ranks
                and b not in ranks
                and a != b
            ):
                chunk.append(pair)
                ranks.update((a, b))
                placed = True
                break
        if not placed:
            chunks.append([pair])
            used.append({a, b})
    return chunks


def run_plan(
    arm: ArmConfig,
    pairs: Sequence[dict[str, Any]],
    plan: Plan,
    *,
    out_dir: Path,
    reference_arrays: str,
    persist_id: str,
    physics: str,
    seed: int,
    pairs_per_process: int,
    dry_run: bool,
    python: str,
) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    index: list[dict[str, Any]] = []
    chunks = chunk_unique(pairs, pairs_per_process)
    for setting in plan.settings:
        for chunk_index, chunk in enumerate(chunks):
            out_json = out_dir / f"{arm.name}_{setting.label}_chunk{chunk_index}.json"
            entry = {
                "arm": arm.name,
                "checkpoint": str(arm.checkpoint),
                "test": setting.test,
                "label": setting.label,
                "start_step": setting.start_step,
                "ramp_steps": setting.ramp_steps,
                "final_alpha": setting.final_alpha,
                "steps": setting.steps,
                "chunk": chunk_index,
                "pairs": chunk,
                "summary": str(out_json),
            }
            if out_json.is_file() and out_json.stat().st_size > 0:
                entry["status"] = "skipped_existing"
                index.append(entry)
                continue
            cmd = [python, str(RUN_EVALUATOR)] + evaluator_args(
                arm,
                chunk,
                setting,
                out_json=out_json,
                reference_arrays=reference_arrays,
                persist_id=persist_id,
                physics=physics,
                seed=seed,
            )
            print(
                f"[PROBE] {arm.name} {setting.test} {setting.label} chunk {chunk_index}: {len(chunk)} pairs"
            )
            if dry_run:
                print(" ".join(cmd))
                entry["status"] = "dry_run"
            else:
                result = subprocess.run(
                    cmd, cwd=str(REPO_ROOT), env=os.environ.copy(), check=False
                )
                entry["status"] = (
                    "ok"
                    if result.returncode == 0 and out_json.is_file()
                    else f"failed_rc{result.returncode}"
                )
                print(f"[PROBE] -> {entry['status']}")
            index.append(entry)
        (out_dir / "index.json").write_text(json.dumps(index, indent=1))
    return index


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--checkpoint", required=True, help="path, or newest:<tree>")
    parser.add_argument("--encoder", required=True)
    parser.add_argument(
        "--agent-entry-point", default="rlopt_ipmd_tuned_cfg_entry_point"
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="extra hydra override, repeatable",
    )
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument(
        "--test", required=True, choices=["held_alpha", "handover", "extrapolate"]
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument(
        "--alphas", type=float, nargs="*", default=[0.0, 0.25, 0.5, 0.75, 1.0]
    )
    parser.add_argument("--ramps", type=int, nargs="*", default=[0, 10, 50])
    parser.add_argument("--switch-steps", type=int, nargs="*", default=[150, 160, 170])
    parser.add_argument("--pairs-per-process", type=int, default=64)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--reference-arrays", required=True)
    parser.add_argument("--persist-id", default="bones_seed_sonic_full_129785@e714bbff")
    parser.add_argument("--physics", default="newton_mjwarp")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    pairs = json.loads(Path(args.pairs).read_text())["pairs"]
    if args.max_pairs is not None:
        pairs = pairs[: args.max_pairs]
    if not pairs:
        raise SystemExit("no pairs in the pairs file")
    arm = ArmConfig(
        name=args.arm,
        checkpoint=resolve_checkpoint(args.checkpoint),
        encoder=Path(args.encoder),
        agent_entry_point=args.agent_entry_point,
        extra_overrides=tuple(args.override),
    )
    plan = make_plan(
        args.test,
        steps=args.steps,
        alphas=args.alphas,
        ramps=args.ramps,
        switch_steps=args.switch_steps,
    )
    print(f"[PROBE] arm {arm.name} checkpoint {arm.checkpoint} encoder {arm.encoder}")
    print(
        f"[PROBE] test {args.test}: {len(plan.settings)} settings x {len(pairs)} pairs"
    )
    index = run_plan(
        arm,
        pairs,
        plan,
        out_dir=args.out / args.test,
        reference_arrays=args.reference_arrays,
        persist_id=args.persist_id,
        physics=args.physics,
        seed=args.seed,
        pairs_per_process=args.pairs_per_process,
        dry_run=args.dry_run,
        python=args.python,
    )
    failed = [e for e in index if str(e.get("status", "")).startswith("failed")]
    print(f"[PROBE] done: {len(index)} runs, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
