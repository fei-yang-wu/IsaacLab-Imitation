#!/usr/bin/env python3
"""Reproduce the command-interface planner-capacity study end to end.

What this measures
------------------
Whether a compressed command interface reduces planner-training complexity, on a
frozen low-level protocol. Two claims:

    iso-parameter    better MPJPE at matched planner parameter count
    iso-performance  fewer planner parameters to reach a fixed MPJPE

The interfaces form a packet-size ladder at a fixed 5 Hz publication rate:

    full_body_trajectory  670  29 qpos + 29 qvel + root
    root_qpos             380  29 qpos + root            (no velocities)
    ee_trajectory         360  4 EE poses, NO root       (the rootless control)
    latent_skill          258  z256 + phase
    root_points5          240  5 keypoint positions + root

Each interface has its OWN low-level controller, trained natively on that
command space. Nothing is reconstructed: a root_qpos controller never receives
joint velocities at all.

Why the stages are ordered this way
-----------------------------------
`qualify` runs first and gates everything. An interface whose own frozen
controller cannot track it makes every downstream planner number meaningless --
and this is not hypothetical: ee_trajectory reached a healthy-looking 41.3 mm on
the TRAINING metric while its true frame-0/700-step floor was 405.2 mm, because
training episodes terminate early and never accumulate drift. Only the gate
distinguishes those two cases.

Usage
-----
    # everything, from a clean study root
    pixi run python experiments/paper/run_interface_capacity_study.py

    # just re-aggregate an existing run
    pixi run python experiments/paper/run_interface_capacity_study.py \\
        stages=[aggregate] gates.refuse_existing_study_root=false

    # one interface, one seed, for a smoke check
    pixi run python experiments/paper/run_interface_capacity_study.py \\
        grid.interfaces=[root_points5] grid.seeds=[0] grid.sizes=[tiny]

    # reproduce the rootless-control measurement (expects UNUSABLE INTERFACE)
    pixi run python experiments/paper/run_interface_capacity_study.py \\
        stages=[qualify] grid.interfaces=[ee_trajectory] \\
        interfaces.ee_trajectory.enabled=true

Every parameter lives in `conf/interface_capacity.yaml` next to this file.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger("interface_capacity")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CAMPAIGN = (
    _REPO_ROOT / "experiments" / "campaigns" / "2026-07-23-lafan1-planner-capacity"
)


class StudyError(RuntimeError):
    """A gate failed. Deliberately fatal: partial studies must not look complete."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _enabled_interfaces(cfg: DictConfig) -> list[str]:
    """Requested interfaces that are also enabled, preserving config order."""
    requested = [str(name) for name in cfg.grid.interfaces]
    unknown = [name for name in requested if name not in cfg.interfaces]
    if unknown:
        raise StudyError(
            f"Unknown interface(s) {unknown}; conf/interface_capacity.yaml defines "
            f"{sorted(cfg.interfaces.keys())}."
        )
    selected = [name for name in requested if bool(cfg.interfaces[name].enabled)]
    if not selected:
        raise StudyError(
            f"None of the requested interfaces {requested} are enabled. "
            "ee_trajectory is disabled by default because it is the rootless "
            "control; enable it explicitly to reproduce that measurement."
        )
    skipped = [name for name in requested if name not in selected]
    if skipped:
        logger.warning("Skipping disabled interface(s): %s", skipped)
    return selected


def _verify_checkpoints(cfg: DictConfig, interfaces: list[str]) -> dict[str, str]:
    """Resolve each interface's checkpoint and check it against its hash."""
    root = Path(cfg.paths.checkpoint_root)
    resolved: dict[str, str] = {}
    for name in interfaces:
        spec = cfg.interfaces[name]
        path = root / str(spec.checkpoint)
        if not path.is_file():
            raise StudyError(
                f"Interface {name!r}: checkpoint not found at {path}. Stage it "
                "under paths.checkpoint_root before running the study."
            )
        if bool(cfg.gates.require_checkpoint_hash):
            actual = _sha256(path)
            expected = str(spec.sha256)
            if actual != expected:
                raise StudyError(
                    f"Interface {name!r}: checkpoint hash mismatch at {path}.\n"
                    f"  expected {expected}\n  actual   {actual}\n"
                    "Checkpoints have been silently substituted and truncated in "
                    "this campaign before; refusing to produce numbers against an "
                    "unverified one."
                )
        resolved[name] = str(path)
        logger.info(
            "%-22s %4d values  %s  oracle %.1f mm",
            name,
            int(spec.packet_values),
            path.name,
            float(spec.oracle_mpjpe_mm),
        )
    return resolved


def _qualification_record(cfg: DictConfig, interface: str) -> Path:
    return (
        Path(cfg.paths.study_root)
        / "qualification"
        / f"{interface}_{cfg.motion.name}"
        / "qualification.json"
    )


def _require_qualification(cfg: DictConfig, interfaces: list[str]) -> None:
    """Refuse to spend planner compute on an unqualified interface."""
    if not bool(cfg.gates.require_qualification):
        logger.warning(
            "gates.require_qualification=false -- planner numbers may be produced "
            "on an interface its own controller cannot track."
        )
        return
    missing: list[str] = []
    for name in interfaces:
        record = _qualification_record(cfg, name)
        if not record.is_file():
            missing.append(f"{name}: no record at {record}")
            continue
        payload = json.loads(record.read_text())
        if str(payload.get("result")) != "PASS":
            missing.append(f"{name}: result={payload.get('result')!r}")
    if missing:
        raise StudyError(
            "Qualification gate failed for:\n  " + "\n  ".join(missing) + "\n"
            "Run stages=[qualify] first. A planner study on an interface whose "
            "controller cannot follow it measures nothing."
        )


def _run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    merged = dict(os.environ)
    merged.update(env or {})
    printable = " ".join(command)
    logger.info("[CMD] %s", printable)
    result = subprocess.run(command, cwd=_REPO_ROOT, env=merged, check=False)
    if result.returncode != 0:
        raise StudyError(f"Command failed (exit {result.returncode}): {printable}")


def _campaign_env(cfg: DictConfig, interface: str, checkpoint: str) -> dict[str, str]:
    """Environment the campaign shell scripts read.

    The campaign scripts are the single implementation; this entrypoint supplies
    their inputs and enforces the gates around them, rather than duplicating the
    orchestration in a second place that could drift.
    """
    spec = cfg.interfaces[interface]
    env = {
        "MOTION_NAME": str(cfg.motion.name),
        "MANIFEST": str(cfg.paths.manifest),
        "STEPS": str(cfg.protocol.control_steps),
        "EVAL_STEPS": str(cfg.protocol.control_steps),
        "NUM_ENVS": str(cfg.protocol.num_envs),
        "NJMAX": str(cfg.protocol.njmax),
        "NCONMAX": str(cfg.protocol.nconmax),
        "TOL_MM": str(cfg.qualify.tolerance_mm),
        "FLOOR_MAX_MM": str(cfg.qualify.floor_max_mm),
        "COLLECT_STEPS": str(cfg.grid.collect_steps),
        "DEMO_ONLY": "1" if bool(cfg.grid.demo_only) else "0",
        "LATENT_TASK": str(cfg.interfaces.latent_skill.task),
        "LATENT_DATASET_PATH": str(cfg.paths.latent_dataset),
        "CHUNK_TASK": str(spec.task) if interface != "latent_skill" else "",
    }
    checkpoint_var = {
        "latent_skill": "LATENT_LOW_LEVEL_CHECKPOINT",
        "full_body_trajectory": "FBCHUNK_LOW_LEVEL_CHECKPOINT",
        "ee_trajectory": "EECHUNK_LOW_LEVEL_CHECKPOINT",
        "root_qpos": "ROOT_QPOS_LOW_LEVEL_CHECKPOINT",
        "root_points5": "ROOT_POINTS5_LOW_LEVEL_CHECKPOINT",
    }[interface]
    env[checkpoint_var] = checkpoint
    if interface == "latent_skill":
        env["LATENT_SKILL_CHECKPOINT"] = str(
            Path(cfg.paths.checkpoint_root) / str(spec.skill_checkpoint)
        )
    return {key: value for key, value in env.items() if value}


def stage_qualify(
    cfg: DictConfig, interfaces: list[str], ckpts: dict[str, str]
) -> None:
    """Replay floor vs 5 Hz oracle stream, per interface."""
    for name in interfaces:
        if name == "latent_skill":
            # The latent interface has no chunked/unchunked distinction to check:
            # its packet is one z held across the window, not a slotted sequence.
            logger.info("qualify: skipping %s (no chunk adapter to certify)", name)
            continue
        out = Path(cfg.paths.study_root) / "qualification" / f"{name}_{cfg.motion.name}"
        _run(
            ["bash", str(_CAMPAIGN / "qualify_interface.sh")],
            env={
                **_campaign_env(cfg, name, ckpts[name]),
                "INTERFACE": name,
                "CHECKPOINT": ckpts[name],
                "OUT_ROOT": str(out),
            },
        )


def stage_oracle(cfg: DictConfig, interfaces: list[str], ckpts: dict[str, str]) -> None:
    """Oracle metrics + the demonstration rows the planners train on.

    One invocation for all interfaces, not one per interface: the script gates
    each block on INTERFACES and skips artifacts that already exist, so a single
    call does exactly the missing work.
    """
    env: dict[str, str] = {}
    for name in interfaces:
        env.update(_campaign_env(cfg, name, ckpts[name]))
    env.update(
        {
            "INTERFACES": " ".join(interfaces),
            "OUTPUT_ROOT": str(Path(cfg.paths.study_root) / "oracle_baselines"),
            "ORACLE_STEPS": str(cfg.protocol.control_steps),
            # Every explicit interface shares this task; the latent row carries
            # its own via LATENT_TASK.
            "CHUNK_TASK": str(cfg.interfaces.full_body_trajectory.task),
        }
    )
    _run(["bash", str(_CAMPAIGN / "prepare_oracle_baselines.sh")], env=env)


def stage_grid(cfg: DictConfig, interfaces: list[str], ckpts: dict[str, str]) -> None:
    """One planner per (interface, size, seed)."""
    _require_qualification(cfg, [n for n in interfaces if n != "latent_skill"])
    for seed in cfg.grid.seeds:
        for size in cfg.grid.sizes:
            for name in interfaces:
                logger.info("grid cell: interface=%s size=%s seed=%s", name, size, seed)
                _run(
                    ["bash", str(_CAMPAIGN / "run_capacity_point.sh")],
                    env={
                        **_campaign_env(cfg, name, ckpts[name]),
                        "INTERFACES": name,
                        "MODEL_SIZE": str(size),
                        "PLANNER_SEED": str(seed),
                        "STUDY_ROOT": str(cfg.paths.study_root),
                        "PLANNER_UPDATES": str(cfg.planner.num_updates),
                        "PLANNER_MAX_SAMPLES": str(cfg.planner.max_samples),
                    },
                )


def stage_aggregate(cfg: DictConfig, interfaces: list[str], _ckpts: object) -> None:
    """Per-seed and across-seed summaries plus the paper tables.

    Each interface is normalized by its OWN oracle: the trackers differ in
    quality (23.6-30.6 mm across the ladder), so absolute MPJPE alone would
    conflate planner quality with tracker quality.
    """
    study_root = Path(cfg.paths.study_root)
    oracle_root = study_root / "oracle_baselines"
    steps = int(cfg.protocol.control_steps)
    sizes = [str(size) for size in cfg.grid.sizes]

    oracle_args: list[str] = []
    for name in interfaces:
        summary = oracle_root / name / f"oracle_frame0_{steps}" / "summary.json"
        if not summary.is_file():
            raise StudyError(
                f"Missing oracle summary for {name!r}: {summary}. Run "
                "stages=[oracle] first -- planner rows cannot be normalized "
                "without their interface's own oracle."
            )
        oracle_args += ["--oracle", f"{name}={summary}"]

    seed_inputs: list[Path] = []
    for seed in cfg.grid.seeds:
        seed_root = study_root / "scaling" / f"seed{seed}"
        out = seed_root / "capacity_summary"
        command = [
            sys.executable,
            "-m",
            "imitation_experiments.capacity.aggregate_one_motion_capacity_scaling",
            "--scaling_root",
            str(seed_root),
            "--sizes",
            *sizes,
            *oracle_args,
            "--output_dir",
            str(out),
            "--overwrite",
        ]
        if bool(cfg.grid.demo_only):
            # A demo-only grid writes no eval_finetuned_* artifacts, so the
            # aggregator has to be told to expect the demonstration stage only.
            # grid.demo_only defaults to true, so without this the default
            # config could never aggregate its own output.
            command += ["--stages", "demonstration_only"]
        _run(command)
        seed_inputs.append(out / "capacity_results.json")

    final = study_root / "capacity_seeds_summary"
    # The across-seed aggregator refuses a non-empty output dir on purpose, so a
    # crashed run cannot be silently overwritten. A crash can also leave the dir
    # empty; clear that case rather than making the user rmdir by hand.
    if final.is_dir() and not any(final.iterdir()):
        final.rmdir()
    input_args: list[str] = []
    for path in seed_inputs:
        input_args += ["--input", str(path)]
    command = [
        sys.executable,
        "-m",
        "imitation_experiments.capacity.aggregate_one_motion_capacity_seeds",
        *input_args,
        "--min_seeds",
        str(len(cfg.grid.seeds)),
        "--survival_target",
        str(cfg.aggregate.survival_target),
        "--normalized_mpjpe_target",
        str(cfg.aggregate.normalized_mpjpe_target),
        "--output_dir",
        str(final),
    ]
    if bool(cfg.aggregate.overwrite):
        command.append("--overwrite")
    _run(command)


_STAGES = {
    "qualify": stage_qualify,
    "oracle": stage_oracle,
    "grid": stage_grid,
    "aggregate": stage_aggregate,
}


@hydra.main(version_base=None, config_path="conf", config_name="interface_capacity")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    stages = [str(stage) for stage in cfg.stages]
    unknown = [stage for stage in stages if stage not in _STAGES]
    if unknown:
        raise StudyError(
            f"Unknown stage(s) {unknown}; expected a subset of {sorted(_STAGES)}."
        )

    study_root = Path(cfg.paths.study_root)
    creating = {"qualify", "oracle", "grid"} & set(stages)
    if (
        creating
        and study_root.exists()
        and any(study_root.iterdir())
        and bool(cfg.gates.refuse_existing_study_root)
    ):
        raise StudyError(
            f"Study root already exists and is not empty: {study_root}\n"
            "Refusing to write into it, so a partial rerun cannot be mistaken for "
            "a complete study. Point paths.study_root elsewhere, or set "
            "gates.refuse_existing_study_root=false for an intentional resume."
        )
    study_root.mkdir(parents=True, exist_ok=True)

    interfaces = _enabled_interfaces(cfg)
    logger.info("Interfaces: %s", interfaces)
    checkpoints = _verify_checkpoints(cfg, interfaces)

    # Record exactly what produced these artifacts, next to them.
    provenance: dict[str, Any] = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "interfaces": interfaces,
        "checkpoints": checkpoints,
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip(),
        "stages": stages,
    }
    (study_root / "study_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True)
    )

    for stage in stages:
        logger.info("===== stage: %s", stage)
        _STAGES[stage](cfg, interfaces, checkpoints)

    logger.info("Study complete under %s", study_root)


if __name__ == "__main__":
    main()
