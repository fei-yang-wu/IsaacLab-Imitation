"""Run the GR00T-head arms closed-loop on the Embodied-Control runtime.

For every (arm, rtc, goal) cell this driver writes an `ec lowlevel run` job —
tracker bundle + `gr00t_service` command source — runs it inside the
Embodied-Control repo's `lowlevel-sim` Pixi environment, and aggregates the
per-run metrics into one summary table.

The EC statistical eval is a deployment-rehearsal signal (MuJoCo actuator
dynamics differ from Newton/PhysX by design); the synced Isaac evaluation
remains the number of record.

Runs in the default Pixi environment:

    pixi run python -m imitation_experiments.evaluation.eval_gr00t_ec \
        --config-dir <campaign>/conf --config-name eval_ec
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import hydra
import yaml
from omegaconf import DictConfig, OmegaConf

from imitation_experiments.paths import REPO_ROOT


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def _service_cmd(cfg: DictConfig, checkpoint: Path, goal: str) -> list[str]:
    return [
        str(cfg.pixi_bin),
        "run",
        "-e",
        "gr00t",
        "python",
        "-m",
        "imitation_experiments.planner.gr00t_chunk_service",
        "--checkpoint",
        str(checkpoint),
        "--goal-features",
        str(_resolve(cfg.goal_features)),
        "--goal",
        goal,
    ]


def _job(cfg: DictConfig, arm: str, arm_cfg: Any, rtc: bool, goal: str, out: Path) -> dict:
    checkpoint = _resolve(arm_cfg.checkpoint)
    if not checkpoint.is_file():
        msg = f"arm {arm}: checkpoint missing: {checkpoint}"
        raise FileNotFoundError(msg)
    return {
        "api_version": "ec.lowlevel/v1alpha1",
        "bundle": str(_resolve(arm_cfg.bundle)),
        "env": {"backend": "mujoco", "model": str(_resolve(cfg.mjcf))},
        "command": {
            "topology": "local",
            "source": "gr00t_service",
            "gr00t": {
                "mode": str(arm_cfg.mode),
                "hold_steps": int(cfg.hold_steps),
                "slots": int(cfg.slots),
                "rtc": bool(rtc),
                "rtc_freeze_steps": int(cfg.rtc_freeze_steps),
                "service_cwd": str(REPO_ROOT),
                "service_cmd": _service_cmd(cfg, checkpoint, goal),
            },
        },
        "rollout": {
            "episodes": int(cfg.episodes),
            "max_steps": int(cfg.max_steps),
            "seed": int(cfg.seed),
            "record_states": bool(cfg.get("record_states", False)),
        },
        "safety": {"min_base_height_m": float(cfg.min_base_height_m)},
        "outputs": {"root": str(out)},
    }


def _run_cell(cfg: DictConfig, job: dict, job_path: Path) -> dict:
    job_path.parent.mkdir(parents=True, exist_ok=True)
    job_path.write_text(yaml.safe_dump(job, sort_keys=False))
    environment = dict(os.environ)
    environment.setdefault("MUJOCO_GL", "egl")
    completed = subprocess.run(
        [str(cfg.pixi_bin), "run", "-e", "lowlevel-sim", "ec", "lowlevel", "run", str(job_path)],
        cwd=str(_resolve(cfg.ec_repo)),
        env=environment,
        capture_output=True,
        text=True,
        timeout=float(cfg.cell_timeout_s),
    )
    out_root = Path(job["outputs"]["root"])
    run_dirs = sorted(out_root.glob("*_lowlevel")) if out_root.is_dir() else []
    record: dict[str, Any] = {
        "returncode": completed.returncode,
        "run_dir": str(run_dirs[-1]) if run_dirs else None,
    }
    # A damped episode (e.g. base_too_low) exits non-zero but writes full
    # artifacts — that is a measured failure, not a driver error. Only a
    # missing run directory or metrics file is an error.
    if not run_dirs or not (run_dirs[-1] / "metrics.json").is_file():
        record["stderr_tail"] = completed.stderr[-2000:]
        return record
    metrics = json.loads((run_dirs[-1] / "metrics.json").read_text())
    episodes = [
        json.loads(line)
        for line in (run_dirs[-1] / "episodes.jsonl").read_text().splitlines()
    ]
    record["metrics"] = metrics
    record["episodes"] = episodes
    record["survived"] = all(
        episode["status"] in {"completed", "reference_finished"}
        for episode in episodes
    )
    record["steps"] = [episode["steps"] for episode in episodes]
    record["damp_causes"] = [episode.get("damp_cause") for episode in episodes]
    return record


def _run_mpjpe(cfg: DictConfig, record: dict, goal: str) -> None:
    run_dir = record.get("run_dir")
    if run_dir is None or not any(Path(run_dir).glob("states_ep*.npz")):
        return
    completed = subprocess.run(
        [
            str(cfg.pixi_bin), "run", "-e", "lowlevel-sim", "python", "-m",
            "embodied_control.lowlevel.eval_mpjpe",
            "--run-dir", str(run_dir),
            "--reference", str(_resolve(cfg.reference_arrays)),
            "--motion", goal,
            "--mjcf", str(_resolve(cfg.mjcf)),
        ],
        cwd=str(_resolve(cfg.ec_repo)),
        capture_output=True,
        text=True,
        timeout=600,
    )
    mpjpe_path = Path(run_dir) / "mpjpe.json"
    if completed.returncode == 0 and mpjpe_path.is_file():
        record["mpjpe"] = json.loads(mpjpe_path.read_text())
    else:
        record["mpjpe_error"] = completed.stderr[-1000:]


@hydra.main(version_base="1.3", config_path=None, config_name=None)
def main(cfg: DictConfig) -> None:
    output_root = _resolve(cfg.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    only_arms = (
        {str(arm) for arm in cfg.only_arms} if cfg.get("only_arms") else None
    )
    results: dict[str, dict] = {}
    for arm, arm_cfg in cfg.arms.items():
        if only_arms is not None and str(arm) not in only_arms:
            continue
        for rtc in cfg.rtc_variants:
            variant = f"{arm}__{'rtc' if rtc else 'basic'}"
            for goal in cfg.goals:
                cell_dir = output_root / variant / goal
                job = _job(cfg, str(arm), arm_cfg, bool(rtc), str(goal), cell_dir)
                record = _run_cell(cfg, job, cell_dir / "job.yaml")
                if bool(cfg.get("compute_mpjpe", False)):
                    _run_mpjpe(cfg, record, str(goal))
                results.setdefault(variant, {})[str(goal)] = record
                status = "PASS" if record.get("survived") else "FAIL"
                steps = record.get("steps")
                print(f"[{status}] {variant} / {goal} steps={steps}", flush=True)

    summary: dict[str, Any] = {"cells": results, "config": OmegaConf.to_container(cfg, resolve=True)}
    lines = [
        "| variant | survived | mean steps | mpjpe-l mm | head p50 ms |",
        "| --- | --- | --- | --- | --- |",
    ]
    for variant, goals in results.items():
        survived = sum(1 for record in goals.values() if record.get("survived"))
        steps = [
            step
            for record in goals.values()
            for step in record.get("steps", [])
        ]
        head = [
            record["metrics"]["planner_head_ms"]["p50"]
            for record in goals.values()
            if record.get("metrics", {}).get("planner_head_ms")
        ]
        mpjpe = [
            episode_record["mpjpe_l_mm"]
            for record in goals.values()
            if record.get("survived")
            for name, episode_record in record.get("mpjpe", {}).items()
            if isinstance(episode_record, dict) and "mpjpe_l_mm" in episode_record
        ]
        mean_steps = sum(steps) / len(steps) if steps else 0.0
        mean_head = sum(head) / len(head) if head else float("nan")
        mean_mpjpe = sum(mpjpe) / len(mpjpe) if mpjpe else float("nan")
        lines.append(
            f"| {variant} | {survived}/{len(goals)} | {mean_steps:.0f} "
            f"| {mean_mpjpe:.1f} | {mean_head:.1f} |"
        )
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2))
    (output_root / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print(f"[PASS] EC eval summary -> {output_root}/summary.md", flush=True)


if __name__ == "__main__":
    main()
