"""Test whether motions chain through language alone, on the EC runtime.

Design: for each ordered goal pair (A, B) run TWO episodes that are identical
in every respect except the language goal after a switch tick.

- control: goal A for the whole episode.
- switch:  goal A, then goal B from `switch_tick` onward.

Both start from the same pose with the same seed and the same tracker, and
neither reads a reference — the EC `gr00t_service` source is driven purely by
the causal robot history plus the language goal. So any divergence after the
switch tick is attributable to the language change and nothing else.

Three runs per pair, so the criterion tests DIRECTION and not merely change:

- control:  goal A throughout.
- switch:   goal A, then goal B from `switch_tick`.
- target:   goal B throughout (the "what B looks like" reference run).

Reported per pair:
- `survived_switch`: the switch episode never fell (`base_too_low`), i.e.
  success under the campaign's fall-only definition.
- `divergence_rad`: post-switch joint distance between switch and control.
  Near zero means the language change did NOT alter behavior.
- `pre_switch_rad`: the same measure BEFORE the switch. The control on the
  control: it must be ~0, or the two runs were not identical and every other
  number is noise.
- `dist_to_A` / `dist_to_B`: post-switch distance from the switch run to the
  A-only and B-only runs. B is aligned by TIME SINCE GOAL ONSET — switch tick
  `T+k` against target tick `k` — because both are then "k steps into goal B".
- `identification_ratio` = `dist_to_A / dist_to_B`. Above 1 means the robot
  moved TOWARD the commanded motion rather than merely away from the old one.
  Divergence alone cannot distinguish "switched to B" from "fell apart".

`dist_to_B` will not reach zero even for perfect chaining: the switch run
enters goal B from A's pose, while the target run enters it from the reset
pose. The RATIO is the interpretable quantity, not either distance alone.

Deliberately reference-free: with the goal switched mid-episode the reference
motion does not follow, so MPJPE and `reference_finished` would score against
a motion the robot is no longer being asked to perform. Chaining quality
against a reference needs a paired reference schedule; this answers the prior
question of whether language moves the robot at all.

Runs in the default Pixi environment:

    pixi run python -m imitation_experiments.evaluation.eval_gr00t_chaining \
        --config-dir <campaign>/conf --config-name chaining
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import yaml
from omegaconf import DictConfig, OmegaConf

from imitation_experiments.paths import REPO_ROOT


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def _job(cfg: DictConfig, goals: list[str], schedule: list[list] | None, out: Path) -> dict:
    checkpoint = _resolve(cfg.arm.checkpoint)
    gr00t: dict[str, Any] = {
        "mode": str(cfg.arm.mode),
        "hold_steps": int(cfg.hold_steps),
        "slots": int(cfg.slots),
        "rtc": False,
        "service_cwd": str(REPO_ROOT),
        "service_cmd": [
            str(cfg.pixi_bin), "run", "-e", "gr00t", "python", "-m",
            "imitation_experiments.planner.gr00t_chunk_service",
            "--checkpoint", str(checkpoint),
            "--goal-features", str(_resolve(cfg.goal_features)),
            "--goal", goals[0],
            # Pin the sampler: control and switch runs must be bit-identical
            # before the switch tick, or post-switch divergence cannot be
            # attributed to the language change.
            "--seed", str(int(cfg.head_seed)),
        ],
        "goal_sequence": list(goals),
    }
    if schedule is not None:
        gr00t["goal_schedule"] = schedule
    return {
        "api_version": "ec.lowlevel/v1alpha1",
        "bundle": str(_resolve(cfg.arm.bundle)),
        "env": {"backend": "mujoco", "model": str(_resolve(cfg.mjcf))},
        "command": {"topology": "local", "source": "gr00t_service", "gr00t": gr00t},
        "rollout": {
            "episodes": 1,
            "max_steps": int(cfg.max_steps),
            "seed": int(cfg.seed),
            "record_states": True,
            # Renders all three runs of a pair, so the switch can be judged
            # against what A alone and B alone actually look like.
            "record_video": bool(cfg.get("record_video", False)),
        },
        "safety": {"min_base_height_m": float(cfg.min_base_height_m)},
        "outputs": {"root": str(out)},
    }


def _run(cfg: DictConfig, job: dict, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(job, sort_keys=False))
    environment = dict(os.environ)
    environment.setdefault("MUJOCO_GL", "egl")
    subprocess.run(
        [str(cfg.pixi_bin), "run", "-e", "lowlevel-sim", "ec", "lowlevel", "run", str(path)],
        cwd=str(_resolve(cfg.ec_repo)), env=environment,
        capture_output=True, text=True, timeout=float(cfg.timeout_s),
    )
    root = Path(job["outputs"]["root"])
    runs = sorted(root.glob("*_lowlevel")) if root.is_dir() else []
    if not runs:
        return {"ok": False}
    run_dir = runs[-1]
    episodes = [
        json.loads(line)
        for line in (run_dir / "episodes.jsonl").read_text().splitlines()
    ]
    states = sorted(run_dir.glob("states_ep*.npz"))
    joints = np.load(states[0])["joint_pos"] if states else None
    return {"ok": True, "run_dir": str(run_dir), "episodes": episodes, "joints": joints}


@hydra.main(version_base="1.3", config_path=None, config_name=None)
def main(cfg: DictConfig) -> None:
    output_root = _resolve(cfg.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    switch = int(cfg.switch_tick)
    results: dict[str, Any] = {}
    for pair in cfg.pairs:
        first, second = str(pair[0]), str(pair[1])
        name = f"{first}__then__{second}"
        control = _run(
            cfg, _job(cfg, [first], None, output_root / name / "control"),
            output_root / name / "control" / "job.yaml",
        )
        switched = _run(
            cfg,
            _job(cfg, [first], [[switch, second]], output_root / name / "switch"),
            output_root / name / "switch" / "job.yaml",
        )
        target = _run(
            cfg, _job(cfg, [second], None, output_root / name / "target"),
            output_root / name / "target" / "job.yaml",
        )
        record: dict[str, Any] = {
            "control_ok": control["ok"],
            "switch_ok": switched["ok"],
            "target_ok": target["ok"],
        }
        if control["ok"] and switched["ok"] and target["ok"]:
            a, b, t = control["joints"], switched["joints"], target["joints"]
            if a is not None and b is not None and t is not None:
                # Align the target by time since goal onset: switch tick T+k is
                # k steps into goal B, which is target tick k.
                span = min(len(a) - switch, len(b) - switch, len(t))
                if span > 0:
                    post_switch = b[switch : switch + span]
                    record["dist_to_A"] = float(
                        np.abs(post_switch - a[switch : switch + span]).mean()
                    )
                    record["dist_to_B"] = float(
                        np.abs(post_switch - t[:span]).mean()
                    )
                    if record["dist_to_B"] > 0:
                        record["identification_ratio"] = (
                            record["dist_to_A"] / record["dist_to_B"]
                        )
        if control["ok"] and switched["ok"]:
            episode = switched["episodes"][0]
            record["switch_status"] = episode["status"]
            record["survived_switch"] = episode["status"] in {
                "completed", "reference_finished"
            }
            a, b = control["joints"], switched["joints"]
            if a is not None and b is not None:
                ticks = min(len(a), len(b))
                record["ticks_compared"] = int(ticks)
                if ticks > switch:
                    record["divergence_rad"] = float(
                        np.abs(a[switch:ticks] - b[switch:ticks]).mean()
                    )
                    record["pre_switch_rad"] = float(
                        np.abs(a[:switch] - b[:switch]).mean()
                    )
        results[name] = record
        ratio = record.get("identification_ratio")
        print(
            f"[{name}] survived={record.get('survived_switch')} "
            f"divergence={record.get('divergence_rad')} "
            f"pre_switch={record.get('pre_switch_rad')} "
            f"dA={record.get('dist_to_A')} dB={record.get('dist_to_B')} "
            f"ratio={'n/a' if ratio is None else round(ratio, 3)}",
            flush=True,
        )

    summary = {"pairs": results, "config": OmegaConf.to_container(cfg, resolve=True)}
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[PASS] chaining summary -> {output_root}/summary.json", flush=True)


if __name__ == "__main__":
    main()
