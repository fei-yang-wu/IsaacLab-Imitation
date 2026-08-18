"""End-to-end checks that need the Embodied-Control ``lowlevel-sim`` env.

These run the real EC worker in its own Pixi environment, so they are skipped
wherever that env is not built (cluster login nodes, fresh checkouts, CI).
The unit tests cover the reductions and the noise math; what only a live run
can prove is that the owned rollout loop still matches the rig it mirrors and
that noise does not leak into the metric path.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from imitation_experiments.evaluation.protocol import SONIC_OBSERVATION_NOISE
from imitation_experiments.paths import REPO_ROOT

EC_REPO = REPO_ROOT / "external/Embodied-Control"
BUNDLE = EC_REPO / "assets/latent_playkit/bundles/fsq64_sonic_4500m"
MODEL = EC_REPO / "assets/latent_playkit/model/g1_29dof_rev_1_0.xml"
REFERENCE = REPO_ROOT / "data/bones_seed_language10_v1/reference_arrays/root_qpos_v1"
WORKER = (
    REPO_ROOT
    / "source/imitation_experiments/imitation_experiments/evaluation"
    / "ec_sidecar_worker.py"
)

_ASSETS_PRESENT = (
    (EC_REPO / ".pixi/envs/lowlevel-sim").is_dir()
    and BUNDLE.is_dir()
    and REFERENCE.is_dir()
    and shutil.which("pixi") is not None
)

# Each case spawns a Pixi subprocess and steps MuJoCo, so these stay opt-in:
# `pixi run test-experiments` must remain fast enough to run on every edit.
#     EC_SIDECAR_INTEGRATION=1 pixi run python -m pytest \
#         source/imitation_experiments/tests/test_ec_sidecar_integration.py
pytestmark = pytest.mark.skipif(
    not _ASSETS_PRESENT or os.environ.get("EC_SIDECAR_INTEGRATION") != "1",
    reason=(
        "needs EC lowlevel-sim env, playkit bundle, reference tree, and "
        "EC_SIDECAR_INTEGRATION=1"
    ),
)


def _run(job: dict, work_dir: Path, *, self_check: bool = False) -> dict:
    work_dir.mkdir(parents=True, exist_ok=True)
    job_path = work_dir / "job.json"
    job_path.write_text(json.dumps(job))
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("PIXI_")
    }
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment.setdefault("MUJOCO_GL", "egl")
    command = [
        "pixi",
        "run",
        "-e",
        "lowlevel-sim",
        "python",
        str(WORKER),
        "--job",
        str(job_path),
    ]
    if self_check:
        command.append("--self-check")
    completed = subprocess.run(
        command,
        cwd=str(EC_REPO),
        env=environment,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    if self_check:
        return {"stdout": completed.stdout}
    return json.loads(Path(job["output"]).read_text())


def _job(work_dir: Path, *, noise: dict, cases: list[dict]) -> dict:
    work_dir.mkdir(parents=True, exist_ok=True)
    return {
        "bundle": str(BUNDLE),
        "model": str(MODEL),
        "reference_root": str(REFERENCE),
        "cases": cases,
        "max_steps": 1200,
        "fall_height_m": 0.4,
        "hold_steps": None,
        "noise": noise,
        "output": str(work_dir / "worker_result.json"),
    }


def _case(rank: int, repeat: int = 0) -> dict:
    return {
        "trajectory_rank": rank,
        "motion_name": None,
        "start_frame": 0,
        "env_seed": 0,
        "repeat_index": repeat,
    }


def test_owned_loop_matches_the_rig_with_noise_disabled(tmp_path: Path) -> None:
    """The drift guard: our loop is a copy, so it must be proven identical."""
    result = _run(_job(tmp_path, noise={}, cases=[_case(0)]), tmp_path, self_check=True)

    assert "SELF-CHECK OK" in result["stdout"]


def test_noise_is_reproducible_across_processes(tmp_path: Path) -> None:
    """Sync lockstep: the same board twice must be bit-identical.

    Two separate processes, so this also covers the seeding being derived from
    the case rather than from anything process-local.
    """
    cases = [_case(rank, repeat) for rank in (0, 8) for repeat in range(2)]
    noise = dict(SONIC_OBSERVATION_NOISE)

    first = _run(_job(tmp_path / "a", noise=noise, cases=cases), tmp_path / "a")
    second = _run(_job(tmp_path / "b", noise=noise, cases=cases), tmp_path / "b")

    def strip(result: dict) -> list[dict]:
        return [
            {key: value for key, value in row.items() if key != "eval_seconds"}
            for row in result["episodes"]
        ]

    assert strip(first) == strip(second)


def test_noise_changes_the_rollout_but_not_the_reference_alignment(
    tmp_path: Path,
) -> None:
    """Noise must move the robot, never the reference it is scored against."""
    cases = [_case(0)]
    clean = _run(_job(tmp_path / "clean", noise={}, cases=cases), tmp_path / "clean")
    noisy = _run(
        _job(tmp_path / "noisy", noise=dict(SONIC_OBSERVATION_NOISE), cases=cases),
        tmp_path / "noisy",
    )

    clean_row, noisy_row = clean["episodes"][0], noisy["episodes"][0]
    # Same motion, so the reference-side facts are identical...
    assert clean_row["motion_name"] == noisy_row["motion_name"]
    assert clean_row["motion_length"] == noisy_row["motion_length"]
    # ...the robot's spawn is still exactly on reference frame 0...
    assert noisy_row["initial_pose_error"]["root_pos_m"] == 0.0
    assert noisy_row["initial_pose_error"]["joint_rad"] == 0.0
    # ...and the tracking error genuinely moved.
    assert noisy_row["mpjpe_l_mm"] != clean_row["mpjpe_l_mm"]


def test_repeats_produce_different_episodes_under_noise(tmp_path: Path) -> None:
    """Repeats are samples now; identical repeats would mean unseeded draws."""
    cases = [_case(0, repeat) for repeat in range(3)]
    result = _run(
        _job(tmp_path, noise=dict(SONIC_OBSERVATION_NOISE), cases=cases), tmp_path
    )

    values = [row["mpjpe_l_mm"] for row in result["episodes"]]
    assert len(set(values)) == len(values), f"repeats were identical: {values}"
