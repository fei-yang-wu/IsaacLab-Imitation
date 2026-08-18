"""Synchronous-lockstep EC/MuJoCo tracking worker for the evaluation sidecar.

This file runs inside the Embodied-Control ``lowlevel-sim`` Pixi environment,
which does not install ``imitation_experiments``. It therefore imports only
the standard library, numpy, and ``embodied_control`` — never anything from
this repository's packages. The orchestrator
(:mod:`imitation_experiments.evaluation.ec_tracker_sidecar`) invokes it by
absolute file path:

    pixi run -e lowlevel-sim python <this file> --job <job.json>

The rollout is a synchronous lockstep loop: publisher and tracker are ticked
once per control step through an in-process buffer and the simulated clock
never waits, so the result is a pure function of (bundle, reference, start
frame, noise seed) — bit-identical under any host load. That is the
sync-lockstep requirement for evaluation on shared compute nodes; the
asynchronous shared-memory path is reserved for deployment-rehearsal timing
gates on idle hosts.

**Sensor noise.** Embodied-Control is the hardware rehearsal rig, and real
hardware does not deliver clean state, so the rehearsal protocol injects
observation noise. Noise is drawn from a seeded generator, never from wall
clock or host timing, so it costs nothing in reproducibility: a noisy run is
still bit-identical when repeated under any load. Magnitudes default to the
uniform ranges the policy was trained against
(``config/g1/common/observations.py``: ``projected_gravity`` +/-0.05,
``base_ang_vel`` +/-0.2 rad/s, ``joint_pos_rel`` +/-0.01 rad,
``joint_vel_rel`` +/-0.5 rad/s), matching Isaac's convention of perturbing the
observation without renormalizing.

Two invariants keep the measurement honest:

* Noise reaches **only the controller's view** of the robot. Metrics are
  computed from the true simulator state, so MPJPE never measures the noise
  that was injected.
* With noise disabled the loop must reproduce
  :meth:`LatentPlayground.rollout` exactly. That equality is the guard that
  this owned loop has not drifted from the rig it mirrors, and it is what
  lets the deterministic and noisy protocols be compared.

Job JSON schema (written by the orchestrator):

    {
      "bundle": "<policy bundle dir>",
      "model": "<G1 MJCF path>",
      "reference_root": "<reference arrays dir>",
      "cases": [
        {"trajectory_rank": 0, "motion_name": null, "start_frame": 0,
         "env_seed": 0, "repeat_index": 0},
        ...
      ],
      "max_steps": 1200,
      "fall_height_m": 0.4,
      "hold_steps": null,
      "noise": {"joint_pos": 0.01, "joint_vel": 0.5,
                "projected_gravity": 0.05, "base_ang_vel": 0.2},
      "output": "<worker_result.json path>"
    }

``"noise": null`` (or all-zero magnitudes) selects the deterministic protocol
used by the cross-backend calibration certificate.

Episode-status semantics (the ``LatentRollout.survived`` property is
deliberately NOT used — it treats only ``"completed"`` as success, while a
full-reference episode ends as ``"reference_finished"``):

* ``fell``            -> ``fell=true``, termination ``base_too_low``
* ``reference_finished`` -> success, requires complete frame coverage
* ``completed``       -> success (safety-cap hit before the reference ended)
* ``no_command``      -> artifact failure: no score, the row carries an
  ``artifact_failure`` reason and the orchestrator writes ``failure.json``
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

# States are `embodied_control.lowlevel.contracts.RobotState`, a FROZEN
# dataclass that only exists inside the Embodied-Control environment and so
# cannot be named here at type-check time. The noise model touches four of its
# fields — joint_pos, joint_vel, projected_gravity, base_ang_vel — and never
# the anchor fields, which are what the metrics are computed from.
RobotStateLike = Any

INITIAL_ROOT_TOLERANCE_M = 1e-4
INITIAL_JOINT_TOLERANCE_RAD = 1e-4

# SONIC's released policy-group noise, verbatim from its `config.yaml`
# (`enable_corruption: true`). The orchestrator passes these explicitly; this
# mapping is the fallback so the worker is never accidentally noise-free.
DEFAULT_NOISE = {
    "projected_gravity": 0.05,
    "base_ang_vel": 0.2,
    "joint_pos": 0.01,
    "joint_vel": 0.5,
}


def _episode_seed(case: dict) -> int:
    """A distinct, reproducible noise stream per (motion, seed, repeat)."""
    return (
        int(case.get("env_seed", 0)) * 1_000_003
        + int(case["trajectory_rank"]) * 1_009
        + int(case.get("repeat_index", 0))
    )


def _sensed(state: RobotStateLike, noise: dict, rng) -> RobotStateLike:
    """The controller's view of the robot: true state plus sensor noise.

    Returns a copy. The caller keeps the true state for metrics, so MPJPE can
    never measure the noise that was injected. Noise is added without
    renormalizing ``projected_gravity``, matching Isaac's ``AdditiveUniform
    NoiseCfg`` convention and therefore the distribution the policy trained on.
    """
    import dataclasses

    if not noise:
        return state

    def perturb(value, scale: float):
        array = np.asarray(value, dtype=np.float32)
        if scale <= 0.0:
            return array
        return (array + rng.uniform(-scale, scale, array.shape)).astype(np.float32)

    # `RobotState` is a FROZEN dataclass, so this builds a new instance rather
    # than assigning through — `setattr` raises `FrozenInstanceError`. Every
    # perturbed field is a new array and the anchor fields are carried over
    # untouched, so the true state is unaffected either way.
    return dataclasses.replace(
        state,
        joint_pos=perturb(state.joint_pos, noise.get("joint_pos", 0.0)),
        joint_vel=perturb(state.joint_vel, noise.get("joint_vel", 0.0)),
        projected_gravity=perturb(
            state.projected_gravity, noise.get("projected_gravity", 0.0)
        ),
        base_ang_vel=perturb(state.base_ang_vel, noise.get("base_ang_vel", 0.0)),
    )


def rollout_with_noise(
    playground, motion, *, start_frame, max_steps, fall_height_m, hold_steps, noise, rng
):
    """One synchronous-lockstep episode with sensor noise on the policy input.

    Mirrors :meth:`LatentPlayground.rollout` tick for tick. It is reimplemented
    here rather than called because that method builds its own backend, leaving
    no seam to inject noise — and because the true/sensed split has to be
    explicit: the logs that feed forward kinematics take the TRUE state while
    the publisher and tracker take the sensed one.

    With ``noise`` empty this must reproduce ``LatentPlayground.rollout``
    exactly; ``--self-check`` asserts it.
    """
    from embodied_control.lowlevel.command_buffer import (  # ty: ignore[unresolved-import]
        InProcessCommandBuffer,
    )
    from embodied_control.lowlevel.envs.mujoco import (  # ty: ignore[unresolved-import]
        MujocoBackend,
    )
    from embodied_control.lowlevel.latent import (  # ty: ignore[unresolved-import]
        LatentRollout,
    )
    from embodied_control.lowlevel.publishers.latent_perturbation import (  # ty: ignore[unresolved-import]
        LatentPublisher,
    )
    from embodied_control.lowlevel.tracker import (  # ty: ignore[unresolved-import]
        BufferedCommandSource,
        LowLevelTracker,
    )

    manifest = playground.bundle.manifest
    backend = MujocoBackend(
        manifest.action,
        playground.model_path,
        control_hz=playground.control_hz,
        timestep=manifest.rates.physics_dt,
        decimation=manifest.rates.decimation,
    )
    backend.reset()
    playground._place_on_reference(backend, motion, int(start_frame))

    source = playground.make_reference_source(motion, start_frame=int(start_frame))
    buffer = InProcessCommandBuffer()
    publisher = LatentPublisher(
        buffer, source, playground.command, hold_steps=hold_steps
    )
    tracker = LowLevelTracker(
        playground.bundle,
        playground.policy,
        BufferedCommandSource(buffer, control_hz=playground.control_hz),
    )
    tracker.reset(_sensed(backend.read_state(), noise, rng))

    joint_log: list[np.ndarray] = []
    anchor_log: list[np.ndarray] = []
    height_log: list[float] = []
    frame_log: list[int] = []
    action_log: list[np.ndarray] = []
    status = "completed"
    steps = 0
    for step in range(int(max_steps)):
        state = backend.read_state()
        # TRUE state feeds the metric pipeline.
        joint_log.append(np.asarray(state.joint_pos, dtype=np.float32).copy())
        anchor_log.append(
            np.concatenate([state.anchor_pos_w, state.anchor_quat_w]).astype(np.float32)
        )
        height_log.append(backend.base_height)
        frame_log.append(int(getattr(source, "cursor", -1)))
        # SENSED state feeds the controller stack, exactly as on hardware.
        sensed = _sensed(state, noise, rng)
        publisher.tick(step, backend.now, sensed)
        joint_command, _ = tracker.step(step, sensed)
        if joint_command is None:
            status = "no_command"
            break
        action_log.append(tracker.last_action.copy())
        backend.write_command(joint_command)
        steps = step + 1
        if backend.base_height < float(fall_height_m):
            status = "fell"
            break
        if publisher.exhausted:
            status = "reference_finished"
            break

    return LatentRollout(
        steps=steps,
        status=status,
        joint_pos=np.stack(joint_log) if joint_log else np.empty((0, 29)),
        anchor_pose_xyzw=np.stack(anchor_log) if anchor_log else np.empty((0, 7)),
        base_height=np.asarray(height_log, dtype=np.float32),
        reference_frame=np.asarray(frame_log, dtype=np.int64),
        z_trace=(
            np.stack(publisher.z_trace)
            if publisher.z_trace
            else np.empty((0, playground.z_dim), dtype=np.float32)
        ),
        action=np.stack(action_log) if action_log else np.empty((0, 29)),
        frames=[],
        video_fps=float(playground.control_hz),
        motion=motion.name,
    )


def _telemetry(rollout) -> dict:
    """Adapt a LatentRollout to the telemetry-record keys the metrics expect."""
    return {
        "reference_frames": rollout.reference_frame,
        "joint_position_log": rollout.joint_pos,
        "anchor_pose_log": rollout.anchor_pose_xyzw,
    }


def _initial_pose_errors(rollout, motion, start_frame: int) -> dict[str, float]:
    """Measure how far tick 0 sits from the reference start pose."""
    root_err = float(
        np.linalg.norm(
            rollout.anchor_pose_xyzw[0, 0:3] - motion.anchor_pos_w[start_frame]
        )
    )
    joint_err = float(
        np.abs(rollout.joint_pos[0] - motion.joint_qpos[start_frame]).max()
    )
    return {"root_pos_m": root_err, "joint_rad": joint_err}


def _frames_consecutive(frames: np.ndarray) -> bool:
    valid = frames[frames >= 0]
    if valid.size <= 1:
        return True
    return bool(np.all(np.diff(valid) == 1))


def run_case(playground, case: dict, job: dict) -> dict:
    # Resolvable only inside the Embodied-Control lowlevel-sim environment.
    from embodied_control.lowlevel.metrics import (  # ty: ignore[unresolved-import]
        oracle_tracking_metrics,
        sonic_success_metrics,
    )

    start_frame = int(case.get("start_frame", 0))
    motion_key = case.get("motion_name")
    motion = playground.arrays.motion(
        motion_key if motion_key is not None else int(case["trajectory_rank"])
    )
    noise = job.get("noise", DEFAULT_NOISE) or {}
    rollout = rollout_with_noise(
        playground,
        motion,
        start_frame=start_frame,
        max_steps=int(job["max_steps"]),
        fall_height_m=float(job["fall_height_m"]),
        hold_steps=job.get("hold_steps"),
        noise=noise,
        rng=np.random.default_rng(_episode_seed(case)),
    )

    row: dict = {
        "trajectory_rank": int(case["trajectory_rank"]),
        "motion_name": motion.name,
        "start_frame": start_frame,
        "env_seed": int(case.get("env_seed", 0)),
        "repeat_index": int(case.get("repeat_index", 0)),
        "status": rollout.status,
        "steps": int(rollout.steps),
        "motion_length": int(motion.length),
        "min_base_height_m": round(rollout.min_base_height, 4),
    }

    if rollout.status == "no_command":
        row["artifact_failure"] = "tracker_received_no_command"
        return row

    initial = _initial_pose_errors(rollout, motion, start_frame)
    row["initial_pose_error"] = initial
    if (
        initial["root_pos_m"] > INITIAL_ROOT_TOLERANCE_M
        or initial["joint_rad"] > INITIAL_JOINT_TOLERANCE_RAD
    ):
        row["artifact_failure"] = (
            "robot not initialized on the reference start pose: "
            f"root {initial['root_pos_m']:.2e} m, joint {initial['joint_rad']:.2e} rad"
        )
        return row

    if not _frames_consecutive(rollout.reference_frame):
        row["artifact_failure"] = "reference frames are not consecutive"
        return row

    telemetry = _telemetry(rollout)
    tracking = oracle_tracking_metrics(
        playground.bundle.manifest.action, playground.model_path, motion, telemetry
    )
    tracking.pop("per_frame", None)
    tracking.pop("tracked_bodies", None)
    sonic = sonic_success_metrics(
        playground.bundle.manifest.action, playground.model_path, motion, telemetry
    )

    fell = rollout.status == "fell"
    complete = bool(sonic.get("complete_motion", False))
    row.update(
        {
            "fell": fell,
            "termination": (
                "base_too_low"
                if fell
                else (
                    "reference_finished"
                    if rollout.status == "reference_finished"
                    else "max_steps"
                )
            ),
            # Success under the fall-only protocol: the robot did not fall.
            # `completed` (safety-cap) still counts, matching the Isaac
            # fall-only pass where a truncated episode is not a failure.
            "success": not fell,
            "reference_finished": rollout.status == "reference_finished",
            "complete_motion": complete,
            "frames_scored": int(tracking["ticks_evaluated"]),
            "mpjpe_l_mm": round(float(tracking["mpjpe_l_mm"]), 3),
            "mpjpe_g_mm": round(float(tracking["mpjpe_g_mm"]), 3),
            "mpjpe_l_mm_p95": round(float(tracking["mpjpe_l_mm_p95"]), 3),
            "mpjpe_g_mm_p95": round(float(tracking["mpjpe_g_mm_p95"]), 3),
            "sonic": {
                key: (bool(value) if isinstance(value, (bool, np.bool_)) else value)
                for key, value in sonic.items()
            },
        }
    )
    # Complete coverage for a finished reference, tolerating the one-tick
    # boundary ambiguity of where the cursor sits when exhaustion is detected.
    # `sonic.complete_motion` (an exact arange comparison) is kept as data.
    scored = int(tracking["ticks_evaluated"])
    if rollout.status == "reference_finished" and scored < motion.length - 2:
        row["artifact_failure"] = (
            "reference_finished without complete frame coverage: "
            f"{scored} of {motion.length - 1} frames"
        )
    return row


def self_check(playground, job: dict) -> None:
    """Assert the owned loop reproduces LatentPlayground.rollout with no noise.

    The loop above is a copy of the rig's loop, so it can silently drift from
    it. Running both noise-free on the same motion and requiring identical
    trajectories is what makes the copy trustworthy — and it is the only
    reason the noisy number can be compared with anything the rig produces.
    """
    case = job["cases"][0]
    start_frame = int(case.get("start_frame", 0))
    motion = playground.arrays.motion(
        case.get("motion_name")
        if case.get("motion_name") is not None
        else int(case["trajectory_rank"])
    )
    mine = rollout_with_noise(
        playground,
        motion,
        start_frame=start_frame,
        max_steps=int(job["max_steps"]),
        fall_height_m=float(job["fall_height_m"]),
        hold_steps=job.get("hold_steps"),
        noise={},
        rng=np.random.default_rng(0),
    )
    theirs = playground.rollout(
        playground.make_reference_source(motion, start_frame=start_frame),
        motion=motion,
        max_steps=int(job["max_steps"]),
        hold_steps=job.get("hold_steps"),
        start_pose_frame=start_frame,
        fall_height_m=float(job["fall_height_m"]),
        record_video=False,
    )
    if mine.status != theirs.status or mine.steps != theirs.steps:
        raise AssertionError(
            f"loop drift: {mine.status}/{mine.steps} vs {theirs.status}/{theirs.steps}"
        )
    for name in ("joint_pos", "anchor_pose_xyzw", "reference_frame", "action"):
        if not np.array_equal(getattr(mine, name), getattr(theirs, name)):
            raise AssertionError(f"loop drift in {name}")
    print(
        f"SELF-CHECK OK: owned loop == LatentPlayground.rollout "
        f"({motion.name}, {mine.steps} steps)"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="verify the owned loop matches the rig's, then exit",
    )
    args = parser.parse_args(argv)

    from embodied_control.lowlevel.latent import (  # ty: ignore[unresolved-import]
        LatentPlayground,
    )

    job = json.loads(args.job.read_text())
    started = time.monotonic()
    playground = LatentPlayground(
        job["bundle"], job["model"], reference_root=job["reference_root"]
    )
    load_seconds = time.monotonic() - started

    if args.self_check:
        self_check(playground, job)
        return 0

    episodes = []
    for case in job["cases"]:
        case_started = time.monotonic()
        row = run_case(playground, case, job)
        row["eval_seconds"] = round(time.monotonic() - case_started, 3)
        episodes.append(row)
        print(json.dumps(row), flush=True)

    result = {
        "schema_version": "ec_sidecar_worker_result_v1",
        "execution_mode": "sync_lockstep",
        "observation_noise": job.get("noise", DEFAULT_NOISE) or {},
        "episodes": episodes,
        "runtime": {
            "playground_load_seconds": round(load_seconds, 3),
            "eval_seconds": round(time.monotonic() - started, 3),
        },
    }
    output = Path(job["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(result, indent=2))
    tmp_path.replace(output)
    print(f"RESULT: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
