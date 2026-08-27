"""Asynchronous planner board on the Embodied-Control DDS plant.

This is the deployment-rehearsal tier of the planner evaluation. The tracker
runs as the native C++ 50 Hz controller and talks ONLY the Unitree G1 DDS wire
protocol; a MuJoCo plant serves that protocol on interface ``lo``. Nothing in
the loop is stepped by the evaluator, so the robot advances whether or not the
planner replied: a late plan is a real deadline miss, not a pause.

The planner is a separate process (the GR00T head in its own Pixi environment)
behind the native request/response mailboxes. The controller owns the schedule:
it publishes its causal history ``lead_ticks`` before the plan in hand runs
out, walks the reply slot by slot, and holds the last command when a reply is
late. One head call therefore covers ``plan_slots * hold_steps`` control ticks,
which is the cadence the Isaac board's leading row uses.

Runs inside the Embodied-Control ``native`` Pixi environment, which does not
install ``imitation_experiments``; like
:mod:`imitation_experiments.evaluation.ec_sidecar_worker` it therefore imports
only the standard library, numpy, and ``embodied_control``, and is invoked by
absolute path::

    pixi run -e native python <this file> --job job.json

Sensor noise is served by the PLANT, on the wire, because a real G1 does not
publish clean state. The board runs with noise on by default (user directive
2026-08-17).

MPJPE is scored from the plant's own state log, never from the controller: the
hardware wire protocol carries no root pose, so the controller genuinely does
not know where the robot is. That asymmetry is the point of this tier.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import signal
import subprocess
import time
import uuid
from pathlib import Path

import numpy as np

from embodied_control.lowlevel.bundle import PolicyBundle
from embodied_control.lowlevel.native_core import NativeMujocoLoop, NativeUnitreeLoop
from embodied_control.lowlevel.publishers.native_pull import (
    NativeLatentPlanWorker,
    StdioChunkService,
)
from embodied_control.lowlevel.reference import ReferenceArrays

# SONIC's policy-group observation noise as uniform half-ranges. `imu_tilt_rad`
# is the plant-side stand-in for SONIC's additive projected-gravity noise: on
# hardware a gravity error IS an orientation error, so the plant tilts the
# published IMU frame instead of perturbing a derived vector.
DEFAULT_SENSOR_NOISE = {
    "joint_pos": 0.01,
    "joint_vel": 0.5,
    "base_ang_vel": 0.2,
    "imu_tilt_rad": 0.05,
}


def _plant_command(job: dict, pose_path: Path, states_path: Path, report_path: Path):
    noise = {**DEFAULT_SENSOR_NOISE, **(job.get("sensor_noise") or {})}
    return [
        job.get("ec_python") or sys.executable,
        "-m",
        "embodied_control.cli",
        "lowlevel",
        "plant",
        str(job["bundle"]),
        "--model",
        str(job["mjcf"]),
        "--network",
        str(job.get("network", "lo")),
        "--dds-domain",
        str(int(job.get("dds_domain", 0))),
        # Default 0.002 is the plant's own choice; the tracker was trained at
        # the bundle's physics_dt, so a rehearsal that wants matched dynamics
        # sets this to that value.
        "--timestep",
        str(float(job.get("plant_timestep", 0.002))),
        "--seconds",
        str(float(job["_episode_seconds"]) + float(job.get("plant_margin_s", 8.0))),
        "--initial-pose",
        str(pose_path),
        "--states",
        str(states_path),
        "--report",
        str(report_path),
        "--noise-joint-pos",
        str(noise["joint_pos"]),
        "--noise-joint-vel",
        str(noise["joint_vel"]),
        "--noise-base-ang-vel",
        str(noise["base_ang_vel"]),
        "--noise-imu-tilt-rad",
        str(noise["imu_tilt_rad"]),
        "--noise-seed",
        str(int(job.get("noise_seed", 0))),
        # The gantry holds the reference start pose until the controller's
        # first command lands, so booting the controller costs no fall.
        *(["--freeze-until-command"] if job.get("freeze_until_command", True) else []),
    ]


def _wait_for_plant(process, timeout_s: float = 60.0) -> None:
    deadline = time.monotonic() + timeout_s
    lines: list[str] = []
    while time.monotonic() < deadline:
        line = process.stdout.readline()
        if not line:
            break
        lines.append(line)
        if "PLANT_READY" in line:
            return
    process.kill()
    raise RuntimeError("the DDS plant did not become ready:\n" + "".join(lines))


def _start_pose(motion, frame: int) -> np.ndarray:
    frame = int(np.clip(frame, 0, motion.length - 1))
    return np.concatenate(
        [
            np.asarray(motion.anchor_pos_w[frame], dtype=np.float32),
            np.asarray(motion.anchor_quat_w[frame], dtype=np.float32),
            np.asarray(motion.joint_qpos[frame], dtype=np.float32),
        ]
    ).astype(np.float32)


def _run_mujoco_episode(
    job: dict, motion, episode: dict, index: int, service, output: Path
):
    """The same controller and planner against in-process MuJoCo.

    One rung below the DDS tier: identical native runtime, identical plan
    protocol, but no wire, no plant process, and no hardware state machine.
    A row that tracks here and falls on the plant blames the transport tier,
    not the planner.
    """
    control_hz = float(job.get("control_hz", 50.0))
    seconds = job.get("episode_seconds")
    if seconds in (None, 0):
        seconds = float(motion.length) / control_hz
    seconds = min(float(seconds), float(job.get("max_episode_seconds", 1e9)))
    ticks = int(round(seconds * control_hz))
    stem = f"ep{index:04d}"
    token = uuid.uuid4().hex[:8]
    request_slot = f"/ec_mj_req_{os.getpid()}_{token}"
    response_slot = f"/ec_mj_res_{os.getpid()}_{token}"
    record: dict = {
        "episode": index,
        "motion": motion.name,
        "start_frame": int(episode.get("start_frame", 0)),
        "repeat": int(episode.get("repeat", 0)),
        "requested_ticks": ticks,
        "episode_seconds": seconds,
        "tier": "mujoco_native",
    }
    worker = None
    try:
        service.goal = motion.name
        worker = NativeLatentPlanWorker(
            request_slot,
            response_slot,
            service,
            z_dim=int(job["z_dim"]),
            plan_slots=int(job["plan_slots"]),
            hold_steps=int(job["hold_steps"]),
            lead_ticks=int(job["lead_ticks"]),
            create_slots=True,
        )
        worker.start()
        loop = NativeMujocoLoop(
            PolicyBundle.load(job["bundle"]),
            str(job["mjcf"]),
            response_slot=response_slot,
            request_slot=request_slot,
            create_slots=False,
            hold_steps=int(job["hold_steps"]),
            lead_ticks=int(job["lead_ticks"]),
            plan_slots=int(job["plan_slots"]),
            latent_plan=True,
            command_stale_ms=float(job.get("command_stale_ms", 500.0)),
            # The plant serves noise on the wire; on this tier the backend
            # puts the same SONIC noise on the controller's view instead.
            sensor_noise={
                "joint_pos": DEFAULT_SENSOR_NOISE["joint_pos"],
                "joint_vel": DEFAULT_SENSOR_NOISE["joint_vel"],
                "base_ang_vel": DEFAULT_SENSOR_NOISE["base_ang_vel"],
                "projected_gravity": 0.05,
                **(job.get("sensor_noise") or {}),
            },
            noise_seed=int(job.get("noise_seed", 0)) + int(episode.get("repeat", 0)),
        )
        loop.set_initial_pose(_start_pose(motion, int(episode.get("start_frame", 0))))
        loop.start(ticks, paced=bool(job.get("paced", True)))
        loop.wait()
        record["stats"] = {k: _plain(v) for k, v in loop.stats().items()}
        record["planner_calls"] = int(worker.requests)
        record["planner_request_ms_mean"] = (
            float(np.mean(worker.request_ms)) if worker.request_ms else None
        )
        record["planner_error"] = (
            None if worker.last_error is None else str(worker.last_error)
        )
        joints = np.asarray(loop.joint_position_log()).reshape(-1, 29)
        anchors = np.asarray(loop.anchor_pose_log()).reshape(-1, 7)
        states_path = output / f"states_{stem}.npz"
        np.savez(
            states_path,
            joint_pos=joints,
            anchor_pose_xyzw=anchors,
            base_heights=np.asarray(loop.base_heights()),
        )
        record["states_path"] = str(states_path)
        record["min_base_height"] = float(np.nanmin(loop.base_heights()))
    finally:
        if worker is not None:
            worker.close()
    return record


def _run_episode(job: dict, motion, episode: dict, index: int, service, output: Path):
    """One paced episode: plant process, controller, planner worker."""
    control_hz = float(job.get("control_hz", 50.0))
    # Each motion runs its own length unless the job fixes a horizon, so a
    # short clip is not scored against dead time at the end.
    seconds = job.get("episode_seconds")
    if seconds in (None, 0):
        seconds = float(motion.length) / control_hz
    seconds = min(float(seconds), float(job.get("max_episode_seconds", 1e9)))
    ticks = int(round(seconds * control_hz))
    episode = {**episode, "_seconds": seconds}
    stem = f"ep{index:04d}"
    pose_path = output / f"{stem}_start_pose.npy"
    states_path = output / f"states_{stem}.npz"
    plant_report_path = output / f"{stem}_plant.json"
    np.save(pose_path, _start_pose(motion, int(episode.get("start_frame", 0))))

    plant = subprocess.Popen(
        _plant_command(
            {**job, "_episode_seconds": seconds},
            pose_path,
            states_path,
            plant_report_path,
        ),
        cwd=str(job["ec_repo"]),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    loop = None
    worker = None
    token = uuid.uuid4().hex[:8]
    request_slot = f"/ec_board_req_{os.getpid()}_{token}"
    response_slot = f"/ec_board_res_{os.getpid()}_{token}"
    record: dict = {
        "episode": index,
        "motion": motion.name,
        "start_frame": int(episode.get("start_frame", 0)),
        "repeat": int(episode.get("repeat", 0)),
        "requested_ticks": ticks,
        "episode_seconds": seconds,
    }
    try:
        _wait_for_plant(plant)
        service.goal = motion.name
        worker = NativeLatentPlanWorker(
            request_slot,
            response_slot,
            service,
            z_dim=int(job["z_dim"]),
            plan_slots=int(job["plan_slots"]),
            hold_steps=int(job["hold_steps"]),
            lead_ticks=int(job["lead_ticks"]),
            create_slots=True,
        )
        worker.start()
        loop = NativeUnitreeLoop(
            PolicyBundle.load(job["bundle"]),
            str(job.get("network", "lo")),
            response_slot=response_slot,
            request_slot=request_slot,
            writes_enabled=True,
            create_slots=False,
            hold_steps=int(job["hold_steps"]),
            lead_ticks=int(job["lead_ticks"]),
            plan_slots=int(job["plan_slots"]),
            latent_plan=True,
            command_stale_ms=float(job.get("command_stale_ms", 500.0)),
            control_cpu=int(job.get("control_cpu", -1)),
            writer_cpu=int(job.get("writer_cpu", -1)),
            control_fifo_priority=int(job.get("control_priority", 0)),
            writer_fifo_priority=int(job.get("writer_priority", 0)),
            lock_memory=bool(job.get("lock_memory", False)),
            require_realtime=bool(job.get("require_realtime", False)),
            dds_domain=int(job.get("dds_domain", 0)),
            policy_threads=int(job.get("policy_threads", 2)),
        )
        if not loop.wait_for_state(20.0):
            raise RuntimeError("no rt/lowstate from the plant before the timeout")

        # Order matters. The gantry lets go the moment the writer publishes its
        # first command, and a reference frame is usually a mid-stride pose
        # that topples in a few hundred milliseconds of static holding. So run
        # the control thread FIRST with the write gate still shut: it requests
        # a plan and fills the command slot while the robot is still frozen.
        # Only then initialise (a no-op ramp that holds the pose) and arm, so
        # the planner takes over within a tick or two of the release.
        loop.start(ticks, paced=True)
        plan_deadline = time.monotonic() + 30.0
        while int(loop.stats()["control_ticks"]) == 0:
            if not loop.running or time.monotonic() > plan_deadline:
                raise RuntimeError(
                    f"no planner command before the arm timeout: stats={loop.stats()}"
                )
            time.sleep(0.005)
        loop.begin_initialization(
            float(job.get("init_seconds", 0.05)),
            hold_current=bool(job.get("init_hold_current", True)),
            # There is no Unitree motion service in front of a simulated
            # plant, and the client's CheckMode blocks for its whole timeout -
            # long enough for the running controller to starve and damp.
            skip_motion_switcher=bool(job.get("skip_motion_switcher", True)),
        )
        if not loop.wait_for_mode(NativeUnitreeLoop.WAIT, 10.0):
            raise RuntimeError(
                "the controller never reached WAIT: "
                f"mode={loop.unitree_mode} writer={loop.writer_stats()}"
            )
        loop.arm_control()
        armed = True
        while loop.running:
            time.sleep(0.02)
        loop.wait()
        record["armed"] = armed
        record["stats"] = {k: _plain(v) for k, v in loop.stats().items()}
        record["writer_stats"] = {k: _plain(v) for k, v in loop.writer_stats().items()}
        record["planner_request_ms_mean"] = (
            float(np.mean(worker.request_ms)) if worker.request_ms else None
        )
        record["planner_request_ms_p99"] = (
            float(np.percentile(worker.request_ms, 99)) if worker.request_ms else None
        )
        record["planner_calls"] = int(worker.requests)
        record["planner_error"] = (
            None if worker.last_error is None else str(worker.last_error)
        )
    finally:
        if loop is not None:
            loop.force_damp()
            loop.stop()
            try:
                loop.wait()
            except Exception:  # noqa: BLE001 - teardown must not mask a fault
                pass
        if worker is not None:
            worker.close()
        if plant.poll() is None:
            plant.send_signal(signal.SIGINT)
        try:
            plant.communicate(timeout=30.0)
        except subprocess.TimeoutExpired:
            plant.kill()
            plant.communicate()
    if plant_report_path.is_file():
        record["plant"] = json.loads(plant_report_path.read_text())
    record["states_path"] = str(states_path) if states_path.is_file() else None
    return record


def _plain(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args(argv)
    job = json.loads(args.job.read_text())

    output = Path(job["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    arrays = ReferenceArrays(job["reference_root"])

    # One head process for the whole board: loading it costs far more than an
    # episode, and the service switches language goals per request.
    service = StdioChunkService(
        list(job["service_command"]),
        action_width=int(job["z_dim"]),
    )
    records = []
    started = time.monotonic()
    try:
        for index, episode in enumerate(job["episodes"]):
            motion = arrays.motion(episode["motion"])
            runner = (
                _run_mujoco_episode
                if str(job.get("tier", "dds_plant")) == "mujoco_native"
                else _run_episode
            )
            record = runner(job, motion, episode, index, service, output)
            records.append(record)
            print(
                json.dumps(
                    {
                        "episode": index,
                        "motion": record["motion"],
                        "deadline_misses": record.get("stats", {}).get(
                            "deadline_misses"
                        ),
                        "plan_slot_advances": record.get("stats", {}).get(
                            "plan_slot_advances"
                        ),
                        "planner_calls": record.get("planner_calls"),
                        "fault": record.get("stats", {}).get("fault"),
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
    finally:
        service.close()

    summary = {
        "schema": "ec_dds_planner_board_v1",
        "label": job.get("label", "ec_dds_planner_board"),
        "tier": str(job.get("tier", "dds_plant")),
        "execution": "async_service",
        "bundle": str(job["bundle"]),
        "mjcf": str(job["mjcf"]),
        "reference_root": str(job["reference_root"]),
        "service_command": list(job["service_command"]),
        "plan_slots": int(job["plan_slots"]),
        "hold_steps": int(job["hold_steps"]),
        "lead_ticks": int(job["lead_ticks"]),
        "z_dim": int(job["z_dim"]),
        "control_hz": float(job.get("control_hz", 50.0)),
        "episode_seconds": job.get("episode_seconds"),
        "sensor_noise": {**DEFAULT_SENSOR_NOISE, **(job.get("sensor_noise") or {})},
        "noise_seed": int(job.get("noise_seed", 0)),
        "dds_domain": int(job.get("dds_domain", 0)),
        "wall_clock_s": round(time.monotonic() - started, 1),
        "episodes": records,
    }
    (output / "board.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"board": str(output / "board.json")}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
