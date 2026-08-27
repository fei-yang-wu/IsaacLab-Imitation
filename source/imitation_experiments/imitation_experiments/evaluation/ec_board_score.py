"""Score an Embodied-Control planner board into the canonical row.

Reads the ``board.json`` an
:mod:`imitation_experiments.evaluation.ec_dds_board_worker` run writes, replays
each episode's saved joint trajectory through MuJoCo forward kinematics, and
reports MPJPE-L / MPJPE-G against the reference plus the fall-free rate.

Metrics come from the CLEAN simulator state. On the DDS tier that is the
plant's own state log, because the hardware wire protocol carries no root pose;
on the in-process tier it is the backend's metric log. Sensor noise reaches the
controller's view only, so MPJPE never measures the noise that was injected.

The headline number is the episode-mean MPJPE-L, which is the aggregation the
Isaac board's paper rows use. The step-weighted micro mean is reported beside
it because the same summary file carries both and they differ by a few
millimetres.

Runs inside the Embodied-Control ``native`` or ``lowlevel-sim`` Pixi
environment, by absolute path::

    pixi run -e native python <this file> --board <run>/board.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from embodied_control.lowlevel.bundle import PolicyBundle
from embodied_control.lowlevel.eval_mpjpe import _align_reference
from embodied_control.lowlevel.metrics import compute_mpjpe, fk_body_positions
from embodied_control.lowlevel.reference import ReferenceArrays


def _episode_row(board: dict, record: dict, arrays, bundle, mjcf: str, fall_m: float):
    states_path = record.get("states_path")
    if not states_path or not Path(states_path).is_file():
        return {**_identity(record), "artifact_failure": "no states file"}
    states = np.load(states_path)
    if "anchor_pose_xyzw" in states:
        joint_pos = np.asarray(states["joint_pos"], dtype=np.float32)
        anchor = np.asarray(states["anchor_pose_xyzw"], dtype=np.float32)
        if "base_heights" in states:
            base_heights = np.asarray(states["base_heights"], dtype=np.float32)
        else:
            # The lockstep runner logs no separate height channel; the anchor
            # pose is the pelvis in world, so its z IS the base height.
            base_heights = anchor[:, 2]
    else:
        # DDS tier: the plant's own log, [pos 3 | quat XYZW 4 | joints 29].
        joint_pos = np.asarray(states["joint_pos"], dtype=np.float32)
        anchor = np.concatenate(
            [
                np.asarray(states["root_pos"], dtype=np.float32),
                np.asarray(states["root_quat_xyzw"], dtype=np.float32),
            ],
            axis=1,
        )
        base_heights = np.asarray(states["root_pos"], dtype=np.float32)[:, 2]
    finite = np.isfinite(joint_pos).all(axis=1) & np.isfinite(anchor).all(axis=1)
    joint_pos = joint_pos[finite]
    anchor = anchor[finite]
    if joint_pos.shape[0] < 2:
        return {**_identity(record), "artifact_failure": "no finite frames"}

    motion = arrays.motion(record["motion"])
    if motion.body_pos_w is None:
        return {**_identity(record), "artifact_failure": "reference has no bodies"}
    start = int(record.get("start_frame", 0))
    frames = np.minimum(np.arange(joint_pos.shape[0]) + start, motion.length - 1)
    robot_body = fk_body_positions(
        mjcf, bundle.manifest.action, joint_pos, anchor, arrays.body_names
    )
    reference_body = motion.body_pos_w[frames]
    aligned_body = _align_reference(
        anchor[0],
        motion.anchor_pos_w[start],
        motion.anchor_quat_w[start],
        reference_body.reshape(-1, 3),
    ).reshape(reference_body.shape)
    aligned_root = _align_reference(
        anchor[0],
        motion.anchor_pos_w[start],
        motion.anchor_quat_w[start],
        motion.anchor_pos_w[frames],
    )
    scores = compute_mpjpe(robot_body, anchor[:, 0:3], aligned_body, aligned_root)
    scores.pop("per_frame_mpjpe_g_mm", None)
    scores.pop("per_frame_mpjpe_l_mm", None)
    min_height = float(np.nanmin(base_heights)) if base_heights.size else float("nan")
    stats = record.get("stats") or {}
    return {
        **_identity(record),
        **scores,
        "frames_scored": int(joint_pos.shape[0]),
        "frame_coverage": float(joint_pos.shape[0]) / float(motion.length),
        "reference_frames": int(motion.length),
        "min_base_height_m": min_height,
        "fell": bool(min_height < fall_m),
        "deadline_misses": int(stats.get("deadline_misses", 0)),
        "plan_slot_advances": int(stats.get("plan_slot_advances", 0)),
        "planner_calls": int(record.get("planner_calls", 0) or 0),
        "planner_request_ms_mean": record.get("planner_request_ms_mean"),
        "fault": int(stats.get("fault", 0)),
        "control_ticks": int(stats.get("control_ticks", 0)),
    }


def _identity(record: dict) -> dict:
    return {
        "episode": record.get("episode"),
        "motion": record.get("motion"),
        "repeat": record.get("repeat", 0),
        "start_frame": record.get("start_frame", 0),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board", type=Path, required=True)
    parser.add_argument("--fall-height-m", type=float, default=0.4)
    parser.add_argument("--output", type=Path, default=None)
    # Older boards did not record these; pass them explicitly for those runs.
    parser.add_argument("--mjcf", default=None)
    parser.add_argument("--reference-root", default=None)
    args = parser.parse_args(argv)

    board = json.loads(args.board.read_text())
    job_bundle = board["bundle"]
    bundle = PolicyBundle.load(job_bundle)
    reference_root = args.reference_root or board.get("reference_root")
    mjcf = args.mjcf or board.get("mjcf")
    if not reference_root or not mjcf:
        raise SystemExit(
            "this board records no mjcf/reference_root; pass --mjcf and "
            "--reference-root"
        )
    arrays = ReferenceArrays(reference_root)

    rows = [
        _episode_row(board, record, arrays, bundle, mjcf, args.fall_height_m)
        for record in board["episodes"]
    ]
    scored = [row for row in rows if "mpjpe_l_mm" in row]
    upright = [row for row in scored if not row["fell"]]
    summary = {
        "schema": "ec_planner_board_score_v1",
        "label": board.get("label"),
        "tier": board.get("tier"),
        "execution": board.get("execution"),
        "plan_slots": board.get("plan_slots"),
        "hold_steps": board.get("hold_steps"),
        "lead_ticks": board.get("lead_ticks"),
        "sensor_noise": board.get("sensor_noise"),
        "episodes": len(rows),
        "scored_episodes": len(scored),
        "fall_free_rate": (
            float(len(upright)) / float(len(scored)) if scored else None
        ),
        # The paper aggregation: mean over episodes of each episode's mean.
        "mpjpe_l_mm_episode_mean": (
            float(np.mean([row["mpjpe_l_mm"] for row in scored])) if scored else None
        ),
        "mpjpe_l_mm_episode_mean_upright": (
            float(np.mean([row["mpjpe_l_mm"] for row in upright])) if upright else None
        ),
        "mpjpe_g_mm_episode_mean": (
            float(np.mean([row["mpjpe_g_mm"] for row in scored])) if scored else None
        ),
        "deadline_misses_total": int(
            sum(row.get("deadline_misses", 0) for row in scored)
        ),
        "planner_calls_total": int(sum(row.get("planner_calls", 0) for row in scored)),
        "planner_request_ms_mean": (
            float(
                np.mean(
                    [
                        row["planner_request_ms_mean"]
                        for row in scored
                        if row.get("planner_request_ms_mean")
                    ]
                )
            )
            if scored
            else None
        ),
        "faulted_episodes": int(sum(1 for row in scored if row.get("fault"))),
        "rows": rows,
    }
    output = args.output or args.board.parent / "score.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "label",
                    "tier",
                    "episodes",
                    "scored_episodes",
                    "fall_free_rate",
                    "mpjpe_l_mm_episode_mean",
                    "deadline_misses_total",
                    "planner_calls_total",
                    "faulted_episodes",
                )
            },
            indent=2,
        )
    )
    print(f"score: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
