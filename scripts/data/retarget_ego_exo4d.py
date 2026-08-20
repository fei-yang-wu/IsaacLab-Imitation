#!/usr/bin/env python3
"""Convert one Ego-Exo4D pose annotation to a named joint reference NPZ."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from iltools.retarget import (
    KeypointJointSpec,
    KeypointRetargeter,
    load_ego_pose_bundle,
    load_ego_pose_trajectory,
    MujocoKeypointRetargeter,
    MujocoPositionTaskSpec,
    save_joint_reference_npz,
    transform_keypoint_trajectory,
)


def _parse_keypoint_names(value: str) -> tuple[str, ...]:
    names = tuple(name.strip() for name in value.split(",") if name.strip())
    if not names:
        raise argparse.ArgumentTypeError("At least one keypoint name is required.")
    if len(set(names)) != len(names):
        raise argparse.ArgumentTypeError("Keypoint names must be unique.")
    return names


def _load_specs(path: Path) -> tuple[KeypointJointSpec, ...]:
    with path.open(encoding="utf-8") as handle:
        payload: Any = json.load(handle)
    if isinstance(payload, dict):
        payload = payload.get("specs")
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{path} must contain a non-empty JSON spec list.")
    try:
        return tuple(KeypointJointSpec(**item) for item in payload)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid keypoint joint spec in {path}: {exc}") from exc


def _load_position_tasks(path: Path) -> tuple[MujocoPositionTaskSpec, ...]:
    with path.open(encoding="utf-8") as handle:
        payload: Any = json.load(handle)
    if isinstance(payload, dict):
        payload = payload.get("tasks")
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{path} must contain a non-empty JSON task list.")
    try:
        return tuple(MujocoPositionTaskSpec(**item) for item in payload)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid MuJoCo position task in {path}: {exc}") from exc


def _load_transform(path: Path) -> tuple[Any, Any, float]:
    with path.open(encoding="utf-8") as handle:
        payload: Any = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a transform object.")
    try:
        rotation = payload["rotation"]
        translation = payload["translation"]
        scale = float(payload.get("scale", 1.0))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{path} must contain rotation, translation, and optional scale."
        ) from exc
    return rotation, translation, scale


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retarget an Ego-Exo4D 3-D pose annotation to joint NPZ."
    )
    parser.add_argument("--annotation", type=Path, action="append", required=True)
    parser.add_argument(
        "--specs-json",
        type=Path,
        help="Hinge specs JSON for the dependency-light retargeter.",
    )
    parser.add_argument(
        "--mujoco-model",
        type=Path,
        help="MJCF model for model-based DLS keypoint retargeting.",
    )
    parser.add_argument(
        "--position-tasks-json",
        type=Path,
        help="Position task JSON used with --mujoco-model.",
    )
    parser.add_argument(
        "--target-joint-names",
        type=_parse_keypoint_names,
        help="Comma-separated hinge/slide joints used with --mujoco-model.",
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--damping", type=float, default=0.02)
    parser.add_argument(
        "--transform-json",
        type=Path,
        help="Explicit world-to-robot transform JSON for 3-D keypoints.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument(
        "--keypoint-names",
        type=_parse_keypoint_names,
        required=True,
        help="Comma-separated annotation3D landmark names in source order.",
    )
    parser.add_argument(
        "--keypoint-group",
        type=_parse_keypoint_names,
        action="append",
        help=(
            "Comma-separated names for one repeated --annotation file. "
            "Use once per file when body and hand annotations are separate."
        ),
    )
    parser.add_argument(
        "--resample-gaps",
        action="store_true",
        help=(
            "Linearly resample valid EgoPose timestamps onto a contiguous "
            "frame grid before retargeting."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mujoco_model is not None:
        if args.position_tasks_json is None or args.target_joint_names is None:
            raise SystemExit(
                "--mujoco-model requires --position-tasks-json and "
                "--target-joint-names."
            )
        if args.specs_json is not None:
            raise SystemExit("Use either --specs-json or --mujoco-model, not both.")
        position_tasks = _load_position_tasks(args.position_tasks_json)
        retargeter = MujocoKeypointRetargeter(
            args.mujoco_model,
            args.target_joint_names,
            position_tasks,
            iterations=args.iterations,
            damping=args.damping,
        )
        target_joint_names = args.target_joint_names
    else:
        if args.specs_json is None:
            raise SystemExit(
                "Provide --specs-json or use --mujoco-model for model-based retargeting."
            )
        if args.position_tasks_json is not None or args.target_joint_names is not None:
            raise SystemExit(
                "--position-tasks-json and --target-joint-names require --mujoco-model."
            )
        specs = _load_specs(args.specs_json)
        retargeter = KeypointRetargeter(args.keypoint_names, specs)
        target_joint_names = tuple(spec.target_name for spec in specs)

    if len(args.annotation) == 1:
        if args.keypoint_group is not None:
            raise SystemExit("--keypoint-group requires repeated --annotation.")
        trajectory = load_ego_pose_trajectory(
            args.annotation[0],
            keypoint_names=args.keypoint_names,
            fps=args.fps,
            strict=True,
            resample_gaps=args.resample_gaps,
        )
    else:
        if args.keypoint_group is None or len(args.keypoint_group) != len(
            args.annotation
        ):
            raise SystemExit(
                "Repeated --annotation requires one --keypoint-group per file."
            )
        merged_names = tuple(name for group in args.keypoint_group for name in group)
        if merged_names != args.keypoint_names:
            raise SystemExit(
                "--keypoint-names must equal the concatenated --keypoint-group "
                "names in order."
            )
        trajectory = load_ego_pose_bundle(
            args.annotation,
            keypoint_names_by_file=args.keypoint_group,
            fps=args.fps,
            strict=True,
            resample_gaps=args.resample_gaps,
        )
    if args.transform_json is not None:
        rotation, translation, scale = _load_transform(args.transform_json)
        trajectory = transform_keypoint_trajectory(
            trajectory,
            rotation=rotation,
            translation=translation,
            scale=scale,
        )
    retargeted = retargeter.retarget(trajectory)
    output = save_joint_reference_npz(
        retargeted,
        args.output,
        joint_names=target_joint_names,
        fps=args.fps,
    )
    print(output)


if __name__ == "__main__":
    main()
