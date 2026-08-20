#!/usr/bin/env python3
"""Convert a small EgoDex HDF5 clip to a Vega-Wuji joint reference.

This is a hand-only first-stage retargeter.  It uses relative 3-D finger
landmarks, so it does not need a human-to-robot frame calibration.  Arm,
head, and unselected-hand joints stay at zero.  The output is suitable for a
small tracking-policy smoke test, not for a paper result.

EgoDex HDF5 reading requires ``h5py``.  In this repository it is available in
the Isaac Lab Pixi environment, so run this script with
``pixi run -e isaaclab``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from iltools.core.trajectory import Trajectory
from iltools.retarget import (
    KeypointJointSpec,
    KeypointRetargeter,
    save_joint_reference_npz,
)
from imitation_experiments.paths import REPO_ROOT


DEFAULT_MODEL = (
    REPO_ROOT
    / "source/isaaclab_imitation/isaaclab_imitation/assets/vega_wuji"
    / "vega_u_wuji_v2_beta1_with_mount.xml"
)
FINGER_SOURCE_NAMES = {
    "index_finger": "IndexFinger",
    "middle_finger": "MiddleFinger",
    "ring_finger": "RingFinger",
    "pinky": "LittleFinger",
}
TARGET_FINGER_NAMES = {
    "index_finger": "index_finger",
    "middle_finger": "middle_finger",
    "ring_finger": "ring_finger",
    "pinky": "pinky",
}


def _parse_hands(value: str) -> tuple[str, ...]:
    hands = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    if not hands or any(hand not in {"left", "right"} for hand in hands):
        raise argparse.ArgumentTypeError("hands must contain left and/or right")
    if len(set(hands)) != len(hands):
        raise argparse.ArgumentTypeError("hands must be unique")
    return hands


def _source_names(hand: str) -> tuple[str, ...]:
    prefix = hand.lower()
    names = [f"{prefix}Hand"]
    names.extend(
        [
            f"{prefix}Thumb{suffix}"
            for suffix in (
                "Knuckle",
                "IntermediateBase",
                "IntermediateTip",
                "Tip",
            )
        ]
    )
    for source_prefix in FINGER_SOURCE_NAMES.values():
        names.extend(
            [
                f"{prefix}{source_prefix}{suffix}"
                for suffix in (
                    "Knuckle",
                    "IntermediateBase",
                    "IntermediateTip",
                    "Tip",
                )
            ]
        )
    return tuple(names)


def _keypoint_specs(
    hand: str,
    limits: dict[str, tuple[float, float]],
) -> tuple[KeypointJointSpec, ...]:
    """Build the explicit source-chain to target-joint map for one hand."""

    prefix = hand.lower()
    target_prefix = "r" if prefix == "right" else "l"

    def spec(
        target_suffix: str, parent: str, joint: str, child: str
    ) -> KeypointJointSpec:
        target_name = f"{target_prefix}_{target_suffix}"
        lower, upper = limits[target_name]
        # A straight human chain has angle pi.  Vega/Wuji uses zero for the
        # corresponding straight hinge, hence q = pi - human_angle.
        return KeypointJointSpec(
            target_name=target_name,
            parent=parent,
            joint=joint,
            child=child,
            lower=lower,
            upper=upper,
            scale=-1.0,
            offset=float(np.pi),
        )

    source_prefix = f"{prefix}Thumb"
    specs = [
        spec(
            "thumb_cmc_flex",
            f"{prefix}Hand",
            f"{source_prefix}Knuckle",
            f"{source_prefix}IntermediateBase",
        ),
        spec(
            "thumb_mcp",
            f"{source_prefix}Knuckle",
            f"{source_prefix}IntermediateBase",
            f"{source_prefix}IntermediateTip",
        ),
        spec(
            "thumb_ip",
            f"{source_prefix}IntermediateBase",
            f"{source_prefix}IntermediateTip",
            f"{source_prefix}Tip",
        ),
    ]
    for source_key, target_key in TARGET_FINGER_NAMES.items():
        source_prefix = f"{prefix}{FINGER_SOURCE_NAMES[source_key]}"
        specs.extend(
            [
                spec(
                    f"{target_key}_mcp_flex",
                    f"{prefix}Hand",
                    f"{source_prefix}Knuckle",
                    f"{source_prefix}IntermediateBase",
                ),
                spec(
                    f"{target_key}_pip",
                    f"{source_prefix}Knuckle",
                    f"{source_prefix}IntermediateBase",
                    f"{source_prefix}IntermediateTip",
                ),
                spec(
                    f"{target_key}_dip",
                    f"{source_prefix}IntermediateBase",
                    f"{source_prefix}IntermediateTip",
                    f"{source_prefix}Tip",
                ),
            ]
        )
    return tuple(specs)


def _read_hdf5(
    path: Path, *, hand: str, fps: float, confidence_threshold: float
) -> tuple[Trajectory, dict[str, Any]]:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "EgoDex conversion requires h5py. Run this command with "
            "`pixi run -e isaaclab python scripts/data/retarget_egodex.py ...`."
        ) from exc

    names = _source_names(hand)
    with h5py.File(path, "r") as handle:
        if "transforms" not in handle or "confidences" not in handle:
            raise ValueError(f"{path} must contain transforms and confidences groups")
        points: list[np.ndarray] = []
        confidence_sources: dict[str, str] = {}
        frame_count: int | None = None
        for name in names:
            transform_path = f"transforms/{name}"
            confidence_path = f"confidences/{name}"
            if transform_path not in handle:
                raise KeyError(f"HDF5 file is missing {transform_path}")
            if confidence_path in handle:
                confidence_source = confidence_path
            else:
                confidence_source = f"confidences/{hand}Hand"
                if confidence_source not in handle:
                    raise KeyError(
                        f"{path} is missing per-keypoint confidence {confidence_path} "
                        f"and hand confidence {confidence_source}"
                    )
            transforms = np.asarray(handle[transform_path], dtype=np.float64)
            confidence = np.asarray(
                handle[confidence_source], dtype=np.float64
            ).reshape(-1)
            if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
                raise ValueError(f"{transform_path} must have shape [frames, 4, 4]")
            if transforms.shape[0] != confidence.shape[0]:
                raise ValueError(f"{name} transform and confidence lengths differ")
            if frame_count is None:
                frame_count = int(transforms.shape[0])
            elif transforms.shape[0] != frame_count:
                raise ValueError("EgoDex keypoints do not have a common frame count")
            if not np.isfinite(transforms).all() or not np.isfinite(confidence).all():
                raise ValueError(f"EgoDex keypoint {name} contains non-finite values")
            if np.any(confidence < confidence_threshold):
                raise ValueError(
                    f"EgoDex keypoint {name} has confidence below "
                    f"{confidence_threshold:g}; choose another clip or lower the threshold"
                )
            points.append(transforms[:, :3, 3])
            confidence_sources[name] = confidence_source

        assert frame_count is not None
        attributes = {key: _json_value(value) for key, value in handle.attrs.items()}

    trajectory = Trajectory(
        observations={"keypoints": np.stack(points, axis=1)},
        infos={
            "source": "EgoDex HDF5",
            "coordinate_frame": "stationary ARKit origin",
            "keypoint_names": list(names),
            "confidence_sources": confidence_sources,
            "attributes": attributes,
        },
        dt=1.0 / fps,
    )
    return trajectory, attributes


def _json_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _resample_qpos(
    qpos: np.ndarray, *, input_fps: float, output_fps: float
) -> np.ndarray:
    if qpos.ndim != 2 or qpos.shape[0] < 2:
        raise ValueError("qpos must have shape [at least two frames, joints]")
    if input_fps <= 0.0 or output_fps <= 0.0:
        raise ValueError("input_fps and output_fps must be positive")
    output_frames = int(np.floor((qpos.shape[0] - 1) * output_fps / input_fps)) + 1
    source_times = np.arange(qpos.shape[0], dtype=np.float64) / input_fps
    target_times = np.arange(output_frames, dtype=np.float64) / output_fps
    return np.column_stack(
        [
            np.interp(target_times, source_times, qpos[:, joint])
            for joint in range(qpos.shape[1])
        ]
    )


def _model_joint_order_and_limits(
    model_path: Path,
) -> tuple[tuple[str, ...], dict[str, tuple[float, float]]]:
    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "MuJoCo is required to read the Vega-Wuji joint limits. Run with "
            "`pixi run -e isaaclab`."
        ) from exc
    if not model_path.is_file():
        raise FileNotFoundError(f"Vega-Wuji model not found: {model_path}")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    joint_names: list[str] = []
    limits: dict[str, tuple[float, float]] = {}
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name is None:
            raise ValueError(f"Actuator {actuator_id} has no joint name")
        if name in limits:
            raise ValueError(f"MuJoCo actuator order repeats joint {name}")
        joint_names.append(name)
        limits[name] = (
            float(model.jnt_range[joint_id, 0]),
            float(model.jnt_range[joint_id, 1]),
        )
    if len(joint_names) != 59:
        raise ValueError(
            f"Expected the 59-joint Vega-Wuji model, found {len(joint_names)}"
        )
    return tuple(joint_names), limits


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_provenance(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", type=Path, required=True, help="One EgoDex HDF5 clip")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--hands", type=_parse_hands, default=("right",))
    parser.add_argument("--input-fps", type=float, default=30.0)
    parser.add_argument("--output-fps", type=float, default=50.0)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--dataset-name", default="EgoDex")
    parser.add_argument("--dataset-url", default="https://github.com/apple/ml-egodex")
    parser.add_argument("--dataset-license", default="CC BY-NC-ND")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance-output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.input_fps <= 0.0 or args.output_fps <= 0.0:
        raise SystemExit("FPS values must be positive")
    if not 0.0 <= args.confidence_threshold <= 1.0:
        raise SystemExit("confidence threshold must be in [0, 1]")
    hdf5_path = args.hdf5.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    if not hdf5_path.is_file():
        raise FileNotFoundError(f"EgoDex HDF5 file not found: {hdf5_path}")

    joint_names, limits = _model_joint_order_and_limits(model_path)
    trajectories: list[Trajectory] = []
    attributes: dict[str, Any] = {}
    mapped_joint_names: list[str] = []
    for hand in args.hands:
        trajectory, attributes = _read_hdf5(
            hdf5_path,
            hand=hand,
            fps=args.input_fps,
            confidence_threshold=args.confidence_threshold,
        )
        specs = _keypoint_specs(hand, limits)
        retargeted = KeypointRetargeter(
            tuple(trajectory.infos["keypoint_names"]), specs
        ).retarget(trajectory)
        trajectories.append(retargeted)
        mapped_joint_names.extend(spec.target_name for spec in specs)

    qpos = np.zeros(
        (len(trajectories[0].observations["qpos"]), len(joint_names)), dtype=np.float64
    )
    joint_index = {name: index for index, name in enumerate(joint_names)}
    for trajectory in trajectories:
        target_names = tuple(trajectory.infos["retarget"]["target_joint_names"])
        for source_index, name in enumerate(target_names):
            qpos[:, joint_index[name]] = trajectory.observations["qpos"][
                :, source_index
            ]
    qpos = _resample_qpos(qpos, input_fps=args.input_fps, output_fps=args.output_fps)
    output_trajectory = Trajectory(observations={"qpos": qpos})
    output = save_joint_reference_npz(
        output_trajectory,
        args.output,
        joint_names=joint_names,
        fps=args.output_fps,
    )

    provenance_output = args.provenance_output or output.with_suffix(".provenance.json")
    _write_provenance(
        provenance_output,
        {
            "source_dataset": args.dataset_name,
            "source_dataset_url": args.dataset_url,
            "source_dataset_license": args.dataset_license,
            "source_file": str(hdf5_path),
            "source_file_sha256": _sha256(hdf5_path),
            "source_attributes": attributes,
            "source_hands": list(args.hands),
            "confidence_sources": sorted(
                {
                    source
                    for trajectory in trajectories
                    for source in trajectory.infos["confidence_sources"].values()
                }
            ),
            "source_fps": args.input_fps,
            "output_fps": args.output_fps,
            "confidence_threshold": args.confidence_threshold,
            "model": str(model_path),
            "model_sha256": _sha256(model_path),
            "joint_names": list(joint_names),
            "mapped_joint_names": mapped_joint_names,
            "retarget_method": "relative landmark hinge angle; q = pi - human_angle",
            "unmapped_joints": [
                name for name in joint_names if name not in mapped_joint_names
            ],
            "qualification": "demo hand-only retarget; arm/head/unselected-hand joints are zero",
        },
    )
    print(f"reference: {output}")
    print(f"provenance: {provenance_output.resolve()}")
    print(f"frames: {qpos.shape[0]}, joints: {qpos.shape[1]}")
    print(f"mapped joints: {len(set(mapped_joint_names))}")


if __name__ == "__main__":
    main()
