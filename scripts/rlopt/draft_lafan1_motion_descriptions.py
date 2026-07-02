#!/usr/bin/env python3
"""Draft LaFAN1 language descriptions from offline NPZ motion statistics.

The output JSON is intentionally compatible with
``build_language_goal_embeddings.py --prompt_json``. Each motion is keyed by its
raw manifest name and stores prompt tiers such as ``robot_instruction``,
``kinematic_description``, ``event_level``, and ``attribute_text``.

This script is a drafting aid, not a final annotator. It computes objective
motion statistics, creates deterministic first-pass captions, and optionally
writes a manifest copy with ``language`` fields for later review.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from language_prompts import humanize_motion_name, prompt_for_motion

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (
    REPO_ROOT / "data" / "lafan1" / "manifests" / "g1_lafan1_manifest.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "data"
    / "lafan1"
    / "language"
    / "g1_lafan1_motion_descriptions.draft.json"
)

SOURCE_TAG = "codex_motion_stats_v0"


def _extract_manifest_entries(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        entries = data.get("dataset", {}).get("trajectories", {}).get("lafan1_csv")
        if entries is None:
            entries = data.get("lafan1_csv", data.get("motions"))
    else:
        entries = data
    if not isinstance(entries, list) or not entries:
        raise ValueError(
            "Manifest must define a non-empty 'dataset.trajectories.lafan1_csv' list."
        )
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Manifest entry #{index} must be a mapping.")
    return entries


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_motion_path(entry: dict[str, Any], manifest_dir: Path) -> Path:
    raw_path = entry.get("path") or entry.get("file")
    if raw_path is None:
        raise ValueError(f"Manifest entry needs 'path' or 'file': {entry}")
    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (manifest_dir / path).resolve()


def _frame_slice(frame_range: Any, total_frames: int) -> slice:
    if frame_range is None:
        return slice(None)
    if (
        not isinstance(frame_range, list | tuple)
        or len(frame_range) != 2
        or total_frames <= 0
    ):
        raise ValueError(f"Invalid frame_range: {frame_range!r}")
    start = int(frame_range[0])
    end = int(frame_range[1])
    if start < 0 or end < start:
        raise ValueError(f"Invalid frame_range bounds: {frame_range!r}")

    # Existing generated manifests use [1, frame_count] for full-range clips.
    # Treat positive starts as 1-based inclusive and zero starts as 0-based.
    if start == 0:
        zero_start = 0
        zero_stop = min(total_frames, end + 1)
    else:
        zero_start = max(0, start - 1)
        zero_stop = min(total_frames, end)
    if zero_stop <= zero_start:
        raise ValueError(
            f"frame_range {frame_range!r} selects no frames from {total_frames}."
        )
    return slice(zero_start, zero_stop)


def _as_float(value: float | None, digits: int = 4) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return round(float(value), digits)


def _percentile(values: np.ndarray, q: float) -> float | None:
    if values.size == 0:
        return None
    return float(np.nanpercentile(values, q))


def _finite_mean(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(np.mean(finite))


def _get_array(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray | None:
    if key not in data.files:
        return None
    array = np.asarray(data[key])
    if not np.issubdtype(array.dtype, np.number):
        return None
    return array.astype(np.float64, copy=False)


def _infer_total_frames(data: np.lib.npyio.NpzFile) -> int:
    for key in (
        "root_pos",
        "qpos",
        "joint_pos",
        "body_pos_w",
        "root_lin_vel",
        "joint_vel",
    ):
        if key in data.files:
            array = np.asarray(data[key])
            if array.ndim >= 1 and array.shape[0] > 0:
                return int(array.shape[0])
    raise ValueError("Could not infer frame count from NPZ arrays.")


def _slice_time(array: np.ndarray | None, frame_slice: slice) -> np.ndarray | None:
    if array is None or array.ndim == 0:
        return array
    return array[frame_slice]


def _root_pos(data: np.lib.npyio.NpzFile, frame_slice: slice) -> np.ndarray:
    root_pos = _slice_time(_get_array(data, "root_pos"), frame_slice)
    if root_pos is None and "qpos" in data.files:
        qpos = _slice_time(_get_array(data, "qpos"), frame_slice)
        if qpos is not None and qpos.ndim == 2 and qpos.shape[1] >= 3:
            root_pos = qpos[:, :3]
    if root_pos is None or root_pos.ndim != 2 or root_pos.shape[1] < 3:
        raise ValueError("NPZ must contain root_pos[T,3] or qpos[T,>=3].")
    return np.asarray(root_pos[:, :3], dtype=np.float64)


def _root_lin_vel(
    data: np.lib.npyio.NpzFile,
    frame_slice: slice,
    root_pos: np.ndarray,
    fps: float,
) -> np.ndarray:
    root_vel = _slice_time(_get_array(data, "root_lin_vel"), frame_slice)
    if root_vel is not None and root_vel.ndim == 2 and root_vel.shape[1] >= 3:
        return np.asarray(root_vel[:, :3], dtype=np.float64)
    if root_pos.shape[0] <= 1:
        return np.zeros_like(root_pos)
    return np.gradient(root_pos, axis=0) * float(fps)


def _joint_names(data: np.lib.npyio.NpzFile) -> list[str]:
    if "joint_names" not in data.files:
        return []
    return [str(name) for name in np.asarray(data["joint_names"]).reshape(-1).tolist()]


def _indices_containing(names: list[str], tokens: tuple[str, ...]) -> list[int]:
    return [
        index
        for index, name in enumerate(names)
        if any(token in name.lower() for token in tokens)
    ]


def _joint_activity(
    joint_vel: np.ndarray | None,
    joint_names: list[str],
) -> dict[str, float | None]:
    if joint_vel is None or joint_vel.ndim != 2 or joint_vel.size == 0:
        return {
            "joint_mean_abs_vel": None,
            "joint_p95_abs_vel": None,
            "arm_mean_abs_vel": None,
            "leg_mean_abs_vel": None,
            "torso_mean_abs_vel": None,
            "arm_leg_activity_ratio": None,
        }
    abs_vel = np.abs(joint_vel)
    arm_indices = _indices_containing(
        joint_names, ("shoulder", "elbow", "wrist", "hand")
    )
    leg_indices = _indices_containing(joint_names, ("hip", "knee", "ankle", "foot"))
    torso_indices = _indices_containing(joint_names, ("waist", "torso", "spine"))

    def group_mean(indices: list[int]) -> float | None:
        valid = [idx for idx in indices if idx < abs_vel.shape[1]]
        if not valid:
            return None
        return _finite_mean(abs_vel[:, valid])

    arm_mean = group_mean(arm_indices)
    leg_mean = group_mean(leg_indices)
    ratio = None
    if arm_mean is not None and leg_mean is not None and leg_mean > 1.0e-8:
        ratio = arm_mean / leg_mean
    return {
        "joint_mean_abs_vel": _finite_mean(abs_vel),
        "joint_p95_abs_vel": _percentile(abs_vel.reshape(-1), 95),
        "arm_mean_abs_vel": arm_mean,
        "leg_mean_abs_vel": leg_mean,
        "torso_mean_abs_vel": group_mean(torso_indices),
        "arm_leg_activity_ratio": ratio,
    }


def _cyclicity_score(
    joint_pos: np.ndarray | None,
    joint_names: list[str],
    *,
    duration_sec: float,
) -> float | None:
    if (
        joint_pos is None
        or joint_pos.ndim != 2
        or joint_pos.shape[0] < 30
        or duration_sec <= 0.0
    ):
        return None

    name_to_index = {name: idx for idx, name in enumerate(joint_names)}
    left = name_to_index.get("left_hip_pitch_joint")
    right = name_to_index.get("right_hip_pitch_joint")
    if left is not None and right is not None:
        signal = joint_pos[:, left] - joint_pos[:, right]
    else:
        leg_indices = _indices_containing(joint_names, ("hip", "knee", "ankle"))
        valid = [idx for idx in leg_indices if idx < joint_pos.shape[1]]
        if not valid:
            return None
        signal = np.mean(joint_pos[:, valid], axis=1)

    if signal.shape[0] > 1200:
        indices = np.linspace(0, signal.shape[0] - 1, 1200).astype(np.int64)
        signal = signal[indices]
    signal = np.asarray(signal, dtype=np.float64)
    signal = signal - np.nanmean(signal)
    if not np.all(np.isfinite(signal)):
        signal = np.nan_to_num(signal, nan=0.0)
    denom = float(np.dot(signal, signal))
    if denom <= 1.0e-12:
        return None

    sample_rate = max(1.0, (signal.shape[0] - 1) / duration_sec)
    min_lag = max(2, int(round(0.25 * sample_rate)))
    max_lag = min(signal.shape[0] // 2, int(round(2.0 * sample_rate)))
    if max_lag <= min_lag:
        return None
    best = 0.0
    for lag in range(min_lag, max_lag + 1):
        a = signal[:-lag]
        b = signal[lag:]
        local_denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if local_denom <= 1.0e-12:
            continue
        best = max(best, float(np.dot(a, b) / local_denom))
    return max(0.0, min(1.0, best))


def _load_motion_stats(
    entry: dict[str, Any],
    *,
    manifest_dir: Path,
) -> dict[str, Any]:
    name = str(entry.get("name") or Path(str(entry.get("path", "motion"))).stem)
    category = humanize_motion_name(name)
    npz_path = _resolve_motion_path(entry, manifest_dir)
    if not npz_path.is_file():
        raise FileNotFoundError(f"Motion NPZ not found for {name}: {npz_path}")

    with np.load(npz_path, allow_pickle=False) as data:
        total_frames = _infer_total_frames(data)
        frame_slice = _frame_slice(entry.get("frame_range"), total_frames)
        fps = float(entry.get("input_fps", 0.0) or 0.0)
        if fps <= 0.0 and "fps" in data.files:
            fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        if fps <= 0.0:
            fps = 50.0

        root_pos = _root_pos(data, frame_slice)
        frames = int(root_pos.shape[0])
        duration_sec = frames / fps if fps > 0.0 else 0.0
        root_vel = _root_lin_vel(data, frame_slice, root_pos, fps)
        root_ang_vel = _slice_time(_get_array(data, "root_ang_vel"), frame_slice)
        joint_pos = _slice_time(_get_array(data, "joint_pos"), frame_slice)
        joint_vel = _slice_time(_get_array(data, "joint_vel"), frame_slice)
        body_lin_vel = _slice_time(_get_array(data, "body_lin_vel_w"), frame_slice)
        joint_names = _joint_names(data)

    root_xy = root_pos[:, :2]
    if frames > 1:
        root_xy_delta = np.diff(root_xy, axis=0)
        root_xy_path_length = float(np.linalg.norm(root_xy_delta, axis=1).sum())
        root_xy_displacement = float(np.linalg.norm(root_xy[-1] - root_xy[0]))
    else:
        root_xy_path_length = 0.0
        root_xy_displacement = 0.0

    root_speed = np.linalg.norm(root_vel[:, :2], axis=1)
    vertical_speed = np.abs(root_vel[:, 2])
    root_height = root_pos[:, 2]

    root_ang_speed = None
    if (
        root_ang_vel is not None
        and root_ang_vel.ndim == 2
        and root_ang_vel.shape[1] >= 3
    ):
        root_ang_speed = np.linalg.norm(root_ang_vel[:, :3], axis=1)

    body_speed = None
    if (
        body_lin_vel is not None
        and body_lin_vel.ndim == 3
        and body_lin_vel.shape[-1] >= 3
    ):
        body_speed = np.linalg.norm(body_lin_vel[..., :3], axis=-1)

    activity = _joint_activity(joint_vel, joint_names)
    cyclicity = _cyclicity_score(
        joint_pos,
        joint_names,
        duration_sec=duration_sec,
    )

    stats: dict[str, Any] = {
        "name": name,
        "category": category,
        "path": str(npz_path),
        "frames": frames,
        "fps": _as_float(fps, 3),
        "duration_sec": _as_float(duration_sec, 3),
        "root_xy_displacement_m": _as_float(root_xy_displacement),
        "root_xy_path_length_m": _as_float(root_xy_path_length),
        "root_xy_straightness": _as_float(
            root_xy_displacement / root_xy_path_length
            if root_xy_path_length > 1.0e-8
            else 0.0
        ),
        "root_mean_speed_mps": _as_float(_finite_mean(root_speed)),
        "root_p95_speed_mps": _as_float(_percentile(root_speed, 95)),
        "root_max_speed_mps": _as_float(_percentile(root_speed, 100)),
        "root_vertical_range_m": _as_float(
            float(root_height.max() - root_height.min())
        ),
        "root_min_height_m": _as_float(float(root_height.min())),
        "root_max_height_m": _as_float(float(root_height.max())),
        "root_mean_vertical_speed_mps": _as_float(_finite_mean(vertical_speed)),
        "root_p95_vertical_speed_mps": _as_float(_percentile(vertical_speed, 95)),
        "root_mean_ang_speed_radps": _as_float(
            _finite_mean(root_ang_speed) if root_ang_speed is not None else None
        ),
        "root_p95_ang_speed_radps": _as_float(
            _percentile(root_ang_speed, 95) if root_ang_speed is not None else None
        ),
        "body_mean_speed_mps": _as_float(
            _finite_mean(body_speed) if body_speed is not None else None
        ),
        "body_p95_speed_mps": _as_float(
            _percentile(body_speed.reshape(-1), 95) if body_speed is not None else None
        ),
        "cyclicity_score": _as_float(cyclicity),
        **{key: _as_float(value) for key, value in activity.items()},
    }
    return stats


def _quantile_thresholds(stats: list[dict[str, Any]], key: str) -> list[float]:
    values = [
        float(row[key])
        for row in stats
        if row.get(key) is not None and math.isfinite(float(row[key]))
    ]
    if len(values) < 2:
        return []
    return [float(value) for value in np.quantile(values, [0.2, 0.4, 0.6, 0.8])]


def _bin_label(value: Any, thresholds: list[float], labels: tuple[str, ...]) -> str:
    if value is None or not thresholds:
        return "unknown"
    numeric = float(value)
    index = 0
    while index < len(thresholds) and numeric > thresholds[index]:
        index += 1
    return labels[min(index, len(labels) - 1)]


def _cyclicity_label(score: Any) -> str:
    if score is None:
        return "unknown rhythm"
    value = float(score)
    if value >= 0.65:
        return "strong repeated rhythm"
    if value >= 0.45:
        return "moderate repeated rhythm"
    if value >= 0.25:
        return "weak repeated rhythm"
    return "irregular timing"


def _arm_leg_label(ratio: Any) -> str:
    if ratio is None:
        return "unknown limb emphasis"
    value = float(ratio)
    if value < 0.55:
        return "leg-dominant motion"
    if value < 0.85:
        return "leg-led motion"
    if value <= 1.25:
        return "balanced arm and leg motion"
    if value <= 1.75:
        return "arm-emphasized motion"
    return "strong arm emphasis"


def _assign_relative_bins(stats: list[dict[str, Any]]) -> None:
    thresholds = {
        "root_mean_speed_mps": _quantile_thresholds(stats, "root_mean_speed_mps"),
        "root_p95_speed_mps": _quantile_thresholds(stats, "root_p95_speed_mps"),
        "root_xy_displacement_m": _quantile_thresholds(stats, "root_xy_displacement_m"),
        "root_vertical_range_m": _quantile_thresholds(stats, "root_vertical_range_m"),
        "root_p95_ang_speed_radps": _quantile_thresholds(
            stats, "root_p95_ang_speed_radps"
        ),
    }
    for row in stats:
        row["relative_bins"] = {
            "mean_speed": _bin_label(
                row.get("root_mean_speed_mps"),
                thresholds["root_mean_speed_mps"],
                ("very slow", "slow", "moderate", "fast", "very fast"),
            ),
            "peak_speed": _bin_label(
                row.get("root_p95_speed_mps"),
                thresholds["root_p95_speed_mps"],
                (
                    "very low peak speed",
                    "low peak speed",
                    "moderate peak speed",
                    "high peak speed",
                    "very high peak speed",
                ),
            ),
            "travel": _bin_label(
                row.get("root_xy_displacement_m"),
                thresholds["root_xy_displacement_m"],
                (
                    "mostly in-place",
                    "short-travel",
                    "moderate-travel",
                    "traveling",
                    "long-travel",
                ),
            ),
            "vertical": _bin_label(
                row.get("root_vertical_range_m"),
                thresholds["root_vertical_range_m"],
                (
                    "very stable height",
                    "low height change",
                    "moderate height change",
                    "large level change",
                    "very large level change",
                ),
            ),
            "turning": _bin_label(
                row.get("root_p95_ang_speed_radps"),
                thresholds["root_p95_ang_speed_radps"],
                (
                    "minimal turning",
                    "mild turning",
                    "noticeable turning",
                    "frequent turning",
                    "strong turning",
                ),
            ),
            "rhythm": _cyclicity_label(row.get("cyclicity_score")),
            "limb_emphasis": _arm_leg_label(row.get("arm_leg_activity_ratio")),
        }


def _fmt(value: Any, unit: str = "") -> str:
    if value is None:
        return "unknown"
    return f"{float(value):.2f}{unit}"


def _has_large_level_change(vertical_label: str) -> bool:
    return vertical_label in {"large level change", "very large level change"}


def _locomotion_level_note(vertical_label: str) -> str:
    if not _has_large_level_change(vertical_label):
        return "steady upright balance"
    return f"non-steady body-height changes ({vertical_label})"


def _language_for_stats(stats: dict[str, Any]) -> dict[str, Any]:
    category = str(stats["category"])
    bins = stats["relative_bins"]
    travel = bins["travel"]
    speed = bins["mean_speed"]
    peak_speed = bins["peak_speed"]
    vertical = bins["vertical"]
    turning = bins["turning"]
    rhythm = bins["rhythm"]
    limb = bins["limb_emphasis"]
    level_note = _locomotion_level_note(vertical)

    if category == "dance":
        short = f"Rhythmic {travel} dance phrase with {limb} and {turning}."
        robot = (
            f"Perform a rhythmic dance sequence with {travel} footwork, "
            f"{limb}, and {turning}."
        )
        kinematic = (
            f"A humanoid performs a dance motion with {travel} root travel, "
            f"{speed} average horizontal speed, {vertical}, {turning}, "
            f"{limb}, and {rhythm}."
        )
        event = (
            "The humanoid shifts weight between the feet, steps through a dance "
            f"phrase, uses {limb}, turns with {turning}, and maintains balance "
            "through the rhythm."
        )
    elif category == "fall and get up":
        short = f"Fall-and-recovery motion with {vertical} and {peak_speed} recovery bursts."
        robot = (
            "Fall toward the ground, stabilize, and push back up to standing "
            f"with {vertical} and {peak_speed}."
        )
        kinematic = (
            "A humanoid descends from standing toward a low body height, makes "
            f"a recovery transition with {peak_speed}, shows {vertical}, "
            f"uses {limb}, and returns to upright balance."
        )
        event = (
            "The humanoid starts upright, loses height, reaches a low recovery "
            "posture, plants the limbs, pushes upward, and regains standing "
            f"balance with {travel} root displacement."
        )
    elif category == "fight":
        short = f"Combat-style motion with {limb}, {turning}, and {peak_speed} actions."
        robot = (
            f"Perform a fighting sequence with guarded stance changes, {limb}, "
            f"{turning}, and {peak_speed}."
        )
        kinematic = (
            f"A humanoid performs a fighting motion with {limb}, {turning}, "
            f"{travel} repositioning, {speed} average speed, and braced torso "
            "control."
        )
        event = (
            "The humanoid sets a guarded stance, shifts weight, performs arm "
            f"strikes or blocks, repositions with {travel} motion, and recovers "
            "balance."
        )
    elif category == "fight and sports":
        short = f"Athletic fight/sport motion with {limb}, {travel} repositioning, and {turning}."
        robot = (
            f"Perform an athletic fight-and-sport sequence with {travel} "
            f"repositioning, {limb}, and {turning}."
        )
        kinematic = (
            "A humanoid combines combat-like gestures with athletic stance "
            f"changes, using {limb}, {travel} root motion, {peak_speed}, "
            f"{turning}, and dynamic balance recovery."
        )
        event = (
            "The humanoid enters an athletic stance, shifts weight, executes "
            "sport-like upper-body actions, steps to reposition, and returns to "
            "a stable posture."
        )
    elif category == "jumps":
        short = f"Jumping sequence with {vertical}, {peak_speed}, and landing recovery."
        robot = (
            f"Perform jumps with a crouch, upward drive, landing absorption, "
            f"{vertical}, and {peak_speed}."
        )
        kinematic = (
            "A humanoid prepares through the legs, drives upward, changes root "
            f"height with {vertical}, moves with {travel} displacement, uses "
            f"{limb}, and stabilizes after landing."
        )
        event = (
            "The humanoid crouches, pushes through the legs, rises into a jump "
            "or hop, lands through the feet, absorbs impact, and stabilizes."
        )
    elif category == "run":
        short = f"Running gait with {speed} average speed, {peak_speed}, and {travel} travel."
        robot = (
            f"Run forward with alternating foot contacts, {speed} average speed, "
            f"{peak_speed}, {travel} travel, and {level_note}."
        )
        kinematic = (
            "A humanoid uses a running gait with alternating foot contacts, "
            f"{travel} root displacement, {speed} average horizontal speed, "
            f"{peak_speed}, {level_note}, reciprocal arm swing, and {rhythm}."
        )
        event = (
            "The humanoid cycles left and right leg drive, swings the arms "
            "opposite the legs, travels forward, and repeats the running gait"
            + (
                " while passing through visible low or recovery-like postures."
                if _has_large_level_change(vertical)
                else "."
            )
        )
    elif category == "sprint":
        short = (
            f"Sprinting gait with {peak_speed}, {travel} travel, and strong leg drive."
        )
        robot = (
            f"Sprint forward with powerful leg drive, {peak_speed}, "
            f"{travel} travel, and {level_note}."
        )
        kinematic = (
            "A humanoid performs sprinting bursts with powerful alternating leg "
            f"drive, {peak_speed}, {travel} root displacement, {level_note}, "
            "strong arm swing, and dynamic balance."
        )
        event = (
            "The humanoid leans into the sprint, drives one leg after the other, "
            "swings the arms strongly, makes brief foot contacts, and stabilizes "
            "between bursts"
            + (
                " with visible low-posture or recovery-like transitions."
                if _has_large_level_change(vertical)
                else "."
            )
        )
    elif category == "walk":
        short = f"Walking gait with {travel} travel, {speed} speed, and {vertical}."
        robot = (
            f"Walk forward with steady alternating steps, {travel} travel, "
            f"{speed} speed, and {level_note}."
        )
        kinematic = (
            "A humanoid uses a controlled walking gait with alternating foot "
            f"contacts, {travel} root displacement, {speed} average speed, "
            f"{level_note}, mild arm swing, and {rhythm}."
        )
        event = (
            "The humanoid places one foot, transfers weight, swings the other "
            "leg forward, places the next foot, and repeats a walking cycle"
            + (
                " that includes a visible low-body or ground-near transition."
                if _has_large_level_change(vertical)
                else "."
            )
        )
    else:
        short = (
            f"{category} motion with {travel} travel, {speed} speed, and {vertical}."
        )
        robot = (
            f"Perform a {category} motion with {travel} travel and controlled balance."
        )
        kinematic = (
            f"A humanoid performs a {category} motion with {travel} root "
            f"displacement, {speed} average speed, {vertical}, {turning}, "
            f"{limb}, and controlled full-body balance."
        )
        event = (
            f"The humanoid starts a {category} action, coordinates the limbs "
            "and torso through the motion, and returns to a balanced state."
        )

    measured_cues = (
        f"Measured cues: root displacement {_fmt(stats.get('root_xy_displacement_m'), ' m')}, "
        f"mean speed {_fmt(stats.get('root_mean_speed_mps'), ' m/s')}, "
        f"p95 speed {_fmt(stats.get('root_p95_speed_mps'), ' m/s')}, "
        f"height range {_fmt(stats.get('root_vertical_range_m'), ' m')}, "
        f"and p95 angular speed {_fmt(stats.get('root_p95_ang_speed_radps'), ' rad/s')}."
    )
    kinematic = f"{kinematic} {measured_cues}"

    attribute = (
        f"action: {category}; travel: {travel}; mean speed: {speed}; "
        f"peak speed: {peak_speed}; vertical motion: {vertical}; "
        f"turning: {turning}; limb emphasis: {limb}; rhythm: {rhythm}; "
        f"root displacement: {_fmt(stats.get('root_xy_displacement_m'), ' m')}; "
        f"root height range: {_fmt(stats.get('root_vertical_range_m'), ' m')}."
    )
    features = [
        f"{travel} root displacement ({_fmt(stats.get('root_xy_displacement_m'), ' m')})",
        f"{speed} mean horizontal speed ({_fmt(stats.get('root_mean_speed_mps'), ' m/s')})",
        f"{peak_speed} ({_fmt(stats.get('root_p95_speed_mps'), ' m/s')} p95)",
        f"{vertical} ({_fmt(stats.get('root_vertical_range_m'), ' m')} root height range)",
        f"{turning} ({_fmt(stats.get('root_p95_ang_speed_radps'), ' rad/s')} p95 angular speed)",
        f"{limb}; {rhythm}",
    ]

    return {
        "category": category,
        "short_caption": short,
        "robot_instruction": robot,
        "kinematic_description": kinematic,
        "event_level": event,
        "attribute_text": attribute,
        "distinguishing_features": features,
        "fallback_category_prompt": prompt_for_motion(stats["name"], "category"),
        "source": SOURCE_TAG,
        "review_status": "draft",
        "needs_visual_review": True,
        "confidence": "medium",
    }


def _storyboard_path_for_name(
    name: str,
    *,
    storyboard_dir: Path | None,
    pattern: str,
) -> str | None:
    if storyboard_dir is None:
        return None
    path = (storyboard_dir / pattern.format(name=name)).expanduser()
    if path.is_file():
        return str(path.resolve())
    return None


def _select_entries(
    entries: list[dict[str, Any]],
    *,
    select: list[str] | None,
    max_motions: int | None,
) -> list[dict[str, Any]]:
    selected = entries
    if select:
        wanted = set(select)
        selected = [
            entry
            for entry in entries
            if str(entry.get("name") or Path(str(entry.get("path", ""))).stem) in wanted
        ]
        missing = sorted(wanted - {str(entry.get("name")) for entry in selected})
        if missing:
            raise ValueError(f"Selected motion names not found in manifest: {missing}")
    if max_motions is not None:
        selected = selected[: max(0, int(max_motions))]
    if not selected:
        raise ValueError("No manifest entries selected.")
    return selected


def _write_manifest_copy(
    *,
    source_manifest: dict[str, Any],
    output_path: Path,
    descriptions: dict[str, dict[str, Any]],
    include_stats: bool,
) -> None:
    manifest_copy = copy.deepcopy(source_manifest)
    entries = _extract_manifest_entries(manifest_copy)
    for entry in entries:
        name = str(entry.get("name") or Path(str(entry.get("path", "motion"))).stem)
        description = descriptions.get(name)
        if description is None:
            continue
        language = {
            key: description[key]
            for key in (
                "category",
                "short_caption",
                "robot_instruction",
                "kinematic_description",
                "event_level",
                "attribute_text",
                "distinguishing_features",
                "source",
                "review_status",
                "confidence",
            )
            if key in description
        }
        if include_stats:
            language["stats_summary"] = description.get("stats_summary", {})
        entry["language"] = language
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest_copy, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _write_markdown(
    *,
    output_path: Path,
    payload: dict[str, Any],
) -> None:
    lines = [
        "# LaFAN1 Draft Motion Descriptions",
        "",
        "These captions are deterministic drafts from NPZ motion statistics. "
        "Use storyboard/contact-sheet review to correct semantics before treating "
        "them as curated labels.",
        "",
        "| Motion | Category | Draft caption | Distinguishing features | Storyboard |",
        "|---|---|---|---|---|",
    ]
    for name, description in payload["prompts"].items():
        features = "<br>".join(description.get("distinguishing_features", []))
        storyboard = description.get("storyboard_path") or ""
        if storyboard:
            storyboard = f"[image]({storyboard})"
        row = [
            name,
            description.get("category", ""),
            description.get("short_caption", ""),
            features,
            storyboard,
        ]
        escaped = [str(cell).replace("|", "\\|").replace("\n", " ") for cell in row]
        lines.append("| " + " | ".join(escaped) + " |")

    lines.extend(
        [
            "",
            "## Review Prompt",
            "",
            "For each motion, compare the storyboard/video with the statistics and "
            "revise only semantically wrong or non-distinct language fields. Keep "
            "the output JSON keyed by raw motion name and preserve the fields "
            "`robot_instruction`, `kinematic_description`, `event_level`, and "
            "`attribute_text`.",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Draft reviewable LaFAN1 language descriptions from offline NPZ stats."
        )
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(DEFAULT_MANIFEST),
        help="LAFAN1 manifest JSON to read.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help=(
            "Output sidecar JSON. This file is prompt-json-compatible with "
            "build_language_goal_embeddings.py."
        ),
    )
    parser.add_argument(
        "--manifest_output",
        type=str,
        default=None,
        help=(
            "Optional manifest copy to write with per-entry language fields. "
            "The source manifest is never modified unless you explicitly point "
            "this at the same path."
        ),
    )
    parser.add_argument(
        "--include_stats_in_manifest",
        action="store_true",
        default=False,
        help="Include stats_summary inside the optional manifest language fields.",
    )
    parser.add_argument(
        "--markdown_output",
        type=str,
        default=None,
        help="Optional Markdown review table to write.",
    )
    parser.add_argument(
        "--storyboard_dir",
        type=str,
        default=None,
        help=(
            "Optional directory of storyboard/contact-sheet images. Existing "
            "paths are recorded in the sidecar for visual review."
        ),
    )
    parser.add_argument(
        "--storyboard_pattern",
        type=str,
        default="{name}.jpg",
        help="Filename pattern under --storyboard_dir; may use {name}.",
    )
    parser.add_argument(
        "--select",
        nargs="+",
        default=None,
        help="Optional raw motion names to draft, preserving manifest order.",
    )
    parser.add_argument(
        "--max_motions",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    if not manifest_path.is_file():
        raise SystemExit(f"Manifest not found: {manifest_path}")
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = _select_entries(
        _extract_manifest_entries(manifest_data),
        select=args.select,
        max_motions=args.max_motions,
    )
    stats = [
        _load_motion_stats(entry, manifest_dir=manifest_path.parent)
        for entry in entries
    ]
    _assign_relative_bins(stats)

    storyboard_dir = (
        Path(args.storyboard_dir).expanduser().resolve()
        if args.storyboard_dir is not None
        else None
    )
    descriptions: dict[str, dict[str, Any]] = {}
    for row in stats:
        description = _language_for_stats(row)
        storyboard_path = _storyboard_path_for_name(
            row["name"],
            storyboard_dir=storyboard_dir,
            pattern=args.storyboard_pattern,
        )
        if storyboard_path is not None:
            description["storyboard_path"] = storyboard_path
        description["stats_summary"] = row
        descriptions[row["name"]] = description

    payload = {
        "schema_version": "lafan1_language_descriptions_v0",
        "source": SOURCE_TAG,
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": _sha256_file(manifest_path),
        "num_motions": len(descriptions),
        "prompt_json_compatible": True,
        "notes": (
            "Draft captions are generated from motion statistics and should be "
            "reviewed against rendered videos or storyboards before final use."
        ),
        "prompts": descriptions,
    }

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    if args.markdown_output is not None:
        _write_markdown(
            output_path=Path(args.markdown_output).expanduser().resolve(),
            payload=payload,
        )

    if args.manifest_output is not None:
        manifest_output = Path(args.manifest_output).expanduser().resolve()
        if manifest_output == manifest_path:
            raise SystemExit(
                "--manifest_output points at the source manifest. Write a copy "
                "first, review it, then replace the source manually if desired."
            )
        _write_manifest_copy(
            source_manifest=manifest_data,
            output_path=manifest_output,
            descriptions=descriptions,
            include_stats=bool(args.include_stats_in_manifest),
        )

    categories = sorted({row["category"] for row in stats})
    print(f"[language-draft] manifest:   {manifest_path}")
    print(f"[language-draft] motions:    {len(descriptions)}")
    print(f"[language-draft] categories: {', '.join(categories)}")
    print(f"[language-draft] sidecar:    {output_path}")
    if args.markdown_output is not None:
        print(
            "[language-draft] markdown:   "
            f"{Path(args.markdown_output).expanduser().resolve()}"
        )
    if args.manifest_output is not None:
        print(
            "[language-draft] manifest copy: "
            f"{Path(args.manifest_output).expanduser().resolve()}"
        )


if __name__ == "__main__":
    main()
