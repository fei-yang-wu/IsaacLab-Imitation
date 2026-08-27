"""Reference-side clip features.

Every number here is computed from the *reference* motion alone. Nothing in
this module reads a policy, a rollout, or a score, so a population built with
it cannot be tuned to flatter a checkpoint.

A 2026-08-17 attempt to define a "hardware-plausible" clip from these features
(upright, moderate speed, no ground contact) was DELETED: SONIC's project site
shows real-robot deployment of squatting, kneeling, hand crawling and elbow
crawling, so an upright-only filter models difficulty, not deployability. Do
not reintroduce one without evidence about what a real G1 can actually run.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "TESTBED_CLIP_RULE_V1",
    "ClipFeatures",
    "compute_clip_features",
    "difficulty_index",
    "select_testbed_ranks",
]

# Column layout of the retained bodies in a reference-arrays tree. The order is
# `G1_TRACKED_BODY_NAMES`; the tree's README is the authority.
_PELVIS = 0
_LEFT_ANKLE = 3
_RIGHT_ANKLE = 6
_LEFT_WRIST = 10
_RIGHT_WRIST = 12

_CONTROL_HZ = 50.0


@dataclass(frozen=True)
class ClipFeatures:
    """Per-clip reference kinematics used by the deployable rule."""

    rank: int
    motion: str
    frames: int
    pel_z_min: float
    pel_z_mean: float
    feet_z_max: float
    root_speed_mean: float
    root_speed_max: float
    jvel_p99: float
    wrist_z_max: float
    travel_m: float


def compute_clip_features(
    reference_arrays_dir: str | Path, ranks: Iterable[int]
) -> list[ClipFeatures]:
    """Read one reference-arrays tree and describe each requested clip.

    The arrays are memory-mapped, so this touches only the requested clips'
    rows. Keep the tree on local storage: memory-mapping reference data off a
    network filesystem collapses throughput.
    """
    root = Path(reference_arrays_dir)
    manifest = json.loads((root / "reference_arrays_manifest.json").read_text())
    info = manifest["traj_info"]
    total = int(info["written"])
    start = np.asarray(info["start_index"])
    end = np.asarray(info["end_index"])
    names = info["ordered_traj_list"]

    body_pos = np.memmap(
        root / "body_pos_w.memmap", dtype=np.float32, mode="r", shape=(total, 14, 3)
    )
    body_vel = np.memmap(
        root / "body_lin_vel_w.memmap", dtype=np.float32, mode="r", shape=(total, 14, 3)
    )
    qpos = np.memmap(
        root / "qpos.memmap", dtype=np.float32, mode="r", shape=(total, 36)
    )

    features: list[ClipFeatures] = []
    for rank in ranks:
        rank = int(rank)
        first, last = int(start[rank]), int(end[rank])
        positions = np.asarray(body_pos[first:last])
        velocities = np.asarray(body_vel[first:last])
        joints = np.asarray(qpos[first:last, 7:])
        pelvis_z = positions[:, _PELVIS, 2]
        feet_z = positions[:, [_LEFT_ANKLE, _RIGHT_ANKLE], 2]
        wrist_z = positions[:, [_LEFT_WRIST, _RIGHT_WRIST], 2]
        root_speed = np.linalg.norm(velocities[:, _PELVIS, :2], axis=-1)
        if last - first > 1:
            joint_speed = np.abs(np.diff(joints, axis=0)) * _CONTROL_HZ
        else:
            joint_speed = np.zeros((1, joints.shape[1]), dtype=np.float32)
        features.append(
            ClipFeatures(
                rank=rank,
                motion=str(names[rank][1]),
                frames=last - first,
                pel_z_min=float(pelvis_z.min()),
                pel_z_mean=float(pelvis_z.mean()),
                feet_z_max=float(feet_z.max()),
                root_speed_mean=float(root_speed.mean()),
                root_speed_max=float(root_speed.max()),
                jvel_p99=float(np.percentile(joint_speed, 99)),
                wrist_z_max=float(wrist_z.max()),
                travel_m=float(
                    np.linalg.norm(
                        positions[-1, _PELVIS, :2] - positions[0, _PELVIS, :2]
                    )
                ),
            )
        )
    return features


# Motion-name tokens marking a clip whose reference is physically conditioned on
# an object or on terrain the evaluation scene does not contain. Tracking such a
# clip in an empty world scores a motion the robot has no reason to be able to
# make, so the failure is the scene's, not the tracker's. This is NOT a
# statement about what a real robot can do: squatting, kneeling, crawling and
# boxing all stay in, because SONIC deploys them on hardware.
TESTBED_EXCLUDED_NAME_TOKENS: frozenset[str] = frozenset(
    {
        "bike",
        "ball",
        "car",
        "chair",
        "crate",
        "door",
        "golf",
        "handle",
        "horse",
        "ladder",
        "lever",
        "phone",
        "riding",
        "rope",
        "stair",
        "stairs",
        "switch",
        "table",
        "tennis",
        "wall",
    }
)

TESTBED_CLIP_RULE_V1: Mapping[str, float] = {
    # Two to thirty seconds at 50 Hz. Shorter clips carry too little tracking to
    # measure; longer ones are usually concatenation artifacts.
    "frames_min": 100.0,
    "frames_max": 1500.0,
    # A reference pelvis below the floor is a retargeting artifact, not a pose.
    "pel_z_min_floor": 0.0,
    # Drop the easiest quarter of the surviving population by `difficulty_index`.
    # Those clips sit at 16-20 mm with a saturated success rate for every
    # tracker measured so far, so they consume a quarter of the board and
    # separate nothing.
    "difficulty_min": 0.25,
}
"""Frozen definition of the canonical comparison testbed population."""


def _percentile_rank(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    # Average ties, matching `scipy.stats.rankdata`, so equal clips get equal
    # difficulty regardless of input order.
    position = 0
    while position < len(order):
        end = position
        while (
            end + 1 < len(order) and values[order[end + 1]] == values[order[position]]
        ):
            end += 1
        shared = (position + end) / 2.0 + 1.0
        for index in order[position : end + 1]:
            ranks[index] = shared / len(values)
        position = end + 1
    return ranks


def difficulty_index(features: Sequence[ClipFeatures]) -> list[float]:
    """A reference-only difficulty score in [0, 1], one per clip.

    The mean of four percentile ranks taken over the population passed in:
    how low the pelvis goes, how fast the root travels, how fast the joints
    move, and how high the feet rise. Equal weights on purpose -- fitting the
    weights to a tracker's error would make the board a function of that
    tracker.

    Validated, not fitted: on the 4,096-clip canonical block the index has
    Spearman +0.46 against the released SONIC checkpoint's per-clip MPJPE-L,
    and its deciles rise monotonically from 16.3 mm to about 30 mm.

    The score is relative to the population it is computed over, so compute it
    once over the whole retained corpus, never per candidate board.
    """
    if not features:
        return []
    axes = (
        [-item.pel_z_min for item in features],
        [item.root_speed_max for item in features],
        [item.jvel_p99 for item in features],
        [item.feet_z_max for item in features],
    )
    ranked = [_percentile_rank(axis) for axis in axes]
    return [
        sum(axis[index] for axis in ranked) / len(ranked)
        for index in range(len(features))
    ]


def select_testbed_ranks(
    features: Sequence[ClipFeatures],
    *,
    count: int = 4096,
    seed: int = 20260818,
    rule: Mapping[str, float] = TESTBED_CLIP_RULE_V1,
    excluded_tokens: frozenset[str] = TESTBED_EXCLUDED_NAME_TOKENS,
) -> list[int]:
    """Ranks of the frozen comparison testbed, drawn from a whole corpus.

    Pass features for the ENTIRE corpus: the difficulty band is defined against
    the corpus population, so a subset would shift it.
    """
    import random
    import re

    retained = [
        item
        for item in features
        if not (set(re.split(r"[_\W]+", item.motion.lower())) & excluded_tokens)
        and rule["frames_min"] <= item.frames <= rule["frames_max"]
        and item.pel_z_min > rule["pel_z_min_floor"]
    ]
    scores = difficulty_index(retained)
    band = sorted(
        item.rank
        for item, score in zip(retained, scores)
        if score >= rule["difficulty_min"]
    )
    if len(band) < int(count):
        raise ValueError(
            f"difficulty band holds {len(band)} clips, fewer than the requested {count}."
        )
    return sorted(random.Random(int(seed)).sample(band, int(count)))
