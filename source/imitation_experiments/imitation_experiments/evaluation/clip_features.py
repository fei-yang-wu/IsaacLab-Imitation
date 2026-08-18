"""Reference-side clip features and the frozen deployable-clip rule.

Every number here is computed from the *reference* motion alone. Nothing in
this module reads a policy, a rollout, or a score, so a selection made with it
cannot be tuned to flatter a checkpoint.

Why the rule exists: SONIC's headline simulation figures (22.3 mm MPJPE-L at
100% success) come from the 123-clip set it also deployed on hardware, not from
a large held-out benchmark. Its large-set rows are test-content 98.7% / 23.2 mm
and test-repetition 99.6%. A 4,096-clip random block of BONES-SEED is the
analogue of the large sets, and it contains deep-crouch and ground-contact
clips that the hardware set does not: on the canonical block the minimum
reference pelvis height alone has Spearman -0.61 against the released SONIC
checkpoint's per-clip MPJPE-L, with the bottom quintile at 36.3 mm against
18.5 mm for the top. Comparing such a block against 22.3 mm compares two
different motion populations.

:data:`DEPLOYABLE_CLIP_RULE_V1` is the frozen answer: an upright, moderate,
hardware-plausible clip. Its thresholds were chosen on the canonical block
(ranks 12288-16383) and then validated unchanged on a held-out block (ranks
20480-24575), where the released checkpoint scored 21.92 mm at SR 0.9995
against 22.16 mm at SR 1.0000 on the block the rule was written on. Do not
retune the thresholds; add a ``_v2`` rule instead.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

__all__ = [
    "DEPLOYABLE_CLIP_RULE_V1",
    "ClipFeatures",
    "compute_clip_features",
    "is_deployable",
    "select_deployable_ranks",
]

# Column layout of the retained bodies in a reference-arrays tree. The order is
# `G1_TRACKED_BODY_NAMES`; the tree's README is the authority.
_PELVIS = 0
_LEFT_ANKLE = 3
_RIGHT_ANKLE = 6
_LEFT_WRIST = 10
_RIGHT_WRIST = 12

_CONTROL_HZ = 50.0

DEPLOYABLE_CLIP_RULE_V1: Mapping[str, float] = {
    # Never squats or goes to the ground: the pelvis stays near standing height.
    "pel_z_min_min": 0.65,
    # No sprint or lunge: peak horizontal pelvis speed stays walk-to-jog.
    "root_speed_max_max": 2.0,
    # No whipped limb: the 99th percentile joint speed stays moderate.
    "jvel_p99_max": 6.0,
    # No high kick or jump: neither ankle rises above mid-shin.
    "feet_z_max_max": 0.35,
    # Three to twelve seconds at 50 Hz.
    "frames_min": 150.0,
    "frames_max": 600.0,
}
"""Frozen, reference-only definition of a hardware-plausible clip."""


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


def is_deployable(
    features: ClipFeatures, rule: Mapping[str, float] = DEPLOYABLE_CLIP_RULE_V1
) -> bool:
    """Whether one clip passes the frozen hardware-plausibility rule."""
    return (
        features.pel_z_min >= rule["pel_z_min_min"]
        and features.root_speed_max <= rule["root_speed_max_max"]
        and features.jvel_p99 <= rule["jvel_p99_max"]
        and features.feet_z_max <= rule["feet_z_max_max"]
        and rule["frames_min"] <= features.frames <= rule["frames_max"]
    )


def select_deployable_ranks(
    features: Iterable[ClipFeatures],
    *,
    count: int | None = None,
    seed: int = 20260817,
    rule: Mapping[str, float] = DEPLOYABLE_CLIP_RULE_V1,
) -> list[int]:
    """Ranks passing the rule, optionally sub-sampled to a fixed size.

    The pool is sorted by rank before sampling so the draw depends only on the
    seed and the rule, never on the order the features were computed in.
    """
    import random

    pool = sorted(
        (item.rank for item in features if is_deployable(item, rule)),
    )
    if count is None:
        return pool
    if len(pool) < int(count):
        raise ValueError(
            f"deployable pool has {len(pool)} clips, fewer than the requested {count}."
        )
    return sorted(random.Random(int(seed)).sample(pool, int(count)))
