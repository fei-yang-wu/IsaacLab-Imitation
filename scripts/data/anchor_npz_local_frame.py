#!/usr/bin/env python3
"""Re-anchor G1 reference NPZs to a local frame (frame-0 root XY at origin).

The imitation env places each reference in the world as ``reference_root_pos +
env_origin`` with identity rotation (see ``ImitationRLEnv._get_reference_alignment_transform``).
For that to be env_origin-independent, every clip must start at a canonical
local origin. LAFAN1 references already satisfy this (root0 XY == 0); BONES-SEED
CSV conversions inherit the raw mocap start offset, so we re-anchor here.

This is a translation-only planar re-anchor: subtract the frame-0 root XY from
all world-frame position fields (``root_pos``, ``body_pos_w``, and ``qpos``
root translation). Heights (Z), orientations, and velocities are unchanged
because a constant XY translation leaves them invariant. Joint-space fields and
``joint_names`` are untouched.

Run from the repo root (default environment, no Isaac needed):

    pixi run python scripts/data/anchor_npz_local_frame.py --npz_dir data/bones_seed_100/npz/g1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# World-frame position fields whose XY must shift with the anchor.
_POS_FIELDS_XY = ("root_pos",)  # (T, 3)
_BODY_POS_FIELDS_XY = ("body_pos_w",)  # (T, B, 3)


def _anchor_one(npz_path: Path, *, tol: float) -> dict[str, float]:
    with np.load(npz_path, allow_pickle=True) as data:
        payload = {key: data[key] for key in data.files}

    if "root_pos" not in payload:
        raise KeyError(f"{npz_path.name}: missing root_pos")

    offset_xy = np.asarray(payload["root_pos"], dtype=np.float64)[0, :2].copy()
    before = float(np.hypot(*offset_xy))

    for key in _POS_FIELDS_XY:
        if key in payload:
            arr = payload[key].astype(np.float32)
            arr[:, :2] -= offset_xy.astype(np.float32)
            payload[key] = arr
    for key in _BODY_POS_FIELDS_XY:
        if key in payload:
            arr = payload[key].astype(np.float32)
            arr[:, :, :2] -= offset_xy.astype(np.float32)
            payload[key] = arr
    # qpos root translation lives in columns [0:3].
    if "qpos" in payload:
        arr = payload["qpos"].astype(np.float32)
        arr[:, :2] -= offset_xy.astype(np.float32)
        payload["qpos"] = arr

    after = float(np.hypot(*np.asarray(payload["root_pos"], dtype=np.float64)[0, :2]))
    if after > tol:
        raise RuntimeError(f"{npz_path.name}: residual root0 XY {after:.4g} > tol {tol}")

    np.savez(npz_path, **payload)
    return {"before": before, "after": after}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz_dir", type=Path, required=True)
    parser.add_argument("--tol", type=float, default=1e-5)
    args = parser.parse_args()

    npz_dir = args.npz_dir.expanduser().resolve()
    files = sorted(npz_dir.glob("*.npz"))
    if not files:
        raise SystemExit(f"[ERROR] no NPZ files under {npz_dir}")

    max_before = 0.0
    for npz_path in files:
        stats = _anchor_one(npz_path, tol=args.tol)
        max_before = max(max_before, stats["before"])
        if stats["before"] > 1e-3:
            print(f"[INFO] {npz_path.name}: root0 XY {stats['before']:.4f} -> {stats['after']:.2e}")
    print(f"[INFO] re-anchored {len(files)} NPZs; max pre-anchor root0 XY = {max_before:.4f} m")
    print("[INFO] all clips now start at local origin (XY==0); Z/orientation/velocity preserved.")


if __name__ == "__main__":
    main()
