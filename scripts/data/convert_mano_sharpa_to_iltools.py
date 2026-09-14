#!/usr/bin/env python3
"""Convert video-to-data MANO/Sharpa Parquet to ILTools v2 data.

Use ``--retarget`` for MANO-only loaded data. Without it, the input must
already contain the source Sharpa IK fields.

Use ``--rigid-object-index`` when a source fixture contains an articulated or
multi-body object and a documented single-body rigid proxy is acceptable.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from iltools.datasets import (
    ManoSharpaLoader,
    make_rigid_proxy_row,
    resample_mano_sharpa_row,
)
from iltools.retarget import (
    SharpaPinkRetargeter,
    SharpaSettleAssets,
    settle_sharpa_row,
)
from iltools.retarget.contact_settling import ContactSettlingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="Parquet file or partition root."
    )
    parser.add_argument(
        "--object-asset", type=Path, required=True, help="Single rigid URDF or USD."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument(
        "--retarget", action="store_true", help="Run Pink MANO-to-Sharpa IK first."
    )
    parser.add_argument("--left-mjcf", type=Path)
    parser.add_argument("--right-mjcf", type=Path)
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Pink iterations per frame; 100 matches the source dataset scripts.",
    )
    parser.add_argument(
        "--mano-to-robot-scale",
        type=float,
        default=None,
        help=(
            "MANO-to-robot scale used by the retarget. The released per-dataset "
            "values are 1.0 for arctic and synthbox and 1.2 for taco, hot3d, "
            "oakink2, grab, h2o, and dexycb. Overrides the value in the row."
        ),
    )
    parser.add_argument(
        "--settle-contacts",
        action="store_true",
        help=(
            "Press the retargeted fingers onto the object in MuJoCo so they "
            "rest on the surface instead of inside it. The wrist is held on "
            "its retargeted trajectory; only finger joints change. Needs "
            "--object-mesh. This is a deliberate departure from the released "
            "pipeline, which does not correct penetration at all."
        ),
    )
    parser.add_argument(
        "--object-mesh",
        type=Path,
        default=None,
        help="Collision mesh used by --settle-contacts, in metres.",
    )
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=60,
        help="Physics steps per frame while settling.",
    )
    parser.add_argument("--target-fps", type=float, default=20.0)
    parser.add_argument("--motion-speed", type=float, default=1.0)
    parser.add_argument(
        "--rigid-object-index",
        type=int,
        default=None,
        help=(
            "Select one source object body as an explicit rigid proxy. "
            "Required for articulated or multi-body source fixtures."
        ),
    )
    return parser.parse_args()


class _RowsLoader(ManoSharpaLoader):
    """Reuse the validated writer for in-memory retargeted rows."""

    _rows: tuple[dict[str, Any], ...]

    def rows(self, *, filters: Any = None) -> tuple[dict[str, Any], ...]:
        del filters
        return self._rows


def main() -> None:
    args = parse_args()
    loader = ManoSharpaLoader(args.input)
    rows = loader.rows()
    if args.rigid_object_index is not None:
        rows = tuple(
            make_rigid_proxy_row(row, object_body_index=args.rigid_object_index)
            for row in rows
        )
    if args.mano_to_robot_scale is not None:
        rows = tuple(
            {**row, "mano_to_robot_scale": float(args.mano_to_robot_scale)}
            for row in rows
        )
    if args.retarget:
        if args.left_mjcf is None or args.right_mjcf is None:
            raise ValueError("--retarget requires --left-mjcf and --right-mjcf.")
        if all(row.get("mano_to_robot_scale") is None for row in rows):
            raise ValueError(
                "The input rows declare no mano_to_robot_scale; pass "
                "--mano-to-robot-scale (1.0 for arctic, 1.2 for taco and others)."
            )
        retargeter = SharpaPinkRetargeter(
            left_mjcf_path=args.left_mjcf,
            right_mjcf_path=args.right_mjcf,
            max_iterations=args.max_iterations,
        )
        rows = tuple(retargeter.retarget_row(row) for row in rows)
    if args.settle_contacts:
        if args.object_mesh is None:
            raise ValueError("--settle-contacts requires --object-mesh.")
        if args.left_mjcf is None or args.right_mjcf is None:
            raise ValueError("--settle-contacts requires --left-mjcf and --right-mjcf.")
        assets = SharpaSettleAssets(
            left_mjcf_path=args.left_mjcf,
            right_mjcf_path=args.right_mjcf,
            object_mesh_path=args.object_mesh,
        )
        config = ContactSettlingConfig(settle_steps=args.settle_steps)
        settled = []
        for row in rows:
            row, report = settle_sharpa_row(
                row,
                assets=assets,
                config=config,
                object_body_index=args.rigid_object_index or 0,
            )
            print(
                f"settled {report.frame_count} frames: worst penetration "
                f"{report.worst_penetration_before_m * -1000:.2f} mm -> "
                f"{report.worst_penetration_after_m * -1000:.2f} mm, "
                f"{report.frames_penetrating_before} -> "
                f"{report.frames_penetrating_after} penetrating frames, "
                f"mean joint shift {report.mean_joint_shift_rad:.4f} rad"
            )
            settled.append(row)
        rows = tuple(settled)
    rows = tuple(
        resample_mano_sharpa_row(
            row,
            target_fps=args.target_fps,
            motion_speed=args.motion_speed,
        )
        for row in rows
    )
    writer = _RowsLoader(args.input)
    writer._rows = rows
    manifest = writer.write_dataset(
        args.output,
        object_asset_path=args.object_asset,
        dataset_name=args.dataset_name,
    )
    print(manifest)


if __name__ == "__main__":
    main()
