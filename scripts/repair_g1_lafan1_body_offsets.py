#!/usr/bin/env python3
"""Remove legacy Isaac scene-grid offsets from G1 LAFAN1 body positions.

The old batched CSV-to-NPZ exporter saved ``root_pos`` in the motion frame but
saved ``body_pos_w`` in Isaac's shared world frame. Each trajectory therefore
carried its environment-grid origin in every body position. This script writes
a separate corrected NPZ tree and refuses data whose offset is not constant.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--constant_tolerance", type=float, default=2.0e-5)
    parser.add_argument("--zero_tolerance", type=float, default=2.0e-5)
    return parser.parse_args()


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _body_root_offset(
    arrays: dict[str, np.ndarray], *, path: Path, constant_tolerance: float
) -> tuple[np.ndarray, float]:
    if "root_pos" not in arrays or "body_pos_w" not in arrays:
        raise ValueError(f"{path} must contain root_pos and body_pos_w.")
    root_pos = arrays["root_pos"]
    body_pos = arrays["body_pos_w"]
    if root_pos.ndim != 2 or root_pos.shape[-1] != 3:
        raise ValueError(f"Unexpected root_pos shape in {path}: {root_pos.shape}")
    if body_pos.ndim != 3 or body_pos.shape[0] != root_pos.shape[0]:
        raise ValueError(f"Unexpected body_pos_w shape in {path}: {body_pos.shape}")
    offsets = body_pos[:, 0, :] - root_pos
    offset = np.median(offsets, axis=0).astype(np.float32)
    max_drift = float(np.max(np.abs(offsets - offset)))
    if max_drift > float(constant_tolerance):
        raise ValueError(
            f"Body/root offset is not constant in {path}: max drift {max_drift:.3g}."
        )
    return offset, max_drift


def main() -> None:
    args = _parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    if output_dir == input_dir:
        raise ValueError("Refusing in-place repair; choose a separate --output_dir.")
    files = sorted(path for path in input_dir.glob("*.npz") if path.is_file())
    if not files:
        raise RuntimeError(f"No NPZ files found in {input_dir}.")
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    for source in files:
        arrays = _load_arrays(source)
        offset, max_drift = _body_root_offset(
            arrays,
            path=source,
            constant_tolerance=float(args.constant_tolerance),
        )
        corrected = arrays["body_pos_w"].astype(np.float32, copy=True)
        corrected -= offset.reshape(1, 1, 3)
        residual = float(
            np.max(np.abs(corrected[:, 0, :] - arrays["root_pos"]))
        )
        if residual > float(args.zero_tolerance):
            raise ValueError(
                f"Corrected body/root residual is too large in {source}: {residual:.3g}."
            )
        arrays["body_pos_w"] = corrected
        destination = output_dir / source.name
        np.savez(destination, **arrays)
        records.append(
            {
                "source": str(source),
                "destination": str(destination),
                "removed_offset_xyz": offset.tolist(),
                "source_offset_max_drift": max_drift,
                "corrected_body_root_max_residual": residual,
            }
        )

    report = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "file_count": len(records),
        "constant_tolerance": float(args.constant_tolerance),
        "zero_tolerance": float(args.zero_tolerance),
        "files": records,
    }
    report_path = (
        args.report.expanduser().resolve()
        if args.report is not None
        else output_dir / "body_offset_repair_report.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[PASS] Corrected {len(records)} NPZ files. Report: {report_path}")


if __name__ == "__main__":
    main()
