#!/usr/bin/env python3
"""Select LAFAN1 trajectories and write one-motion manifests."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def trajectory_entries(payload: dict[str, Any], manifest_path: Path) -> list[dict[str, Any]]:
    entries = payload.get("dataset", {}).get("trajectories", {}).get("lafan1_csv")
    if entries is None:
        entries = payload.get("lafan1_csv", payload.get("motions"))
    if not isinstance(entries, list):
        raise SystemExit(f"Could not find trajectory list in {manifest_path}")
    return entries


def parse_ranks(rank_spec: str, total: int) -> list[int]:
    spec = rank_spec.strip().lower()
    if spec in ("", "all"):
        return list(range(total))

    ranks: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_s, end_s = chunk.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise SystemExit(f"Invalid rank range: {chunk}")
            ranks.extend(range(start, end + 1))
        else:
            ranks.append(int(chunk))

    unique = list(dict.fromkeys(ranks))
    bad = [rank for rank in unique if rank < 0 or rank >= total]
    if bad:
        raise SystemExit(f"Ranks out of range [0, {total - 1}]: {bad}")
    return unique


def clean_motion_name(name: str, rank: int) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()).strip("._-") or f"rank_{rank:04d}"


def entry_name(entry: dict[str, Any], rank: int) -> str:
    raw_path = entry.get("path") or entry.get("file") or rank
    return str(entry.get("name") or Path(str(raw_path)).stem).replace("\t", " ")


def absolutize_entry_path(entry: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    entry = dict(entry)
    path_key = "path" if "path" in entry else "file" if "file" in entry else None
    if path_key is None:
        return entry

    source_path = Path(str(entry[path_key])).expanduser()
    if not source_path.is_absolute():
        source_path = (manifest_path.parent / source_path).resolve()
    else:
        source_path = source_path.resolve()
    entry[path_key] = str(source_path)
    return entry


def write_single_manifest(
    payload: dict[str, Any],
    entry: dict[str, Any],
    output_path: Path,
) -> None:
    single = dict(payload)
    if "dataset" in single and isinstance(single["dataset"], dict):
        single["dataset"] = dict(single["dataset"])
        trajectories = dict(single["dataset"].get("trajectories", {}))
        trajectories["lafan1_csv"] = [entry]
        single["dataset"]["trajectories"] = trajectories
    elif "lafan1_csv" in single:
        single["lafan1_csv"] = [entry]
    else:
        single["motions"] = [entry]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(single, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def infer_steps(
    entry: dict[str, Any],
    manifest_path: Path,
    *,
    fallback_steps: int | None,
) -> int:
    frame_range = entry.get("frame_range")
    if isinstance(frame_range, list) and len(frame_range) == 2:
        return max(1, int(frame_range[1]) - int(frame_range[0]))

    for key in ("num_frames", "length"):
        if key in entry:
            return max(1, int(entry[key]))

    path_key = "path" if "path" in entry else "file" if "file" in entry else None
    if path_key is not None:
        source_path = Path(str(entry[path_key])).expanduser()
        if not source_path.is_absolute():
            source_path = manifest_path.parent / source_path
        if source_path.is_file() and source_path.suffix.lower() == ".npz":
            try:
                import numpy as np

                with np.load(source_path, allow_pickle=False) as npz:
                    for npz_key in (
                        "body_pos_w",
                        "body_pos",
                        "joint_pos",
                        "dof_pos",
                        "root_pos_w",
                        "root_pos",
                        "motion",
                    ):
                        value = npz.get(npz_key)
                        if value is not None and getattr(value, "ndim", 0) >= 1:
                            return max(1, int(value.shape[0]))
                    for npz_key in npz.files:
                        value = npz[npz_key]
                        if getattr(value, "ndim", 0) >= 1:
                            return max(1, int(value.shape[0]))
            except Exception:
                if fallback_steps is None:
                    raise

    if fallback_steps is not None:
        return max(1, int(fallback_steps))
    raise SystemExit(f"Could not infer steps for trajectory: {entry}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--ranks", default="all")
    parser.add_argument("--limit", default=0, type=int)
    parser.add_argument("--output_root", required=True, type=Path)
    parser.add_argument(
        "--fallback_steps",
        default=0,
        type=int,
        help="Step count to use when motion length cannot be inferred.",
    )
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = trajectory_entries(payload, manifest_path)
    ranks = parse_ranks(args.ranks, len(entries))
    if args.limit > 0:
        ranks = ranks[: args.limit]

    for rank in ranks:
        entry = entries[rank]
        name = entry_name(entry, rank)
        clean_name = clean_motion_name(name, rank)
        trajectory_root = args.output_root / f"rank_{rank}_{clean_name}"
        single_manifest = trajectory_root / "manifest_single.json"
        write_single_manifest(payload, absolutize_entry_path(entry, manifest_path), single_manifest)
        steps = infer_steps(
            entry,
            manifest_path,
            fallback_steps=args.fallback_steps if args.fallback_steps > 0 else None,
        )
        print(f"{rank}\t{name}\t{clean_name}\t{trajectory_root}\t{single_manifest}\t{steps}")


if __name__ == "__main__":
    main()
