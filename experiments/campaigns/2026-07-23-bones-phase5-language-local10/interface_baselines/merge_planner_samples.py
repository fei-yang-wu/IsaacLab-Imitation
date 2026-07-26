#!/usr/bin/env python3
"""Merge demonstration and planner-rollout samples for one DAgger round."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))

from interface_planner_common import load_rollout_samples  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument(
        "--source_limit",
        type=int,
        action="append",
        default=None,
        help=(
            "Optional row limit for each --source, in the same order. Use 0 "
            "for all rows. Selection matches planner-training --max_samples."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--replace_incomplete",
        action="store_true",
        default=False,
        help="Rebuild copied sample files when no completed merge manifest exists.",
    )
    return parser.parse_args()


def _selected_indices(*, rows: int, limit: int, seed: int) -> torch.Tensor:
    indices = torch.arange(int(rows), dtype=torch.long)
    if int(limit) <= 0 or int(limit) >= int(rows):
        return indices
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    indices = torch.randperm(int(rows), generator=generator)[: int(limit)]
    indices, _ = torch.sort(indices)
    return indices


def _slice_sample(
    sample: dict[str, object], indices: torch.Tensor
) -> dict[str, object]:
    target = sample.get("causal_target")
    if not isinstance(target, torch.Tensor) or target.ndim == 0:
        raise ValueError("Planner sample is missing a row-shaped target tensor.")
    rows = int(target.shape[0])
    selected = [int(index) for index in indices.tolist()]
    result: dict[str, object] = {}
    for key, value in sample.items():
        if (
            isinstance(value, torch.Tensor)
            and value.ndim > 0
            and int(value.shape[0]) == rows
        ):
            result[key] = value.index_select(0, indices).contiguous()
        elif isinstance(value, list) and len(value) == rows:
            result[key] = [value[index] for index in selected]
        else:
            result[key] = value
    planner_step = result.get("planner_step")
    if isinstance(planner_step, torch.Tensor) and planner_step.numel() > 0:
        result["step"] = int(planner_step.min().item())
    return result


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_samples = list(output_dir.glob("sample_step_*.pt"))
    completed_manifest = output_dir / "merge_manifest.json"
    if existing_samples and args.replace_incomplete and not completed_manifest.exists():
        for path in existing_samples:
            path.unlink()
        existing_samples = []
    if existing_samples:
        raise FileExistsError(f"Output already contains planner samples: {output_dir}.")

    limits = list(args.source_limit or [])
    if limits and len(limits) != len(args.source):
        raise ValueError("Provide exactly one --source_limit for every --source.")
    if not limits:
        limits = [0] * len(args.source)

    sources: list[dict[str, object]] = []
    output_index = 0
    expected_interface: str | None = None
    for source_index, (raw_source, source_limit) in enumerate(
        zip(args.source, limits, strict=True)
    ):
        source = raw_source.expanduser().resolve()
        data, metadata = load_rollout_samples(source)
        interface = str(metadata["interface"])
        if expected_interface is None:
            expected_interface = interface
        elif interface != expected_interface:
            raise ValueError(
                f"Cannot merge interface {interface!r} with {expected_interface!r}."
            )
        paths = sorted(source.glob("sample_step_*.pt"))
        row_count = int(data["causal_target"].shape[0])
        selected = _selected_indices(
            rows=row_count,
            limit=int(source_limit),
            seed=int(args.seed) + source_index,
        )
        selected_set = set(int(index) for index in selected.tolist())
        global_row = 0
        selected_row_count = 0
        for path in paths:
            sample = torch.load(path, map_location="cpu", weights_only=False)
            target = sample.get("causal_target")
            if not isinstance(target, torch.Tensor) or target.ndim == 0:
                raise ValueError(
                    f"Sample {path} is missing a row-shaped target tensor."
                )
            file_rows = int(target.shape[0])
            local_indices = [
                row - global_row
                for row in range(global_row, global_row + file_rows)
                if row in selected_set
            ]
            global_row += file_rows
            if not local_indices:
                continue
            destination = output_dir / f"sample_step_{output_index:08d}.pt"
            if len(local_indices) == file_rows:
                shutil.copy2(path, destination)
            else:
                sliced = _slice_sample(
                    sample, torch.as_tensor(local_indices, dtype=torch.long)
                )
                torch.save(sliced, destination)
            output_index += 1
            selected_row_count += len(local_indices)
        if global_row != row_count or selected_row_count != int(selected.numel()):
            raise RuntimeError(
                f"Selected-row accounting failed for {source}: "
                f"loaded={row_count}, visited={global_row}, selected={selected_row_count}."
            )
        sources.append(
            {
                "path": str(source),
                "file_count": len(paths),
                "row_count": row_count,
                "source_limit": int(source_limit),
                "selected_row_count": selected_row_count,
                "collection_stage": metadata.get("collection_stage"),
            }
        )

    merged, merged_metadata = load_rollout_samples(output_dir)
    manifest = {
        "interface": expected_interface,
        "sample_format": merged_metadata["sample_format"],
        "file_count": output_index,
        "row_count": int(merged["causal_target"].shape[0]),
        "sources": sources,
    }
    (output_dir / "merge_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[INFO] Merged {manifest['row_count']} rows from {len(sources)} sources "
        f"into {output_dir}."
    )


if __name__ == "__main__":
    main()
