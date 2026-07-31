#!/usr/bin/env python3
"""Materialize standard planner shards from one paired-target collection."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


PAIRED_TARGET_KEYS = (
    "latent_skill_target",
    "encoder_input_packet_target",
)


@dataclass(frozen=True)
class MaterializationTarget:
    key: str
    motion_name: str | None
    output_dir: Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples_dir", type=Path, required=True)
    parser.add_argument(
        "--target",
        action="append",
        required=True,
        metavar="KEY[:MOTION]=OUTPUT_DIR",
        help=(
            "Paired target tensor to promote and its fresh output directory. "
            "The optional motion filter supports splitting one multi-motion "
            "Isaac collection into specialist datasets without another sim run."
        ),
    )
    return parser.parse_args()


def _parse_targets(values: list[str]) -> list[MaterializationTarget]:
    targets: list[MaterializationTarget] = []
    identities: set[tuple[str, str | None]] = set()
    outputs: set[Path] = set()
    for value in values:
        raw_selector, separator, raw_output = value.partition("=")
        key, motion_separator, raw_motion = raw_selector.partition(":")
        motion_name = raw_motion.strip() if motion_separator else None
        if not separator or key not in PAIRED_TARGET_KEYS or not raw_output.strip():
            raise ValueError(
                "--target must be KEY[:MOTION]=OUTPUT_DIR with KEY in "
                f"{PAIRED_TARGET_KEYS}; got {value!r}."
            )
        if motion_separator and not motion_name:
            raise ValueError(f"Empty motion filter in --target {value!r}.")
        identity = (key, motion_name)
        output = Path(raw_output).expanduser().resolve()
        if identity in identities:
            raise ValueError(f"Duplicate paired target selector {identity!r}.")
        if output in outputs:
            raise ValueError(f"Duplicate paired target output directory {output}.")
        identities.add(identity)
        outputs.add(output)
        targets.append(MaterializationTarget(key, motion_name, output))
    return targets


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row_count(sample: dict[str, Any], *, path: Path) -> int:
    target = sample.get("causal_target")
    if not isinstance(target, torch.Tensor) or target.ndim != 2:
        raise ValueError(f"{path} has no rank-2 causal_target.")
    return int(target.shape[0])


def _materialize_sample(
    sample: dict[str, Any], *, key: str, path: Path
) -> dict[str, Any]:
    target = sample.get(key)
    if not isinstance(target, torch.Tensor) or target.ndim != 2:
        raise ValueError(f"{path} has no rank-2 paired target {key!r}.")
    if int(target.shape[0]) != _row_count(sample, path=path):
        raise ValueError(f"{path} paired target {key!r} has a row-count mismatch.")
    metadata = copy.deepcopy(sample.get("metadata"))
    if not isinstance(metadata, dict):
        raise ValueError(f"{path} has no metadata mapping.")
    specs = metadata.get("paired_interface_target_specs")
    if not isinstance(specs, dict) or not isinstance(specs.get(key), dict):
        raise ValueError(f"{path} has no paired target spec for {key!r}.")
    spec = copy.deepcopy(specs[key])
    result = {
        name: value
        for name, value in sample.items()
        if name not in PAIRED_TARGET_KEYS and name != "z_target"
    }
    promoted = target.detach().cpu().contiguous()
    result["causal_target"] = promoted
    result["demonstration_target"] = promoted
    metadata["interface"] = str(spec["interface"])
    metadata["target_spec"] = spec
    metadata["materialized_paired_target_key"] = key
    interval = int(metadata.get("planner_interval_steps", 10))
    metadata["command_future_steps"] = (
        interval if key == "latent_skill_target" else interval - 1
    )
    result["metadata"] = metadata
    if key == "latent_skill_target":
        result["z_target"] = promoted
    return result


def _select_motion(
    sample: dict[str, Any], *, motion_name: str, path: Path
) -> dict[str, Any] | None:
    rows = _row_count(sample, path=path)
    names = sample.get("motion_name")
    if not isinstance(names, list) or len(names) != rows:
        raise ValueError(f"{path} has no row-aligned motion_name field.")
    selected = [index for index, name in enumerate(names) if name == motion_name]
    if not selected:
        return None
    indices = torch.as_tensor(selected, dtype=torch.long)
    result: dict[str, Any] = {}
    for key, value in sample.items():
        if (
            isinstance(value, torch.Tensor)
            and value.ndim > 0
            and int(value.shape[0]) == rows
        ):
            result[key] = value.index_select(0, indices)
        elif isinstance(value, list) and len(value) == rows:
            result[key] = [value[index] for index in selected]
        else:
            result[key] = value
    return result


def main() -> None:
    args = _parse_args()
    source_dir = args.samples_dir.expanduser().resolve()
    source_paths = sorted(source_dir.glob("sample_step_*.pt"))
    if not source_paths:
        raise FileNotFoundError(f"No sample_step_*.pt files found in {source_dir}.")
    targets = _parse_targets(args.target)
    for target in targets:
        if target.output_dir.exists():
            raise FileExistsError(
                f"Refusing existing output directory: {target.output_dir}"
            )
        target.output_dir.mkdir(parents=True)

    manifests = {
        (target.key, target.motion_name): {
            "format": "materialized_paired_interface_samples",
            "version": 2,
            "paired_target_key": target.key,
            "motion_name_filter": target.motion_name,
            "source_dir": str(source_dir),
            "source_files": [],
            "output_files": [],
            "row_count": 0,
        }
        for target in targets
    }
    for source_path in source_paths:
        sample = torch.load(source_path, map_location="cpu", weights_only=False)
        if not isinstance(sample, dict):
            raise TypeError(f"{source_path} is not a planner sample mapping.")
        source_record = {"name": source_path.name, "sha256": _sha256(source_path)}
        for target in targets:
            selected_sample = (
                sample
                if target.motion_name is None
                else _select_motion(
                    sample,
                    motion_name=target.motion_name,
                    path=source_path,
                )
            )
            if selected_sample is None:
                continue
            rows = _row_count(selected_sample, path=source_path)
            output_path = target.output_dir / source_path.name
            torch.save(
                _materialize_sample(selected_sample, key=target.key, path=source_path),
                output_path,
            )
            manifest = manifests[(target.key, target.motion_name)]
            manifest["source_files"].append(source_record)
            manifest["output_files"].append(
                {"name": output_path.name, "sha256": _sha256(output_path)}
            )
            manifest["row_count"] += rows

    for target in targets:
        manifest = manifests[(target.key, target.motion_name)]
        if int(manifest["row_count"]) <= 0:
            raise ValueError(
                f"No rows matched target {target.key!r} motion {target.motion_name!r}."
            )
        manifest_path = target.output_dir / "materialization_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            f"[PASS] {target.key} motion={target.motion_name or 'all'}: "
            f"{manifest['row_count']} rows -> {target.output_dir}"
        )


if __name__ == "__main__":
    main()
