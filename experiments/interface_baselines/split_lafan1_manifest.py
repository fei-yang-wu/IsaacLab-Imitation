#!/usr/bin/env python3
"""Create train/eval LAFAN1 manifest splits for held-out interface baselines."""

from __future__ import annotations

import argparse
import copy
import fnmatch
import json
import os
import random
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train_manifest", type=Path, default=None)
    parser.add_argument("--eval_manifest", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument(
        "--heldout_names",
        type=str,
        default="",
        help="Comma-separated exact motion names to place in the eval manifest.",
    )
    parser.add_argument(
        "--heldout_patterns",
        type=str,
        default="",
        help="Comma-separated fnmatch patterns for eval motion names.",
    )
    parser.add_argument(
        "--heldout_count",
        type=int,
        default=0,
        help="Randomly hold out this many motions after explicit filters.",
    )
    parser.add_argument(
        "--heldout_fraction",
        type=float,
        default=0.0,
        help="Randomly hold out this fraction after explicit filters.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow_empty_train", action="store_true", default=False)
    parser.add_argument("--allow_empty_eval", action="store_true", default=False)
    return parser.parse_args()


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    entries = manifest.get("dataset", {}).get("trajectories", {}).get("lafan1_csv")
    if not isinstance(entries, list):
        raise ValueError("Manifest must define dataset.trajectories.lafan1_csv.")
    if not entries:
        raise ValueError("Manifest contains no lafan1_csv entries.")
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Manifest entry #{index} must be a mapping.")
        if "name" not in entry:
            raise ValueError(f"Manifest entry #{index} is missing a name.")
    return entries


def _entry_count(manifest: dict[str, Any]) -> int:
    entries = manifest.get("dataset", {}).get("trajectories", {}).get("lafan1_csv")
    if not isinstance(entries, list):
        raise ValueError("Manifest must define dataset.trajectories.lafan1_csv.")
    return len(entries)


def _default_paths(
    manifest_path: Path, *, output_dir: Path | None, prefix: str | None
) -> tuple[Path, Path]:
    root = output_dir if output_dir is not None else manifest_path.parent
    stem = prefix or manifest_path.stem
    return root / f"{stem}_train.json", root / f"{stem}_heldout.json"


def _selected_eval_indices(
    entries: list[dict[str, Any]],
    *,
    heldout_names: list[str],
    heldout_patterns: list[str],
    heldout_count: int,
    heldout_fraction: float,
    seed: int,
) -> set[int]:
    if int(heldout_count) < 0:
        raise ValueError("--heldout_count must be non-negative.")
    if not 0.0 <= float(heldout_fraction) <= 1.0:
        raise ValueError("--heldout_fraction must be in [0, 1].")

    names = [str(entry["name"]) for entry in entries]
    unknown_names = sorted(set(heldout_names) - set(names))
    if unknown_names:
        raise ValueError(f"Held-out names not found in manifest: {unknown_names}")

    selected = {
        index
        for index, name in enumerate(names)
        if name in heldout_names
        or any(fnmatch.fnmatchcase(name, pattern) for pattern in heldout_patterns)
    }
    remaining = [index for index in range(len(entries)) if index not in selected]
    rng = random.Random(int(seed))
    rng.shuffle(remaining)

    random_count = max(0, int(heldout_count))
    if float(heldout_fraction) > 0.0:
        random_count = max(
            random_count,
            round(float(heldout_fraction) * float(len(entries))),
        )
    random_count = min(random_count, len(remaining))
    selected.update(remaining[:random_count])
    return selected


def split_manifest(
    manifest: dict[str, Any],
    *,
    heldout_names: list[str] | None = None,
    heldout_patterns: list[str] | None = None,
    heldout_count: int = 0,
    heldout_fraction: float = 0.0,
    seed: int = 0,
    allow_empty_train: bool = False,
    allow_empty_eval: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    entries = _entries(manifest)
    eval_indices = _selected_eval_indices(
        entries,
        heldout_names=heldout_names or [],
        heldout_patterns=heldout_patterns or [],
        heldout_count=int(heldout_count),
        heldout_fraction=float(heldout_fraction),
        seed=int(seed),
    )
    if not eval_indices and not allow_empty_eval:
        raise ValueError(
            "Eval split is empty. Specify held-out names/patterns/count/fraction."
        )

    train_entries = [
        copy.deepcopy(entry)
        for index, entry in enumerate(entries)
        if index not in eval_indices
    ]
    eval_entries = [
        copy.deepcopy(entry)
        for index, entry in enumerate(entries)
        if index in eval_indices
    ]
    if not train_entries and not allow_empty_train:
        raise ValueError("Train split is empty. Reduce the held-out set.")

    train_manifest = _manifest_with_entries(manifest, train_entries, split_name="train")
    eval_manifest = _manifest_with_entries(manifest, eval_entries, split_name="heldout")
    return train_manifest, eval_manifest


def _manifest_with_entries(
    manifest: dict[str, Any], entries: list[dict[str, Any]], *, split_name: str
) -> dict[str, Any]:
    split_manifest_data = copy.deepcopy(manifest)
    split_manifest_data.setdefault("dataset", {}).setdefault("trajectories", {})[
        "lafan1_csv"
    ] = entries
    metadata = split_manifest_data.setdefault("metadata", {})
    if isinstance(metadata, dict):
        metadata["num_motions"] = len(entries)
        metadata["split"] = str(split_name)
        metadata["split_motion_names"] = [str(entry["name"]) for entry in entries]
    return split_manifest_data


def _rebase_manifest_paths(
    manifest: dict[str, Any], *, source_manifest_path: Path, target_manifest_path: Path
) -> dict[str, Any]:
    """Keep relative motion paths valid when writing a split beside another manifest."""

    metadata = manifest.get("metadata", {})
    relative_to_manifest = True
    if isinstance(metadata, dict):
        relative_to_manifest = bool(
            metadata.get("paths_are_relative_to_manifest", True)
        )
    if not relative_to_manifest:
        return manifest

    source_dir = source_manifest_path.expanduser().resolve().parent
    target_dir = target_manifest_path.expanduser().resolve().parent
    rebased = copy.deepcopy(manifest)
    for entry in _entries(rebased):
        raw_path = entry.get("path")
        if raw_path in (None, ""):
            continue
        motion_path = Path(str(raw_path)).expanduser()
        if motion_path.is_absolute():
            continue
        source_motion_path = (source_dir / motion_path).resolve()
        entry["path"] = Path(os.path.relpath(source_motion_path, target_dir)).as_posix()
    return rebased


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_manifest, eval_manifest = split_manifest(
        manifest,
        heldout_names=_split_csv(args.heldout_names),
        heldout_patterns=_split_csv(args.heldout_patterns),
        heldout_count=int(args.heldout_count),
        heldout_fraction=float(args.heldout_fraction),
        seed=int(args.seed),
        allow_empty_train=bool(args.allow_empty_train),
        allow_empty_eval=bool(args.allow_empty_eval),
    )
    default_train, default_eval = _default_paths(
        manifest_path,
        output_dir=args.output_dir.expanduser().resolve() if args.output_dir else None,
        prefix=args.prefix,
    )
    train_path = (
        args.train_manifest.expanduser().resolve()
        if args.train_manifest
        else default_train
    )
    eval_path = (
        args.eval_manifest.expanduser().resolve()
        if args.eval_manifest
        else default_eval
    )
    train_manifest = _rebase_manifest_paths(
        train_manifest,
        source_manifest_path=manifest_path,
        target_manifest_path=train_path,
    )
    eval_manifest = _rebase_manifest_paths(
        eval_manifest,
        source_manifest_path=manifest_path,
        target_manifest_path=eval_path,
    )
    _write_manifest(train_path, train_manifest)
    _write_manifest(eval_path, eval_manifest)
    print(f"[INFO] Wrote train manifest: {train_path}")
    print(f"[INFO] Wrote eval manifest: {eval_path}")
    print(
        "[INFO] Split sizes: "
        f"train={_entry_count(train_manifest)} eval={_entry_count(eval_manifest)}"
    )


if __name__ == "__main__":
    main()
