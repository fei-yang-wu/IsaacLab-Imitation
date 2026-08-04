#!/usr/bin/env python3
"""Create a training manifest for a subset of a converted BONES-SEED export.

The source manifest may contain far more motion data than fits in the current
GPU-backed reference replay buffer. This tool selects a reproducible subset and
rewrites every trajectory path relative to the output manifest. The public
SONIC keyword-exclusion filter is always applied before any other filtering or
selection, so ``--count N`` selects only from SONIC-eligible motions.

Selection modes are mutually exclusive:

* ``--count N`` keeps the first N eligible entries. Add ``--shuffle-seed S``
  to first put the complete candidate pool in a stable SHA-256 permutation;
  using the same pool and seed makes every smaller N an exact prefix of every
  larger N.
* ``--random-count N`` independently samples N candidates with ``--seed``.
* ``--names-file PATH`` selects explicit manifest names or NPZ filenames.
* ``--shard-ids ...`` selects every eligible member of specific tar shards.
* ``--all`` keeps every SONIC-eligible candidate after optional filters.

Category and mirror filters require the matching BONES-SEED language sidecar.
When the source manifest came directly from the Hugging Face bucket, pass
``--npz-root`` to the directory where the shard members were extracted.
During incremental extraction, ``--available-files-only`` limits selection to
NPZ files that currently exist. Freeze that available pool before producing
multiple nested subset sizes; otherwise newly extracted files change the pool
and therefore its permutation.

Examples:

    # First 5,000 eligible motions from an incrementally extracted NPZ tree.
    pixi run python experiments/scaling_motion/select_bones_seed_subset_manifest.py \
        --source-manifest /data/bones/source/g1_bones_seed_sonic_full_manifest.json \
        --npz-root /data/bones/npz/g1 \
        --output-manifest /data/bones/manifests/subset_first_5000.json \
        --count 5000 --shuffle-seed 0 \
        --available-files-only --require-files

    # Every motion in four selected bucket shards, plus a download list.
    pixi run python experiments/scaling_motion/select_bones_seed_subset_manifest.py \
        --source-manifest /data/bones/source/g1_bones_seed_sonic_full_manifest.json \
        --shard-index /data/bones/source/shard_index.json \
        --npz-root /data/bones/npz/g1 \
        --output-manifest /data/bones/manifests/subset_s4.json \
        --shard-ids 0000 0034 0068 0102 \
        --required-shards-output /data/bones/subset_s4_shards.txt
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from filter_bones_seed_sonic_exclusions import (  # noqa: E402
    SONIC_RELEASE_FILTER_KEYWORDS,
    matched_keywords,
)


SONIC_FILTER_PATH = SCRIPTS_DIR / "filter_bones_seed_sonic_exclusions.py"


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _manifest_entries(manifest: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    entries = manifest.get("dataset", {}).get("trajectories", {}).get("lafan1_csv")
    if not isinstance(entries, list) or not entries:
        raise ValueError(
            "Manifest must define a non-empty "
            f"dataset.trajectories.lafan1_csv list: {path}"
        )
    if not all(isinstance(entry, dict) for entry in entries):
        raise TypeError(f"Manifest entries must be JSON objects: {path}")

    names = [str(entry.get("name", "")).strip() for entry in entries]
    if any(not name for name in names):
        raise ValueError(f"Every manifest entry must have a non-empty name: {path}")
    if len(names) != len(set(names)):
        raise ValueError(f"Manifest contains duplicate motion names: {path}")
    for index, entry in enumerate(entries):
        if not entry.get("path") and not entry.get("file"):
            raise ValueError(
                f"Manifest entry {index} must have a non-empty path or file: {path}"
            )
    filenames = [_entry_filename(entry) for entry in entries]
    if len(filenames) != len(set(filenames)):
        raise ValueError(
            "Manifest contains duplicate NPZ filenames, which are ambiguous with "
            f"--npz-root: {path}"
        )
    return entries


def _language_motions(language: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    motions = language.get("motions")
    if not isinstance(motions, list) or not motions:
        raise ValueError(
            f"Language sidecar must define a non-empty motions list: {path}"
        )
    if not all(isinstance(motion, dict) for motion in motions):
        raise TypeError(f"Language motions must be JSON objects: {path}")
    names = [str(motion.get("name", "")).strip() for motion in motions]
    if any(not name for name in names):
        raise ValueError(f"Every language motion must have a non-empty name: {path}")
    if len(names) != len(set(names)):
        raise ValueError(f"Language sidecar contains duplicate names: {path}")
    return motions


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _selected_names_sha256(names: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for name in names:
        digest.update(str(name).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _entry_filename(entry: dict[str, Any]) -> str:
    raw_path = entry.get("path") or entry.get("file")
    if raw_path is None:
        raise ValueError(f"Manifest entry has no path/file: {entry}")
    return Path(str(raw_path)).name


def _resolve_source_path(
    entry: dict[str, Any],
    *,
    source_manifest: Path,
    npz_root: Path | None,
) -> Path:
    if npz_root is not None:
        return (npz_root / _entry_filename(entry)).resolve()
    raw_path = Path(str(entry.get("path") or entry.get("file"))).expanduser()
    if raw_path.is_absolute():
        return raw_path.resolve()
    return (source_manifest.parent / raw_path).resolve()


def _rebase_entry(
    entry: dict[str, Any],
    *,
    source_manifest: Path,
    output_manifest: Path,
    npz_root: Path | None,
) -> dict[str, Any]:
    result = copy.deepcopy(entry)
    source_path = _resolve_source_path(
        result,
        source_manifest=source_manifest,
        npz_root=npz_root,
    )
    result.pop("file", None)
    result["path"] = os.path.relpath(source_path, output_manifest.parent)
    return result


def _parse_mirror(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n", ""}:
        return False
    raise ValueError(f"Cannot interpret is_mirror value: {value!r}")


def _read_names_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        names = [
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    else:
        if not isinstance(value, list):
            raise TypeError(f"Names JSON must contain a list: {path}")
        names = []
        for item in value:
            if isinstance(item, str):
                names.append(item.strip())
            elif isinstance(item, dict):
                candidate = (
                    item.get("name")
                    or item.get("filename")
                    or item.get("bones_seed_filename")
                )
                if candidate is None:
                    raise ValueError(f"Names entry has no supported identifier: {item}")
                names.append(str(candidate).strip())
            else:
                raise TypeError(f"Unsupported names entry: {item!r}")
    if not names or any(not name for name in names):
        raise ValueError(f"Names file contains no usable identifiers: {path}")
    if len(names) != len(set(names)):
        raise ValueError(f"Names file contains duplicate identifiers: {path}")
    return names


def _entry_aliases(entry: dict[str, Any]) -> set[str]:
    filename = _entry_filename(entry)
    return {str(entry["name"]), filename, Path(filename).stem}


def _indices_from_names(
    entries: list[dict[str, Any]], identifiers: list[str]
) -> set[int]:
    alias_to_indices: dict[str, list[int]] = {}
    for index, entry in enumerate(entries):
        for alias in _entry_aliases(entry):
            alias_to_indices.setdefault(alias, []).append(index)

    selected: set[int] = set()
    missing: list[str] = []
    ambiguous: dict[str, list[str]] = {}
    for identifier in identifiers:
        matches = alias_to_indices.get(identifier, [])
        if not matches:
            missing.append(identifier)
        elif len(matches) > 1:
            ambiguous[identifier] = [str(entries[index]["name"]) for index in matches]
        else:
            selected.add(matches[0])
    if missing:
        raise ValueError(f"Names not found after filtering: {missing[:10]}")
    if ambiguous:
        raise ValueError(f"Ambiguous names: {ambiguous}")
    return selected


def _normalize_shard_name(value: str) -> str:
    normalized = str(value).strip()
    if re.fullmatch(r"\d+", normalized):
        return f"bones_seed_g1-{int(normalized):04d}.tar"
    if re.fullmatch(r"bones_seed_g1-\d{4}\.tar", normalized):
        return normalized
    raise ValueError(
        f"Invalid shard ID {value!r}; expected 34, 0034, or bones_seed_g1-0034.tar."
    )


def _shard_records(index: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    records = index.get("shards")
    if not isinstance(records, list) or not records:
        raise ValueError(f"Shard index must define a non-empty shards list: {path}")
    if not all(
        isinstance(record, dict)
        and isinstance(record.get("shard"), str)
        and isinstance(record.get("members"), list)
        for record in records
    ):
        raise TypeError(f"Malformed shard record in: {path}")
    return records


def _member_to_shard(
    records: list[dict[str, Any]],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for record in records:
        shard = str(record["shard"])
        for raw_member in record["members"]:
            member = str(raw_member)
            if member in result:
                raise ValueError(
                    f"Shard index assigns member {member!r} to multiple shards."
                )
            result[member] = shard
    return result


def _candidate_indices(
    entries: list[dict[str, Any]],
    *,
    language_by_name: dict[str, dict[str, Any]] | None,
    categories: list[str],
    mirror_mode: str,
) -> tuple[list[int], list[str]]:
    if (categories or mirror_mode != "all") and language_by_name is None:
        raise ValueError("--category and --mirror-mode require --language-sidecar.")

    selected: list[int] = []
    sonic_excluded: list[str] = []
    missing_language: list[str] = []
    category_set = set(categories)
    for index, entry in enumerate(entries):
        name = str(entry["name"])
        if matched_keywords(name, SONIC_RELEASE_FILTER_KEYWORDS):
            sonic_excluded.append(name)
            continue
        language = None if language_by_name is None else language_by_name.get(name)
        if language_by_name is not None and language is None:
            missing_language.append(name)
            continue
        if category_set and str(language.get("category", "")) not in category_set:
            continue
        if mirror_mode != "all":
            is_mirror = _parse_mirror(language.get("is_mirror", False))
            if mirror_mode == "exclude" and is_mirror:
                continue
            if mirror_mode == "only" and not is_mirror:
                continue
        selected.append(index)

    if missing_language:
        raise ValueError(
            "Language sidecar is missing manifest motions, for example: "
            f"{missing_language[:10]}"
        )
    return selected, sonic_excluded


def _available_candidate_indices(
    entries: list[dict[str, Any]],
    candidate_indices: list[int],
    *,
    source_manifest: Path,
    npz_root: Path | None,
) -> tuple[list[int], int]:
    available = [
        index
        for index in candidate_indices
        if _resolve_source_path(
            entries[index],
            source_manifest=source_manifest,
            npz_root=npz_root,
        ).is_file()
    ]
    return available, len(candidate_indices) - len(available)


def _stable_shuffle_key(entry: dict[str, Any], seed: int) -> bytes:
    digest = hashlib.sha256()
    digest.update(str(seed).encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(entry["name"]).encode("utf-8"))
    return digest.digest()


def _stable_shuffled_indices(
    entries: list[dict[str, Any]],
    candidate_indices: list[int],
    seed: int,
) -> list[int]:
    return sorted(
        candidate_indices,
        key=lambda index: (_stable_shuffle_key(entries[index], seed), index),
    )


def _select_indices(
    args: argparse.Namespace,
    entries: list[dict[str, Any]],
    candidate_indices: list[int],
    shard_records: list[dict[str, Any]] | None,
) -> tuple[list[int], dict[str, Any]]:
    candidate_set = set(candidate_indices)
    if args.count is not None:
        if args.count <= 0:
            raise ValueError("--count must be positive.")
        if args.count > len(candidate_indices):
            raise ValueError(
                f"--count={args.count} exceeds {len(candidate_indices)} candidates."
            )
        if args.shuffle_seed is None:
            ordered = candidate_indices[: args.count]
            method = {
                "mode": "count",
                "count": args.count,
                "order": "source_manifest",
            }
        else:
            shuffle_seed = int(args.shuffle_seed)
            ordered = _stable_shuffled_indices(
                entries,
                candidate_indices,
                shuffle_seed,
            )[: args.count]
            method = {
                "mode": "count",
                "count": args.count,
                "order": "stable_sha256_shuffle",
                "shuffle_seed": shuffle_seed,
                "shuffle_algorithm": "sha256(str(seed) + NUL + motion_name)",
            }
    elif args.random_count is not None:
        if args.random_count <= 0:
            raise ValueError("--random-count must be positive.")
        if args.random_count > len(candidate_indices):
            raise ValueError(
                f"--random-count={args.random_count} exceeds "
                f"{len(candidate_indices)} candidates."
            )
        seed = 0 if args.seed is None else int(args.seed)
        rng = random.Random(seed)
        selected = set(rng.sample(candidate_indices, args.random_count))
        ordered = [index for index in range(len(entries)) if index in selected]
        method = {
            "mode": "random_count",
            "count": args.random_count,
            "seed": seed,
        }
    elif args.names_file is not None:
        identifiers = _read_names_file(args.names_file)
        filtered_entries = [entries[index] for index in candidate_indices]
        relative_indices = _indices_from_names(filtered_entries, identifiers)
        selected = {candidate_indices[index] for index in relative_indices}
        ordered = [index for index in range(len(entries)) if index in selected]
        method = {
            "mode": "names_file",
            "names_file": str(args.names_file.resolve()),
            "identifier_count": len(identifiers),
        }
    elif args.shard_ids is not None:
        if shard_records is None:
            raise ValueError("--shard-ids requires --shard-index.")
        requested = {_normalize_shard_name(value) for value in args.shard_ids}
        records_by_name = {str(record["shard"]): record for record in shard_records}
        missing = sorted(requested - set(records_by_name))
        if missing:
            raise ValueError(f"Shard IDs not found in shard index: {missing}")
        members = {
            str(member)
            for shard in requested
            for member in records_by_name[shard]["members"]
        }
        selected = {
            index
            for index in candidate_indices
            if _entry_filename(entries[index]) in members
        }
        ordered = [index for index in range(len(entries)) if index in selected]
        missing_members = members - {
            _entry_filename(entries[index]) for index in range(len(entries))
        }
        if missing_members:
            raise ValueError(
                "Shard members are missing from the source manifest, for example: "
                f"{sorted(missing_members)[:10]}"
            )
        method = {"mode": "shards", "shards": sorted(requested)}
    else:
        ordered = [index for index in range(len(entries)) if index in candidate_set]
        method = {"mode": "all"}

    if not ordered:
        raise ValueError("Selection produced zero motions.")
    return ordered, method


def _write_json(path: Path, value: dict[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_required_shards(
    path: Path,
    *,
    selected_entries: list[dict[str, Any]],
    member_to_shard: dict[str, str],
    overwrite: bool,
) -> list[str]:
    missing = [
        _entry_filename(entry)
        for entry in selected_entries
        if _entry_filename(entry) not in member_to_shard
    ]
    if missing:
        raise ValueError(
            "Selected manifest members are missing from the shard index, for example: "
            f"{missing[:10]}"
        )
    required = sorted(
        {member_to_shard[_entry_filename(entry)] for entry in selected_entries}
    )
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{shard}\n" for shard in required), encoding="utf-8")
    return required


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument(
        "--npz-root",
        type=Path,
        default=None,
        help=(
            "Directory containing extracted NPZ files. When omitted, source paths "
            "are resolved relative to --source-manifest."
        ),
    )
    parser.add_argument("--language-sidecar", type=Path, default=None)
    parser.add_argument("--output-language-sidecar", type=Path, default=None)
    parser.add_argument("--shard-index", type=Path, default=None)
    parser.add_argument("--required-shards-output", type=Path, default=None)

    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--count", type=int)
    selection.add_argument("--random-count", type=int)
    selection.add_argument("--names-file", type=Path)
    selection.add_argument("--shard-ids", nargs="+")
    selection.add_argument("--all", action="store_true")

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed used only by --random-count (default: 0).",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help=(
            "With --count, order the complete candidate pool by a stable SHA-256 "
            "permutation before taking its prefix. Reuse the same seed and frozen "
            "candidate pool to create nested scale-up subsets."
        ),
    )
    parser.add_argument(
        "--category",
        action="append",
        default=[],
        help="Keep an exact language-sidecar category. Repeat for multiple categories.",
    )
    parser.add_argument(
        "--mirror-mode",
        choices=("all", "exclude", "only"),
        default="all",
        help="Keep all, only canonical non-mirrors, or only mirrored motions.",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Dataset name written to the subset manifest.",
    )
    parser.add_argument(
        "--require-files",
        action="store_true",
        help="Fail unless every selected rebased NPZ path exists.",
    )
    parser.add_argument(
        "--available-files-only",
        action="store_true",
        help=(
            "Before selection, keep only NPZ files that currently exist. "
            "Useful for a stable snapshot during incremental extraction."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.seed is not None and args.random_count is None:
        raise ValueError(
            "--seed is valid only with --random-count; "
            "use --shuffle-seed with --count for nested shuffled prefixes."
        )
    if args.shuffle_seed is not None and args.count is None:
        raise ValueError("--shuffle-seed is valid only with --count.")
    source_manifest = args.source_manifest.expanduser().resolve()
    output_manifest = args.output_manifest.expanduser().resolve()
    language_path = (
        None
        if args.language_sidecar is None
        else args.language_sidecar.expanduser().resolve()
    )
    shard_index_path = (
        None if args.shard_index is None else args.shard_index.expanduser().resolve()
    )
    output_language_path = (
        None
        if args.output_language_sidecar is None
        else args.output_language_sidecar.expanduser().resolve()
    )
    required_shards_path = (
        None
        if args.required_shards_output is None
        else args.required_shards_output.expanduser().resolve()
    )
    npz_root = None if args.npz_root is None else args.npz_root.expanduser().resolve()
    outputs = [
        path
        for path in (output_manifest, output_language_path, required_shards_path)
        if path is not None
    ]
    inputs = [
        path
        for path in (source_manifest, language_path, shard_index_path)
        if path is not None
    ]
    overlapping_paths = sorted(str(path) for path in set(outputs) & set(inputs))
    if overlapping_paths:
        raise ValueError(
            f"Output paths must not overwrite input files: {overlapping_paths}"
        )
    if len(outputs) != len(set(outputs)):
        raise ValueError(
            "Output manifest, language sidecar, and shard list must differ."
        )
    if not args.overwrite:
        existing_outputs = [str(path) for path in outputs if path.exists()]
        if existing_outputs:
            raise FileExistsError(
                f"Refusing to overwrite existing output(s): {existing_outputs}"
            )
    if not source_manifest.is_file():
        raise FileNotFoundError(f"Source manifest not found: {source_manifest}")
    if npz_root is not None and args.require_files and not npz_root.is_dir():
        raise NotADirectoryError(f"NPZ root not found: {npz_root}")

    manifest = _json_object(source_manifest)
    entries = _manifest_entries(manifest, source_manifest)

    language = None
    language_motions = None
    language_by_name = None
    if language_path is not None:
        if not language_path.is_file():
            raise FileNotFoundError(f"Language sidecar not found: {language_path}")
        language = _json_object(language_path)
        language_motions = _language_motions(language, language_path)
        language_by_name = {str(motion["name"]): motion for motion in language_motions}
    if output_language_path is not None and language is None:
        raise ValueError("--output-language-sidecar requires --language-sidecar.")

    shard_records = None
    member_to_shard = None
    if shard_index_path is not None:
        if not shard_index_path.is_file():
            raise FileNotFoundError(f"Shard index not found: {shard_index_path}")
        shard_index = _json_object(shard_index_path)
        shard_records = _shard_records(shard_index, shard_index_path)
        member_to_shard = _member_to_shard(shard_records)
    if required_shards_path is not None and member_to_shard is None:
        raise ValueError("--required-shards-output requires --shard-index.")

    candidate_indices, sonic_excluded = _candidate_indices(
        entries,
        language_by_name=language_by_name,
        categories=args.category,
        mirror_mode=args.mirror_mode,
    )
    candidate_count_before_availability = len(candidate_indices)
    unavailable_candidate_count = 0
    if args.available_files_only:
        candidate_indices, unavailable_candidate_count = _available_candidate_indices(
            entries,
            candidate_indices,
            source_manifest=source_manifest,
            npz_root=npz_root,
        )
    selected_indices, method = _select_indices(
        args,
        entries,
        candidate_indices,
        shard_records,
    )
    selected_source_entries = [entries[index] for index in selected_indices]
    selected_entries = [
        _rebase_entry(
            entry,
            source_manifest=source_manifest,
            output_manifest=output_manifest,
            npz_root=npz_root,
        )
        for entry in selected_source_entries
    ]

    if args.require_files:
        missing_files = [
            str((output_manifest.parent / entry["path"]).resolve())
            for entry in selected_entries
            if not (output_manifest.parent / entry["path"]).resolve().is_file()
        ]
        if missing_files:
            raise FileNotFoundError(
                f"Selected NPZ files are missing, for example: {missing_files[:10]}"
            )

    selected_names = [str(entry["name"]) for entry in selected_entries]
    dataset_name = (
        str(args.dataset_name)
        if args.dataset_name
        else f"{manifest.get('dataset_name', 'bones_seed')}_subset_{len(selected_entries)}"
    )
    output = copy.deepcopy(manifest)
    output["dataset_name"] = dataset_name
    output.setdefault("dataset", {}).setdefault("trajectories", {})["lafan1_csv"] = (
        selected_entries
    )
    metadata = output.setdefault("metadata", {})
    metadata["num_motions"] = len(selected_entries)
    if args.require_files:
        metadata["num_missing_npz"] = 0
    else:
        metadata.pop("num_missing_npz", None)
    metadata["paths_are_relative_to_manifest"] = True
    metadata["sonic_exclusion_filter_applied"] = True
    metadata["sonic_exclusion_filter_source"] = (
        "scripts/filter_bones_seed_sonic_exclusions.py"
    )
    metadata["sonic_exclusion_filter_source_sha256"] = _sha256(SONIC_FILTER_PATH)
    metadata["sonic_exclusion_filter_keywords"] = list(SONIC_RELEASE_FILTER_KEYWORDS)
    if output_language_path is not None:
        metadata["language_annotations_path"] = os.path.relpath(
            output_language_path,
            output_manifest.parent,
        )
    else:
        metadata.pop("language_annotations_path", None)
    metadata["subset"] = {
        **method,
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": _sha256(source_manifest),
        "source_count": len(entries),
        "sonic_excluded_count": len(sonic_excluded),
        "sonic_excluded_names_sha256": _selected_names_sha256(sonic_excluded),
        "sonic_eligible_count": len(entries) - len(sonic_excluded),
        "available_files_only": bool(args.available_files_only),
        "candidate_count_before_availability_filter": (
            candidate_count_before_availability
        ),
        "unavailable_candidate_file_count": unavailable_candidate_count,
        "candidate_count_after_filters": len(candidate_indices),
        "selected_count": len(selected_entries),
        "categories": list(args.category),
        "mirror_mode": args.mirror_mode,
        "selected_names_sha256": _selected_names_sha256(selected_names),
        **(
            {
                "shard_index": str(shard_index_path),
                "shard_index_sha256": _sha256(shard_index_path),
            }
            if shard_index_path is not None
            else {}
        ),
    }

    required_shards: list[str] | None = None
    if required_shards_path is not None:
        required_shards = _write_required_shards(
            required_shards_path,
            selected_entries=selected_source_entries,
            member_to_shard=member_to_shard,
            overwrite=args.overwrite,
        )

    _write_json(output_manifest, output, overwrite=args.overwrite)

    if output_language_path is not None:
        assert language is not None and language_by_name is not None
        missing_language = [
            name for name in selected_names if name not in language_by_name
        ]
        if missing_language:
            raise ValueError(
                "Selected motions are missing from the language sidecar: "
                f"{missing_language[:10]}"
            )
        subset_language = copy.deepcopy(language)
        subset_language["dataset_name"] = dataset_name
        subset_language["manifest"] = os.path.relpath(
            output_manifest, output_language_path.parent
        )
        subset_language["motions"] = [
            copy.deepcopy(language_by_name[name]) for name in selected_names
        ]
        subset_language["subset"] = copy.deepcopy(metadata["subset"])
        _write_json(
            output_language_path,
            subset_language,
            overwrite=args.overwrite,
        )

    print(f"[PASS] selected {len(selected_entries)} / {len(entries)} motions")
    print(
        "[INFO] "
        f"sonic_eligible={len(entries) - len(sonic_excluded)} "
        f"sonic_excluded={len(sonic_excluded)}"
    )
    if args.available_files_only:
        print(
            "[INFO] "
            f"available_candidates={len(candidate_indices)} "
            f"unavailable_candidates={unavailable_candidate_count}"
        )
    print(f"[INFO] dataset_name={dataset_name}")
    print(f"[INFO] manifest={output_manifest}")
    if output_language_path is not None:
        print(f"[INFO] language_sidecar={output_language_path}")
    if required_shards is not None:
        print(
            f"[INFO] required_shards={len(required_shards)} list={required_shards_path}"
        )
    print(f"[INFO] selected_names_sha256={metadata['subset']['selected_names_sha256']}")


if __name__ == "__main__":
    main()
