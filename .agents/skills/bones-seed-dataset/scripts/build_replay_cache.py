#!/usr/bin/env python3
"""Safely build or validate a content-specific ILTools replay cache.

The ILTools loader rebuilds a persisted buffer automatically when its selection
does not match the existing sidecar.  That is convenient for disposable caches
but dangerous for the 95 GiB BONES-SEED buffer: a one-motion evaluation can
otherwise replace the full-dataset manifest in the same directory.  This
wrapper only builds into an empty directory and treats any mismatched nonempty
directory as an error.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from iltools.datasets.utils import make_rb_from


DEFAULT_KEYS = (
    "qpos",
    "qvel",
    "root_pos",
    "root_quat",
    "root_lin_vel",
    "root_ang_vel",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)
MANIFEST_NAME = "iltools_rb_manifest.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zarr",
        type=Path,
        required=True,
        help="Existing Zarr root containing every motion to persist.",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        required=True,
        help="Empty output directory, or an existing matching cache to validate.",
    )
    parser.add_argument(
        "--persist-id",
        required=True,
        help="Stable source-content identity, normally including a manifest hash.",
    )
    parser.add_argument("--expected-motions", type=int, required=True)
    parser.add_argument(
        "--expected-transitions",
        type=int,
        default=None,
        help="Optional exact written-transition count.",
    )
    parser.add_argument(
        "--keys",
        nargs="+",
        default=list(DEFAULT_KEYS),
        help="Exact replay fields to persist.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the sidecar and memmap metadata without building.",
    )
    parser.add_argument(
        "--verify-load",
        action="store_true",
        help=(
            "Reopen through ILTools, reconcile every trajectory span, and "
            "compare every field at distributed first/middle/last steps."
        ),
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Could not read JSON at {path}: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object at {path}.")
    return payload


def _validation_errors(
    persist_dir: Path,
    *,
    persist_id: str,
    keys: list[str],
    expected_motions: int,
    expected_transitions: int | None,
) -> list[str]:
    manifest_path = persist_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        return [f"missing {manifest_path}"]

    manifest = _read_json(manifest_path)
    key = manifest.get("key")
    traj_info = manifest.get("traj_info")
    errors: list[str] = []
    if manifest.get("format_version") != 1:
        errors.append(f"format_version={manifest.get('format_version')!r}, expected 1")
    if not isinstance(key, dict):
        errors.append("key is missing or is not an object")
        key = {}
    if key.get("source") != {"persist_id": persist_id}:
        errors.append(
            f"source={key.get('source')!r}, expected persist_id={persist_id!r}"
        )
    for selection_name in ("datasets", "motions", "trajectories"):
        if key.get(selection_name) is not None:
            errors.append(
                f"{selection_name} selection must be null for the full cache; "
                f"got {key.get(selection_name)!r}"
            )
    if key.get("keys") != keys:
        errors.append(f"keys={key.get('keys')!r}, expected {keys!r}")

    if not isinstance(traj_info, dict):
        errors.append("traj_info is missing or is not an object")
        traj_info = {}
    ordered = traj_info.get("ordered_traj_list")
    motion_count = len(ordered) if isinstance(ordered, list) else -1
    if motion_count != expected_motions:
        errors.append(f"motion_count={motion_count}, expected {expected_motions}")
    written = traj_info.get("written")
    if expected_transitions is not None and written != expected_transitions:
        errors.append(
            f"written={written!r}, expected transitions={expected_transitions}"
        )
    if (
        expected_transitions is not None
        and traj_info.get("capacity") != expected_transitions
    ):
        errors.append(
            f"capacity={traj_info.get('capacity')!r}, expected {expected_transitions}"
        )
    starts = traj_info.get("start_index")
    ends = traj_info.get("end_index")
    if not isinstance(starts, list) or len(starts) != expected_motions:
        errors.append("start_index length does not match expected motion count")
    if not isinstance(ends, list) or len(ends) != expected_motions:
        errors.append("end_index length does not match expected motion count")
    if isinstance(ends, list) and ends and written is not None and ends[-1] != written:
        errors.append(f"last end_index={ends[-1]!r}, expected written={written!r}")

    meta_path = persist_dir / "meta.json"
    if not meta_path.is_file():
        errors.append(f"missing {meta_path}")
    else:
        meta = _read_json(meta_path)
        for field in keys:
            field_meta = meta.get(field)
            field_path = persist_dir / f"{field}.memmap"
            if not isinstance(field_meta, dict):
                errors.append(f"meta.json has no metadata for {field!r}")
            else:
                shape = field_meta.get("shape")
                if (
                    not isinstance(shape, list)
                    or not shape
                    or (
                        expected_transitions is not None
                        and shape[0] != expected_transitions
                    )
                ):
                    errors.append(
                        f"meta shape for {field!r} is {shape!r}; expected first "
                        f"dimension {expected_transitions!r}"
                    )
            if not field_path.is_file() or field_path.stat().st_size <= 0:
                errors.append(f"missing or empty memmap: {field_path}")
    return errors


def _validate_or_raise(
    persist_dir: Path,
    *,
    persist_id: str,
    keys: list[str],
    expected_motions: int,
    expected_transitions: int | None,
) -> None:
    errors = _validation_errors(
        persist_dir,
        persist_id=persist_id,
        keys=keys,
        expected_motions=expected_motions,
        expected_transitions=expected_transitions,
    )
    if errors:
        detail = "\n  - ".join(errors)
        raise RuntimeError(f"Replay cache validation failed:\n  - {detail}")
    manifest = _read_json(persist_dir / MANIFEST_NAME)
    print(
        "[PASS] replay cache: "
        f"{manifest['traj_info']['written']:,} transitions, "
        f"{len(manifest['traj_info']['ordered_traj_list']):,} motions, "
        f"{persist_id}"
    )


def _resolve_zarr_path(path: Path) -> Path:
    zarr_path = path.expanduser().resolve()
    nested_zarr = zarr_path / "trajectories.zarr"
    if nested_zarr.exists():
        zarr_path = nested_zarr
    if not zarr_path.is_dir():
        raise FileNotFoundError(f"Zarr root not found: {zarr_path}")
    return zarr_path


def _verify_reload_against_zarr(
    *,
    zarr_path: Path,
    persist_dir: Path,
    persist_id: str,
    keys: list[str],
    expected_motions: int,
    expected_transitions: int | None,
) -> None:
    replay_buffer, traj_info = make_rb_from(
        zarr_path=zarr_path,
        datasets=None,
        motions=None,
        trajectories=None,
        keys=keys,
        device="cpu",
        persist_dir=persist_dir,
        persist_id=persist_id,
        persist_rebuild=False,
        verbose_tree=False,
        pin_memory=False,
        prefetch=0,
        batch_size=1,
    )
    ordered = traj_info.get("ordered_traj_list", [])
    starts = traj_info.get("start_index", [])
    ends = traj_info.get("end_index", [])
    if len(ordered) != expected_motions:
        raise RuntimeError(
            f"Reloaded motion count is {len(ordered)}, expected {expected_motions}."
        )
    if (
        expected_transitions is not None
        and traj_info.get("written") != expected_transitions
    ):
        raise RuntimeError(
            f"Reloaded transition count is {traj_info.get('written')}, "
            f"expected {expected_transitions}."
        )

    storage = getattr(replay_buffer, "_storage", None)
    source = getattr(storage, "_storage", None)
    if source is None:
        raise RuntimeError("Reloaded replay buffer has no tensor-backed storage.")
    root = zarr.open(zarr_path, mode="r")
    source_spans: dict[tuple[str, str, str], int] = {}
    for dataset in root.group_keys():
        info_list = root[dataset].attrs.get("trajectory_info_list")
        if not isinstance(info_list, list):
            raise RuntimeError(
                f"Source dataset {dataset!r} has no trajectory_info_list metadata."
            )
        for info in info_list:
            if not isinstance(info, dict):
                raise RuntimeError(
                    f"Source dataset {dataset!r} has malformed trajectory metadata."
                )
            motion = str(info["motion"])
            trajectory = str(
                info.get(
                    "trajectory",
                    f"trajectory_{int(info.get('trajectory_in_motion', 0))}",
                )
            )
            identity = (str(dataset), motion, trajectory)
            if identity in source_spans:
                raise RuntimeError(
                    f"Duplicate source trajectory metadata: {identity!r}."
                )
            source_spans[identity] = int(info["length"]) - 1

    reloaded_identities = {tuple(str(item) for item in entry) for entry in ordered}
    source_identities = set(source_spans)
    if reloaded_identities != source_identities:
        missing = sorted(source_identities - reloaded_identities)[:5]
        extra = sorted(reloaded_identities - source_identities)[:5]
        raise RuntimeError(
            "Reloaded replay trajectory identities differ from source metadata: "
            f"missing={missing!r}, extra={extra!r}."
        )
    if len(source_spans) != expected_motions:
        raise RuntimeError(
            f"Source metadata contains {len(source_spans)} motions, "
            f"expected {expected_motions}."
        )

    previous_end = 0
    for rank, entry in enumerate(ordered):
        identity = tuple(str(item) for item in entry)
        start = int(starts[rank])
        end = int(ends[rank])
        if start != previous_end:
            raise RuntimeError(
                f"Replay span is not contiguous at rank {rank}: "
                f"start={start}, previous_end={previous_end}."
            )
        actual_span = end - start
        expected_span = source_spans[identity]
        if actual_span != expected_span:
            raise RuntimeError(
                f"Replay span differs from source at rank {rank} {identity!r}: "
                f"actual={actual_span}, expected={expected_span}."
            )
        previous_end = end
    if expected_transitions is not None and previous_end != expected_transitions:
        raise RuntimeError(
            f"Reconciled source spans total {previous_end}, "
            f"expected {expected_transitions}."
        )

    sample_ranks = sorted(
        {
            0,
            expected_motions // 4,
            expected_motions // 2,
            (3 * expected_motions) // 4,
            expected_motions - 1,
        }
    )
    comparisons = 0
    for rank in sample_ranks:
        dataset, motion, trajectory = ordered[rank]
        group = root[str(dataset)][str(motion)][str(trajectory)]
        start = int(starts[rank])
        end = int(ends[rank])
        transition_count = end - start
        if transition_count <= 0:
            raise RuntimeError(f"Trajectory rank {rank} has no transitions.")
        for local_step in sorted({0, transition_count // 2, transition_count - 1}):
            global_step = start + local_step
            for key in keys:
                expected = np.asarray(group[key][local_step])
                actual = source[key][global_step].cpu().numpy()
                if not np.array_equal(actual, expected):
                    raise RuntimeError(
                        "Reloaded replay data differs from Zarr at "
                        f"rank={rank} local_step={local_step} key={key!r}."
                    )
                comparisons += 1
    del replay_buffer
    print(
        "[PASS] replay reload: "
        f"{expected_motions:,} motions, {traj_info['written']:,} transitions, "
        "every trajectory identity/span reconciled with source metadata, "
        f"{comparisons} representative field comparisons"
    )


def main() -> None:
    args = _parse_args()
    if args.expected_motions <= 0:
        raise ValueError("--expected-motions must be positive.")
    if args.expected_transitions is not None and args.expected_transitions <= 0:
        raise ValueError("--expected-transitions must be positive when provided.")
    keys = [str(key) for key in args.keys]
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("--keys must be a nonempty list without duplicates.")

    persist_dir = args.persist_dir.expanduser().resolve()
    zarr_path = _resolve_zarr_path(args.zarr)
    manifest_path = persist_dir / MANIFEST_NAME
    if args.validate_only:
        _validate_or_raise(
            persist_dir,
            persist_id=str(args.persist_id),
            keys=keys,
            expected_motions=int(args.expected_motions),
            expected_transitions=args.expected_transitions,
        )
        if args.verify_load:
            _verify_reload_against_zarr(
                zarr_path=zarr_path,
                persist_dir=persist_dir,
                persist_id=str(args.persist_id),
                keys=keys,
                expected_motions=int(args.expected_motions),
                expected_transitions=args.expected_transitions,
            )
        return

    if persist_dir.exists() and any(persist_dir.iterdir()):
        if manifest_path.is_file():
            try:
                _validate_or_raise(
                    persist_dir,
                    persist_id=str(args.persist_id),
                    keys=keys,
                    expected_motions=int(args.expected_motions),
                    expected_transitions=args.expected_transitions,
                )
                print("[INFO] Existing cache is valid; no build was performed.")
                if args.verify_load:
                    _verify_reload_against_zarr(
                        zarr_path=zarr_path,
                        persist_dir=persist_dir,
                        persist_id=str(args.persist_id),
                        keys=keys,
                        expected_motions=int(args.expected_motions),
                        expected_transitions=args.expected_transitions,
                    )
                return
            except RuntimeError as error:
                raise RuntimeError(
                    f"Refusing to rebuild mismatched nonempty directory {persist_dir}. "
                    "Choose a fresh, versioned --persist-dir instead.\n"
                    f"{error}"
                ) from error
        raise RuntimeError(
            f"Refusing to build into nonempty directory without {MANIFEST_NAME}: "
            f"{persist_dir}. Choose a fresh, versioned --persist-dir."
        )

    persist_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    print(f"[BUILD] zarr       : {zarr_path}")
    print(f"[BUILD] persist dir: {persist_dir}")
    print(f"[BUILD] persist id : {args.persist_id}")
    print(f"[BUILD] keys       : {','.join(keys)}")
    replay_buffer, traj_info = make_rb_from(
        zarr_path=zarr_path,
        datasets=None,
        motions=None,
        trajectories=None,
        keys=keys,
        device="cpu",
        persist_dir=persist_dir,
        persist_id=str(args.persist_id),
        persist_rebuild=False,
        verbose_tree=False,
        pin_memory=False,
        prefetch=0,
        batch_size=1,
    )
    del replay_buffer
    if len(traj_info.get("ordered_traj_list", [])) != args.expected_motions:
        raise RuntimeError(
            "Built cache motion count does not match --expected-motions; "
            "leave this directory quarantined and build into a new path."
        )
    _validate_or_raise(
        persist_dir,
        persist_id=str(args.persist_id),
        keys=keys,
        expected_motions=int(args.expected_motions),
        expected_transitions=args.expected_transitions,
    )
    if args.verify_load:
        _verify_reload_against_zarr(
            zarr_path=zarr_path,
            persist_dir=persist_dir,
            persist_id=str(args.persist_id),
            keys=keys,
            expected_motions=int(args.expected_motions),
            expected_transitions=args.expected_transitions,
        )


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"[FATAL] {error}") from None
