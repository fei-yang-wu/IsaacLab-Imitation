#!/usr/bin/env python3
"""Repair a prefix-damaged ILTools replay from its hashed packed copy.

This is deliberately narrower than a generic unpacker.  A mismatched ILTools
selection can refill the leading rows of an existing full-size memmap and then
replace its sidecar.  When a pristine packed copy exists, rewriting only that
known prefix and checking every full-file hash is much cheaper than rereading
millions of small Zarr files or reassembling another 95 GiB copy.

The full-cache sidecar is installed only after every repaired target member
matches the SHA-256 recorded by ``buffer_pack_index.json``.  An interruption
therefore leaves a cache that still fails closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, BinaryIO


PACK_INDEX_NAME = "buffer_pack_index.json"
REPLAY_MANIFEST_NAME = "iltools_rb_manifest.json"
REPAIR_RECORD_NAME = "cache_repair_record.json"
REPAIR_IN_PROGRESS_NAME = "cache_repair_in_progress.json"
COPY_CHUNK_BYTES = 8 << 20
PROBE_BYTES = 64 << 10


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packed-dir", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--repair-rows", type=int, required=True)
    parser.add_argument("--expected-persist-id", required=True)
    parser.add_argument("--expected-motions", type=int, required=True)
    parser.add_argument("--expected-transitions", type=int, required=True)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Patch the target after all read-only preflight checks pass.",
    )
    parser.add_argument(
        "--confirm-repair",
        default="",
        help="With --apply, must equal 'repair-packed-prefix'.",
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(COPY_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(f"{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


class _PackedMemberReader:
    """Read one logical member across its ordered packed parts."""

    def __init__(self, packed_dir: Path, member: dict[str, Any]) -> None:
        self._paths = [packed_dir / str(name) for name in member["parts"]]
        self._sizes = [path.stat().st_size for path in self._paths]
        if sum(self._sizes) != int(member["size"]):
            raise RuntimeError(
                f"Packed parts for {member['name']!r} total {sum(self._sizes)}, "
                f"expected {member['size']}."
            )

    def read_at(self, offset: int, size: int) -> bytes:
        if offset < 0 or size < 0:
            raise ValueError("offset and size must be non-negative")
        remaining = size
        result = bytearray()
        cursor = 0
        for path, part_size in zip(self._paths, self._sizes, strict=True):
            part_end = cursor + part_size
            if offset >= part_end:
                cursor = part_end
                continue
            local_offset = max(0, offset - cursor)
            take = min(remaining, part_size - local_offset)
            with path.open("rb") as handle:
                handle.seek(local_offset)
                chunk = handle.read(take)
            if len(chunk) != take:
                raise RuntimeError(f"Short read from packed part {path}.")
            result.extend(chunk)
            remaining -= take
            offset += take
            cursor = part_end
            if remaining == 0:
                break
        if remaining:
            raise RuntimeError(f"Packed member is short by {remaining} bytes.")
        return bytes(result)

    def copy_prefix_to(self, target: BinaryIO, size: int) -> None:
        remaining = size
        target.seek(0)
        for path, part_size in zip(self._paths, self._sizes, strict=True):
            if remaining == 0:
                break
            with path.open("rb") as source:
                part_remaining = min(remaining, part_size)
                while part_remaining:
                    chunk = source.read(min(COPY_CHUNK_BYTES, part_remaining))
                    if not chunk:
                        raise RuntimeError(f"Short read from packed part {path}.")
                    target.write(chunk)
                    part_remaining -= len(chunk)
                    remaining -= len(chunk)
        if remaining:
            raise RuntimeError(f"Packed member is short by {remaining} bytes.")


def _validate_full_manifest(
    manifest: dict[str, Any],
    *,
    persist_id: str,
    expected_motions: int,
    expected_transitions: int,
) -> None:
    key = manifest.get("key", {})
    traj = manifest.get("traj_info", {})
    if manifest.get("format_version") != 1:
        raise RuntimeError("Packed replay manifest format_version is not 1.")
    if key.get("source") != {"persist_id": persist_id}:
        raise RuntimeError(f"Packed replay persist identity differs: {key!r}.")
    for selection in ("datasets", "motions", "trajectories"):
        if key.get(selection) is not None:
            raise RuntimeError(
                f"Packed replay is not full-dataset: {selection}={key.get(selection)!r}."
            )
    ordered = traj.get("ordered_traj_list")
    if not isinstance(ordered, list) or len(ordered) != expected_motions:
        raise RuntimeError(
            f"Packed replay has {len(ordered) if isinstance(ordered, list) else -1} "
            f"motions, expected {expected_motions}."
        )
    for count_name in ("capacity", "written"):
        if int(traj.get(count_name, -1)) != expected_transitions:
            raise RuntimeError(
                f"Packed replay {count_name}={traj.get(count_name)!r}, "
                f"expected {expected_transitions}."
            )


def _preflight(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    packed_dir = args.packed_dir.expanduser().resolve()
    target_dir = args.target_dir.expanduser().resolve()
    if packed_dir == target_dir:
        raise RuntimeError("--packed-dir and --target-dir must differ.")
    if args.repair_rows <= 0 or args.repair_rows > args.expected_transitions:
        raise RuntimeError("--repair-rows is outside the replay capacity.")

    index_path = packed_dir / PACK_INDEX_NAME
    index = _read_json(index_path)
    if index.get("format_version") != 1:
        raise RuntimeError("Packed index format_version is not 1.")
    if index.get("persist_id") != args.expected_persist_id:
        raise RuntimeError(
            f"Packed persist_id={index.get('persist_id')!r}, "
            f"expected {args.expected_persist_id!r}."
        )
    members = index.get("members")
    if not isinstance(members, list) or not members:
        raise RuntimeError("Packed index has no members.")

    packed_manifest = _read_json(packed_dir / REPLAY_MANIFEST_NAME)
    _validate_full_manifest(
        packed_manifest,
        persist_id=args.expected_persist_id,
        expected_motions=args.expected_motions,
        expected_transitions=args.expected_transitions,
    )
    target_manifest_path = target_dir / REPLAY_MANIFEST_NAME
    target_manifest = _read_json(target_manifest_path)
    target_traj = target_manifest.get("traj_info", {})
    if int(target_traj.get("capacity", -1)) != args.expected_transitions:
        raise RuntimeError("Target replay capacity differs from the packed cache.")
    target_written = int(target_traj.get("written", -1))
    if target_written < 0 or target_written > args.repair_rows:
        raise RuntimeError(
            f"Target sidecar written={target_written} is incompatible with "
            f"repair_rows={args.repair_rows}."
        )
    target_source = target_manifest.get("key", {}).get("source")
    if target_source != {"persist_id": args.expected_persist_id}:
        raise RuntimeError(f"Target persist identity differs: {target_source!r}.")

    total_repair_bytes = 0
    for member in members:
        name = str(member["name"])
        target = target_dir / name
        if not target.is_file():
            raise RuntimeError(f"Target member is missing: {target}.")
        if name == REPLAY_MANIFEST_NAME:
            if not bool(member["verbatim"]):
                raise RuntimeError("Packed replay manifest must be verbatim.")
            continue
        expected_size = int(member["size"])
        if target.stat().st_size != expected_size:
            raise RuntimeError(
                f"Target {name} has {target.stat().st_size} bytes, "
                f"expected {expected_size}."
            )
        expected_hash = member.get("sha256")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise RuntimeError(f"Packed member {name} has no usable SHA-256.")
        if bool(member["verbatim"]):
            actual_hash = _sha256_file(target)
            if actual_hash != expected_hash:
                raise RuntimeError(
                    f"Unrepairable verbatim target {name} differs from packed hash."
                )
            print(f"[PASS] unchanged descriptor: {name}")
            continue

        if expected_size % args.expected_transitions:
            raise RuntimeError(
                f"Target {name} size is not row-aligned to replay capacity."
            )
        row_bytes = expected_size // args.expected_transitions
        repair_bytes = args.repair_rows * row_bytes
        reader = _PackedMemberReader(packed_dir, member)
        for offset in sorted(
            {
                repair_bytes,
                expected_size // 4,
                expected_size // 2,
                (3 * expected_size) // 4,
                max(0, expected_size - PROBE_BYTES),
            }
        ):
            probe_size = min(PROBE_BYTES, expected_size - offset)
            with target.open("rb") as handle:
                handle.seek(offset)
                actual = handle.read(probe_size)
            expected = reader.read_at(offset, probe_size)
            if actual != expected:
                raise RuntimeError(
                    f"Target {name} still differs at/after repair boundary "
                    f"offset={offset}; increase --repair-rows or restore the full member."
                )
        total_repair_bytes += repair_bytes
        member["row_bytes"] = row_bytes
        member["repair_bytes"] = repair_bytes
        print(
            f"[PASS] repair boundary: {name} row_bytes={row_bytes} "
            f"repair_bytes={repair_bytes:,}"
        )

    print(
        f"[PASS] packed-prefix repair preflight: rows={args.repair_rows:,}, "
        f"bytes={total_repair_bytes:,}"
    )
    return index, members


def _apply(
    args: argparse.Namespace, index: dict[str, Any], members: list[dict[str, Any]]
) -> None:
    if args.confirm_repair != "repair-packed-prefix":
        raise RuntimeError("--apply requires --confirm-repair repair-packed-prefix.")
    packed_dir = args.packed_dir.expanduser().resolve()
    target_dir = args.target_dir.expanduser().resolve()
    target_manifest_path = target_dir / REPLAY_MANIFEST_NAME
    damaged_manifest_bytes = target_manifest_path.read_bytes()
    damaged_manifest_sha = hashlib.sha256(damaged_manifest_bytes).hexdigest()
    backup_path = (
        target_dir / f"{REPLAY_MANIFEST_NAME}.pre_repair_{damaged_manifest_sha[:12]}"
    )
    if backup_path.exists():
        if backup_path.read_bytes() != damaged_manifest_bytes:
            raise RuntimeError(f"Existing sidecar backup differs: {backup_path}.")
    else:
        shutil.copy2(target_manifest_path, backup_path)

    record: dict[str, Any] = {
        "format_version": 1,
        "status": "in_progress",
        "packed_dir": str(packed_dir),
        "packed_index_sha256": _sha256_file(packed_dir / PACK_INDEX_NAME),
        "target_dir": str(target_dir),
        "persist_id": args.expected_persist_id,
        "motions": args.expected_motions,
        "transitions": args.expected_transitions,
        "repair_rows": args.repair_rows,
        "damaged_manifest_sha256": damaged_manifest_sha,
        "damaged_manifest_backup": str(backup_path),
        "member_sha256": {},
    }
    in_progress_path = target_dir / REPAIR_IN_PROGRESS_NAME
    _write_json_atomic(in_progress_path, record)

    started = time.perf_counter()
    for member in members:
        if bool(member["verbatim"]):
            continue
        name = str(member["name"])
        repair_bytes = int(member["repair_bytes"])
        reader = _PackedMemberReader(packed_dir, member)
        with (target_dir / name).open("r+b", buffering=0) as target:
            reader.copy_prefix_to(target, repair_bytes)
            target.flush()
            os.fsync(target.fileno())
        print(f"[REPAIRED] {name}: {repair_bytes:,} prefix bytes", flush=True)

    for member in members:
        name = str(member["name"])
        if name == REPLAY_MANIFEST_NAME:
            continue
        actual_hash = _sha256_file(target_dir / name)
        expected_hash = str(member["sha256"])
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Full-file verification failed for {name}: "
                f"actual={actual_hash}, expected={expected_hash}. The full sidecar "
                "was not installed."
            )
        record["member_sha256"][name] = actual_hash
        _write_json_atomic(in_progress_path, record)
        print(f"[PASS] full SHA-256: {name} {actual_hash}", flush=True)

    packed_manifest_path = packed_dir / REPLAY_MANIFEST_NAME
    packed_manifest_hash = _sha256_file(packed_manifest_path)
    manifest_member = next(
        member for member in members if member["name"] == REPLAY_MANIFEST_NAME
    )
    if packed_manifest_hash != manifest_member["sha256"]:
        raise RuntimeError("Packed full replay manifest fails its recorded SHA-256.")
    tmp_manifest = target_manifest_path.with_name(f"{REPLAY_MANIFEST_NAME}.tmp")
    shutil.copy2(packed_manifest_path, tmp_manifest)
    with tmp_manifest.open("rb") as handle:
        os.fsync(handle.fileno())
    tmp_manifest.replace(target_manifest_path)

    record["status"] = "complete"
    record["full_manifest_sha256"] = packed_manifest_hash
    record["elapsed_seconds"] = time.perf_counter() - started
    _write_json_atomic(in_progress_path, record)
    in_progress_path.replace(target_dir / REPAIR_RECORD_NAME)
    print(
        f"[PASS] repaired replay is byte-identical to packed index in "
        f"{record['elapsed_seconds']:.1f} seconds",
        flush=True,
    )


def main() -> None:
    args = _parse_args()
    index, members = _preflight(args)
    if not args.apply:
        print(
            "[INFO] dry run only; pass --apply with the confirmation token to repair."
        )
        return
    _apply(args, index, members)


if __name__ == "__main__":
    try:
        main()
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"[FATAL] {error}") from None
