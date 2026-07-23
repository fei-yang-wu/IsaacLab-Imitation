#!/usr/bin/env python3
"""Pack the converted BONES-SEED NPZ tree into WebDataset-style tar shards.

129k loose NPZ files make a Hugging Face dataset repo painful (git-LFS pointer
per file, slow clones). This packs them into deterministic ~1 GB tar shards that
stream well and upload fast. Each shard is a plain ``.tar`` whose members are the
individual ``<motion_name>.npz`` files, so a WebDataset / stream loader can read
them directly, and any consumer can ``tar xf`` a shard to recover loose NPZs.

Outputs under ``--out_dir``:
  * ``shards/bones_seed_g1-{NNNN}.tar``   -- the packed motions,
  * ``shard_index.json``                  -- shard -> members, sizes, sha256,
  * a copy of the manifest and language sidecar if provided.

Deterministic: motions are packed in sorted filename order, so identical inputs
produce identical shards.

    pixi run python scripts/pack_bones_seed_shards.py \
        --npz_dir ~/Storage/bones_seed_full/npz/g1 \
        --out_dir ~/Storage/bones_seed_full/hf \
        --shard_bytes 1000000000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import time
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument(
        "--shard_bytes",
        type=int,
        default=1_000_000_000,
        help="Approximate uncompressed bytes per shard before rolling over.",
    )
    parser.add_argument("--shard_prefix", type=str, default="bones_seed_g1")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--language", type=Path, default=None)
    parser.add_argument("--selection", type=Path, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Allow writing into a non-empty shards/ directory.",
    )
    args = parser.parse_args()

    npz_dir = args.npz_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(shards_dir.glob("*.tar"))
    if existing and not args.overwrite:
        raise SystemExit(
            f"[ERROR] {len(existing)} shard(s) already exist in {shards_dir}. "
            "Use --overwrite to repack."
        )
    for stale in existing:
        stale.unlink()

    npz_files = sorted(npz_dir.glob("*.npz"))
    if not npz_files:
        raise SystemExit(f"[ERROR] no NPZ files under {npz_dir}")
    print(f"[pack] {len(npz_files)} NPZ -> shards of ~{args.shard_bytes / 1e9:.2f} GB")

    shard_index: list[dict[str, object]] = []
    shard_id = 0
    members: list[str] = []
    cur_bytes = 0
    tar: tarfile.TarFile | None = None
    shard_path: Path | None = None
    total_bytes = 0
    t0 = time.time()

    def _open_shard(idx: int) -> tuple[tarfile.TarFile, Path]:
        path = shards_dir / f"{args.shard_prefix}-{idx:04d}.tar"
        return tarfile.open(path, mode="w"), path

    def _close_shard() -> None:
        nonlocal tar, shard_path, members, cur_bytes
        if tar is None or shard_path is None:
            return
        tar.close()
        shard_index.append(
            {
                "shard": shard_path.name,
                "num_motions": len(members),
                "uncompressed_bytes": cur_bytes,
                "sha256": _sha256(shard_path),
                "members": members,
            }
        )
        print(
            f"[pack] wrote {shard_path.name}: {len(members)} motions, "
            f"{cur_bytes / 1e9:.2f} GB"
        )
        members = []
        cur_bytes = 0

    for i, npz in enumerate(npz_files):
        size = npz.stat().st_size
        if tar is None:
            tar, shard_path = _open_shard(shard_id)
        elif cur_bytes + size > args.shard_bytes and members:
            _close_shard()
            shard_id += 1
            tar, shard_path = _open_shard(shard_id)
        tar.add(npz, arcname=npz.name)
        members.append(npz.name)
        cur_bytes += size
        total_bytes += size
        if (i + 1) % 10000 == 0:
            print(f"[pack] {i + 1}/{len(npz_files)} packed")
    _close_shard()

    index_payload = {
        "dataset_name": "bones_seed_sonic_full",
        "npz_dir": str(npz_dir),
        "num_motions": len(npz_files),
        "num_shards": len(shard_index),
        "total_uncompressed_bytes": total_bytes,
        "shard_bytes_target": args.shard_bytes,
        "shards": shard_index,
    }
    (out_dir / "shard_index.json").write_text(
        json.dumps(index_payload, indent=2) + "\n", encoding="utf-8"
    )

    # Copy sidecar metadata next to the shards so the repo is self-describing.
    for label, src in (
        ("manifest", args.manifest),
        ("language", args.language),
        ("selection", args.selection),
    ):
        if src is None:
            continue
        src = src.expanduser().resolve()
        if not src.is_file():
            print(f"[pack] WARNING: {label} not found: {src}")
            continue
        dst = out_dir / src.name
        dst.write_bytes(src.read_bytes())
        print(f"[pack] copied {label}: {dst}")

    print(
        f"[pack] done: {len(shard_index)} shard(s), {total_bytes / 1e9:.2f} GB total, "
        f"in {time.time() - t0:.1f}s"
    )
    print(f"[pack] index: {out_dir / 'shard_index.json'}")


if __name__ == "__main__":
    main()
