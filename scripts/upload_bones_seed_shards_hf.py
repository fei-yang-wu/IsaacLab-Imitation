#!/usr/bin/env python3
"""Upload the packed BONES-SEED shard tree to a (private) Hugging Face dataset repo.

Uses ``upload_large_folder`` (resumable, parallel, integrity-checked) because the
tree is ~100 GB across ~100 tar shards. The repo-owned NPZ uploader
(``setup_g1_bones_seed_npz_dataset.py``) is NPZ-tree specific and rejects a
shard-only folder, so this is a dedicated entrypoint.

    pixi run python scripts/upload_bones_seed_shards_hf.py \
        --repo_id GeorgiaTech/g1_bones_seed_sonic_129k_50hz \
        --folder ~/Storage/bones_seed_full/hf --private
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--private", action="store_true", default=False)
    parser.add_argument("--token", default=None, help="HF token (default: cached login).")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help="Create the repo and list what would upload without uploading.",
    )
    args = parser.parse_args()

    from huggingface_hub import HfApi

    folder = args.folder.expanduser().resolve()
    if not folder.is_dir():
        raise NotADirectoryError(f"folder does not exist: {folder}")
    shards = sorted((folder / "shards").glob("*.tar"))
    sidecars = sorted(p.name for p in folder.glob("*.json")) + (
        ["README.md"] if (folder / "README.md").is_file() else []
    )
    if not shards:
        raise RuntimeError(f"No .tar shards under {folder / 'shards'}")

    total_bytes = sum(p.stat().st_size for p in shards)
    print(f"[upload] repo_id={args.repo_id} private={args.private}")
    print(f"[upload] folder={folder}")
    print(f"[upload] shards={len(shards)} ({total_bytes / 1e9:.1f} GB) sidecars={sidecars}")

    api = HfApi()
    print(f"[upload] ensuring dataset repo exists: {args.repo_id}")
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=bool(args.private),
        exist_ok=True,
        token=args.token,
    )
    if args.dry_run:
        print("[upload] dry-run: repo ensured; not uploading.")
        return

    api.upload_large_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(folder),
        print_report=True,
    )
    print(f"[upload] done: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
