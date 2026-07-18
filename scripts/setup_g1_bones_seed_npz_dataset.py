#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sync the G1 BONES-SEED-100 NPZ dataset with a Hugging Face dataset repo.

Companion to ``scripts/setup_g1_lafan1_npz_dataset.py`` for the curated
100-motion BONES-SEED subset. The repo stores a self-contained dataset tree:

    npz/g1/*.npz                             local-frame G1 references (joint_names embedded)
    manifests/g1_bones_seed_100_manifest.json training manifest (canonical joint order)
    language/g1_bones_seed_100_language.json  natural-language descriptions + metadata
    curated/bones_seed_100_provenance.json    per-clip source provenance

Examples:
    # Pull the dataset into data/bones_seed_100 (default):
    pixi run python scripts/setup_g1_bones_seed_npz_dataset.py

    # Create the (private) repo and push a freshly built local tree:
    pixi run python scripts/setup_g1_bones_seed_npz_dataset.py \
        --mode upload --create-repo --private --token "$HF_TOKEN"
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, snapshot_download
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "huggingface_hub is required. Run this script through the Pixi "
        "default or lerobot environment."
    ) from exc


DEFAULT_REPO_ID = "GeorgiaTech/g1_bones_seed_100_50hz"
# Subtrees kept in sync with the remote dataset repo.
DATASET_SUBDIRS = ("npz/g1", "manifests", "language", "curated")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_data_root() -> Path:
    return _repo_root() / "data" / "bones_seed_100"


def _dataset_repo_url(repo_id: str) -> str:
    return f"https://huggingface.co/datasets/{repo_id}"


def _count_npz_files(root: Path) -> int:
    npz_root = root / "npz" / "g1"
    if not npz_root.is_dir():
        return 0
    return sum(1 for path in npz_root.rglob("*.npz") if path.is_file())


def _allow_patterns() -> list[str]:
    patterns: list[str] = ["README.md"]
    for subdir in DATASET_SUBDIRS:
        patterns.extend([f"{subdir}/*", f"{subdir}/**"])
    return patterns


def _format_command(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download or upload the Hugging Face G1 BONES-SEED-100 NPZ dataset."
    )
    parser.add_argument("--mode", choices=("download", "upload"), default="download")
    parser.add_argument("--repo_id", default=DEFAULT_REPO_ID)
    parser.add_argument("--data_root", default=str(_default_data_root()))
    parser.add_argument("--revision", default="main")
    parser.add_argument("--token", default=None)
    parser.add_argument("--force-download", action="store_true", default=False)
    parser.add_argument("--allow-empty", action="store_true", default=False)
    parser.add_argument("--create-repo", action="store_true", default=False)
    parser.add_argument("--private", action="store_true", default=False)
    parser.add_argument("--commit-message", default=None)
    parser.add_argument("--python", default=sys.executable)
    return parser


def _download_dataset(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root).expanduser().resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Downloading dataset from: {_dataset_repo_url(args.repo_id)}")
    print(f"[INFO] Revision: {args.revision}  Local target: {data_root}")
    snapshot_path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(data_root),
        allow_patterns=_allow_patterns(),
        force_download=bool(args.force_download),
        token=args.token,
    )
    npz_count = _count_npz_files(data_root)
    print(f"[INFO] Download snapshot root: {snapshot_path}")
    print(f"[INFO] Local NPZ count under {data_root / 'npz' / 'g1'}: {npz_count}")
    if npz_count == 0 and not args.allow_empty:
        raise RuntimeError(
            f"No .npz files found after download under {data_root / 'npz' / 'g1'}. "
            f"Check {_dataset_repo_url(args.repo_id)} or rerun with --allow-empty."
        )


def _upload_dataset(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root).expanduser().resolve()
    npz_count = _count_npz_files(data_root)
    if not data_root.is_dir():
        raise NotADirectoryError(f"Local data root does not exist: {data_root}")
    if npz_count == 0:
        raise RuntimeError(f"No .npz files found under: {data_root / 'npz' / 'g1'}")

    api = HfApi()
    if args.create_repo:
        print(f"[INFO] Ensuring dataset repo exists: {_dataset_repo_url(args.repo_id)}")
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            private=bool(args.private),
            exist_ok=True,
            token=args.token,
        )

    commit_message = args.commit_message or "Upload BONES-SEED-100 G1 dataset"
    print(f"[INFO] Uploading local dataset tree to: {_dataset_repo_url(args.repo_id)}")
    print(f"[INFO] Local source: {data_root}  NPZ count: {npz_count}")
    commit_info = api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(data_root),
        allow_patterns=_allow_patterns(),
        revision=args.revision,
        commit_message=commit_message,
        token=args.token,
    )
    print(f"[INFO] Upload complete: {commit_info}")


def main() -> None:
    args = _build_parser().parse_args()
    if args.mode == "download":
        _download_dataset(args)
        print("\n[NEXT] Point training at the manifest, e.g.:")
        print(
            "  "
            + _format_command(
                [
                    args.python,
                    "scripts/rlopt/train.py",
                    "--task",
                    "Isaac-Imitation-G1-Latent-v0",
                    "--algo",
                    "IPMD",
                    "--headless",
                    f"env.lafan1_manifest_path={Path(args.data_root) / 'manifests' / 'g1_bones_seed_100_manifest.json'}",
                ]
            )
        )
    else:
        _upload_dataset(args)


if __name__ == "__main__":
    main()
