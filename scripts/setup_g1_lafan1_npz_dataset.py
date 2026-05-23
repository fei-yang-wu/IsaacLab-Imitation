#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sync the G1 LAFAN1 NPZ dataset with a Hugging Face dataset repo.

This script is a companion to ``scripts/setup_lafan1_dataset.py``. Unlike the
CSV-based workflow, it only handles an ``npz/g1`` subtree that is already ready
for this repo's NPZ loader tooling.

Examples:
    conda run -n SL python scripts/setup_g1_lafan1_npz_dataset.py

    conda run -n SL python scripts/setup_g1_lafan1_npz_dataset.py \
        --mode upload --token "$HF_TOKEN"
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path, PurePosixPath

try:
    from huggingface_hub import HfApi, snapshot_download
except ImportError as exc:  # pragma: no cover - import guard for misconfigured envs
    raise ImportError(
        "huggingface_hub is required. Install it in the SL environment "
        "and rerun this script."
    ) from exc


DEFAULT_REPO_ID = "GeorgiaTech/g1_lafan1_50hz"
DEFAULT_REPO_SUBDIR = "npz/g1"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_data_root() -> Path:
    return _repo_root() / "data" / "lafan1"


def _normalize_repo_subdir(repo_subdir: str) -> str:
    normalized_path = PurePosixPath(repo_subdir.strip("/"))
    normalized = str(normalized_path)
    if normalized in {"", "."}:
        raise ValueError("--repo_subdir must not be empty.")
    if ".." in normalized_path.parts:
        raise ValueError("--repo_subdir must not contain parent-directory segments.")
    return normalized


def _repo_subdir_path(repo_subdir: str) -> Path:
    return Path(*PurePosixPath(repo_subdir).parts)


def _dataset_repo_url(repo_id: str) -> str:
    return f"https://huggingface.co/datasets/{repo_id}"


def _count_npz_files(root: Path) -> int:
    if not root.is_dir():
        return 0
    return sum(1 for path in root.rglob("*.npz") if path.is_file())


def _allow_patterns(repo_subdir: str) -> list[str]:
    return [f"{repo_subdir}/*", f"{repo_subdir}/**"]


def _format_command(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download or upload the Hugging Face G1 LAFAN1 NPZ dataset subtree "
            "(defaults to npz/g1)."
        )
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=("download", "upload"),
        default="download",
        help="Whether to pull data from Hugging Face or push a local NPZ tree to it.",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=DEFAULT_REPO_ID,
        help="Hugging Face dataset repo id.",
    )
    parser.add_argument(
        "--repo_subdir",
        type=str,
        default=DEFAULT_REPO_SUBDIR,
        help="Dataset repo subdirectory to sync.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=str(_default_data_root()),
        help=(
            "Local root used to mirror the Hugging Face dataset subtree. "
            "With the default repo_subdir this writes under data/lafan1/npz/g1."
        ),
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Dataset repo revision or branch name.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Optional Hugging Face token. Uploads usually require this or a prior hf login.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        default=False,
        help="Force Hugging Face to refresh downloaded files.",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        default=False,
        help=(
            "Allow download mode to succeed even when the remote repo does not yet "
            "contain any .npz files under repo_subdir."
        ),
    )
    parser.add_argument(
        "--create-repo",
        action="store_true",
        default=False,
        help="Create the dataset repo before upload if it does not already exist.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        default=False,
        help="Create the dataset repo as private when combined with --create-repo.",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default=None,
        help="Optional commit message for upload mode.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable included in printed follow-up commands.",
    )
    return parser


def _download_dataset(
    *,
    repo_id: str,
    repo_subdir: str,
    data_root: Path,
    revision: str,
    token: str | None,
    force_download: bool,
    allow_empty: bool,
) -> None:
    target_dir = data_root / _repo_subdir_path(repo_subdir)
    data_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Downloading dataset subtree from: {_dataset_repo_url(repo_id)}")
    print(f"[INFO] Revision: {revision}")
    print(f"[INFO] Repo subdir: {repo_subdir}")
    print(f"[INFO] Local target: {target_dir}")
    print(f"[INFO] Allow patterns: {_allow_patterns(repo_subdir)}")

    snapshot_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=str(data_root),
        allow_patterns=_allow_patterns(repo_subdir),
        force_download=bool(force_download),
        token=token,
    )

    npz_count = _count_npz_files(target_dir)
    print(f"[INFO] Download snapshot root: {snapshot_path}")
    print(f"[INFO] Local NPZ count under {target_dir}: {npz_count}")

    if npz_count == 0 and not allow_empty:
        raise RuntimeError(
            "No .npz files were found after download. "
            f"Expected them under {target_dir}. "
            f"Check whether {_dataset_repo_url(repo_id)} contains files under {repo_subdir}, "
            "or rerun with --allow-empty if the repo has been created but not populated yet."
        )

    if npz_count == 0:
        print(
            "[INFO] Remote dataset subtree is currently empty; nothing else to download."
        )


def _upload_dataset(
    *,
    repo_id: str,
    repo_subdir: str,
    data_root: Path,
    revision: str,
    token: str | None,
    create_repo: bool,
    private: bool,
    commit_message: str | None,
) -> None:
    source_dir = data_root / _repo_subdir_path(repo_subdir)
    npz_count = _count_npz_files(source_dir)

    if not source_dir.is_dir():
        raise NotADirectoryError(f"Local NPZ directory does not exist: {source_dir}")
    if npz_count == 0:
        raise RuntimeError(f"No .npz files found under local source dir: {source_dir}")

    api = HfApi()
    if create_repo:
        print(f"[INFO] Ensuring dataset repo exists: {_dataset_repo_url(repo_id)}")
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=bool(private),
            exist_ok=True,
            token=token,
        )

    commit_message = commit_message or f"Upload {repo_subdir} NPZ data"

    print(f"[INFO] Uploading local NPZ subtree to: {_dataset_repo_url(repo_id)}")
    print(f"[INFO] Revision: {revision}")
    print(f"[INFO] Repo subdir: {repo_subdir}")
    print(f"[INFO] Local source: {source_dir}")
    print(f"[INFO] Local NPZ count: {npz_count}")

    commit_info = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(source_dir),
        path_in_repo=repo_subdir,
        revision=revision,
        token=token,
        allow_patterns=["*.npz", "**/*.npz"],
        commit_message=commit_message,
    )

    commit_url = getattr(commit_info, "commit_url", None)
    commit_oid = getattr(commit_info, "oid", None)
    print(f"[INFO] Upload finished for {npz_count} NPZ files.")
    if commit_oid:
        print(f"[INFO] Commit OID: {commit_oid}")
    if commit_url:
        print(f"[INFO] Commit URL: {commit_url}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    repo_subdir = _normalize_repo_subdir(args.repo_subdir)
    data_root = Path(args.data_root).expanduser().resolve()
    local_npz_dir = data_root / _repo_subdir_path(repo_subdir)

    print(f"[INFO] Local NPZ directory resolved to: {local_npz_dir}")

    if args.mode == "download":
        _download_dataset(
            repo_id=args.repo_id,
            repo_subdir=repo_subdir,
            data_root=data_root,
            revision=args.revision,
            token=args.token,
            force_download=bool(args.force_download),
            allow_empty=bool(args.allow_empty),
        )

        preview_cmd = [
            args.python,
            str(_repo_root() / "scripts" / "write_lafan1_npz_manifest.py"),
            "--npz_dir",
            str(local_npz_dir),
            "--manifest_path",
            str(data_root / "manifests" / "g1_lafan1_manifest.json"),
            "--recursive",
        ]
        print("[INFO] If you want a manifest for the loader, run:")
        print(f"[CMD]  {_format_command(preview_cmd)}")
        return

    _upload_dataset(
        repo_id=args.repo_id,
        repo_subdir=repo_subdir,
        data_root=data_root,
        revision=args.revision,
        token=args.token,
        create_repo=bool(args.create_repo),
        private=bool(args.private),
        commit_message=args.commit_message,
    )


if __name__ == "__main__":
    main()
