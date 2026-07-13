#!/usr/bin/env python3
"""Prepare a small BONES-SEED G1 CSV subset as LAFAN1-style NPZ references.

The BONES-SEED Hugging Face repo stores Unitree G1 CSV motions in a gated
``g1.tar.gz`` archive. This helper keeps the task-specific bookkeeping outside
the generic LAFAN1 converter:

1. read a timeline shortlist JSON,
2. find or extract the selected ``<filename>.csv`` files,
3. copy them under ``data/bones_seed/raw/g1/``,
4. normalize BONES G1 CSV layout to this repo's converter CSV layout,
5. run ``prepare_lafan1_from_csv.py`` to export NPZs and a manifest, and
6. write a sidecar JSON with the natural-language timeline labels.

Run it from the repo root with the Isaac Lab Pixi environment, for example:

    pixi run -e isaaclab python scripts/prepare_bones_seed_subset.py \
        --archive data/bones_seed/raw/g1.tar.gz --headless --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHORTLIST = (
    REPO_ROOT / "data" / "bones_seed" / "curated" / "bones_seed_10_shortlist.timeline.json"
)
DEFAULT_CSV_DIR = REPO_ROOT / "data" / "bones_seed" / "raw" / "g1"
DEFAULT_NPZ_DIR = REPO_ROOT / "data" / "bones_seed" / "npz" / "g1"
DEFAULT_MANIFEST = (
    REPO_ROOT / "data" / "bones_seed" / "manifests" / "g1_bones_seed_10_manifest.json"
)
DEFAULT_LANGUAGE = (
    REPO_ROOT / "data" / "bones_seed" / "language" / "g1_bones_seed_10_language.json"
)
DEFAULT_METADATA_CSV = (
    REPO_ROOT / "data" / "bones_seed" / "raw" / "metadata" / "seed_metadata_v004.csv"
)
DEFAULT_SOURCE_DIRS = (
    REPO_ROOT / "data" / "bones_seed" / "raw" / "g1" / "csv",
    REPO_ROOT / "data" / "bones_seed" / "raw" / "g1",
    REPO_ROOT / "data" / "bones_seed" / "raw" / "extracted" / "g1" / "csv",
    REPO_ROOT / "data" / "bones_seed" / "raw" / "extracted" / "g1",
)
ILTOOLS_ZARR_SHARD_SIZE = 512
BONES_G1_ROOT_ROTATION_CONVENTION = (
    "CSV root_rotateX/Y/Z columns are degrees. BONES seed-viewer applies them "
    "as Three.js Euler(rx, ry, rz, 'ZYX'), equivalent to SciPy lowercase "
    "from_euler('xyz', [rx, ry, rz]) / extrinsic XYZ."
)


def _load_unitree_g1_joint_order() -> tuple[tuple[str, ...], str]:
    """Load the repo's canonical G1 order without importing Isaac task modules."""
    module_path = (
        REPO_ROOT
        / "source"
        / "isaaclab_imitation"
        / "isaaclab_imitation"
        / "assets"
        / "robots"
        / "unitree_joint_order.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_isaaclab_imitation_unitree_joint_order", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Unitree joint-order module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return (
        tuple(module.UNITREE_G1_29DOF_SDK_JOINT_NAMES),
        str(module.UNITREE_G1_29DOF_JOINT_ORDER_SOURCE),
    )


UNITREE_G1_29DOF_JOINT_NAMES, UNITREE_G1_29DOF_JOINT_ORDER_SOURCE = (
    _load_unitree_g1_joint_order()
)
BONES_G1_ROOT_COLUMNS: tuple[str, ...] = (
    "Frame",
    "root_translateX",
    "root_translateY",
    "root_translateZ",
    "root_rotateX",
    "root_rotateY",
    "root_rotateZ",
)
BONES_G1_EXPECTED_COLUMNS: tuple[str, ...] = BONES_G1_ROOT_COLUMNS + tuple(
    f"{joint_name}_dof" for joint_name in UNITREE_G1_29DOF_JOINT_NAMES
)


def _sanitize_motion_name(path_without_suffix: Path) -> str:
    name = path_without_suffix.as_posix()
    name = re.sub(r"[^A-Za-z0-9_\-/]+", "_", name)
    name = name.replace("/", "__").replace("-", "_")
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "motion"


def _load_shortlist(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) == 0:
        raise ValueError(f"Shortlist must be a non-empty JSON list: {path}")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(payload):
        if not isinstance(row, dict) or not row.get("filename"):
            raise ValueError(f"Shortlist entry #{index} needs a filename field.")
        rows.append(row)
    return rows


def _wanted_csv_names(rows: list[dict[str, Any]]) -> list[str]:
    return [f"{str(row['filename']).removesuffix('.csv')}.csv" for row in rows]


def _candidate_source_dirs(explicit_dirs: list[str] | None) -> list[Path]:
    dirs = [Path(value).expanduser().resolve() for value in explicit_dirs or []]
    dirs.extend(path.resolve() for path in DEFAULT_SOURCE_DIRS)
    return list(dict.fromkeys(dirs))


def _find_existing_csvs(
    wanted_csv_names: list[str],
    *,
    source_dirs: list[Path],
) -> dict[str, Path]:
    wanted = set(wanted_csv_names)
    found: dict[str, Path] = {}
    for source_dir in source_dirs:
        if not source_dir.is_dir():
            continue
        for path in source_dir.rglob("*.csv"):
            if path.name in wanted and path.name not in found:
                found[path.name] = path.resolve()
        if len(found) == len(wanted):
            break
    return found


def _extract_missing_from_archive(
    archive_path: Path,
    *,
    missing_csv_names: set[str],
    csv_dir: Path,
) -> dict[str, Path]:
    extracted: dict[str, Path] = {}
    if len(missing_csv_names) == 0:
        return extracted
    if not archive_path.is_file():
        raise FileNotFoundError(f"BONES-SEED archive not found: {archive_path}")

    csv_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Scanning archive for {len(missing_csv_names)} CSV(s): {archive_path}")
    with tarfile.open(archive_path, mode="r:*") as tar:
        for member in tar:
            if not member.isfile():
                continue
            member_name = Path(member.name).name
            if member_name not in missing_csv_names:
                continue
            target = csv_dir / member_name
            source = tar.extractfile(member)
            if source is None:
                raise RuntimeError(f"Could not extract archive member: {member.name}")
            with source, target.open("wb") as handle:
                shutil.copyfileobj(source, handle)
            extracted[member_name] = target.resolve()
            print(f"[INFO] Extracted {member.name} -> {target}")
            if len(extracted) == len(missing_csv_names):
                break
    return extracted


def _read_csv_header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return next(csv.reader(handle), [])


def _is_bones_g1_csv(path: Path) -> bool:
    first_row = _read_csv_header(path)
    return len(first_row) == len(BONES_G1_EXPECTED_COLUMNS) and tuple(
        first_row[: len(BONES_G1_ROOT_COLUMNS)]
    ) == BONES_G1_ROOT_COLUMNS


def _validate_bones_g1_columns(path: Path) -> list[str]:
    header = _read_csv_header(path)
    if tuple(header) != BONES_G1_EXPECTED_COLUMNS:
        expected_joint_columns = list(BONES_G1_EXPECTED_COLUMNS[len(BONES_G1_ROOT_COLUMNS) :])
        actual_joint_columns = header[len(BONES_G1_ROOT_COLUMNS) :]
        raise ValueError(
            "BONES-SEED G1 CSV joint columns do not match this repo's Unitree "
            f"G1 order for {path}.\n"
            f"Expected: {expected_joint_columns}\n"
            f"Actual:   {actual_joint_columns}"
        )
    return header


def _write_converter_csv(source: Path, target: Path) -> str:
    """Write a converter-ready CSV and return the detected source format."""
    if not _is_bones_g1_csv(source):
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        return "lafan1_numeric"

    _validate_bones_g1_columns(source)
    data = np.loadtxt(source, delimiter=",", skiprows=1, dtype=np.float64)
    data = np.atleast_2d(data)
    if data.shape[1] != 36:
        raise ValueError(
            f"BONES-SEED G1 CSV must have 36 columns including Frame, got "
            f"{data.shape[1]} in {source}"
        )

    root_pos_m = data[:, 1:4] * 0.01
    # BONES seed-viewer uses Three.js Euler(rx, ry, rz, "ZYX"). In SciPy,
    # lowercase "xyz" with the CSV X/Y/Z columns is the matching extrinsic XYZ
    # convention.
    root_quat_xyzw = Rotation.from_euler(
        "xyz", data[:, 4:7], degrees=True
    ).as_quat()
    joint_pos_rad = np.deg2rad(data[:, 7:])
    output = np.concatenate([root_pos_m, root_quat_xyzw, joint_pos_rad], axis=1)
    if output.shape[1] != 36:
        raise RuntimeError(f"Converter CSV shape bug for {source}: {output.shape}")

    temp_target = target.with_suffix(target.suffix + ".tmp")
    np.savetxt(temp_target, output.astype(np.float32), delimiter=",", fmt="%.9g")
    temp_target.replace(target)
    return "bones_g1_euler_degrees_cm"


def _stage_csvs(
    found_csvs: dict[str, Path],
    *,
    csv_dir: Path,
    copy_mode: str,
) -> dict[str, Path]:
    csv_dir.mkdir(parents=True, exist_ok=True)
    staged: dict[str, Path] = {}
    for csv_name, source in sorted(found_csvs.items()):
        target = (csv_dir / csv_name).resolve()
        source_format = None
        if _is_bones_g1_csv(source):
            source_format = _write_converter_csv(source, target)
            staged[csv_name] = target
            print(f"[INFO] Normalized {source_format}: {source} -> {target}")
            continue
        if source.resolve() == target:
            staged[csv_name] = target
            continue
        target.unlink(missing_ok=True)
        if copy_mode == "copy":
            shutil.copy2(source, target)
        elif copy_mode == "symlink":
            target.symlink_to(source)
        elif copy_mode == "hardlink":
            os.link(source, target)
        else:
            raise ValueError(f"Unsupported copy mode: {copy_mode}")
        staged[csv_name] = target
        print(f"[INFO] Staged {source_format or 'csv'}: {source} -> {target}")
    return staged


def _load_metadata_by_filename(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row["filename"]: row
            for row in csv.DictReader(handle)
            if row.get("filename")
        }


def _write_language_sidecar(
    rows: list[dict[str, Any]],
    *,
    output_path: Path,
    manifest_path: Path,
    input_fps: float,
    output_fps: float,
    metadata_by_filename: dict[str, dict[str, str]],
) -> None:
    motions: list[dict[str, Any]] = []
    for row in rows:
        filename = str(row["filename"]).removesuffix(".csv")
        motion_name = _sanitize_motion_name(Path(filename))
        metadata = metadata_by_filename.get(filename, {})
        motions.append(
            {
                "name": motion_name,
                "bones_seed_filename": filename,
                "category": metadata.get("category"),
                "move_duration_frames": int(metadata.get("move_duration_frames", 0))
                if metadata.get("move_duration_frames")
                else None,
                "language_goal": row.get("overview_description", ""),
                "natural_descriptions": [
                    metadata[key]
                    for key in (
                        "content_natural_desc_1",
                        "content_natural_desc_2",
                        "content_natural_desc_3",
                        "content_natural_desc_4",
                    )
                    if metadata.get(key)
                ],
                "short_descriptions": [
                    metadata[key]
                    for key in (
                        "content_short_description",
                        "content_short_description_2",
                    )
                    if metadata.get(key)
                ],
                "technical_description": metadata.get("content_technical_description"),
                "events": row.get("events", []),
                "num_events": int(row.get("num_events", len(row.get("events", [])))),
                "propagated_from_filename": row.get("propagated_from_filename"),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_name": "bones_seed",
        "source": "bones-studio/seed",
        "manifest": str(manifest_path),
        "input_fps": float(input_fps),
        "output_fps": float(output_fps),
        "motions": motions,
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[INFO] Wrote language sidecar: {output_path}")


def _run_converter(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "prepare_lafan1_from_csv.py"),
        "--csv_dir",
        str(args.csv_dir),
        "--npz_dir",
        str(args.npz_dir),
        "--manifest_path",
        str(args.manifest_path),
        "--recursive",
        "--dataset_name",
        "bones_seed",
        "--input_fps",
        str(float(args.input_fps)),
        "--output_fps",
        str(float(args.output_fps)),
    ]
    if args.headless:
        cmd.append("--headless")
    if args.device is not None:
        cmd.extend(["--device", str(args.device)])
    if args.overwrite:
        cmd.append("--overwrite")
    if args.skip_existing:
        cmd.append("--skip_existing")
    if args.record_videos:
        cmd.append("--record_videos")
        if args.overwrite_videos:
            cmd.append("--overwrite_videos")

    print(f"[CMD] {' '.join(shlex.quote(part) for part in cmd)}")
    subprocess.run(cmd, check=True)


def _pad_npz_arrays_to_transition_multiple(
    npz_dir: Path, multiple: int
) -> dict[str, dict[str, int]]:
    if multiple <= 1:
        return {}
    summary: dict[str, dict[str, int]] = {}
    for npz_path in sorted(npz_dir.glob("*.npz")):
        with np.load(npz_path, allow_pickle=False) as data:
            payload = {key: data[key] for key in data.files}
        if "joint_pos" not in payload:
            continue
        frame_count = int(payload["joint_pos"].shape[0])
        transition_count = max(frame_count - 1, 1)
        padded_transition_count = (
            (transition_count + multiple - 1) // multiple
        ) * multiple
        padded_transition_count = max(
            padded_transition_count, ILTOOLS_ZARR_SHARD_SIZE
        )
        padded_frame_count = padded_transition_count + 1
        if padded_frame_count == frame_count:
            continue
        pad_count = padded_frame_count - frame_count
        padded_payload: dict[str, np.ndarray] = {}
        for key, value in payload.items():
            if value.ndim >= 1 and int(value.shape[0]) == frame_count:
                tail = np.repeat(value[-1:], pad_count, axis=0)
                padded_payload[key] = np.concatenate([value, tail], axis=0)
            else:
                padded_payload[key] = value
        original_frame_count = int(
            np.asarray(payload.get("original_frame_count", [frame_count])).reshape(-1)[0]
        )
        padded_payload["original_frame_count"] = np.asarray(
            [original_frame_count], dtype=np.int64
        )
        padded_payload["padded_frame_count"] = np.asarray(
            [padded_frame_count], dtype=np.int64
        )
        padded_payload["padding_multiple"] = np.asarray([multiple], dtype=np.int64)
        np.savez(npz_path, **padded_payload)
        summary[npz_path.name] = {
            "original_frames": original_frame_count,
            "previous_frames": frame_count,
            "padded_frames": padded_frame_count,
            "padded_transitions": padded_transition_count,
        }
        print(
            f"[INFO] Padded NPZ to {multiple}-transition multiple: {npz_path.name} "
            f"{frame_count} -> {padded_frame_count}"
        )
    return summary


def _annotate_manifest(
    manifest_path: Path,
    *,
    language_path: Path,
    input_fps: float,
    output_fps: float,
    padding_summary: dict[str, dict[str, int]],
) -> None:
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = manifest.setdefault("metadata", {})
    metadata["source_dataset"] = "bones-studio/seed"
    metadata["language_annotations_path"] = os.path.relpath(
        language_path, manifest_path.parent
    )
    metadata["input_fps_note"] = (
        "BONES-SEED dataset card reports total duration at 120 fps; "
        "use --input_fps to override if a CSV source says otherwise."
    )
    metadata["input_fps"] = float(input_fps)
    metadata["output_fps"] = float(output_fps)
    metadata["joint_order"] = {
        "source": "BONES-SEED G1 CSV header columns after root fields",
        "matches": UNITREE_G1_29DOF_JOINT_ORDER_SOURCE,
        "joint_names": list(UNITREE_G1_29DOF_JOINT_NAMES),
    }
    metadata["root_rotation_convention"] = BONES_G1_ROOT_ROTATION_CONVENTION
    metadata["npz_padding_default"] = (
        "disabled; use ILTools loader chunk_size=1 for short trajectories instead "
        "of repeating final frames."
    )
    metadata["loader_kwargs"] = {
        "chunk_size": 1,
        "shard_size": ILTOOLS_ZARR_SHARD_SIZE,
    }
    if padding_summary:
        metadata["npz_padding"] = {
            "reason": (
                "Pad short trajectories by repeating the final frame so the "
                "Zarr loader full-frame and next-frame arrays use shard sizes "
                "divisible by the inner chunk size."
            ),
            "motions": padding_summary,
        }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[INFO] Annotated manifest metadata: {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a curated BONES-SEED subset as LAFAN1-style G1 NPZ data."
    )
    parser.add_argument("--shortlist", type=Path, default=DEFAULT_SHORTLIST)
    parser.add_argument(
        "--source_dir",
        action="append",
        default=None,
        help="Directory to search for extracted BONES-SEED CSVs. Can be repeated.",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="Optional local BONES-SEED g1.tar.gz path to scan/extract selected CSVs.",
    )
    parser.add_argument("--csv_dir", type=Path, default=DEFAULT_CSV_DIR)
    parser.add_argument("--npz_dir", type=Path, default=DEFAULT_NPZ_DIR)
    parser.add_argument("--manifest_path", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--language_path", type=Path, default=DEFAULT_LANGUAGE)
    parser.add_argument("--metadata_csv", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument(
        "--input_fps",
        type=float,
        default=120.0,
        help="Source CSV FPS. BONES-SEED is documented as 120 fps.",
    )
    parser.add_argument(
        "--output_fps",
        type=float,
        default=50.0,
        help="Exported NPZ FPS. 50 Hz matches existing LAFAN1-style references.",
    )
    parser.add_argument(
        "--copy_mode",
        choices=("copy", "symlink", "hardlink"),
        default="copy",
        help="How selected CSVs are staged into data/bones_seed/raw/g1.",
    )
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--skip_existing", action="store_true", default=False)
    parser.add_argument("--record_videos", action="store_true", default=False)
    parser.add_argument("--overwrite_videos", action="store_true", default=False)
    parser.add_argument(
        "--pad_npz_multiple",
        type=int,
        default=1,
        help=(
            "Pad exported NPZ trajectories so their N-1 transition count is a "
            "multiple of this value by repeating the final frame. Use 1 to "
            "disable. Padding is disabled by default because it creates frozen "
            "tails during reference replay."
        ),
    )
    parser.add_argument(
        "--no_convert",
        action="store_true",
        default=False,
        help="Only stage CSVs and write language sidecar; do not run Isaac conversion.",
    )
    parser.add_argument("--dry_run", action="store_true", default=False)
    args = parser.parse_args()

    args.shortlist = args.shortlist.expanduser().resolve()
    args.csv_dir = args.csv_dir.expanduser().resolve()
    args.npz_dir = args.npz_dir.expanduser().resolve()
    args.manifest_path = args.manifest_path.expanduser().resolve()
    args.language_path = args.language_path.expanduser().resolve()
    args.metadata_csv = args.metadata_csv.expanduser().resolve()
    if args.archive is not None:
        args.archive = args.archive.expanduser().resolve()

    rows = _load_shortlist(args.shortlist)
    metadata_by_filename = _load_metadata_by_filename(args.metadata_csv)
    wanted_csv_names = _wanted_csv_names(rows)
    source_dirs = _candidate_source_dirs(args.source_dir)
    found: dict[str, Path] = {}
    if args.archive is not None and args.overwrite:
        print("[INFO] Refreshing selected raw CSVs from archive because --overwrite is set.")
        found.update(
            _extract_missing_from_archive(
                args.archive,
                missing_csv_names=set(wanted_csv_names),
                csv_dir=args.csv_dir,
            )
        )
    found.update(_find_existing_csvs(wanted_csv_names, source_dirs=source_dirs))
    missing = set(wanted_csv_names) - set(found)

    if missing and args.archive is not None:
        found.update(
            _extract_missing_from_archive(
                args.archive, missing_csv_names=missing, csv_dir=args.csv_dir
            )
        )
        missing = set(wanted_csv_names) - set(found)

    print(f"[INFO] Shortlist motions: {len(wanted_csv_names)}")
    print(f"[INFO] Found CSVs:        {len(found)}")
    if missing:
        print("[ERROR] Missing selected CSVs:")
        for csv_name in sorted(missing):
            print(f"  - {csv_name}")
        searched = "\n".join(f"  - {path}" for path in source_dirs)
        raise SystemExit(
            "Selected BONES-SEED CSVs are not available yet. Search dirs:\n"
            f"{searched}\n"
            "Download/extract the gated BONES-SEED G1 archive, or pass --archive "
            "with a local g1.tar.gz."
        )

    if args.dry_run:
        for csv_name, path in sorted(found.items()):
            print(f"[DRY-RUN] {csv_name}: {path}")
        return

    _stage_csvs(found, csv_dir=args.csv_dir, copy_mode=str(args.copy_mode))
    _write_language_sidecar(
        rows,
        output_path=args.language_path,
        manifest_path=args.manifest_path,
        input_fps=float(args.input_fps),
        output_fps=float(args.output_fps),
        metadata_by_filename=metadata_by_filename,
    )
    if not args.no_convert:
        _run_converter(args)
        padding_summary = _pad_npz_arrays_to_transition_multiple(
            args.npz_dir, int(args.pad_npz_multiple)
        )
        _annotate_manifest(
            args.manifest_path,
            language_path=args.language_path,
            input_fps=float(args.input_fps),
            output_fps=float(args.output_fps),
            padding_summary=padding_summary,
        )

    print("[INFO] BONES-SEED subset prep complete.")


if __name__ == "__main__":
    main()
