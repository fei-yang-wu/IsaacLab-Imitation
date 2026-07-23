#!/usr/bin/env python3
"""Convert the full SONIC-filtered BONES-SEED G1 set to LAFAN1-style NPZ at scale.

The single-session ``batch_csv_to_npz.py`` pads every motion in a batch to the
batch's longest trajectory, so throwing all ~130k motions (durations 69 -> 21617
source frames) into one Isaac session would need >100 GB of host RAM. This
orchestrator keeps ``batch_csv_to_npz.py`` byte-for-byte unchanged (so the output
matches the existing lafan1 / bones_seed_100 trees exactly) and instead:

  1. extract  -- pull the selected CSVs out of ``g1.tar.gz`` in one streaming pass,
  2. normalize -- convert BONES euler/deg/cm CSV layout to this repo's converter
                  CSV layout (pos_m, quat_xyzw, joint_rad), in parallel,
  3. convert   -- length-bucket the motions under an ``envs x max_frames`` memory
                  budget and run ``batch_csv_to_npz.py`` once per bucket (PhysX),
  4. anchor    -- re-anchor each NPZ so frame-0 root XY == (0, 0), and
  5. audit     -- verify the 15-key set and canonical joint/body name ordering.

Every stage is resumable: re-running skips CSVs / NPZs that already exist.

Run from the repo root in the Isaac Lab env for the convert stage:

    pixi run -e isaaclab python scripts/convert_bones_seed_full.py \
        --selection ~/Storage/bones_seed_full/selection/g1_bones_seed_sonic_selection.json \
        --archive data/bones_seed/raw/g1.tar.gz \
        --work_root ~/Storage/bones_seed_full \
        --device cuda:0 --headless

Stages 1-2 and 4-5 need no Isaac Sim; pass ``--stages extract,normalize`` in the
default env, then ``--stages convert`` in ``-e isaaclab``, then the rest.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
ALL_STAGES = ("extract", "normalize", "convert", "anchor", "audit", "manifest")
EXPECTED_NPZ_KEYS = {
    "fps",
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
    "joint_names",
    "body_names",
}


def _load_prepare_module():
    """Import prepare_bones_seed_subset for its (Isaac-free) CSV normalizer."""
    module_path = REPO_ROOT / "scripts" / "prepare_bones_seed_subset.py"
    spec = importlib.util.spec_from_file_location(
        "_prepare_bones_seed_subset", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------ selection ---------------------------------


def _load_selection(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "motions" not in payload or not isinstance(payload["motions"], list):
        raise ValueError(f"Selection JSON missing 'motions' list: {path}")
    return payload


def _staged_csv_path(csv_dir: Path, filename: str) -> Path:
    return csv_dir / f"{filename}.csv"


def _npz_path(npz_dir: Path, filename: str) -> Path:
    return npz_dir / f"{filename}.npz"


# ------------------------------ stage: extract ----------------------------


def stage_extract(
    selection: dict[str, Any], archive: Path, raw_dir: Path
) -> dict[str, int]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    wanted: dict[str, str] = {m["g1_path"]: m["filename"] for m in selection["motions"]}
    # Skip already-extracted CSVs.
    remaining = {
        g1_path: filename
        for g1_path, filename in wanted.items()
        if not (raw_dir / f"{filename}.csv").is_file()
    }
    print(
        f"[extract] wanted={len(wanted)} already_present={len(wanted) - len(remaining)} "
        f"to_extract={len(remaining)}"
    )
    if not remaining:
        return {"extracted": 0, "present": len(wanted)}
    if not archive.is_file():
        raise FileNotFoundError(f"BONES-SEED archive not found: {archive}")

    extracted = 0
    t0 = time.time()
    with tarfile.open(archive, mode="r:*") as tar:
        for member in tar:
            if not member.isfile():
                continue
            filename = remaining.get(member.name)
            if filename is None:
                continue
            target = raw_dir / f"{filename}.csv"
            source = tar.extractfile(member)
            if source is None:
                raise RuntimeError(f"Could not extract archive member: {member.name}")
            with source, target.open("wb") as handle:
                # Copy in chunks to bound memory.
                while True:
                    chunk = source.read(1 << 20)
                    if not chunk:
                        break
                    handle.write(chunk)
            extracted += 1
            if extracted % 5000 == 0:
                rate = extracted / max(time.time() - t0, 1e-6)
                print(f"[extract] {extracted}/{len(remaining)} ({rate:.0f}/s)")
            if extracted == len(remaining):
                # All wanted members found; stop scanning the rest of the archive.
                break
    missing = [
        filename
        for g1_path, filename in remaining.items()
        if not (raw_dir / f"{filename}.csv").is_file()
    ]
    if missing:
        raise RuntimeError(
            f"{len(missing)} selected CSV(s) were not found in the archive, e.g. "
            f"{missing[:5]}"
        )
    print(f"[extract] done: extracted={extracted} in {time.time() - t0:.1f}s")
    return {"extracted": extracted, "present": len(wanted)}


# ------------------------------ stage: normalize --------------------------

_PREPARE = None


def _normalize_one(args: tuple[str, str]) -> tuple[str, bool, str]:
    """Worker: normalize a raw BONES CSV to a converter CSV. Returns (name, ok, msg)."""
    global _PREPARE
    if _PREPARE is None:
        _PREPARE = _load_prepare_module()
    raw_path_str, staged_path_str = args
    raw_path = Path(raw_path_str)
    staged_path = Path(staged_path_str)
    try:
        if not _PREPARE._is_bones_g1_csv(raw_path):
            return (raw_path.name, False, "not a BONES G1 CSV (unexpected header)")
        _PREPARE._write_converter_csv(raw_path, staged_path)
        return (raw_path.name, True, "")
    except Exception as exc:  # noqa: BLE001 - report per-file, keep going
        return (raw_path.name, False, f"{type(exc).__name__}: {exc}")


def stage_normalize(
    selection: dict[str, Any],
    raw_dir: Path,
    csv_dir: Path,
    workers: int,
) -> dict[str, int]:
    csv_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[str, str]] = []
    for m in selection["motions"]:
        filename = m["filename"]
        raw_path = raw_dir / f"{filename}.csv"
        staged_path = _staged_csv_path(csv_dir, filename)
        if staged_path.is_file():
            continue
        if not raw_path.is_file():
            raise FileNotFoundError(f"Missing extracted CSV: {raw_path}")
        jobs.append((str(raw_path), str(staged_path)))

    print(
        f"[normalize] total={len(selection['motions'])} to_normalize={len(jobs)} "
        f"workers={workers}"
    )
    if not jobs:
        return {"normalized": 0, "failed": 0}

    normalized = 0
    failed = 0
    failures: list[str] = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_normalize_one, job) for job in jobs]
        for i, future in enumerate(as_completed(futures), start=1):
            name, ok, msg = future.result()
            if ok:
                normalized += 1
            else:
                failed += 1
                failures.append(f"{name}: {msg}")
            if i % 5000 == 0:
                rate = i / max(time.time() - t0, 1e-6)
                print(f"[normalize] {i}/{len(jobs)} ({rate:.0f}/s) failed={failed}")
    if failures:
        print(f"[normalize] {failed} failure(s), first 10:")
        for line in failures[:10]:
            print(f"  - {line}")
    print(
        f"[normalize] done: normalized={normalized} failed={failed} "
        f"in {time.time() - t0:.1f}s"
    )
    return {"normalized": normalized, "failed": failed}


# ------------------------------ stage: convert ----------------------------


def _output_frames_estimate(
    move_duration_frames: int | None,
    staged_csv: Path,
    input_fps: float,
    output_fps: float,
) -> int:
    if move_duration_frames and move_duration_frames > 0:
        src = int(move_duration_frames)
    else:
        with staged_csv.open("r", encoding="utf-8") as handle:
            src = sum(1 for _ in handle)
    # batch_csv_to_npz resamples with torch.arange(0, duration, output_dt); use a
    # conservative ceil so the memory budget is never under-estimated.
    return max(2, int(np.ceil(src * output_fps / input_fps)) + 1)


def _plan_buckets(
    items: list[tuple[str, int]],
    max_envs: int,
    cell_budget: int,
) -> list[list[str]]:
    """Length-bucket motions so each batch obeys len(batch)*max_frames <= budget.

    ``items`` is (filename, output_frames), sorted ascending by output_frames.
    """
    buckets: list[list[str]] = []
    current: list[str] = []
    current_max = 0
    for filename, frames in items:
        prospective_max = max(current_max, frames)
        if current and (
            len(current) + 1 > max_envs
            or (len(current) + 1) * prospective_max > cell_budget
        ):
            buckets.append(current)
            current = [filename]
            current_max = frames
        else:
            current.append(filename)
            current_max = prospective_max
    if current:
        buckets.append(current)
    return buckets


def stage_convert(
    selection: dict[str, Any],
    csv_dir: Path,
    npz_dir: Path,
    batch_dir: Path,
    input_fps: float,
    output_fps: float,
    max_envs: int,
    cell_budget: int,
    device: str,
    headless: bool,
    python_exe: str,
) -> dict[str, Any]:
    npz_dir.mkdir(parents=True, exist_ok=True)
    batch_dir.mkdir(parents=True, exist_ok=True)
    batch_converter = REPO_ROOT / "scripts" / "batch_csv_to_npz.py"

    dur_by_name = {
        m["filename"]: m.get("move_duration_frames") for m in selection["motions"]
    }
    # One scandir per directory instead of 2*130k per-file stat calls (much faster
    # on a large-directory spinning disk, and makes resume near-instant).
    existing_npz = {
        entry.name[:-4]
        for entry in os.scandir(npz_dir)
        if entry.name.endswith(".npz")
    }
    existing_staged = {
        entry.name[:-4]
        for entry in os.scandir(csv_dir)
        if entry.name.endswith(".csv")
    }
    pending: list[tuple[str, int]] = []
    done = 0
    for m in selection["motions"]:
        filename = m["filename"]
        if filename in existing_npz:
            done += 1
            continue
        if filename not in existing_staged:
            raise FileNotFoundError(
                f"Missing staged CSV (run normalize first): "
                f"{_staged_csv_path(csv_dir, filename)}"
            )
        frames = _output_frames_estimate(
            dur_by_name.get(filename),
            _staged_csv_path(csv_dir, filename),
            input_fps,
            output_fps,
        )
        pending.append((filename, frames))

    pending.sort(key=lambda x: x[1])
    buckets = _plan_buckets(pending, max_envs=max_envs, cell_budget=cell_budget)
    # Build the name -> frames map ONCE; rebuilding it inside the bucket loops is
    # O(n^2) over ~130k motions.
    frames_by_name = dict(pending)
    total_cells = (
        sum(len(b) * max(frames_by_name[f] for f in b) for b in buckets)
        if buckets
        else 0
    )
    print(
        f"[convert] total={len(selection['motions'])} already_done={done} "
        f"pending={len(pending)} buckets={len(buckets)} "
        f"est_cell_frames={total_cells:,}"
    )
    if not buckets:
        return {"buckets": 0, "converted": 0, "already_done": done}

    converted = 0
    t0 = time.time()
    for bi, bucket in enumerate(buckets):
        bmax = max(frames_by_name[f] for f in bucket)
        jobs = [
            {
                "input_file": str(_staged_csv_path(csv_dir, f)),
                "output_name": str(_npz_path(npz_dir, f)),
            }
            for f in bucket
        ]
        jobs_json = batch_dir / f"bucket_{bi:04d}.json"
        jobs_json.write_text(json.dumps(jobs), encoding="utf-8")
        cmd = [
            python_exe,
            str(batch_converter),
            "--jobs_json",
            str(jobs_json),
            "--input_fps",
            str(input_fps),
            "--output_fps",
            str(output_fps),
            "--device",
            device,
        ]
        if headless:
            cmd.append("--headless")
        print(
            f"[convert] bucket {bi + 1}/{len(buckets)}: envs={len(bucket)} "
            f"max_frames={bmax} cells={len(bucket) * bmax:,}"
        )
        print(f"[CMD] {' '.join(shlex.quote(c) for c in cmd)}")
        subprocess.run(cmd, check=True)
        missing = [f for f in bucket if not _npz_path(npz_dir, f).is_file()]
        if missing:
            raise RuntimeError(
                f"bucket {bi} produced {len(missing)} missing NPZ(s), e.g. {missing[:5]}"
            )
        converted += len(bucket)
        rate = converted / max(time.time() - t0, 1e-6)
        print(
            f"[convert] bucket {bi + 1} ok | converted={converted}/{len(pending)} "
            f"({rate:.1f} motions/s)"
        )
    print(f"[convert] done: converted={converted} in {time.time() - t0:.1f}s")
    return {"buckets": len(buckets), "converted": converted, "already_done": done}


# ------------------------------ stage: anchor -----------------------------


def stage_anchor(npz_dir: Path, tol: float, python_exe: str) -> None:
    anchor_script = REPO_ROOT / "scripts" / "anchor_npz_local_frame.py"
    cmd = [
        python_exe,
        str(anchor_script),
        "--npz_dir",
        str(npz_dir),
        "--tol",
        str(tol),
    ]
    print(f"[anchor] {' '.join(shlex.quote(c) for c in cmd)}")
    subprocess.run(cmd, check=True)


# ------------------------------ stage: audit ------------------------------


G1_ENV_CFG = (
    REPO_ROOT
    / "source"
    / "isaaclab_imitation"
    / "isaaclab_imitation"
    / "tasks"
    / "manager_based"
    / "imitation"
    / "config"
    / "g1"
    / "imitation_g1_env_cfg.py"
)


def _canonical_dataset_body_names() -> list[str] | None:
    """Parse ``G1_29DOF_DATASET_BODY_NAMES`` from the env cfg without importing Isaac.

    This list is the authoritative reference body set the training env binds to
    by name (30 bodies; the fixed rubber-hand links are intentionally excluded).
    """
    import ast

    try:
        tree = ast.parse(G1_ENV_CFG.read_text(encoding="utf-8"))
    except OSError:
        return None
    for node in tree.body:
        # The constant is an annotated assignment: ``NAME: list[str] = [...]``.
        if isinstance(node, ast.AnnAssign):
            target = node.target
            value = node.value
        elif isinstance(node, ast.Assign):
            target = node.targets[0] if len(node.targets) == 1 else None
            value = node.value
        else:
            continue
        if (
            isinstance(target, ast.Name)
            and target.id == "G1_29DOF_DATASET_BODY_NAMES"
            and value is not None
        ):
            return [str(x) for x in ast.literal_eval(value)]
    return None


def stage_audit(
    selection: dict[str, Any],
    npz_dir: Path,
    sample: int,
) -> dict[str, Any]:
    canonical_bodies = _canonical_dataset_body_names()
    if canonical_bodies is None:
        print("[audit] WARNING: could not parse G1_29DOF_DATASET_BODY_NAMES from cfg")
    else:
        print(
            f"[audit] canonical G1_29DOF_DATASET_BODY_NAMES: {len(canonical_bodies)} bodies"
        )

    names = [m["filename"] for m in selection["motions"]]
    if sample > 0 and sample < len(names):
        step = max(1, len(names) // sample)
        names = names[::step][:sample]
    print(f"[audit] auditing {len(names)} NPZ(s)")

    # Joint names have no separate literal; enforce self-consistency across the
    # tree (all NPZ must agree) plus a 29-count check.
    ref_joint_names: list[str] | None = None
    ref_body_names: list[str] | None = None
    problems: list[str] = []
    checked = 0
    for filename in names:
        path = _npz_path(npz_dir, filename)
        if not path.is_file():
            problems.append(f"{filename}: NPZ missing")
            continue
        with np.load(path, allow_pickle=False) as data:
            keys = set(data.files)
            if not EXPECTED_NPZ_KEYS.issubset(keys):
                problems.append(
                    f"{filename}: missing keys {sorted(EXPECTED_NPZ_KEYS - keys)}"
                )
                continue
            joint_names = [str(x) for x in data["joint_names"].tolist()]
            body_names = [str(x) for x in data["body_names"].tolist()]
            root_pos = data["root_pos"]
            if ref_joint_names is None:
                ref_joint_names = joint_names
            if ref_body_names is None:
                ref_body_names = body_names
            if len(joint_names) != 29:
                problems.append(f"{filename}: {len(joint_names)} joints (expected 29)")
            if joint_names != ref_joint_names:
                problems.append(f"{filename}: joint_names differ across tree")
            if body_names != ref_body_names:
                problems.append(f"{filename}: body_names differ across tree")
            if canonical_bodies is not None and body_names != canonical_bodies:
                problems.append(
                    f"{filename}: body_names != canonical "
                    f"G1_29DOF_DATASET_BODY_NAMES (got {len(body_names)} bodies)"
                )
            xy0 = np.abs(root_pos[0, :2])
            if float(xy0.max()) > 1e-4:
                problems.append(
                    f"{filename}: root0 XY not anchored (max |xy|={float(xy0.max()):.2e})"
                )
        checked += 1
    print(f"[audit] checked={checked} problems={len(problems)}")
    for line in problems[:20]:
        print(f"  - {line}")
    return {
        "checked": checked,
        "problems": len(problems),
        "problem_examples": problems[:20],
        "canonical_body_match": canonical_bodies is not None
        and ref_body_names == canonical_bodies,
        "joint_names": ref_joint_names or [],
        "body_names": ref_body_names or [],
    }


# ------------------------------ stage: manifest ---------------------------


def _sanitize_motion_name(filename: str) -> str:
    import re

    name = re.sub(r"[^A-Za-z0-9_\-/]+", "_", filename)
    name = name.replace("/", "__").replace("-", "_")
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "motion"


def stage_manifest(
    selection: dict[str, Any],
    npz_dir: Path,
    manifest_path: Path,
    language_path: Path,
    input_fps: float,
    output_fps: float,
) -> dict[str, int]:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    language_path.parent.mkdir(parents=True, exist_ok=True)

    trajectories: list[dict[str, Any]] = []
    language_motions: list[dict[str, Any]] = []
    missing = 0
    for m in selection["motions"]:
        filename = m["filename"]
        npz_file = _npz_path(npz_dir, filename)
        if not npz_file.is_file():
            missing += 1
            continue
        motion_name = _sanitize_motion_name(filename)
        trajectories.append(
            {
                "name": motion_name,
                "path": os.path.relpath(npz_file, manifest_path.parent),
                "input_fps": float(output_fps),
            }
        )
        language_motions.append(
            {
                "name": motion_name,
                "bones_seed_filename": filename,
                "category": m.get("category"),
                "is_mirror": m.get("is_mirror"),
                "language_goal": m.get("language_goal") or "",
                "move_duration_frames": m.get("move_duration_frames"),
            }
        )

    manifest = {
        "dataset_name": "bones_seed_sonic_full",
        "dataset": {"trajectories": {"lafan1_csv": trajectories}},
        "metadata": {
            "source_dataset": "bones-studio/seed",
            "selection": "sonic_keyword_exclusion",
            "selection_sha256": selection.get("selection_sha256"),
            "num_motions": len(trajectories),
            "num_missing_npz": missing,
            "input_fps": float(input_fps),
            "output_fps": float(output_fps),
            "language_annotations_path": os.path.relpath(
                language_path, manifest_path.parent
            ),
            "paths_are_relative_to_manifest": True,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    language = {
        "dataset_name": "bones_seed_sonic_full",
        "source": "bones-studio/seed",
        "manifest": str(manifest_path),
        "input_fps": float(input_fps),
        "output_fps": float(output_fps),
        "motions": language_motions,
    }
    language_path.write_text(json.dumps(language, indent=2) + "\n", encoding="utf-8")

    print(
        f"[manifest] wrote {len(trajectories)} trajectories "
        f"(missing_npz={missing}) -> {manifest_path}"
    )
    return {"trajectories": len(trajectories), "missing_npz": missing}


# ------------------------------ main --------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--work_root", type=Path, required=True)
    parser.add_argument(
        "--archive",
        type=Path,
        default=REPO_ROOT / "data" / "bones_seed" / "raw" / "g1.tar.gz",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="all",
        help=f"Comma list of stages to run, or 'all'. Choices: {','.join(ALL_STAGES)}",
    )
    parser.add_argument("--input_fps", type=float, default=120.0)
    parser.add_argument("--output_fps", type=float, default=50.0)
    parser.add_argument("--max_envs", type=int, default=8192)
    parser.add_argument(
        "--cell_budget",
        type=int,
        default=4_000_000,
        help="Max envs*max_frames per Isaac batch (bounds host/GPU memory).",
    )
    parser.add_argument(
        "--normalize_workers", type=int, default=max(1, (os.cpu_count() or 8) - 2)
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--anchor_tol", type=float, default=1e-5)
    parser.add_argument(
        "--audit_sample",
        type=int,
        default=2000,
        help="Audit this many NPZs (evenly sampled). 0 audits all.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Debug: only process the first N selected motions (0 = all).",
    )
    parser.add_argument("--python", type=str, default=sys.executable)
    args = parser.parse_args()

    stages = (
        ALL_STAGES
        if args.stages == "all"
        else tuple(s.strip() for s in args.stages.split(",") if s.strip())
    )
    for s in stages:
        if s not in ALL_STAGES:
            parser.error(f"unknown stage: {s}")

    work_root = args.work_root.expanduser().resolve()
    raw_dir = work_root / "raw_csv"
    csv_dir = work_root / "staged_csv"
    npz_dir = work_root / "npz" / "g1"
    batch_dir = work_root / "batch_jobs"
    record_dir = work_root / "records"
    record_dir.mkdir(parents=True, exist_ok=True)

    selection = _load_selection(args.selection.expanduser().resolve())
    if args.limit > 0:
        selection = {**selection, "motions": selection["motions"][: args.limit]}
        print(f"[main] LIMIT active: processing {len(selection['motions'])} motions")

    print(
        f"[main] motions={len(selection['motions'])} stages={stages} work_root={work_root}"
    )
    record: dict[str, Any] = {
        "selection": str(args.selection),
        "selection_sha256": selection.get("selection_sha256"),
        "num_motions": len(selection["motions"]),
        "stages": list(stages),
        "input_fps": args.input_fps,
        "output_fps": args.output_fps,
        "max_envs": args.max_envs,
        "cell_budget": args.cell_budget,
    }

    if "extract" in stages:
        record["extract"] = stage_extract(
            selection, args.archive.expanduser().resolve(), raw_dir
        )
    if "normalize" in stages:
        record["normalize"] = stage_normalize(
            selection, raw_dir, csv_dir, args.normalize_workers
        )
    if "convert" in stages:
        record["convert"] = stage_convert(
            selection,
            csv_dir,
            npz_dir,
            batch_dir,
            args.input_fps,
            args.output_fps,
            args.max_envs,
            args.cell_budget,
            args.device,
            args.headless,
            args.python,
        )
    if "anchor" in stages:
        stage_anchor(npz_dir, args.anchor_tol, args.python)
        record["anchor"] = {"tol": args.anchor_tol}
    if "audit" in stages:
        record["audit"] = stage_audit(selection, npz_dir, args.audit_sample)
    if "manifest" in stages:
        manifest_path = (
            work_root / "manifests" / "g1_bones_seed_sonic_full_manifest.json"
        )
        language_path = (
            work_root / "language" / "g1_bones_seed_sonic_full_language.json"
        )
        record["manifest"] = stage_manifest(
            selection,
            npz_dir,
            manifest_path,
            language_path,
            args.input_fps,
            args.output_fps,
        )

    record_path = record_dir / "convert_record.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"[main] wrote record: {record_path}")

    audit = record.get("audit")
    if audit and audit["problems"] > 0:
        print(f"[main] AUDIT FAILED with {audit['problems']} problem(s).")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
