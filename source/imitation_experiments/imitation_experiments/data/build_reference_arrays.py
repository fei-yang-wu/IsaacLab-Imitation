#!/usr/bin/env python3
"""Build training-shaped reference arrays straight from an NPZ motion tree.

The generic path to reference data is NPZ -> Zarr -> a wide TorchRL replay ->
two derived caches rebuilt inside every training process. Each layer after the
NPZ serves the generic dataset API, and training pays for all of it. On the
129,785-motion BONES-SEED set:

* The Zarr turns 129,785 files into roughly 5.3 million (about 41 per
  trajectory, including eight unused ``next_*`` duplicates), so the one-time
  fill is IOPS-bound rather than bandwidth-bound.
* The replay stores 30 bodies at full width, so the macro and runtime caches
  read about 133 GB to keep about 55 GB, and ``body_pos_w`` plus ``body_quat_w``
  are read twice -- once by each.

This module writes the caches' contents directly, once, from the NPZ files.
Those are ``STORED`` zip members, so each array is a contiguous byte range that
a worker reads with no decompression, and the whole build is one file open per
trajectory. Trajectory offsets are known up front, so workers write disjoint row
ranges with no cursor, no locking, and no coordination.

Nothing here is specific to one robot or one dataset. The joint count, body
count, ``qpos``/``qvel`` widths, body names, and joint names all come from the
data; which bodies to retain and which body to anchor on are arguments; and a
source that lacks, say, body velocities simply yields a smaller array set that
consumers check for by name. It works for any NPZ tree this repo's CSV
converters produce -- LAFAN1, Dance102, BONES-SEED, or a new one.

Quaternion conventions follow the consumers exactly, and they differ:

* ``body_quat_w`` keeps the dataset's WXYZ order, because the runtime cache
  stores raw source values and the environment swizzles at sample time.
* ``anchor_quat_w`` is pre-swizzled to XYZW, because that is how the
  ``root_qpos`` macro cache holds it.

``joint_pos``/``joint_vel`` are not written: they are ``qpos[:, 7:]`` and
``qvel[:, 6:]``, and aliasing costs a 1.24x read amplification instead of 11 GB
of duplication on the 129k set.

Equivalence with the Zarr path was measured, not assumed. Against the packed
129,785-motion replay, over 360 rows sampled from 60 random trajectories, root
position, all 29 joint positions, all 35 ``qvel`` components, and all 30 bodies'
positions and quaternions are bit-identical. The only difference is the root
quaternion -- ``qpos[3:7]`` and the ``root_quat`` field -- at most 1.79e-07,
about one to three float32 ULP at unit magnitude, because the Zarr export
re-normalized it before rounding. Cross-checks against a replay must allow that
tolerance on those four components; checks against the NPZs, which is what
``verify_against_source`` does, are exact.

This lives in the shared package rather than in a campaign directory, and it
parses the NPZ manifest itself rather than importing
``isaaclab_imitation.tasks.manager_based.imitation.motion_manifest``, because
that package is only installed in the ``isaaclab`` Pixi environment and this
builder and its test must run in the default one.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import dataclasses
import hashlib
import io
import json
import logging
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Literal
import zipfile

import numpy as np


logger = logging.getLogger(__name__)

SIDECAR_NAME = "reference_arrays_manifest.json"
FORMAT_VERSION = 1

#: Ordered bodies the G1 v2 interface tracks, mirroring
#: ``G1_TRACKED_BODY_NAMES`` in ``config/g1/common/constants.py``. Exported as a
#: convenience for launchers, never used as a default: the default keeps every
#: body the dataset has, which is lossless and robot-agnostic.
G1_TRACKED_BODY_NAMES: tuple[str, ...] = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)

#: WXYZ -> XYZW, matching ``_WXYZ_TO_XYZW`` in ``expert_data_plane.py``.
WXYZ_TO_XYZW = [1, 2, 3, 0]

#: Per-frame members with a single trailing width.
FLAT_MEMBERS = ("qpos", "qvel")

#: Per-frame members shaped (frames, num_bodies, width).
BODY_MEMBERS = ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")

#: Only ``qpos`` is structural. ``root_*`` and ``joint_*`` are deliberately
#: never written: each is a slice of ``qpos``/``qvel`` or of ``body_*``.
REQUIRED_MEMBERS = ("qpos",)

#: Anchor array -> the body member it is extracted from.
ANCHOR_SOURCES = {"anchor_pos_w": "body_pos_w", "anchor_quat_w": "body_quat_w"}

DTYPE = np.float32


# --------------------------------------------------------------------------- #
# Manifest and trajectory-order sources
# --------------------------------------------------------------------------- #


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Could not read JSON at {path}: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object at {path}.")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_npz_paths(manifest: Path, *, npz_root: Path | None = None) -> dict[str, Path]:
    """Map motion name -> NPZ path from a clip manifest.

    Mirrors the entry shape ``load_clip_manifest`` accepts: entries live under
    ``dataset.trajectories.lafan1_csv`` with ``lafan1_csv``, ``motions``, and a
    bare list as fallbacks. ``npz_root`` replaces each entry's directory so a
    manifest can be reused against a relocated copy of the tree without being
    rewritten.
    """
    payload = _read_json(manifest)
    entries: Any = None
    dataset = payload.get("dataset")
    if isinstance(dataset, dict):
        trajectories = dataset.get("trajectories")
        if isinstance(trajectories, dict):
            entries = trajectories.get("lafan1_csv")
    for fallback in ("lafan1_csv", "motions"):
        if entries is None:
            entries = payload.get(fallback)
    if not isinstance(entries, list) or not entries:
        raise RuntimeError(f"No clip entries found in {manifest}.")

    metadata = payload.get("metadata")
    relative = True
    if isinstance(metadata, dict):
        relative = bool(metadata.get("paths_are_relative_to_manifest", True))
    base = manifest.parent

    paths: dict[str, Path] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise RuntimeError(f"Malformed clip entry in {manifest}: {entry!r}")
        raw = entry.get("path") or entry.get("file")
        if raw is None:
            raise RuntimeError(f"Clip entry without a path in {manifest}: {entry!r}")
        path = Path(str(raw)).expanduser()
        if not path.is_absolute() and relative:
            path = base / path
        path = path.resolve()
        if npz_root is not None:
            path = (npz_root / path.name).resolve()
        name = str(entry.get("name") or path.stem)
        if name in paths:
            raise RuntimeError(f"Duplicate motion name {name!r} in {manifest}.")
        paths[name] = path
    return paths


def npz_members(path: Path) -> set[str]:
    """Array names an NPZ holds, from its zip directory alone."""
    with zipfile.ZipFile(path) as archive:
        return {
            name[: -len(".npy")] for name in archive.namelist() if name.endswith(".npy")
        }


def npz_member_shape(path: Path, member: str) -> tuple[int, ...]:
    """Shape of one NPZ member, read from its header without loading data."""
    with zipfile.ZipFile(path) as archive:
        with archive.open(f"{member}.npy") as handle:
            version = np.lib.format.read_magic(handle)
            if version == (1, 0):
                shape, _fortran, _dtype = np.lib.format.read_array_header_1_0(handle)
            elif version == (2, 0):
                shape, _fortran, _dtype = np.lib.format.read_array_header_2_0(handle)
            else:  # pragma: no cover - numpy has only shipped 1.0 and 2.0
                raise RuntimeError(f"{path}:{member} has .npy version {version}.")
    if not shape:
        raise RuntimeError(f"{path}:{member} is a scalar; expected a time axis.")
    return tuple(int(value) for value in shape)


def trajectory_plan_from_traj_info(
    traj_info_path: Path, npz_paths: dict[str, Path]
) -> list[dict[str, Any]]:
    """Canonical order and offsets from an existing ILTools replay sidecar.

    Using the persisted sidecar keeps trajectory ranks byte-compatible with the
    replay that earlier runs indexed, which matters because planner goal indices
    are trajectory ranks.
    """
    payload = _read_json(traj_info_path)
    traj_info = payload.get("traj_info")
    if not isinstance(traj_info, dict):
        raise RuntimeError(f"{traj_info_path} has no traj_info object.")
    ordered = traj_info.get("ordered_traj_list")
    starts = traj_info.get("start_index")
    ends = traj_info.get("end_index")
    if (
        not isinstance(ordered, list)
        or not isinstance(starts, list)
        or not isinstance(ends, list)
    ):
        raise RuntimeError(
            f"{traj_info_path} is missing ordered_traj_list/start_index/end_index."
        )
    if not len(ordered) == len(starts) == len(ends):
        raise RuntimeError(
            f"{traj_info_path} span lengths disagree: ordered={len(ordered)}, "
            f"start={len(starts)}, end={len(ends)}."
        )

    plan: list[dict[str, Any]] = []
    seen: set[str] = set()
    spans: list[tuple[Any, Any, Any]] = list(zip(ordered, starts, ends))
    for rank, (entry, start, end) in enumerate(spans):
        if not isinstance(entry, (list, tuple)) or len(entry) != 3:
            raise RuntimeError(
                f"Malformed ordered_traj_list entry at rank {rank}: {entry!r}"
            )
        dataset, motion, trajectory = (str(part) for part in entry)
        if motion in seen:
            # One NPZ per clip is this repo's norm. A motion with several takes
            # cannot be filled from a single file, and writing the same frames
            # into two spans would look plausible and be wrong.
            raise RuntimeError(
                f"Motion {motion!r} appears more than once in {traj_info_path.name}. "
                "This builder maps one NPZ per motion and cannot fill multi-take "
                "motions."
            )
        seen.add(motion)
        npz_path = npz_paths.get(motion)
        if npz_path is None:
            raise RuntimeError(
                f"Motion {motion!r} at rank {rank} is in {traj_info_path.name} but "
                "not in the NPZ manifest."
            )
        plan.append(
            {
                "rank": rank,
                "dataset": dataset,
                "motion": motion,
                "trajectory": trajectory,
                "path": str(npz_path),
                "start": int(start),
                "end": int(end),
            }
        )
    return plan


def trajectory_plan_from_manifest(
    npz_paths: dict[str, Path], *, dataset_name: str = "lafan1"
) -> list[dict[str, Any]]:
    """Manifest-order plan, reading only NPZ headers to get lengths.

    A trajectory contributes ``T - 1`` rows: the last frame is dropped, matching
    ``make_rb_from``'s treatment of non-transition-aligned keys.
    """
    plan: list[dict[str, Any]] = []
    cursor = 0
    for rank, (motion, path) in enumerate(npz_paths.items()):
        rows = npz_member_shape(path, "qpos")[0] - 1
        if rows <= 0:
            raise RuntimeError(f"{path} has too few frames to yield a transition.")
        plan.append(
            {
                "rank": rank,
                "dataset": dataset_name,
                "motion": motion,
                "trajectory": "trajectory_0",
                "path": str(path),
                "start": cursor,
                "end": cursor + rows,
            }
        )
        cursor += rows
    return plan


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class BuildLayout:
    """Everything a worker needs to open the outputs and place its rows.

    Widths and member names are discovered from the data, never assumed, so one
    layout describes any NPZ tree this repo's converters produce.
    """

    output_dir: str
    total_rows: int
    members: tuple[str, ...]
    widths: dict[str, int]
    body_names: tuple[str, ...]
    joint_names: tuple[str, ...]
    anchor_body: str | None

    @property
    def body_members(self) -> tuple[str, ...]:
        return tuple(name for name in self.members if name in BODY_MEMBERS)

    @property
    def anchor_arrays(self) -> tuple[str, ...]:
        if self.anchor_body is None:
            return ()
        return tuple(
            name for name, source in ANCHOR_SOURCES.items() if source in self.members
        )

    @property
    def array_names(self) -> tuple[str, ...]:
        return self.members + self.anchor_arrays

    def array_shape(self, name: str) -> tuple[int, ...]:
        if name in ANCHOR_SOURCES:
            return (self.total_rows, self.widths[ANCHOR_SOURCES[name]])
        if name in BODY_MEMBERS:
            return (self.total_rows, len(self.body_names), self.widths[name])
        return (self.total_rows, self.widths[name])

    def array_path(self, name: str) -> Path:
        return Path(self.output_dir) / f"{name}.memmap"


def discover_layout_members(path: Path) -> tuple[tuple[str, ...], dict[str, int]]:
    """Which known members an NPZ carries, and their trailing widths."""
    present = npz_members(path)
    missing = [name for name in REQUIRED_MEMBERS if name not in present]
    if missing:
        raise RuntimeError(f"{path} lacks required members {missing}.")
    members: list[str] = []
    widths: dict[str, int] = {}
    for name in (*FLAT_MEMBERS, *BODY_MEMBERS):
        if name not in present:
            continue
        members.append(name)
        widths[name] = npz_member_shape(path, name)[-1]
    return tuple(members), widths


def _open_outputs(
    layout: BuildLayout, mode: Literal["r", "r+", "w+"]
) -> dict[str, np.memmap]:
    return {
        name: np.memmap(
            layout.array_path(name),
            dtype=DTYPE,
            mode=mode,
            shape=layout.array_shape(name),
        )
        for name in layout.array_names
    }


def _write_trajectories(
    layout: BuildLayout, entries: list[dict[str, Any]]
) -> tuple[int, int]:
    """Write one contiguous slice of the plan. Returns (rows, bytes read)."""
    outputs = _open_outputs(layout, "r+")
    reference_bodies = list(layout.body_names)
    body_members = layout.body_members
    anchor_arrays = layout.anchor_arrays
    rows_written = 0
    bytes_read = 0

    for entry in entries:
        path = Path(entry["path"])
        start = int(entry["start"])
        end = int(entry["end"])
        rows = end - start
        # One sequential read of the whole file, rather than letting np.load
        # seek to each member in turn. Measured on the 129,785-clip tree from a
        # 7200-rpm disk: 107 files/s versus 91 member-wise, both single-worker.
        # The largest clip is about 46 MB, so a worker's peak is bounded.
        with np.load(io.BytesIO(path.read_bytes())) as npz:
            if body_members or anchor_arrays:
                names = [str(value) for value in npz["body_names"]]
                missing = [name for name in reference_bodies if name not in names]
                if missing:
                    raise RuntimeError(
                        f"{path} lacks bodies {missing}; its body_names are {names}."
                    )
                if layout.anchor_body is not None and layout.anchor_body not in names:
                    raise RuntimeError(
                        f"{path} lacks the anchor body {layout.anchor_body!r}."
                    )
                # Look bodies up by name per file: a tree whose members list
                # bodies in different orders still lands in the right columns.
                body_ids = np.array(
                    [names.index(name) for name in reference_bodies], dtype=np.intp
                )
                anchor_id = (
                    names.index(layout.anchor_body)
                    if layout.anchor_body is not None
                    else -1
                )
            else:
                body_ids = np.empty(0, dtype=np.intp)
                anchor_id = -1

            if layout.joint_names:
                # Joint order is a column position in qpos/qvel, so a
                # heterogeneous tree must fail here rather than silently permute
                # one trajectory's columns against the rest.
                joints = tuple(str(value) for value in npz["joint_names"])
                if joints != layout.joint_names:
                    raise RuntimeError(
                        f"{path} declares joint order {list(joints)}, which differs "
                        "from the first trajectory's."
                    )

            source = {member: npz[member] for member in layout.members}

        available = int(source["qpos"].shape[0]) - 1
        if available != rows:
            raise RuntimeError(
                f"{path} yields {available} transitions but the plan reserves "
                f"{rows} rows at [{start}, {end})."
            )
        bytes_read += sum(int(array.nbytes) for array in source.values())

        for member in layout.members:
            if member in BODY_MEMBERS:
                outputs[member][start:end] = source[member][:rows, body_ids]
            else:
                outputs[member][start:end] = source[member][:rows]
        for name in anchor_arrays:
            column = source[ANCHOR_SOURCES[name]][:rows, anchor_id]
            if name == "anchor_quat_w":
                # The macro cache stores the anchor rotation as XYZW; the
                # body_quat_w array above deliberately keeps the source's WXYZ.
                column = column[:, WXYZ_TO_XYZW]
            outputs[name][start:end] = column
        rows_written += rows

    for array in outputs.values():
        array.flush()
    return rows_written, bytes_read


def _worker(payload: tuple[BuildLayout, list[dict[str, Any]]]) -> tuple[int, int]:
    layout, entries = payload
    return _write_trajectories(layout, entries)


def _contiguous_shards(
    entries: list[dict[str, Any]], workers: int
) -> list[list[dict[str, Any]]]:
    """Split into contiguous, roughly row-balanced slices.

    Contiguous rather than round-robin: on a spinning source disk that keeps
    each worker's reads local, and on any device it keeps each worker's writes
    to one sequential span of every output array.
    """
    if workers <= 1 or len(entries) <= 1:
        return [entries]
    total_rows = sum(entry["end"] - entry["start"] for entry in entries)
    target = total_rows / workers
    shards: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    accumulated = 0
    for entry in entries:
        current.append(entry)
        accumulated += entry["end"] - entry["start"]
        if accumulated >= target * (len(shards) + 1) and len(shards) < workers - 1:
            shards.append(current)
            current = []
    if current:
        shards.append(current)
    return shards


def build_reference_arrays(
    *,
    manifest: Path,
    output_dir: Path,
    persist_id: str,
    body_names: list[str] | None = None,
    anchor_body: str | None = None,
    traj_info: Path | None = None,
    npz_root: Path | None = None,
    dataset_name: str = "lafan1",
    workers: int = 6,
    expected_motions: int | None = None,
    expected_transitions: int | None = None,
) -> dict[str, Any]:
    """Materialize the reference arrays and write the sidecar last.

    ``body_names=None`` keeps every body the dataset carries, which is lossless
    and the right default for an unfamiliar tree. Naming a subset is what makes
    the artifact small: on the 129,785-motion G1 set, the 14 tracked bodies turn
    a 95 GB artifact into 49 GB.
    """
    npz_paths = load_npz_paths(manifest, npz_root=npz_root)
    plan = (
        trajectory_plan_from_traj_info(traj_info, npz_paths)
        if traj_info is not None
        else trajectory_plan_from_manifest(npz_paths, dataset_name=dataset_name)
    )
    if expected_motions is not None and len(plan) != expected_motions:
        raise RuntimeError(
            f"Plan holds {len(plan)} trajectories, expected {expected_motions}."
        )
    total_rows = plan[-1]["end"]
    if expected_transitions is not None and total_rows != expected_transitions:
        raise RuntimeError(
            f"Plan holds {total_rows} transitions, expected {expected_transitions}."
        )
    for previous, entry in zip(plan, plan[1:]):
        if entry["start"] != previous["end"]:
            raise RuntimeError(
                f"Plan has a gap or overlap at rank {entry['rank']}: "
                f"{previous['end']} -> {entry['start']}."
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = output_dir / SIDECAR_NAME
    if sidecar_path.exists():
        raise RuntimeError(
            f"Refusing to rebuild over a complete cache at {output_dir}. Choose a "
            "fresh, versioned directory; an interrupted build must be quarantined, "
            "not resumed."
        )

    first_path = Path(plan[0]["path"])
    members, widths = discover_layout_members(first_path)
    with np.load(first_path) as first:
        dataset_body_names = (
            tuple(str(value) for value in first["body_names"])
            if "body_names" in first
            else ()
        )
        joint_names = (
            tuple(str(value) for value in first["joint_names"])
            if "joint_names" in first
            else ()
        )
    if body_names is None:
        body_names = list(dataset_body_names)
    if anchor_body is not None and anchor_body not in dataset_body_names:
        raise RuntimeError(
            f"Anchor body {anchor_body!r} is not in the dataset's bodies "
            f"{list(dataset_body_names)}."
        )
    if anchor_body is not None and not (set(ANCHOR_SOURCES.values()) <= set(members)):
        raise RuntimeError(
            f"--anchor_body needs {sorted(ANCHOR_SOURCES.values())}, but the source "
            f"carries only {list(members)}."
        )

    layout = BuildLayout(
        output_dir=str(output_dir),
        total_rows=int(total_rows),
        members=members,
        widths=widths,
        body_names=tuple(body_names),
        joint_names=joint_names,
        anchor_body=anchor_body,
    )

    required = sum(
        int(np.prod(layout.array_shape(name))) * np.dtype(DTYPE).itemsize
        for name in layout.array_names
    )
    free = shutil.disk_usage(output_dir).free
    if required > free:
        raise RuntimeError(
            f"{output_dir} has {free / 1e9:.1f} GB free but the arrays need "
            f"{required / 1e9:.1f} GB. Preallocation is sparse, so building anyway "
            "would fail partway through and leave a directory to quarantine."
        )
    logger.info(
        "Allocating %.1f GB across %s arrays in %s (%.1f GB free): %s.",
        required / 1e9,
        len(layout.array_names),
        output_dir,
        free / 1e9,
        ", ".join(layout.array_names),
    )

    # Preallocate before fanning out, so every worker maps an existing file.
    for array in _open_outputs(layout, "w+").values():
        del array

    shards = _contiguous_shards(plan, workers)
    started = time.perf_counter()
    rows_written = 0
    bytes_read = 0
    if len(shards) == 1:
        rows_written, bytes_read = _write_trajectories(layout, shards[0])
    else:
        with ProcessPoolExecutor(max_workers=len(shards)) as pool:
            for rows, read in pool.map(_worker, [(layout, shard) for shard in shards]):
                rows_written += rows
                bytes_read += read
    elapsed = time.perf_counter() - started

    if rows_written != total_rows:
        raise RuntimeError(
            f"Workers wrote {rows_written} rows but the plan reserves {total_rows}."
        )
    logger.info(
        "Wrote %s rows across %s trajectories in %.1f s (%.2f GB read, %.0f MB/s).",
        f"{rows_written:,}",
        f"{len(plan):,}",
        elapsed,
        bytes_read / 1e9,
        bytes_read / 1e6 / max(elapsed, 1e-9),
    )

    sidecar = {
        "format_version": FORMAT_VERSION,
        "key": {
            "source": {"persist_id": persist_id},
            "manifest": str(manifest.resolve()),
            "manifest_sha256": sha256_file(manifest),
            "body_names": list(body_names),
            "anchor_body": anchor_body,
            # Carried so a consumer never has to open the Zarr for metadata: the
            # dataset-level zarr.json is 145 MB of JSON on the 129k set.
            "joint_names": list(joint_names),
            "dataset_body_names": list(dataset_body_names),
            "dtype": np.dtype(DTYPE).name,
            "arrays": {
                name: {
                    "shape": list(layout.array_shape(name)),
                    "dtype": np.dtype(DTYPE).name,
                    "quaternion_order": (
                        "xyzw"
                        if name == "anchor_quat_w"
                        else ("wxyz" if name.endswith("quat_w") else None)
                    ),
                }
                for name in layout.array_names
            },
        },
        "traj_info": {
            "capacity": int(total_rows),
            "written": int(total_rows),
            "start_index": [entry["start"] for entry in plan],
            "end_index": [entry["end"] for entry in plan],
            "ordered_traj_list": [
                [entry["dataset"], entry["motion"], entry["trajectory"]]
                for entry in plan
            ],
        },
    }
    tmp_path = sidecar_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(sidecar), encoding="utf-8")
    tmp_path.replace(sidecar_path)
    logger.info("Wrote %s.", sidecar_path)
    return sidecar


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def validation_errors(
    output_dir: Path,
    *,
    persist_id: str,
    body_names: list[str] | None = None,
    anchor_body: str | None = None,
    expected_motions: int | None = None,
    expected_transitions: int | None = None,
) -> list[str]:
    """Sidecar and on-disk-size checks, without reading any array contents."""
    sidecar_path = output_dir / SIDECAR_NAME
    if not sidecar_path.is_file():
        return [f"missing {sidecar_path}"]

    sidecar = _read_json(sidecar_path)
    errors: list[str] = []
    if sidecar.get("format_version") != FORMAT_VERSION:
        errors.append(
            f"format_version={sidecar.get('format_version')!r}, "
            f"expected {FORMAT_VERSION}"
        )
    key = sidecar.get("key")
    if not isinstance(key, dict):
        return errors + ["key is missing or is not an object"]
    if key.get("source") != {"persist_id": persist_id}:
        errors.append(
            f"source={key.get('source')!r}, expected persist_id={persist_id!r}"
        )
    if body_names is not None and key.get("body_names") != list(body_names):
        errors.append(
            f"body_names={key.get('body_names')!r}, expected {list(body_names)!r}"
        )
    if anchor_body is not None and key.get("anchor_body") != anchor_body:
        errors.append(
            f"anchor_body={key.get('anchor_body')!r}, expected {anchor_body!r}"
        )

    traj_info = sidecar.get("traj_info")
    if not isinstance(traj_info, dict):
        return errors + ["traj_info is missing or is not an object"]
    ordered = traj_info.get("ordered_traj_list")
    motions = len(ordered) if isinstance(ordered, list) else -1
    if expected_motions is not None and motions != expected_motions:
        errors.append(f"motion_count={motions}, expected {expected_motions}")
    written = traj_info.get("written")
    if expected_transitions is not None and written != expected_transitions:
        errors.append(f"written={written!r}, expected {expected_transitions}")
    ends = traj_info.get("end_index")
    if isinstance(ends, list) and ends and ends[-1] != written:
        errors.append(f"end_index[-1]={ends[-1]!r} disagrees with written={written!r}")

    arrays = key.get("arrays")
    if not isinstance(arrays, dict):
        return errors + ["key.arrays is missing or is not an object"]
    for name, spec in arrays.items():
        path = output_dir / f"{name}.memmap"
        if not path.is_file():
            errors.append(f"missing {path}")
            continue
        shape = tuple(int(value) for value in spec["shape"])
        expected_bytes = int(np.prod(shape)) * np.dtype(spec["dtype"]).itemsize
        actual_bytes = path.stat().st_size
        if actual_bytes != expected_bytes:
            errors.append(
                f"{path.name} is {actual_bytes} bytes, expected {expected_bytes} for "
                f"shape {shape}"
            )
    return errors


def open_reference_arrays(
    output_dir: Path,
) -> tuple[dict[str, np.memmap], dict[str, Any]]:
    """Memory-map every array read-only, alongside its sidecar."""
    sidecar = _read_json(output_dir / SIDECAR_NAME)
    arrays = {}
    for name, spec in sidecar["key"]["arrays"].items():
        arrays[name] = np.memmap(
            output_dir / f"{name}.memmap",
            dtype=np.dtype(spec["dtype"]),
            mode="r",
            shape=tuple(int(value) for value in spec["shape"]),
        )
    return arrays, sidecar


def verify_against_source(
    output_dir: Path,
    *,
    manifest: Path,
    npz_root: Path | None = None,
    samples: int = 5,
) -> None:
    """Compare the built arrays against the NPZs at distributed rows.

    Checks the first, middle, and last row of ``samples`` trajectories spread
    across the plan, for every array, so a shard that silently wrote the wrong
    span cannot pass.
    """
    arrays, sidecar = open_reference_arrays(output_dir)
    traj_info = sidecar["traj_info"]
    key = sidecar["key"]
    body_names = list(key["body_names"])
    anchor_body = key.get("anchor_body")
    npz_paths = load_npz_paths(manifest, npz_root=npz_root)

    ordered = traj_info["ordered_traj_list"]
    starts = traj_info["start_index"]
    ends = traj_info["end_index"]
    count = len(ordered)
    ranks = sorted(
        {int(round(i * (count - 1) / max(samples - 1, 1))) for i in range(samples)}
    )

    for rank in ranks:
        motion = str(ordered[rank][1])
        start, end = int(starts[rank]), int(ends[rank])
        with np.load(npz_paths[motion]) as npz:
            source_bodies = (
                [str(value) for value in npz["body_names"]]
                if "body_names" in npz
                else []
            )
            body_ids = [source_bodies.index(name) for name in body_names]
            rows = end - start
            for offset in sorted({0, rows // 2, rows - 1}):
                row = start + offset
                for name in arrays:
                    if name in ANCHOR_SOURCES:
                        column = npz[ANCHOR_SOURCES[name]][offset][
                            source_bodies.index(str(anchor_body))
                        ]
                        expected = (
                            column[WXYZ_TO_XYZW] if name == "anchor_quat_w" else column
                        )
                    elif name in BODY_MEMBERS:
                        expected = npz[name][offset][body_ids]
                    else:
                        expected = npz[name][offset]
                    _assert_row(arrays[name][row], expected, motion, name, row)
    logger.info(
        "Verified %s trajectories against %s at first/middle/last rows.",
        len(ranks),
        manifest,
    )


def _assert_row(
    actual: np.ndarray, expected: np.ndarray, motion: str, name: str, row: int
) -> None:
    if not np.array_equal(np.asarray(actual), np.asarray(expected, dtype=DTYPE)):
        raise RuntimeError(
            f"{name} row {row} of motion {motion!r} does not match the source NPZ."
        )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--persist_id",
        required=True,
        help="Stable source-content identity, normally including a manifest hash.",
    )
    parser.add_argument(
        "--traj_info",
        type=Path,
        default=None,
        help=(
            "Existing iltools_rb_manifest.json supplying the canonical trajectory "
            "order and row offsets. Without it, manifest order is used and "
            "trajectory ranks may differ from an earlier replay."
        ),
    )
    parser.add_argument(
        "--npz_root",
        type=Path,
        default=None,
        help="Directory holding the NPZ files, overriding the manifest's paths.",
    )
    parser.add_argument(
        "--body_names",
        nargs="+",
        default=None,
        help=(
            "Ordered bodies to retain, in the order the consuming environment "
            "expects. Default keeps every body the dataset has; naming the "
            "tracked subset is what makes the artifact small."
        ),
    )
    parser.add_argument(
        "--anchor_body",
        default=None,
        help=(
            "Body to extract anchor_pos_w/anchor_quat_w for. Required by the "
            "root_qpos macro cache; omit for a dataset that needs no anchor."
        ),
    )
    parser.add_argument(
        "--dataset_name",
        default="lafan1",
        help="Dataset group label recorded in ordered_traj_list.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help=(
            "Reader processes. Measured on the 129,785-clip tree from a 7200-rpm "
            "disk: 107 files/s at 1 worker, 113 at 2, 120 at 4, and 120 at 8 -- "
            "the drive saturates, so past 4 more workers buy nothing. Raise it "
            "when the source tree is on NVMe."
        ),
    )
    parser.add_argument("--expected_motions", type=int, default=None)
    parser.add_argument("--expected_transitions", type=int, default=None)
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Check the sidecar and array sizes without building.",
    )
    parser.add_argument(
        "--verify_load",
        action="store_true",
        help="Additionally compare built rows against the source NPZs.",
    )
    parser.add_argument("--verify_samples", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = _parse_args(argv)

    if not args.validate_only:
        build_reference_arrays(
            manifest=args.manifest,
            output_dir=args.output_dir,
            persist_id=args.persist_id,
            body_names=list(args.body_names) if args.body_names else None,
            anchor_body=args.anchor_body,
            traj_info=args.traj_info,
            npz_root=args.npz_root,
            dataset_name=args.dataset_name,
            workers=args.workers,
            expected_motions=args.expected_motions,
            expected_transitions=args.expected_transitions,
        )

    errors = validation_errors(
        args.output_dir,
        persist_id=args.persist_id,
        body_names=list(args.body_names) if args.body_names else None,
        anchor_body=args.anchor_body,
        expected_motions=args.expected_motions,
        expected_transitions=args.expected_transitions,
    )
    if errors:
        for error in errors:
            print(f"[FAIL] {error}", file=sys.stderr)
        return 1

    if args.verify_load:
        verify_against_source(
            args.output_dir,
            manifest=args.manifest,
            npz_root=args.npz_root,
            samples=args.verify_samples,
        )

    sidecar = _read_json(args.output_dir / SIDECAR_NAME)
    total = sum(
        int(np.prod(spec["shape"])) * np.dtype(spec["dtype"]).itemsize
        for spec in sidecar["key"]["arrays"].values()
    )
    print(
        f"[PASS] {args.output_dir}: "
        f"{len(sidecar['traj_info']['ordered_traj_list']):,} trajectories, "
        f"{sidecar['traj_info']['written']:,} transitions, "
        f"{total / 1e9:.1f} GB across {len(sidecar['key']['arrays'])} arrays"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
