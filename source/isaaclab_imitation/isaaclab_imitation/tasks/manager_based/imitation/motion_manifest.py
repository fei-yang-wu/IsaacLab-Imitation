"""Single schema authority for motion-clip manifests.

A *clip manifest* names an ordered set of motion clips -- each a file (NPZ, or
a CSV that still needs resampling) plus its source frame rate and an optional
frame range. It is the only way motion data enters training, and it is dataset
agnostic: LAFAN1, Unitree Dance102, BONES-SEED, and cross-dataset merges are
all just different *contents* of the same blob, distinguished by the free-form
``metadata.family`` label rather than by code paths::

    {
      "dataset_name": "...",
      "dataset": {"trajectories": {"lafan1_csv": [
        {"name": ..., "path": ..., "input_fps": ...[, "frame_range": [a, b]]}
      ]}},
      "metadata": {..., "family": ..., "role": ...}
    }

``lafan1_csv`` in that payload is NOT a dataset name: it is the key of the
ILTools clip loader that consumes the blob (see :data:`CLIP_LOADER_KEY`), owned
by the ImitationLearningTools submodule. It is the one place that spelling is
allowed to appear, and nothing in this repo should branch on it.

This module owns reading, validating, and writing the blob. It is deliberately
stdlib-only (json/pathlib/hashlib) so scripts and tests can load a manifest by
file path without importing the ``isaaclab_imitation`` package (which registers
Isaac tasks on import). The environment-facing configuration built on top of it
lives in :mod:`motion_data`.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

# Resolve the package root directly from this module to avoid importing the
# top-level package, which also registers Isaac tasks on import.
PACKAGE_ROOT = Path(__file__).resolve().parents[3]
MANIFESTS_DIR = PACKAGE_ROOT / "manifests"

# The ILTools loader that consumes a clip manifest, and the payload key it
# reads its entries from. This is a submodule contract (see
# ``iltools/datasets/loaders.py``), which is why the spelling is historical:
# it is the ONLY place a dataset name appears as a code identifier, and it is
# passed through to the loader rather than branched on.
CLIP_LOADER_KEY = "lafan1_csv"

# ``metadata.family`` names the motion-data lineage a manifest describes and
# ``metadata.role`` what it is for in the experiment workflow. Both are
# free-form labels about the *data*: they are carried through, used to make
# cache directories self-describing, and never branched on, so a new dataset
# needs no code change here. These are the families in use today.
KNOWN_MANIFEST_FAMILIES = ("lafan1", "dance102", "bones_seed", "unified")

# Entry keys carried through normalization beyond the required
# name/path/input_fps(/frame_range) set. The ILTools ``Lafan1CsvLoader``
# reads entries with ``Mapping.get`` and ignores unknown keys, so preserving
# these provenance keys is safe end to end.
_PRESERVED_ENTRY_KEYS = ("source_dataset", "source_motion_name")


def normalize_clip_entries(
    entries_like: list[dict[str, Any]],
    *,
    base_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Normalize LAFAN1 source entries into absolute-path loader entries.

    Provenance keys (``source_dataset``/``source_motion_name``, as written by
    the unified-manifest merge) are preserved when present.
    """
    resolved_base_dir = None
    if base_dir is not None:
        resolved_base_dir = Path(base_dir).expanduser().resolve()

    entries: list[dict[str, Any]] = []
    for index, entry_like in enumerate(entries_like):
        if not isinstance(entry_like, dict):
            raise ValueError(
                f"Manifest entry #{index} must be a mapping, got {type(entry_like)}."
            )

        path_value = entry_like.get("path") or entry_like.get("file")
        if path_value is None:
            raise ValueError(
                f"Manifest entry #{index} must include `path` (or `file`)."
            )
        if "input_fps" not in entry_like:
            raise ValueError(f"Manifest entry #{index} must include `input_fps`.")

        source_path = Path(str(path_value)).expanduser()
        if source_path.is_absolute():
            source_path = source_path.resolve()
        elif resolved_base_dir is not None:
            source_path = (resolved_base_dir / source_path).resolve()
        else:
            source_path = source_path.resolve()

        entries.append(
            {
                "name": str(entry_like.get("name") or source_path.stem),
                "path": str(source_path),
                "input_fps": float(entry_like["input_fps"]),
                **(
                    {"frame_range": entry_like["frame_range"]}
                    if "frame_range" in entry_like
                    else {}
                ),
                **{
                    key: entry_like[key]
                    for key in _PRESERVED_ENTRY_KEYS
                    if key in entry_like
                },
            }
        )

    return entries


def load_clip_manifest(
    manifest_path: str | Path,
) -> tuple[Path, list[dict[str, Any]]]:
    """Load manifest entries and resolve relative clip paths against the manifest file."""
    manifest_file = Path(manifest_path).expanduser().resolve()
    if not manifest_file.is_file():
        raise FileNotFoundError(f"Clip manifest not found: {manifest_file}")

    data = json.loads(manifest_file.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        entries_like = (
            data.get("dataset", {}).get("trajectories", {}).get(CLIP_LOADER_KEY)
        )
        if entries_like is None:
            entries_like = data.get(CLIP_LOADER_KEY, data.get("motions", data))
    else:
        entries_like = data

    if not isinstance(entries_like, list) or len(entries_like) == 0:
        raise ValueError(
            f"Manifest must define a non-empty `dataset.trajectories.{CLIP_LOADER_KEY}` "
            f"list: {manifest_file}"
        )

    entries = normalize_clip_entries(entries_like, base_dir=manifest_file.parent)
    return manifest_file, entries


def load_clip_loader_options(manifest_path: str | Path) -> dict[str, int]:
    """Load optional ILTools loader options from manifest metadata."""
    manifest_file = Path(manifest_path).expanduser().resolve()
    data = json.loads(manifest_file.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    metadata = data.get("metadata", {})
    if not isinstance(metadata, dict):
        return {}
    raw_options = metadata.get("loader_kwargs", {})
    if not isinstance(raw_options, dict):
        return {}

    options: dict[str, int] = {}
    for key in ("chunk_size", "shard_size"):
        if key not in raw_options or raw_options[key] is None:
            continue
        value = int(raw_options[key])
        if value <= 0:
            raise ValueError(f"Manifest loader_kwargs.{key} must be positive.")
        options[key] = value
    return options


def infer_manifest_control_freq(entries: list[dict[str, Any]]) -> float | None:
    """Infer a single control frequency from NPZ manifest entries.

    CSV manifests often describe source data that still needs resampling, so timing is
    inferred only when every source is an NPZ and all declared ``input_fps`` values
    agree.
    """
    if len(entries) == 0:
        return None
    fps_values: list[float] = []
    for entry in entries:
        path = Path(str(entry["path"]))
        if path.suffix.lower() != ".npz":
            return None
        fps = float(entry["input_fps"])
        if fps <= 0.0:
            return None
        fps_values.append(fps)
    first = fps_values[0]
    if all(abs(value - first) <= 1.0e-6 for value in fps_values):
        return first
    return None


def build_clip_loader_kwargs(
    *,
    entries: list[dict[str, Any]],
    sim_dt: float,
    decimation: int,
    joint_names: list[str],
    control_freq: float | None = None,
    dataset_name: str = "lafan1",
    canonical_joint_names: list[str] | None = None,
) -> dict[str, Any]:
    """Build the ILTools clip-loader call arguments from resolved clip entries.

    These are call arguments, not configuration: everything here is derived
    from the environment config and the manifest, so the result is built at
    resolution time and never stored as a config field.

    ``joint_names`` is the *native* joint-order fallback used only for sources
    whose NPZ carries no ``joint_names``. ``canonical_joint_names`` (when given)
    is the order every trajectory is unified to at zarr-build time (the robot
    articulation order), so training reads a single canonical layout.
    ``dataset_name`` is a provenance label the loader records; it defaults to
    the historical value so cache identities stay stable.
    """
    normalized_entries = normalize_clip_entries(copy.deepcopy(entries))
    if len(normalized_entries) == 0:
        raise ValueError("Clip loader entries must be a non-empty list.")
    if control_freq is None:
        control_freq = 1.0 / (float(sim_dt) * float(decimation))

    loader_kwargs: dict[str, Any] = {
        "dataset_name": str(dataset_name),
        "dataset": {"trajectories": {CLIP_LOADER_KEY: normalized_entries}},
        "control_freq": float(control_freq),
        "sim": {"dt": float(sim_dt)},
        "decimation": int(decimation),
        "joint_names": list(joint_names),
    }
    if canonical_joint_names is not None:
        loader_kwargs["canonical_joint_names"] = list(canonical_joint_names)
    return loader_kwargs


def _sanitize_cache_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return name or "manifest"


def cache_dir_from_entries(
    entries: list[dict[str, Any]],
    *,
    manifest_path: str | Path | None = None,
    family: str | None = None,
) -> str:
    """Create a stable cache path tied to the manifest identity and entries.

    The directory name is prefixed with the manifest's ``family`` so caches
    self-describe (``iltools_g1_bones_seed_tracking_*``). The two ``"lafan1"``
    fallbacks below are frozen spellings, not defaults worth generalizing:
    changing either would rename -- and therefore invalidate -- every cache
    built before this function existed.
    """
    cache_root = Path(
        os.environ.get("ISAACLAB_IMITATION_MOTION_CACHE_ROOT", "/tmp")
    ).expanduser()
    resolved_manifest_path = None
    manifest_name = "lafan1"
    if manifest_path is not None:
        resolved_manifest_path = str(Path(manifest_path).expanduser().resolve())
        manifest_name = _sanitize_cache_name(Path(resolved_manifest_path).stem)

    family_name = "lafan1" if family is None else _sanitize_cache_name(str(family))

    signature = json.dumps(
        {
            "manifest_path": resolved_manifest_path,
            "entries": entries,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:12]
    return str(
        cache_root / f"iltools_g1_{family_name}_tracking_{manifest_name}_{digest}"
    )


def manifest_family(payload: Any) -> str | None:
    """Return ``metadata.family`` from a manifest payload, or None."""
    if not isinstance(payload, dict):
        return None
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return None
    value = metadata.get("family")
    return None if value is None else str(value)


def manifest_role(payload: Any) -> str | None:
    """Return ``metadata.role`` from a manifest payload, or None."""
    if not isinstance(payload, dict):
        return None
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return None
    value = metadata.get("role")
    return None if value is None else str(value)


def load_manifest_family(manifest_path: str | Path) -> str | None:
    """Read ``metadata.family`` from a manifest file, tolerating legacy blobs."""
    manifest_file = Path(manifest_path).expanduser().resolve()
    if not manifest_file.is_file():
        return None
    try:
        payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return manifest_family(payload)


def read_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Read and validate a motion-dataset manifest, returning the raw payload.

    Motion paths are NOT resolved here; use :func:`load_clip_manifest` for
    resolved loader entries. Legacy manifests without ``metadata.family`` or
    ``metadata.role`` are accepted (the accessors return None for them).
    """
    manifest_file = Path(manifest_path).expanduser().resolve()
    if not manifest_file.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_file}")
    payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest root must be a JSON object: {manifest_file}")

    entries = payload.get("dataset", {}).get("trajectories", {}).get(CLIP_LOADER_KEY)
    if not isinstance(entries, list) or len(entries) == 0:
        raise ValueError(
            f"Manifest must define a non-empty `dataset.trajectories.{CLIP_LOADER_KEY}` "
            f"list: {manifest_file}"
        )
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(
                f"Manifest entry #{index} must be a mapping: {manifest_file}"
            )
        if (entry.get("path") or entry.get("file")) is None:
            raise ValueError(
                f"Manifest entry #{index} must include `path` (or `file`): "
                f"{manifest_file}"
            )
        if "input_fps" not in entry:
            raise ValueError(
                f"Manifest entry #{index} must include `input_fps`: {manifest_file}"
            )

    metadata = payload.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError(f"Manifest `metadata` must be a mapping: {manifest_file}")

    return payload


def write_manifest(
    manifest_path: str | Path,
    *,
    dataset_name: str,
    entries: list[dict[str, Any]],
    metadata: dict[str, Any],
    family: str,
    role: str | None = None,
) -> dict[str, Any]:
    """Write the canonical clip-manifest blob and return the payload.

    ``metadata`` is carried through verbatim with ``family`` and ``role``
    merged in. Both are free-form labels describing the data (see
    :data:`KNOWN_MANIFEST_FAMILIES`); a new dataset needs no change here. The
    file is written with sorted keys, 2-space indentation, and a trailing
    newline.
    """
    if not str(family).strip():
        raise ValueError("family must be a non-empty label naming the data lineage.")
    if role is not None and not str(role).strip():
        raise ValueError("role must be a non-empty label when given.")
    if not isinstance(entries, list) or len(entries) == 0:
        raise ValueError("Manifest entries must be a non-empty list.")
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Manifest entry #{index} must be a mapping.")
        if (entry.get("path") or entry.get("file")) is None:
            raise ValueError(
                f"Manifest entry #{index} must include `path` (or `file`)."
            )
        if "input_fps" not in entry:
            raise ValueError(f"Manifest entry #{index} must include `input_fps`.")
    if not isinstance(metadata, dict):
        raise ValueError("Manifest metadata must be a mapping.")

    merged_metadata = dict(metadata)
    merged_metadata["family"] = str(family)
    if role is not None:
        merged_metadata["role"] = str(role)
    payload: dict[str, Any] = {
        "dataset_name": str(dataset_name),
        "dataset": {
            "trajectories": {CLIP_LOADER_KEY: [dict(entry) for entry in entries]}
        },
        "metadata": merged_metadata,
    }

    manifest_file = Path(manifest_path).expanduser()
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload
