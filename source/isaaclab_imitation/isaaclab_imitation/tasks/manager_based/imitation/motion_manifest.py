"""Single schema authority for LAFAN1-style motion-dataset manifests.

Every motion dataset in this repo (LAFAN1, Unitree Dance102, BONES-SEED, and
the unified merge) enters training through the same ``loader_type="lafan1_csv"``
manifest blob::

    {
      "dataset_name": "...",
      "dataset": {"trajectories": {"lafan1_csv": [
        {"name": ..., "path": ..., "input_fps": ...[, "frame_range": [a, b]]}
      ]}},
      "metadata": {..., "family": ..., "role": ...}
    }

This module owns reading, validating, and writing that blob. It is
deliberately stdlib-only (json/pathlib/hashlib) so scripts and tests can load
it by file path without importing the ``isaaclab_imitation`` package (which
registers Isaac tasks on import). ``lafan1_manifest.py`` remains as a thin
re-export shim for legacy imports.
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

# Known dataset families and manifest roles. ``metadata.family`` identifies
# which motion-data lineage a manifest belongs to; ``metadata.role`` records
# what the manifest is for in the experiment workflow.
MOTION_MANIFEST_FAMILIES = ("lafan1", "dance102", "bones_seed", "unified")
MOTION_MANIFEST_ROLES = ("debug", "daily", "testing", "headline")

# Conventional role for each family when a writer does not pass one
# explicitly: dance102 is the quick debug set, lafan1 the daily driver,
# unified the cross-dataset testing merge, and bones_seed the headline data.
DEFAULT_ROLE_BY_FAMILY = {
    "lafan1": "daily",
    "dance102": "debug",
    "bones_seed": "headline",
    "unified": "testing",
}

# Entry keys carried through normalization beyond the required
# name/path/input_fps(/frame_range) set. The ILTools ``Lafan1CsvLoader``
# reads entries with ``Mapping.get`` and ignores unknown keys, so preserving
# these provenance keys is safe end to end.
_PRESERVED_ENTRY_KEYS = ("source_dataset", "source_motion_name")


def normalize_lafan1_entries(
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


def load_lafan1_manifest(
    manifest_path: str | Path,
) -> tuple[Path, list[dict[str, Any]]]:
    """Load manifest entries and resolve relative motion paths against the manifest file."""
    manifest_file = Path(manifest_path).expanduser().resolve()
    if not manifest_file.is_file():
        raise FileNotFoundError(f"lafan1_manifest_path not found: {manifest_file}")

    data = json.loads(manifest_file.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        entries_like = data.get("dataset", {}).get("trajectories", {}).get("lafan1_csv")
        if entries_like is None:
            entries_like = data.get("lafan1_csv", data.get("motions", data))
    else:
        entries_like = data

    if not isinstance(entries_like, list) or len(entries_like) == 0:
        raise ValueError(
            "Manifest must define a non-empty `dataset.trajectories.lafan1_csv` list."
        )

    entries = normalize_lafan1_entries(entries_like, base_dir=manifest_file.parent)
    return manifest_file, entries


def load_lafan1_manifest_loader_options(manifest_path: str | Path) -> dict[str, int]:
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


def infer_npz_manifest_control_freq(entries: list[dict[str, Any]]) -> float | None:
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


def build_lafan1_loader_kwargs(
    *,
    entries: list[dict[str, Any]],
    sim_dt: float,
    decimation: int,
    joint_names: list[str],
    control_freq: float | None = None,
    dataset_name: str = "lafan1",
    canonical_joint_names: list[str] | None = None,
) -> dict[str, Any]:
    """Build normalized LAFAN1 loader kwargs from resolved source entries.

    ``joint_names`` is the *native* joint-order fallback used only for sources
    whose NPZ carries no ``joint_names``. ``canonical_joint_names`` (when given)
    is the order every trajectory is unified to at zarr-build time (the robot
    articulation order), so training reads a single canonical layout.
    """
    normalized_entries = normalize_lafan1_entries(copy.deepcopy(entries))
    if len(normalized_entries) == 0:
        raise ValueError("LAFAN1 loader entries must be a non-empty list.")
    if control_freq is None:
        control_freq = 1.0 / (float(sim_dt) * float(decimation))

    loader_kwargs: dict[str, Any] = {
        "dataset_name": str(dataset_name),
        "dataset": {"trajectories": {"lafan1_csv": normalized_entries}},
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


def dataset_path_from_entries(
    entries: list[dict[str, Any]],
    *,
    manifest_path: str | Path | None = None,
    family: str | None = None,
) -> str:
    """Create a stable cache path tied to the manifest identity and entries.

    The cache directory name is prefixed with the manifest's ``family``
    (fallback ``"lafan1"`` when absent) so caches self-describe, e.g.
    ``iltools_g1_bones_seed_tracking_*`` instead of a family-agnostic
    ``iltools_g1_lafan1_tracking_*``. This changes cache directory names for
    NEW resolutions only: the content digest is computed exactly as before,
    previously built caches simply become cold, and ``refresh_zarr_dataset``
    semantics are untouched.
    """
    cache_root = Path(
        os.environ.get("ISAACLAB_IMITATION_LAFAN1_ZARR_CACHE_ROOT", "/tmp")
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

    Motion paths are NOT resolved here; use :func:`load_lafan1_manifest` for
    resolved loader entries. Legacy manifests without ``metadata.family`` or
    ``metadata.role`` are accepted (the accessors return None for them).
    """
    manifest_file = Path(manifest_path).expanduser().resolve()
    if not manifest_file.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_file}")
    payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest root must be a JSON object: {manifest_file}")

    entries = payload.get("dataset", {}).get("trajectories", {}).get("lafan1_csv")
    if not isinstance(entries, list) or len(entries) == 0:
        raise ValueError(
            "Manifest must define a non-empty `dataset.trajectories.lafan1_csv` "
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

    family = manifest_family(payload)
    if family is not None and family not in MOTION_MANIFEST_FAMILIES:
        raise ValueError(
            f"Manifest metadata.family={family!r} is not one of "
            f"{MOTION_MANIFEST_FAMILIES}: {manifest_file}"
        )
    role = manifest_role(payload)
    if role is not None and role not in MOTION_MANIFEST_ROLES:
        raise ValueError(
            f"Manifest metadata.role={role!r} is not one of "
            f"{MOTION_MANIFEST_ROLES}: {manifest_file}"
        )

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
    """Write the canonical motion-dataset manifest blob and return the payload.

    ``metadata`` is carried through verbatim with ``family`` and ``role``
    merged in. When ``role`` is None it defaults from
    :data:`DEFAULT_ROLE_BY_FAMILY`. The file is written with sorted keys,
    2-space indentation, and a trailing newline.
    """
    if family not in MOTION_MANIFEST_FAMILIES:
        raise ValueError(f"family={family!r} is not one of {MOTION_MANIFEST_FAMILIES}.")
    if role is None:
        role = DEFAULT_ROLE_BY_FAMILY[family]
    if role not in MOTION_MANIFEST_ROLES:
        raise ValueError(f"role={role!r} is not one of {MOTION_MANIFEST_ROLES}.")
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
    merged_metadata["family"] = family
    merged_metadata["role"] = role
    payload: dict[str, Any] = {
        "dataset_name": str(dataset_name),
        "dataset": {"trajectories": {"lafan1_csv": [dict(entry) for entry in entries]}},
        "metadata": merged_metadata,
    }

    manifest_file = Path(manifest_path).expanduser()
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload
