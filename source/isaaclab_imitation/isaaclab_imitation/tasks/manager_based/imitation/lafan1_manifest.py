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


def normalize_lafan1_entries(
    entries_like: list[dict[str, Any]],
    *,
    base_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Normalize LAFAN1 source entries into absolute-path loader entries."""
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
    """Build normalized LAFAN1 loader kwargs from resolved source entries."""
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
) -> str:
    """Create a stable cache path tied to the manifest identity and entries."""
    cache_root = Path(
        os.environ.get("ISAACLAB_IMITATION_LAFAN1_ZARR_CACHE_ROOT", "/tmp")
    ).expanduser()
    resolved_manifest_path = None
    manifest_name = "lafan1"
    if manifest_path is not None:
        resolved_manifest_path = str(Path(manifest_path).expanduser().resolve())
        manifest_name = _sanitize_cache_name(Path(resolved_manifest_path).stem)

    signature = json.dumps(
        {
            "manifest_path": resolved_manifest_path,
            "entries": entries,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:12]
    return str(cache_root / f"iltools_g1_lafan1_tracking_{manifest_name}_{digest}")
