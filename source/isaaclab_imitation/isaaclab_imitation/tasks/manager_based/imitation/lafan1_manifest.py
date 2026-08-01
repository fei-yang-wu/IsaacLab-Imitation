"""Backward-compatible re-export shim for the motion-manifest schema authority.

The implementation moved to :mod:`.motion_manifest`, which is the single
schema authority for LAFAN1-style motion-dataset manifests. Repo-owned code
should import from ``motion_manifest`` directly; this module only keeps the
historical import path working.
"""

from __future__ import annotations

from .motion_manifest import (
    MANIFESTS_DIR,
    PACKAGE_ROOT,
    build_lafan1_loader_kwargs,
    dataset_path_from_entries,
    infer_npz_manifest_control_freq,
    load_lafan1_manifest,
    load_lafan1_manifest_loader_options,
    normalize_lafan1_entries,
)

__all__ = [
    "MANIFESTS_DIR",
    "PACKAGE_ROOT",
    "build_lafan1_loader_kwargs",
    "dataset_path_from_entries",
    "infer_npz_manifest_control_freq",
    "load_lafan1_manifest",
    "load_lafan1_manifest_loader_options",
    "normalize_lafan1_entries",
]
