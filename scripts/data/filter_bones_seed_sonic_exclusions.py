#!/usr/bin/env python3
"""Drop BONES-SEED trajectories matching the public SONIC release's exclusion filter.

Applies the exact keyword list from ``filter_and_copy_bones_data.py`` in the
public SONIC release (NVlabs/GR00T-WholeBodyControl) to a G1 BONES-SEED
manifest's trajectory names, and writes a new manifest with the matches
dropped. Motions are excluded only if their trajectory *name* contains a
keyword; NPZ files are left untouched, so this is a manifest-only filter.
"""

import argparse
import hashlib
import importlib.util
from pathlib import Path

_MOTION_MANIFEST_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "source"
    / "isaaclab_imitation"
    / "isaaclab_imitation"
    / "tasks"
    / "manager_based"
    / "imitation"
    / "motion_manifest.py"
)


def _load_motion_manifest_module():
    """Load the stdlib-only manifest schema authority by file path.

    Avoids importing the ``isaaclab_imitation`` package, which registers Isaac
    tasks on import and is unavailable in the default Pixi environment.
    """
    spec = importlib.util.spec_from_file_location(
        "motion_manifest", _MOTION_MANIFEST_MODULE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not load motion manifest module: {_MOTION_MANIFEST_MODULE_PATH}"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Verbatim default --filter-keywords list from
# gear_sonic/data_process/filter_and_copy_bones_data.py in
# NVlabs/GR00T-WholeBodyControl (the public SONIC release).
SONIC_RELEASE_FILTER_KEYWORDS = [
    "bed",
    "bike",
    "chair",
    "climb",
    "com_up_50cm",
    "sitting",
    "step_on",
    "seat",
    "table",
    "_sit_",
    "sit_",
    "ladder",
    "crutch",
    "_bed_",
    "_ride_",
    "scooter",
    "stepdown",
    "acrobatics_",
    "box_HSPU",
    "cartwheel",
    "50cm_box_",
    "on_box",
    "fall_from",
    "handstand_ff_",
    "on_1m",
    "form_box",
    "off_1m",
    "230m",
    "jump_over_obstacle_",
    "lift_crate_come_up_",
    "jump_to_shoulder_roll",
    "kozak_dance",
    "stair",
    "handstand",
    "box_jump",
    "monkey_jump",
    "safety_roll",
    "box_dips",
    "walking_on_edge",
    "push_obstacle",
]


def matched_keywords(name: str, keywords: list[str]) -> list[str]:
    name_lower = name.lower()
    return [kw for kw in keywords if kw.lower() in name_lower]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--dest-manifest", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    motion_manifest = _load_motion_manifest_module()
    manifest = motion_manifest.read_manifest(args.source_manifest)
    trajectories = manifest["dataset"]["trajectories"]["lafan1_csv"]

    kept = []
    excluded = []
    for traj in trajectories:
        hits = matched_keywords(traj["name"], SONIC_RELEASE_FILTER_KEYWORDS)
        if hits:
            excluded.append({"name": traj["name"], "matched_keywords": hits})
        else:
            kept.append(traj)

    print(f"Source: {args.source_manifest} ({len(trajectories)} trajectories)")
    print(f"Excluded {len(excluded)} trajectories matching the SONIC release filter:")
    for entry in excluded:
        print(f"  {entry['name']} -> {entry['matched_keywords']}")
    print(f"Kept {len(kept)} trajectories.")

    if args.dry_run:
        return

    metadata = dict(manifest.get("metadata") or {})
    metadata["num_motions"] = len(kept)
    metadata["sonic_exclusion_filter_applied"] = True
    metadata["sonic_exclusion_filter_source"] = (
        "NVlabs/GR00T-WholeBodyControl gear_sonic/data_process/"
        "filter_and_copy_bones_data.py default --filter-keywords"
    )
    metadata["sonic_excluded_trajectories"] = excluded
    metadata["sonic_exclusion_filter_source_manifest"] = str(args.source_manifest)

    motion_manifest.write_manifest(
        args.dest_manifest,
        dataset_name=manifest.get("dataset_name", "bones_seed"),
        entries=kept,
        metadata=metadata,
        family="bones_seed",
        role="headline",
    )

    digest = hashlib.sha256(args.dest_manifest.read_bytes()).hexdigest()
    print(f"Wrote: {args.dest_manifest}")
    print(f"SHA-256: {digest}")


if __name__ == "__main__":
    main()
