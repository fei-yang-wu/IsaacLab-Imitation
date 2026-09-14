#!/usr/bin/env python3
"""Run the upstream video-to-data MANO-to-Sharpa retarget natively.

The released pipeline runs its retarget scripts inside the robotic-grounding
Docker image. This driver calls the same scripts from an adjacent
``video_to_data`` checkout inside this repository's ``isaaclab`` environment,
which already carries Pinocchio, Pink, and DAQP. Visualization modules are
stubbed out because the stage runs headless.

Its only purpose is evidence: the output is the reference the ILTools Pink
port is compared against by ``scripts/audit/compare_sharpa_references.py``.
Production data comes from ``convert_mano_sharpa_to_iltools.py --retarget``.

Example:

    pixi run -e isaaclab python scripts/data/run_source_sharpa_retarget.py \\
        --dataset arctic \\
        --input-dir data/dexmanip/human_motion_data/arctic/arctic_loaded \\
        --output-dir /tmp/arctic_processed_source \\
        --sequence-id dataset_s07_box_grab_01 --mano-to-robot-scale 1.0
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

DEFAULT_SOURCE_ROOT = Path("/home/fwu91/Documents/DexManip/video_to_data")
SUPPORTED = ("arctic", "synthbox", "taco", "hot3d", "oakink2", "grab", "h2o", "dexycb")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=SUPPORTED, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--mesh-dir", type=Path, default=None)
    parser.add_argument("--sequence-id", default=None)
    parser.add_argument("--sequence-pattern", default=None)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--mano-to-robot-scale", type=float, default=None)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def install_headless_stubs() -> None:
    """Stub the visualization modules the retarget scripts import at module scope."""

    viser = types.ModuleType("viser")
    viser.ViserServer = object
    sys.modules.setdefault("viser", viser)

    rate_module = types.ModuleType("loop_rate_limiters")

    class RateLimiter:
        def __init__(self, frequency: float, warn: bool = False) -> None:
            del warn
            self.period = 1.0 / float(frequency)

    rate_module.RateLimiter = RateLimiter
    sys.modules.setdefault("loop_rate_limiters", rate_module)

    visualizer = types.ModuleType(
        "robotic_grounding.retarget.pinocchio_viser_visualizer"
    )
    visualizer.ViserVisualizer = object
    sys.modules.setdefault(
        "robotic_grounding.retarget.pinocchio_viser_visualizer", visualizer
    )


def main() -> int:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    package = source_root / "robotic_grounding/source/robotic_grounding"
    scripts = source_root / "robotic_grounding/scripts/retarget"
    for path in (package, scripts):
        if not path.is_dir():
            raise SystemExit(f"Adjacent video-to-data checkout is incomplete: {path}")

    install_headless_stubs()
    sys.path.insert(0, str(package))
    sys.path.insert(0, str(scripts))
    module = __import__(f"{args.dataset}_to_sharpa")

    # The released scripts read one argparse namespace. Build it here instead
    # of re-parsing, so this driver stays the only command-line surface.
    namespace = argparse.Namespace(
        input_dir=args.input_dir.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        device=args.device,
        save=True,
        visualize=False,
        visualize_object_point_clouds=False,
        sequence_id=args.sequence_id,
        sequence_pattern=args.sequence_pattern,
        max_sequences=args.max_sequences,
        shard_id=0,
        num_shards=1,
    )
    if args.mano_to_robot_scale is not None:
        namespace.mano_to_robot_scale = float(args.mano_to_robot_scale)
    elif hasattr(module, "parse_args"):
        namespace.mano_to_robot_scale = 1.0
    if args.mesh_dir is not None:
        namespace.mesh_dir = args.mesh_dir.expanduser().resolve()
    elif hasattr(module, "ARCTIC_MESH_DIR"):
        namespace.mesh_dir = module.ARCTIC_MESH_DIR

    module.main(namespace)
    print(f"wrote {namespace.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
