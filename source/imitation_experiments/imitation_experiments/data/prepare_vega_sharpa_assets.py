"""Build shared mounted-robot and scene assets, then convert in a fresh Kit process."""

import argparse
from pathlib import Path
import subprocess
import sys

from iltools.core import load_dexterous_reference_set
from imitation_experiments.data.vega_sharpa_model import compose_model
from imitation_experiments.data.vega_sharpa_scene import build_scene_assets
from imitation_experiments.paths import REPO_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--object-mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stand-radius", type=float, default=0.055)
    args = parser.parse_args()
    output = args.output.resolve()
    source = load_dexterous_reference_set(args.source_manifest)[0]
    model = compose_model(
        REPO_ROOT / "source/isaaclab_imitation/isaaclab_imitation/assets",
        output / "model/vega_u_sharpa.xml",
    )
    build_scene_assets(
        args.object_mesh.resolve(),
        source.object_poses_w[0, 0],
        output / "scene",
        stand_radius_m=args.stand_radius,
    )
    # USD was imported above. Kit must start in a fresh process for the
    # pinned Isaac Sim version, using this worktree's own Pixi interpreter.
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/data/convert_vega_sharpa_assets_to_usd.py"),
            "--model",
            str(model),
            "--output-dir",
            str(output / "model/usd"),
            "--headless",
        ],
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
