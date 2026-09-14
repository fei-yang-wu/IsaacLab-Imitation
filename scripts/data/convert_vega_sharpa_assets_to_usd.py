#!/usr/bin/env python3
"""Convert the composed Vega / Sharpa MJCF for both Isaac physics backends."""

import argparse
import json
from pathlib import Path


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app
    try:
        from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg

        cfg = MjcfConverterCfg(
            asset_path=str(args.model.resolve()),
            usd_dir=str(args.output_dir.resolve()),
            usd_file_name="vega_u_sharpa.usd",
            fix_base=True,
            self_collision=True,
            collision_from_visuals=False,
            collision_type="Convex Hull",
            run_asset_transformer=True,
            run_multi_physics_conversion=True,
            force_usd_conversion=True,
        )
        converter = MjcfConverter(cfg)
        from imitation_experiments.data.vega_sharpa_model import (
            author_robot_gravity_compensation,
            author_robot_shape_collision_exclusions,
            describe_robot_asset,
        )

        author_robot_gravity_compensation(Path(converter.usd_path))
        author_robot_shape_collision_exclusions(Path(converter.usd_path))

        (args.output_dir / "asset.json").write_text(
            json.dumps(
                describe_robot_asset(Path(converter.usd_path), args.model),
                indent=2,
            )
            + "\n"
        )
        print(f"Converted mounted robot: {converter.usd_path}", flush=True)
    finally:
        app.close()


if __name__ == "__main__":
    main()
