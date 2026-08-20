"""Kit-first Newton training entrypoint for MJCF-backed Isaac tasks.

Newton itself does not require the Isaac Sim Kit runtime, but Isaac Lab's
MJCF-to-USD converter does. This wrapper owns one Kit application before the
shared RLOpt dispatcher creates the Vega-Wuji environment.
"""

from __future__ import annotations

import argparse
import sys

from runtime_bootstrap import requested_backend


def _parse_launcher_args(argv: list[str]):
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--video", action="store_true", default=False)
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args(argv)
    if args_cli.video:
        args_cli.enable_cameras = True
    return args_cli


def main(argv: list[str] | None = None) -> int:
    """Launch Kit once, then run the shared Newton RLOpt dispatcher."""

    if argv is None:
        argv = sys.argv[1:]
    if requested_backend(argv) != "newton":
        raise RuntimeError(
            "scripts/rlopt/train_newton.py requires an explicit "
            "physics=newton_mjwarp override."
        )

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(_parse_launcher_args(argv))
    try:
        from train import run

        status = run(argv, require_running_kit=True)
        return 0 if status is None else status
    finally:
        app_launcher.app.close()


if __name__ == "__main__":
    raise SystemExit(main())
