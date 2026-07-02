#!/usr/bin/env python3
"""Run the LaFAN1 language pipeline from the cluster Python launcher."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _parse_env_assignment(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"Expected KEY=VALUE for --set, got {value!r}."
        )
    key, raw = value.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError(f"Empty environment key in {value!r}.")
    return key, raw


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--set",
        dest="env",
        action="append",
        default=[],
        type=_parse_env_assignment,
        metavar="KEY=VALUE",
        help="Environment variable forwarded to run_lafan1_language_pipeline.sh.",
    )
    parser.add_argument(
        "--script",
        default="scripts/rlopt/run_lafan1_language_pipeline.sh",
        help="Pipeline shell script to execute from the repository root.",
    )
    args = parser.parse_args()

    env = os.environ.copy()
    for key, value in args.env:
        env[key] = value
    env.setdefault("PYTHON_BIN", "/isaac-sim/python.sh")

    script = Path(args.script)
    if not script.is_absolute():
        script = REPO_ROOT / script
    subprocess.run(["bash", str(script)], cwd=REPO_ROOT, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
