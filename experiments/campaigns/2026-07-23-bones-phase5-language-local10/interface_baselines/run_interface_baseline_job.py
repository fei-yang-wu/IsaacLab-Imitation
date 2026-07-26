#!/usr/bin/env python3
"""Launch interface-baseline workflows with the split CU130 runtime."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


# Resolve the implementation as a sibling so this keeps working regardless of
# how many directory levels sit above interface_baselines (the 2026-07-23
# reorg moved it under experiments/paper/, which broke the old parents[2] +
# hardcoded experiments/interface_baselines path).
IMPLEMENTATION = Path(__file__).resolve().parent / "run_interface_baseline_job_impl.py"


def _configure_container_runtime(env: dict[str, str]) -> None:
    if not Path("/isaac-sim/python.sh").is_file():
        return
    runtime_command = "bash scripts/rlopt/runtime_python.sh"
    env.setdefault("INTERFACE_BASELINE_PYTHON_CMD", runtime_command)
    env.setdefault("INTERFACE_BASELINE_ISAACLAB_PYTHON_CMD", runtime_command)


def main() -> int:
    env = dict(os.environ)
    _configure_container_runtime(env)
    os.environ.clear()
    os.environ.update(env)
    sys.argv[0] = str(IMPLEMENTATION)
    runpy.run_path(str(IMPLEMENTATION), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
