#!/usr/bin/env python3
"""Run an Isaac/Kit command with the immutable CU130 Torch bridge enabled."""

from __future__ import annotations

import ctypes
import runpy
import sys
from pathlib import Path

from runtime_bootstrap import (
    configure_cu130_bridge,
    find_runtime_site_packages,
    verify_cu130_torch,
)


def _preload_runtime_nccl(site_packages: Path) -> None:
    nccl = site_packages / "nvidia" / "nccl" / "lib" / "libnccl.so.2"
    if not nccl.is_file():
        raise FileNotFoundError(nccl)
    ctypes.CDLL(str(nccl), mode=ctypes.RTLD_GLOBAL)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        raise SystemExit("usage: run_kit_command.py SCRIPT [ARGS ...]")

    site_packages = find_runtime_site_packages(required=True)
    assert site_packages is not None
    _preload_runtime_nccl(site_packages)
    configure_cu130_bridge(required=True)
    verify_cu130_torch(site_packages)

    script = Path(argv[0]).expanduser()
    if not script.is_absolute():
        script = Path.cwd() / script
    if not script.is_file():
        raise FileNotFoundError(script)

    sys.argv = [str(script), *argv[1:]]
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
