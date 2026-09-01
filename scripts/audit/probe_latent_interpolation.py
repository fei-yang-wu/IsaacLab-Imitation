#!/usr/bin/env python3
"""Cluster entrypoint for the latent-interpolation probe.

The analysis lives in
`imitation_experiments.capacity.probe_latent_interpolation`; this file exists
so the cluster control plane has a `scripts/` path to name as a stage
executable. On a workstation, run the module directly with `python -m`.

Under the cluster's split runtime the interpreter is Kit's Python, which has no
Torch of its own: the CU130 stack has to be appended to `sys.path` by
`configure_cu130_bridge` BEFORE anything imports Torch. That is why the probe
is imported inside `main` rather than at module scope, and why the imports
above it are stdlib only. Importing it at the top raises
`ModuleNotFoundError: No module named 'torch'` (ICE jobs 5598895/5598896).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "source" / "imitation_experiments"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "rlopt"))

from runtime_bootstrap import (  # noqa: E402
    configure_cu130_bridge,
    verify_cu130_torch,
)


def main() -> int:
    site_packages = configure_cu130_bridge(
        required=os.environ.get("ISAACLAB_REQUIRE_CU130_RUNTIME") == "1"
    )
    if site_packages is not None:
        verify_cu130_torch(site_packages)

    from imitation_experiments.capacity.probe_latent_interpolation import (
        main as probe_main,
    )

    probe_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
