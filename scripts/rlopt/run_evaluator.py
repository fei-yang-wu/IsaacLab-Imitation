"""Run `imitation_experiments.lowlevel.evaluate_checkpoint` with the container's
torch bridge in place.

`eval_checkpoint_tree.py` does the same for a checkpoint tree; this is the
single-run form the composition probe driver spawns per setting. Every
argument goes to the evaluator verbatim.

    python scripts/rlopt/run_evaluator.py --task Isaac-Imitation-G1-v2 ...
"""

from __future__ import annotations

import os
from pathlib import Path
import runpy
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "source" / "imitation_experiments"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime_bootstrap import (  # noqa: E402
    configure_cu130_bridge,
    verify_cu130_torch,
)

EVALUATOR = "imitation_experiments.lowlevel.evaluate_checkpoint"


def main() -> int:
    site_packages = configure_cu130_bridge(
        required=os.environ.get("ISAACLAB_REQUIRE_CU130_RUNTIME") == "1"
    )
    if site_packages is not None:
        verify_cu130_torch(site_packages)
    sys.argv = [f"{EVALUATOR}.py"] + sys.argv[1:]
    runpy.run_module(EVALUATOR, run_name="__main__", alter_sys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
