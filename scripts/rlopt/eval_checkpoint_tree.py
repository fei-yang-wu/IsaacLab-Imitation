#!/usr/bin/env python3
"""Score every milestone of one checkpoint tree in a single process.

`imitation_experiments.lowlevel.evaluate_checkpoint` starts Isaac Sim at import
time, so scoring N checkpoints as N processes pays N simulation starts. At the
256-clip board that start is most of the per-cell cost. An arm fixes the
command interface, so the only thing that changes between its milestones is the
policy weights: one start can serve the whole budget axis.

This entrypoint plans the cells with
`imitation_experiments.evaluation.score_tree`, then hands the whole list to the
evaluator through `--checkpoints` / `--output_jsons`. Every argument after `--`
goes to the evaluator verbatim, so the interface overrides stay exactly the
ones the campaign launcher would pass for a single cell.

    python scripts/rlopt/eval_checkpoint_tree.py \
        --tree /data/pareto_stack/jepa_h1_ee_wide_seed0 \
        --output_root /data/eval/pareto_stack --arm jepa_h1_ee_wide --seed 0 \
        --row milestone -- --task Isaac-Imitation-G1-v2 --algo IPMD ...

It exists as a script rather than a `-m` invocation because a cluster stage
runs one script path inside the container.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import runpy
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "source" / "imitation_experiments"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Stdlib-only, and safe before Isaac Sim or Torch: `score_tree` plans cells from
# the filesystem alone, and `runtime_bootstrap` is the split-runtime helper.
from imitation_experiments.evaluation.score_tree import plan_cells  # noqa: E402
from runtime_bootstrap import (  # noqa: E402
    configure_cu130_bridge,
    verify_cu130_torch,
)

EVALUATOR = "imitation_experiments.lowlevel.evaluate_checkpoint"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--arm", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--row", type=str, default="milestone")
    # A board name instead of 256 or 4096 literal ranks in a config file. The
    # ranks and `--num_envs` are expanded here, so a campaign declares the
    # board it scores and cannot drift from the board's definition.
    parser.add_argument("--board", type=str, default=None)
    parser.add_argument("--final_only", action="store_true", default=False)
    # Off by default: an interrupted job resumes instead of repeating cells.
    parser.add_argument("--rescore", action="store_true", default=False)
    parser.add_argument(
        "evaluator_args",
        nargs=argparse.REMAINDER,
        help="Arguments after `--`, passed to the evaluator verbatim.",
    )
    args = parser.parse_args()

    passthrough = list(args.evaluator_args)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    if not passthrough:
        parser.error("pass the evaluator arguments after `--`.")

    if args.board is not None:
        for flag in ("--trajectory_ranks", "--num_envs"):
            if flag in passthrough:
                parser.error(f"--board already sets {flag}; pass one or the other.")
        from imitation_experiments.evaluation.protocol import BOARDS

        if args.board not in BOARDS:
            parser.error(f"unknown board: {args.board}")
        ranks = [str(case.trajectory_rank) for case in BOARDS[args.board].cases]
        if not ranks:
            parser.error(f"board {args.board} has no cases.")
        passthrough += ["--num_envs", str(len(ranks)), "--trajectory_ranks", *ranks]
        print(f"[INFO] board {args.board}: {len(ranks)} clips")

    cells = plan_cells(
        args.tree,
        args.output_root,
        arm=args.arm,
        seed=args.seed,
        row=args.row,
        final_only=args.final_only,
        skip_scored=not args.rescore,
    )
    if not cells:
        print(
            f"[INFO] nothing to score: {args.arm} seed{args.seed} {args.row} "
            f"({args.tree})"
        )
        return 0

    args.output_root.mkdir(parents=True, exist_ok=True)
    print(
        f"[INFO] {args.arm} seed{args.seed} {args.row}: {len(cells)} cell(s) "
        f"in one simulation start"
    )
    for cell in cells:
        print(f"[INFO]   f{cell.frames} -> {cell.output_json.name}")

    # In the split container runtime, Kit's Python is the only interpreter with
    # both Kit and Torch -- but Torch lives in the CU130 site-packages and has
    # to be put on the path first. The evaluator imports Torch at module level,
    # so this has to happen before `run_module`. Outside the container the
    # bridge finds nothing and does nothing (ICE job 5593233).
    site_packages = configure_cu130_bridge(
        required=os.environ.get("ISAACLAB_REQUIRE_CU130_RUNTIME") == "1"
    )
    if site_packages is not None:
        # Import the CU130 Torch stack before Kit extensions can load Isaac
        # Sim's bundled NCCL into the process.
        verify_cu130_torch(site_packages)

    sys.argv = (
        [f"{EVALUATOR}.py"]
        + passthrough
        + ["--checkpoints"]
        + [str(cell.checkpoint) for cell in cells]
        + ["--output_jsons"]
        + [str(cell.output_json) for cell in cells]
    )
    runpy.run_module(EVALUATOR, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
