#!/usr/bin/env python3
"""Assert every interface's demonstrations were collected under identical settings.

Why this exists
---------------
A capacity comparison may vary exactly one thing: the command interface. It may
not vary how the training data was collected. The 2026-07-28 audit found three
knobs that differed between the latent and explicit demonstration sets, none of
them intentional -- all three were emergent defaults of two different collector
scripts:

    knob                    latent      explicit
    random_reset_step_max   0           200
    episode_length_s        10.0        20.04
    tracking terminations   disabled    ACTIVE

The last is the one that bites. With those terms active an explicit episode was
reset the moment its tracker drifted, so the explicit demonstration sets contain
only well-tracked states while the latent set contains drifted ones too. A
planner that never saw a drifted state cannot learn to recover from one, which
biases the closed-loop result against the explicit arms for a reason that has
nothing to do with the interface.

`random_reset_step_max` differed because
`eval_skill_commander_closed_loop.py:1282-1289` forces it back to 0 unless
`--allow_random_reset` is given -- silently, *after* Hydra applies the override
the launcher was passing. The launcher looked correct and was not.

Run this before training any planner on a demonstration set.

Usage
-----
    pixi run python .../audit_demonstration_symmetry.py \\
        --oracle_root logs/interface_baselines/lafan1_interface_capacity/oracle_baselines
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Recorded metadata keys that must agree across every interface. Each collector
# writes these into its demonstration summary.json.
REQUIRED_MATCHING_KEYS = (
    "random_reset_step_min",
    "random_reset_step_max",
    "reset_schedule",
    "wrap_steps",
    "episode_length_s",
    "num_envs",
    "motion_name",
    "motion_manifest",
    "tracking_terminations_enabled",
)
# Derived from disabled_tracking_termination_terms when the collector records
# the list instead of the boolean.
TRACKING_TERMINATION_NAMES = ("anchor_pos", "anchor_ori", "ee_body_pos")


def _load(path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text())
    metadata = summary.get("metadata")
    if not isinstance(metadata, dict):
        raise SystemExit(f"{path} has no metadata block.")
    record = {key: metadata.get(key) for key in REQUIRED_MATCHING_KEYS}
    # Normalize the two ways the collectors express the same fact.
    disabled = metadata.get("disabled_tracking_termination_terms")
    if record["tracking_terminations_enabled"] is None and disabled is not None:
        record["tracking_terminations_enabled"] = not set(
            TRACKING_TERMINATION_NAMES
        ).issubset(set(disabled))
    # episode_length_s is a float from two different code paths; compare rounded
    # so 10.0 vs 10.000000001 is not reported as a difference.
    if isinstance(record["episode_length_s"], (int, float)):
        record["episode_length_s"] = round(float(record["episode_length_s"]), 3)
    return record


def _budget(demo_dir: Path) -> dict[str, Any]:
    """Row and trajectory counts, read from the sample files themselves.

    A trajectory is the (env_id, episode_id) pair -- episode_id alone is a
    per-environment reset counter, so grouping on it would merge distinct
    rollouts that happen to share a number.
    """
    import torch

    files = sorted((demo_dir / "rollout_training_samples").glob("*.pt"))
    if not files:
        return {}
    env_ids, episode_ids = [], []
    rows = 0
    for path in files:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if "env_id" not in payload or "episode_id" not in payload:
            return {
                "rows": int(payload["causal_target"].shape[0]),
                "trajectories": None,
            }
        env_ids.append(payload["env_id"].reshape(-1))
        episode_ids.append(payload["episode_id"].reshape(-1))
        rows += int(payload["causal_target"].shape[0])
    keys = torch.stack([torch.cat(env_ids), torch.cat(episode_ids)], dim=1)
    trajectories = int(torch.unique(keys, dim=0).shape[0])
    return {
        "rows": rows,
        "trajectories": trajectories,
        "rows_per_trajectory": round(rows / max(trajectories, 1), 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle_root", type=Path, required=True)
    parser.add_argument(
        "--interfaces",
        nargs="*",
        default=None,
        help="Restrict the check. Default: every interface with a demo summary.",
    )
    args = parser.parse_args()

    root = args.oracle_root.expanduser().resolve()
    summaries = sorted(root.glob("*/oracle_demonstrations/summary.json"))
    if args.interfaces:
        wanted = set(args.interfaces)
        summaries = [p for p in summaries if p.parent.parent.name in wanted]
    if len(summaries) < 2:
        raise SystemExit(
            f"Need at least two demonstration summaries under {root}; found "
            f"{len(summaries)}."
        )

    records = {path.parent.parent.name: _load(path) for path in summaries}
    sample_paths = {path.parent.parent.name: path.parent for path in summaries}
    interfaces = sorted(records)

    width = max(len(name) for name in interfaces) + 2
    print(f"{'knob':38}" + "".join(f"{name:>{width}}" for name in interfaces))
    mismatched: list[str] = []
    for key in REQUIRED_MATCHING_KEYS:
        values = [records[name][key] for name in interfaces]
        agree = all(value == values[0] for value in values)
        if not agree:
            mismatched.append(key)
        marker = "  " if agree else " !"
        rendered = "".join(f"{str(value)[: width - 1]:>{width}}" for value in values)
        print(f"{key:36}{marker}{rendered}")

    budgets = {
        name: budget
        for name, path in sample_paths.items()
        if (budget := _budget(path))
    }
    if len(budgets) == len(interfaces):
        print()
        print("Budget (from the sample files, not the summary):")
        for key, label in (
            ("rows", "rows (must match)"),
            ("trajectories", "trajectories (reported)"),
            ("rows_per_trajectory", "rows / trajectory"),
        ):
            values = [budgets.get(name, {}).get(key) for name in interfaces]
            agree = all(value == values[0] for value in values)
            marker = "  " if agree else " !"
            rendered = "".join(
                f"{str(value)[: width - 1]:>{width}}" for value in values
            )
            print(f"{label:36}{marker}{rendered}")
        row_counts = {budgets[name]["rows"] for name in budgets}
        if len(row_counts) > 1:
            mismatched.append("demonstration_rows")
        print(
            "\nRows are the controlled quantity: they set how much the optimizer\n"
            "sees, and collection stops at the exact row budget. Trajectory counts\n"
            "differ by a few because environments reset asynchronously under the\n"
            "0-200 random start, so the row budget binds before the last episode\n"
            "closes. Quote the actual per-interface trajectory count; do not round\n"
            "it to the requested DEMO_TRAJECTORIES."
        )

    print()
    if mismatched:
        print(
            f"[FAIL] {len(mismatched)} collection setting(s) or budget(s) differ across "
            f"interfaces: {', '.join(mismatched)}.\n"
            "       The demonstration sets are not a controlled comparison. "
            "Re-collect with\n"
            "       prepare_oracle_baselines.sh before training any planner on "
            "them.",
            file=sys.stderr,
        )
        return 1
    print(
        f"[PASS] All {len(REQUIRED_MATCHING_KEYS)} settings agree across {interfaces}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
