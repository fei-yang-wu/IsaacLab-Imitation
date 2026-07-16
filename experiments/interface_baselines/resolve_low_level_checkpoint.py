#!/usr/bin/env python3
"""Resolve one final RLOpt checkpoint from an exact recorded experiment name."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log_root", type=Path, required=True)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--checkpoint_basename", required=True)
    return parser.parse_args()


def _recorded_experiment_name(command_path: Path) -> str | None:
    tokens = shlex.split(command_path.read_text(encoding="utf-8"))
    prefix = "agent.logger.exp_name="
    names = [token.removeprefix(prefix) for token in tokens if token.startswith(prefix)]
    if len(names) > 1:
        raise ValueError(f"Multiple experiment names in {command_path}")
    return names[0] if names else None


def resolve_checkpoint(
    log_root: Path,
    *,
    run_id: str,
    checkpoint_basename: str,
) -> Path:
    root = log_root.expanduser().resolve()
    expected_name = f"{run_id}_oracle_low_level"
    matches = [
        command_path.parent
        for command_path in sorted(root.glob("*/command.txt"))
        if _recorded_experiment_name(command_path) == expected_name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one low-level run named {expected_name!r} under "
            f"{root}; found {len(matches)}."
        )
    checkpoint = matches[0] / "models" / checkpoint_basename
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Exact final checkpoint is missing for {expected_name!r}: {checkpoint}"
        )
    return checkpoint.resolve()


def main() -> None:
    args = _parse_args()
    print(
        resolve_checkpoint(
            args.log_root,
            run_id=args.run_id,
            checkpoint_basename=args.checkpoint_basename,
        )
    )


if __name__ == "__main__":
    main()
