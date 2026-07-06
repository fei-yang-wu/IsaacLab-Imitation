#!/usr/bin/env python3
"""Cluster Python wrapper for the Dance102 DiT latent M3 shell pipeline."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_env_args(argv: list[str]) -> tuple[dict[str, str], list[str]]:
    env_updates: dict[str, str] = {}
    passthrough: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--env":
            i += 1
            if i >= len(argv) or "=" not in argv[i]:
                raise ValueError("--env requires KEY=VALUE")
            key, value = argv[i].split("=", 1)
            env_updates[key] = value
        elif arg.startswith("--env="):
            assignment = arg.removeprefix("--env=")
            if "=" not in assignment:
                raise ValueError("--env requires KEY=VALUE")
            key, value = assignment.split("=", 1)
            env_updates[key] = value
        elif arg == "--dry_run":
            env_updates["DRY_RUN"] = "1"
        elif arg == "--no-dry_run":
            env_updates["DRY_RUN"] = "0"
        else:
            passthrough.append(arg)
        i += 1
    return env_updates, passthrough


def main() -> None:
    repo_root = _repo_root()
    env = dict(os.environ)
    env_updates, passthrough = _parse_env_args(sys.argv[1:])
    env.update(env_updates)
    env.setdefault("PYTHON_BIN", sys.executable)

    script = repo_root / "scripts" / "rlopt" / "run_dance102_dit_latent_m3.sh"
    cmd = ["bash", str(script), *passthrough]
    print("[CMD] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=repo_root, env=env, check=True)


if __name__ == "__main__":
    main()
