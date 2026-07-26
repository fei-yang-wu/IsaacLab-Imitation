"""Location-independent repository paths for the shared interface-baseline code.

Earlier reorganizations moved this directory twice and silently broke every
``Path(__file__).resolve().parents[N]`` that hard-coded a nesting depth. Resolve
the repository root by walking up to its markers instead, so a future move stays
correct without editing each caller.
"""

from __future__ import annotations

from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    """Return the repository root containing ``pixi.toml`` and ``source/``."""
    origin = (start or Path(__file__)).resolve()
    for candidate in origin.parents:
        if (candidate / "pixi.toml").is_file() and (candidate / "source").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate the repository root above {origin}")


REPO_ROOT = find_repo_root()

#: Stable, release-facing paper entrypoints kept outside this campaign directory.
PAPER_DIR = REPO_ROOT / "experiments/paper"
