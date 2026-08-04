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

#: Stable, release-facing paper entrypoints kept outside the package.
PAPER_DIR = REPO_ROOT / "experiments/paper"


def load_paper_entrypoint(script_name: str):
    """Import a release-facing entrypoint from ``experiments/paper`` by path.

    Those scripts are entrypoints, not library modules: they stay at the paths
    protocols and gates cite, and they are deliberately not importable by name.
    Tests still need to exercise their internals, and loading the file at its
    documented location says exactly that -- unlike putting
    ``experiments/paper`` on ``sys.path``, which makes every module in it
    silently importable from everywhere and is what this repo's layout rules
    forbid.

    Args:
        script_name: File stem under ``experiments/paper``, e.g.
            ``"build_paper_release_bundle"``.
    """
    import importlib.util

    script_path = PAPER_DIR / f"{script_name}.py"
    if not script_path.is_file():
        raise FileNotFoundError(f"No paper entrypoint at {script_path}")
    spec = importlib.util.spec_from_file_location(script_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load paper entrypoint {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def module_source_rel(module_name: str) -> str:
    """Repo-relative source path of a package module, for provenance hashing."""
    import importlib.util

    spec = importlib.util.find_spec(module_name)
    if spec is None or not spec.origin:
        raise RuntimeError(f"Cannot resolve source file for {module_name}")
    return Path(spec.origin).resolve().relative_to(REPO_ROOT).as_posix()
