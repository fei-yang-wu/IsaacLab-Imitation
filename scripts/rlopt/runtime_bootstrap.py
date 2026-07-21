"""Process-level runtime selection helpers for split Isaac Lab execution.

This module intentionally depends only on the Python standard library.  It is
safe to import before Isaac Sim, PyTorch, Isaac Lab, and the project task
registry have been imported.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

KIT_MODULE_PREFIXES = (
    "isaacsim",
    "omni.kit",
    "omni.isaac.kit",
)
COMPUTE_ONLY_GPU_MARKERS = ("A100", "H100", "H200")
RUNTIME_ROOT_CANDIDATES = (
    Path("/opt/isaaclab-imitation-runtime"),
    Path("/opt/isaaclab-imitation-runtime-spec/.pixi/envs/container-runtime"),
)


def _matches_prefix(module_name: str, prefixes: tuple[str, ...]) -> bool:
    return any(
        module_name == prefix or module_name.startswith(f"{prefix}.")
        for prefix in prefixes
    )


def loaded_kit_modules() -> tuple[str, ...]:
    """Return Kit/SimulationApp modules already loaded in this process."""
    return tuple(
        sorted(
            name for name in sys.modules if _matches_prefix(name, KIT_MODULE_PREFIXES)
        )
    )


def assert_kit_not_loaded() -> None:
    """Fail when strict kit-less execution has imported any Kit modules."""
    loaded = loaded_kit_modules()
    if loaded:
        preview = ", ".join(loaded[:8])
        if len(loaded) > 8:
            preview += f", ... ({len(loaded)} total)"
        raise RuntimeError(
            f"Strict kit-less execution imported forbidden Kit modules: {preview}"
        )


class _KitImportGuard(importlib.abc.MetaPathFinder):
    """Reject Kit imports while allowing Isaac Lab's optional-import probes."""

    def find_spec(self, fullname: str, path: object = None, target: object = None):
        del path, target
        if _matches_prefix(fullname, KIT_MODULE_PREFIXES):
            raise ModuleNotFoundError(
                f"Strict kit-less execution blocked import of {fullname!r}.",
                name=fullname,
            )
        return None


def install_kit_import_guard() -> _KitImportGuard:
    """Install and return the strict Kit import guard."""
    assert_kit_not_loaded()
    for finder in sys.meta_path:
        if isinstance(finder, _KitImportGuard):
            return finder
    guard = _KitImportGuard()
    sys.meta_path.insert(0, guard)
    return guard


def requested_backend(argv: list[str], explicit: str | None = None) -> str:
    """Resolve ``physx`` or ``newton`` from raw CLI tokens without imports.

    The split runtime defaults to PhysX because that is the default Isaac Lab
    task backend.  Strict Newton must be selected explicitly through
    ``physics=newton_mjwarp`` or ``CLUSTER_SIM_BACKEND=newton``.
    """
    choice = (explicit or os.environ.get("CLUSTER_SIM_BACKEND", "auto")).strip().lower()
    if choice not in {"", "auto", "newton", "physx"}:
        raise ValueError(
            f"Unsupported simulation backend {choice!r}; expected auto, newton, or physx."
        )

    token_choices: set[str] = set()
    for token in argv:
        if token == "--assert-kitless":
            token_choices.add("newton")
        elif token.startswith("physics="):
            physics_name = token.split("=", 1)[1].strip().lower()
            if physics_name == "physx":
                token_choices.add("physx")
            elif physics_name.startswith("newton"):
                token_choices.add("newton")
        elif token.startswith("presets="):
            presets = {
                item.strip().lower() for item in token.split("=", 1)[1].split(",")
            }
            if any(item.startswith("newton") for item in presets):
                token_choices.add("newton")

    if len(token_choices) > 1:
        raise ValueError(
            f"Conflicting simulation backend arguments: {sorted(token_choices)}"
        )
    token_choice = next(iter(token_choices), None)
    if (
        choice not in {"", "auto"}
        and token_choice is not None
        and choice != token_choice
    ):
        raise ValueError(
            f"CLUSTER_SIM_BACKEND={choice} conflicts with CLI-selected backend {token_choice}."
        )
    if choice not in {"", "auto"}:
        return choice
    return token_choice or "physx"


def detect_gpu_names() -> tuple[str, ...]:
    """Read visible GPU model names without importing CUDA or PyTorch."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ()
    return tuple(line.strip() for line in result.stdout.splitlines() if line.strip())


def validate_gpu_policy(
    backend: str,
    gpu_names: tuple[str, ...],
    *,
    allow_compute_only_physx: bool = False,
) -> None:
    """Reject Kit/PhysX on compute-only GPU models before AppLauncher starts."""
    if backend != "physx":
        return
    unsupported = [
        name
        for name in gpu_names
        if any(marker in name.upper() for marker in COMPUTE_ONLY_GPU_MARKERS)
    ]
    if unsupported and allow_compute_only_physx:
        return
    if unsupported:
        raise RuntimeError(
            "PhysX/Kit requires an RT-capable GPU; compute-only GPU(s) were selected: "
            + ", ".join(unsupported)
            + ". Use strict Newton on A100/H100/H200."
        )


def find_runtime_site_packages(required: bool) -> Path | None:
    """Find the immutable CU130 runtime's site-packages directory."""
    explicit = os.environ.get("ISAACLAB_CU130_SITE_PACKAGES")
    if explicit:
        candidates = [Path(explicit)]
    else:
        runtime_root = os.environ.get("ISAACLAB_CU130_RUNTIME_ROOT")
        roots = [Path(runtime_root)] if runtime_root else list(RUNTIME_ROOT_CANDIDATES)
        candidates = []
        for root in roots:
            candidates.extend(sorted(root.glob("lib/python*/site-packages")))

    for candidate in candidates:
        if candidate.is_dir() and (candidate / "torch").is_dir():
            return candidate.resolve()
    if required:
        rendered = ", ".join(str(path) for path in candidates) or "<no candidates>"
        raise RuntimeError(
            f"Could not locate the CU130 runtime site-packages; checked: {rendered}"
        )
    return None


def configure_cu130_bridge(required: bool) -> Path | None:
    """Append the runtime packages and prefer only its ``nvidia`` namespace."""
    site_packages = find_runtime_site_packages(required=required)
    if site_packages is None:
        return None

    site_text = str(site_packages)
    if site_text not in sys.path:
        sys.path.append(site_text)

    nvidia_path = site_packages / "nvidia"
    loaded_nvidia = sys.modules.get("nvidia")
    if loaded_nvidia is not None and nvidia_path.is_dir():
        namespace_path = getattr(loaded_nvidia, "__path__", None)
        if namespace_path is not None and str(nvidia_path) not in namespace_path:
            namespace_path.insert(0, str(nvidia_path))
    elif nvidia_path.is_dir():
        finder = _NvidiaNamespaceFinder(site_packages)
        if not any(
            isinstance(item, _NvidiaNamespaceFinder)
            and item.site_packages == site_packages
            for item in sys.meta_path
        ):
            sys.meta_path.insert(0, finder)
    return site_packages


class _NvidiaNamespaceFinder(importlib.abc.MetaPathFinder):
    """Resolve the top-level ``nvidia`` namespace from the CU130 runtime."""

    def __init__(self, site_packages: Path):
        self.site_packages = site_packages

    def find_spec(self, fullname: str, path: object = None, target: object = None):
        del path, target
        if fullname != "nvidia":
            return None
        return importlib.machinery.PathFinder.find_spec(
            fullname, [str(self.site_packages)]
        )


def verify_cu130_torch(site_packages: Path) -> Any:
    """Import Torch before Kit startup and verify the immutable CU130 package."""
    import torch

    torch_file = Path(torch.__file__).resolve()
    expected_root = site_packages.resolve()
    if (
        torch_file != expected_root / "torch" / "__init__.py"
        and expected_root not in torch_file.parents
    ):
        raise RuntimeError(
            f"Torch resolved from {torch_file}, expected the CU130 runtime under {expected_root}."
        )
    if torch.__version__.split("+", 1)[0] != "2.11.0":
        raise RuntimeError(
            f"Expected PyTorch 2.11.0, found {torch.__version__} at {torch_file}."
        )
    if torch.version.cuda != "13.0":
        raise RuntimeError(
            f"Expected torch.version.cuda == '13.0', found {torch.version.cuda!r}."
        )
    print(
        "[INFO] Verified CU130 bridge: "
        f"torch={torch.__version__}, cuda={torch.version.cuda}, origin={torch_file}"
    )
    return torch


def config_contains_type_name(config: object, type_name: str) -> bool:
    """Recursively find a config node by concrete class name."""
    visited: set[int] = set()
    stack: list[object] = [config]
    while stack:
        node = stack.pop()
        node_id = id(node)
        if node_id in visited:
            continue
        visited.add(node_id)
        if type(node).__name__ == type_name:
            return True
        if isinstance(node, dict):
            stack.extend(node.values())
            continue
        if isinstance(node, (list, tuple, set)):
            stack.extend(node)
            continue
        try:
            children = vars(node).values()
        except TypeError:
            continue
        for child in children:
            if child is None or isinstance(child, (bool, int, float, str, bytes)):
                continue
            if isinstance(child, dict):
                stack.extend(child.values())
            elif isinstance(child, (list, tuple, set)):
                stack.extend(child)
            else:
                stack.append(child)
    return False
