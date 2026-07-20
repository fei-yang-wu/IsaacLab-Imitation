"""CPU-only tests for split RLOpt runtime selection and Kit ownership."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "rlopt"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from runtime_bootstrap import (  # noqa: E402
    _KitImportGuard,
    assert_kit_not_loaded,
    config_contains_type_name,
    requested_backend,
    validate_gpu_policy,
)


def _load_script(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_DIR / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_backend_selection_is_explicit_and_conflict_checked(monkeypatch):
    monkeypatch.delenv("CLUSTER_SIM_BACKEND", raising=False)
    assert requested_backend([]) == "physx"
    assert requested_backend(["physics=physx"]) == "physx"
    assert requested_backend(["physics=newton_mjwarp"]) == "newton"
    assert requested_backend(["presets=newton_mjwarp,newton_warp_renderer"]) == "newton"
    assert requested_backend(["--assert-kitless"]) == "newton"
    with pytest.raises(ValueError, match="conflicts"):
        requested_backend(["physics=physx"], explicit="newton")


def test_kit_import_guard_and_loaded_module_assertion(monkeypatch):
    guard = _KitImportGuard()
    with pytest.raises(ModuleNotFoundError, match="Strict kit-less"):
        guard.find_spec("isaacsim")
    assert guard.find_spec("torch") is None

    monkeypatch.setitem(
        sys.modules, "omni.kit.runtime_test", types.ModuleType("omni.kit.runtime_test")
    )
    with pytest.raises(RuntimeError, match="forbidden Kit modules"):
        assert_kit_not_loaded()


def test_gpu_policy_rejects_compute_only_physx():
    with pytest.raises(RuntimeError, match="RT-capable"):
        validate_gpu_policy("physx", ("NVIDIA H100 80GB HBM3",))
    validate_gpu_policy("newton", ("NVIDIA H100 80GB HBM3",))
    validate_gpu_policy("physx", ("NVIDIA L40S",))
    validate_gpu_policy("physx", ("NVIDIA A40",))


def test_gpu_policy_allows_explicit_compute_only_physx_experiment():
    validate_gpu_policy(
        "physx",
        ("NVIDIA H100 80GB HBM3",),
        allow_compute_only_physx=True,
    )


def test_config_type_scan_handles_nested_containers():
    NewtonCfg = type("NewtonCfg", (), {})

    class Root:
        pass

    root = Root()
    root.physics = {"selected": [NewtonCfg()]}
    assert config_contains_type_name(root, "NewtonCfg")
    assert not config_contains_type_name(root, "PhysxCfg")


def test_train_dispatcher_import_is_runtime_light():
    watched = ("torch", "gymnasium", "isaaclab", "isaacsim", "wandb", "rlopt")
    before = {name for name in watched if name in sys.modules}
    _load_script("test_train_dispatcher", "train.py")
    after = {name for name in watched if name in sys.modules}
    assert after == before


def test_newton_robot_asset_preflight(tmp_path):
    module = _load_script("test_train_asset_preflight", "train.py")
    NewtonCfg = type("NewtonCfg", (), {})
    root = types.SimpleNamespace(
        physics=NewtonCfg(),
        scene=types.SimpleNamespace(
            robot=types.SimpleNamespace(
                spawn=types.SimpleNamespace(usd_path=str(tmp_path / "g1.usd"))
            )
        ),
    )
    with pytest.raises(FileNotFoundError, match="preconverted G1 USD"):
        module._validate_newton_robot_asset(root)
    (tmp_path / "g1.usd").touch()
    module._validate_newton_robot_asset(root)


def test_physx_bootstrap_owns_exactly_one_app(monkeypatch):
    module = _load_script("test_train_physx", "train_physx.py")
    events: list[object] = []

    class FakeApp:
        def close(self):
            events.append("close")

    class FakeAppLauncher:
        @staticmethod
        def add_app_launcher_args(parser: argparse.ArgumentParser):
            parser.add_argument("--headless", action="store_true")

        def __init__(self, args):
            events.append(("launch", args.headless))
            self.app = FakeApp()

    fake_isaaclab = types.ModuleType("isaaclab")
    fake_isaaclab_app = types.ModuleType("isaaclab.app")
    fake_isaaclab_app.AppLauncher = FakeAppLauncher
    fake_train = types.ModuleType("train")

    def fake_run(argv, *, require_running_kit):
        events.append(("run", tuple(argv), require_running_kit))
        return 17

    fake_train.run = fake_run
    monkeypatch.setitem(sys.modules, "isaaclab", fake_isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.app", fake_isaaclab_app)
    monkeypatch.setitem(sys.modules, "train", fake_train)
    monkeypatch.setattr(module, "detect_gpu_names", lambda: ("NVIDIA L40S",))
    monkeypatch.setattr(module, "configure_cu130_bridge", lambda required: None)

    assert module.main(["--headless", "physics=physx"]) == 17
    assert events == [
        ("launch", True),
        ("run", ("--headless", "physics=physx"), True),
        "close",
    ]


def test_physx_bootstrap_rejects_newton_before_app_launch(monkeypatch):
    module = _load_script("test_train_physx_reject_newton", "train_physx.py")
    monkeypatch.delenv("CLUSTER_SIM_BACKEND", raising=False)
    monkeypatch.setattr(
        module,
        "detect_gpu_names",
        lambda: pytest.fail("GPU detection must not run for Newton."),
    )
    with pytest.raises(RuntimeError, match="only owns the PhysX/Kit runtime"):
        module.main(["physics=newton_mjwarp"])
