from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).parent / "isaaclab_imitation"
_AGENT_CONFIG_PATH = (
    _PACKAGE_ROOT
    / "tasks"
    / "manager_based"
    / "imitation"
    / "config"
    / "g1"
    / "agents"
    / "rlopt_ipmd_cfg.py"
)
_PLANNER_MODULE_PATH = _PACKAGE_ROOT / "envs" / "causal_planner_observation.py"
_SUPERVISION_KEY = ("policy_supervision", "expert_action")


def _literal_assignment(path: Path, name: str) -> object:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == name and node.value is not None:
                return ast.literal_eval(node.value)
    raise AssertionError(f"Could not find literal assignment {name} in {path}.")


def _method_attribute_assignment(
    path: Path,
    *,
    class_name: str,
    method_name: str,
    attribute_path: tuple[str, ...],
) -> object:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for item in node.body:
            if not isinstance(item, ast.FunctionDef) or item.name != method_name:
                continue
            for assignment in ast.walk(item):
                if not isinstance(assignment, ast.Assign):
                    continue
                for target in assignment.targets:
                    parts: list[str] = []
                    current = target
                    while isinstance(current, ast.Attribute):
                        parts.append(current.attr)
                        current = current.value
                    if isinstance(current, ast.Name):
                        parts.append(current.id)
                    if tuple(reversed(parts)) == attribute_path:
                        return ast.literal_eval(assignment.value)
    raise AssertionError(
        f"Could not find {'.'.join(attribute_path)} in "
        f"{class_name}.{method_name}."
    )


def _planner_observation_spec() -> dict[str, object]:
    module_spec = importlib.util.spec_from_file_location(
        "causal_planner_observation_contract", _PLANNER_MODULE_PATH
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module.causal_planner_observation_spec(history_steps=9)


def test_training_action_label_is_absent_from_every_low_level_actor_input() -> None:
    for constant_name in (
        "VANILLA_POLICY_INPUT_KEYS",
        "LATENT_POLICY_INPUT_KEYS",
        "PROPRIO_POLICY_INPUT_KEYS",
        "FULL_BODY_TRAJECTORY_COMMAND_KEYS",
        "EE_TRAJECTORY_COMMAND_KEYS",
    ):
        actor_keys = _literal_assignment(_AGENT_CONFIG_PATH, constant_name)
        assert isinstance(actor_keys, list)
        assert _SUPERVISION_KEY not in actor_keys


def test_training_action_label_is_absent_from_causal_planner_input() -> None:
    spec = _planner_observation_spec()
    assert "expert_action" not in spec["feature_names"]
    assert "policy_supervision" not in spec["feature_names"]
    assert spec["reference_features"] == []


def test_future_cvae_command_width_excludes_phase_features() -> None:
    def assignment(path: tuple[str, ...]) -> object:
        return _method_attribute_assignment(
            _AGENT_CONFIG_PATH,
            class_name="G1ImitationLatentFutureCVAERLOptIPMDConfig",
            method_name="__post_init__",
            attribute_path=path,
        )

    assert assignment(("self", "ipmd", "latent_dim")) == 256
    assert assignment(("self", "ipmd", "latent_learning", "code_latent_dim")) == 256
    assert (
        assignment(("self", "ipmd", "latent_learning", "command_phase_mode"))
        == "none"
    )
