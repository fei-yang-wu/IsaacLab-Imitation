"""Golden observation-layout contract for every registered G1 imitation task.

The ordered term list of each observation group IS the checkpoint input
contract: reordering, adding, or removing a term silently changes the actor's
input layout and invalidates every checkpoint trained against it. The env
consolidation refactor (recipe x command-config) must therefore reproduce
these layouts exactly for every task id that predates it.

Regenerate the golden file only when a layout change is intentional:

    REGENERATE_G1_TASK_LAYOUT_GOLDEN=1 pixi run -e isaaclab pytest -q \
        source/isaaclab_imitation/test_g1_task_layout_contract.py

and say so in the commit message.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import os
from pathlib import Path

import pytest

import isaaclab_imitation.tasks  # noqa: F401  (registers the gym tasks)
import gymnasium as gym

GOLDEN_PATH = Path(__file__).with_name("g1_task_layout_golden.json")

TASK_IDS = sorted(
    spec_id
    for spec_id in gym.registry
    if spec_id.startswith("Isaac-Imitation-G1")
)


def _load_env_cfg(task_id: str):
    entry = gym.spec(task_id).kwargs["env_cfg_entry_point"]
    module_name, class_name = entry.split(":")
    return getattr(importlib.import_module(module_name), class_name)()


def _group_terms(group) -> list[str]:
    terms = []
    for field in dataclasses.fields(group):
        value = getattr(group, field.name)
        if value is not None and hasattr(value, "func"):
            terms.append(field.name)
    return terms


def _layout(task_id: str) -> dict:
    cfg = _load_env_cfg(task_id)
    groups = {}
    for field in dataclasses.fields(cfg.observations):
        group = getattr(cfg.observations, field.name)
        if group is None:
            continue
        groups[field.name] = _group_terms(group)
    return {
        "env_cfg_class": type(cfg).__name__,
        "observations_class": type(cfg.observations).__name__,
        "rewards_class": type(cfg.rewards).__name__,
        "terminations_class": type(cfg.terminations).__name__,
        "events_class": type(cfg.events).__name__,
        "actions_class": type(cfg.actions).__name__,
        "curriculum": type(cfg.curriculum).__name__ if cfg.curriculum else None,
        "anchor_body": getattr(cfg, "expert_anchor_body_name", None),
        "latent_command_dim": getattr(cfg, "latent_command_dim", None),
        "latent_patch_past_steps": getattr(cfg, "latent_patch_past_steps", None),
        "latent_patch_future_steps": getattr(cfg, "latent_patch_future_steps", None),
        "command_hold_steps": getattr(cfg, "command_hold_steps", None),
        "random_reset_step_min": getattr(cfg, "random_reset_step_min", None),
        "random_reset_step_max": getattr(cfg, "random_reset_step_max", None),
        "random_reset_full_trajectory": getattr(
            cfg, "random_reset_full_trajectory", None
        ),
        "groups": groups,
    }


def test_registered_task_layouts_match_golden() -> None:
    current = {task_id: _layout(task_id) for task_id in TASK_IDS}
    if os.environ.get("REGENERATE_G1_TASK_LAYOUT_GOLDEN") == "1":
        GOLDEN_PATH.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
        pytest.skip(f"regenerated {GOLDEN_PATH.name}")
    assert GOLDEN_PATH.is_file(), (
        f"{GOLDEN_PATH} missing; regenerate with REGENERATE_G1_TASK_LAYOUT_GOLDEN=1"
    )
    golden = json.loads(GOLDEN_PATH.read_text())
    assert sorted(current) == sorted(golden), (
        "Task-id set changed. If intentional, regenerate the golden file and "
        "record the reason in the commit message."
    )
    for task_id in TASK_IDS:
        assert current[task_id] == golden[task_id], (
            f"{task_id}: layout or protocol drift vs golden. If intentional "
            "(checkpoint-breaking!), regenerate the golden file."
        )
