"""Default observation-layout contract for every registered G1 imitation task.

The ordered term list of each observation group IS the checkpoint input
contract: reordering, adding, or removing a term silently changes the actor's
input layout and invalidates every checkpoint trained against it. The env
consolidation refactor (recipe x command-config) must therefore reproduce
these layouts exactly for every task id that predates it.

Regenerate the recorded default layout only when a layout change is
intentional:

    REGENERATE_G1_TASK_LAYOUT_DEFAULT=1 pixi run -e isaaclab pytest -q \
        source/isaaclab_imitation/tests/test_g1_task_layout_contract.py

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

DEFAULT_LAYOUT_PATH = Path(__file__).with_name("g1_task_layout_default.json")

TASK_IDS = sorted(
    spec_id for spec_id in gym.registry if spec_id.startswith("Isaac-Imitation-G1")
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


def test_registered_task_layouts_match_default() -> None:
    current = {task_id: _layout(task_id) for task_id in TASK_IDS}
    if os.environ.get("REGENERATE_G1_TASK_LAYOUT_DEFAULT") == "1":
        DEFAULT_LAYOUT_PATH.write_text(
            json.dumps(current, indent=2, sort_keys=True) + "\n"
        )
        pytest.skip(f"regenerated {DEFAULT_LAYOUT_PATH.name}")
    assert DEFAULT_LAYOUT_PATH.is_file(), (
        f"{DEFAULT_LAYOUT_PATH} missing; regenerate with "
        "REGENERATE_G1_TASK_LAYOUT_DEFAULT=1"
    )
    recorded = json.loads(DEFAULT_LAYOUT_PATH.read_text())
    assert sorted(current) == sorted(recorded), (
        "Task-id set changed. If intentional, regenerate the recorded default "
        "layout and record the reason in the commit message."
    )
    for task_id in TASK_IDS:
        assert current[task_id] == recorded[task_id], (
            f"{task_id}: layout or protocol drift vs the recorded default. If "
            "intentional (checkpoint-breaking!), regenerate it."
        )


ROOT_QPOS_TRIO = ("expert_motion_qpos", "expert_anchor_pos_b", "expert_anchor_ori_b")


@pytest.mark.parametrize(
    "task_id",
    ["Isaac-Imitation-G1-v1", "Isaac-Imitation-G1-Latent-v0"],
)
def test_stable_explicit_root_qpos_cli_override_layout(task_id: str) -> None:
    """Stable recipe + explicit root_qpos trio through the real CLI path.

    Isaac Lab 3.0's ``register_task`` applies plain ``env.*``/``agent.*``
    overrides with a direct ``setattr`` on the configs (no ``from_dict``
    round-trip) whenever no other Hydra args remain, so config-time pruning
    from ``__post_init__`` runs before the overrides land. ``ImitationRLEnv``
    re-derives the pruned command-term set at construction via
    ``_refresh_command_observation_terms``; this test exercises that exact
    sequence without booting a simulation.

    Parametrized over both ``-G1-v1`` (the default) and ``-Latent-v0`` (kept
    for back-compat) -- they share ``_LATENT_STABLE_TASK_KWARGS``, but only
    this exercises that the override path (not just the recorded layout)
    behaves identically for both ids.
    """
    import sys

    from isaaclab_tasks.utils import resolve_task_config

    overrides = [
        "env.command_mode=explicit",
        "env.command_observation_terms="
        "[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b]",
        "agent.ipmd.use_latent_command=false",
        "agent.command_components=[joint_qpos,root_pos,root_ori]",
    ]
    original_argv = sys.argv
    sys.argv = [original_argv[0]] + overrides
    try:
        env_cfg, agent_cfg = resolve_task_config(task_id, "rlopt_ipmd_cfg_entry_point")
    finally:
        sys.argv = original_argv

    # What ImitationRLEnv.__init__ does before any manager reads the config.
    refresh = getattr(env_cfg, "_refresh_command_observation_terms", None)
    assert callable(refresh)
    refresh()

    policy_terms = _group_terms(env_cfg.observations.policy)
    critic_terms = _group_terms(env_cfg.observations.critic)
    assert "latent_command" not in policy_terms
    assert "latent_command" not in critic_terms
    for name in ROOT_QPOS_TRIO:
        assert name in policy_terms, policy_terms
        assert name in critic_terms, critic_terms
    assert "expert_motion" not in policy_terms, policy_terms
    # Kept explicit command terms carry no observation noise (frozen protocol).
    for name in ROOT_QPOS_TRIO:
        assert getattr(env_cfg.observations.policy, name).noise is None

    # The agent contract selected by the same overrides must resolve against
    # the pruned groups (this is where the original failure surfaced as a
    # KeyError from the actor's input-key lookup).
    agent_cfg.sync_input_keys()
    for group_name, key in agent_cfg.policy.input_keys:
        assert group_name == "policy" and key in policy_terms, (group_name, key)
    for group_name, key in agent_cfg.value_function.input_keys:
        assert group_name == "critic" and key in critic_terms, (group_name, key)

    # The from_dict path (real Hydra runs with extra non-dotted args) must
    # produce the identical layout, including the raw "[a,b,c]" string form.
    cfg2 = type(env_cfg)()
    cfg2.from_dict(
        {
            "command_mode": "explicit",
            "command_observation_terms": (
                "[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b]"
            ),
        }
    )
    assert _group_terms(cfg2.observations.policy) == policy_terms
    assert _group_terms(cfg2.observations.critic) == critic_terms


def test_refresh_is_identity_for_default_tasks() -> None:
    """The env-construction refresh must not move any registered default."""
    for task_id in TASK_IDS:
        cfg = _load_env_cfg(task_id)
        refresh = getattr(cfg, "_refresh_command_observation_terms", None)
        if not callable(refresh):
            continue
        before = _layout(task_id)["groups"]
        refresh()
        after = {
            field.name: _group_terms(getattr(cfg.observations, field.name))
            for field in dataclasses.fields(cfg.observations)
            if getattr(cfg.observations, field.name) is not None
        }
        assert after == before, task_id
