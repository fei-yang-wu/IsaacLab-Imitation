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
    # Record the env-construction layout: the flat v2 configs finalize their
    # command-mode / whitelist / toggle derivation in resolve_late_overrides()
    # (legacy v0/v1 surfaces in _refresh_command_observation_terms), which is
    # what the env actually builds. The fixture is the fixed point either way.
    resolve = getattr(cfg, "resolve_late_overrides", None)
    if not callable(resolve):
        refresh = getattr(cfg, "_refresh_command_observation_terms", None)
        if callable(refresh):
            refresh()
    else:
        resolve()
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

    Parametrized over both ``-G1-v1`` (frozen since 2026-08-01) and
    ``-Latent-v0`` (kept for back-compat) -- they share
    ``_LATENT_STABLE_TASK_KWARGS``, but only
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
    """The env-construction resolution must not move any registered default."""
    for task_id in TASK_IDS:
        cfg = _load_env_cfg(task_id)
        resolve = getattr(cfg, "resolve_late_overrides", None)
        if not callable(resolve):
            # Legacy v0/v1 surfaces keep the refresh path.
            refresh = getattr(cfg, "_refresh_command_observation_terms", None)
            if not callable(refresh):
                continue
            resolve = refresh
        before = _layout(task_id)["groups"]
        resolve()
        after = {
            field.name: _group_terms(getattr(cfg.observations, field.name))
            for field in dataclasses.fields(cfg.observations)
            if getattr(cfg.observations, field.name) is not None
        }
        assert after == before, task_id


MACRO_TRIO = ("expert_motion", "expert_anchor_pos_b", "expert_anchor_ori_b")


def test_expert_window_and_goal_observation_pruning_knobs() -> None:
    """Opt-in observation-cost knobs on the default task.

    `expert_window_observation_terms` whitelists expert_window terms (must
    retain the active macro-state terms) and
    `enable_expert_goal_observations=False` drops the expert_goal group.
    Both default to no-change; that is covered by the recorded-default test.
    """
    cfg = _load_env_cfg("Isaac-Imitation-G1-v1")
    cfg.from_dict(
        {
            "expert_window_observation_terms": (
                "[expert_motion,expert_anchor_pos_b,expert_anchor_ori_b]"
            ),
            "enable_expert_goal_observations": False,
        }
    )
    assert _group_terms(cfg.observations.expert_window) == list(MACRO_TRIO)
    assert cfg.observations.expert_goal is None
    # The env-construction refresh path must be a fixed point.
    cfg._refresh_command_observation_terms()
    assert _group_terms(cfg.observations.expert_window) == list(MACRO_TRIO)
    assert cfg.observations.expert_goal is None
    # Kept window terms still follow the task's window/params sync.
    for name in MACRO_TRIO:
        term = getattr(cfg.observations.expert_window, name)
        assert term.params["past_steps"] == cfg.latent_patch_past_steps
        assert term.params["future_steps"] == cfg.latent_patch_future_steps

    # Turning the knobs back off restores the full declared layout.
    cfg.expert_window_observation_terms = None
    cfg.enable_expert_goal_observations = True
    cfg._refresh_command_observation_terms()
    default_cfg = _load_env_cfg("Isaac-Imitation-G1-v1")
    assert _group_terms(cfg.observations.expert_window) == _group_terms(
        default_cfg.observations.expert_window
    )
    assert cfg.observations.expert_goal is not None
    assert _group_terms(cfg.observations.expert_goal) == _group_terms(
        default_cfg.observations.expert_goal
    )


REWARD_INPUT_TERMS = ["expert_motion", "expert_anchor_pos_b", "expert_anchor_ori_b"]


def test_v2_parks_reward_input_group_with_opt_in_knob() -> None:
    """-G1-v2 parks the IPMD reward-estimation (IRL) stack by default.

    The reward_input group feeds only the IPMD reward estimator, so the
    single v2 env defaults `enable_reward_input_observations=False` and the
    env-construction resolution drops the group; v0/v1 keep it (pinned by the
    recorded-default test). The opt-in knob keeps the exact v0/v1 term list
    when enabled before construction.
    """
    v2_cfg = _load_env_cfg("Isaac-Imitation-G1-v2")
    assert v2_cfg.enable_reward_input_observations is False
    v2_cfg.resolve_late_overrides()
    assert getattr(v2_cfg.observations, "reward_input", None) is None

    v1_cfg = _load_env_cfg("Isaac-Imitation-G1-v1")
    assert v1_cfg.enable_reward_input_observations is True
    assert _group_terms(v1_cfg.observations.reward_input) == REWARD_INPUT_TERMS
    v0_cfg = _load_env_cfg("Isaac-Imitation-G1-v0")
    assert _group_terms(v0_cfg.observations.reward_input) == REWARD_INPUT_TERMS

    # The single v2 env keeps the v1 opt-in semantics: the env-construction
    # resolution drops the group at the default (parked) toggle and keeps the
    # exact v0/v1 term list when the knob is enabled before construction.
    cfg = _load_env_cfg("Isaac-Imitation-G1-v2")
    cfg.enable_reward_input_observations = True
    cfg.resolve_late_overrides()
    assert _group_terms(cfg.observations.reward_input) == REWARD_INPUT_TERMS
    for name in ("expert_anchor_pos_b", "expert_anchor_ori_b"):
        term = getattr(cfg.observations.reward_input, name)
        assert term.params["anchor_body_name"] == cfg.expert_anchor_body_name
    # Resolution stays a fixed point.
    cfg.resolve_late_overrides()
    assert _group_terms(cfg.observations.reward_input) == REWARD_INPUT_TERMS


def test_reward_estimation_agent_switch_defaults_parked() -> None:
    """The declarative agent switch mirrors the env knob: parked by default."""
    from isaaclab_imitation.tasks.manager_based.imitation.config.g1.agents.rlopt_ipmd_cfg import (  # noqa: E501
        REWARD_INPUT_KEYS,
        G1ImitationLatentSonicRLOptIPMDConfig,
        G1ImitationRLOptIPMDConfig,
    )

    coeff_names = (
        "reward_loss_coeff",
        "reward_l2_coeff",
        "reward_grad_penalty_coeff",
        "reward_logit_reg_coeff",
        "reward_param_weight_decay_coeff",
    )
    for cls in (G1ImitationRLOptIPMDConfig, G1ImitationLatentSonicRLOptIPMDConfig):
        agent_cfg = cls()
        assert agent_cfg.reward_estimation is False, cls.__name__
        assert agent_cfg.ipmd.reward_input_keys is None, cls.__name__
        for name in coeff_names:
            assert float(getattr(agent_cfg.ipmd, name)) == 0.0, (cls.__name__, name)
        # Opting in restores the historical vanilla estimator wiring, also
        # through the post-override sync path the train entrypoint uses.
        agent_cfg.reward_estimation = True
        agent_cfg.sync_input_keys()
        assert agent_cfg.ipmd.reward_input_keys == list(REWARD_INPUT_KEYS)
        assert float(agent_cfg.ipmd.reward_loss_coeff) == 1.0, cls.__name__
        for name in coeff_names[1:]:
            assert float(getattr(agent_cfg.ipmd, name)) == 0.0, (cls.__name__, name)


def test_expert_window_whitelist_must_cover_macro_state_terms() -> None:
    cfg = _load_env_cfg("Isaac-Imitation-G1-v1")
    with pytest.raises(ValueError, match="expert_macro_state_terms"):
        cfg.from_dict(
            {
                "expert_window_observation_terms": (
                    "[expert_anchor_pos_b,expert_anchor_ori_b]"
                )
            }
        )


def test_reset_start_mode_config_surface() -> None:
    """Every registered G1 task exposes a valid reset_start_mode default."""
    for task_id in TASK_IDS:
        cfg = _load_env_cfg(task_id)
        assert getattr(cfg, "reset_start_mode", None) in (
            "auto",
            "fixed",
            "random",
            "adaptive",
        ), f"{task_id}: bad reset_start_mode default {cfg.reset_start_mode!r}"

    # Explicit modes survive config validation and are normalized.
    for mode in ("fixed", "random", "adaptive"):
        cfg = _load_env_cfg("Isaac-Imitation-G1-v1")
        cfg.reset_start_mode = mode
        cfg.__post_init__()
        assert cfg.reset_start_mode == mode

    # Case/whitespace is normalized by the config validation.
    cfg = _load_env_cfg("Isaac-Imitation-G1-v1")
    cfg.reset_start_mode = " Adaptive "
    cfg.__post_init__()
    assert cfg.reset_start_mode == "adaptive"

    # Unknown modes fail loudly at config time.
    cfg = _load_env_cfg("Isaac-Imitation-G1-v1")
    cfg.reset_start_mode = "bogus"
    with pytest.raises(ValueError, match="reset_start_mode"):
        cfg.__post_init__()

    # Legacy full-trajectory variants keep the SONIC joint rank+frame path.
    full_trajectory_ids = [
        task_id
        for task_id in TASK_IDS
        if getattr(_load_env_cfg(task_id), "random_reset_full_trajectory", False)
    ]
    assert full_trajectory_ids, "expected at least one full-trajectory task"
    for task_id in full_trajectory_ids:
        cfg = _load_env_cfg(task_id)
        assert cfg.random_reset_full_trajectory is True
        assert cfg.reset_start_mode == "auto"
