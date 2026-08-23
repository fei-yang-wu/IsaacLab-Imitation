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
    """The task's config as the environment sees it: constructed, then resolved.

    Resolution is what turns a v2 config's declarations into its actual
    surface (preset selections collapsed, command terms narrowed, anchors
    stamped), so a layout recorded before it would not be the layout any run
    uses. Configs with no motion data resolve fine -- the dataset is only
    required when something actually loads it.
    """
    entry = gym.spec(task_id).kwargs["env_cfg_entry_point"]
    module_name, class_name = entry.split(":")
    cfg = getattr(importlib.import_module(module_name), class_name)()
    resolve = getattr(cfg, "resolve", None)
    if callable(resolve):
        resolve()
    else:
        # Legacy v0/v1 configs derive their command surface in their own step.
        refresh = getattr(cfg, "_refresh_command_observation_terms", None)
        if callable(refresh):
            refresh()
    return cfg


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
        "anchor_body": _anchor_body(cfg),
        "command": _command_surface(cfg),
        "groups": groups,
    }


def _anchor_body(cfg) -> str | None:
    """The body anchor-relative terms are expressed in.

    v2 keeps it on the command interface's reference channel, its single home;
    the frozen v0/v1 configs keep the flat environment field.
    """
    interface = getattr(cfg, "command_interface", None)
    if interface is not None:
        return interface.reference.anchor_body_name
    return getattr(cfg, "expert_anchor_body_name", None)


def _command_surface(cfg) -> dict:
    """The task's command protocol, whichever generation declares it.

    v2 declares one command interface; the frozen v0/v1 configs keep the flat
    environment knobs. Recording both keeps this fixture a real protocol gate
    for every registered task.
    """
    interface = getattr(cfg, "command_interface", None)
    if interface is None:
        return {
            "generation": "legacy",
            "latent_command_dim": getattr(cfg, "latent_command_dim", None),
            "latent_patch_past_steps": getattr(cfg, "latent_patch_past_steps", None),
            "latent_patch_future_steps": getattr(
                cfg, "latent_patch_future_steps", None
            ),
            "command_hold_steps": getattr(cfg, "command_hold_steps", None),
            "random_reset_step_min": getattr(cfg, "random_reset_step_min", None),
            "random_reset_step_max": getattr(cfg, "random_reset_step_max", None),
            "random_reset_full_trajectory": getattr(
                cfg, "random_reset_full_trajectory", None
            ),
        }
    selection = interface.reference.selection
    encoder = interface.encoder
    return {
        "generation": "interface",
        "actor_kind": interface.actor_kind(),
        "actor_source": interface.actor.source,
        "actor_components": list(interface.actor_components()),
        "actor_latent_dim": getattr(interface.actor, "dim", None),
        "actor_hold_steps": getattr(interface.actor, "hold_steps", None),
        "actor_horizon": getattr(interface.actor, "horizon", None),
        "critic_channels": list(interface.critic_channels),
        "critic_components": list(interface.critic_components()),
        "encoder_components": (
            list(encoder.components) if encoder is not None else None
        ),
        "encoder_window": (
            [int(encoder.past_steps), int(encoder.future_steps)]
            if encoder is not None
            else None
        ),
        "selection_schedule": selection.schedule,
        "selection_start_mode": selection.start_mode,
        "selection_start_frame": int(selection.start_frame),
        "selection_random_step_min": int(selection.random_step_min),
        "selection_random_step_max": int(selection.random_step_max),
        "selection_full_trajectory": bool(selection.full_trajectory),
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
        resolve = getattr(cfg, "resolve", None) or getattr(
            cfg, "_refresh_command_observation_terms", None
        )
        if not callable(resolve):
            continue
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
    assert getattr(v2_cfg.observations, "reward_input", None) is None

    v1_cfg = _load_env_cfg("Isaac-Imitation-G1-v1")
    assert v1_cfg.enable_reward_input_observations is True
    assert _group_terms(v1_cfg.observations.reward_input) == REWARD_INPUT_TERMS
    v0_cfg = _load_env_cfg("Isaac-Imitation-G1-v0")
    assert _group_terms(v0_cfg.observations.reward_input) == REWARD_INPUT_TERMS

    # The single v2 env keeps the v1 opt-in semantics: resolution drops the
    # group at the default (parked) toggle and keeps the exact v0/v1 term list
    # when the knob is enabled beforehand -- which is the only order that can
    # work, because resolution only ever removes from the declared surface.
    entry = gym.spec("Isaac-Imitation-G1-v2").kwargs["env_cfg_entry_point"]
    module_name, class_name = entry.split(":")
    cfg = getattr(importlib.import_module(module_name), class_name)()
    cfg.enable_reward_input_observations = True
    cfg.resolve()
    assert _group_terms(cfg.observations.reward_input) == REWARD_INPUT_TERMS
    for name in ("expert_anchor_pos_b", "expert_anchor_ori_b"):
        term = getattr(cfg.observations.reward_input, name)
        assert term.params["anchor_body_name"] == _anchor_body(cfg)
    # Resolution stays a fixed point.
    cfg.resolve()
    assert _group_terms(cfg.observations.reward_input) == REWARD_INPUT_TERMS


def test_v2_reward_input_terms_are_normalized_and_legacy_stay_raw() -> None:
    """The v2 reward_input group is the normalized [0, 1] feature map.

    v2 serves joint positions (soft-limit normalized), the relative anchor
    position, and the relative anchor rot6d through the shared normalization
    helpers; the byte-frozen v0/v1 surfaces keep the raw robot-motion terms
    that pair with `ImitationRLEnvLegacy`'s expert-side cache.
    """
    from isaaclab_imitation.tasks.manager_based.imitation import mdp

    entry = gym.spec("Isaac-Imitation-G1-v2").kwargs["env_cfg_entry_point"]
    module_name, class_name = entry.split(":")
    cfg = getattr(importlib.import_module(module_name), class_name)()
    cfg.enable_reward_input_observations = True
    cfg.resolve()
    group = cfg.observations.reward_input
    assert group.expert_motion.func is mdp.reward_robot_joint_pos
    assert group.expert_anchor_pos_b.func is mdp.reward_expert_anchor_pos_b
    assert group.expert_anchor_ori_b.func is mdp.reward_expert_anchor_ori_b

    v1_group = _load_env_cfg("Isaac-Imitation-G1-v1").observations.reward_input
    assert v1_group.expert_motion.func is mdp.robot_motion
    assert v1_group.expert_anchor_pos_b.func is mdp.expert_anchor_pos_b
    assert v1_group.expert_anchor_ori_b.func is mdp.expert_anchor_ori_b


def test_reward_input_normalization_helpers() -> None:
    """The shared [0, 1] feature map both reward-estimator sides import."""
    import torch

    from isaaclab_imitation.envs.reward_input_normalization import (
        identity_rot6d_unit,
        normalize_anchor_pos_unit,
        normalize_joint_pos_unit,
        normalize_rot6d_unit,
    )

    lower = torch.tensor([-1.0, 0.0])
    upper = torch.tensor([1.0, 2.0])
    q = torch.tensor([[-1.0, 0.0], [1.0, 2.0], [0.0, 1.0], [-3.0, 5.0]])
    normalized = normalize_joint_pos_unit(q, lower, upper)
    expected = torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5], [0.0, 1.0]])
    assert torch.allclose(normalized, expected)

    zero_error = normalize_anchor_pos_unit(torch.zeros(3))
    assert torch.allclose(zero_error, torch.full((3,), 0.5))
    saturated = normalize_anchor_pos_unit(torch.tensor([2.0, -2.0, 0.5]))
    assert torch.allclose(saturated, torch.tensor([1.0, 0.0, 0.75]))

    identity = identity_rot6d_unit(torch.device("cpu"))
    assert torch.allclose(identity, torch.tensor([1.0, 0.5, 0.5, 0.5, 1.0, 0.5]))
    assert torch.allclose(
        normalize_rot6d_unit(torch.tensor([-1.0, 1.0, 0.0])),
        torch.tensor([0.0, 1.0, 0.5]),
    )


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
        # The declarative regularizer knobs survive the post-override sync
        # (a direct agent.ipmd.* override would be zeroed by the switch) and
        # park again with the stack.
        agent_cfg.reward_estimation_grad_penalty_coeff = 0.5
        agent_cfg.reward_estimation_logit_reg_coeff = 0.01
        agent_cfg.sync_input_keys()
        assert float(agent_cfg.ipmd.reward_grad_penalty_coeff) == 0.5, cls.__name__
        assert float(agent_cfg.ipmd.reward_logit_reg_coeff) == 0.01, cls.__name__
        agent_cfg.reward_estimation = False
        agent_cfg.sync_input_keys()
        assert float(agent_cfg.ipmd.reward_grad_penalty_coeff) == 0.0, cls.__name__
        assert float(agent_cfg.ipmd.reward_logit_reg_coeff) == 0.0, cls.__name__


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


def _reference_selection(cfg):
    """The reference-selection surface of a task, whichever generation it is.

    v2 declares it on the command interface's reference channel; the frozen
    v0/v1 configs keep the flat environment fields.
    """
    interface = getattr(cfg, "command_interface", None)
    if interface is not None:
        selection = interface.reference.selection
        return selection.start_mode, bool(selection.full_trajectory)
    return (
        getattr(cfg, "reset_start_mode", None),
        bool(getattr(cfg, "random_reset_full_trajectory", False)),
    )


def test_reset_start_mode_config_surface() -> None:
    """Every registered G1 task exposes a valid start-frame policy."""
    for task_id in TASK_IDS:
        start_mode, _ = _reference_selection(_load_env_cfg(task_id))
        assert start_mode in ("auto", "fixed", "random", "adaptive"), (
            f"{task_id}: bad start mode default {start_mode!r}"
        )

    # Explicit modes survive validation and are normalized (legacy surface).
    for mode in ("fixed", "random", "adaptive"):
        cfg = _load_env_cfg("Isaac-Imitation-G1-v1")
        cfg.reset_start_mode = mode
        cfg.__post_init__()
        assert cfg.reset_start_mode == mode

    cfg = _load_env_cfg("Isaac-Imitation-G1-v1")
    cfg.reset_start_mode = " Adaptive "
    cfg.__post_init__()
    assert cfg.reset_start_mode == "adaptive"

    cfg = _load_env_cfg("Isaac-Imitation-G1-v1")
    cfg.reset_start_mode = "bogus"
    with pytest.raises(ValueError, match="reset_start_mode"):
        cfg.__post_init__()

    # ... and on the v2 surface, where the selection cfg validates itself.
    cfg = _load_env_cfg("Isaac-Imitation-G1-v2")
    cfg.command_interface.reference.selection.start_mode = " Adaptive "
    cfg.command_interface.reference.selection.resolve()
    assert cfg.command_interface.reference.selection.start_mode == "adaptive"

    cfg = _load_env_cfg("Isaac-Imitation-G1-v2")
    cfg.command_interface.reference.selection.start_mode = "bogus"
    with pytest.raises(ValueError, match="start_mode"):
        cfg.command_interface.reference.selection.resolve()

    # Full-trajectory variants keep the SONIC joint rank+frame path.
    full_trajectory_ids = [
        task_id
        for task_id in TASK_IDS
        if _reference_selection(_load_env_cfg(task_id))[1]
    ]
    assert full_trajectory_ids, "expected at least one full-trajectory task"
    for task_id in full_trajectory_ids:
        start_mode, full_trajectory = _reference_selection(_load_env_cfg(task_id))
        assert full_trajectory is True
        assert start_mode == "auto"
