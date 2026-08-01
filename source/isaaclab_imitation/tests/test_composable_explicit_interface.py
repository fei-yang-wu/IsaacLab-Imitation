from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from isaaclab_imitation.envs import imitation_rl_env_legacy
from isaaclab_imitation.envs.imitation_rl_env_legacy import ImitationRLEnvLegacy
from isaaclab_imitation.tasks.manager_based.imitation.config.g1.agents.rlopt_ipmd_cfg import (
    G1ImitationRLOptIPMDConfig,
    command_component_input_keys,
    normalize_command_components,
)
from isaaclab_imitation.tasks.manager_based.imitation.config.g1.imitation_g1_env_cfg import (
    ImitationG1LafanTrackEnvCfg,
)


def test_command_components_are_alias_normalized_and_canonically_ordered() -> None:
    components = normalize_command_components(
        ["root_orientation", "qpos", "root_position"]
    )
    assert components == ("joint_qpos", "root_pos", "root_ori")
    assert command_component_input_keys(components) == [
        ("policy", "expert_motion_qpos"),
        ("policy", "expert_anchor_pos_b"),
        ("policy", "expert_anchor_ori_b"),
    ]


def test_composed_root_keypoint_pose_builds_actor_and_critic_contracts() -> None:
    cfg = G1ImitationRLOptIPMDConfig()
    cfg.command_components = [
        "root_ori",
        "keypoint_pos",
        "root_pos",
        "keypoint_ori",
    ]
    cfg.command_spec_name = "root_points5_pose"
    cfg.sync_input_keys()

    assert cfg.command_components == [
        "keypoint_pos",
        "keypoint_ori",
        "root_pos",
        "root_ori",
    ]
    assert cfg.policy.input_keys[:4] == [
        ("policy", "expert_keypoint_pos_b"),
        ("policy", "expert_keypoint_ori_b"),
        ("policy", "expert_anchor_pos_b"),
        ("policy", "expert_anchor_ori_b"),
    ]
    assert cfg.value_function is not None
    assert cfg.value_function.input_keys[:4] == [
        ("critic", "expert_keypoint_pos_b"),
        ("critic", "expert_keypoint_ori_b"),
        ("critic", "expert_anchor_pos_b"),
        ("critic", "expert_anchor_ori_b"),
    ]


def test_composed_explicit_interface_rejects_latent_tracker_mode() -> None:
    cfg = G1ImitationRLOptIPMDConfig()
    cfg.command_components = ["joint_qpos", "root_pos", "root_ori"]
    cfg.ipmd.use_latent_command = True
    with pytest.raises(ValueError, match="cannot be combined"):
        cfg.sync_input_keys()


def test_g1_observation_config_registers_keypoint_orientations() -> None:
    cfg = ImitationG1LafanTrackEnvCfg()
    assert cfg.observations.policy.expert_keypoint_ori_b is not None
    assert cfg.observations.critic.expert_keypoint_ori_b is not None
    assert cfg.observations.expert_window.expert_keypoint_ori_b is not None


def _fake_achieved_env(
    *,
    slices: dict[str, tuple[int, int]],
    state: torch.Tensor,
    state_history: torch.Tensor,
) -> SimpleNamespace:
    batch_size = int(state.shape[0])
    batch = TensorDict(
        {
            "hl": TensorDict(
                {
                    "state": state.clone(),
                    "state_history": state_history.clone(),
                    "future_window": torch.full(
                        (batch_size, 10, int(state.shape[-1])), 9.0
                    ),
                    "target": torch.full((batch_size, int(state.shape[-1])), 8.0),
                },
                batch_size=[batch_size],
            )
        },
        batch_size=[batch_size],
    )
    joint_pos = torch.arange(batch_size * 29, dtype=torch.float32).reshape(
        batch_size, 29
    )
    joint_vel = joint_pos + 100.0
    env = SimpleNamespace(
        device=torch.device("cpu"),
        num_envs=batch_size,
        robot=SimpleNamespace(
            data=SimpleNamespace(
                joint_pos=SimpleNamespace(torch=joint_pos),
                joint_vel=SimpleNamespace(torch=joint_vel),
            )
        ),
        _command_ee_body_names=(),
        _command_keypoint_body_names=(),
        _expert_anchor_body_name="torso_link",
    )
    env.current_expert_macro_transition_batch = lambda *args, **kwargs: batch
    env.expert_macro_feature_slices = lambda *args, **kwargs: slices
    return env


def test_achieved_macro_state_supports_root_qpos_components() -> None:
    slices = {
        "expert_motion_qpos": (0, 29),
        "expert_anchor_pos_b": (29, 32),
        "expert_anchor_ori_b": (32, 38),
    }
    state = torch.full((2, 38), -1.0)
    history = torch.full((2, 10, 38), -2.0)
    env = _fake_achieved_env(slices=slices, state=state, state_history=history)

    result = ImitationRLEnvLegacy.current_achieved_macro_transition_batch(
        env, horizon_steps=10, state_history_steps=9
    )

    expected_qpos = env.robot.data.joint_pos.torch
    assert torch.equal(result["hl", "state"][:, :29], expected_qpos)
    assert torch.equal(result["hl", "state"][:, 29:], state[:, 29:])
    assert torch.equal(result["hl", "state_history"][:, -1, :29], expected_qpos)
    assert torch.equal(result["hl", "state_history"][:, :-1], history[:, :-1])
    assert torch.equal(result["hl", "future_window"], torch.full((2, 10, 38), 9.0))
    assert torch.equal(result["hl", "target"], torch.full((2, 38), 8.0))


def test_achieved_macro_state_supports_root_keypoint_pose_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slices = {
        "expert_keypoint_pos_b": (0, 15),
        "expert_keypoint_ori_b": (15, 45),
        "expert_anchor_pos_b": (45, 48),
        "expert_anchor_ori_b": (48, 54),
    }
    state = torch.full((2, 54), -1.0)
    history = torch.full((2, 10, 54), -2.0)
    env = _fake_achieved_env(slices=slices, state=state, state_history=history)
    env._command_keypoint_body_names = tuple(f"body_{index}" for index in range(5))
    env._get_robot_body_ids_by_name_fast = lambda names: torch.arange(len(names))
    achieved_pos = torch.arange(30, dtype=torch.float32).reshape(2, 5, 3)
    achieved_quat = torch.zeros((2, 5, 4), dtype=torch.float32)
    env._get_robot_body_state_in_anchor_frame_fast = lambda body_ids, anchor_name: (
        achieved_pos,
        achieved_quat,
    )
    achieved_ori = torch.arange(60, dtype=torch.float32).reshape(2, 5, 6)
    monkeypatch.setattr(
        imitation_rl_env_legacy,
        "_get_mdp_compiled_module",
        lambda: SimpleNamespace(quat_to_rot6d_flat=lambda quat: achieved_ori),
    )

    result = ImitationRLEnvLegacy.current_achieved_macro_transition_batch(
        env, horizon_steps=10, state_history_steps=9
    )

    assert torch.equal(result["hl", "state"][:, :15], achieved_pos.reshape(2, 15))
    assert torch.equal(result["hl", "state"][:, 15:45], achieved_ori.reshape(2, 30))
    assert torch.equal(result["hl", "state"][:, 45:], state[:, 45:])
    assert torch.equal(
        result["hl", "state_history"][:, -1, :45],
        torch.cat([achieved_pos.reshape(2, 15), achieved_ori.reshape(2, 30)], dim=-1),
    )
