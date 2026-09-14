"""Contract tests for the dual floating-hand Sharpa PhysX task."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import gymnasium as gym
import torch

import isaaclab_imitation.tasks  # noqa: F401
from isaaclab_imitation.assets.sharpa import ASSET_DIR
from isaaclab_imitation.tasks.manager_based.dexmanip.config.sharpa.sharpa_rsl_rl_cfg import (
    SharpaV2DPPORunnerCfg,
)
from isaaclab_imitation.tasks.manager_based.dexmanip.sharpa_actions import (
    SharpaResidualActionCfg,
    _compute_wrist_wrench_and_finger_target,
)
from isaaclab_imitation.tasks.manager_based.dexmanip.sharpa_command import (
    reference_advance_mask,
)
from isaaclab_imitation.tasks.manager_based.dexmanip.config.sharpa.sharpa_env_cfg import (
    SharpaContactSensorCfg,
    SharpaPhysicsCfg,
)
from isaaclab_imitation.tasks.manager_based.dexmanip import sharpa_mdp


def _movable_joint_names(path: Path) -> tuple[str, ...]:
    root = ET.parse(path).getroot()
    return tuple(
        str(joint.attrib["name"])
        for joint in root.findall("joint")
        if joint.attrib.get("type") != "fixed"
    )


def test_sharpa_task_and_assets_are_registered() -> None:
    spec = gym.spec("Isaac-Sharpa-V2D-PhysX-v0")
    assert spec.entry_point == "isaaclab.envs:ManagerBasedRLEnv"
    assert spec.kwargs["env_cfg_entry_point"].endswith(":SharpaV2DEnvCfg")
    assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(":SharpaV2DPPORunnerCfg")
    for side in ("left", "right"):
        path = ASSET_DIR / "urdfs" / "sharpawave" / f"{side}_sharpa_wave.urdf"
        assert len(_movable_joint_names(path)) == 22
    assert (ASSET_DIR / "meshes" / "LICENSE").is_file()
    assert (ASSET_DIR / "meshes" / "NOTICE").is_file()


def test_sharpa_action_and_ppo_source_contract() -> None:
    action = SharpaResidualActionCfg(asset_name="right_robot")
    assert action.wrist_position_scale == 0.05
    assert action.wrist_orientation_scale == 0.15
    assert action.finger_joint_scale == 0.15
    assert action.ema_factor == 0.3
    assert action.max_force == action.max_torque == 60.0

    cfg = SharpaV2DPPORunnerCfg()
    assert cfg.num_steps_per_env == 24
    assert cfg.max_iterations == 20_000
    assert cfg.policy.actor_hidden_dims == [1024, 512, 256, 128]
    assert cfg.policy.critic_hidden_dims == [1024, 512, 256, 128]
    assert cfg.policy.activation == "elu"
    assert cfg.actor.hidden_dims == [1024, 512, 256, 128]
    assert cfg.critic.hidden_dims == [1024, 512, 256, 128]
    assert cfg.actor.distribution_cfg.init_std == 0.1
    assert cfg.algorithm.num_learning_epochs == 5
    assert cfg.algorithm.num_mini_batches == 4
    assert cfg.algorithm.learning_rate == 1.0e-3
    assert cfg.algorithm.gamma == 0.99
    assert cfg.algorithm.lam == 0.95
    assert cfg.algorithm.clip_param == 0.1
    assert cfg.algorithm.entropy_coef == 0.001
    assert cfg.algorithm.desired_kl == 0.005


def test_sharpa_exposes_matching_physx_and_newton_presets() -> None:
    physics = SharpaPhysicsCfg()
    assert type(physics.default).__name__ == "PhysxCfg"
    assert type(physics.physx).__name__ == "PhysxCfg"
    assert type(physics.newton_mjwarp).__name__ == "NewtonCfg"
    sensors = SharpaContactSensorCfg()
    assert type(sensors.default).__name__ == "ContactSensorCfg"
    assert type(sensors.physx).__name__ == "ContactSensorCfg"
    assert type(sensors.newton_mjwarp).__module__.startswith("isaaclab_newton")


def test_reference_advances_immediately_by_default_and_legacy_mode_holds() -> None:
    timestep = torch.tensor([0, 4])
    horizon = torch.tensor([8, 5])
    first_step = torch.tensor([1, 1])
    corrected = reference_advance_mask(
        timestep,
        horizon,
        first_step,
        legacy_reset_hold=False,
        hold_steps=20,
    )
    legacy = reference_advance_mask(
        timestep,
        horizon,
        first_step,
        legacy_reset_hold=True,
        hold_steps=20,
    )
    assert corrected.tolist() == [True, False]
    assert legacy.tolist() == [False, False]
    after_hold = reference_advance_mask(
        timestep,
        horizon,
        torch.tensor([21, 21]),
        legacy_reset_hold=True,
        hold_steps=20,
    )
    assert after_hold.tolist() == [True, False]
    # The released rule releases the hold when the counter reaches the
    # decay-step count, not one step later.
    at_hold_boundary = reference_advance_mask(
        timestep,
        horizon,
        torch.tensor([20, 20]),
        legacy_reset_hold=True,
        hold_steps=20,
    )
    assert at_hold_boundary.tolist() == [True, False]


def test_sharpa_wrist_controller_uses_isaac_xyzw_order() -> None:
    """Identity target must not generate a 180-degree correction torque."""

    force, torque, finger_target = _compute_wrist_wrench_and_finger_target(
        wrist_position=torch.zeros(1, 3),
        wrist_quat_xyzw=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        wrist_link_vel_w=torch.zeros(1, 6),
        wrist_link_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        wrist_com_pos_b=torch.zeros(1, 3),
        wrist_pose_target=torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
        finger_joint_target=torch.zeros(1, 1),
        gravity_comp_e=torch.zeros(1, 6),
        joint_damping=torch.ones(1, 1),
        joint_vel=torch.zeros(1, 1),
        joint_effort_limits=torch.ones(1, 1),
        joint_stiffness=torch.ones(1, 1),
        joint_pos=torch.zeros(1, 1),
        linear_stiffness=50.0,
        linear_damping=10.0,
        angular_stiffness=12.0,
        angular_damping=0.1,
        max_force=60.0,
        max_torque=60.0,
    )
    torch.testing.assert_close(force, torch.zeros_like(force))
    torch.testing.assert_close(torque, torch.zeros_like(torque))
    torch.testing.assert_close(finger_target, torch.zeros_like(finger_target))


def _objective_env(
    object_position: torch.Tensor, lift: torch.Tensor
) -> SimpleNamespace:
    command = SimpleNamespace(
        object_position_e=object_position[:, None],
        object_orientation_e=torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0]]] * len(object_position)
        ),
        object_body_position_command_e=object_position[:, None].clone(),
        object_body_wxyz_command_e=torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0]]] * len(object_position)
        ),
        object_goal_pose_e=torch.cat(
            (
                object_position,
                torch.tensor([[1.0, 0.0, 0.0, 0.0]] * len(object_position)),
            ),
            dim=-1,
        ),
        object_lift_height=lift,
    )
    return SimpleNamespace(
        num_envs=len(object_position),
        device=object_position.device,
        command_manager=SimpleNamespace(get_term=lambda _name: command),
    )


def test_object_objective_terms_are_identity_maximized_and_thresholded() -> None:
    env = _objective_env(
        torch.tensor([[0.0, 0.0, 0.2], [0.2, 0.0, 0.2]]),
        torch.tensor([0.1, 0.02]),
    )
    torch.testing.assert_close(
        sharpa_mdp.object_pose_tracking_exp(env), torch.tensor([1.0, 1.0])
    )
    env.command_manager.get_term("sharpa_reference").object_goal_pose_e[1, :3] = 0.0
    goal = sharpa_mdp.object_goal_tracking_exp(env)
    assert float(goal[0]) == 1.0
    assert float(goal[1]) < 1.0
    torch.testing.assert_close(
        sharpa_mdp.object_lift_progress(env, target_height=0.1),
        torch.tensor([1.0, 0.2]),
    )
    torch.testing.assert_close(
        sharpa_mdp.object_task_success(env), torch.tensor([1.0, 0.0])
    )


def test_live_contact_reduces_history_and_body_axes() -> None:
    forces = torch.zeros(2, 2, 3, 3)
    forces[0, 1, 2, 0] = 0.2
    env = SimpleNamespace(
        num_envs=2,
        device=forces.device,
        scene=SimpleNamespace(
            sensors={
                "left_hand_object_contacts": SimpleNamespace(
                    data=SimpleNamespace(net_forces_w=forces)
                )
            }
        ),
    )
    torch.testing.assert_close(
        sharpa_mdp.live_contact(env, "left"), torch.tensor([True, False])
    )


def test_sharpa_rlopt_config_matches_the_rsl_rl_recipe() -> None:
    """The RLOpt translation states the same recipe as the RSL-RL config."""

    import math

    from isaaclab_imitation.tasks.manager_based.dexmanip.config.sharpa import (
        sharpa_agents,
    )
    from isaaclab_imitation.tasks.manager_based.dexmanip.curriculum import (
        DEFAULT_NUM_STEPS_PER_ENV,
    )

    spec = gym.spec("Isaac-Sharpa-V2D-PhysX-v0")
    assert spec.kwargs["rlopt_cfg_entry_point"].endswith(":SharpaRLOptPPOConfig")
    assert spec.kwargs["rlopt_ppo_cfg_entry_point"].endswith(":SharpaRLOptPPOConfig")

    rsl = SharpaV2DPPORunnerCfg()
    cfg = sharpa_agents.SharpaRLOptPPOConfig()
    batch = sharpa_agents.SHARPA_REFERENCE_NUM_ENVS * rsl.num_steps_per_env

    # The curriculum converts PPO updates to control steps with this horizon.
    assert cfg.collector.frames_per_batch == rsl.num_steps_per_env
    assert rsl.num_steps_per_env == DEFAULT_NUM_STEPS_PER_ENV
    assert (
        sharpa_agents.SHARPA_CURRICULUM_NUM_STEPS_PER_ENV == DEFAULT_NUM_STEPS_PER_ENV
    )

    assert cfg.collector.init_random_frames == 0
    assert cfg.collector.total_frames == rsl.max_iterations * batch
    assert cfg.replay_buffer.size == batch
    assert cfg.loss.epochs == rsl.algorithm.num_learning_epochs
    assert cfg.loss.mini_batch_size == batch // rsl.algorithm.num_mini_batches
    assert cfg.mini_batches_per_rollout == rsl.algorithm.num_mini_batches
    assert cfg.loss.loss_critic_type == "l2"
    assert cfg.loss.gamma == rsl.algorithm.gamma
    assert cfg.ppo.gae_lambda == rsl.algorithm.lam
    assert cfg.ppo.clip_epsilon == rsl.algorithm.clip_param
    assert cfg.ppo.clip_value == rsl.algorithm.use_clipped_value_loss
    assert cfg.ppo.critic_coeff == rsl.algorithm.value_loss_coef
    assert cfg.ppo.entropy_coeff == rsl.algorithm.entropy_coef
    assert cfg.ppo.normalize_advantage and cfg.ppo.normalize_advantage_global
    assert cfg.ppo.truncation_bootstrap == "rsl_rl"
    assert cfg.ppo.log_std_init == math.log(rsl.actor.distribution_cfg.init_std)
    assert cfg.optim.optimizer == "adam"
    assert cfg.optim.lr == rsl.algorithm.learning_rate
    assert cfg.optim.max_grad_norm == rsl.algorithm.max_grad_norm
    assert cfg.optim.scheduler == "adaptive"
    assert cfg.optim.desired_kl == rsl.algorithm.desired_kl
    assert cfg.optim.lr_adaptation_factor == 1.5
    assert cfg.optim.kl_adapt_step == "update"
    assert (cfg.optim.min_lr, cfg.optim.max_lr) == (1.0e-5, 1.0e-2)
    assert cfg.policy.num_cells == rsl.actor.hidden_dims
    assert cfg.value_function.num_cells == rsl.critic.hidden_dims
    assert cfg.policy.activation_fn == cfg.value_function.activation_fn == "elu"
    assert cfg.policy.normalize_input and cfg.value_function.normalize_input
    assert cfg.policy.input_keys is None and cfg.value_function.input_keys is None
    assert cfg.save_interval == rsl.save_interval * batch
    assert cfg.save_interval_iterations == rsl.save_interval
    assert cfg.trainer.log_interval == cfg.log_interval_iterations * batch
    assert cfg.logger.backend == ""
