"""Contract tests for the Vega-Wuji Newton dexterous task."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import gymnasium as gym
from iltools.core import ContactSequence, DexterousReference, ScenePhysics
import numpy as np
import pytest
import torch

import isaaclab_imitation.tasks  # noqa: F401
from isaaclab_imitation.envs.vega_wuji_imitation_env import (
    _add_contact_sensors,
    _apply_scene_inertial_contract,
    _mjcf_actuator_joint_names,
    _pack_references,
    _scene_spec,
    _sync_motion_command_cfg,
    _validate_reference_timing,
    _validate_robot_only_mjcf,
    _validate_wrist_frame_cfg,
)
from isaaclab_imitation.tasks.manager_based.dexmanip import mdp
from isaaclab_imitation.tasks.manager_based.dexmanip.actions import (
    VegaWujiReferenceResidualAction,
)
from isaaclab_imitation.tasks.manager_based.dexmanip.command import (
    VegaWujiReferenceCommand,
    dominant_contact_object_indices,
    match_newton_counterpart_columns,
)
from isaaclab_imitation.tasks.manager_based.dexmanip.curriculum import (
    DEFAULT_TIMESTEP_SCHEDULE,
    DEFAULT_VIRTUAL_OBJECT_CONTROL_SCALE_FACTOR,
)
from isaaclab_imitation.tasks.manager_based.dexmanip.newton_scene_material import (
    configure_scene_material,
)
from isaaclab_imitation.tasks.manager_based.dexmanip.config.vega_wuji.vega_wuji_agents import (
    VEGA_WUJI_POLICY_INPUT_KEYS,
    VegaWujiRLOptPPOConfig,
)
from isaaclab_imitation.tasks.manager_based.dexmanip.config.vega_wuji.vega_wuji_env_cfg import (
    VegaWujiImitationEnvCfg,
)


_WRIST_FRAME_NAMES = {"right": "right_palm", "left": "left_palm"}


def _poses(*shape: int) -> np.ndarray:
    value = np.zeros((*shape, 7), dtype=np.float32)
    value[..., 3] = 1.0
    return value


def _reference(
    *,
    sequence_id: str,
    frame_count: int,
    qpos_offset: float = 0.0,
) -> DexterousReference:
    side_count = 2
    slot_count = 2
    active = np.zeros((frame_count, side_count, slot_count), dtype=np.bool_)
    active[..., 0] = True
    object_indices = np.full(active.shape, -1, dtype=np.int32)
    object_indices[..., 0] = 0
    link_normals = np.zeros((frame_count, side_count, slot_count, 3), dtype=np.float32)
    object_normals = np.zeros_like(link_normals)
    link_normals[..., 0, 2] = 1.0
    object_normals[..., 0, 2] = -1.0
    contacts = ContactSequence(
        hand_sides=("right", "left"),
        link_names=np.asarray((("r_tip", ""), ("l_tip", ""))),
        link_positions_w=np.zeros_like(link_normals),
        link_normals_w=link_normals,
        object_positions_w=np.zeros_like(link_normals),
        object_normals_w=object_normals,
        object_indices=object_indices,
        active=active,
    )
    qpos = np.stack(
        (
            np.arange(frame_count, dtype=np.float32) + qpos_offset,
            np.arange(frame_count, dtype=np.float32) + 10.0 + qpos_offset,
        ),
        axis=-1,
    )
    return DexterousReference(
        sequence_id=sequence_id,
        robot_name="vega_wuji",
        fps=20.0,
        joint_names=("joint_b", "joint_a"),
        qpos=qpos,
        qvel=np.zeros_like(qpos),
        fixed_root_pose_w=np.asarray((0.0, 0.0, 0.19, 1.0, 0.0, 0.0, 0.0)),
        left_wrist_pose_w=_poses(frame_count),
        right_wrist_pose_w=_poses(frame_count),
        left_wrist_frame_name=_WRIST_FRAME_NAMES["left"],
        right_wrist_frame_name=_WRIST_FRAME_NAMES["right"],
        object_names=("cube",),
        object_poses_w=_poses(frame_count, 1),
        object_asset_paths=("cube.usd",),
        object_scales=np.asarray(((1.0, 1.0, 1.0),), dtype=np.float32),
        object_radii=np.asarray((0.1,), dtype=np.float32),
        left_hand_frame_names=("l_tip",),
        left_hand_frame_poses_w=_poses(frame_count, 1),
        right_hand_frame_names=("r_tip",),
        right_hand_frame_poses_w=_poses(frame_count, 1),
        contacts=contacts,
    )


def test_vega_wuji_task_is_registered_for_rlopt_only() -> None:
    spec = gym.spec("Isaac-Imitation-Vega-Wuji-v0")
    assert spec.entry_point == "isaaclab_imitation.envs:VegaWujiImitationEnv"
    assert spec.kwargs["env_cfg_entry_point"].endswith(":VegaWujiImitationEnvCfg")
    assert spec.kwargs["rlopt_cfg_entry_point"].endswith(":VegaWujiRLOptPPOConfig")
    assert spec.kwargs["rlopt_ppo_cfg_entry_point"].endswith(":VegaWujiRLOptPPOConfig")
    assert "rsl_rl_cfg_entry_point" not in spec.kwargs


def test_config_matches_full_joint_control_and_newton_contract() -> None:
    cfg = VegaWujiImitationEnvCfg()

    assert cfg.sim.dt == 0.01
    assert cfg.decimation == 5
    assert cfg.scene.num_envs == 4096
    assert cfg.scene.env_spacing == 1.5
    assert cfg.scene.replicate_physics is True
    assert cfg.scene.filter_collisions is False
    assert type(cfg.sim.physics.default).__name__ == "NewtonCfg"
    assert cfg.sim.physics.default.default_shape_cfg.margin == 0.0
    assert cfg.sim.physics.default.default_shape_cfg.gap == 0.0
    assert not hasattr(cfg.sim.physics, "physx")
    assert cfg.scene.robot.default.spawn.fix_base is True
    assert cfg.reference_fps == 20.0
    assert cfg.reference_path == ""
    assert cfg.reference_motion_index == 0
    assert cfg.require_training_qualified_reference is True
    assert cfg.commands.motion.fixed_motion_index == 0
    hand_material = cfg.events.wuji_hand_material
    assert hand_material.mode == "startup"
    assert hand_material.params == {
        "static_friction_range": (2.0, 2.01),
        "dynamic_friction_range": (2.0, 2.01),
        "restitution_range": (0.0, 0.0),
        "num_buckets": 64,
    }
    assert cfg.events.scene_material.mode == "startup"
    assert cfg.events.scene_material.func is configure_scene_material
    assert cfg.events.scene_material.params == {}

    action = cfg.actions.joint_residual
    assert action.class_type is VegaWujiReferenceResidualAction
    assert action.command_name == "motion"
    assert action.lift_joint_name == "Lift"
    assert action.right_arm_joint_expr == "R_arm_j[1-7]"
    assert action.left_arm_joint_expr == "L_arm_j[1-7]"
    assert action.right_finger_joint_expr == "r_.*"
    assert action.left_finger_joint_expr == "l_.*"
    assert action.lift_joint_scale == 0.05
    assert action.revolute_joint_scale == 0.15
    assert action.ema_factor == 0.3
    command = cfg.commands.motion
    assert command.class_type is VegaWujiReferenceCommand
    assert command.right_wrist_body_name == "r_mount"
    assert command.left_wrist_body_name == "l_mount"
    assert command.right_wrist_reference_frame_name == "right_palm"
    assert command.left_wrist_reference_frame_name == "left_palm"
    assert cfg.commands.motion.friction_coefficient == pytest.approx(0.1)
    assert cfg.commands.motion.reset_to_first_frame_prob == pytest.approx(0.1)
    assert cfg.commands.motion.randomize_reset_finger_openness is False
    assert cfg.commands.motion.force_unassisted_object_control is False
    assert cfg.observations.policy.concatenate_terms is False
    assert cfg.observations.policy.processed_right_actions.params == {
        "action_name": "joint_residual",
        "side": "right",
    }
    assert cfg.observations.policy.processed_left_actions.params == {
        "action_name": "joint_residual",
        "side": "left",
    }
    assert cfg.rewards.action_norm.params == {"action_name": "joint_residual"}
    assert cfg.rewards.robot_joint_position_tracking_exp.weight == pytest.approx(5.0)
    assert cfg.rewards.robot_joint_velocity_tracking_exp.weight == pytest.approx(0.5)
    assert cfg.rewards.object_position_tracking_exp.weight == pytest.approx(1.0)
    assert cfg.rewards.object_orientation_tracking_exp.weight == pytest.approx(1.0)
    assert cfg.rewards.object_twist_tracking_exp.weight == pytest.approx(0.25)
    assert cfg.rewards.object_twist_tracking_exp.params == {
        "command_name": "motion",
        "var": 1.0,
    }
    assert cfg.rewards.joint_position_limit.weight == pytest.approx(-1.0e-2)
    assert cfg.terminations.robot_joint_away_from_trajectory.params[
        "normalized_position_threshold"
    ] == pytest.approx(0.35)


def test_late_motion_selection_override_reaches_command_cfg() -> None:
    cfg = VegaWujiImitationEnvCfg()
    cfg.randomize_reference = False
    cfg.reference_motion_index = 3

    _sync_motion_command_cfg(cfg)

    assert cfg.commands.motion.randomize_reference is False
    assert cfg.commands.motion.fixed_motion_index == 3


def test_reference_timing_is_exactly_one_frame_per_control_transition() -> None:
    cfg = VegaWujiImitationEnvCfg()

    assert _validate_reference_timing(cfg) == pytest.approx(0.05)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("reference_fps", 25.0),
        ("sim.dt", 0.02),
        ("decimation", 4),
    ),
)
def test_reference_timing_rejects_post_hydra_rate_drift(
    field: str,
    value: float,
) -> None:
    cfg = VegaWujiImitationEnvCfg()
    if field == "sim.dt":
        cfg.sim.dt = value
    else:
        setattr(cfg, field, value)

    with pytest.raises(ValueError, match="one ILTools Reference frame"):
        _validate_reference_timing(cfg)


def test_wrist_frame_config_rejects_command_reference_divergence() -> None:
    cfg = VegaWujiImitationEnvCfg()
    _validate_wrist_frame_cfg(cfg)

    cfg.commands.motion.right_wrist_body_name = "R_ee"
    with pytest.raises(ValueError, match="command right wrist body.*r_mount"):
        _validate_wrist_frame_cfg(cfg)


def test_actuator_groups_match_mjcf_position_drives() -> None:
    cfg = VegaWujiImitationEnvCfg().scene.robot.default
    assert cfg.actuators["lift"].stiffness == 2000.0
    assert cfg.actuators["lift"].damping == 100.0
    assert cfg.actuators["torso"].effort_limit_sim == 500.0
    assert cfg.actuators["head"].stiffness == 120.0
    assert cfg.actuators["arms"].effort_limit_sim == 300.0
    hand = cfg.actuators["hands"]
    assert hand.stiffness["[lr]_thumb_cmc_abd"] == pytest.approx(0.6858601064)
    assert hand.damping["[lr]_index_finger_dip"] == pytest.approx(0.0082402121)
    assert (
        hand.effort_limit_sim[
            "[lr]_(index_finger|middle_finger|ring_finger|pinky)_mcp_flex"
        ]
        == 2.0
    )


def test_observation_reward_termination_and_curriculum_terms_match_released_recipe() -> (
    None
):
    cfg = VegaWujiImitationEnvCfg()
    observation_names = tuple(name for _, name in VEGA_WUJI_POLICY_INPUT_KEYS)
    assert all(hasattr(cfg.observations.policy, name) for name in observation_names)

    assert cfg.rewards.action_rate.weight == -5.0e-3
    assert cfg.rewards.action_norm.weight == -2.0e-3
    assert cfg.rewards.contact_wrench_support_reward.weight == 10.0
    assert cfg.rewards.unintended_contact_penalty.weight == -10.0
    assert cfg.rewards.missed_contact_penalty.weight == -1.0
    assert cfg.rewards.termination_penalty.weight == -100.0
    assert cfg.terminations.wrist_away_from_trajectory.params["threshold"] == 0.2
    assert (
        cfg.terminations.object_away_from_trajectory.params["orientation_threshold"]
        == 0.7
    )
    curriculum = cfg.curriculum.fixed_timestep.params
    assert curriculum["timestep_schedule"] == list(DEFAULT_TIMESTEP_SCHEDULE)
    assert curriculum["virtual_object_control_scale_factor"] == list(
        DEFAULT_VIRTUAL_OBJECT_CONTROL_SCALE_FACTOR
    )


def test_vega_wuji_agent_uses_all_named_policy_observations() -> None:
    cfg = VegaWujiRLOptPPOConfig()
    assert cfg.policy.input_keys == VEGA_WUJI_POLICY_INPUT_KEYS
    assert getattr(cfg.value_function, "input_keys") == VEGA_WUJI_POLICY_INPUT_KEYS
    assert cfg.policy.num_cells == [1024, 512, 256, 128]
    assert getattr(cfg.value_function, "num_cells") == [1024, 512, 256, 128]
    assert cfg.policy.activation_fn == "elu"
    assert getattr(cfg.value_function, "activation_fn") == "elu"
    assert cfg.policy.normalize_input is True
    assert getattr(cfg.value_function, "normalize_input") is True
    assert cfg.ppo.clip_log_std is True
    assert cfg.ppo.log_std_init == pytest.approx(math.log(0.1))
    assert cfg.ppo.clip_epsilon == pytest.approx(0.1)
    assert cfg.ppo.entropy_coeff == pytest.approx(0.001)
    assert cfg.ppo.gae_lambda == pytest.approx(0.95)
    assert cfg.optim.optimizer == "adam"
    assert cfg.optim.lr == pytest.approx(1.0e-3)
    assert cfg.optim.desired_kl == pytest.approx(0.005)
    assert cfg.collector.frames_per_batch == 24
    assert cfg.logger.backend == ""
    assert cfg.collector.total_frames == 20_000 * 4096 * 24
    assert cfg.save_interval == 200 * 4096 * 24


def test_virtual_object_control_angular_gains_stay_inertia_stable() -> None:
    """Keep the object controller's rotational loop inside its stability limit.

    The angular gains act on an object inertia far below its mass, so the
    rotational loop reaches the discrete stability limit before the linear one.
    ``k dt^2 / I`` must stay under 4 and ``d dt / I`` under 2 at the task's
    20 Hz control rate. A gain that tumbles the object is not a cosmetic
    problem: the controller wrench is body-frame, so a tumbling object
    misdirects its own gravity compensation and is then flung out of the scene.

    The inertia here is measured from the live Newton model for the corn can
    (2026-08-20): 0.01 on each diagonal, with a mass of 1.0 kg. An earlier
    version of this test used 0.00125, which is eight times too small, and
    concluded that the current gains sit on their stability boundary. They do
    not. At 1.0 / 0.05 both ratios are 0.25, so the loop has a wide margin and
    the earlier gain reduction cannot be explained by discrete instability.
    """

    from isaaclab_imitation.tasks.manager_based.dexmanip.newton_voc import (
        NewtonVirtualRigidObjectControlCfg,
    )

    cfg = NewtonVirtualRigidObjectControlCfg()
    control_dt = 1.0 / 20.0
    measured_inertia = 0.01
    stability_ratio = cfg.angular_stiffness * control_dt**2 / measured_inertia
    assert stability_ratio < 4.0, (
        f"angular_stiffness={cfg.angular_stiffness} gives k dt^2 / I = "
        f"{stability_ratio:.3f}, at or past the discrete stability limit of 4"
    )
    damping_ratio = cfg.angular_damping * control_dt / measured_inertia
    assert damping_ratio < 2.0, (
        f"angular_damping={cfg.angular_damping} gives d dt / I = "
        f"{damping_ratio:.3f}, at or past the discrete stability limit of 2"
    )
    # The linear loop acts on mass, not inertia, and stays far from its limit.
    assert cfg.linear_stiffness * control_dt**2 / 1.0 < 4.0
    assert cfg.linear_damping * control_dt / 1.0 < 2.0


def test_vega_wuji_agent_matches_the_released_ppo_recipe() -> None:
    """Pin the remaining released PPO scalars and the counts behind them.

    The released recipe states its rollout split and checkpoint cadence as
    counts (four minibatches, a checkpoint every two hundred iterations). This
    config sizes both in frames, which only reproduces the counts at
    ``reference_num_envs``. Pin the frame values and the counts together so a
    later edit cannot move one without the other.
    """

    cfg = VegaWujiRLOptPPOConfig()
    assert cfg.loss.epochs == 5
    assert cfg.loss.gamma == pytest.approx(0.99)
    assert cfg.ppo.critic_coeff == pytest.approx(1.0)
    assert cfg.ppo.clip_value is True
    assert cfg.optim.max_grad_norm == pytest.approx(1.0)
    assert cfg.optim.scheduler == "adaptive"

    reference_batch = cfg.reference_num_envs * cfg.collector.frames_per_batch
    assert cfg.reference_num_envs == 4096
    assert cfg.mini_batches_per_rollout == 4
    assert cfg.save_interval_iterations == 200
    assert reference_batch // cfg.loss.mini_batch_size == cfg.mini_batches_per_rollout
    assert cfg.save_interval // reference_batch == cfg.save_interval_iterations
    assert cfg.trainer.log_interval // reference_batch == cfg.log_interval_iterations
    assert cfg.replay_buffer.size == reference_batch


def test_transition_conditioning_is_full_state_and_action_frame_aligned() -> None:
    """The visible command frame is already the next transition target."""

    joint_count = 59
    object_count = 2
    motion_count = 2
    frame_count = 4
    command = object.__new__(VegaWujiReferenceCommand)
    command._env = SimpleNamespace(num_envs=2)
    command.motion_index = torch.tensor([1, 0])
    command.timestep_counter = torch.tensor([2, 1])
    command._reference_qpos = torch.arange(
        motion_count * frame_count * joint_count, dtype=torch.float32
    ).reshape(motion_count, frame_count, joint_count)
    command._reference_qvel = command._reference_qpos + 10_000.0
    command._reference_object_pose_wxyz = torch.arange(
        motion_count * frame_count * object_count * 7, dtype=torch.float32
    ).reshape(motion_count, frame_count, object_count, 7)
    command._reference_object_twist_w = torch.arange(
        motion_count * frame_count * object_count * 6, dtype=torch.float32
    ).reshape(motion_count, frame_count, object_count, 6)
    live_position = torch.arange(2 * joint_count, dtype=torch.float32).reshape(
        2, joint_count
    )
    live_velocity = live_position + 1_000.0
    command.robot = SimpleNamespace(
        data=SimpleNamespace(
            joint_pos=SimpleNamespace(torch=live_position),
            joint_vel=SimpleNamespace(torch=live_velocity),
        )
    )
    command._env.scene = SimpleNamespace(env_origins=torch.zeros(2, 3))
    object_positions = (
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]),
    )
    object_twists = (
        torch.arange(12, dtype=torch.float32).reshape(2, 6),
        torch.arange(12, 24, dtype=torch.float32).reshape(2, 6),
    )
    identity_xyzw = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(2, 1)
    command.objects = [
        SimpleNamespace(
            data=SimpleNamespace(
                root_link_pos_w=SimpleNamespace(torch=position),
                root_link_quat_w=SimpleNamespace(torch=identity_xyzw),
                root_link_vel_w=SimpleNamespace(torch=twist),
            )
        )
        for position, twist in zip(object_positions, object_twists, strict=True)
    ]

    # Do not add one: ManagerBasedRLEnv advances commands before it emits the
    # observation from which the next action is selected.
    torch.testing.assert_close(command.action_target_frame, torch.tensor([2, 1]))
    expected_qpos = command._reference_qpos[
        command.motion_index, command.timestep_counter
    ]
    expected_qvel = command._reference_qvel[
        command.motion_index, command.timestep_counter
    ]
    torch.testing.assert_close(
        command.current_robot_state,
        torch.cat((live_position, live_velocity), dim=-1),
    )
    torch.testing.assert_close(
        command.next_desired_robot_state,
        torch.cat((expected_qpos, expected_qvel), dim=-1),
    )
    expected_object_pose = command._reference_object_pose_wxyz[
        command.motion_index, command.timestep_counter
    ]
    expected_object_twist = command._reference_object_twist_w[
        command.motion_index, command.timestep_counter
    ]
    expected_object = torch.cat(
        (expected_object_pose, expected_object_twist), dim=-1
    ).reshape(2, -1)
    torch.testing.assert_close(command.next_desired_object_state, expected_object)
    expected_live_pose = torch.cat(
        (
            torch.stack(object_positions, dim=1),
            torch.tensor([1.0, 0.0, 0.0, 0.0]).view(1, 1, 4).repeat(2, 2, 1),
        ),
        dim=-1,
    )
    expected_live = torch.cat(
        (expected_live_pose, torch.stack(object_twists, dim=1)), dim=-1
    ).reshape(2, -1)
    torch.testing.assert_close(command.current_object_state, expected_live)
    assert command.current_robot_state.shape == (2, 118)
    assert command.next_desired_robot_state.shape == (2, 118)
    assert command.current_object_state.shape == (2, 26)
    assert command.next_desired_object_state.shape == (2, 26)


def test_transition_conditioning_adds_keys_without_reordering_observation_surface() -> (
    None
):
    legacy_keys = [
        ("policy", "wrist_position"),
        ("policy", "wrist_orientation"),
        ("policy", "wrist_velocity"),
        ("policy", "finger_joint_pos"),
        ("policy", "finger_joint_vel"),
        ("policy", "object_position"),
        ("policy", "object_orientation"),
        ("policy", "command"),
        ("policy", "last_action"),
        ("policy", "processed_right_actions"),
        ("policy", "processed_left_actions"),
        ("policy", "contact_position_direction"),
    ]
    transition_keys = [
        ("policy", "current_robot_state"),
        ("policy", "next_desired_robot_state"),
        ("policy", "next_desired_object_state"),
        ("policy", "current_object_state"),
    ]

    assert VEGA_WUJI_POLICY_INPUT_KEYS == legacy_keys + transition_keys
    cfg = VegaWujiImitationEnvCfg()
    for _, name in transition_keys:
        term = getattr(cfg.observations.policy, name)
        assert term.params == {"command_name": "motion"}


def test_vega_runtime_joint_count_pins_transition_state_width() -> None:
    cfg = VegaWujiImitationEnvCfg()
    mjcf_path = Path(cfg.scene.robot.default.spawn.asset_path)

    assert len(_mjcf_actuator_joint_names(mjcf_path)) == 59
    assert 2 * len(_mjcf_actuator_joint_names(mjcf_path)) == 118


def test_iltools_references_pack_joint_order_scene_frames_and_contacts() -> None:
    long_reference = _reference(sequence_id="long", frame_count=3)
    short_reference = _reference(sequence_id="short", frame_count=2, qpos_offset=100.0)
    short_reference.object_twists_w[..., 0] = [[1.0], [2.0]]
    packed = _pack_references(
        (
            long_reference,
            short_reference,
        ),
        expected_robot_name="vega_wuji",
        expected_fps=20.0,
        joint_names=("joint_a", "joint_b"),
        fingertip_names={"right": ("r_tip",), "left": ("l_tip",)},
        wrist_frame_names=_WRIST_FRAME_NAMES,
    )

    assert packed["joint_names"] == ("joint_a", "joint_b")
    np.testing.assert_allclose(packed["qpos"][0, 0], (10.0, 0.0))
    np.testing.assert_allclose(packed["qpos"][1, 2], (111.0, 101.0))
    np.testing.assert_allclose(packed["object_twist_w"][1, :, 0, 0], (1.0, 2.0, 2.0))
    assert packed["lengths"].tolist() == [3, 2]
    assert packed["contact_link_names"] == {
        "right": ("r_tip",),
        "left": ("l_tip",),
    }
    assert packed["contacts"]["right"]["active"].shape == (2, 3, 1)


def test_pack_keeps_repeated_reference_slots_but_senses_each_link_once() -> None:
    reference = _reference(sequence_id="repeated", frame_count=2)
    reference.contacts.link_names[0, 1] = "r_tip"
    reference.contacts.active[:, 0, 1] = True
    reference.contacts.object_indices[:, 0, 1] = 0

    packed = _pack_references(
        (reference,),
        expected_robot_name="vega_wuji",
        expected_fps=20.0,
        joint_names=("joint_a", "joint_b"),
        fingertip_names={"right": ("r_tip",), "left": ("l_tip",)},
        wrist_frame_names=_WRIST_FRAME_NAMES,
    )

    assert packed["contact_link_names"]["right"] == ("r_tip",)
    assert packed["contacts"]["right"]["active"].shape == (1, 2, 2)


def test_dominant_contact_object_backfills_each_hand_frame() -> None:
    object_indices = torch.tensor([[[-1, -1], [-1, -1], [1, 1], [0, 1], [-1, -1]]])
    result = dominant_contact_object_indices(
        object_indices,
        object_indices >= 0,
        torch.tensor([4]),
        num_objects=2,
    )

    assert result.tolist() == [[1, 1, 1, 0, 0]]


def test_newton_counterpart_columns_use_full_unique_body_labels() -> None:
    object_labels = (
        "/World/envs/env_0/cube/base",
        "/World/envs/env_0/mug/base",
    )
    counterpart_labels = (
        "/World/envs/env_0/mug/base",
        "/World/envs/env_0/cube/base",
    )

    assert match_newton_counterpart_columns(object_labels, counterpart_labels) == (1, 0)
    with pytest.raises(ValueError, match="matched 0"):
        match_newton_counterpart_columns(
            object_labels,
            ("/World/envs/env_0/cube/base",),
        )


def test_newton_contact_sensors_use_hand_and_object_subtree_globs() -> None:
    cfg = VegaWujiImitationEnvCfg()

    _add_contact_sensors(
        cfg,
        {"right": ("r_tip",), "left": ("l_tip",)},
        ("workpiece",),
        ("table",),
    )

    assert cfg.scene.right_hand_object_contacts.prim_path == (
        "{ENV_REGEX_NS}/Robot/.*/r_mount.*"
    )
    assert cfg.scene.left_hand_object_contacts.prim_path == (
        "{ENV_REGEX_NS}/Robot/.*/l_mount.*"
    )
    assert cfg.scene.right_hand_object_contacts.filter_prim_paths_expr == [
        "{ENV_REGEX_NS}/workpiece",
        "{ENV_REGEX_NS}/workpiece/.*",
    ]
    assert cfg.scene.robot_support_contacts.prim_path == "{ENV_REGEX_NS}/Robot/.*"
    assert cfg.scene.robot_support_contacts.filter_shape_prim_expr == [
        "{ENV_REGEX_NS}/table",
        "{ENV_REGEX_NS}/table/.*",
    ]
    assert cfg.rewards.support_contact_force_penalty is not None
    assert cfg.terminations.excessive_support_contact_force is not None


def test_free_space_scene_disables_support_contact_terms() -> None:
    cfg = VegaWujiImitationEnvCfg()

    _add_contact_sensors(
        cfg,
        {"right": ("r_tip",), "left": ("l_tip",)},
        ("workpiece",),
    )

    assert not hasattr(cfg.scene, "robot_support_contacts")
    assert cfg.rewards.support_contact_force_penalty is None
    assert cfg.terminations.excessive_support_contact_force is None


def test_pack_rejects_a_scene_change_between_motions() -> None:
    first = _reference(sequence_id="first", frame_count=2)
    second = _reference(sequence_id="second", frame_count=2)
    second.object_radii[:] = 0.2

    with pytest.raises(ValueError, match="changes object_radii"):
        _pack_references(
            (first, second),
            expected_robot_name="vega_wuji",
            expected_fps=20.0,
            joint_names=("joint_a", "joint_b"),
            fingertip_names={"right": ("r_tip",), "left": ("l_tip",)},
            wrist_frame_names=_WRIST_FRAME_NAMES,
        )


def test_pack_rejects_scene_physics_change_between_motions() -> None:
    first = _reference(sequence_id="first", frame_count=2)
    second = _reference(sequence_id="second", frame_count=2)
    physics_args = {
        "object_mass_kg": np.asarray((0.42,), dtype=np.float32),
        "object_center_of_mass_m": np.zeros((1, 3), dtype=np.float32),
        "object_diagonal_inertia_kg_m2": np.asarray(
            ((0.004, 0.005, 0.006),), dtype=np.float32
        ),
        "object_static_friction": np.asarray((0.8,), dtype=np.float32),
        "object_dynamic_friction": np.asarray((0.7,), dtype=np.float32),
        "object_restitution": np.asarray((0.0,), dtype=np.float32),
        "support_static_friction": np.empty(0, dtype=np.float32),
        "support_dynamic_friction": np.empty(0, dtype=np.float32),
        "support_restitution": np.empty(0, dtype=np.float32),
    }
    first.scene_physics = ScenePhysics(**physics_args)
    second.scene_physics = ScenePhysics(
        **{**physics_args, "object_mass_kg": np.asarray((0.43,), dtype=np.float32)}
    )
    first.scene_physics.validate(object_count=1, support_count=0)
    second.scene_physics.validate(object_count=1, support_count=0)

    with pytest.raises(ValueError, match="scene_physics.object_mass_kg"):
        _pack_references(
            (first, second),
            expected_robot_name="vega_wuji",
            expected_fps=20.0,
            joint_names=("joint_a", "joint_b"),
            fingertip_names={"right": ("r_tip",), "left": ("l_tip",)},
            wrist_frame_names=_WRIST_FRAME_NAMES,
        )


def test_iltools_scene_physics_drives_spawn_material_and_newton_inertia() -> None:
    reference = _reference(sequence_id="physics", frame_count=2)
    reference.scene_physics = ScenePhysics(
        object_mass_kg=np.asarray((0.42,), dtype=np.float32),
        object_center_of_mass_m=np.asarray(((0.01, -0.02, 0.03),), dtype=np.float32),
        object_diagonal_inertia_kg_m2=np.asarray(
            ((0.004, 0.005, 0.006),), dtype=np.float32
        ),
        object_static_friction=np.asarray((0.8,), dtype=np.float32),
        object_dynamic_friction=np.asarray((0.7,), dtype=np.float32),
        object_restitution=np.asarray((0.1,), dtype=np.float32),
        support_static_friction=np.empty(0, dtype=np.float32),
        support_dynamic_friction=np.empty(0, dtype=np.float32),
        support_restitution=np.empty(0, dtype=np.float32),
    )
    reference.scene_physics.validate(object_count=1, support_count=0)

    spec = _scene_spec(reference)
    assert spec.objects[0].mass == pytest.approx(0.42)
    assert spec.objects[0].material is not None
    assert spec.objects[0].material.static_friction == pytest.approx(0.8)
    assert spec.objects[0].material.dynamic_friction == pytest.approx(0.7)
    assert spec.objects[0].material.restitution == pytest.approx(0.1)

    class _Object:
        def set_masses_index(self, *, masses) -> None:
            self.masses = masses

        def set_coms_index(self, *, coms) -> None:
            self.coms = coms

        def set_inertias_index(self, *, inertias) -> None:
            self.inertias = inertias

    item = _Object()
    _apply_scene_inertial_contract(
        [item], reference.scene_physics, num_envs=2, device="cpu"
    )
    torch.testing.assert_close(item.masses, torch.full((2, 1), 0.42))
    torch.testing.assert_close(
        item.coms,
        torch.tensor([[[0.01, -0.02, 0.03]], [[0.01, -0.02, 0.03]]]),
    )
    torch.testing.assert_close(
        item.inertias[0, 0].reshape(3, 3),
        torch.diag(torch.tensor([0.004, 0.005, 0.006])),
    )


def test_mjcf_order_and_robot_only_contract(tmp_path: Path) -> None:
    path = tmp_path / "robot.xml"
    path.write_text(
        """
        <mujoco>
          <worldbody>
            <body name="R_ee">
              <body name="r_mount"><site name="right_palm"/></body>
            </body>
            <body name="L_ee">
              <body name="l_mount"><site name="left_palm"/></body>
            </body>
          </worldbody>
          <actuator>
            <position name="b" joint="joint_b"/>
            <position name="a" joint="joint_a"/>
          </actuator>
        </mujoco>
        """,
        encoding="utf-8",
    )

    _validate_robot_only_mjcf(path)
    assert _mjcf_actuator_joint_names(path) == ("joint_b", "joint_a")

    root = ET.parse(path).getroot()
    ET.SubElement(root.find("worldbody"), "body", {"name": "desk"})
    ET.ElementTree(root).write(path)
    with pytest.raises(ValueError, match="scene bodies: desk"):
        _validate_robot_only_mjcf(path)


@pytest.mark.parametrize(
    "site_xml",
    (
        '<site name="right_palm" pos="0.01 0 0"/>',
        '<site name="right_palm" quat="0 1 0 0"/>',
    ),
)
def test_robot_only_contract_rejects_non_identity_palm_site(
    tmp_path: Path,
    site_xml: str,
) -> None:
    path = tmp_path / "bad_palm.xml"
    path.write_text(
        f"""
        <mujoco>
          <worldbody>
            <body name="R_ee"><body name="r_mount">{site_xml}</body></body>
            <body name="L_ee">
              <body name="l_mount"><site name="left_palm"/></body>
            </body>
          </worldbody>
        </mujoco>
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="right_palm must be an identity site"):
        _validate_robot_only_mjcf(path)


def test_pack_rejects_wrong_reference_wrist_frame() -> None:
    reference = _reference(sequence_id="wrong_frame", frame_count=2)
    reference.right_wrist_frame_name = "R_ee"

    with pytest.raises(ValueError, match="right wrist frame.*expected 'right_palm'"):
        _pack_references(
            (reference,),
            expected_robot_name="vega_wuji",
            expected_fps=20.0,
            joint_names=("joint_a", "joint_b"),
            fingertip_names={"right": ("r_tip",), "left": ("l_tip",)},
            wrist_frame_names=_WRIST_FRAME_NAMES,
        )


def test_termination_penalty_uses_only_non_timeout_termination_mask() -> None:
    env = SimpleNamespace(
        termination_manager=SimpleNamespace(terminated=torch.tensor([True, False]))
    )
    torch.testing.assert_close(mdp.termination_penalty(env), torch.tensor([1.0, 0.0]))


def test_command_debug_visualization_is_deferred_until_the_reference_loads() -> None:
    """``set_debug_vis`` must not touch Reference state.

    ``CommandTerm.__init__`` calls ``set_debug_vis`` before this term loads its
    Reference, so building markers there raises ``AttributeError`` on
    ``_reference_contacts``. Marker construction belongs in the first
    visualization callback instead.
    """

    from isaaclab_imitation.tasks.manager_based.dexmanip.command import (
        VegaWujiReferenceCommand,
    )

    for name in (
        "_set_debug_vis_impl",
        "_build_debug_markers",
        "_debug_vis_callback",
        "_has_active_contact_geometry",
    ):
        assert callable(getattr(VegaWujiReferenceCommand, name, None)), (
            f"the command term must define {name}"
        )

    # A bare instance stands in for the term mid-construction: no Reference
    # arrays, no scene. Both entry points must stay quiet on it.
    bare = object.__new__(VegaWujiReferenceCommand)
    VegaWujiReferenceCommand._set_debug_vis_impl(bare, True)
    assert bare._debug_vis_requested is True
    assert not getattr(bare, "_debug_markers", None), (
        "markers must not be built before the Reference is available"
    )
    VegaWujiReferenceCommand._set_debug_vis_impl(bare, False)
    assert bare._debug_vis_requested is False

    # With visualization off, the callback must return before it reads state.
    VegaWujiReferenceCommand._debug_vis_callback(bare, None)


def test_command_debug_visualization_defaults_to_off() -> None:
    from isaaclab_imitation.tasks.manager_based.dexmanip.command import (
        VegaWujiReferenceCommandCfg,
    )

    assert VegaWujiReferenceCommandCfg().debug_vis is False
