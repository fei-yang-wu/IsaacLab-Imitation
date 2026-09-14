"""Contract tests for the source-parity Sharpa task (``Isaac-Sharpa-V2D-Source-v0``).

These tests pin the released ``Sharpa-V2D-v0`` recipe of video-to-data
without constructing an Isaac environment: registration, observation order
and inert noise, the 65-value command layout, reward set and weights, the
six-term curriculum, terminations, hand-material randomization, the
released-direction PhysX contact sensor, the step-mode hold, and the source
action penalty.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch

import isaaclab_imitation.tasks  # noqa: F401
from isaaclab_imitation.assets.sharpa import HAND_CONTACT_BODIES
from isaaclab_imitation.tasks.manager_based.dexmanip import mdp as dex_mdp
from isaaclab_imitation.tasks.manager_based.dexmanip import sharpa_source_mdp
from isaaclab_imitation.tasks.manager_based.dexmanip.config.sharpa import (
    sharpa_env_cfg,
)
from isaaclab_imitation.tasks.manager_based.dexmanip.config.sharpa import (
    sharpa_source_env_cfg as parity,
)
from isaaclab_imitation.tasks.manager_based.dexmanip.curriculum import (
    DEFAULT_TIMESTEP_SCHEDULE,
    DEFAULT_VIRTUAL_OBJECT_CONTROL_SCALE_FACTOR,
    SOURCE_REWARD_WEIGHT_SCHEDULES,
)
from isaaclab_imitation.tasks.manager_based.dexmanip.sharpa_source_command import (
    SharpaSourceReferenceCommand,
    SharpaSourceReferenceCommandCfg,
)

SOURCE_OBSERVATION_ORDER = (
    "wrist_position_e",
    "wrist_orientation_e",
    "wrist_velocity_b",
    "finger_joint_pos",
    "finger_joint_vel",
    "object_position_e",
    "object_orientation_e",
    "command",
    "actions",
    "processed_right_actions",
    "processed_left_actions",
    "contact_position_direction_in_wrist",
)
SOURCE_NOISY_TERMS = SOURCE_OBSERVATION_ORDER[:7]
SOURCE_REWARD_WEIGHTS = {
    "action_rate_l2": -5.0e-3,
    "action_l1": -2.0e-3,
    "object_keypoints_tracking_exp": 0.0,
    "hand_keypoints_tracking_exp": 0.0,
    "hand_joint_pos_tracking_exp": 0.0,
    "termination_penalty": -100.0,
    "contact_wrench_support_reward": 10.0,
    "unintended_contact_penalty": -10.0,
    "missed_contact_penalty": -1.0,
}


def _term_names(cfg: object) -> list[str]:
    """Declared configclass field names in declaration order."""

    return [field.name for field in dataclasses.fields(cfg)]


def test_source_parity_task_is_registered_for_both_trainers() -> None:
    spec = gym.spec("Isaac-Sharpa-V2D-Source-v0")
    assert spec.entry_point == "isaaclab.envs:ManagerBasedRLEnv"
    assert spec.kwargs["env_cfg_entry_point"].endswith(":SharpaV2DSourceParityEnvCfg")
    assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(":SharpaV2DPPORunnerCfg")
    assert spec.kwargs["rlopt_ppo_cfg_entry_point"].endswith(":SharpaRLOptPPOConfig")


def test_observation_terms_follow_the_released_order_with_inert_noise() -> None:
    policy = parity.SharpaSourceObservationsCfg.PolicyCfg()
    terms = [name for name in _term_names(policy) if name in SOURCE_OBSERVATION_ORDER]
    assert tuple(terms) == SOURCE_OBSERVATION_ORDER
    assert policy.enable_corruption is False
    assert policy.concatenate_terms is True
    for name in SOURCE_OBSERVATION_ORDER:
        term = getattr(policy, name)
        if name in SOURCE_NOISY_TERMS:
            assert term.noise is not None
            assert (term.noise.n_min, term.noise.n_max) == (-0.01, 0.01)
        else:
            assert term.noise is None
    assert policy.processed_right_actions.params == {"action_name": "right_hand"}
    assert policy.processed_left_actions.params == {"action_name": "left_hand"}


def test_reward_set_and_weights_match_the_released_recipe() -> None:
    rewards = parity.SharpaSourceRewardsCfg()
    names = {
        name
        for name in _term_names(rewards)
        if hasattr(getattr(rewards, name), "weight")
    }
    assert names == set(SOURCE_REWARD_WEIGHTS)
    for name, weight in SOURCE_REWARD_WEIGHTS.items():
        assert getattr(rewards, name).weight == weight, name
    assert rewards.contact_wrench_support_reward.params["tolerance"] == 0.1
    assert rewards.contact_wrench_support_reward.params["var"] == 0.1
    assert rewards.object_keypoints_tracking_exp.params["var"] == 0.1
    assert rewards.hand_keypoints_tracking_exp.params["var"] == 0.1
    assert rewards.hand_joint_pos_tracking_exp.params["var"] == 1.0
    assert rewards.action_l1.params["action_names"] == ["right_hand", "left_hand"]
    assert rewards.action_l1.func is sharpa_source_mdp.action_norm_sum
    assert rewards.termination_penalty.func is dex_mdp.termination_penalty


def test_curriculum_drives_only_the_six_released_terms() -> None:
    params = parity.SharpaSourceCurriculumCfg().fixed_timestep.params
    assert params["command_name"] == "sharpa_reference"
    assert params["num_steps_per_env"] == 24
    assert params["timestep_schedule"] == list(DEFAULT_TIMESTEP_SCHEDULE)
    assert params["virtual_object_control_scale_factor"] == list(
        DEFAULT_VIRTUAL_OBJECT_CONTROL_SCALE_FACTOR
    )
    schedules = params["reward_weight_schedules"]
    assert set(schedules) == set(SOURCE_REWARD_WEIGHT_SCHEDULES)
    assert schedules["object_keypoints_tracking_exp"] == [
        0.0,
        0.1,
        0.25,
        0.25,
        0.5,
        0.5,
        1.0,
        1.0,
        1.0,
        20.0,
    ]
    assert schedules["hand_keypoints_tracking_exp"] == 0.25
    assert schedules["hand_joint_pos_tracking_exp"] == 0.25
    assert schedules["contact_wrench_support_reward"] == 10.0
    assert schedules["unintended_contact_penalty"] == -10.0
    assert schedules["missed_contact_penalty"] == -1.0


def test_terminations_match_the_released_thresholds() -> None:
    terminations = parity.SharpaSourceTerminationsCfg()
    assert terminations.time_out.time_out is True
    assert terminations.time_out.func is sharpa_source_mdp.reference_timeout
    assert terminations.hand_wrist_away_from_trajectory.params["threshold"] == 0.2
    away = terminations.object_away_from_trajectory.params
    assert (away["position_threshold"], away["orientation_threshold"]) == (0.2, 0.7)


def test_hand_material_randomization_matches_the_released_event() -> None:
    events = parity.SharpaSourceEventsCfg()
    for side in ("right", "left"):
        term = getattr(events, f"{side}_physics_material")
        assert term.mode == "startup"
        assert term.params["asset_cfg"].name == f"{side}_robot"
        assert term.params["static_friction_range"] == (2.0, 2.01)
        assert term.params["dynamic_friction_range"] == (2.0, 2.01)
        assert term.params["restitution_range"] == (0.0, 0.0)
        assert term.params["num_buckets"] == 64


def test_command_cfg_matches_the_released_contact_and_reset_settings() -> None:
    cfg = SharpaSourceReferenceCommandCfg()
    assert cfg.class_type is SharpaSourceReferenceCommand
    assert cfg.contact_link_names == tuple(HAND_CONTACT_BODIES)
    assert len(cfg.contact_link_names) == 17
    assert cfg.num_wrench_basis == 512
    assert cfg.num_friction_cone_edges == 8
    assert cfg.friction_coefficient == 0.1
    assert cfg.contact_force_threshold == 0.1
    assert cfg.reset_to_first_frame_prob == 0.1
    assert cfg.reset_finger_openness == 0.7
    assert cfg.virtual_object_control_decay_steps == 20
    assert cfg.virtual_object_control_decay_mode == "step"
    assert cfg.make_quat_unique is True


def test_contact_link_prim_paths_follow_the_imported_kinematic_tree() -> None:
    paths = parity.hand_link_prim_paths("right", HAND_CONTACT_BODIES)
    assert len(paths) == 17
    prefix = "{ENV_REGEX_NS}/right_robot/Geometry/right_hand_C_MC"
    assert paths[0] == prefix
    assert paths[1] == f"{prefix}/right_thumb_CMC_VL/right_thumb_MC"
    assert paths[6] == (
        f"{prefix}/right_index_MCP_VL/right_index_PP/right_index_MP/right_index_DP"
    )
    assert len(set(paths)) == 17
    filters, link_index = parity.hand_link_contact_filters("right", HAND_CONTACT_BODIES)
    assert len(filters) == len(link_index) == 22  # 17 links + 5 fingertip elastomers
    assert sorted(set(link_index)) == list(range(17))
    assert filters[0] == f"{prefix}/right_hand_C_MC_cut"
    thumb_dp = [f for f, i in zip(filters, link_index, strict=True) if i == 3]
    assert thumb_dp == [
        f"{prefix}/right_thumb_CMC_VL/right_thumb_MC/right_thumb_MCP_VL/"
        "right_thumb_PP/right_thumb_DP/right_thumb_DP",
        f"{prefix}/right_thumb_CMC_VL/right_thumb_MC/right_thumb_MCP_VL/"
        "right_thumb_PP/right_thumb_DP/thumb_elastomer",
    ]
    with pytest.raises(ValueError, match="resolved"):
        parity.hand_link_prim_paths("right", (".*_no_such_link",))


def test_command_layout_has_the_released_65_values() -> None:
    envs = 3
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(envs, 4)
    zeros = torch.zeros(envs, 3)
    fake = SimpleNamespace(
        right_hand_wrist_position_e=zeros,
        left_hand_wrist_position_e=zeros,
        right_hand_wrist_wxyz_e=identity,
        left_hand_wrist_wxyz_e=identity,
        right_hand_wrist_pose_command_e=torch.cat((zeros + 0.1, identity), dim=-1),
        left_hand_wrist_pose_command_e=torch.cat((zeros - 0.1, identity), dim=-1),
        right_hand_finger_joint_pos_command=torch.ones(envs, 22),
        left_hand_finger_joint_pos_command=torch.ones(envs, 22),
        right_hand_finger_joint_pos=torch.zeros(envs, 22),
        left_hand_finger_joint_pos=torch.zeros(envs, 22),
        object_position_e=zeros[:, None],
        object_orientation_e=identity[:, None],
        object_body_position_command_e=(zeros + 0.2)[:, None],
        object_body_wxyz_command_e=identity[:, None],
    )
    command = SharpaSourceReferenceCommand.command.fget(fake)
    assert command.shape == (envs, 65)
    torch.testing.assert_close(command[:, :3], zeros + 0.1)
    torch.testing.assert_close(command[:, 3:7], identity)
    torch.testing.assert_close(command[:, 7:10], zeros - 0.1)
    torch.testing.assert_close(command[:, 14:36], torch.ones(envs, 22))
    torch.testing.assert_close(command[:, 58:61], zeros + 0.2)
    torch.testing.assert_close(command[:, 61:65], identity)


def test_pose_delta_keeps_the_raw_relative_quaternion_when_not_unique() -> None:
    current = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    target = torch.tensor([[-1.0, 0.0, 0.0, 0.0]])
    raw = dex_mdp.pose_delta_command(
        torch.zeros(1, 3), current, torch.zeros(1, 3), target, unique=False
    )
    unique = dex_mdp.pose_delta_command(
        torch.zeros(1, 3), current, torch.zeros(1, 3), target
    )
    assert raw[0, 3].item() == pytest.approx(-1.0)
    assert unique[0, 3].item() == pytest.approx(1.0)


def test_step_mode_holds_the_reference_and_assistance_for_the_settling_window() -> None:
    envs = 4
    fake = SimpleNamespace(
        cfg=SimpleNamespace(virtual_object_control_decay_steps=20),
        steps_since_last_reset=torch.tensor([0, 18, 19, 40]),
        virtual_object_controller_curriculum_scale=torch.tensor(0.25),
        virtual_object_controller_scale_factor_per_env=torch.ones(envs, 1),
        _lengths=torch.tensor([10]),
        motion_index=torch.zeros(envs, dtype=torch.long),
        timestep_counter=torch.tensor([0, 0, 0, 9]),
        trajectory_manager=SimpleNamespace(
            advance_cursors=lambda ids: advanced.append(ids)
        ),
    )
    advanced: list[torch.Tensor] = []
    SharpaSourceReferenceCommand._update_command(fake)
    # Steps 1, 19: still settling. Step 20: released. Step 41: at the horizon.
    scale = fake.virtual_object_controller_scale_factor_per_env[:, 0].tolist()
    assert scale == [1.0, 1.0, 0.25, 0.25]
    assert advanced[0].tolist() == [2]


def test_action_norm_sums_squares_over_both_hands() -> None:
    terms = {
        "right_hand": SimpleNamespace(raw_actions=torch.tensor([[1.0, 2.0]])),
        "left_hand": SimpleNamespace(raw_actions=torch.tensor([[3.0, 0.0]])),
    }
    env = SimpleNamespace(action_manager=SimpleNamespace(get_term=terms.__getitem__))
    value = sharpa_source_mdp.action_norm_sum(env, ["right_hand", "left_hand"])
    assert value.tolist() == [14.0]


def test_source_motion_speed_is_the_released_half_speed() -> None:
    assert parity.SOURCE_MOTION_SPEED == 0.5
    field = next(
        item
        for item in dataclasses.fields(parity.SharpaV2DSourceParityEnvCfg)
        if item.name == "require_source_motion_speed"
    )
    default = field.default
    if (
        default is dataclasses.MISSING
        and field.default_factory is not dataclasses.MISSING
    ):
        default = field.default_factory()
    assert default is True


def test_object_body_prim_path_follows_the_asset_kind(tmp_path) -> None:
    """A URDF nests its single link under ``Geometry``; two links are refused.

    The USD side of this contract is covered below, against real stages: the
    body's location is read from the asset instead of assumed, because a USD
    written by Isaac Lab's URDF converter keeps the same nesting.
    """

    from isaaclab_imitation.tasks.manager_based.dexmanip.config.sharpa.sharpa_env_cfg import (
        object_body_prim_path,
    )

    urdf = tmp_path / "object.urdf"
    urdf.write_text("<robot name='box'><link name='object'/></robot>", encoding="utf-8")
    assert object_body_prim_path(str(urdf)) == "{ENV_REGEX_NS}/object/Geometry/object"

    articulated = tmp_path / "two.urdf"
    articulated.write_text(
        "<robot name='box'><link name='bottom'/><link name='top'/></robot>",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly one link"):
        object_body_prim_path(str(articulated))


# --- Object rigid-body prim path ------------------------------------------
#
# The scene points its contact sensor and its rigid-object view at this path.
# When it names a prim that carries no rigid body, PhysX reports "Failed to
# find rigid body" and the object silently stops being simulated, so the path
# must be read from the asset rather than assumed.

_USDA_NESTED_BODY = """#usda 1.0
(
    defaultPrim = "arctic_box_bottom"
)

def Xform "arctic_box_bottom"
{
    def Scope "Geometry"
    {
        def Xform "object" (
            prepend apiSchemas = ["PhysicsRigidBodyAPI"]
        )
        {
        }
    }
}
"""

_USDA_ROOT_BODY = """#usda 1.0
(
    defaultPrim = "proxy"
)

def Xform "proxy" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI"]
)
{
}
"""

_USDA_NO_BODY = """#usda 1.0
(
    defaultPrim = "proxy"
)

def Xform "proxy"
{
}
"""


def _write_usda(tmp_path, name: str, body: str):
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def test_object_body_path_follows_the_converter_nesting(tmp_path) -> None:
    """A converted USD keeps the URDF importer's ``Geometry/<link>`` nesting."""

    usda = _write_usda(tmp_path, "object.usda", _USDA_NESTED_BODY)
    assert (
        sharpa_env_cfg.object_body_prim_path(str(usda))
        == "{ENV_REGEX_NS}/object/Geometry/object"
    )


def test_object_body_path_accepts_a_body_on_the_default_prim(tmp_path) -> None:
    """A hand-authored proxy may carry its rigid body at the asset root."""

    usda = _write_usda(tmp_path, "proxy.usda", _USDA_ROOT_BODY)
    assert sharpa_env_cfg.object_body_prim_path(str(usda)) == "{ENV_REGEX_NS}/object"


def test_object_body_path_refuses_a_usd_without_a_rigid_body(tmp_path) -> None:
    """Fail loudly instead of returning a path that simulates nothing."""

    usda = _write_usda(tmp_path, "empty.usda", _USDA_NO_BODY)
    with pytest.raises(ValueError, match="rigid body"):
        sharpa_env_cfg.object_body_prim_path(str(usda))


def test_object_body_path_agrees_between_the_urdf_and_its_converted_usd(
    tmp_path,
) -> None:
    """Converting an asset must not move the body the scene points at."""

    urdf = tmp_path / "object.urdf"
    urdf.write_text(
        '<?xml version="1.0"?>\n<robot name="proxy"><link name="object"/></robot>\n',
        encoding="utf-8",
    )
    usda = _write_usda(tmp_path, "object.usda", _USDA_NESTED_BODY)
    assert sharpa_env_cfg.object_body_prim_path(str(urdf)) == (
        sharpa_env_cfg.object_body_prim_path(str(usda))
    )
