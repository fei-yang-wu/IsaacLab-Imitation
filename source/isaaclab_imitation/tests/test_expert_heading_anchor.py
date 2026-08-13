"""Contract for the ``expert_heading`` macro-window frame convention.

The historical "robot" convention splits the DiffSR macro window across two
frames: pretraining anchors every slot at the expert's own full pose at the
window's first slot, while the live encoder input anchors at the robot's full
live anchor pose -- so a frozen encoder's rollout input differs from its
pretraining input by exactly the live tracking error, a distribution it never
saw. ``expert_heading`` uses one convention for both: the expert's slot-0
heading (yaw-only) frame with an xy-only origin.

Three things are pinned:

1. :func:`heading_anchor_frame` cancels exactly the transformations the
   re-rooted tracking reward is invariant under -- global yaw and xy
   translation -- and nothing more: absolute height and roll/pitch relative
   to gravity pass through.
2. Under ``expert_heading`` the macro builder collapses both the pretrain
   (``expert``) and live (``rollout``) contexts onto one frame, so the same
   reference cursor yields identical encoder inputs in both.
3. The macro state has the same width in both modes, so an invalid mode must
   be refused at configuration time -- a wrong mode cannot be caught by any
   shape check downstream.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
from isaaclab.utils.math import quat_apply

from isaaclab_imitation.envs.expert_data_plane import ExpertDataPlane
from isaaclab_imitation.tasks.manager_based.imitation.mdp._compiled import (
    body_pose_in_anchor_frame,
    heading_anchor_frame,
)

_ATOL = 1e-5


def _quat_xyzw(roll: float, pitch: float, yaw: float) -> torch.Tensor:
    """Build an Isaac Lab (x, y, z, w) quaternion from ZYX Euler angles."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return torch.tensor(
        [
            sr * cp * cy - cr * sp * sy,  # x
            cr * sp * cy + sr * cp * sy,  # y
            cr * cp * sy - sr * sp * cy,  # z
            cr * cp * cy + sr * sp * sy,  # w
        ],
        dtype=torch.float32,
    )


def test_heading_anchor_frame_zeroes_xy_and_extracts_heading():
    anchor_pos = torch.tensor([[1.5, -2.0, 0.83]])
    anchor_quat = _quat_xyzw(roll=0.2, pitch=-0.1, yaw=1.1).unsqueeze(0)

    origin, heading = heading_anchor_frame(anchor_pos, anchor_quat)

    assert torch.allclose(origin, torch.tensor([[1.5, -2.0, 0.0]]), atol=_ATOL)
    # Heading is a pure z-rotation: x and y components are exactly zero.
    assert torch.allclose(heading[:, :2], torch.zeros(1, 2), atol=_ATOL)
    assert torch.allclose(heading.norm(dim=-1), torch.ones(1), atol=_ATOL), (
        "heading quat must be renormalized"
    )


def test_expert_heading_frame_keeps_height_and_tilt_cancels_yaw_and_xy():
    """The frame's invariance group must equal the re-rooted reward's."""
    yaw, xy = 0.9, torch.tensor([3.0, -1.0])
    roll, pitch, z0 = 0.25, -0.15, 0.78

    # A two-slot window: the anchor slot plus one future slot 0.2 m ahead of
    # it along the anchor's own heading (the swing-twist heading, which under
    # combined roll+pitch is NOT the ZYX Euler yaw), 0.05 m higher, 0.1 rad
    # more yaw.
    anchor0_pos = torch.tensor([xy[0], xy[1], z0]).unsqueeze(0)
    anchor0_quat = _quat_xyzw(roll, pitch, yaw).unsqueeze(0)
    _, heading0 = heading_anchor_frame(anchor0_pos, anchor0_quat)
    step_world = quat_apply(heading0, torch.tensor([[0.2, 0.0, 0.0]]))[
        0
    ] + torch.tensor([0.0, 0.0, 0.05])
    anchor1_pos = anchor0_pos + step_world
    anchor1_quat = _quat_xyzw(roll, pitch, yaw + 0.1).unsqueeze(0)
    window_pos = torch.stack([anchor0_pos[0], anchor1_pos[0]]).unsqueeze(0)
    window_quat = torch.cat([anchor0_quat, anchor1_quat]).unsqueeze(0)

    origin, heading = heading_anchor_frame(anchor0_pos, anchor0_quat)
    pos_b, quat_b = body_pose_in_anchor_frame(origin, heading, window_pos, window_quat)

    # Slot 0: xy cancelled, absolute height preserved.
    assert torch.allclose(pos_b[0, 0], torch.tensor([0.0, 0.0, z0]), atol=_ATOL)
    # Slot 1: forward step lands on +x of the heading frame; height absolute.
    assert torch.allclose(pos_b[0, 1], torch.tensor([0.2, 0.0, z0 + 0.05]), atol=_ATOL)
    # Slot 0 orientation keeps roll/pitch (yaw removed): it must equal the
    # tilt-only quaternion, not identity.
    tilt_only = _quat_xyzw(roll, pitch, 0.0)
    dot = torch.abs((quat_b[0, 0] * tilt_only).sum())
    assert dot > 1.0 - 1e-4, f"slot-0 tilt lost: |dot|={dot.item():.6f}"

    # The whole construction is invariant to the anchor's world yaw and xy:
    # rebuilding the same relative window at a different yaw/xy gives
    # identical terms.
    yaw2, xy2 = -2.3, torch.tensor([-7.0, 4.0])
    b0_pos = torch.tensor([xy2[0], xy2[1], z0]).unsqueeze(0)
    b0_quat = _quat_xyzw(roll, pitch, yaw2).unsqueeze(0)
    _, heading_b0 = heading_anchor_frame(b0_pos, b0_quat)
    step2 = quat_apply(heading_b0, torch.tensor([[0.2, 0.0, 0.0]]))[0] + torch.tensor(
        [0.0, 0.0, 0.05]
    )
    b1_pos = b0_pos + step2
    b1_quat = _quat_xyzw(roll, pitch, yaw2 + 0.1).unsqueeze(0)
    window_pos2 = torch.stack([b0_pos[0], b1_pos[0]]).unsqueeze(0)
    window_quat2 = torch.cat([b0_quat, b1_quat]).unsqueeze(0)
    origin2, heading2 = heading_anchor_frame(b0_pos, b0_quat)
    pos_b2, quat_b2 = body_pose_in_anchor_frame(
        origin2, heading2, window_pos2, window_quat2
    )
    assert torch.allclose(pos_b, pos_b2, atol=_ATOL)
    dots = torch.abs((quat_b * quat_b2).sum(dim=-1))
    assert bool((dots > 1.0 - 1e-4).all())


def _data_plane_with_mode(mode: object) -> ExpertDataPlane:
    plane = object.__new__(ExpertDataPlane)
    plane._env = SimpleNamespace(  # type: ignore[assignment]
        cfg=SimpleNamespace(expert_macro_anchor_mode=mode),
        _command_ee_body_names=[],
        _command_keypoint_body_names=[],
    )
    return plane


def test_macro_builder_collapses_both_contexts_under_expert_heading():
    """Pretrain (expert) and live (rollout) macro windows must use ONE frame
    under expert_heading; under robot they must keep their split semantics.

    ``robot_heading`` is SONIC v1.1's convention: the live path anchors at the
    LIVE robot's heading frame, while the pretrain path -- which has no robot
    -- keeps the expert slot-0 heading frame.
    """
    for mode, expected in (
        ("expert_heading", {"expert": "expert_heading", "rollout": "expert_heading"}),
        ("robot", {"expert": "expert", "rollout": "rollout"}),
        (
            "robot_heading",
            {"expert": "expert_heading", "rollout": "robot_heading"},
        ),
    ):
        plane = _data_plane_with_mode(mode)
        seen: list[str] = []

        def _capture(_window, _env_ids, *, context, **_kwargs):
            seen.append(context)
            return {}

        plane._build_expert_window_terms = _capture  # type: ignore[method-assign]
        plane._expert_macro_feature_term_order = lambda: (  # type: ignore[method-assign]
            "expert_motion_qpos",
        )
        for caller_context in ("expert", "rollout"):
            seen.clear()
            plane._build_expert_macro_window_terms(
                None,  # type: ignore[arg-type]
                torch.zeros(1, dtype=torch.long),
                context=caller_context,
                past_steps=0,
            )
            assert seen == [expected[caller_context]], (
                f"mode={mode} caller={caller_context}: got {seen}"
            )


def _window_plane(robot_pos: torch.Tensor, robot_quat: torch.Tensor) -> ExpertDataPlane:
    """A data plane whose only live state is one robot anchor pose."""
    plane = _data_plane_with_mode("robot_heading")
    plane._env.device = torch.device("cpu")  # type: ignore[attr-defined]
    plane._get_joint_ids_tensor_fast = lambda ids: ids  # type: ignore[method-assign]
    plane._transform_reference_pose_to_world = (  # type: ignore[method-assign]
        lambda pos, quat=None, env_ids=None: (pos, quat)
    )
    plane._get_robot_anchor_state_w_fast = (  # type: ignore[method-assign]
        lambda _name: (robot_pos, robot_quat)
    )
    return plane


def test_robot_heading_context_uses_the_live_robot_heading_frame():
    """`robot_heading` must anchor at the LIVE robot, yaw-only.

    SONIC v1.1's encoder reads
    ``inv(get_heading_q(robot_anchor_quat_w)) * ref``. Two things are pinned
    here: the frame comes from the robot (not the expert slot), and only its
    heading is cancelled, so the reference keeps its tilt relative to gravity.
    """
    robot_pos = torch.tensor([[0.4, -0.7, 0.79]])
    robot_quat = _quat_xyzw(roll=0.18, pitch=-0.12, yaw=0.65).unsqueeze(0)
    ref_pos = torch.tensor([[[0.9, -0.2, 0.83], [1.1, 0.1, 0.86]]])
    ref_quat = torch.stack(
        [_quat_xyzw(0.05, 0.02, 0.9), _quat_xyzw(0.04, 0.03, 1.0)]
    ).unsqueeze(0)
    window = {
        "joint_pos": torch.zeros(1, 2, 3),
        "_macro_anchor_pos_w": ref_pos,
        "_macro_anchor_quat_w": ref_quat,
    }
    env_ids = torch.zeros(1, dtype=torch.long)

    plane = _window_plane(robot_pos, robot_quat)
    terms = plane._build_expert_window_terms(
        window,  # type: ignore[arg-type]
        env_ids,
        context="robot_heading",
        past_steps=0,
    )

    origin, heading = heading_anchor_frame(robot_pos, robot_quat)
    expected_pos, expected_quat = body_pose_in_anchor_frame(
        origin, heading, ref_pos, ref_quat
    )
    assert torch.allclose(
        terms["expert_anchor_pos_b"], expected_pos.reshape(1, -1), atol=_ATOL
    )

    # The full-pose "rollout" convention is a DIFFERENT frame whenever the
    # robot is tilted: it also cancels the robot's roll/pitch.
    rollout_terms = _window_plane(robot_pos, robot_quat)._build_expert_window_terms(
        window,  # type: ignore[arg-type]
        env_ids,
        context="rollout",
        past_steps=0,
    )
    assert not torch.allclose(
        rollout_terms["expert_anchor_pos_b"],
        terms["expert_anchor_pos_b"],
        atol=1e-3,
    )


def test_unknown_anchor_mode_is_refused():
    plane = _data_plane_with_mode("world")
    with pytest.raises(ValueError, match="expert_macro_anchor_mode"):
        plane._expert_macro_anchor_mode()


def test_missing_cfg_field_defaults_to_robot():
    plane = object.__new__(ExpertDataPlane)
    plane._env = SimpleNamespace(cfg=SimpleNamespace())  # type: ignore[assignment]
    assert plane._expert_macro_anchor_mode() == "robot"
