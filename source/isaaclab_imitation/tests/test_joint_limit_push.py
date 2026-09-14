"""The joint_limit_push reward charges outward PD torque past the soft limit only."""

from __future__ import annotations

import types

import torch

from isaaclab_imitation.tasks.manager_based.imitation.mdp.rewards import (
    joint_limit_push,
)


def _env(joint_pos: torch.Tensor, torque: torch.Tensor, lower: float, upper: float):
    num_envs, dim = joint_pos.shape
    limits = torch.zeros(num_envs, dim, 2)
    limits[..., 0] = lower
    limits[..., 1] = upper
    asset = types.SimpleNamespace(
        data=types.SimpleNamespace(
            joint_pos=types.SimpleNamespace(torch=joint_pos),
            soft_joint_pos_limits=types.SimpleNamespace(torch=limits),
            applied_torque=types.SimpleNamespace(torch=torque),
        )
    )
    return types.SimpleNamespace(scene={"robot": asset})


def _cfg(dim: int):
    return types.SimpleNamespace(name="robot", joint_ids=slice(0, dim))


def test_zero_inside_the_range_whatever_the_torque() -> None:
    joint_pos = torch.tensor([[0.0, 0.40, -0.79]])
    torque = torch.tensor([[50.0, -50.0, 80.0]])
    env = _env(joint_pos, torque, lower=-0.803, upper=0.454)

    assert joint_limit_push(env, _cfg(3)).tolist() == [0.0]


def test_charges_only_outward_torque_past_each_limit() -> None:
    # joint 0: past the upper limit, pushed further out (charged)
    # joint 1: past the upper limit, pulled back (free)
    # joint 2: past the lower limit, pushed further out (charged)
    # joint 3: past the lower limit, pulled back (free)
    joint_pos = torch.tensor([[0.50, 0.50, -0.90, -0.90]])
    torque = torch.tensor([[30.0, -30.0, -12.0, 12.0]])
    env = _env(joint_pos, torque, lower=-0.803, upper=0.454)

    assert joint_limit_push(env, _cfg(4)).tolist() == [42.0]


def test_per_environment_rows_are_independent() -> None:
    joint_pos = torch.tensor([[0.50, 0.0], [0.0, 0.50]])
    torque = torch.tensor([[10.0, 99.0], [99.0, 5.0]])
    env = _env(joint_pos, torque, lower=-0.803, upper=0.454)

    assert joint_limit_push(env, _cfg(2)).tolist() == [10.0, 5.0]
