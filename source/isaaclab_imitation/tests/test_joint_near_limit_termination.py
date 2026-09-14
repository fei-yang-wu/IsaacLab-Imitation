"""joint_pos_near_hard_limit ends the episode on the measured joint, not the target."""

from __future__ import annotations

import types

import torch

from isaaclab_imitation.tasks.manager_based.imitation.mdp.terminations import (
    joint_pos_near_hard_limit,
)


def _env(joint_pos: torch.Tensor, soft: tuple[float, float], hard: tuple[float, float]):
    num_envs, dim = joint_pos.shape
    soft_t = torch.zeros(num_envs, dim, 2)
    soft_t[..., 0], soft_t[..., 1] = soft
    hard_t = torch.zeros(num_envs, dim, 2)
    hard_t[..., 0], hard_t[..., 1] = hard
    asset = types.SimpleNamespace(
        data=types.SimpleNamespace(
            joint_pos=types.SimpleNamespace(torch=joint_pos),
            soft_joint_pos_limits=types.SimpleNamespace(torch=soft_t),
            joint_pos_limits=types.SimpleNamespace(torch=hard_t),
        )
    )
    return types.SimpleNamespace(
        scene={"robot": asset}, num_envs=num_envs, device="cpu"
    )


def _cfg(dim: int):
    return types.SimpleNamespace(name="robot", joint_ids=slice(0, dim))


SOFT = (-0.803, 0.454)  # G1 ankle pitch training limits
HARD = (-0.8727, 0.5236)


def test_disabled_by_default_never_fires() -> None:
    env = _env(torch.tensor([[0.60, 0.0], [-0.95, 0.0]]), SOFT, HARD)
    assert joint_pos_near_hard_limit(env, _cfg(2), fraction=None).tolist() == [
        False,
        False,
    ]


def test_fires_past_the_fraction_of_the_soft_to_hard_gap() -> None:
    # fraction 0.5 -> upper bar 0.454 + 0.5 * 0.0696 = 0.4888, lower -0.803 - 0.5 * 0.0697 = -0.8378
    joint_pos = torch.tensor([[0.48, 0.0], [0.50, 0.0], [-0.83, 0.0], [-0.85, 0.0]])
    env = _env(joint_pos, SOFT, HARD)
    assert joint_pos_near_hard_limit(env, _cfg(2), fraction=0.5).tolist() == [
        False,
        True,
        False,
        True,
    ]


def test_fraction_zero_is_the_soft_limit_and_one_the_hard_limit() -> None:
    env = _env(torch.tensor([[0.46], [0.53]]), SOFT, HARD)
    assert joint_pos_near_hard_limit(env, _cfg(1), fraction=0.0).tolist() == [
        True,
        True,
    ]
    assert joint_pos_near_hard_limit(env, _cfg(1), fraction=1.0).tolist() == [
        False,
        True,
    ]


def test_any_joint_ends_the_episode() -> None:
    env = _env(torch.tensor([[0.0, 0.0, 0.52]]), SOFT, HARD)
    assert joint_pos_near_hard_limit(env, _cfg(3), fraction=0.5).tolist() == [True]
