"""`action_acc_l2`: the second difference of the raw action, per environment.

The term keeps its own ``a_{t-2}`` buffer because ``ActionManager`` exposes only
``action`` and ``prev_action``. These tests drive it with a stub manager so no
simulator starts.
"""

from __future__ import annotations

import types

import pytest
import torch

torch.manual_seed(0)

isaaclab_managers = pytest.importorskip("isaaclab.managers")
from isaaclab_imitation.tasks.manager_based.imitation.mdp.rewards import (  # noqa: E402
    action_acc_l2,
)


def _env(num_envs: int, width: int) -> types.SimpleNamespace:
    manager = types.SimpleNamespace(
        action=torch.zeros(num_envs, width),
        prev_action=torch.zeros(num_envs, width),
    )
    return types.SimpleNamespace(num_envs=num_envs, device="cpu", action_manager=manager)


def _term(env) -> action_acc_l2:
    cfg = isaaclab_managers.RewardTermCfg(func=action_acc_l2, weight=1.0)
    return action_acc_l2(cfg, env)


def _step(env, new_action: torch.Tensor) -> None:
    env.action_manager.prev_action = env.action_manager.action.clone()
    env.action_manager.action = new_action.clone()


def test_constant_ramp_scores_zero_and_alternation_does_not() -> None:
    """A constant-velocity ramp has zero second difference; a zigzag does not."""
    env = _env(2, 3)
    term = _term(env)
    ramp = torch.tensor([[0.1, 0.1, 0.1], [0.0, 0.0, 0.0]])
    zig = torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.2, 0.2]])
    # env 0 ramps by +0.1 per step; env 1 alternates 0, 0.2, 0, 0.2 ...
    _step(env, torch.stack([ramp[0] * 1, zig[1]]))
    term(env)
    _step(env, torch.stack([ramp[0] * 2, zig[0]]))
    term(env)
    _step(env, torch.stack([ramp[0] * 3, zig[1]]))
    out = term(env)
    assert torch.isclose(out[0], torch.tensor(0.0), atol=1e-6), out
    # a_t - 2 a_{t-1} + a_{t-2} = 0.2 - 0 + 0.2 = 0.4 per joint, squared, x3 joints
    assert torch.isclose(out[1], torch.tensor(3 * 0.4**2), atol=1e-6), out


def test_matches_explicit_second_difference_over_a_random_sequence() -> None:
    env = _env(4, 29)
    term = _term(env)
    history = [torch.zeros(4, 29), torch.zeros(4, 29)]
    for _ in range(6):
        nxt = torch.randn(4, 29)
        _step(env, nxt)
        out = term(env)
        history.append(nxt)
        a_t, a_1, a_2 = history[-1], history[-2], history[-3]
        expected = torch.sum((a_t - 2 * a_1 + a_2) ** 2, dim=1)
        torch.testing.assert_close(out, expected)


def test_reset_clears_the_buffer_for_reset_envs_only() -> None:
    """A reset environment must not carry the previous episode's a_{t-2}."""
    env = _env(3, 2)
    term = _term(env)
    for value in (1.0, 2.0, 3.0):
        _step(env, torch.full((3, 2), value))
        term(env)
    # Reset env 1 only. Its buffer becomes the current action, so the next
    # step scores against a flat history; env 0 and 2 keep theirs.
    term.reset(env_ids=[1])
    _step(env, torch.full((3, 2), 4.0))
    out = term(env)
    # unreset envs: 4 - 2*3 + 2 = 0 (the ramp continues) -> zero
    assert torch.isclose(out[0], torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(out[2], torch.tensor(0.0), atol=1e-6)
    # reset env: a_{t-2} was overwritten with 3.0 (the action at reset time),
    # a_{t-1} = 3.0, a_t = 4.0 -> 4 - 6 + 3 = 1 per joint, squared, x2 joints
    assert torch.isclose(out[1], torch.tensor(2.0), atol=1e-6), out


def test_torque_and_energy_terms_are_registered_off() -> None:
    """The v2 reward set carries `joint_torques_l2` and `energy_consumption`
    at weight 0.0 so existing rows stay byte-identical until a campaign
    turns them on."""
    from isaaclab_imitation.tasks.manager_based.imitation.config.g1.common.rewards import (
        G1SonicRewardsCfg,
    )

    cfg = G1SonicRewardsCfg()
    assert cfg.joint_torques_l2.weight == 0.0
    assert cfg.joint_torques_l2.func.__name__ == "joint_torques_l2"
    assert cfg.energy_consumption.weight == 0.0
