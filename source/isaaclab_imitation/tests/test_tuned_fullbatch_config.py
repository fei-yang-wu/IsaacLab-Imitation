"""Contracts for the full-batch tuned optimizer geometry (2026-08-30).

Two things must hold, and both are easy to break silently:

* the frozen ``rlopt_ipmd_tuned_cfg_entry_point`` contract keeps its exact
  optimizer geometry, so the 46.5B/50B chains stay reproducible;
* the new config resolves to ONE minibatch per epoch at any environment count,
  which it achieves through the clamp in ``scripts/rlopt/train_impl.py`` rather
  than by naming a literal batch size.
"""

from __future__ import annotations

import gymnasium as gym
import pytest

import isaaclab_imitation.tasks  # noqa: F401  (registers the gym tasks)
from isaaclab_imitation.tasks.manager_based.imitation.config.g1.agents.rlopt_ipmd_cfg import (
    G1ImitationTunedFullBatchLinearLRRLOptIPMDConfig,
    G1ImitationTunedFullBatchRLOptIPMDConfig,
    G1ImitationTunedRLOptIPMDConfig,
)

TUNED_TASK_IDS = (
    "Isaac-Imitation-G1-v2",
    "Isaac-Imitation-G1-Explicit-v2",
    "Isaac-Imitation-G1-Chunk-v2",
)


def _resolved_minibatch(mini_batch_size: int, num_envs: int, horizon: int) -> int:
    """Reproduce `train_impl.py`'s on-policy clamp: minibatch <= collected batch."""
    return min(int(mini_batch_size), num_envs * horizon)


def test_tuned_contract_is_unchanged():
    """The frozen recipe: 5 epochs and a minibatch well below the batch."""
    cfg = G1ImitationTunedRLOptIPMDConfig()
    assert cfg.loss.epochs == 5
    assert cfg.loss.mini_batch_size == 4096 * 24 // 4


@pytest.mark.parametrize("num_envs", [4096, 16384, 20480, 24576])
def test_fullbatch_resolves_to_one_minibatch_per_epoch(num_envs):
    cfg = G1ImitationTunedFullBatchRLOptIPMDConfig()
    horizon = cfg.collector.frames_per_batch  # per-env rollout steps, 24
    batch = num_envs * horizon
    resolved = _resolved_minibatch(cfg.loss.mini_batch_size, num_envs, horizon)

    assert resolved == batch, "one gradient step per epoch requires the whole batch"
    # 3 epochs x 1 minibatch, against the tuned recipe's 5 x 2 at the 3/4 rule.
    assert cfg.loss.epochs == 3


def test_fullbatch_changes_only_the_optimizer_geometry():
    """Everything outside `loss.epochs` / `loss.mini_batch_size` must match."""
    tuned = G1ImitationTunedRLOptIPMDConfig()
    full = G1ImitationTunedFullBatchRLOptIPMDConfig()

    assert full.ipmd.actor_learning_rate == tuned.ipmd.actor_learning_rate
    assert full.ipmd.critic_learning_rate == tuned.ipmd.critic_learning_rate
    assert full.optim.min_lr == tuned.optim.min_lr
    assert full.optim.max_lr == tuned.optim.max_lr
    assert full.optim.max_grad_norm == tuned.optim.max_grad_norm
    assert full.optim.desired_kl == tuned.optim.desired_kl
    assert full.optim.kl_adapt_step == tuned.optim.kl_adapt_step
    assert full.optim.weight_decay == tuned.optim.weight_decay
    assert full.ppo.entropy_coeff == tuned.ppo.entropy_coeff
    assert full.ppo.clip_epsilon == tuned.ppo.clip_epsilon
    assert full.ppo.clip_log_std == tuned.ppo.clip_log_std
    assert full.loss.gamma == tuned.loss.gamma
    assert full.collector.frames_per_batch == tuned.collector.frames_per_batch
    assert full.policy.num_cells == tuned.policy.num_cells
    assert full.value_function is not None and tuned.value_function is not None
    assert full.value_function.num_cells == tuned.value_function.num_cells


@pytest.mark.parametrize("task_id", TUNED_TASK_IDS)
def test_both_entry_points_are_registered(task_id):
    """The frozen entry point stays; the new one is additive, never a redirect."""
    kwargs = gym.spec(task_id).kwargs
    tuned = kwargs["rlopt_ipmd_tuned_cfg_entry_point"]
    full = kwargs["rlopt_ipmd_tuned_fullbatch_cfg_entry_point"]

    assert tuned.endswith(":G1ImitationTunedRLOptIPMDConfig")
    assert full.endswith(":G1ImitationTunedFullBatchRLOptIPMDConfig")
    assert tuned != full


def test_linearlr_moves_only_the_actor_schedule():
    """Linear actor decay: scheduler, start lr, and the critic pin; nothing else."""
    full = G1ImitationTunedFullBatchRLOptIPMDConfig()
    lin = G1ImitationTunedFullBatchLinearLRRLOptIPMDConfig()

    assert full.optim.scheduler == "adaptive"
    assert lin.optim.scheduler == "linearlr"
    assert lin.optim.scheduler_step == "update"
    assert set(lin.optim.scheduler_kwargs) == {
        "start_factor",
        "end_factor",
        "total_iters",
    }
    assert lin.optim.scheduler_kwargs["start_factor"] == 1.0
    assert lin.ipmd.actor_learning_rate == 2.0e-4
    # 2e-4 -> 1e-5, the adaptive rule's floor.
    assert lin.ipmd.actor_learning_rate * lin.optim.scheduler_kwargs[
        "end_factor"
    ] == pytest.approx(full.optim.min_lr)
    # The critic is pinned, not decayed: final == start.
    assert lin.ipmd.critic_lr_schedule == "linear"
    assert lin.ipmd.critic_learning_rate == full.ipmd.critic_learning_rate
    assert lin.ipmd.critic_lr_final == lin.ipmd.critic_learning_rate

    assert lin.loss.epochs == full.loss.epochs
    assert lin.loss.mini_batch_size == full.loss.mini_batch_size
    assert lin.optim.weight_decay == full.optim.weight_decay
    assert lin.optim.max_grad_norm == full.optim.max_grad_norm
    assert lin.ppo.entropy_coeff == full.ppo.entropy_coeff
    assert lin.ppo.clip_epsilon == full.ppo.clip_epsilon
    assert lin.loss.gamma == full.loss.gamma
    assert lin.policy.num_cells == full.policy.num_cells


@pytest.mark.parametrize("task_id", TUNED_TASK_IDS)
def test_linearlr_entry_point_is_registered(task_id):
    kwargs = gym.spec(task_id).kwargs
    lin = kwargs["rlopt_ipmd_tuned_fullbatch_linearlr_cfg_entry_point"]
    assert lin.endswith(":G1ImitationTunedFullBatchLinearLRRLOptIPMDConfig")
    assert lin != kwargs["rlopt_ipmd_tuned_fullbatch_cfg_entry_point"]
