"""Trajectory-wise train/validation split for the interface planners.

The property under test is that no trajectory contributes rows to both sides.
A row-wise split satisfies every shape assertion and still leaks, because
consecutive planner publishes within one rollout share nearly all of their state
history -- so these tests check membership, not sizes.
"""

from __future__ import annotations


import pytest
import torch
from imitation_experiments.planner.train_chunked_transformer_planner import _trajectory_split  # noqa: E402


def _samples(num_envs: int, episodes_per_env: int, rows_per_episode: int) -> dict:
    env_id, episode_id = [], []
    for env in range(num_envs):
        for episode in range(episodes_per_env):
            env_id += [env] * rows_per_episode
            episode_id += [episode] * rows_per_episode
    rows = len(env_id)
    return {
        "causal_target": torch.zeros(rows, 4),
        "env_id": torch.tensor(env_id),
        "episode_id": torch.tensor(episode_id),
    }


def test_split_is_by_trajectory_not_by_row() -> None:
    data = _samples(num_envs=10, episodes_per_env=10, rows_per_episode=50)
    train, val, summary = _trajectory_split(data, val_fraction=0.2, seed=0)

    assert summary["num_trajectories"] == 100
    assert summary["num_val_trajectories"] == 20
    assert summary["num_train_trajectories"] == 80
    assert summary["num_train_rows"] == 80 * 50
    assert summary["num_val_rows"] == 20 * 50
    assert train.numel() + val.numel() == 100 * 50
    assert set(train.tolist()).isdisjoint(val.tolist())

    def trajectories(rows: torch.Tensor) -> set[tuple[int, int]]:
        return {
            (int(data["env_id"][i]), int(data["episode_id"][i])) for i in rows.tolist()
        }

    # The actual guarantee: a trajectory is wholly in one split or the other.
    assert trajectories(train).isdisjoint(trajectories(val))
    assert len(trajectories(val)) == 20


def test_episode_id_alone_would_merge_environments() -> None:
    """env 0 episode 1 and env 3 episode 1 are different rollouts.

    Grouping on episode_id alone would see 10 trajectories here instead of 100,
    and any split of those 10 would put rows from the same environment on both
    sides. The split must use the pair.
    """
    data = _samples(num_envs=10, episodes_per_env=10, rows_per_episode=50)
    _, _, summary = _trajectory_split(data, val_fraction=0.2, seed=0)
    assert summary["num_trajectories"] == 100, (
        "grouping collapsed to episode_id alone, which merges environments"
    )


def test_split_is_deterministic_for_a_seed_and_varies_across_seeds() -> None:
    data = _samples(num_envs=10, episodes_per_env=10, rows_per_episode=50)
    a, _, _ = _trajectory_split(data, val_fraction=0.2, seed=0)
    b, _, _ = _trajectory_split(data, val_fraction=0.2, seed=0)
    c, _, _ = _trajectory_split(data, val_fraction=0.2, seed=1)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_disabled_split_returns_every_row_for_training() -> None:
    data = _samples(num_envs=2, episodes_per_env=2, rows_per_episode=5)
    train, val, summary = _trajectory_split(data, val_fraction=0.0, seed=0)
    assert train.numel() == 20
    assert val.numel() == 0
    assert summary == {}


def test_missing_env_id_is_an_error_not_a_silent_fallback() -> None:
    data = _samples(num_envs=2, episodes_per_env=2, rows_per_episode=5)
    del data["env_id"]
    with pytest.raises(KeyError, match="env_id"):
        _trajectory_split(data, val_fraction=0.2, seed=0)


def test_placeholder_env_id_from_legacy_writers_is_rejected() -> None:
    data = _samples(num_envs=2, episodes_per_env=2, rows_per_episode=5)
    data["env_id"] = torch.full_like(data["env_id"], -1)
    with pytest.raises(ValueError, match="-1 placeholders"):
        _trajectory_split(data, val_fraction=0.2, seed=0)


def test_fraction_too_small_for_the_budget_is_rejected() -> None:
    # 4 trajectories at 10% rounds to 0 held out -- a silently empty validation
    # set would report a meaningless number rather than fail.
    data = _samples(num_envs=2, episodes_per_env=2, rows_per_episode=5)
    with pytest.raises(ValueError, match="trajectories for validation"):
        _trajectory_split(data, val_fraction=0.1, seed=0)
