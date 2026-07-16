from __future__ import annotations

import pytest
import torch

from planner_publish_schedule import planner_renew_env_ids


def test_renewals_follow_each_environments_episode_step() -> None:
    episode_steps = torch.tensor([0, 1, 9, 10, 11, 20], dtype=torch.long)

    renew_env_ids = planner_renew_env_ids(episode_steps, 10)

    assert torch.equal(renew_env_ids, torch.tensor([0, 3, 5]))


def test_async_resets_renew_independently_of_other_environments() -> None:
    # The initial call publishes for every environment even when attaching to
    # counters that did not all start at zero.
    episode_steps = torch.tensor([2, 1, 3], dtype=torch.long)
    assert torch.equal(
        planner_renew_env_ids(
            episode_steps,
            4,
            initial_publication=True,
        ),
        torch.tensor([0, 1, 2]),
    )

    # Environment 1 reset asynchronously, while environment 2 independently
    # reached its regular publication boundary.
    episode_steps = torch.tensor([3, 0, 4], dtype=torch.long)
    assert torch.equal(
        planner_renew_env_ids(episode_steps, 4),
        torch.tensor([1, 2]),
    )

    # A later reset of environment 0 renews only that environment.
    episode_steps = torch.tensor([0, 1, 5], dtype=torch.long)
    assert torch.equal(
        planner_renew_env_ids(episode_steps, 4),
        torch.tensor([0]),
    )


def test_empty_batch_returns_empty_long_tensor() -> None:
    renew_env_ids = planner_renew_env_ids(torch.empty(0, dtype=torch.int32), 5)

    assert renew_env_ids.dtype == torch.long
    assert renew_env_ids.shape == (0,)


@pytest.mark.parametrize("interval", [0, -1])
def test_interval_must_be_positive(interval: int) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        planner_renew_env_ids(torch.zeros(2, dtype=torch.long), interval)


def test_episode_steps_must_be_one_dimensional_integers() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        planner_renew_env_ids(torch.zeros(2, 1, dtype=torch.long), 4)
    with pytest.raises(TypeError, match="integer dtype"):
        planner_renew_env_ids(torch.zeros(2), 4)
