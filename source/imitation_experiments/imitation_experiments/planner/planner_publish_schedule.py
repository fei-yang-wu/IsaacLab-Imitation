"""Per-environment publication schedule for explicit interface planners."""

from __future__ import annotations

import torch


def planner_renew_env_ids(
    episode_length_buf: torch.Tensor,
    planner_interval_steps: int,
    *,
    initial_publication: bool = False,
) -> torch.Tensor:
    """Return environment IDs that need a newly published planner command.

    Normal renewals occur at episode-local control steps ``0, interval, ...``.
    Because reset environments return to episode step zero, this schedule keeps
    independently resetting vectorized environments aligned without relying on
    a global rollout step. Set ``initial_publication`` on the first call to
    publish for every environment even if its episode counter is not zero.

    The returned one-dimensional ``torch.long`` tensor stays on the same device
    as ``episode_length_buf``.
    """
    if episode_length_buf.ndim != 1:
        raise ValueError(
            "episode_length_buf must be one-dimensional, "
            f"got shape {tuple(episode_length_buf.shape)}."
        )
    if (
        episode_length_buf.dtype == torch.bool
        or episode_length_buf.is_floating_point()
        or episode_length_buf.is_complex()
    ):
        raise TypeError("episode_length_buf must use an integer dtype.")

    interval = int(planner_interval_steps)
    if interval <= 0:
        raise ValueError("planner_interval_steps must be positive.")

    if initial_publication:
        return torch.arange(
            episode_length_buf.numel(),
            device=episode_length_buf.device,
            dtype=torch.long,
        )

    renew_mask = episode_length_buf.remainder(interval) == 0
    return torch.nonzero(renew_mask, as_tuple=False).flatten()
