"""Reset samplers owned by the imitation environment.

The official SONIC sampler remains in :mod:`iltools.datasets.reset_sampling`.
This module contains repo-specific compositions around that sampler so they do
not change the reproduced SONIC behavior.
"""

from __future__ import annotations

import torch

from iltools.datasets.reset_sampling import SonicAdaptiveResetSampler


class RandomTrajectoryAdaptiveResetSampler:
    """Mix uniform-trajectory starts with SONIC adaptive starts.

    The random branch first chooses a trajectory uniformly, then chooses a
    local frame uniformly from the leading ``random_start_fraction`` of that
    trajectory.  The adaptive branch delegates unchanged to ``adaptive``.

    This is deliberately different from SONIC's ``uniform_sampling_rate``:
    that setting mixes probability over trajectory-frame bins before applying
    sequence-length weights, while this class makes an explicit per-reset
    branch choice and gives every trajectory equal probability on the random
    branch.
    """

    def __init__(
        self,
        lengths: torch.Tensor,
        *,
        adaptive: SonicAdaptiveResetSampler,
        random_sampling_ratio: float = 0.8,
        random_start_fraction: float = 0.5,
        generator: torch.Generator | None = None,
    ) -> None:
        if lengths.ndim != 1 or lengths.numel() == 0:
            raise ValueError("lengths must be a non-empty one-dimensional tensor.")
        if torch.any(lengths <= 0):
            raise ValueError("All trajectory lengths must be positive.")
        if not 0.0 <= float(random_sampling_ratio) <= 1.0:
            raise ValueError("random_sampling_ratio must be in [0, 1].")
        if not 0.0 < float(random_start_fraction) <= 1.0:
            raise ValueError("random_start_fraction must be in (0, 1].")

        self.lengths = lengths.to(dtype=torch.long)
        self.adaptive = adaptive
        self.random_sampling_ratio = float(random_sampling_ratio)
        self.random_start_fraction = float(random_start_fraction)
        self.generator = generator

    def sampling_probabilities(self) -> torch.Tensor:
        """Return the current adaptive-branch probability snapshot."""
        return self.adaptive.sampling_probabilities()

    def sample(
        self,
        batch_size: int,
        *,
        probabilities: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample trajectory ranks and local frames for ``batch_size`` resets."""
        count = int(batch_size)
        if count < 0:
            raise ValueError("batch_size must be non-negative.")
        if count == 0:
            empty = torch.empty(0, device=self.lengths.device, dtype=torch.long)
            return empty, empty.clone()
        if self.random_sampling_ratio <= 0.0:
            return self.adaptive.sample(count, probabilities=probabilities)

        device = self.lengths.device
        random_mask = (
            torch.rand(count, device=device, generator=self.generator)
            < self.random_sampling_ratio
        )
        if self.random_sampling_ratio >= 1.0:
            random_mask.fill_(True)

        # Sampling full candidate batches avoids a device synchronization just
        # to discover the stochastic branch counts. At reset-batch scale this
        # is cheaper than pulling ``random_mask.sum()`` back to Python.
        adaptive_ranks, adaptive_steps = self.adaptive.sample(
            count,
            probabilities=probabilities,
        )
        random_ranks = torch.randint(
            int(self.lengths.numel()),
            (count,),
            device=device,
            generator=self.generator,
        )
        selected_lengths = self.lengths.index_select(0, random_ranks)
        # ceil(fraction * length) makes the half-open range non-empty even for
        # a one-frame trajectory and defines 50% of odd lengths without
        # silently discarding their middle frame.
        random_span = torch.ceil(
            selected_lengths.to(dtype=torch.float64) * self.random_start_fraction
        ).to(dtype=torch.long)
        random_steps = torch.floor(
            torch.rand(
                count,
                device=device,
                dtype=torch.float64,
                generator=self.generator,
            )
            * random_span
        ).to(dtype=torch.long)
        return (
            torch.where(random_mask, random_ranks, adaptive_ranks),
            torch.where(random_mask, random_steps, adaptive_steps),
        )


__all__ = ["RandomTrajectoryAdaptiveResetSampler"]
