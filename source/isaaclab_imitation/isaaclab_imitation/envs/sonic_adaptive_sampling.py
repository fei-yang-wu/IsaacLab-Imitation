from __future__ import annotations

import torch


class SonicAdaptiveResetSampler:
    """Failure-aware motion/frame sampler matching SONIC's public motion library.

    Each trajectory is split into fixed-size frame bins. Sampling probabilities
    are based on the observed failure rate in each bin, blended with a uniform
    component and weighted so long sequences do not dominate solely because
    they contain more bins. A sampled frame is shifted backwards by a random
    lead-in so the policy can act before the difficult segment.
    """

    def __init__(
        self,
        trajectory_lengths: torch.Tensor,
        *,
        bin_size: int = 50,
        sequence_length_agnostic: bool = True,
        init_num_failures: float = 1.0,
        uniform_sampling_rate: float = 0.1,
        pre_failure_sample_window: int = 200,
        failure_rate_max_over_mean: float = 200.0,
    ) -> None:
        lengths = torch.as_tensor(
            trajectory_lengths,
            dtype=torch.long,
            device=trajectory_lengths.device,
        ).reshape(-1)
        if lengths.numel() == 0:
            raise ValueError("trajectory_lengths must contain at least one trajectory.")
        if torch.any(lengths <= 0):
            raise ValueError("trajectory_lengths must all be positive.")
        if int(bin_size) <= 0:
            raise ValueError("bin_size must be positive.")
        if float(init_num_failures) <= 0.0:
            raise ValueError("init_num_failures must be positive.")
        if not 0.0 <= float(uniform_sampling_rate) <= 1.0:
            raise ValueError("uniform_sampling_rate must be in [0, 1].")
        if int(pre_failure_sample_window) < 0:
            raise ValueError("pre_failure_sample_window must be >= 0.")
        if float(failure_rate_max_over_mean) <= 0.0:
            raise ValueError("failure_rate_max_over_mean must be positive.")

        self.device = lengths.device
        self.trajectory_lengths = lengths
        self.bin_size = int(bin_size)
        self.sequence_length_agnostic = bool(sequence_length_agnostic)
        self.uniform_sampling_rate = float(uniform_sampling_rate)
        self.pre_failure_sample_window = int(pre_failure_sample_window)
        self.failure_rate_max_over_mean = float(failure_rate_max_over_mean)

        bins: list[torch.Tensor] = []
        trajectory_bin_ids: list[torch.Tensor] = []
        next_bin_id = 0
        for trajectory_rank, trajectory_length in enumerate(lengths.tolist()):
            starts = torch.arange(
                0,
                trajectory_length,
                self.bin_size,
                device=self.device,
                dtype=torch.long,
            )
            ends = torch.minimum(
                starts + self.bin_size,
                torch.full_like(starts, trajectory_length),
            )
            ranks = torch.full_like(starts, trajectory_rank)
            bins.append(torch.stack((ranks, starts, ends), dim=-1))
            trajectory_bin_ids.append(
                torch.arange(
                    next_bin_id,
                    next_bin_id + starts.numel(),
                    device=self.device,
                    dtype=torch.long,
                )
            )
            next_bin_id += starts.numel()

        self.bins = torch.cat(bins, dim=0)
        self.num_bins = int(self.bins.shape[0])
        self.trajectory_bin_ids = trajectory_bin_ids
        self.first_bin_ids = torch.stack([ids[0] for ids in trajectory_bin_ids])
        self.bin_lengths = self.bins[:, 2] - self.bins[:, 1]
        peer_bin_counts = torch.empty(
            self.num_bins, device=self.device, dtype=torch.float32
        )
        for ids in trajectory_bin_ids:
            peer_bin_counts.index_fill_(0, ids, float(ids.numel()))

        self.bin_weights = self.bin_lengths.to(dtype=torch.float32)
        self.bin_weights /= self.bin_weights.mean()
        if self.sequence_length_agnostic:
            self.bin_weights /= peer_bin_counts

        initial_count = float(init_num_failures)
        self.num_failures = torch.full(
            (self.num_bins,), initial_count, device=self.device, dtype=torch.float32
        )
        self.num_visits = torch.full_like(self.num_failures, initial_count)

    def _bin_ids(
        self, trajectory_ranks: torch.Tensor, frame_steps: torch.Tensor
    ) -> torch.Tensor:
        ranks = torch.as_tensor(
            trajectory_ranks, device=self.device, dtype=torch.long
        ).reshape(-1)
        steps = torch.as_tensor(
            frame_steps, device=self.device, dtype=torch.long
        ).reshape(-1)
        if ranks.shape != steps.shape:
            raise ValueError(
                "trajectory_ranks and frame_steps must have matching shapes."
            )
        if torch.any((ranks < 0) | (ranks >= self.trajectory_lengths.numel())):
            raise ValueError("trajectory_ranks contains an out-of-range value.")
        max_steps = self.trajectory_lengths.index_select(0, ranks) - 1
        steps = torch.minimum(torch.maximum(steps, torch.zeros_like(steps)), max_steps)
        local_bins = torch.div(steps, self.bin_size, rounding_mode="floor")
        return self.first_bin_ids.index_select(0, ranks) + local_bins

    def record_visits(
        self, trajectory_ranks: torch.Tensor, frame_steps: torch.Tensor
    ) -> None:
        """Record one control-step visit per environment, normalized by bin length."""
        bin_ids = self._bin_ids(trajectory_ranks, frame_steps)
        counts = torch.bincount(bin_ids, minlength=self.num_bins).to(torch.float32)
        self.num_visits.add_(counts / self.bin_lengths.to(torch.float32))

    def record_failures(
        self, trajectory_ranks: torch.Tensor, frame_steps: torch.Tensor
    ) -> None:
        """Record terminal tracking failures at their motion-local frame bins."""
        if torch.as_tensor(trajectory_ranks).numel() == 0:
            return
        bin_ids = self._bin_ids(trajectory_ranks, frame_steps)
        self.num_failures.add_(
            torch.bincount(bin_ids, minlength=self.num_bins).to(torch.float32)
        )

    def sampling_probabilities(self) -> torch.Tensor:
        """Return the current SONIC-style global trajectory-bin distribution."""
        failure_rate = self.num_failures / self.num_visits
        upper_bound = failure_rate.mean() * self.failure_rate_max_over_mean
        clipped = torch.clamp(failure_rate, min=0.0, max=upper_bound)
        clipped_sum = clipped.sum()
        if not torch.isfinite(clipped_sum) or clipped_sum <= 0.0:
            failure_prob = torch.full_like(clipped, 1.0 / self.num_bins)
        else:
            failure_prob = clipped / clipped_sum
        uniform_prob = torch.full_like(failure_prob, 1.0 / self.num_bins)
        probabilities = (
            failure_prob * (1.0 - self.uniform_sampling_rate)
            + uniform_prob * self.uniform_sampling_rate
        )
        probabilities *= self.bin_weights
        return probabilities / probabilities.sum()

    def sample(self, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample trajectory ranks and random local starts with SONIC's lead-in."""
        count = int(count)
        if count < 0:
            raise ValueError("count must be >= 0.")
        if count == 0:
            empty = torch.empty(0, device=self.device, dtype=torch.long)
            return empty, empty
        sampled_bin_ids = torch.multinomial(
            self.sampling_probabilities(), count, replacement=True
        )
        sampled_bins = self.bins.index_select(0, sampled_bin_ids)
        trajectory_ranks = sampled_bins[:, 0]
        bin_starts = sampled_bins[:, 1]
        bin_ends = sampled_bins[:, 2]
        frame_steps = (
            torch.rand(count, device=self.device) * (bin_ends - bin_starts)
        ).floor().to(torch.long) + bin_starts
        if self.pre_failure_sample_window > 0:
            lead_in = torch.randint(
                self.pre_failure_sample_window,
                (count,),
                device=self.device,
                dtype=torch.long,
            )
            frame_steps = (frame_steps - lead_in).clamp_min(0)
        return trajectory_ranks, frame_steps
