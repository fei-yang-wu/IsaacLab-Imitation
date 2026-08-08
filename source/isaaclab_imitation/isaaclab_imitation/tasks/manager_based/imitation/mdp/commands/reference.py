# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The reference command channel: the privileged, always-present channel.

:class:`ReferenceCommandTerm` is what the environment tracks. It is always
dataset-backed -- in every mode, including planner evaluation, where the
reference drives nothing and exists only to score the rollout -- and it owns:

* **selection**: which trajectory and which start frame each environment resets
  onto (:class:`ReferenceSelectionCfg`). Trajectory choice is the parallel
  trajectory manager's schedule (including ``custom``, where an evaluation
  driver or per-goal collector supplies the ranks); start-frame choice is the
  ``StartFrameSampler`` / adaptive full-trajectory sampler this term builds
  from its own config.
* **the adaptive-failure bookkeeping** those samplers consume
  (:meth:`record_visits` / :meth:`record_failures`).
* **the tracking metrics** ``Metrics/reference/{mpjpe_mm, anchor_pos_err_m,
  anchor_ori_err_rad}``, logged natively by Isaac Lab's CommandManager.
* **emission** of any command component for privileged consumers -- the critic's
  command view and the skill encoder's windowed view both read
  :meth:`ReferenceCommandTerm.component`.

Reset ordering: ``ManagerBasedRLEnv._reset_idx`` applies the reset-mode events
(where ``reset_reference_state`` reads ``current_expert_frame`` at the new
cursor to write the robot's reset pose) *before* ``command_manager.reset``
reaches ``_resample_command``, so cursor resampling cannot happen there. The
environment calls :meth:`resample_reference` ahead of ``super()._reset_idx``
instead, and :meth:`record_visits` pre-physics in ``step`` -- the points at
which the tracked cursor still belongs to the episode being scored.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING, Any

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils.configclass import configclass

from iltools.datasets.reset_sampling import (
    SonicAdaptiveResetSampler,
    StartFrameSampler,
)

from ...command_components import (
    COMMAND_COMPONENT_TERM_NAMES,
    is_missing,
    normalize_command_components,
)
from .._compiled import (
    body_pose_in_anchor_frame,
    quat_error_squared,
    quat_to_rot6d_flat,
)
from .reset_sampling import RandomTrajectoryAdaptiveResetSampler

if TYPE_CHECKING:
    from isaaclab_imitation.envs import ImitationRLEnv

_METRES_TO_MM = 1000.0

_REFERENCE_SCHEDULES = frozenset({"random", "sequential", "round_robin", "custom"})
_START_MODES = frozenset({"auto", "fixed", "random", "adaptive"})


@configclass
class ReferenceSelectionCfg:
    """Which motion and which start frame each environment resets onto.

    The environment never decides this. ``schedule="custom"`` with ``custom_fn``
    is how an evaluation driver, a per-goal collector, or a paper protocol pins
    the trajectory rank per environment; combine it with ``start_mode="fixed"``
    to pin the frame as well.
    """

    schedule: str = "random"
    """Trajectory selection: random | sequential | round_robin | custom."""

    custom_fn: Callable | None = None
    """``(env_ids, num_trajectories) -> ranks`` for ``schedule="custom"``."""

    start_mode: str = "auto"
    """Start-frame policy: auto | fixed | random | adaptive.

    ``auto`` resolves to ``random`` when ``random_step_max > random_step_min``
    and to ``fixed`` otherwise.
    """

    start_frame: int = 0
    """Fixed start frame (``start_mode="fixed"``)."""

    random_step_min: int = 0
    random_step_max: int = 200
    """Inclusive start-frame range for ``start_mode="random"``."""

    full_trajectory: bool = False
    """SONIC joint rank+frame sampling from the adaptive failure distribution.

    When set, the adaptive sampler picks the trajectory AND the frame, so
    ``schedule`` no longer applies. A configured explicit random-trajectory
    branch may wrap this sampler without changing its failure bookkeeping.
    """

    adaptive_weight_fn: Callable | None = None
    """``(ranks, steps) -> weights`` replacing the SONIC failure weighting."""

    adaptive_bin_size: int = 50
    adaptive_sequence_length_agnostic: bool = True
    adaptive_init_num_failures: float = 1.0
    adaptive_uniform_ratio: float = 0.1
    adaptive_pre_failure_window: int = 200
    adaptive_failure_rate_max_over_mean: float = 50.0

    random_trajectory_sampling_ratio: float = 0.0
    """Explicit random-trajectory branch probability for full-trajectory sampling."""

    random_trajectory_start_fraction: float = 0.5
    """Leading trajectory fraction available to the explicit random branch."""

    rng_seed: int | None = None
    """Dedicated reference/reset RNG seed; ``None`` inherits the environment seed."""

    def resolve(self) -> None:
        """Normalize and validate in place. Idempotent."""
        self.schedule = str(self.schedule).strip().lower().replace("-", "_")
        if self.schedule not in _REFERENCE_SCHEDULES:
            raise ValueError(
                f"Unsupported reference schedule {self.schedule!r}; expected one "
                f"of {sorted(_REFERENCE_SCHEDULES)}."
            )
        if self.schedule == "custom" and self.custom_fn is None:
            raise ValueError(
                "schedule='custom' requires custom_fn(env_ids, num_trajectories)."
            )
        if self.custom_fn is not None and not callable(self.custom_fn):
            raise ValueError("custom_fn must be callable.")
        self.start_mode = str(self.start_mode).strip().lower()
        if self.start_mode not in _START_MODES:
            raise ValueError(
                f"Unsupported start_mode {self.start_mode!r}; expected one of "
                f"{sorted(_START_MODES)}."
            )
        if int(self.start_frame) < 0:
            raise ValueError("start_frame must be >= 0.")
        if int(self.random_step_min) < 0:
            raise ValueError("random_step_min must be >= 0.")
        if int(self.random_step_max) < int(self.random_step_min):
            raise ValueError("random_step_max must be >= random_step_min.")
        if self.adaptive_weight_fn is not None:
            if not callable(self.adaptive_weight_fn):
                raise ValueError("adaptive_weight_fn must be callable.")
            if self.full_trajectory:
                raise ValueError(
                    "adaptive_weight_fn is incompatible with full_trajectory "
                    "(SONIC joint rank+frame sampling owns both)."
                )
        if int(self.adaptive_bin_size) <= 0:
            raise ValueError("adaptive_bin_size must be positive.")
        if float(self.adaptive_init_num_failures) <= 0.0:
            raise ValueError("adaptive_init_num_failures must be positive.")
        if not 0.0 <= float(self.adaptive_uniform_ratio) <= 1.0:
            raise ValueError("adaptive_uniform_ratio must be in [0, 1].")
        if int(self.adaptive_pre_failure_window) < 0:
            raise ValueError("adaptive_pre_failure_window must be >= 0.")
        if float(self.adaptive_failure_rate_max_over_mean) <= 0.0:
            raise ValueError("adaptive_failure_rate_max_over_mean must be positive.")
        if not 0.0 <= float(self.random_trajectory_sampling_ratio) <= 1.0:
            raise ValueError("random_trajectory_sampling_ratio must be in [0, 1].")
        if not 0.0 < float(self.random_trajectory_start_fraction) <= 1.0:
            raise ValueError("random_trajectory_start_fraction must be in (0, 1].")
        if float(self.random_trajectory_sampling_ratio) > 0.0 and not bool(
            self.full_trajectory
        ):
            raise ValueError(
                "random_trajectory_sampling_ratio requires full_trajectory=true."
            )

    def resolved_start_mode(self) -> str:
        """The concrete start mode, with ``auto`` decided."""
        if self.start_mode != "auto":
            return self.start_mode
        return (
            "random"
            if int(self.random_step_max) > int(self.random_step_min)
            else "fixed"
        )


class ReferenceCommandTerm(CommandTerm):
    """The reference channel: selection, reset sampling, metrics, and emission.

    ``command`` is the full-body reference frame in the flat 67-D layout
    ``[joint_pos + joint_vel (2 * num_joints), anchor_pos_b (3), anchor_ori_b
    rot6d (6)]``, refreshed lazily on access so a consumer reading
    ``get_command("reference")`` after ``env.step()`` sees the post-step
    reference frame. Individual components (including windowed views) come from
    :meth:`component`.

    Constructor-ordering note: the CommandManager builds this term inside
    ``load_managers()``, before the first reference frame exists, and
    ``CommandTerm.__init__`` only allocates buffers. Every environment accessor
    here is therefore guarded on ``current_expert_frame`` and every id
    resolution is lazy.
    """

    # pyrefly: ignore[bad-override-mutable-attribute]  # Isaac Lab term idiom
    cfg: ReferenceChannelCfg

    def __init__(self, cfg: ReferenceChannelCfg, env: ImitationRLEnv):
        super().__init__(cfg, env)
        self._joint_ids: Sequence[int] | slice | None = None
        self._command: torch.Tensor | None = None
        self._mpjpe_bodies_validated = False
        # Reset-start samplers, built eagerly: the CommandManager constructs
        # this term inside load_managers(), which the environment triggers
        # after the trajectory manager exists.
        self._start_frame_sampler: StartFrameSampler | None = None
        self._adaptive_failure_reset_sampler: SonicAdaptiveResetSampler | None = None
        self._full_trajectory_reset_sampler: (
            SonicAdaptiveResetSampler | RandomTrajectoryAdaptiveResetSampler | None
        ) = None
        self._predicted_reset_ranks: torch.Tensor | None = None
        self._predicted_reset_steps: torch.Tensor | None = None
        self._predicted_reset_probabilities: torch.Tensor | None = None
        self._build_reset_samplers()
        # Per-env metric buffers; CommandTerm.reset() averages these over the
        # resetting envs into `Metrics/reference/<name>` and zeroes them.
        # `mpjpe_l_mm` is root-relative (the SONIC/PHC "local" metric) and
        # `mpjpe_g_mm` is the world-frame counterpart, which counts the drift
        # MPJPE-L removes. GLOBAL is the one to rank on: ranking on the local
        # metric while reading a world-frame EE number compares two different
        # frames and inverts the ordering.
        #
        # These hold the RUNNING EPISODE MEAN, not the current step's value.
        # `CommandTerm.reset` logs `mean(metric[env_ids])` of whatever is in the
        # buffer at the reset step and then zeroes it, so a buffer holding the
        # instantaneous error reports the error AT THE MOMENT THE EPISODE ENDED
        # -- and since most episodes end on a tracking-error termination, that
        # is by construction a sample taken at the failure threshold. It read
        # ~55 mm during training against ~20 mm when the same checkpoint was
        # evaluated over a rollout. Accumulating instead makes the logged number
        # an episode mean, which is what evaluation reports and what everyone
        # already assumed this was.
        self.metrics["mpjpe_l_mm"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["mpjpe_g_mm"] = torch.zeros(self.num_envs, device=self.device)
        self._mpjpe_l_sum = torch.zeros(self.num_envs, device=self.device)
        self._mpjpe_g_sum = torch.zeros(self.num_envs, device=self.device)
        self._mpjpe_steps = torch.zeros(self.num_envs, device=self.device)
        self.metrics["anchor_pos_err_m"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["anchor_ori_err_rad"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self._anchor_pos_sum = torch.zeros(self.num_envs, device=self.device)
        self._anchor_ori_sum = torch.zeros(self.num_envs, device=self.device)
        self._anchor_steps = torch.zeros(self.num_envs, device=self.device)
        # The value at the final in-episode step, kept alongside the mean.
        self.metrics["anchor_pos_err_final_m"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["anchor_ori_err_final_rad"] = torch.zeros(
            self.num_envs, device=self.device
        )

    def __str__(self) -> str:
        selection = self.cfg.selection
        msg = "ReferenceCommandTerm (the tracked reference channel):\n"
        msg += f"\tCommand dimension: {self._command_dim()}\n"
        msg += f"\tAnchor body: {self.cfg.anchor_body_name}\n"
        msg += f"\tMPJPE bodies: {len(self.cfg.mpjpe_body_names)}\n"
        msg += f"\tSelection: {selection.schedule} trajectories, "
        msg += f"{selection.resolved_start_mode()} start frame"
        return msg

    """
    Properties.
    """

    @property
    def command(self) -> torch.Tensor:
        """Full-body reference command. Shape is (num_envs, 2 * num_joints + 9)."""
        self._refresh_command()
        assert self._command is not None
        return self._command

    """
    Emission.
    """

    def component(
        self,
        name: str,
        *,
        past_steps: int = 0,
        future_steps: int = 0,
        frame_stride: int = 1,
    ) -> torch.Tensor:
        """One command component, single-frame or windowed.

        ``past_steps == future_steps == 0`` reads the per-step-cached fast
        paths directly; a non-trivial window is built by the data plane's
        window path. Both express anchor-relative quantities in the robot's
        current anchor frame, so the windowed view with ``0/0`` and the
        single-frame view are the same values. ``frame_stride`` spaces the
        window slots that many reference frames apart (SONIC's tokenizer uses
        stride 5 at 50 Hz for 0.1 s spacing).
        """
        term_name = _term_name_of(name)
        if int(past_steps) == 0 and int(future_steps) == 0:
            return self._single_frame_component(name)
        return self._imitation_env().get_current_expert_window_term(
            term_name=term_name,
            past_steps=int(past_steps),
            future_steps=int(future_steps),
            frame_stride=int(frame_stride),
            joint_ids=self._resolve_joint_ids(),
            anchor_body_name=self.cfg.anchor_body_name,
            reference_body_names=self._body_names_for(name),
        )

    """
    Selection (applied by the environment's reset path).
    """

    def prepare_predicted_resets(self) -> None:
        """Snapshot the adaptive distribution and stage candidates before physics.

        This is active only for ``reference_prefetch_mode=next_and_reset``.
        Current-step failures are intentionally applied to the following
        snapshot, making the one-step adaptation lag explicit and causal.
        """
        env = self._imitation_env()
        if env.expert_data_plane.reference_prefetch_mode != "next_and_reset":
            return
        if not self.cfg.selection.full_trajectory:
            raise RuntimeError(
                "next_and_reset currently requires full-trajectory "
                "selection so one snapshotted distribution owns both rank and frame."
            )
        sampler = self._full_trajectory_reset_sampler
        if sampler is None:
            raise RuntimeError("Full-trajectory reset sampling is not initialized.")
        if self._predicted_reset_ranks is not None:
            raise RuntimeError("A predictive reset pool is already pending.")
        count = int(env.cfg.data.reference_prefetch_reset_pool_size)
        probabilities = sampler.sampling_probabilities().clone()
        ranks, steps = sampler.sample(count, probabilities=probabilities)
        self._predicted_reset_probabilities = probabilities
        self._predicted_reset_ranks = ranks
        self._predicted_reset_steps = steps
        env.expert_data_plane.begin_predicted_reset_reference(ranks, steps)

    def finish_predicted_reset_step(self) -> None:
        """Discard unused candidate metadata after the staged pool is drained."""
        self._predicted_reset_probabilities = None
        self._predicted_reset_ranks = None
        self._predicted_reset_steps = None

    def resample_reference(self, env_ids: torch.Tensor) -> int:
        """Resample the reference cursor for the given environments.

        Called by ``ImitationRLEnv._reset_idx`` before ``super()._reset_idx``,
        so the ``reset_reference_state`` event reads the reference at the freshly
        selected cursor. With ``selection.full_trajectory`` the SONIC sampler
        picks trajectory ranks and frames jointly; otherwise the trajectory
        manager's schedule picks the trajectory (including the ``custom``
        selector) and the start-frame sampler picks the local start.
        Returns the number of rows already present in the predictive GPU pool;
        callers synchronously fetch only any overflow. The ordinary exact path
        returns zero.
        """
        env = self._imitation_env()
        tm = env.trajectory_manager
        env_ids_tm = env_ids.to(device=tm.state_device, dtype=torch.long)
        if (
            env.expert_data_plane.reference_prefetch_mode == "next_and_reset"
            and self._predicted_reset_ranks is not None
            and self._predicted_reset_steps is not None
            and self._predicted_reset_probabilities is not None
        ):
            sampler = self._full_trajectory_reset_sampler
            if sampler is None:
                raise RuntimeError("Predictive reset sampler is unavailable.")
            count = int(env_ids_tm.numel())
            prefetched_count = min(count, int(self._predicted_reset_ranks.numel()))
            ranks = self._predicted_reset_ranks[:prefetched_count]
            steps = self._predicted_reset_steps[:prefetched_count]
            overflow = count - prefetched_count
            if overflow > 0:
                overflow_ranks, overflow_steps = sampler.sample(
                    overflow,
                    probabilities=self._predicted_reset_probabilities,
                )
                ranks = torch.cat((ranks, overflow_ranks))
                steps = torch.cat((steps, overflow_steps))
            tm.reset_envs(env_ids_tm, ranks=ranks, steps=steps)
            self.finish_predicted_reset_step()
            return prefetched_count
        if self.cfg.selection.full_trajectory:
            sampler = self._full_trajectory_reset_sampler
            if sampler is None:
                raise RuntimeError("Full-trajectory reset sampler is not enabled.")
            reset_ranks, reset_steps = sampler.sample(env_ids_tm.numel())
            tm.reset_envs(env_ids_tm, ranks=reset_ranks, steps=reset_steps)
            return 0
        if self._start_frame_sampler is None:
            raise RuntimeError("Reference start-frame sampler was not built.")
        ranks = tm.env_traj_rank.index_select(0, env_ids_tm)
        reset_steps = self._start_frame_sampler.sample_steps(ranks)
        tm.reset_envs(env_ids_tm, steps=reset_steps)
        return 0

    def record_visits(self) -> None:
        """Record the current cursor as a visit in the SONIC failure sampler.

        Called pre-physics in ``ImitationRLEnv.step``: the recorded
        (trajectory rank, local step) pair is the one the episode is scored
        against, so the terminal frame of a failing episode carries both a visit
        and (at reset) a failure record. No-op unless the SONIC weighting is in
        use.
        """
        sampler = self._adaptive_failure_reset_sampler
        if sampler is None:
            return
        env = self._imitation_env()
        sampler.record_visits(
            env.trajectory_manager.env_traj_rank,
            env._current_reference_local_step,
        )

    def record_failures(self, env_ids: torch.Tensor) -> None:
        """Record terminal failure bins for the resetting environments.

        Called by ``ImitationRLEnv._reset_idx`` before any reassignment -- the
        last point at which the tracked cursor still belongs to the episode that
        is ending. Failure is any non-time-out, non-``reference_finished``
        termination. No-op unless the SONIC weighting is in use.
        """
        sampler = self._adaptive_failure_reset_sampler
        if sampler is None:
            return
        env = self._imitation_env()
        tm = env.trajectory_manager
        env_ids_device = env_ids.to(device=env.device, dtype=torch.long)
        env_ids_tm = env_ids.to(device=tm.state_device, dtype=torch.long)
        failed_mask = self._reset_tracking_failure_mask().index_select(
            0, env_ids_device
        )
        if not torch.any(failed_mask):
            return
        failed_mask_tm = failed_mask.to(device=tm.state_device)
        sampler.record_failures(
            tm.env_traj_rank.index_select(0, env_ids_tm)[failed_mask_tm],
            env._current_reference_local_step.index_select(0, env_ids_device)[
                failed_mask
            ],
        )

    def set_weight_fn(self, weight_fn: Any) -> None:
        """Install a custom adaptive start-frame weight function.

        The callable takes ``(trajectory_ranks, frame_steps)`` and returns one
        non-negative weight per pair. It replaces the SONIC failure weighting
        and switches the term to adaptive start-frame sampling; trajectory
        selection still follows the configured schedule.
        """
        if not callable(weight_fn):
            raise ValueError("weight_fn must be a callable.")
        selection = self.cfg.selection
        if selection.full_trajectory:
            raise RuntimeError(
                "A custom adaptive weight function is incompatible with "
                "selection.full_trajectory (SONIC joint rank+frame sampling)."
            )
        selection.adaptive_weight_fn = weight_fn
        selection.start_mode = StartFrameSampler.ADAPTIVE
        tm = self._imitation_env().trajectory_manager
        self._start_frame_sampler = StartFrameSampler(
            tm.length,
            mode="adaptive",
            weight_fn=weight_fn,
            device=tm.state_device,
            generator=tm.reset_generator,
        )

    """
    Implementation specific functions.
    """

    def _update_command(self):
        self._refresh_command()

    def _update_metrics(self):
        env = self._imitation_env()
        if getattr(env, "current_expert_frame", None) is None:
            return
        # Root-relative MPJPE, delegated to the data plane's metric fast path
        # (metres; converted to mm at this logging boundary). It returns None
        # when the environment was built without an MPJPE body set, and the
        # metric then stays at zero because there is nothing to measure.
        mpjpe_pair = env._compute_mpjpe_metrics()
        if mpjpe_pair is None:
            self.metrics["mpjpe_l_mm"].zero_()
            self.metrics["mpjpe_g_mm"].zero_()
        else:
            self._validate_mpjpe_bodies(env)
            mpjpe_local_m, mpjpe_global_m = mpjpe_pair
            local_mm = mpjpe_local_m * _METRES_TO_MM
            global_mm = mpjpe_global_m * _METRES_TO_MM
            self._mpjpe_l_sum += local_mm
            self._mpjpe_g_sum += global_mm
            self._mpjpe_steps += 1.0
            steps = self._mpjpe_steps.clamp(min=1.0)
            # Store the running mean, so whichever step `reset` happens to
            # sample, it reads the episode mean rather than one instant. This is
            # the accumulate-then-average shape `Episode_Reward` uses, and it is
            # what makes the logged value comparable to what evaluation reports.
            self.metrics["mpjpe_l_mm"][:] = self._mpjpe_l_sum / steps
            self.metrics["mpjpe_g_mm"][:] = self._mpjpe_g_sum / steps
        robot_anchor_pos_w, robot_anchor_quat_w = env._get_robot_anchor_state_w_fast(
            self.cfg.anchor_body_name
        )
        ref_anchor_pos_w, ref_anchor_quat_w = env._get_reference_body_pose_w_fast(
            (self.cfg.anchor_body_name,)
        )
        anchor_pos_err = torch.linalg.vector_norm(
            ref_anchor_pos_w[:, 0, :] - robot_anchor_pos_w, dim=-1
        )
        anchor_ori_err = torch.sqrt(
            quat_error_squared(robot_anchor_quat_w, ref_anchor_quat_w[:, 0, :])
        )
        # Same accumulate-then-average shape as MPJPE above, and for the same
        # reason: these used to hold the instantaneous error, so `reset` logged
        # the value AT THE STEP THE EPISODE ENDED. Most episodes end on a
        # tracking-error termination, so that sample was taken at the failure
        # threshold by construction - a metric that reports roughly its own
        # termination bound no matter how well the policy tracks. The two MPJPE
        # metrics were converted to episode means; these two were left behind.
        self._anchor_pos_sum += anchor_pos_err
        self._anchor_ori_sum += anchor_ori_err
        self._anchor_steps += 1.0
        anchor_steps = self._anchor_steps.clamp(min=1.0)
        self.metrics["anchor_pos_err_m"][:] = self._anchor_pos_sum / anchor_steps
        self.metrics["anchor_ori_err_rad"][:] = self._anchor_ori_sum / anchor_steps
        # The terminal value is still worth having - for a failure it is the
        # error that tripped the threshold, and for a completed motion it is the
        # accumulated drift at the end. Kept under its own name so it can never
        # be mistaken for the episode mean.
        self.metrics["anchor_pos_err_final_m"][:] = anchor_pos_err
        self.metrics["anchor_ori_err_final_rad"][:] = anchor_ori_err

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Log the metrics, then clear this episode's MPJPE accumulators.

        ``super().reset`` reads the buffers and zeroes them; the running sums
        behind them have to be cleared too or the next episode's mean would be
        contaminated by the previous one.
        """
        extras = super().reset(env_ids)
        selected: Sequence[int] | slice = slice(None) if env_ids is None else env_ids
        self._mpjpe_l_sum[selected] = 0.0
        self._mpjpe_g_sum[selected] = 0.0
        self._mpjpe_steps[selected] = 0.0
        self._anchor_pos_sum[selected] = 0.0
        self._anchor_ori_sum[selected] = 0.0
        self._anchor_steps[selected] = 0.0
        return extras

    def _resample_command(self, env_ids: Sequence[int]):
        """No-op: Isaac Lab's reset ordering forbids cursor sampling here.

        See the module docstring; the environment calls
        :meth:`resample_reference` at the correct point instead, and
        ``cfg.resampling_time_range`` is effectively-never so the manager's
        timer cannot fire this mid-episode either.
        """

    def _set_debug_vis_impl(self, debug_vis: bool):
        """No-op: the reference channel carries no debug visualization."""

    def _build_reset_samplers(self) -> None:
        """Construct the start-frame samplers from this term's own config."""
        selection = self.cfg.selection
        tm = self._imitation_env().trajectory_manager
        if (
            self._imitation_env().expert_data_plane.reference_prefetch_mode
            == "next_and_reset"
            and not selection.full_trajectory
        ):
            raise ValueError(
                "reference_prefetch_mode=next_and_reset requires "
                "selection.full_trajectory=true (the SONIC joint rank/frame sampler)."
            )
        start_mode = selection.resolved_start_mode()
        if selection.full_trajectory or start_mode == StartFrameSampler.ADAPTIVE:
            self._adaptive_failure_reset_sampler = SonicAdaptiveResetSampler(
                tm.length,
                bin_size=int(selection.adaptive_bin_size),
                sequence_length_agnostic=bool(
                    selection.adaptive_sequence_length_agnostic
                ),
                init_num_failures=float(selection.adaptive_init_num_failures),
                uniform_sampling_rate=float(selection.adaptive_uniform_ratio),
                pre_failure_sample_window=int(selection.adaptive_pre_failure_window),
                failure_rate_max_over_mean=float(
                    selection.adaptive_failure_rate_max_over_mean
                ),
                generator=tm.reset_generator,
            )
        if selection.full_trajectory:
            # SONIC picks ranks AND frames jointly from the bin distribution;
            # the generic start sampler is unused. A repo-owned mixture may
            # wrap SONIC without changing its failure bookkeeping.
            assert self._adaptive_failure_reset_sampler is not None
            if float(selection.random_trajectory_sampling_ratio) > 0.0:
                self._full_trajectory_reset_sampler = (
                    RandomTrajectoryAdaptiveResetSampler(
                        tm.length,
                        adaptive=self._adaptive_failure_reset_sampler,
                        random_sampling_ratio=float(
                            selection.random_trajectory_sampling_ratio
                        ),
                        random_start_fraction=float(
                            selection.random_trajectory_start_fraction
                        ),
                        generator=tm.reset_generator,
                    )
                )
            else:
                self._full_trajectory_reset_sampler = (
                    self._adaptive_failure_reset_sampler
                )
            self._start_frame_sampler = None
            return
        if start_mode == StartFrameSampler.ADAPTIVE:
            weight_fn = selection.adaptive_weight_fn
            if weight_fn is None:
                weight_fn = self._adaptive_failure_reset_sampler
            if weight_fn is None:
                raise ValueError(
                    "start_mode='adaptive' requires the SONIC reset sampler or a "
                    "custom selection.adaptive_weight_fn."
                )
            self._start_frame_sampler = StartFrameSampler(
                tm.length,
                mode="adaptive",
                weight_fn=weight_fn,
                device=tm.state_device,
                generator=tm.reset_generator,
            )
            return
        self._start_frame_sampler = StartFrameSampler(
            tm.length,
            mode=start_mode,
            fixed_step=int(selection.start_frame),
            random_step_min=int(selection.random_step_min),
            random_step_max=int(selection.random_step_max),
            device=tm.state_device,
            generator=tm.reset_generator,
        )

    def _reset_tracking_failure_mask(self) -> torch.Tensor:
        env = self._imitation_env()
        failure_mask = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        for term_name in env.termination_manager.active_terms:
            term_cfg = env.termination_manager.get_term_cfg(term_name)
            if term_cfg.time_out or term_name == "reference_finished":
                continue
            failure_mask |= env.termination_manager.get_term(term_name)
        return failure_mask

    """
    Helper functions.
    """

    def _imitation_env(self) -> ImitationRLEnv:
        return self._env  # type: ignore[return-value]

    def _command_dim(self) -> int:
        if self.cfg.joint_names is not None:
            num_joints = len(self.cfg.joint_names)
        else:
            num_joints = int(self._imitation_env().scene["robot"].num_joints)
        return 2 * num_joints + 9

    def _resolve_joint_ids(self) -> Sequence[int] | slice:
        joint_ids = self._joint_ids
        if joint_ids is None:
            if self.cfg.joint_names is None:
                joint_ids = slice(None)
            else:
                joint_ids, _ = (
                    self._imitation_env()
                    .scene["robot"]
                    .find_joints(self.cfg.joint_names, preserve_order=True)
                )
            self._joint_ids = joint_ids
        return joint_ids

    def _body_names_for(self, component: str) -> tuple[str, ...]:
        """Reference bodies a body-set component is built from."""
        if component in ("ee_pos", "ee_ori"):
            return tuple(self.cfg.ee_body_names)
        if component in ("keypoint_pos", "keypoint_ori"):
            return tuple(self.cfg.keypoint_body_names)
        return ()

    def _single_frame_component(self, component: str) -> torch.Tensor:
        env = self._imitation_env()
        if component == "joint_qpos_qvel":
            return env._get_expert_motion_command_fast(self._resolve_joint_ids())
        if component == "joint_qpos":
            return env.get_expert_motion_qpos_command(self._resolve_joint_ids())
        if component in ("root_pos", "root_ori"):
            pos_b, ori_b = self._anchor_frame_pose((self.cfg.anchor_body_name,))
            if component == "root_pos":
                return pos_b[:, 0, :]
            return quat_to_rot6d_flat(ori_b[:, 0, :])
        body_names = self._body_names_for(component)
        if not body_names:
            raise ValueError(
                f"Command component {component!r} needs a configured body set on "
                "the reference channel."
            )
        pos_b, ori_b = self._anchor_frame_pose(body_names)
        if component.endswith("_pos"):
            return pos_b.reshape(env.num_envs, -1)
        return quat_to_rot6d_flat(ori_b).reshape(env.num_envs, -1)

    def _anchor_frame_pose(
        self, body_names: Sequence[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reference body poses expressed in the robot's current anchor frame."""
        env = self._imitation_env()
        robot_anchor_pos_w, robot_anchor_quat_w = env._get_robot_anchor_state_w_fast(
            self.cfg.anchor_body_name
        )
        ref_pos_w, ref_quat_w = env._get_reference_body_pose_w_fast(tuple(body_names))
        return body_pose_in_anchor_frame(
            robot_anchor_pos_w,
            robot_anchor_quat_w,
            ref_pos_w,
            ref_quat_w,
        )

    def _refresh_command(self) -> None:
        env = self._imitation_env()
        if self._command is None:
            self._command = torch.zeros(
                self.num_envs, self._command_dim(), device=self.device
            )
        if getattr(env, "current_expert_frame", None) is None:
            # Environment not fully constructed yet: keep zeros.
            return
        motion = self._single_frame_component("joint_qpos_qvel")
        self._command[:, : motion.shape[-1]] = motion
        self._command[:, -9:-6] = self._single_frame_component("root_pos")
        self._command[:, -6:] = self._single_frame_component("root_ori")

    def _validate_mpjpe_bodies(self, env: ImitationRLEnv) -> None:
        """Check the cfg body set matches the environment's own set once.

        The metric delegates to ``env._compute_mpjpe_metric()``, which measures
        over the environment's configured ``mpjpe_metric_body_names``; a
        mismatched cfg here would silently label the wrong measurement.
        """
        if self._mpjpe_bodies_validated:
            return
        env_body_names = getattr(env, "_mpjpe_metric_body_names", None)
        if env_body_names is not None and list(env_body_names) != list(
            self.cfg.mpjpe_body_names
        ):
            raise ValueError(
                "ReferenceChannelCfg.mpjpe_body_names does not match the "
                f"environment's mpjpe_metric_body_names: "
                f"{list(self.cfg.mpjpe_body_names)} vs {list(env_body_names)}."
            )
        self._mpjpe_bodies_validated = True


@configclass
class ReferenceChannelCfg(CommandTermCfg):
    """Configuration of the reference command channel."""

    class_type: type = ReferenceCommandTerm

    # Selection happens on the environment's reset path, not on the manager's
    # resampling timer (see the module docstring), so the timer never fires.
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

    anchor_body_name: str = "pelvis"
    """Body whose frame anchor-relative reference quantities are expressed in."""

    joint_names: list[str] | None = None
    """Pinned joint order of the joint-space components (None = live order)."""

    # pyrefly: ignore[bad-assignment]  # Isaac Lab required-field idiom
    mpjpe_body_names: list[str] = MISSING
    """Bodies of the root-relative MPJPE tracking metric (required).

    The mdp layer must not import robot-specific config, so the environment cfg
    supplies the tracked body set. It must match the environment's
    ``mpjpe_metric_body_names`` while the metric delegates to the data plane.
    """

    ee_body_names: list[str] = []
    """Bodies behind the ``ee_pos`` / ``ee_ori`` components."""

    keypoint_body_names: list[str] = []
    """Bodies behind the ``keypoint_pos`` / ``keypoint_ori`` components."""

    selection: ReferenceSelectionCfg = ReferenceSelectionCfg()

    critic_components: tuple[str, ...] | None = None
    """Command components the critic reads from this channel.

    ``None`` derives them: the actor's components for an explicit or chunk
    actor (the critic sees the command the actor is judged on), and the
    full-body trio for a latent actor (whose critic would otherwise reach the
    expert's joint state only through the latent).
    """

    def resolve(self) -> None:
        """Normalize and validate in place. Idempotent."""
        self.selection.resolve()
        if is_missing(self.mpjpe_body_names):
            raise ValueError(
                "ReferenceChannelCfg.mpjpe_body_names is required: the reference "
                "channel owns the MPJPE tracking metric."
            )
        if self.critic_components is not None:
            self.critic_components = normalize_command_components(
                self.critic_components
            )


def _term_name_of(component: str) -> str:
    try:
        return COMMAND_COMPONENT_TERM_NAMES[component]
    except KeyError as err:
        raise KeyError(f"Unknown command component {component!r}.") from err


__all__ = [
    "ReferenceChannelCfg",
    "ReferenceCommandTerm",
    "ReferenceSelectionCfg",
]
