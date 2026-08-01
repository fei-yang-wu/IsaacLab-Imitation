"""Motion-tracking command term for the manager-based CommandManager surface.

Strangler-pattern adapter (v2 redesign, first increment): :class:`MotionCommand`
adapts over the existing :class:`~isaaclab_imitation.envs.ImitationRLEnv`
reference machinery instead of owning it. Its two jobs in this phase are:

1. Expose the motion-tracking command tensor through
   ``env.command_manager.get_command("motion")`` with the exact values the
   v1 observation terms (``expert_motion_command`` / ``expert_anchor_pos_b`` /
   ``expert_anchor_ori_b``) produce.
2. Own the tracking metrics via ``_update_metrics()`` so the CommandManager
   logs ``Metrics/motion/...`` natively at episode reset (the
   beyondmimic/SONIC idiom).

Reset-start sampling (v2, step 3c, fully absorbed): with ``cfg.owns_reset=True``
this term owns the reference reset-start samplers
(``iltools.datasets.reset_sampling.StartFrameSampler`` /
``SonicAdaptiveResetSampler``) AND the adaptive-failure bookkeeping that
feeds them: :meth:`MotionCommand.record_visits` /
:meth:`MotionCommand.record_failures` are called by
``ImitationRLEnvV2.step`` / ``_reset_idx`` at exactly the points the legacy
env-inline hooks ran (visit recording before the physics step, failure-bin
recording before trajectory reassignment), and :meth:`MotionCommand.resample_reference`
applies the samplers where the env-inline sampling used to run -- all before
``super()._reset_idx``, so the ``reset_reference_state`` reset event reads
the reference at the freshly sampled cursor. Exactly one sampler instance
set exists (term-owned; no env-side mirrors).
``_resample_command`` remains a documented no-op because Isaac Lab's reset
chain runs ``event_manager.apply(mode="reset")`` -- where the
``reset_reference_state`` event reads ``current_expert_frame`` at the *new*
cursor to write the robot's reset pose -- before ``command_manager.reset``
ever reaches the term, so cursor resampling inside ``_resample_command``
would arrive too late.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch

from dataclasses import MISSING

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils.configclass import configclass

from iltools.datasets.reset_sampling import (
    SonicAdaptiveResetSampler,
    StartFrameSampler,
)

from .._compiled import (
    body_pose_in_anchor_frame,
    quat_error_squared,
    quat_to_rot6d_flat,
)

if TYPE_CHECKING:
    from isaaclab_imitation.envs import ImitationRLEnv

_METRES_TO_MM = 1000.0


class MotionCommand(CommandTerm):
    """Reference-motion tracking command, adapted from the env's live buffers.

    Command layout (per env): ``[joint_pos + joint_vel (2 * num_joints),
    anchor_pos_b (3), anchor_ori_b rot6d (6)]`` -- the same term-major values
    the v1 explicit command observation terms deliver, refreshed from the
    env's fast-path accessors (``_get_expert_motion_command_fast``,
    ``_get_robot_anchor_state_w_fast``, ``_get_reference_body_pose_w_fast``).
    Those accessors cache per env step and are invalidated whenever the env
    refreshes ``current_expert_frame``, so reading them here never poisons the
    values the observation/reward managers later consume.

    The ``command`` property refreshes lazily on access so a consumer reading
    ``get_command("motion")`` after ``env.step()`` sees the post-step reference
    frame -- identical values to the v1 observation terms -- rather than the
    pre-refresh frame the manager's ``compute()`` ran against.

    Constructor-ordering note: the CommandManager builds this term inside
    ``load_managers()``, before the imitation dataset machinery has produced a
    reference frame, and ``CommandTerm.__init__`` only allocates buffers and
    registers debug-vis. Every env accessor here is therefore guarded on
    ``current_expert_frame`` being available and the joint-id/buffer
    resolution is lazy (first touch happens on the first ``compute()`` /
    ``command`` access, after the env is fully constructed).
    """

    # pyrefly: ignore[bad-override-mutable-attribute]  # Isaac Lab term idiom
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ImitationRLEnv):
        super().__init__(cfg, env)
        # Lazy state: resolved on first use, after the env is fully built.
        self._joint_ids: Sequence[int] | slice | None = None
        self._command: torch.Tensor | None = None
        self._mpjpe_bodies_validated = False
        # Reset-start samplers (owned only when cfg.owns_reset). Constructed
        # eagerly: the CommandManager builds this term inside load_managers(),
        # which `ImitationRLEnvV2.__init__` triggers *after* it has parsed the
        # reset cfg fields and created the trajectory manager, so everything
        # needed is available and exactly one sampler instance set ever
        # exists (term-owned; the env's record/sample hooks delegate here).
        self._start_frame_sampler: StartFrameSampler | None = None
        self._adaptive_failure_reset_sampler: SonicAdaptiveResetSampler | None = None
        if cfg.owns_reset:
            self._build_reset_samplers()
        # Per-env metric buffers; CommandTerm.reset() averages these over the
        # resetting envs into `Metrics/motion/<name>` extras and zeroes them.
        self.metrics["mpjpe_mm"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["anchor_pos_err_m"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["anchor_ori_err_rad"] = torch.zeros(
            self.num_envs, device=self.device
        )

    def __str__(self) -> str:
        msg = "MotionCommand (adapter over ImitationRLEnv reference machinery):\n"
        msg += f"\tCommand dimension: {self._command_dim()}\n"
        msg += f"\tAnchor body: {self.cfg.anchor_body_name}\n"
        msg += f"\tMPJPE bodies: {len(self.cfg.mpjpe_body_names)}"
        return msg

    """
    Properties.
    """

    @property
    def command(self) -> torch.Tensor:
        """Motion command tensor. Shape is (num_envs, 2 * num_joints + 9)."""
        self._refresh_command()
        assert self._command is not None
        return self._command

    """
    Implementation specific functions.
    """

    def _update_command(self):
        self._refresh_command()

    def _update_metrics(self):
        env = self._imitation_env()
        if getattr(env, "current_expert_frame", None) is None:
            return
        # Root-relative MPJPE, delegated to the env's existing metric fast
        # path (metres; converted to mm at this logging boundary, matching
        # `Metrics/mpjpe_mm`). It returns None when the env was built without
        # an MPJPE body set (`mpjpe_metric_body_names` empty); the metric then
        # stays at zero because there is nothing to measure it against.
        mpjpe_m = env._compute_mpjpe_metric()
        if mpjpe_m is None:
            self.metrics["mpjpe_mm"].zero_()
        else:
            self._validate_mpjpe_bodies(env)
            self.metrics["mpjpe_mm"][:] = mpjpe_m * _METRES_TO_MM
        # Global anchor position/orientation error, reusing the same fast-path
        # accessors as the anchor reward/termination terms.
        robot_anchor_pos_w, robot_anchor_quat_w = env._get_robot_anchor_state_w_fast(
            self.cfg.anchor_body_name
        )
        ref_anchor_pos_w, ref_anchor_quat_w = env._get_reference_body_pose_w_fast(
            (self.cfg.anchor_body_name,)
        )
        self.metrics["anchor_pos_err_m"][:] = torch.linalg.vector_norm(
            ref_anchor_pos_w[:, 0, :] - robot_anchor_pos_w, dim=-1
        )
        self.metrics["anchor_ori_err_rad"][:] = torch.sqrt(
            quat_error_squared(robot_anchor_quat_w, ref_anchor_quat_w[:, 0, :])
        )

    def resample_reference(self, env_ids: torch.Tensor) -> None:
        """Resample the reference reset-start cursor for the given envs.

        Only valid when ``cfg.owns_reset``. Called by
        ``ImitationRLEnvV2._reset_idx`` at the exact point where the env-inline
        sampling used to run -- *before* ``super()._reset_idx`` so the
        ``reset_reference_state`` reset event (applied by
        ``event_manager.apply(mode="reset")`` inside Isaac Lab's reset chain)
        reads the reference at the freshly sampled cursor. The caller records
        terminal failure bins (:meth:`record_failures`) immediately before
        this, at the same point the legacy env-inline path did.

        Semantics are identical to the env-inline path: with
        ``random_reset_full_trajectory`` the SONIC sampler picks trajectory
        ranks and frames jointly; otherwise the trajectory manager's
        ``reset_schedule`` keeps trajectory selection and the
        ``StartFrameSampler`` picks the local start (fixed/random/adaptive).
        """
        env = self._imitation_env()
        tm = env.trajectory_manager
        env_ids_tm = env_ids.to(device=tm._state_device, dtype=torch.long)
        if env._random_reset_full_trajectory:
            if self._adaptive_failure_reset_sampler is None:
                raise RuntimeError("Adaptive failure reset sampler is not enabled.")
            reset_ranks, reset_steps = self._adaptive_failure_reset_sampler.sample(
                env_ids_tm.numel()
            )
            tm.reset_envs(env_ids_tm, ranks=reset_ranks, steps=reset_steps)
            return
        if self._start_frame_sampler is None:
            raise RuntimeError(
                "MotionCommand.resample_reference requires cfg.owns_reset=True "
                "(no start-frame sampler was built)."
            )
        ranks = tm.env_traj_rank.index_select(0, env_ids_tm)
        reset_steps = self._start_frame_sampler.sample_steps(ranks)
        tm.reset_envs(env_ids_tm, steps=reset_steps)

    def record_visits(self) -> None:
        """Record the current cursor as a visit in the SONIC failure sampler.

        Called by ``ImitationRLEnvV2.step`` right after the pre-step reference
        advance, exactly where the legacy env-inline hook ran: the recorded
        (trajectory rank, local step) pair is the one the episode is being
        scored against, so the terminal frame of a failing episode carries
        both a visit and (at reset) a failure record. No-op unless the SONIC
        weight function is in use.
        """
        sampler = self._adaptive_failure_reset_sampler
        if sampler is None:
            return
        env = self._imitation_env()
        tm = env.trajectory_manager
        sampler.record_visits(
            tm.env_traj_rank,
            env._current_reference_local_step,
        )

    def record_failures(self, env_ids: torch.Tensor) -> None:
        """Record terminal failure bins for the resetting envs.

        Called by ``ImitationRLEnvV2._reset_idx`` before any trajectory
        reassignment or reset write -- the last point at which the tracked
        cursor still belongs to the episode that is ending. Failure is any
        non-time-out, non-``reference_finished`` termination. No-op unless
        the SONIC weight function is in use.
        """
        sampler = self._adaptive_failure_reset_sampler
        if sampler is None:
            return
        env = self._imitation_env()
        tm = env.trajectory_manager
        env_ids_device = env_ids.to(device=env.device, dtype=torch.long)
        env_ids_tm = env_ids.to(device=tm._state_device, dtype=torch.long)
        failed_mask = self._reset_tracking_failure_mask().index_select(
            0, env_ids_device
        )
        if not torch.any(failed_mask):
            return
        failed_mask_tm = failed_mask.to(device=tm._state_device)
        sampler.record_failures(
            tm.env_traj_rank.index_select(0, env_ids_tm)[failed_mask_tm],
            env._current_reference_local_step.index_select(0, env_ids_device)[
                failed_mask
            ],
        )

    def set_weight_fn(self, weight_fn: Any) -> None:
        """Provide a custom adaptive starting-frame weight function.

        The callable must accept ``(trajectory_ranks, frame_steps)`` tensors
        and return one non-negative weight per (rank, step) pair, as expected
        by ``iltools.datasets.reset_sampling.StartFrameSampler``. It replaces
        the SONIC failure-weight function and switches the term to
        ``reset_start_mode='adaptive'`` (trajectory ranks still come from the
        manager's reset schedule).
        """
        env = self._imitation_env()
        if env._random_reset_full_trajectory:
            raise RuntimeError(
                "A custom adaptive weight function is incompatible with "
                "random_reset_full_trajectory (SONIC joint rank+frame sampling)."
            )
        if not callable(weight_fn):
            raise ValueError("weight_fn must be a callable.")
        env._adaptive_reset_weight_fn = weight_fn
        env._reset_start_mode = StartFrameSampler.ADAPTIVE
        self._start_frame_sampler = StartFrameSampler(
            env.trajectory_manager._length,
            mode="adaptive",
            weight_fn=weight_fn,
            device=env.trajectory_manager._state_device,
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

    def _resample_command(self, env_ids: Sequence[int]):
        """No-op: Isaac Lab's reset ordering forbids cursor sampling here.

        ``ManagerBasedRLEnv._reset_idx`` applies the reset-mode events
        (``reset_reference_state`` reads ``current_expert_frame`` at the new
        cursor to write the robot's reset pose) *before* it reaches
        ``command_manager.reset`` -> ``_resample_command``, so a cursor
        resampled here would arrive after the robot was already teleported to
        the stale pre-reset reference. The term therefore exposes
        :meth:`resample_reference` instead, which ``ImitationRLEnvV2._reset_idx``
        calls ahead of ``super()._reset_idx`` (v2, ``cfg.owns_reset``);
        without ownership the legacy env's inline path runs unchanged (v0/v1).
        ``cfg.resampling_time_range`` is effectively-never so the manager's
        timer never fires this hook mid-episode either.
        """

    def _build_reset_samplers(self) -> None:
        """Construct the reset-start samplers this term owns (v2).

        Mirrors ``ImitationRLEnv._setup_adaptive_failure_reset_sampler``
        exactly, reading the env's already-parsed reset cfg fields
        (``reset_start_mode``, ``random_reset_step_min/max``,
        ``random_reset_full_trajectory``, ``reference_start_frame`` and the
        ``adaptive_failure_reset_*`` knobs -- one parse, in the env). The
        built instances are the ONLY ones that exist: the bookkeeping hooks
        (:meth:`record_visits` / :meth:`record_failures`) and
        :meth:`resample_reference` are term methods, so nothing is mirrored
        onto the env.
        """
        env = self._imitation_env()
        tm = env.trajectory_manager
        if env._random_reset_full_trajectory or env._reset_start_mode == "adaptive":
            self._adaptive_failure_reset_sampler = SonicAdaptiveResetSampler(
                tm._length,
                bin_size=env._adaptive_failure_reset_bin_size,
                sequence_length_agnostic=(
                    env._adaptive_failure_reset_sequence_length_agnostic
                ),
                init_num_failures=env._adaptive_failure_reset_init_num_failures,
                uniform_sampling_rate=env._adaptive_failure_reset_uniform_ratio,
                pre_failure_sample_window=(
                    env._adaptive_failure_reset_pre_failure_window
                ),
                failure_rate_max_over_mean=(
                    env._adaptive_failure_reset_failure_rate_max_over_mean
                ),
            )
        if env._random_reset_full_trajectory:
            # Legacy full-trajectory path: SONIC picks ranks AND frames jointly
            # from the bin distribution; the generic start sampler is unused.
            self._start_frame_sampler = None
        else:
            mode = env._reset_start_mode
            if mode == "auto":
                mode = (
                    StartFrameSampler.RANDOM
                    if env._random_reset_step_max > env._random_reset_step_min
                    else StartFrameSampler.FIXED
                )
            if mode == StartFrameSampler.ADAPTIVE:
                weight_fn = env._adaptive_reset_weight_fn
                if weight_fn is None:
                    weight_fn = self._adaptive_failure_reset_sampler
                if weight_fn is None:
                    raise ValueError(
                        "reset_start_mode='adaptive' requires the SONIC reset "
                        "sampler or a custom `cfg.adaptive_reset_weight_fn`."
                    )
                self._start_frame_sampler = StartFrameSampler(
                    tm._length,
                    mode="adaptive",
                    weight_fn=weight_fn,
                    device=tm._state_device,
                )
            else:
                self._start_frame_sampler = StartFrameSampler(
                    tm._length,
                    mode=mode,
                    fixed_step=env._reference_start_frame,
                    random_step_min=env._random_reset_step_min,
                    random_step_max=env._random_reset_step_max,
                    device=tm._state_device,
                )
        # Single-instance guarantee (the aliasing invariant): exactly one
        # sampler instance set exists. The legacy env's inline record hooks
        # (``_record_adaptive_failure_reset_visits`` / ``_bins``, still in
        # ``imitation_rl_env_legacy.py`` for v0/v1 AND for legacy-env runs of
        # the v2 cfg, e.g. the equivalence certificate) read these two env
        # attributes; the v2 env's term-owned hooks
        # (:meth:`record_visits` / :meth:`record_failures`) read the term's
        # own attributes. Both sides must feed the same objects.
        env._adaptive_failure_reset_sampler = self._adaptive_failure_reset_sampler
        env._start_frame_sampler = self._start_frame_sampler

    def _set_debug_vis_impl(self, debug_vis: bool):
        """No-op: no debug visualization in the adapter phase."""

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

    def _refresh_command(self) -> None:
        env = self._imitation_env()
        if self._command is None:
            self._command = torch.zeros(
                self.num_envs, self._command_dim(), device=self.device
            )
        if getattr(env, "current_expert_frame", None) is None:
            # Env not fully constructed / no reference sampled yet: keep zeros.
            return
        motion = env._get_expert_motion_command_fast(self._resolve_joint_ids())
        robot_anchor_pos_w, robot_anchor_quat_w = env._get_robot_anchor_state_w_fast(
            self.cfg.anchor_body_name
        )
        ref_anchor_pos_w, ref_anchor_quat_w = env._get_reference_body_pose_w_fast(
            (self.cfg.anchor_body_name,)
        )
        anchor_pos_b, anchor_ori_b = body_pose_in_anchor_frame(
            robot_anchor_pos_w,
            robot_anchor_quat_w,
            ref_anchor_pos_w,
            ref_anchor_quat_w,
        )
        self._command[:, : motion.shape[-1]] = motion
        self._command[:, -9:-6] = anchor_pos_b[:, 0, :]
        self._command[:, -6:] = quat_to_rot6d_flat(anchor_ori_b[:, 0, :])

    def _validate_mpjpe_bodies(self, env: ImitationRLEnv) -> None:
        """Check the cfg body set matches the env's own MPJPE body set once.

        In the adapter phase the metric is delegated to
        ``env._compute_mpjpe_metric()``, which measures over the env's
        configured ``mpjpe_metric_body_names``; a mismatched cfg here would
        silently label the wrong measurement.
        """
        if self._mpjpe_bodies_validated:
            return
        env_body_names = getattr(env, "_mpjpe_metric_body_names", None)
        if env_body_names is not None and list(env_body_names) != list(
            self.cfg.mpjpe_body_names
        ):
            raise ValueError(
                "MotionCommandCfg.mpjpe_body_names does not match the env's "
                f"mpjpe_metric_body_names: {list(self.cfg.mpjpe_body_names)} vs "
                f"{list(env_body_names)}. The adapter-phase metric delegates to "
                "the env, so the two sets must be identical."
            )
        self._mpjpe_bodies_validated = True


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion-tracking command term."""

    class_type: type = MotionCommand

    # Motion resampling happens through the env's reset path in the adapter
    # phase (see MotionCommand._resample_command), so the manager-side timer
    # is set to effectively never fire.
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

    anchor_body_name: str = "pelvis"
    """Body used to express the reference anchor pose in the robot frame."""

    # pyrefly: ignore[bad-assignment]  # Isaac Lab required-field idiom
    mpjpe_body_names: list[str] = MISSING
    """Bodies of the MPJPE tracking metric.

    Required (no default): the mdp layer must not import robot-specific
    config, so the env cfg supplies the tracked body set (for G1 that is
    ``config.g1.common.constants.G1_TRACKED_BODY_NAMES``). Must match the env's
    ``mpjpe_metric_body_names`` while the metric is delegated to the env.
    """

    joint_names: list[str] | None = None
    """Joint names (in command order) for the joint pos/vel command half.

    ``None`` uses every joint in live articulation order. Supply the pinned
    joint-name list (resolved with ``preserve_order=True``) to match the
    ordering contract of the v1 ``expert_motion_command`` observation term.
    """

    owns_reset: bool = False
    """Whether this term owns reference reset-start sampling (v2 step 3c).

    ``True``: the term constructs the ``StartFrameSampler`` /
    ``SonicAdaptiveResetSampler`` pair (from the env cfg's reset fields) and
    ``ImitationRLEnv._reset_idx`` calls :meth:`MotionCommand.resample_reference`
    instead of running its inline sampling; the env builds no sampler of its
    own. ``False`` (default): the env-inline path runs unchanged (v0/v1).
    """
