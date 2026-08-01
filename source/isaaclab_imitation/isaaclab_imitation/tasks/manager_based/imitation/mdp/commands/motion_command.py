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

Motion (re)sampling stays in the env's reset path for now;
``_resample_command`` is a documented no-op that will absorb the reset-time
reference sampler in a later step of the redesign.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from dataclasses import MISSING

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils.configclass import configclass

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

    def _resample_command(self, env_ids: Sequence[int]):
        """No-op in the adapter phase.

        Motion resampling currently happens through the env's reset path
        (trajectory manager + start-frame sampler in ``_reset_idx``);
        ``cfg.resampling_time_range`` is set to effectively-never so the
        manager's timer never fights it. A later step of the redesign moves
        the reset-time reference sampler into this hook.
        """

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
