"""Held explicit-chunk command term for the manager-based CommandManager surface.

Strangler-pattern adapter (v2 redesign, step 4b): :class:`HeldChunkCommand`
wraps the env's EXISTING held-window machinery
(``ImitationRLEnv._agent_trajectory_command_terms`` behind
``set_agent_full_body_trajectory_command`` / ``capture_held_command_anchor`` /
``current_full_body_tracker_command_term`` /
``reset_agent_trajectory_command``) rather than owning new buffers. It is the
env-side twin of
:class:`~isaaclab_imitation.contracts.command_publisher.ChunkCommandPublisher`:
an external planner publishes one full-body packet (current frame plus future
frames of the ``expert_motion`` / ``expert_anchor_pos_b`` /
``expert_anchor_ori_b`` trio) per hold window, and the env's window path
phase-shifts the packet and re-expresses the anchor terms in the robot's
current anchor frame on every consumption step.

Ownership of the held buffers, the phase shift, and the anchor re-expression
moves into this term in a later step of the redesign; in this phase the term is
the CommandManager-facing view plus the published/hold bookkeeping from
:class:`PublishedCommandTerm`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.utils.configclass import configclass

from .._compiled import body_pose_in_anchor_frame, quat_to_rot6d_flat
from .published_command import PublishedCommandTerm, PublishedCommandTermCfg

if TYPE_CHECKING:
    from isaaclab_imitation.envs import ImitationRLEnv

_CHUNK_POLICY_COMMAND_MODES = (
    "explicit_chunk_current_slot",
    "full_body_chunk_current_slot",
    "ee_chunk_current_slot",
)

_FULL_BODY_PACKET_TERMS = (
    "expert_motion",
    "expert_anchor_pos_b",
    "expert_anchor_ori_b",
)


class HeldChunkCommand(PublishedCommandTerm):
    """Planner-published full-body chunk, adapted from the env's held window.

    ``command`` is the CURRENT consumed slot of the held packet: the flat
    67-D concatenation ``[joint_pos + joint_vel (2 * num_joints),
    anchor_pos_b (3), anchor_ori_b rot6d (6)]`` -- exactly the values the
    chunk-mode policy observation funcs (``mdp.policy_expert_motion_command``,
    ``mdp.policy_expert_anchor_pos_b``, ``mdp.policy_expert_anchor_ori_b``)
    deliver to the frozen vanilla tracker on this control step. All four reads
    go through the same accessor,
    ``ImitationRLEnv.current_full_body_tracker_command_term`` (the env's
    phase-shift + anchor re-expression + slot-0 selection path), so the term
    and the observation funcs cannot disagree.

    Constructor-ordering note: ``ImitationRLEnv`` parses
    ``policy_command_mode`` / ``command_hold_steps`` and allocates
    ``_agent_trajectory_command_terms`` before ``super().__init__`` runs
    ``load_managers()``, so everything checked below exists by the time this
    term is constructed and a cfg/env mismatch fails loudly here.

    Unpublished-consumption policy (adapter phase): serve the env buffer
    as-is. The env zero-fills it on reset and (under
    ``command_observation_source="planner_oracle"``) refills it itself inside
    observation computation, so failing loudly here would break the existing
    oracle-streamed training loop; the fail-loud policy arrives when this term
    owns the storage. ``metrics["command_err"]`` stays at zero until an
    external writer has actually published through :meth:`publish`.
    """

    # pyrefly: ignore[bad-override-mutable-attribute]  # Isaac Lab term idiom
    cfg: HeldChunkCommandCfg

    def __init__(self, cfg: HeldChunkCommandCfg, env: ImitationRLEnv):
        super().__init__(cfg, env)
        if not hasattr(env, "_agent_trajectory_command_terms"):
            raise RuntimeError(
                "HeldChunkCommand requires an ImitationRLEnv with the held "
                "trajectory-command buffers (`_agent_trajectory_command_terms`); "
                f"they were not found on {type(env).__name__}."
            )
        env_mode = str(getattr(env, "_policy_command_mode", "reference"))
        if env_mode not in _CHUNK_POLICY_COMMAND_MODES:
            raise RuntimeError(
                "HeldChunkCommand requires a `*_chunk_current_slot` "
                f"policy_command_mode; the env is configured with {env_mode!r}."
            )
        env_hold_steps = int(getattr(env, "_command_hold_steps", 0))
        if env_hold_steps != self.hold_steps:
            raise ValueError(
                "HeldChunkCommandCfg.hold_steps does not match the env's "
                f"command_hold_steps: {self.hold_steps} vs {env_hold_steps}. "
                "The adapter-phase term serves the env's held window, so the "
                "two must be identical."
            )
        # Lazy joint-id resolution: the articulation exists only after the env
        # is fully constructed (same idiom as MotionCommand).
        self._joint_ids: Sequence[int] | slice | None = None
        # Per-env metric buffer; CommandTerm.reset() averages it over the
        # resetting envs into `Metrics/chunk/command_err` extras and zeroes it.
        self.metrics["command_err"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "HeldChunkCommand (adapter over ImitationRLEnv held window):\n"
        msg += f"\tCommand dimension: {self._command_dim()}\n"
        msg += f"\tAnchor body: {self.cfg.anchor_body_name}\n"
        msg += f"\tHold steps: {self.hold_steps}"
        return msg

    """
    Properties.
    """

    @property
    def command(self) -> torch.Tensor:
        """Consumed slot of the held packet. Shape is (num_envs, 2*J + 9).

        Reads the trio through
        ``current_full_body_tracker_command_term`` -- the same accessor the
        chunk-mode ``mdp.policy_expert_*`` observation funcs use -- so the
        values are identical to what the tracker consumes this step.
        """
        env = self._imitation_env()
        joint_ids = self._resolve_joint_ids()
        anchor = self.cfg.anchor_body_name
        motion = env.current_full_body_tracker_command_term(
            "expert_motion", joint_ids=joint_ids
        )
        anchor_pos_b = env.current_full_body_tracker_command_term(
            "expert_anchor_pos_b", anchor_body_name=anchor
        )
        anchor_ori_b = env.current_full_body_tracker_command_term(
            "expert_anchor_ori_b", anchor_body_name=anchor
        )
        return torch.cat((motion, anchor_pos_b, anchor_ori_b), dim=-1)

    """
    Implementation specific functions.
    """

    def _apply_published_payload(
        self, env_ids: torch.Tensor, payload: Mapping[str, torch.Tensor]
    ) -> None:
        """Write the packet and pin its anchor frame, atomically.

        Mirrors the external-publisher sequence documented on
        ``ImitationRLEnv.capture_held_command_anchor`` (and used by
        ``contracts/command_publisher.ChunkCommandPublisher`` clients):
        ``set_agent_full_body_trajectory_command`` immediately followed by the
        anchor capture, so the packet is interpreted against the anchor pose
        of the instant it was published rather than a stale one. The payload's
        ``*_b`` terms must therefore be expressed in the robot's CURRENT
        anchor frame; joint-space values must already be in the env's pinned
        command joint order (invariant 1 of ``contracts/command_publisher``).
        """
        keys = set(payload.keys())
        if keys != set(_FULL_BODY_PACKET_TERMS):
            raise KeyError(
                "HeldChunkCommand.publish expects exactly the full-body packet "
                f"terms {sorted(_FULL_BODY_PACKET_TERMS)}, got {sorted(keys)}."
            )
        env = self._imitation_env()
        env.set_agent_full_body_trajectory_command(
            expert_motion=payload["expert_motion"],
            expert_anchor_pos_b=payload["expert_anchor_pos_b"],
            expert_anchor_ori_b=payload["expert_anchor_ori_b"],
            env_ids=env_ids,
        )
        env.capture_held_command_anchor(self.cfg.anchor_body_name, env_ids=env_ids)

    def _update_command(self):
        """No-op: the packet lives in the env buffers between publications."""

    def _update_metrics(self):
        """Consumed-slot vs live-oracle error, only where a packet was published.

        ``command_err`` is the mean absolute difference between the slot the
        tracker consumes this step (:attr:`command`) and the live oracle
        reference command in the MotionCommand 67-D layout, rebuilt from the
        same per-step-cached env fast paths MotionCommand uses. While nothing
        has been published through :meth:`publish` (default latent runs,
        ``planner_oracle`` self-filled windows) the metric stays at zero.
        """
        env = self._imitation_env()
        if getattr(env, "current_expert_frame", None) is None:
            return
        if not bool(self._published.any()):
            self.metrics["command_err"].zero_()
            return
        err = (self.command - self._live_reference_command()).abs().mean(dim=-1)
        self.metrics["command_err"][:] = torch.where(
            self._published, err, torch.zeros_like(err)
        )

    def _resample_command(self, env_ids: Sequence[int]):
        """Documented no-op for the buffers; only clears the published mask.

        ``ImitationRLEnv._reset_idx`` already calls
        ``reset_agent_trajectory_command(env_ids)`` for the resetting envs
        *before* ``super()._reset_idx`` triggers the CommandManager reset that
        lands here (same ordering as the agent-latent buffer), so delegating
        another buffer reset would be a double reset. The base class clears
        the published mask (fail-loud policy is deferred; see the class
        docstring).
        """
        super()._resample_command(env_ids)

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

    def _live_reference_command(self) -> torch.Tensor:
        """Live oracle command in the MotionCommand 67-D layout.

        Same fast-path calls as ``MotionCommand._refresh_command`` (cached per
        env step, so this adds no new gathers when the ``motion`` term already
        ran this step).
        """
        env = self._imitation_env()
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
        return torch.cat(
            (
                motion,
                anchor_pos_b[:, 0, :],
                quat_to_rot6d_flat(anchor_ori_b[:, 0, :]),
            ),
            dim=-1,
        )


@configclass
class HeldChunkCommandCfg(PublishedCommandTermCfg):
    """Configuration for the held explicit-chunk command term."""

    class_type: type = HeldChunkCommand

    anchor_body_name: str = "torso_link"
    """Body whose frame the packet's ``*_b`` terms are expressed in.

    Must match the anchor the chunk-mode observation funcs consume with (the
    env cfg's ``expert_anchor_body_name`` after ``_set_anchor_body``); the v2
    env cfg wires it in lockstep, like the ``motion`` term's anchor.
    """

    joint_names: list[str] | None = None
    """Joint names (in command order) for the joint pos/vel packet half.

    ``None`` uses every joint in live articulation order. Supply the pinned
    joint-name list (resolved with ``preserve_order=True``) to match the
    ordering contract of the chunk-mode ``policy_expert_motion_command``
    observation term (invariant 1 of ``contracts/command_publisher``).
    """
