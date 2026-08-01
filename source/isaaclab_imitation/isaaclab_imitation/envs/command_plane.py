"""Command-plane components for the v2 env fork (``ImitationRLEnvV2``).

Owned components of the published-command surface that the legacy env
(``envs/imitation_rl_env.py``) still implements inline for v0/v1:

- :class:`LatentCommandBuffer`: the agent-published latent skill command
  buffer (z + phase), served to the ``skill`` command term, the RLOpt
  wrapper, and the ``latent_command`` observation terms.
- :class:`HeldCommandPlane`: the held explicit-chunk buffers plus the
  publish surface that :class:`SkillCommand` /
  :class:`HeldChunkCommand`, ``envs/rlopt.py``, and the
  ``imitation_experiments`` planners call. Holds the chunk window buffers,
  the renewal-anchor bookkeeping, and the phase-shift / anchor re-expression
  middleware; the diagnostic command trace that the legacy env carried under
  ``ISAACLAB_COMMAND_TRACE`` is deliberately not part of the v2 surface.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import isaaclab.utils.math as math_utils
import torch

if TYPE_CHECKING:
    from isaaclab_imitation.envs.imitation_rl_env_v2 import ImitationRLEnvV2


class LatentCommandBuffer:
    """Agent-published latent skill command (z + phase) buffer.

    Exactly the storage behind ``ImitationRLEnv._agent_latent_command``:
    ``num_envs x latent_dim`` float32, zero-filled on reset. The env exposes
    the legacy names (``get_agent_latent_command`` / ``set_agent_latent_command`` /
    ``reset_agent_latent_command``) as thin delegators so the RLOpt wrapper,
    the ``skill`` command term, and the observation funcs keep working
    unchanged.
    """

    def __init__(self, num_envs: int, latent_dim: int, device: torch.device) -> None:
        latent_dim = int(latent_dim)
        if latent_dim <= 0:
            raise ValueError("latent_command_dim must be > 0.")
        self._dim = latent_dim
        self._command = torch.zeros(
            (int(num_envs), self._dim), device=device, dtype=torch.float32
        )

    @property
    def dim(self) -> int:
        """Width of the latent command (z + phase)."""
        return self._dim

    def get(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Return the current agent-published latent command buffer."""
        if env_ids is None:
            return self._command
        env_ids = env_ids.to(device=self._command.device, dtype=torch.long)
        return self._command.index_select(0, env_ids)

    def set(
        self,
        latent_command: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """Publish the latest agent latent command into the observation state."""
        latent_command = latent_command.to(
            device=self._command.device, dtype=torch.float32
        )
        if env_ids is None:
            if latent_command.ndim != 2 or latent_command.shape != self._command.shape:
                raise ValueError(
                    "Latent command shape mismatch. "
                    f"Expected {tuple(self._command.shape)}, got "
                    f"{tuple(latent_command.shape)}."
                )
            self._command.copy_(latent_command)
            return

        env_ids = env_ids.to(device=self._command.device, dtype=torch.long)
        if latent_command.ndim != 2 or latent_command.shape != (
            env_ids.shape[0],
            self._dim,
        ):
            raise ValueError(
                "Latent command shape mismatch for indexed update. "
                f"Expected {(env_ids.shape[0], self._dim)}, got "
                f"{tuple(latent_command.shape)}."
            )
        self._command.index_copy_(0, env_ids, latent_command)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset latent commands for the selected environments to zeros."""
        if env_ids is None:
            self._command.zero_()
            return
        env_ids = env_ids.to(device=self._command.device, dtype=torch.long)
        self._command.index_fill_(0, env_ids, 0.0)


class HeldCommandPlane:
    """Held explicit-command chunk buffers and the planner publish surface.

    Mirrors the legacy env's held-window machinery
    (``ImitationRLEnv._agent_trajectory_command_terms`` and friends) as an
    owned component. An external planner publishes one full-body packet per
    hold window via :meth:`set_full_body_command` / :meth:`set_command`,
    optionally pinning the publish-time anchor frame with
    :meth:`capture_held_command_anchor`; the env's command-window path
    phase-shifts the packet and re-expresses the anchor terms in the robot's
    current anchor frame on every consumption step.

    This plane is the env-side twin of
    :class:`~isaaclab_imitation.contracts.command_publisher.ChunkCommandPublisher`.
    """

    def __init__(
        self,
        env: ImitationRLEnvV2,
        *,
        num_envs: int,
        device: torch.device,
        window_steps: int,
        num_joints: int,
        num_ee_bodies: int,
        num_keypoint_bodies: int = 0,
    ) -> None:
        self._env = env
        self._window_steps = int(window_steps)
        self._terms = self._allocate_terms(
            window_steps=self._window_steps,
            num_joints=int(num_joints),
            num_ee_bodies=int(num_ee_bodies),
            num_keypoint_bodies=int(num_keypoint_bodies),
            device=device,
        )
        # Anchor pose at each env's last command renewal, per anchor body:
        # published chunks are expressed in the publish-time anchor frame and
        # re-expressed into the current frame each step (odometry middleware).
        self._held_command_anchor_pose: dict[
            str, tuple[torch.Tensor, torch.Tensor]
        ] = {}
        # Set once an external publisher calls capture_held_command_anchor();
        # from then on it owns the held reference pose and the automatic
        # phase-0 recapture is suppressed. See capture_held_command_anchor().
        self._external_command_anchor_owner = False
        # Optional in-step planner publication hook (set_planner_command_provider).
        self._planner_command_provider: Any = None
        self._planner_command_provider_token: int | None = None

    @property
    def terms(self) -> dict[str, torch.Tensor]:
        """Flat per-term command window buffers (legacy ``_agent_trajectory_command_terms``)."""
        return self._terms

    @property
    def window_steps(self) -> int:
        """Frames per held command window (past + current + future)."""
        return self._window_steps

    @staticmethod
    def command_window_steps_from_offsets(past_steps: int, future_steps: int) -> int:
        past_steps = int(past_steps)
        future_steps = int(future_steps)
        if past_steps < 0 or future_steps < 0:
            raise ValueError("Command window steps must be >= 0.")
        return past_steps + future_steps + 1

    @staticmethod
    def _allocate_terms(
        *,
        window_steps: int,
        num_joints: int,
        num_ee_bodies: int,
        num_keypoint_bodies: int = 0,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        def zeros(width: int) -> torch.Tensor:
            return torch.zeros((0, width), device=device, dtype=torch.float32)

        window_steps = int(window_steps)
        return {
            "expert_motion": zeros(window_steps * 2 * int(num_joints)),
            # root_qpos: the position half only, so the packet carries no joint
            # velocities at all rather than zero-filling them.
            "expert_motion_qpos": zeros(window_steps * int(num_joints)),
            "expert_anchor_pos_b": zeros(window_steps * 3),
            "expert_anchor_ori_b": zeros(window_steps * 6),
            "expert_ee_pos_b": zeros(window_steps * int(num_ee_bodies) * 3),
            "expert_ee_ori_b": zeros(window_steps * int(num_ee_bodies) * 6),
            # Keypoint positions and orientations share a body set of their
            # own so their packet slots never collide with the EE interface's.
            "expert_keypoint_pos_b": zeros(window_steps * int(num_keypoint_bodies) * 3),
            "expert_keypoint_ori_b": zeros(window_steps * int(num_keypoint_bodies) * 6),
        }

    def _ensure_buffers(self) -> None:
        env = self._env
        num_envs = int(env.num_envs)
        device = torch.device(env.device)
        for term_name, term in tuple(self._terms.items()):
            if term.shape[0] == num_envs and term.device == device:
                continue
            self._terms[term_name] = torch.zeros(
                (num_envs, int(term.shape[1])),
                device=device,
                dtype=torch.float32,
            )

    def get_term(
        self, term_name: str, env_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        self._ensure_buffers()
        try:
            value = self._terms[str(term_name)]
        except KeyError as err:
            raise KeyError(f"Unknown trajectory command term: {term_name!r}.") from err
        if env_ids is None:
            return value
        env_ids = env_ids.to(device=self._env.device, dtype=torch.long)
        return value.index_select(0, env_ids)

    def set_command(
        self,
        command_terms: Mapping[str, torch.Tensor],
        env_ids: torch.Tensor | None = None,
    ) -> None:
        self._ensure_buffers()
        if env_ids is not None:
            env_ids = env_ids.to(device=self._env.device, dtype=torch.long)
        for term_name, command in command_terms.items():
            key = str(term_name)
            if key not in self._terms:
                raise KeyError(f"Unknown trajectory command term: {key!r}.")
            target = self._terms[key]
            command = command.to(device=self._env.device, dtype=torch.float32)
            if env_ids is None:
                if command.ndim != 2 or command.shape != target.shape:
                    raise ValueError(
                        f"Trajectory command term {key!r} shape mismatch. "
                        f"Expected {tuple(target.shape)}, got "
                        f"{tuple(command.shape)}."
                    )
                target.copy_(command)
                continue
            expected_shape = (int(env_ids.shape[0]), int(target.shape[1]))
            if command.ndim != 2 or tuple(command.shape) != expected_shape:
                raise ValueError(
                    f"Trajectory command term {key!r} indexed shape mismatch. "
                    f"Expected {expected_shape}, got {tuple(command.shape)}."
                )
            target.index_copy_(0, env_ids, command)

    def set_full_body_command(
        self,
        *,
        expert_motion: torch.Tensor,
        expert_anchor_pos_b: torch.Tensor,
        expert_anchor_ori_b: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        self.set_command(
            {
                "expert_motion": expert_motion,
                "expert_anchor_pos_b": expert_anchor_pos_b,
                "expert_anchor_ori_b": expert_anchor_ori_b,
            },
            env_ids=env_ids,
        )

    def set_ee_command(
        self,
        *,
        expert_ee_pos_b: torch.Tensor,
        expert_ee_ori_b: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        self.set_command(
            {
                "expert_ee_pos_b": expert_ee_pos_b,
                "expert_ee_ori_b": expert_ee_ori_b,
            },
            env_ids=env_ids,
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        self._ensure_buffers()
        if env_ids is None:
            for command in self._terms.values():
                command.zero_()
            return
        env_ids = env_ids.to(device=self._env.device, dtype=torch.long)
        for command in self._terms.values():
            command.index_fill_(0, env_ids, 0.0)

    def set_planner_command_provider(self, provider: Any) -> None:
        """Register a callback that produces planner command packets in-step.

        ``planner_oracle`` fills the command buffer from the expert *inside*
        the observation pass, so the packet is expressed in the anchor frame
        of the very step that consumes it and its re-expression is exactly
        the identity at publication. An external publisher writing between
        steps cannot match that: it fetches body-frame quantities one physics
        step early, which silently biases the root command.

        Registering a provider gives a planner the same in-step contract. The
        callback receives the environment ids being renewed and returns a
        mapping of command term name to tensor for those environments.
        """
        self._planner_command_provider = provider
        self._planner_command_provider_token = None

    def maybe_fill_from_planner_provider(self, phase: torch.Tensor) -> None:
        """Publish the registered planner packet for envs at hold phase zero."""
        provider = self._planner_command_provider
        if provider is None:
            return
        # get_current_command_window_term runs once per command term; the
        # planner must be evaluated once per control step, not once per term.
        token = self._env.common_step_counter
        if self._planner_command_provider_token == token:
            return
        renew_ids = torch.nonzero(phase == 0, as_tuple=False).flatten()
        self._planner_command_provider_token = token
        if renew_ids.numel() == 0:
            return
        terms = provider(renew_ids)
        if terms:
            self.set_command(terms, env_ids=renew_ids)

    def capture_held_command_anchor(
        self,
        anchor_body_name: str = "torso_link",
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """Pin the held command anchor pose to the robot's current anchor.

        Published chunks are stored in the anchor frame at publish time and
        re-expressed into the current anchor frame on every consumption step.
        The env-filled (``planner_oracle``) path writes the chunk and captures
        that reference pose atomically inside observation computation, so its
        re-expression is exactly the identity at publication.

        An external publisher writes the buffer at a different instant, so its
        chunk is interpreted against a stale reference pose and the root
        command is systematically wrong. Calling this immediately after
        :meth:`set_command` restores the atomicity.
        """
        env = self._env
        anchor_pos_w, anchor_quat_w = env._get_robot_anchor_state_w_fast(
            anchor_body_name
        )
        anchor_pos_w = anchor_pos_w.reshape(-1, 3)
        anchor_quat_w = anchor_quat_w.reshape(-1, 4)
        # From here on the publisher owns this reference pose: the automatic
        # phase-0 recapture must not clobber it, or the packet (expressed in
        # the publish-time anchor frame) would be re-expressed as if it were
        # already in the consuming step's frame, losing exactly one step of
        # robot motion.
        self._external_command_anchor_owner = True
        stored = self._held_command_anchor_pose.get(anchor_body_name)
        if stored is None:
            self._held_command_anchor_pose[anchor_body_name] = (
                anchor_pos_w.clone(),
                anchor_quat_w.clone(),
            )
            return
        if env_ids is None:
            stored[0].copy_(anchor_pos_w)
            stored[1].copy_(anchor_quat_w)
            return
        env_ids = env_ids.to(device=env.device, dtype=torch.long)
        stored[0].index_copy_(0, env_ids, anchor_pos_w.index_select(0, env_ids))
        stored[1].index_copy_(0, env_ids, anchor_quat_w.index_select(0, env_ids))

    def hold_phase(self) -> torch.Tensor:
        """Per-env step offset within the current command hold window."""
        env = self._env
        hold_steps = int(env._command_hold_steps)
        return env.episode_length_buf.to(dtype=torch.long) % hold_steps

    def update_held_command_anchor_pose(
        self, anchor_body_name: str, phase: torch.Tensor
    ) -> None:
        """Track the anchor pose at each env's last command renewal.

        Published chunks are expressed in the anchor frame at publish time;
        re-expressing them each step needs that reference pose.
        """
        env = self._env
        anchor_pos_w, anchor_quat_w = env._get_robot_anchor_state_w_fast(
            anchor_body_name
        )
        anchor_pos_w = anchor_pos_w.reshape(-1, 3)
        anchor_quat_w = anchor_quat_w.reshape(-1, 4)
        stored = self._held_command_anchor_pose.get(anchor_body_name)
        if (
            stored is None
            or stored[0].shape != anchor_pos_w.shape
            or stored[0].device != anchor_pos_w.device
        ):
            self._held_command_anchor_pose[anchor_body_name] = (
                anchor_pos_w.clone(),
                anchor_quat_w.clone(),
            )
            return
        if self._external_command_anchor_owner:
            # An external publisher pins this pose at publish time;
            # recapturing it here would discard the frame the packet was
            # expressed in.
            return
        renew_mask = phase == 0
        if bool(renew_mask.any()):
            stored[0][renew_mask] = anchor_pos_w[renew_mask]
            stored[1][renew_mask] = anchor_quat_w[renew_mask]

    def reexpress_window_in_current_anchor_frame(
        self,
        flat: torch.Tensor,
        *,
        term_name: str,
        anchor_body_name: str,
        window_steps: int,
    ) -> torch.Tensor:
        """Re-express a held chunk from its publish-time anchor frame.

        Standard VLA-WBC middleware refreshes command coordinates with
        odometry each control step; only the chunk *content* is held at the
        planner rate. Position (``*_pos_b``) and rot6d (``*_ori_b``) terms
        are rigidly transformed from the renewal-time anchor frame into the
        current one; joint-space terms are frame-invariant.
        """
        is_position = term_name.endswith("_pos_b")
        is_orientation = term_name.endswith("_ori_b")
        if not is_position and not is_orientation:
            return flat
        env = self._env
        stored = self._held_command_anchor_pose.get(anchor_body_name)
        if stored is None:
            return flat
        renewal_pos_w, renewal_quat_w = stored
        current_pos_w, current_quat_w = env._get_robot_anchor_state_w_fast(
            anchor_body_name
        )
        current_pos_w = current_pos_w.reshape(-1, 3)
        current_quat_w = current_quat_w.reshape(-1, 4)
        # Relative transform from renewal anchor frame to current anchor frame.
        delta_quat = math_utils.quat_mul(
            math_utils.quat_inv(current_quat_w), renewal_quat_w
        )
        delta_pos = math_utils.quat_apply_inverse(
            current_quat_w, renewal_pos_w - current_pos_w
        )
        num_envs, width = flat.shape
        if is_position:
            if width % 3 != 0:
                raise ValueError(
                    f"Position command term {term_name!r} width {width} is not "
                    "divisible by 3."
                )
            vectors = flat.reshape(num_envs, -1, 3)
            num_vectors = vectors.shape[1]
            delta_quat_exp = (
                delta_quat[:, None, :].expand(-1, num_vectors, -1).reshape(-1, 4)
            )
            rotated = math_utils.quat_apply(
                delta_quat_exp, vectors.reshape(-1, 3)
            ).reshape(num_envs, num_vectors, 3)
            return (rotated + delta_pos[:, None, :]).reshape(num_envs, width)
        if width % 6 != 0:
            raise ValueError(
                f"Orientation command term {term_name!r} width {width} is not "
                "divisible by 6."
            )
        delta_mat = math_utils.matrix_from_quat(delta_quat)
        columns = flat.reshape(num_envs, -1, 3, 2)
        rotated = torch.matmul(delta_mat[:, None, :, :], columns)
        return rotated.reshape(num_envs, width)

    @staticmethod
    def shift_window_by_phase(
        flat: torch.Tensor,
        phase: torch.Tensor,
        *,
        window_steps: int,
    ) -> torch.Tensor:
        """Time-align a held command chunk to the current control step.

        Shifts the frame-major flattened window ``[N, W * D]`` forward by
        each env's hold phase so the leading slot stays time-aligned with
        the current control step, repeating the final frame past the chunk
        end.
        """
        num_envs, width = flat.shape
        window_steps = int(window_steps)
        if window_steps <= 0 or width % window_steps != 0:
            raise ValueError(
                "Held command window width must be divisible by window steps, "
                f"got width={width}, window_steps={window_steps}."
            )
        per_step_dim = width // window_steps
        view = flat.reshape(num_envs, window_steps, per_step_dim)
        offsets = torch.arange(window_steps, device=flat.device, dtype=torch.long)
        indices = (
            offsets[None, :] + phase.to(device=flat.device, dtype=torch.long)[:, None]
        ).clamp_(max=window_steps - 1)
        shifted = view.gather(
            1, indices[:, :, None].expand(num_envs, window_steps, per_step_dim)
        )
        return shifted.reshape(num_envs, width)

    @staticmethod
    def clamp_window_to_hold_boundary(
        flat: torch.Tensor,
        phase: torch.Tensor,
        *,
        window_steps: int,
    ) -> torch.Tensor:
        """Limit a fresh command window to the current hold's information.

        The fresh window at phase ``k`` covers frames ``[t, t + W - 1]`` while
        the chunk published at the last renewal only knew frames up to the
        hold boundary at slot ``W - 1 - k``. Slots past the boundary repeat
        the boundary frame (tail padding), so no post-renewal information
        leaks into the command observation.
        """
        num_envs, width = flat.shape
        window_steps = int(window_steps)
        if window_steps <= 0 or width % window_steps != 0:
            raise ValueError(
                "Held command window width must be divisible by window steps, "
                f"got width={width}, window_steps={window_steps}."
            )
        per_step_dim = width // window_steps
        view = flat.reshape(num_envs, window_steps, per_step_dim)
        offsets = torch.arange(window_steps, device=flat.device, dtype=torch.long)
        boundary = (
            window_steps - 1 - phase.to(device=flat.device, dtype=torch.long)
        ).clamp_(min=0)
        indices = torch.minimum(offsets[None, :], boundary[:, None])
        clamped = view.gather(
            1, indices[:, :, None].expand(num_envs, window_steps, per_step_dim)
        )
        return clamped.reshape(num_envs, width)
