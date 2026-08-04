# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The actor command channel: one family, three sibling emitters.

Exactly one of these terms is built per environment, under the fixed manager
name ``actor``. They differ only in who produces the command:

* :class:`ExplicitActorCommand` -- derived from the reference channel
  (``source="reference"``). The oracle / direct explicit tracker row.
* :class:`LatentActorCommand` -- published by the agent (``source="agent"``).
  The skill-latent row: z + phase, written by the RLOpt latent-command
  pipeline, held between publications.
* :class:`ChunkActorCommand` -- published as a packet (``source="external"``,
  or ``source="reference"`` for the oracle self-fill used to train and certify
  the streamed row). The planner row: one packet of ``horizon`` frames per hold
  window, consumed one phase-aligned slot per control step and re-expressed in
  the robot's current anchor frame, which is what real VLA-WBC middleware does
  with odometry.

All three answer :meth:`component`, so an observation term reads
``command_manager.get_term("actor").component(name)`` without knowing which
emitter is installed. The explicit and chunk emitters serve the same component
names as the reference channel; the latent emitter serves only its own
``latent_command``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import isaaclab.utils.math as math_utils
import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_imitation.contracts.command_channels import (
    ACTOR_TERM_NAME,
    REFERENCE_TERM_NAME,
)

from ...command_components import (
    COMMAND_COMPONENT_TERM_NAMES,
    COMMAND_TERM_NAME_COMPONENTS,
    FULL_BODY_COMPONENTS,
    LATENT_COMMAND_TERM_NAME,
    component_term_names,
    normalize_command_components,
)
from .published_command import PublishedCommandTerm, PublishedCommandTermCfg

if TYPE_CHECKING:
    from isaaclab_imitation.envs import ImitationRLEnv

_ACTOR_SOURCES = frozenset({"reference", "agent", "external"})


def _resolve_source(cfg: Any, *, allowed: frozenset[str]) -> None:
    """Normalize and validate an actor command's declared producer."""
    source = str(cfg.source).strip().lower()
    if source not in _ACTOR_SOURCES:
        raise ValueError(
            f"Unsupported actor command source {source!r}; expected one of "
            f"{sorted(_ACTOR_SOURCES)}."
        )
    if source not in allowed:
        raise ValueError(
            f"{type(cfg).__name__} does not support source={source!r}; expected "
            f"one of {sorted(allowed)}."
        )
    cfg.source = source


def _check_component(cfg: Any, name: str) -> None:
    if name not in cfg.components:
        raise KeyError(
            f"Component {name!r} is not part of the actor command "
            f"{list(cfg.components)}."
        )


def _reject_window_override(past_steps: int, future_steps: int) -> None:
    """The actor channel's window is its own; a caller may not widen it.

    A consumer that wants a different window (the skill encoder's, say) reads
    the reference channel, which is what that view is: privileged reference
    data, not the actor's command.
    """
    if int(past_steps) != 0 or int(future_steps) != 0:
        raise ValueError(
            "The actor command channel serves its configured window; ask the "
            "reference channel for a different one."
        )


def _reference_term(env: ImitationRLEnv):
    """The reference channel term, resolved through the CommandManager."""
    try:
        return env.command_manager.get_term(REFERENCE_TERM_NAME)
    except (KeyError, AttributeError) as err:
        raise RuntimeError(
            "The actor command channel needs the always-present "
            f"{REFERENCE_TERM_NAME!r} command term; it was not found."
        ) from err


# ---------------------------------------------------------------------------
# Explicit: derived from the reference channel.
# ---------------------------------------------------------------------------


class ExplicitActorCommand(CommandTerm):
    """Explicit actor command read straight off the reference channel.

    ``past_steps``/``future_steps`` are the actor's command window; ``0/0`` (the
    default) is the single-frame command the explicit trackers are trained on,
    and a non-trivial window is the encoder-style view over the same components.
    """

    # pyrefly: ignore[bad-override-mutable-attribute]  # Isaac Lab term idiom
    cfg: ExplicitCommandCfg

    def __init__(self, cfg: ExplicitCommandCfg, env: ImitationRLEnv):
        super().__init__(cfg, env)
        self._command: torch.Tensor | None = None

    def __str__(self) -> str:
        msg = "ExplicitActorCommand (derived from the reference channel):\n"
        msg += f"\tComponents: {list(self.cfg.components)}\n"
        msg += f"\tWindow: past={self.cfg.past_steps}, future={self.cfg.future_steps}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """Every selected component, concatenated in canonical order."""
        parts = [self.component(name) for name in self.cfg.components]
        self._command = torch.cat(parts, dim=-1)
        return self._command

    def component(
        self, name: str, *, past_steps: int = 0, future_steps: int = 0
    ) -> torch.Tensor:
        """One component of the actor command, over the actor's own window."""
        _check_component(self.cfg, name)
        _reject_window_override(past_steps, future_steps)
        return _reference_term(self._imitation_env()).component(
            name,
            past_steps=int(self.cfg.past_steps),
            future_steps=int(self.cfg.future_steps),
        )

    def _update_command(self):
        """No-op: the command is a live view of the reference channel."""

    def _update_metrics(self):
        """No metrics: the reference channel owns the tracking metrics."""

    def _resample_command(self, env_ids: Sequence[int]):
        """No-op: the reference channel owns reference resampling."""

    def _set_debug_vis_impl(self, debug_vis: bool):
        """No-op: actor command terms carry no debug visualization."""

    def _imitation_env(self) -> ImitationRLEnv:
        return self._env  # type: ignore[return-value]

    def _command_dim(self) -> int:
        return int(
            sum(int(self.component(name).shape[-1]) for name in self.cfg.components)
        )


@configclass
class ExplicitCommandCfg(CommandTermCfg):
    """Explicit actor command derived from the reference channel."""

    class_type: type = ExplicitActorCommand

    # The reference channel owns resampling; this term is a view.
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

    source: str = "reference"
    components: tuple[str, ...] = FULL_BODY_COMPONENTS
    past_steps: int = 0
    future_steps: int = 0

    def resolve(self) -> None:
        _resolve_source(self, allowed=frozenset({"reference"}))
        self.components = normalize_command_components(self.components)
        if int(self.past_steps) < 0 or int(self.future_steps) < 0:
            raise ValueError("command window steps must be >= 0.")

    def command_terms(self) -> tuple[str, ...]:
        return component_term_names(self.components)


# ---------------------------------------------------------------------------
# Latent: published by the agent.
# ---------------------------------------------------------------------------


class LatentActorCommand(PublishedCommandTerm):
    """Agent-published skill latent (z + phase).

    Owns the buffer the RLOpt latent-command pipeline writes through
    :meth:`set` (or the publisher-facing :meth:`publish`), the observation term
    reads, and the environment's reset path zero-fills.
    """

    # pyrefly: ignore[bad-override-mutable-attribute]  # Isaac Lab term idiom
    cfg: LatentCommandCfg

    def __init__(self, cfg: LatentCommandCfg, env: ImitationRLEnv):
        super().__init__(cfg, env)
        dim = int(cfg.dim)
        if dim <= 0:
            raise ValueError("LatentCommandCfg.dim must be > 0.")
        self._latent = torch.zeros(
            (self.num_envs, dim), device=self.device, dtype=torch.float32
        )

    def __str__(self) -> str:
        return (
            "LatentActorCommand (agent-published skill latent):\n"
            f"\tCommand dimension: {int(self.cfg.dim)}\n"
            f"\tHold steps: {self.hold_steps}"
        )

    @property
    def command(self) -> torch.Tensor:
        """The published latent command. Shape is (num_envs, dim)."""
        return self._latent

    def component(
        self, name: str, *, past_steps: int = 0, future_steps: int = 0
    ) -> torch.Tensor:
        if name != LATENT_COMMAND_TERM_NAME:
            raise KeyError(
                f"A latent actor command serves only "
                f"{LATENT_COMMAND_TERM_NAME!r}, not {name!r}."
            )
        _reject_window_override(past_steps, future_steps)
        return self._latent

    def get(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Read the buffer, optionally for a subset of environments."""
        if env_ids is None:
            return self._latent
        env_ids = env_ids.to(device=self._latent.device, dtype=torch.long)
        return self._latent.index_select(0, env_ids)

    def set(
        self, latent_command: torch.Tensor, env_ids: torch.Tensor | None = None
    ) -> None:
        """Write the buffer (the agent-facing publication path)."""
        latent_command = latent_command.to(
            device=self._latent.device, dtype=torch.float32
        )
        if env_ids is None:
            if latent_command.shape != self._latent.shape:
                raise ValueError(
                    "Latent command shape mismatch. Expected "
                    f"{tuple(self._latent.shape)}, got {tuple(latent_command.shape)}."
                )
            self._latent.copy_(latent_command)
            self._published.fill_(True)
            return
        env_ids = env_ids.to(device=self._latent.device, dtype=torch.long)
        expected = (int(env_ids.shape[0]), int(self._latent.shape[1]))
        if latent_command.ndim != 2 or tuple(latent_command.shape) != expected:
            raise ValueError(
                "Latent command shape mismatch for indexed update. Expected "
                f"{expected}, got {tuple(latent_command.shape)}."
            )
        self._latent.index_copy_(0, env_ids, latent_command)
        self._published.index_fill_(0, env_ids, True)

    def reset_command(self, env_ids: torch.Tensor | None = None) -> None:
        """Zero the buffer for the selected environments."""
        if env_ids is None:
            self._latent.zero_()
            self._published.fill_(False)
            return
        env_ids = env_ids.to(device=self._latent.device, dtype=torch.long)
        self._latent.index_fill_(0, env_ids, 0.0)
        self._published.index_fill_(0, env_ids, False)

    def _apply_published_payload(
        self, env_ids: torch.Tensor, payload: torch.Tensor
    ) -> None:
        self.set(payload, env_ids=env_ids)

    def _update_command(self):
        """No-op: the latent lives in the buffer between publications."""

    def _update_metrics(self):
        """No metrics: the reference channel owns the tracking metrics."""

    def _resample_command(self, env_ids: Sequence[int]):
        """Zero the latent for resetting environments and clear the mask.

        A stale latent from the previous episode is a valid-looking command; the
        environment's reset path must not leave one in place.
        """
        super()._resample_command(env_ids)
        self._latent[env_ids] = 0.0

    def _command_dim(self) -> int:
        return int(self.cfg.dim)


@configclass
class LatentCommandCfg(PublishedCommandTermCfg):
    """Implicit (skill-latent) actor command, published by the agent."""

    class_type: type = LatentActorCommand

    source: str = "agent"
    dim: int = 258

    def resolve(self) -> None:
        _resolve_source(self, allowed=frozenset({"agent"}))
        if int(self.dim) <= 0:
            raise ValueError("latent command dim must be positive.")
        if int(self.hold_steps) <= 0:
            raise ValueError("latent command hold_steps must be positive.")

    def command_terms(self) -> tuple[str, ...]:
        return (LATENT_COMMAND_TERM_NAME,)


# ---------------------------------------------------------------------------
# Chunk: a published packet, consumed one slot per control step.
# ---------------------------------------------------------------------------


def shift_window_by_phase(
    flat: torch.Tensor, phase: torch.Tensor, *, window_steps: int
) -> torch.Tensor:
    """Time-align a held packet to the current control step.

    Shifts the frame-major flattened window ``[N, W * D]`` forward by each
    environment's hold phase so the leading slot stays aligned with the current
    control step, repeating the final frame past the packet end.
    """
    num_envs, width = flat.shape
    window_steps = int(window_steps)
    if window_steps <= 0 or width % window_steps != 0:
        raise ValueError(
            "Held command window width must be divisible by window steps, got "
            f"width={width}, window_steps={window_steps}."
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


class ChunkActorCommand(PublishedCommandTerm):
    """A published explicit packet, consumed one phase-aligned slot per step.

    Owns the packet buffers, the publish-time anchor pose, the phase shift, and
    the anchor re-expression: published coordinates are expressed in the anchor
    frame of the instant they were published and refreshed into the robot's
    current anchor frame on every consumption step, exactly as odometry-based
    target re-expression works on a real stack. Joint-space components are
    frame-invariant and pass through.

    ``source="reference"`` self-fills the packet from the reference channel at
    each renewal, inside the observation pass, so the packet is expressed in the
    frame of the step that consumes it and its re-expression is the identity at
    publication -- the oracle row used to train and to certify the streamed
    interface. ``source="external"`` expects a planner to call :meth:`publish`
    (or to register a provider through :meth:`set_publisher`).

    Unpublished consumption serves the zero-filled buffer rather than failing:
    an environment computes observations at reset, before any external publisher
    has had a step to run. The ``published`` mask and the ``command_err`` metric
    record what has actually been written.
    """

    # pyrefly: ignore[bad-override-mutable-attribute]  # Isaac Lab term idiom
    cfg: ChunkCommandCfg

    def __init__(self, cfg: ChunkCommandCfg, env: ImitationRLEnv):
        super().__init__(cfg, env)
        self._packet: dict[str, torch.Tensor] = {}
        self._anchor_pose: tuple[torch.Tensor, torch.Tensor] | None = None
        self._external_anchor_owner = False
        self._publisher: Any = None
        self._fill_token: int | None = None
        self.metrics["command_err"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        return (
            "ChunkActorCommand (published packet, current slot):\n"
            f"\tComponents: {list(self.cfg.components)}\n"
            f"\tHorizon: {int(self.cfg.horizon)} frames\n"
            f"\tHold steps: {self.hold_steps}\n"
            f"\tSource: {self.cfg.source}"
        )

    """
    Properties.
    """

    @property
    def window_steps(self) -> int:
        """Frames per published packet."""
        return int(self.cfg.horizon)

    @property
    def command(self) -> torch.Tensor:
        """Every selected component's current slot, concatenated."""
        return torch.cat([self.component(name) for name in self.cfg.components], dim=-1)

    """
    Operations.
    """

    def component(
        self, name: str, *, past_steps: int = 0, future_steps: int = 0
    ) -> torch.Tensor:
        """The slot of the packet time-aligned with this control step."""
        _check_component(self.cfg, name)
        _reject_window_override(past_steps, future_steps)
        self._maybe_fill_from_source()
        window = self._packet_window(name)
        per_step_dim = window.shape[1] // self.window_steps
        return window.reshape(-1, self.window_steps, per_step_dim)[:, 0, :]

    def set_publisher(self, provider: Any) -> None:
        """Register a callback producing packets in-step.

        The callback receives the environment ids being renewed and returns a
        mapping of component name to packet tensor for those environments. It is
        evaluated inside the observation pass, so the packet is expressed in the
        anchor frame of the step that consumes it -- an external publisher
        writing between steps fetches body-frame quantities one physics step
        early, which silently biases the root command.
        """
        self._publisher = provider
        self._fill_token = None

    """
    Implementation specific functions.
    """

    def _apply_published_payload(
        self, env_ids: torch.Tensor, payload: Mapping[str, torch.Tensor]
    ) -> None:
        keys = {_component_of(key) for key in payload}
        expected = set(self.cfg.components)
        if keys != expected:
            raise KeyError(
                "A chunk packet must carry exactly the configured components "
                f"{sorted(expected)}, got {sorted(keys)}."
            )
        self._write_packet(payload, env_ids=env_ids)
        self._capture_anchor_pose(env_ids=env_ids)

    def _update_command(self):
        """No-op: the packet lives in the buffers between publications."""

    def _update_metrics(self):
        """Consumed slot vs the live reference, where a packet was published."""
        env = self._imitation_env()
        if getattr(env, "current_expert_frame", None) is None:
            return
        if not bool(self._published.any()):
            self.metrics["command_err"].zero_()
            return
        reference = _reference_term(env)
        consumed = self.command
        live = torch.cat(
            [reference.component(name) for name in self.cfg.components], dim=-1
        )
        err = (consumed - live).abs().mean(dim=-1)
        self.metrics["command_err"][:] = torch.where(
            self._published, err, torch.zeros_like(err)
        )

    def _resample_command(self, env_ids: Sequence[int]):
        """Zero the packet for resetting environments and clear the mask."""
        super()._resample_command(env_ids)
        for buffer in self._packet.values():
            buffer[env_ids] = 0.0

    def _command_dim(self) -> int:
        reference = _reference_term(self._imitation_env())
        return int(
            sum(
                int(reference.component(name).shape[-1]) for name in self.cfg.components
            )
        )

    """
    Helper functions.
    """

    def _imitation_env(self) -> ImitationRLEnv:
        return self._env  # type: ignore[return-value]

    def _reference_window(self, name: str) -> torch.Tensor:
        """The reference channel's packet-shaped window for one component."""
        return _reference_term(self._imitation_env()).component(
            name, past_steps=0, future_steps=self.window_steps - 1
        )

    def _ensure_buffer(self, name: str) -> torch.Tensor:
        buffer = self._packet.get(name)
        if buffer is not None:
            return buffer
        width = int(self._reference_window(name).shape[-1])
        buffer = torch.zeros(
            (self.num_envs, width), device=self.device, dtype=torch.float32
        )
        self._packet[name] = buffer
        return buffer

    def _write_packet(
        self,
        packet: Mapping[str, torch.Tensor],
        env_ids: torch.Tensor | None = None,
    ) -> None:
        for key, value in packet.items():
            name = _component_of(key)
            target = self._ensure_buffer(name)
            value = value.to(device=self.device, dtype=torch.float32)
            if env_ids is None:
                if value.shape != target.shape:
                    raise ValueError(
                        f"Chunk component {name!r} shape mismatch. Expected "
                        f"{tuple(target.shape)}, got {tuple(value.shape)}."
                    )
                target.copy_(value)
                continue
            env_ids = env_ids.to(device=self.device, dtype=torch.long)
            expected = (int(env_ids.shape[0]), int(target.shape[1]))
            if value.ndim != 2 or tuple(value.shape) != expected:
                raise ValueError(
                    f"Chunk component {name!r} indexed shape mismatch. Expected "
                    f"{expected}, got {tuple(value.shape)}."
                )
            target.index_copy_(0, env_ids, value)

    def _maybe_fill_from_source(self) -> None:
        """Renew the packet once per control step, for the environments due.

        ``component`` runs once per observation term; the fill must run once per
        control step, so it is gated on the environment's step counter.
        """
        env = self._imitation_env()
        token = int(env.common_step_counter)
        if self._fill_token == token:
            return
        self._fill_token = token
        renew_ids = self.due()
        if renew_ids.numel() == 0:
            return
        if self.cfg.source == "reference":
            packet = {
                name: self._reference_window(name).index_select(0, renew_ids)
                for name in self.cfg.components
            }
        elif self._publisher is not None:
            packet = self._publisher(renew_ids)
        else:
            return
        if packet:
            self.publish(renew_ids, packet)

    def _packet_window(self, name: str) -> torch.Tensor:
        """The stored packet, phase-shifted and re-expressed for this step."""
        buffer = self._ensure_buffer(name)
        phase = self.hold_phase
        self._update_anchor_pose(phase)
        # Observations must not alias the mutable packet buffers.
        window = shift_window_by_phase(
            buffer.clone(), phase, window_steps=self.window_steps
        )
        return self._reexpress_in_current_anchor_frame(window, name)

    def _anchor_body_name(self) -> str:
        return str(_reference_term(self._imitation_env()).cfg.anchor_body_name)

    def _robot_anchor_state_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        env = self._imitation_env()
        pos_w, quat_w = env._get_robot_anchor_state_w_fast(self._anchor_body_name())
        return pos_w.reshape(-1, 3), quat_w.reshape(-1, 4)

    def _capture_anchor_pose(self, env_ids: torch.Tensor | None = None) -> None:
        """Pin the anchor frame a packet is expressed in, at publish time."""
        pos_w, quat_w = self._robot_anchor_state_w()
        # From here on the publisher owns this pose: the automatic phase-0
        # recapture must not clobber it, or the packet would be re-expressed as
        # if it were already in the consuming step's frame, losing exactly one
        # step of robot motion.
        self._external_anchor_owner = True
        if self._anchor_pose is None:
            self._anchor_pose = (pos_w.clone(), quat_w.clone())
            return
        stored_pos, stored_quat = self._anchor_pose
        if env_ids is None:
            stored_pos.copy_(pos_w)
            stored_quat.copy_(quat_w)
            return
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        stored_pos.index_copy_(0, env_ids, pos_w.index_select(0, env_ids))
        stored_quat.index_copy_(0, env_ids, quat_w.index_select(0, env_ids))

    def _update_anchor_pose(self, phase: torch.Tensor) -> None:
        """Track the anchor pose at each environment's last renewal."""
        pos_w, quat_w = self._robot_anchor_state_w()
        if self._anchor_pose is None:
            self._anchor_pose = (pos_w.clone(), quat_w.clone())
            return
        if self._external_anchor_owner:
            return
        stored_pos, stored_quat = self._anchor_pose
        renew_mask = phase == 0
        if bool(renew_mask.any()):
            stored_pos[renew_mask] = pos_w[renew_mask]
            stored_quat[renew_mask] = quat_w[renew_mask]

    def _reexpress_in_current_anchor_frame(
        self, flat: torch.Tensor, name: str
    ) -> torch.Tensor:
        """Refresh packet coordinates from the publish-time anchor frame.

        Position and rot6d components are rigidly transformed from the renewal
        anchor frame into the current one; joint-space components are
        frame-invariant.
        """
        is_position = name.endswith("_pos")
        is_orientation = name.endswith("_ori")
        if not is_position and not is_orientation:
            return flat
        if self._anchor_pose is None:
            return flat
        renewal_pos_w, renewal_quat_w = self._anchor_pose
        current_pos_w, current_quat_w = self._robot_anchor_state_w()
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
                    f"Position component {name!r} width {width} is not divisible by 3."
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
                f"Orientation component {name!r} width {width} is not divisible by 6."
            )
        delta_mat = math_utils.matrix_from_quat(delta_quat)
        columns = flat.reshape(num_envs, -1, 3, 2)
        rotated = torch.matmul(delta_mat[:, None, :, :], columns)
        return rotated.reshape(num_envs, width)


@configclass
class ChunkCommandCfg(PublishedCommandTermCfg):
    """Held explicit packet: a publisher writes ``horizon`` frames per window."""

    class_type: type = ChunkActorCommand

    source: str = "external"
    components: tuple[str, ...] = FULL_BODY_COMPONENTS

    horizon: int = 10
    """Frames per published packet (current frame plus ``horizon - 1`` future)."""

    hold_steps: int = 10
    """Control steps a packet is held before renewal is due."""

    def resolve(self) -> None:
        _resolve_source(self, allowed=frozenset({"external", "reference"}))
        self.components = normalize_command_components(self.components)
        if int(self.horizon) <= 0:
            raise ValueError("chunk horizon must be positive.")
        if int(self.hold_steps) <= 0:
            raise ValueError("chunk hold_steps must be positive.")
        if int(self.horizon) < int(self.hold_steps):
            raise ValueError(
                "A chunk needs at least one command frame per held control step: "
                f"horizon={int(self.horizon)}, hold_steps={int(self.hold_steps)}."
            )

    def command_terms(self) -> tuple[str, ...]:
        return component_term_names(self.components)

    @property
    def past_steps(self) -> int:
        """Held consumption is defined for future-only windows."""
        return 0

    @property
    def future_steps(self) -> int:
        return int(self.horizon) - 1


def _component_of(key: str) -> str:
    """Accept either a component name or its observation term name."""
    name = str(key)
    if name in COMMAND_COMPONENT_TERM_NAMES:
        return name
    try:
        return COMMAND_TERM_NAME_COMPONENTS[name]
    except KeyError as err:
        raise KeyError(f"Unknown command component {key!r}.") from err


ActorCommandCfg = LatentCommandCfg | ExplicitCommandCfg | ChunkCommandCfg
"""The three actor command kinds; exactly one is built per environment."""

__all__ = [
    "ACTOR_TERM_NAME",
    "REFERENCE_TERM_NAME",
    "ActorCommandCfg",
    "ChunkActorCommand",
    "ChunkCommandCfg",
    "ExplicitActorCommand",
    "ExplicitCommandCfg",
    "LatentActorCommand",
    "LatentCommandCfg",
    "shift_window_by_phase",
]
