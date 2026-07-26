"""Shared command-publication contract for latent and explicit interfaces.

Motivation
----------
Latent and explicit interfaces currently reach the low-level tracker through
*structurally different* control planes:

* latent: the command is produced inside the agent during ``env.step``; it is a
  stateless 258-value code (z256 + phase) with no window, no anchor frame and
  nothing to re-index;
* explicit: the packet is written from the outer loop into a held buffer that is
  phase-shifted and re-expressed against a stored anchor pose on every
  consumption step.

Any latent-vs-explicit conclusion is therefore confounded by implementation, not
just by command content. This module gives both interfaces one publication
contract so they differ only in *what* they carry.

The three invariants
--------------------
Each caused a real, silent defect in this campaign:

1. **Joint order.** The env consumes joint-space commands through a term pinned
   to ``G1_29DOF_ISAACLAB_JOINT_NAMES`` (``preserve_order=True``), while planners
   predict in the live articulation order, which is backend-specific. Publishing
   without re-indexing delivers every joint target to the wrong joint -- costing
   roughly 380mm vs 118mm of tracking error, while still producing a perfectly
   well-formed packet.
2. **Anchor frame.** ``_reexpress_window_in_current_anchor_frame`` rigidly maps
   ``*_pos_b`` / ``*_ori_b`` terms *from the renewal-time anchor frame* into the
   current one. The buffer contract is therefore "values expressed in the frame
   captured at renewal". A publisher that fetches body-frame quantities at a
   different instant silently biases the root command. Joint-space terms are
   frame-invariant and are unaffected, which is exactly why joint channels can
   look bit-perfect while anchors are wrong.
3. **Reset.** ``reset_agent_trajectory_command`` zeroes the buffer. Under
   ``command_observation_source="planner"`` nothing refills it, so the tracker
   consumes a zero command for at least one step after every reset. Invisible in
   a no-reset protocol, fatal for episodic evaluation.

A publisher owns all three so no call site can get them wrong individually.
"""

from __future__ import annotations

from typing import Any, Protocol

import torch


def renewal_env_ids(
    episode_length_buf: torch.Tensor,
    hold_steps: int,
    *,
    initial: bool = False,
) -> torch.Tensor:
    """Environment ids whose hold window restarts on this control step.

    Uses the same ``episode_length_buf % hold_steps`` phase the environment uses
    internally, so publisher and consumer cannot disagree about when a window
    begins. Reset environments return to episode step zero, which keeps
    asynchronously-resetting environments aligned without a global step counter.
    """
    if episode_length_buf.ndim != 1:
        raise ValueError(
            f"episode_length_buf must be 1-D, got {tuple(episode_length_buf.shape)}."
        )
    if int(hold_steps) <= 0:
        raise ValueError(f"hold_steps must be positive, got {hold_steps}.")
    if initial:
        return torch.arange(
            episode_length_buf.numel(),
            device=episode_length_buf.device,
            dtype=torch.long,
        )
    phase = episode_length_buf.to(dtype=torch.long) % int(hold_steps)
    return torch.nonzero(phase == 0, as_tuple=False).flatten()


class CommandPublisher(Protocol):
    """One control plane for every interface.

    ``publish`` stores what the planner produced; ``consume`` returns what the
    tracker should receive on the current control step. Interfaces differ only
    in how a held prediction becomes a per-step command.
    """

    hold_steps: int

    def due(self, episode_length_buf: torch.Tensor, *, initial: bool = False): ...
    def publish(self, env_ids: torch.Tensor, planner_output: Any) -> None: ...
    def consume(self, env_ids: torch.Tensor | None = None) -> Any: ...
    def reset(self, env_ids: torch.Tensor) -> None: ...


class LatentCommandPublisher:
    """Hold one latent code for ``hold_steps``, advancing only its phase.

    The code is held constant across the window while the phase vector moves, so
    the tracker sees a continuously varying command built from a piecewise
    constant prediction. There is no window to shift and no anchor frame, so
    invariants 1 and 2 are structurally inapplicable -- which is precisely why
    the latent rows were untouched by the joint-order defect.
    """

    def __init__(
        self,
        num_envs: int,
        latent_dim: int,
        hold_steps: int,
        *,
        device: torch.device | str = "cpu",
        phase_mode: str = "sin_cos",
    ) -> None:
        if phase_mode not in ("sin_cos", "linear"):
            raise ValueError(f"Unsupported phase_mode {phase_mode!r}.")
        self.hold_steps = int(hold_steps)
        self.phase_mode = phase_mode
        self._z = torch.zeros(int(num_envs), int(latent_dim), device=device)
        self._phase = torch.zeros(int(num_envs), dtype=torch.long, device=device)
        self._published = torch.zeros(int(num_envs), dtype=torch.bool, device=device)

    def due(self, episode_length_buf: torch.Tensor, *, initial: bool = False):
        return renewal_env_ids(episode_length_buf, self.hold_steps, initial=initial)

    def publish(self, env_ids: torch.Tensor, planner_output: torch.Tensor) -> None:
        env_ids = env_ids.to(device=self._z.device, dtype=torch.long)
        z = planner_output.to(device=self._z.device, dtype=self._z.dtype)
        if z.shape[0] != env_ids.shape[0]:
            raise ValueError(
                f"planner_output rows {z.shape[0]} != env_ids {env_ids.shape[0]}."
            )
        self._z.index_copy_(0, env_ids, z)
        self._phase.index_fill_(0, env_ids, 0)
        self._published.index_fill_(0, env_ids, True)

    def step(self) -> None:
        """Advance the phase of every environment by one control step."""
        self._phase = (self._phase + 1) % self.hold_steps

    def consume(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        if not bool(self._published.all()):
            raise RuntimeError(
                "consume() before every environment has a published command; the "
                "tracker would receive a zeroed latent (see invariant 3)."
            )
        z, phase = self._z, self._phase
        if env_ids is not None:
            env_ids = env_ids.to(device=z.device, dtype=torch.long)
            z, phase = z.index_select(0, env_ids), phase.index_select(0, env_ids)
        frac = phase.to(z.dtype) / float(self.hold_steps)
        if self.phase_mode == "linear":
            extra = frac.unsqueeze(-1)
        else:
            ang = 2.0 * torch.pi * frac
            extra = torch.stack((torch.sin(ang), torch.cos(ang)), dim=-1)
        return torch.cat((z, extra), dim=-1)

    def reset(self, env_ids: torch.Tensor) -> None:
        env_ids = env_ids.to(device=self._z.device, dtype=torch.long)
        self._phase.index_fill_(0, env_ids, 0)
        # Deliberately NOT zeroing z: a zeroed command is a valid-looking but
        # meaningless latent. Mark unpublished so consume() fails loudly instead.
        self._published.index_fill_(0, env_ids, False)


class ChunkCommandPublisher:
    """Hold an explicit window and emit one slot per control step.

    Owns all three invariants for joint-space + anchor packets:
    joint re-indexing at publish time, the renewal-time anchor pose the packet is
    expressed in, and refusal to serve an unpublished (zeroed) buffer.
    """

    def __init__(
        self,
        num_envs: int,
        term_widths: dict[str, int],
        hold_steps: int,
        window_steps: int,
        *,
        device: torch.device | str = "cpu",
        joint_reindex: torch.Tensor | None = None,
        joint_term: str = "expert_motion",
    ) -> None:
        if window_steps < hold_steps:
            raise ValueError(
                f"window_steps {window_steps} < hold_steps {hold_steps}: some held "
                "control step would have no command slot."
            )
        self.hold_steps = int(hold_steps)
        self.window_steps = int(window_steps)
        self.joint_reindex = joint_reindex
        self.joint_term = joint_term
        self._buf = {
            name: torch.zeros(int(num_envs), int(w), device=device)
            for name, w in term_widths.items()
        }
        self._phase = torch.zeros(int(num_envs), dtype=torch.long, device=device)
        self._published = torch.zeros(int(num_envs), dtype=torch.bool, device=device)
        self._anchor_pos = torch.zeros(int(num_envs), 3, device=device)
        self._anchor_quat = torch.zeros(int(num_envs), 4, device=device)
        self._anchor_quat[:, 0] = 1.0

    def due(self, episode_length_buf: torch.Tensor, *, initial: bool = False):
        return renewal_env_ids(episode_length_buf, self.hold_steps, initial=initial)

    def pin_joint_order(
        self, terms: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Re-index joint-space channels into the order the env consumes."""
        if self.joint_reindex is None or self.joint_term not in terms:
            return terms
        value = terms[self.joint_term]
        width = int(value.shape[-1])
        per_frame = width // self.window_steps
        half = per_frame // 2
        if half != int(self.joint_reindex.numel()):
            raise ValueError(
                f"{self.joint_term} has {half} joints per half but the pinned order "
                f"defines {int(self.joint_reindex.numel())}."
            )
        idx = self.joint_reindex.to(value.device)
        frames = value.view(-1, self.window_steps, per_frame)
        out = torch.cat(
            (
                frames[..., :half].index_select(-1, idx),
                frames[..., half:].index_select(-1, idx),
            ),
            dim=-1,
        )
        result = dict(terms)
        result[self.joint_term] = out.reshape(-1, width)
        return result

    def publish(
        self,
        env_ids: torch.Tensor,
        planner_output: dict[str, torch.Tensor],
        *,
        anchor_pos: torch.Tensor | None = None,
        anchor_quat: torch.Tensor | None = None,
    ) -> None:
        """Store a packet together with the anchor frame it is expressed in.

        ``anchor_pos``/``anchor_quat`` must be the robot anchor pose at the
        instant ``planner_output`` was produced. Consumption re-expresses the
        packet from this pose, so supplying a pose from a different instant
        reintroduces invariant 2.
        """
        env_ids = env_ids.to(device=self._phase.device, dtype=torch.long)
        terms = self.pin_joint_order(planner_output)
        for name, buf in self._buf.items():
            if name not in terms:
                raise KeyError(f"publish() missing command term {name!r}.")
            buf.index_copy_(0, env_ids, terms[name].to(buf.device, buf.dtype))
        self._phase.index_fill_(0, env_ids, 0)
        self._published.index_fill_(0, env_ids, True)
        if anchor_pos is not None:
            self._anchor_pos.index_copy_(
                0, env_ids, anchor_pos.to(self._anchor_pos.device)
            )
        if anchor_quat is not None:
            self._anchor_quat.index_copy_(
                0, env_ids, anchor_quat.to(self._anchor_quat.device)
            )

    def step(self) -> None:
        self._phase = (self._phase + 1) % self.hold_steps

    def held_anchor(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the renewal-time anchor pose each held packet is expressed in."""
        return self._anchor_pos, self._anchor_quat

    def consume(self, env_ids: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Return this step's slot from each held term (pre re-expression)."""
        if not bool(self._published.all()):
            raise RuntimeError(
                "consume() before every environment has a published packet; the "
                "tracker would receive a zeroed command (see invariant 3)."
            )
        phase = self._phase
        out: dict[str, torch.Tensor] = {}
        for name, buf in self._buf.items():
            value = buf
            ph = phase
            if env_ids is not None:
                ids = env_ids.to(device=buf.device, dtype=torch.long)
                value, ph = buf.index_select(0, ids), phase.index_select(0, ids)
            per_frame = int(value.shape[-1]) // self.window_steps
            frames = value.view(-1, self.window_steps, per_frame)
            slot = ph.clamp_max(self.window_steps - 1)
            out[name] = frames[
                torch.arange(frames.shape[0], device=frames.device), slot
            ]
        return out

    def reset(self, env_ids: torch.Tensor) -> None:
        env_ids = env_ids.to(device=self._phase.device, dtype=torch.long)
        self._phase.index_fill_(0, env_ids, 0)
        # Mark unpublished rather than zeroing: a zeroed packet is well formed and
        # would be consumed silently. Invariant 3 says fail loudly instead.
        self._published.index_fill_(0, env_ids, False)
