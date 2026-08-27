"""Publish a GR00T explicit head's root_qpos window into the chunk actor term.

Option 1 of the explicit-arm evaluation: the head's `[B, horizon, 38]`
prediction is the literal robot command. `ChunkActorCommand` already owns
the packet buffers, the publish-time anchor capture, and the per-step
re-expression into the live robot anchor, so this publisher only has to
produce the packet and hand it over on the term's renewal schedule.

Packet contract (`ChunkActorCommand._apply_published_payload`): a mapping
component name -> `[B, horizon * width]`, each block term-major (frames
flattened row-major within the component). The component set must match the
configured `env.command_interface.actor.components` exactly.

Run the environment with::

    env.command_interface.actor=chunk
    env.command_interface.actor.source=external
    env.command_interface.actor.horizon=30
    env.command_interface.actor.hold_steps=10
    env.command_interface.actor.components=[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b]

so the consumed slot is a 38-wide root_qpos command an explicit tracker
reads directly. Horizon 30 with hold 10 means the packet carries two
renewals of slack; horizon 30 with hold 30 consumes the whole prediction
open loop.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor

from imitation_experiments.planner.gr00t_isaac_sampler import Gr00tSkillCommandSampler

# The same 38 root_qpos values are named by TWO different vocabularies, and
# the two routes each validate against their own. Mixing them fails loudly at
# run time, which is how this pair was found.
#
# Command space — what `ChunkCommandCfg.components` accepts, so this is the
# payload keying for the native chunk route.
ROOT_QPOS_COMMAND_COMPONENTS: tuple[tuple[str, int], ...] = (
    ("joint_qpos", 29),
    ("root_pos", 3),
    ("root_ori", 6),
)
# Expert-macro-state terms — what the environment's frame layout exposes, so
# this is what a PacketLayout must use on the encoded (BB1) route.
ROOT_QPOS_MACRO_TERMS: tuple[tuple[str, int], ...] = (
    ("expert_motion_qpos", 29),
    ("expert_anchor_pos_b", 3),
    ("expert_anchor_ori_b", 6),
)
# Back-compat alias for the publish payload.
ROOT_QPOS_COMPONENT_WIDTHS = ROOT_QPOS_COMMAND_COMPONENTS


class Gr00tChunkPublisher(Gr00tSkillCommandSampler):
    """Drive `ChunkActorCommand` from a GR00T explicit (chunk-target) head."""

    def __init__(
        self,
        *,
        chunk_term: Any,
        causal_observation_fn: Callable[..., Any],
        state_history_steps: int,
        gr00t_checkpoint: str | Path,
        goal_features_path: str | Path,
        goal_name: str | Sequence[str],
        num_envs: int,
        device: torch.device | str = "cuda",
        components: tuple[tuple[str, int], ...] = ROOT_QPOS_COMPONENT_WIDTHS,
        pin_anchor_state_fn: Callable[[], tuple[Tensor, Tensor]] | None = None,
    ) -> None:
        self._chunk_term = chunk_term
        self._causal_observation_fn = causal_observation_fn
        self._causal_history_steps = int(state_history_steps)
        self._components = tuple(components)
        # When the head's training frame is NOT the full anchor pose (a
        # robot_heading collection driving a robot-frame tracker), the term's
        # publish-time anchor capture is wrong for this packet: re-pin the
        # frame the prediction actually lives in, fetched at publish time.
        self._pin_anchor_state_fn = pin_anchor_state_fn
        self.provenance = self.configure_gr00t(
            checkpoint_path=gr00t_checkpoint,
            goal_features_path=goal_features_path,
            goal_name=goal_name,
            num_envs=num_envs,
            consumption="fresh",  # a packet is republished, never slot-cached
            device=device,
            expected_target_mode="chunk",
        )
        frame_width = sum(width for _, width in self._components)
        if self._gr00t_action_dim != frame_width:
            msg = (
                f"head frame width {self._gr00t_action_dim} does not match the "
                f"configured components {self._components} (sum {frame_width})."
            )
            raise ValueError(msg)
        # `chunk_term` is None on the encoded route, where the packet never
        # reaches a chunk actor term — it goes through the frozen encoder.
        if chunk_term is not None:
            term_horizon = int(getattr(chunk_term, "window_steps", 0))
            if term_horizon != self._gr00t_horizon:
                msg = (
                    f"chunk term horizon {term_horizon} != head action_horizon "
                    f"{self._gr00t_horizon}; configure "
                    f"env.command_interface.actor.horizon={self._gr00t_horizon}."
                )
                raise ValueError(msg)
        self.publications = 0

    def publish(self, env_ids: Tensor) -> None:
        """Predict and publish one packet for each environment in `env_ids`."""
        if self._chunk_term is None:
            msg = "publish() needs a chunk actor term; this is the encoded route."
            raise ValueError(msg)
        env_ids = torch.as_tensor(env_ids, dtype=torch.long).reshape(-1)
        if int(env_ids.numel()) == 0:
            return
        batch = self._causal_observation_fn(
            env_ids=env_ids, history_steps=self._causal_history_steps
        )
        planner_state = (
            batch.get(("planner", "state_history"))
            .reshape(int(env_ids.numel()), -1)
            .to(device=self._gr00t_device, dtype=torch.float32)
        )
        prediction = self._gr00t_predict(
            planner_state, self._gr00t_goal_index[env_ids.to(self._gr00t_device)]
        )
        payload: dict[str, Tensor] = {}
        cursor = 0
        for name, width in self._components:
            block = prediction[:, :, cursor : cursor + width]
            payload[name] = block.reshape(int(env_ids.numel()), -1).contiguous()
            cursor += width
        self._chunk_term.publish(env_ids.to(prediction.device), payload)
        # getattr, not attribute access: existing tests build this publisher
        # through __new__ and set only the fields publish() needs.
        pin_fn = getattr(self, "_pin_anchor_state_fn", None)
        if pin_fn is not None:
            pin_pos, pin_quat = pin_fn()
            index = env_ids.to(pin_pos.device)
            self._chunk_term.pin_anchor_pose(
                env_ids,
                pin_pos.index_select(0, index),
                pin_quat.index_select(0, index),
            )
        self.publications += 1

    def report(self) -> dict[str, Any]:
        record = dict(self.provenance)
        record.update(self.gr00t_stats())
        record["publications"] = int(self.publications)
        return record


class Gr00tPacketPlanner(torch.nn.Module):
    """Adapter presenting a chunk head as a BB1 packet planner.

    Option 2 of the explicit-arm evaluation: the same explicit head, but its
    predicted window is routed through the frozen skill encoder and published
    as a latent, so the head can be scored on a latent tracker.
    `install_packet_encoder_command_source` calls a planner as
    ``planner(causal_state, num_inference_steps=..., inference_noise_std=...)``
    and expects a term-major packet `[B, frames * frame_width]`; the flow
    arguments are meaningless for a GR00T head and are accepted and ignored.

    The returned packet spans the head's full horizon. BB1 slices the leading
    frames the encoder needs (or ensembles the overlap), so a horizon-30 head
    feeds a 10-frame encoder without any change here.

    Per-environment goals: BB1's planner signature carries no environment ids,
    so `note_env_ids` must be called with the ids for the rows about to be
    predicted. BB1 calls `causal_state_provider(env_ids)` immediately before
    `planner(...)` in the same function, which is where the eval entrypoint
    records them. Without that call the adapter refuses to guess a goal.
    """

    def __init__(
        self,
        publisher: Gr00tChunkPublisher,
        *,
        components: tuple[tuple[str, int], ...] = ROOT_QPOS_COMPONENT_WIDTHS,
    ) -> None:
        super().__init__()
        self._publisher = publisher
        self._components = tuple(components)
        # `install_packet_encoder_command_source` reads a device off the
        # planner's parameters; register one so it resolves without exposing
        # the head's weights as trainable.
        self.register_parameter(
            "_device_anchor",
            torch.nn.Parameter(
                torch.zeros(1, device=publisher._gr00t_device), requires_grad=False
            ),
        )
        self._pending_env_ids: Tensor | None = None

    def note_env_ids(self, env_ids: Tensor) -> None:
        """Record which environments the next `forward` call predicts for."""
        self._pending_env_ids = torch.as_tensor(env_ids, dtype=torch.long).reshape(-1)

    def forward(
        self,
        causal_state: Tensor,
        *,
        num_inference_steps: int = 0,
        inference_noise_std: float = 0.0,
    ) -> Tensor:
        del num_inference_steps, inference_noise_std
        if self._pending_env_ids is None:
            msg = (
                "no environment ids recorded for this packet; the eval "
                "entrypoint must call note_env_ids in its causal-state provider."
            )
            raise RuntimeError(msg)
        env_ids = self._pending_env_ids.to(self._publisher._gr00t_device)
        self._pending_env_ids = None
        if int(env_ids.numel()) != int(causal_state.shape[0]):
            msg = (
                f"recorded {int(env_ids.numel())} env ids but the packet batch "
                f"has {int(causal_state.shape[0])} rows."
            )
            raise ValueError(msg)
        prediction = self._publisher._gr00t_predict(
            causal_state, self._publisher._gr00t_goal_index[env_ids]
        )
        rows, frames, _ = prediction.shape
        blocks: list[Tensor] = []
        cursor = 0
        for _, width in self._components:
            block = prediction[:, :, cursor : cursor + width]
            blocks.append(block.reshape(rows, frames * width))
            cursor += width
        return torch.cat(blocks, dim=-1).to(
            device=causal_state.device, dtype=causal_state.dtype
        )

    def report(self) -> dict[str, Any]:
        """Provenance counter block merged into the summary's gr00t_planner."""
        return {"packet_execution": "per_publication"}


__all__ = [
    "Gr00tChunkPublisher",
    "Gr00tPacketPlanner",
    "ROOT_QPOS_COMPONENT_WIDTHS",
]
