"""Asynchronous and cursor-based execution for explicit chunk heads (phase D1).

The latent async sampler (:mod:`gr00t_async_sampler`) consumes a chunk of
latents slot by slot, so a late service reply is absorbed by sliding deeper
into the previous chunk. The explicit routes had no such machinery: the
native route republished a fresh packet at every renewal, and the encoded
route re-ran the head at every publication and consumed only the packet's
leading encoder window. This module gives both routes the same receding
horizon consumption, in a sync and an async variant, so the explicit rows
are cadence-matched with the latent rows they are compared against.

Time and frame conventions (from the collection join,
``eval_skill_commander_closed_loop._save_planner_samples`` and
``prepare_gr00t_dataset``): a head prediction made from the causal state at
control step ``t`` has frames ``k = 0..H-1`` meaning absolute step ``t + k``,
every frame expressed in the robot's anchor frame AT ``t``. Therefore:

- consuming a packet ``a`` steps after its request must start at frame ``a``
  (``align_packet_frames``), and
- its position / rot6d blocks must be rigidly re-expressed from the
  request-time anchor into the current anchor (``reexpress_root_qpos_frames``)
  before an encoder that expects current-frame input sees them. The native
  chunk term does that re-expression itself, so the native route instead
  pins the request-time anchor on the term
  (``ChunkActorCommand.pin_anchor_pose``).

Reply lifecycle mirrors the latent async sampler: a reply is STAGED and only
swapped in at the boundary that needs it (never mid-window), a boundary with
no staged reply slides into the live packet's tail (a deadline miss,
counted), the tail exhausts into holding the last frame/window, and only an
environment's first packet after a reset blocks (``startup_syncs``). Reset
detection is local: an environment whose ``episode_length_buf`` moved
backwards has reset, which bumps its epoch so stale replies are dropped.

Quaternions are wxyz, matching Isaac Lab and
``ImitationRLEnv._get_robot_anchor_state_w_fast``.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor

from imitation_experiments.planner.gr00t_async_sampler import _ServiceLink
from imitation_experiments.planner.gr00t_chunk_publisher import (
    ROOT_QPOS_COMPONENT_WIDTHS,
)
from imitation_experiments.planner.gr00t_service_protocol import STATE_VALUES


# ---------------------------------------------------------------------------
# Pure-torch frame math (wxyz quaternions), so the default test environment
# needs no Isaac Lab import.
# ---------------------------------------------------------------------------


def quat_inv(q: Tensor) -> Tensor:
    """Inverse of unit quaternions ``[..., 4]`` (wxyz)."""
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_mul(a: Tensor, b: Tensor) -> Tensor:
    """Hamilton product of wxyz quaternions ``[..., 4]``."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def quat_apply(q: Tensor, v: Tensor) -> Tensor:
    """Rotate vectors ``[..., 3]`` by wxyz quaternions ``[..., 4]``."""
    qvec = q[..., 1:]
    t = 2.0 * torch.linalg.cross(qvec, v, dim=-1)
    return v + q[..., :1] * t + torch.linalg.cross(qvec, t, dim=-1)


def quat_apply_inverse(q: Tensor, v: Tensor) -> Tensor:
    return quat_apply(quat_inv(q), v)


def matrix_from_quat(q: Tensor) -> Tensor:
    """Rotation matrices ``[..., 3, 3]`` from wxyz quaternions."""
    w, x, y, z = q.unbind(-1)
    row0 = torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
        ],
        dim=-1,
    )
    row1 = torch.stack(
        [
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
        ],
        dim=-1,
    )
    row2 = torch.stack(
        [
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    )
    return torch.stack([row0, row1, row2], dim=-2)


def align_packet_frames(frames: Tensor, elapsed: int) -> Tensor:
    """Shift ``[H, W]`` frames forward by ``elapsed`` steps, repeating the tail.

    Frame ``k`` of the result is frame ``min(k + elapsed, H - 1)`` of the
    input, so frame 0 lines up with the current control step and a packet
    consumed past its end holds its final frame — the same "hold, never
    fabricate" rule the latent async sampler applies to its last slot.
    """
    horizon = int(frames.shape[0])
    if elapsed <= 0:
        return frames
    index = (
        torch.arange(horizon, device=frames.device, dtype=torch.long) + int(elapsed)
    ).clamp_(max=horizon - 1)
    return frames.index_select(0, index)


def reexpress_root_qpos_frames(
    frames: Tensor,
    *,
    request_anchor: tuple[Tensor, Tensor],
    current_anchor: tuple[Tensor, Tensor],
    components: tuple[tuple[str, int], ...] = ROOT_QPOS_COMPONENT_WIDTHS,
) -> Tensor:
    """Re-express ``[H, W]`` frames from the request anchor into the current one.

    Same rigid transform as
    ``ChunkActorCommand._reexpress_in_current_anchor_frame``: ``*_pos``
    blocks rotate and translate, ``*_ori`` rot6d blocks rotate, joint blocks
    pass through.
    """
    request_pos, request_quat = request_anchor
    current_pos, current_quat = current_anchor
    delta_quat = quat_mul(quat_inv(current_quat), request_quat)
    delta_pos = quat_apply_inverse(current_quat, request_pos - current_pos)
    horizon = int(frames.shape[0])
    out: list[Tensor] = []
    cursor = 0
    for name, width in components:
        block = frames[:, cursor : cursor + width]
        cursor += width
        if name.endswith("_pos"):
            vectors = block.reshape(-1, 3)
            rotated = quat_apply(delta_quat.expand(vectors.shape[0], 4), vectors)
            block = (rotated + delta_pos.reshape(1, 3)).reshape(horizon, width)
        elif name.endswith("_ori"):
            columns = block.reshape(horizon, -1, 3, 2)
            rotated = torch.matmul(matrix_from_quat(delta_quat)[None, None], columns)
            block = rotated.reshape(horizon, width)
        out.append(block)
    return torch.cat(out, dim=-1)


def frames_to_term_major_rows(
    frames: Tensor, components: tuple[tuple[str, int], ...]
) -> Tensor:
    """``[B, H, W]`` frame-major to ``[B, H * W]`` term-major (packet layout)."""
    rows, horizon, _ = frames.shape
    blocks: list[Tensor] = []
    cursor = 0
    for _, width in components:
        blocks.append(
            frames[:, :, cursor : cursor + width].reshape(rows, horizon * width)
        )
        cursor += width
    return torch.cat(blocks, dim=-1)


# ---------------------------------------------------------------------------
# Shared per-environment packet bookkeeping
# ---------------------------------------------------------------------------


class _PacketStore:
    """Per-env live packet, staged reply, and epoch fencing for chunk routes.

    ``live`` is what consumption reads; ``staged`` is a delivered reply
    waiting for the boundary that swaps it in. Requests carry the episode
    step and anchor pose of their instant, because the reply's frames are
    expressed in that frame.
    """

    def __init__(self, num_envs: int, horizon: int, width: int, device: Any) -> None:
        self.num_envs = int(num_envs)
        self.horizon = int(horizon)
        self.packet = torch.zeros(num_envs, horizon, width, device=device)
        self.have = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.request_step = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.anchor_pos = torch.zeros(num_envs, 3, device=device)
        self.anchor_quat = torch.zeros(num_envs, 4, device=device)
        self.staged_packet = torch.zeros(num_envs, horizon, width, device=device)
        self.staged = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.staged_request_step = torch.zeros(
            num_envs, dtype=torch.long, device=device
        )
        self.staged_anchor_pos = torch.zeros(num_envs, 3, device=device)
        self.staged_anchor_quat = torch.zeros(num_envs, 4, device=device)
        self.epoch = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.outstanding = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.pending_request_step = torch.zeros(
            num_envs, dtype=torch.long, device=device
        )
        self.pending_anchor_pos = torch.zeros(num_envs, 3, device=device)
        self.pending_anchor_quat = torch.zeros(num_envs, 4, device=device)
        self.last_episode_step = torch.full(
            (num_envs,), -1, dtype=torch.long, device=device
        )

    def note_resets(self, episode_steps: Tensor) -> None:
        """Bump epochs for envs whose episode counter moved backwards."""
        reset = episode_steps < self.last_episode_step
        if bool(reset.any()):
            ids = reset.nonzero(as_tuple=False).reshape(-1)
            self.epoch[ids] += 1
            self.have[ids] = False
            self.staged[ids] = False
            self.outstanding[ids] = False
        self.last_episode_step = episode_steps.clone()

    def stage_request(
        self, env: int, episode_step: int, anchor: tuple[Tensor, Tensor]
    ) -> None:
        self.outstanding[env] = True
        self.pending_request_step[env] = int(episode_step)
        self.pending_anchor_pos[env] = anchor[0]
        self.pending_anchor_quat[env] = anchor[1]

    def accept_reply(self, env: int, frames: Tensor) -> None:
        """Stage a delivered reply; it becomes live only at ``swap``."""
        self.staged_packet[env] = frames.to(
            self.staged_packet.device, self.staged_packet.dtype
        )
        self.staged_request_step[env] = self.pending_request_step[env]
        self.staged_anchor_pos[env] = self.pending_anchor_pos[env]
        self.staged_anchor_quat[env] = self.pending_anchor_quat[env]
        self.staged[env] = True
        self.outstanding[env] = False

    def swap(self, env: int) -> None:
        self.packet[env] = self.staged_packet[env]
        self.request_step[env] = self.staged_request_step[env]
        self.anchor_pos[env] = self.staged_anchor_pos[env]
        self.anchor_quat[env] = self.staged_anchor_quat[env]
        self.have[env] = True
        self.staged[env] = False

    def live_age(self, env: int, episode_step: int) -> int:
        return int(episode_step) - int(self.request_step[env])


class _AsyncChunkMixin:
    """Service link plus the shared request/drain/startup loop."""

    def _init_async(
        self,
        *,
        publisher: Any,  # Gr00tChunkPublisher or a duck-typed test stub
        store: _PacketStore,
        service_endpoint: str,
        lead_steps: int,
        anchor_state_fn: Callable[[], tuple[Tensor, Tensor]],
        service_timeout_s: float = 30.0,
    ) -> None:
        self._publisher = publisher
        self._store = store
        self._link: _ServiceLink | None = _ServiceLink(
            service_endpoint, num_envs=store.num_envs, timeout_s=service_timeout_s
        )
        self._endpoint = str(service_endpoint)
        self._lead = int(lead_steps)
        self._anchor_state_fn = anchor_state_fn
        self._timeout_s = float(service_timeout_s)
        self.deadline_misses = 0
        self.startup_syncs = 0

    def _causal_state_rows(self, env_ids: Tensor) -> Tensor:
        publisher = self._publisher
        batch = publisher._causal_observation_fn(
            env_ids=env_ids, history_steps=publisher._causal_history_steps
        )
        return (
            batch.get(("planner", "state_history"))
            .reshape(int(env_ids.numel()), -1)
            .to(device=publisher._gr00t_device, dtype=torch.float32)
        )

    def _send_requests(self, env_ids: Tensor, episode_steps: Tensor) -> None:
        if int(env_ids.numel()) == 0:
            return
        states = self._causal_state_rows(env_ids)
        anchor_pos, anchor_quat = self._anchor_state_fn()
        link = self._link
        assert link is not None
        for row, env in enumerate(env_ids.reshape(-1).tolist()):
            env = int(env)
            state = states[row].reshape(1, -1).detach().cpu().numpy()
            state = np.ascontiguousarray(state, dtype=np.float32)
            if state.shape[1] != STATE_VALUES:
                msg = (
                    f"causal state has {state.shape[1]} values, expected {STATE_VALUES}"
                )
                raise ValueError(msg)
            self._store.stage_request(
                env,
                int(episode_steps[env]),
                (anchor_pos[env].clone(), anchor_quat[env].clone()),
            )
            link.enqueue(
                env,
                int(self._store.epoch[env]),
                int(self._publisher._gr00t_goal_index[env]),
                state[0],
            )

    def _drain_replies(self) -> None:
        link = self._link
        assert link is not None
        store = self._store
        outstanding = store.outstanding.nonzero(as_tuple=False).reshape(-1)
        width = store.packet.shape[-1]
        for env in outstanding.tolist():
            chunk = link.take_reply(env, int(store.epoch[env]))
            if chunk is None:
                continue
            frames = torch.from_numpy(np.ascontiguousarray(chunk)).reshape(
                store.horizon, width
            )
            store.accept_reply(env, frames)

    def _startup_block(self, env: int, episode_steps: Tensor) -> None:
        store = self._store
        if not bool(store.outstanding[env]):
            self._send_requests(
                torch.tensor([env], dtype=torch.long, device=episode_steps.device),
                episode_steps,
            )
        link = self._link
        assert link is not None
        chunk = link.wait_reply(env, int(store.epoch[env]), timeout_s=self._timeout_s)
        if chunk is None:
            msg = (
                f"planner service produced no startup packet for env {env} "
                f"within {self._timeout_s}s"
            )
            raise RuntimeError(msg)
        frames = torch.from_numpy(np.ascontiguousarray(chunk)).reshape(
            store.horizon, store.packet.shape[-1]
        )
        store.accept_reply(env, frames)
        store.swap(env)
        self.startup_syncs += 1

    def _async_report(self) -> dict[str, Any]:
        link = self._link
        latency = link.latency_snapshot() if link is not None else []
        return {
            "planner_execution": "async_service",
            "service_endpoint": self._endpoint,
            "lead_steps": self._lead,
            "deadline_misses": int(self.deadline_misses),
            "startup_syncs": int(self.startup_syncs),
            "service_roundtrip_ms": (
                {
                    "count": len(latency),
                    "p50": float(np.quantile(latency, 0.5)),
                    "p95": float(np.quantile(latency, 0.95)),
                    "max": float(np.max(latency)),
                }
                if latency
                else None
            ),
        }

    def close(self) -> None:
        if self._link is not None:
            self._link.close()
            self._link = None


# ---------------------------------------------------------------------------
# Native route: async publisher for the chunk actor term
# ---------------------------------------------------------------------------


class Gr00tAsyncChunkPublisher(_AsyncChunkMixin):
    """Service-backed publisher for ``--gr00t_route chunk_native``.

    Wraps a term-less :class:`Gr00tChunkPublisher` for its goal binding and
    causal-state plumbing, but produces packets from the zmq service. The
    evaluator must call :meth:`note_step` once per control step (before the
    renewal handling) and :meth:`publish` with the renewal env ids, exactly
    where the sync publisher is called.

    A renewal with a staged reply swaps it in and publishes it aligned to
    its age. A renewal without one is a deadline miss: the LIVE packet is
    republished time-shifted (``align_packet_frames``), so the term consumes
    its tail instead of replaying its leading frames. Every publish re-pins
    the request-time anchor on the term because the packet's coordinates
    live in that frame, not in the publish-instant frame ``publish()``
    captures.

    Request scheduling is anchored on the term's renewal phase
    (``episode_length_buf % hold_steps``), not on the live packet's age, so
    the causal state a request carries is always ``lead_steps`` before the
    boundary its reply is meant for.
    """

    def __init__(
        self,
        *,
        publisher: Any,  # Gr00tChunkPublisher or a duck-typed test stub
        chunk_term: Any,
        hold_steps: int,
        service_endpoint: str,
        lead_steps: int,
        anchor_state_fn: Callable[[], tuple[Tensor, Tensor]],
        episode_step_fn: Callable[[], Tensor],
        num_envs: int,
        service_timeout_s: float = 30.0,
    ) -> None:
        if chunk_term is None:
            msg = "the native async route needs the chunk actor term."
            raise ValueError(msg)
        if not hasattr(chunk_term, "pin_anchor_pose"):
            msg = (
                "this ChunkActorCommand has no pin_anchor_pose; the async "
                "native route cannot re-pin the request-time anchor."
            )
            raise ValueError(msg)
        if not 0 < int(lead_steps) <= int(hold_steps):
            msg = f"lead_steps must be in 1..hold_steps, got {lead_steps}."
            raise ValueError(msg)
        self._chunk_term = chunk_term
        self._hold = int(hold_steps)
        self._episode_step_fn = episode_step_fn
        store = _PacketStore(
            num_envs,
            publisher._gr00t_horizon,
            publisher._gr00t_action_dim,
            publisher._gr00t_device,
        )
        self._init_async(
            publisher=publisher,
            store=store,
            service_endpoint=service_endpoint,
            lead_steps=lead_steps,
            anchor_state_fn=anchor_state_fn,
            service_timeout_s=service_timeout_s,
        )
        self.publications = 0
        self.provenance = dict(publisher.provenance)

    def gr00t_assert_goal_matches(self, env_ids: Tensor, names: list[str]) -> None:
        self._publisher.gr00t_assert_goal_matches(env_ids, names)

    def note_step(self) -> None:
        store = self._store
        episode_steps = self._episode_step_fn().to(store.epoch.device)
        store.note_resets(episode_steps)
        self._drain_replies()
        until_boundary = (self._hold - episode_steps.remainder(self._hold)).remainder(
            self._hold
        )
        due = (
            (until_boundary > 0)
            & (until_boundary <= self._lead)
            & store.have
            & ~store.outstanding
            & ~store.staged
        )
        if bool(due.any()):
            self._send_requests(due.nonzero(as_tuple=False).reshape(-1), episode_steps)

    def publish(self, env_ids: Tensor) -> None:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long).reshape(-1)
        if int(env_ids.numel()) == 0:
            return
        store = self._store
        episode_steps = self._episode_step_fn().to(store.epoch.device)
        store.note_resets(episode_steps)
        self._drain_replies()
        for env in env_ids.tolist():
            env = int(env)
            if store.staged[env]:
                store.swap(env)
            elif store.have[env]:
                self.deadline_misses += 1
            else:
                self._startup_block(env, episode_steps)
            self._publish_env(env, int(episode_steps[env]))

    def _publish_env(self, env: int, episode_step: int) -> None:
        store = self._store
        frames = align_packet_frames(
            store.packet[env], store.live_age(env, episode_step)
        )
        env_ids = torch.tensor([env], dtype=torch.long, device=frames.device)
        payload: dict[str, Tensor] = {}
        cursor = 0
        for name, width in self._publisher._components:
            payload[name] = (
                frames[:, cursor : cursor + width].reshape(1, -1).contiguous()
            )
            cursor += width
        self._chunk_term.publish(env_ids, payload)
        self._chunk_term.pin_anchor_pose(
            env_ids,
            store.anchor_pos[env].reshape(1, 3),
            store.anchor_quat[env].reshape(1, 4),
        )
        self.publications += 1

    def report(self) -> dict[str, Any]:
        record = dict(self.provenance)
        record.update(self._publisher.gr00t_stats())
        record["publications"] = int(self.publications)
        record.update(self._async_report())
        return record


# ---------------------------------------------------------------------------
# Encoded route: cursor packet planners (sync-matched and async)
# ---------------------------------------------------------------------------


class Gr00tCursorPacketPlanner(torch.nn.Module):
    """Receding-horizon adapter for ``--gr00t_route chunk_encoded``.

    Where :class:`Gr00tPacketPlanner` re-runs the head at every publication,
    this adapter re-plans every ``consume_frames`` control steps and serves
    the intermediate publications from the cached packet: the returned
    term-major packet leads with the window at the packet's current age,
    re-expressed into the current anchor, so
    ``install_packet_encoder_command_source`` slices exactly the frames that
    are time-aligned with this control step. That matches the latent hold-1
    arm's cadence (a 30-slot plan, re-planned every ``consume_frames``).

    The sync variant runs the head inline at each re-plan boundary; the
    async subclass replaces that call with the service link.
    """

    def __init__(
        self,
        publisher: Any,  # Gr00tChunkPublisher or a duck-typed test stub
        *,
        consume_frames: int,
        encoder_frames: int,
        anchor_state_fn: Callable[[], tuple[Tensor, Tensor]],
        episode_step_fn: Callable[[], Tensor],
        num_envs: int,
        components: tuple[tuple[str, int], ...] = ROOT_QPOS_COMPONENT_WIDTHS,
    ) -> None:
        super().__init__()
        self._publisher = publisher
        self._components = tuple(components)
        self._consume = int(consume_frames)
        self._encoder_frames = int(encoder_frames)
        self._anchor_state_fn = anchor_state_fn
        self._episode_step_fn = episode_step_fn
        horizon = publisher._gr00t_horizon
        if self._consume > horizon - self._encoder_frames + 1:
            msg = (
                f"consume_frames {self._consume} leaves no full "
                f"{self._encoder_frames}-frame window inside horizon {horizon}."
            )
            raise ValueError(msg)
        self._max_offset = horizon - self._encoder_frames
        self._store = _PacketStore(
            num_envs, horizon, publisher._gr00t_action_dim, publisher._gr00t_device
        )
        self.register_parameter(
            "_device_anchor",
            torch.nn.Parameter(
                torch.zeros(1, device=publisher._gr00t_device), requires_grad=False
            ),
        )
        self._pending_env_ids: Tensor | None = None
        self.replans = 0

    def note_env_ids(self, env_ids: Tensor) -> None:
        self._pending_env_ids = torch.as_tensor(env_ids, dtype=torch.long).reshape(-1)

    # -- packet refresh, overridden by the async subclass ------------------- #

    def _refresh(
        self, env_ids: Tensor, causal_state: Tensor, episode_steps: Tensor
    ) -> None:
        """Sync re-plan: run the head inline for envs at their due boundary."""
        store = self._store
        due_rows = [
            row
            for row, env in enumerate(env_ids.tolist())
            if not bool(store.have[int(env)])
            or store.live_age(int(env), int(episode_steps[int(env)])) >= self._consume
        ]
        if not due_rows:
            return
        rows = torch.tensor(due_rows, dtype=torch.long, device=env_ids.device)
        due_envs = env_ids.index_select(0, rows)
        prediction = self._publisher._gr00t_predict(
            causal_state.index_select(0, rows.to(causal_state.device)).to(
                device=self._publisher._gr00t_device, dtype=torch.float32
            ),
            self._publisher._gr00t_goal_index[due_envs],
        )
        anchor_pos, anchor_quat = self._anchor_state_fn()
        for row, env in enumerate(due_envs.tolist()):
            env = int(env)
            store.stage_request(
                env,
                int(episode_steps[env]),
                (anchor_pos[env].clone(), anchor_quat[env].clone()),
            )
            store.accept_reply(env, prediction[row])
            store.swap(env)
            self.replans += 1

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
        env_ids = self._pending_env_ids.to(self._store.epoch.device)
        self._pending_env_ids = None
        if int(env_ids.numel()) != int(causal_state.shape[0]):
            msg = (
                f"recorded {int(env_ids.numel())} env ids but the packet batch "
                f"has {int(causal_state.shape[0])} rows."
            )
            raise ValueError(msg)
        store = self._store
        episode_steps = self._episode_step_fn().to(store.epoch.device)
        store.note_resets(episode_steps)
        self._refresh(env_ids, causal_state, episode_steps)
        anchor_pos, anchor_quat = self._anchor_state_fn()
        rows: list[Tensor] = []
        for env in env_ids.tolist():
            env = int(env)
            offset = store.live_age(env, int(episode_steps[env]))
            offset = max(0, min(offset, self._max_offset))
            frames = align_packet_frames(store.packet[env], offset)
            frames = reexpress_root_qpos_frames(
                frames,
                request_anchor=(store.anchor_pos[env], store.anchor_quat[env]),
                current_anchor=(anchor_pos[env], anchor_quat[env]),
                components=self._components,
            )
            rows.append(frames)
        packet = torch.stack(rows, dim=0)
        return frames_to_term_major_rows(packet, self._components).to(
            device=causal_state.device, dtype=causal_state.dtype
        )

    def report(self) -> dict[str, Any]:
        return {
            "packet_execution": "cursor",
            "consume_frames": int(self._consume),
            "replans": int(self.replans),
        }


class Gr00tAsyncPacketPlanner(Gr00tCursorPacketPlanner, _AsyncChunkMixin):
    """Async encoded route: the cursor planner fed by the zmq service.

    Requests are issued ``lead_steps`` before the due boundary (age-based:
    cursor boundaries are anchored on the live packet's request step). A
    boundary reached with no staged reply is a deadline miss, counted once
    per ``consume_frames``, and consumption slides deeper into the live
    packet — the horizon past ``consume_frames`` is the latency tail —
    holding the final window once the tail is spent. A staged reply swaps in
    at the first refresh at or past the boundary, time-aligned by its age.
    """

    def __init__(
        self,
        publisher: Any,  # Gr00tChunkPublisher or a duck-typed test stub
        *,
        service_endpoint: str,
        lead_steps: int,
        consume_frames: int,
        encoder_frames: int,
        anchor_state_fn: Callable[[], tuple[Tensor, Tensor]],
        episode_step_fn: Callable[[], Tensor],
        num_envs: int,
        service_timeout_s: float = 30.0,
        components: tuple[tuple[str, int], ...] = ROOT_QPOS_COMPONENT_WIDTHS,
    ) -> None:
        if not 0 < int(lead_steps) <= int(consume_frames):
            msg = f"lead_steps must be in 1..consume_frames, got {lead_steps}."
            raise ValueError(msg)
        Gr00tCursorPacketPlanner.__init__(
            self,
            publisher,
            consume_frames=consume_frames,
            encoder_frames=encoder_frames,
            anchor_state_fn=anchor_state_fn,
            episode_step_fn=episode_step_fn,
            num_envs=num_envs,
            components=components,
        )
        self._init_async(
            publisher=publisher,
            store=self._store,
            service_endpoint=service_endpoint,
            lead_steps=lead_steps,
            anchor_state_fn=anchor_state_fn,
            service_timeout_s=service_timeout_s,
        )

    def _refresh(
        self, env_ids: Tensor, causal_state: Tensor, episode_steps: Tensor
    ) -> None:
        del causal_state
        self._drain_replies()
        store = self._store
        for env in env_ids.tolist():
            env = int(env)
            if not store.have[env]:
                self._startup_block(env, episode_steps)
                continue
            age = store.live_age(env, int(episode_steps[env]))
            if age >= self._consume:
                if store.staged[env]:
                    store.swap(env)
                elif age % self._consume == 0:
                    self.deadline_misses += 1
        due = (
            store.have
            & ~store.outstanding
            & ~store.staged
            & (episode_steps - store.request_step >= (self._consume - self._lead))
        )
        if bool(due.any()):
            self._send_requests(due.nonzero(as_tuple=False).reshape(-1), episode_steps)

    def report(self) -> dict[str, Any]:
        record = Gr00tCursorPacketPlanner.report(self)
        record.update(self._async_report())
        record["packet_execution"] = "cursor_async"
        return record


__all__ = [
    "Gr00tAsyncChunkPublisher",
    "Gr00tAsyncPacketPlanner",
    "Gr00tCursorPacketPlanner",
    "align_packet_frames",
    "frames_to_term_major_rows",
    "matrix_from_quat",
    "quat_apply",
    "quat_apply_inverse",
    "quat_inv",
    "quat_mul",
    "reexpress_root_qpos_frames",
]
