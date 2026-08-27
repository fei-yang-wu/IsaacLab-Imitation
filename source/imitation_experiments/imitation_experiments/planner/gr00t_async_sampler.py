"""Asynchronous service-backed GR00T latent sampler (phase D1).

`Gr00tAsyncLatentCommandSampler` keeps every contract of the synchronous
`Gr00tLatentCommandSampler` — hold countdown, per-environment renewal, FSQ
snap, goal binding — but produces chunks by talking to the batched zmq
service (:mod:`gr00t_batch_service`) instead of running the head inline.

Semantics, from `wiki/gr00t-planner-deployment.md` (request loop):

- **Lead-time request.** For environment `i`, when the number of control
  steps until its next head-needed renewal equals ``lead_steps``, its causal
  state is enqueued. Requests accumulated during one service forward form
  the next batch — batching emerges from service latency.
- **Swap at expiry, time-aligned.** A reply is staged per environment and
  swapped in at the renewal that needs it. The training join puts slot ``k``
  at ``request_step + hold * k``, so the cursor starts at
  ``floor(elapsed_steps / hold)`` — on time that is slot ``lead // hold``
  (0 for a within-hold lead, exactly the elapsed publications at hold 1),
  late replies skip further. Never replay the originally planned frames.
- **Deadline miss: hold, never block, never fabricate.** A needing renewal
  with no staged reply consumes the previous chunk's next unconsumed slot;
  when the tail is exhausted it re-publishes the last slot. Every such
  renewal counts as a miss.
- **Startup/reset is the one blocking point.** An environment with no live
  chunk (first publication of an episode) blocks on its reply — a deployed
  robot also waits for its first command. Counted separately
  (``startup_syncs``), never mixed into the miss statistic.

Not supported in async mode (fail loudly, do not degrade): temporal
ensembling, ``fresh`` consumption, ``samples_per_publication > 1``. Their
sync semantics presume an inline head call per publication.

An async row is labelled ``planner_execution: async_service`` in its
provenance and is never poolable with a sync row (user gate, 2026-08-18:
the D0 sync companion is reported next to it; no numeric equivalence bound).
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor

from imitation_experiments.planner.gr00t_latent_sampler import (
    Gr00tLatentCommandSampler,
)
from imitation_experiments.planner.gr00t_service_protocol import (
    STATE_VALUES,
    decode_chunks_reply,
    encode_act_request,
)


class _ServiceLink:
    """Background REQ/REP worker: batches pending requests, deposits replies.

    One outstanding service call at a time (REQ/REP lockstep); everything
    enqueued while a forward is in flight becomes the next batch. Replies are
    deposited keyed by environment with the epoch they belong to, so a reply
    that raced a reset is identifiable and droppable.
    """

    def __init__(self, endpoint: str, *, num_envs: int, timeout_s: float) -> None:
        import zmq

        self._zmq = zmq
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, int(timeout_s * 1000))
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(endpoint)
        self._num_envs = int(num_envs)
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._pending: dict[int, tuple[int, int, np.ndarray]] = {}
        self._replies: dict[int, tuple[int, np.ndarray]] = {}
        self._latency_ms: list[float] = []
        self._sent_at: dict[int, float] = {}
        self._fault: BaseException | None = None
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def enqueue(self, env_id: int, epoch: int, goal_id: int, state: np.ndarray) -> None:
        with self._lock:
            # A newer request for the same env supersedes the old one.
            self._pending[env_id] = (epoch, goal_id, state)
        self._wake.set()

    def take_reply(self, env_id: int, epoch: int) -> np.ndarray | None:
        with self._lock:
            if self._fault is not None:
                raise RuntimeError("planner service link fault") from self._fault
            entry = self._replies.get(env_id)
            if entry is None or entry[0] != epoch:
                return None
            del self._replies[env_id]
            return entry[1]

    def wait_reply(
        self, env_id: int, epoch: int, *, timeout_s: float
    ) -> np.ndarray | None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            chunk = self.take_reply(env_id, epoch)
            if chunk is not None:
                return chunk
            time.sleep(0.001)
        return None

    def latency_snapshot(self) -> list[float]:
        with self._lock:
            return list(self._latency_ms)

    def close(self) -> None:
        self._stop = True
        self._wake.set()
        self._thread.join(timeout=5.0)
        self._socket.close(0)
        self._context.term()

    def _run(self) -> None:
        try:
            while not self._stop:
                self._wake.wait(timeout=0.05)
                self._wake.clear()
                with self._lock:
                    batch = dict(self._pending)
                    self._pending.clear()
                if not batch:
                    continue
                env_ids = sorted(batch)
                states = np.stack([batch[e][2] for e in env_ids])
                goal_ids = np.asarray([batch[e][1] for e in env_ids], dtype=np.int64)
                epochs = {e: batch[e][0] for e in env_ids}
                request_ids = np.asarray(
                    [epochs[e] * self._num_envs + e for e in env_ids], dtype=np.int64
                )
                sent = time.monotonic()
                self._socket.send_multipart(
                    encode_act_request(states, goal_ids, request_ids)
                )
                chunks, echoed, _head_ms = decode_chunks_reply(
                    self._socket.recv_multipart()
                )
                elapsed_ms = (time.monotonic() - sent) * 1000.0
                with self._lock:
                    for row, request_id in enumerate(echoed.tolist()):
                        env = int(request_id) % self._num_envs
                        epoch = int(request_id) // self._num_envs
                        self._replies[env] = (epoch, chunks[row])
                        self._latency_ms.append(elapsed_ms)
        except BaseException as exc:  # noqa: BLE001 - surfaced on the main thread
            with self._lock:
                self._fault = exc


class Gr00tAsyncLatentCommandSampler(Gr00tLatentCommandSampler):
    def __init__(
        self,
        *,
        service_endpoint: str,
        lead_steps: int = 5,
        service_timeout_s: float = 30.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if getattr(self, "_gr00t_ensemble", "none") != "none":
            msg = "async mode does not support temporal ensembling"
            raise ValueError(msg)
        if self._gr00t_consumption != "open_loop":
            msg = "async mode requires open_loop consumption"
            raise ValueError(msg)
        if getattr(self, "_gr00t_samples", 1) != 1:
            msg = "async mode requires samples_per_publication=1"
            raise ValueError(msg)
        if self.latent_steps_min != self.latent_steps_max:
            msg = "async lead-time scheduling requires a fixed hold"
            raise ValueError(msg)
        num_envs = int(self._gr00t_cursor.shape[0])
        self._async_link = _ServiceLink(
            service_endpoint, num_envs=num_envs, timeout_s=service_timeout_s
        )
        self._async_endpoint = str(service_endpoint)
        self._async_lead = int(lead_steps)
        self._async_timeout_s = float(service_timeout_s)
        self._async_hold = int(self.latent_steps_min)
        device = self._gr00t_device
        self._async_step = 0
        self._async_epoch = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._async_have_chunk = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._async_outstanding = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._async_request_step = torch.full(
            (num_envs,), -1, dtype=torch.long, device=device
        )
        self._async_ready = torch.zeros_like(self._gr00t_cache)
        self._async_ready_ok = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._async_ready_from = torch.zeros(num_envs, dtype=torch.long, device=device)
        # The cursor value at which the NEXT head chunk is needed; varies per
        # env because a time-aligned swap starts mid-chunk.
        self._async_exhaust_at = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.deadline_misses = 0
        self.held_republished = 0
        self.startup_syncs = 0

    # ------------------------------------------------------------------ #
    # Per-step scheduling
    # ------------------------------------------------------------------ #

    def sample_for_step(
        self, td: Any, *, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        self._async_step += 1
        self._drain_replies()
        self._schedule_requests()
        return super().sample_for_step(td, device=device, dtype=dtype)

    def _drain_replies(self) -> None:
        outstanding = self._async_outstanding.nonzero(as_tuple=False).reshape(-1)
        for env in outstanding.tolist():
            chunk = self._async_link.take_reply(env, int(self._async_epoch[env]))
            if chunk is None:
                continue
            self._async_ready[env] = torch.from_numpy(np.ascontiguousarray(chunk)).to(
                self._async_ready.device, self._async_ready.dtype
            )
            self._async_ready_ok[env] = True
            self._async_ready_from[env] = self._async_request_step[env]
            self._async_outstanding[env] = False

    def _steps_until_head_needed(self) -> Tensor:
        remaining_pubs = (self._async_exhaust_at - self._gr00t_cursor).clamp(min=0)
        countdown = (
            self._latent_steps.to(self._gr00t_device)
            if self._latent_steps is not None
            else torch.zeros_like(remaining_pubs)
        )
        return countdown.clamp(min=0) + remaining_pubs * self._async_hold

    def _schedule_requests(self) -> None:
        if self._latent_steps is None:
            return  # nothing sampled yet; startup path handles the first call
        due = (
            (self._steps_until_head_needed() <= self._async_lead)
            & ~self._async_outstanding
            & ~self._async_ready_ok
            & self._async_have_chunk  # startup requests are issued at need
        )
        if not bool(due.any()):
            return
        env_ids = due.nonzero(as_tuple=False).reshape(-1)
        self._issue_requests(env_ids)

    def _issue_requests(self, env_ids: Tensor) -> None:
        states = self._causal_planner_state(env_ids)
        for row, env in enumerate(env_ids.reshape(-1).tolist()):
            self._send_one(int(env), states[row : row + 1])

    # ------------------------------------------------------------------ #
    # Chunk production (replaces the inline head call)
    # ------------------------------------------------------------------ #

    def gr00t_z(self, planner_state: Tensor, env_ids: Tensor) -> Tensor:
        env_ids = env_ids.to(self._gr00t_device).reshape(-1)
        needs = self._gr00t_cursor[env_ids] >= self._async_exhaust_at[env_ids]
        for row in needs.nonzero(as_tuple=False).reshape(-1).tolist():
            env = int(env_ids[row])
            if self._async_ready_ok[env]:
                self._swap_in(env)
                continue
            if self._async_have_chunk[env]:
                horizon = self._gr00t_horizon
                if int(self._gr00t_cursor[env]) < horizon:
                    self._async_exhaust_at[env] = int(self._gr00t_cursor[env]) + 1
                else:
                    self._gr00t_cursor[env] = horizon - 1
                    self._async_exhaust_at[env] = horizon
                    self.held_republished += 1
                self.deadline_misses += 1
                continue
            self._startup_sync(env, planner_state[row : row + 1])
        cursor = self._gr00t_cursor[env_ids]
        z = self._gr00t_cache[env_ids, cursor]
        self._gr00t_cursor[env_ids] = cursor + 1
        if self._gr00t_fsq_half is not None:
            half = self._gr00t_fsq_half
            z = torch.clamp(torch.round(z * half), -half, half - 1.0) / half
        return z.to(dtype=torch.float32)

    def _swap_in(self, env: int) -> None:
        elapsed = max(0, self._async_step - int(self._async_ready_from[env]))
        skip = min(elapsed // self._async_hold, self._gr00t_horizon - 1)
        self._gr00t_cache[env] = self._async_ready[env]
        self._gr00t_cursor[env] = skip
        self._async_exhaust_at[env] = min(skip + self._gr00t_slots, self._gr00t_horizon)
        self._async_have_chunk[env] = True
        self._async_ready_ok[env] = False

    def _startup_sync(self, env: int, state_row: Tensor) -> None:
        epoch = int(self._async_epoch[env])
        if not bool(self._async_outstanding[env]):
            self._send_one(env, state_row)
        chunk = self._async_link.wait_reply(env, epoch, timeout_s=self._async_timeout_s)
        if chunk is None:
            msg = (
                f"planner service produced no startup chunk for env {env} "
                f"within {self._async_timeout_s}s"
            )
            raise RuntimeError(msg)
        self._async_ready[env] = torch.from_numpy(np.ascontiguousarray(chunk)).to(
            self._async_ready.device, self._async_ready.dtype
        )
        self._async_ready_ok[env] = True
        self._async_ready_from[env] = self._async_request_step[env]
        self._async_outstanding[env] = False
        self.startup_syncs += 1
        self._swap_in(env)

    def _send_one(self, env: int, state_row: Tensor) -> None:
        state = state_row.reshape(1, -1).detach().cpu().numpy().astype(np.float32)
        if state.shape[1] != STATE_VALUES:
            msg = f"causal state has {state.shape[1]} values, expected {STATE_VALUES}"
            raise ValueError(msg)
        self._async_link.enqueue(
            env,
            int(self._async_epoch[env]),
            int(self._gr00t_goal_index[env]),
            state[0],
        )
        self._async_outstanding[env] = True
        self._async_request_step[env] = self._async_step

    # ------------------------------------------------------------------ #
    # Resets and reporting
    # ------------------------------------------------------------------ #

    def gr00t_reset(self, env_ids: Tensor | None = None) -> None:
        super().gr00t_reset(env_ids)
        if env_ids is None:
            selected = torch.arange(
                self._async_epoch.shape[0], device=self._gr00t_device
            )
        else:
            selected = env_ids.to(self._gr00t_device).reshape(-1)
        self._async_epoch[selected] += 1
        self._async_have_chunk[selected] = False
        self._async_ready_ok[selected] = False
        self._async_outstanding[selected] = False
        self._async_exhaust_at[selected] = 0

    def gr00t_report(self) -> dict[str, Any]:
        record = super().gr00t_report()
        latency = self._async_link.latency_snapshot()
        record.update(
            {
                "planner_execution": "async_service",
                "service_endpoint": self._async_endpoint,
                "lead_steps": self._async_lead,
                "deadline_misses": int(self.deadline_misses),
                "held_republished": int(self.held_republished),
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
        )
        return record

    def close(self) -> None:
        self._async_link.close()


__all__ = ["Gr00tAsyncLatentCommandSampler"]
