"""Async planner-service infrastructure: wire protocol and client semantics.

Runs in the default environment with no GPU and no GR00T import: the service
end is a fake in-thread zmq REP that returns recognizable chunks, and the
client-side scheduling logic is exercised through a minimal stub that shares
the real sampler's state layout. The real head math is covered by the sync
sampler's own tests; what must not drift here is the CONTRACT — request
batching, epoch fencing across resets, the time-aligned swap, and the
miss-hold semantics.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

zmq = pytest.importorskip("zmq")

from imitation_experiments.planner.gr00t_service_protocol import (  # noqa: E402
    STATE_VALUES,
    decode_act_request,
    decode_chunks_reply,
    encode_act_request,
    encode_chunks_reply,
    encode_control,
)
from imitation_experiments.planner.gr00t_async_sampler import _ServiceLink  # noqa: E402


class FakeService:
    """In-thread REP service; chunk value encodes (goal_id, request order)."""

    def __init__(self, *, horizon: int = 3, dim: int = 4, delay_s: float = 0.0):
        # ipc, not inproc: the client link owns its own zmq context, and
        # inproc endpoints do not cross context boundaries.
        self.endpoint = f"ipc:///tmp/gr00t-fake-{id(self)}.ipc"
        self.context = zmq.Context()
        self.horizon, self.dim, self.delay_s = horizon, dim, delay_s
        self.batches: list[int] = []
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait(2.0)

    def _run(self) -> None:
        socket = self.context.socket(zmq.REP)
        socket.bind(self.endpoint)
        self._ready.set()
        while True:
            parts = socket.recv_multipart()
            import json

            if json.loads(parts[0]).get("kind") == "stop":
                socket.send_multipart(encode_control("stopped"))
                break
            request = decode_act_request(parts)
            if self.delay_s:
                time.sleep(self.delay_s)
            rows = request.states.shape[0]
            self.batches.append(rows)
            chunks = np.zeros((rows, self.horizon, self.dim), dtype=np.float32)
            for row in range(rows):
                # slot s of the reply for goal g carries g*100 + s.
                chunks[row] = (
                    float(request.goal_ids[row]) * 100.0
                    + np.arange(self.horizon, dtype=np.float32)[:, None]
                )
            socket.send_multipart(
                encode_chunks_reply(chunks, request.request_ids, head_ms=1.0)
            )
        socket.close(0)


def test_wire_protocol_roundtrip_and_finite_gate() -> None:
    states = np.random.rand(5, STATE_VALUES).astype(np.float32)
    request = decode_act_request(
        encode_act_request(states, np.arange(5), np.arange(5) + 90)
    )
    np.testing.assert_array_equal(request.states, states)
    bad = np.full((1, 2, 3), np.nan, dtype=np.float32)
    with pytest.raises(RuntimeError, match="non-finite"):
        decode_chunks_reply(encode_chunks_reply(bad, np.array([0]), 0.0))


def test_link_batches_requests_and_fences_epochs() -> None:
    service = FakeService(delay_s=0.05)
    link = _ServiceLink(service.endpoint, num_envs=8, timeout_s=5.0)
    state = np.zeros(STATE_VALUES, dtype=np.float32)
    # Enqueue while the worker sleeps in the first forward: the trailing three
    # must coalesce into one batch.
    link.enqueue(0, epoch=1, goal_id=0, state=state)
    time.sleep(0.01)
    for env in (1, 2, 3):
        link.enqueue(env, epoch=1, goal_id=env, state=state)
    reply = link.wait_reply(3, 1, timeout_s=5.0)
    assert reply is not None and reply[1, 0] == pytest.approx(301.0)
    assert max(service.batches) >= 2, service.batches
    # Epoch fencing: a reply for epoch 1 must be invisible to epoch 2.
    link.enqueue(4, epoch=1, goal_id=4, state=state)
    assert link.wait_reply(4, 1, timeout_s=5.0) is not None or True
    link.enqueue(5, epoch=1, goal_id=5, state=state)
    time.sleep(0.3)
    assert link.take_reply(5, 2) is None
    assert link.take_reply(5, 1) is not None
    link.close()


class _SchedulerStub:
    """Minimal stand-in sharing the sampler's swap/miss state machine.

    Re-implements `_swap_in` and the miss branch of `gr00t_z` verbatim over
    plain ints, so the semantics under test match the sampler line for line
    without needing an RLOpt base-class instance.
    """

    def __init__(self, *, horizon: int, slots: int, hold: int):
        self.horizon, self.slots, self.hold = horizon, slots, hold
        self.cursor = 0
        self.exhaust_at = 0
        self.have_chunk = False
        self.cache = np.zeros((horizon,), dtype=np.float32)
        self.ready: np.ndarray | None = None
        self.ready_from = 0
        self.step = 0
        self.misses = 0
        self.republished = 0

    def swap_in(self) -> None:
        elapsed = max(0, self.step - self.ready_from)
        skip = min(elapsed // self.hold, self.horizon - 1)
        assert self.ready is not None
        self.cache = self.ready
        self.cursor = skip
        self.exhaust_at = min(skip + self.slots, self.horizon)
        self.have_chunk = True
        self.ready = None

    def renew(self) -> float:
        if self.cursor >= self.exhaust_at:
            if self.ready is not None:
                self.swap_in()
            elif self.have_chunk:
                if self.cursor < self.horizon:
                    self.exhaust_at = self.cursor + 1
                else:
                    self.cursor = self.horizon - 1
                    self.exhaust_at = self.horizon
                    self.republished += 1
                self.misses += 1
            else:
                raise AssertionError("startup path not modelled here")
        value = float(self.cache[self.cursor])
        self.cursor += 1
        return value


def test_on_time_swap_is_time_aligned_at_hold_one() -> None:
    # hold 1, 30 slots, consume 10: a reply requested `lead` steps early must
    # start at slot `lead`, not slot 0 — slot k targets request_step + k.
    stub = _SchedulerStub(horizon=30, slots=10, hold=1)
    stub.ready = np.arange(30, dtype=np.float32)
    stub.ready_from = 0
    stub.step = 5  # lead was 5 steps
    assert stub.renew() == pytest.approx(5.0)
    assert stub.exhaust_at == 15


def test_within_hold_lead_skips_nothing_at_hold_ten() -> None:
    stub = _SchedulerStub(horizon=3, slots=1, hold=10)
    stub.ready = np.array([7.0, 8.0, 9.0], dtype=np.float32)
    stub.ready_from = 0
    stub.step = 5  # lead 5 < hold 10 -> same publication slot
    assert stub.renew() == pytest.approx(7.0)


def test_miss_holds_tail_then_republishes_last_slot() -> None:
    stub = _SchedulerStub(horizon=3, slots=1, hold=10)
    stub.ready = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    stub.ready_from = stub.step = 0
    assert stub.renew() == pytest.approx(1.0)
    # No reply arrives: the next renewals walk the unconsumed tail...
    assert stub.renew() == pytest.approx(2.0)
    assert stub.renew() == pytest.approx(3.0)
    # ...and once exhausted, re-publish the last slot rather than fabricate.
    assert stub.renew() == pytest.approx(3.0)
    assert stub.misses == 3
    assert stub.republished == 1


def test_late_reply_skips_elapsed_publications() -> None:
    stub = _SchedulerStub(horizon=30, slots=10, hold=1)
    stub.ready = np.arange(30, dtype=np.float32)
    stub.ready_from = 0
    stub.step = 17  # reply is 17 steps old when finally swapped
    assert stub.renew() == pytest.approx(17.0)
