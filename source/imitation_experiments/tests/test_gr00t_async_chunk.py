"""Contract tests for the explicit chunk routes' cursor and async execution.

Default environment, no GPU, no GR00T import: the head is a stub whose
prediction values encode (env, frame), the service end is a fake in-thread
zmq REP, and the chunk actor term is a recorder. What must not drift is the
CONTRACT — frame/time alignment (frame k of a request at step t means step
t + k), anchor re-expression, staged swap at the boundary, per-boundary miss
accounting with tail hold, and epoch fencing across resets.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
import pytest
import torch

zmq = pytest.importorskip("zmq")

from imitation_experiments.planner.gr00t_async_chunk import (  # noqa: E402
    Gr00tAsyncChunkPublisher,
    Gr00tAsyncPacketPlanner,
    Gr00tCursorPacketPlanner,
    align_packet_frames,
    frames_to_term_major_rows,
    matrix_from_quat,
    quat_apply,
    quat_inv,
    quat_mul,
    reexpress_root_qpos_frames,
)
from imitation_experiments.planner.gr00t_service_protocol import (  # noqa: E402
    STATE_VALUES,
    decode_act_request,
    encode_chunks_reply,
    encode_control,
)

COMPONENTS = (("joint_qpos", 2), ("root_pos", 3), ("root_ori", 6))
WIDTH = 11
HORIZON = 30
ENCODER_FRAMES = 10
CONSUME = 10
NUM_ENVS = 3


def _frame_value(env: int, request_step: int, frame: int) -> float:
    """The stub head writes prediction frame k of env e requested at t."""
    return float(env) * 10_000.0 + float(request_step) * 100.0 + float(frame)


class StubBatch:
    def __init__(self, tensor: torch.Tensor) -> None:
        self._tensor = tensor

    def get(self, key: Any) -> torch.Tensor:
        assert key == ("planner", "state_history")
        return self._tensor


class StubPublisher:
    """Duck-typed stand-in for Gr00tChunkPublisher: attrs the routes touch.

    The causal state's first value carries the CURRENT episode step of the
    first requested env, so a service reply can reconstruct the request step
    and the frame convention stays checkable end to end.
    """

    def __init__(self, clock: "Clock") -> None:
        self._gr00t_horizon = HORIZON
        self._gr00t_action_dim = WIDTH
        self._gr00t_device = torch.device("cpu")
        self._gr00t_goal_index = torch.arange(NUM_ENVS, dtype=torch.long)
        self._causal_history_steps = 9
        self._components = COMPONENTS
        self.provenance = {"stub": True}
        self._clock = clock
        self.head_calls = 0

    def _causal_observation_fn(self, *, env_ids: torch.Tensor, history_steps: int):
        del history_steps
        rows = int(env_ids.numel())
        state = torch.zeros(rows, STATE_VALUES)
        for row, env in enumerate(env_ids.reshape(-1).tolist()):
            state[row, 0] = float(env)
            state[row, 1] = float(self._clock.steps[int(env)])
        return StubBatch(state)

    def _gr00t_predict(
        self, planner_state: torch.Tensor, goal_index: torch.Tensor
    ) -> torch.Tensor:
        del goal_index
        self.head_calls += 1
        rows = int(planner_state.shape[0])
        out = torch.zeros(rows, HORIZON, WIDTH)
        for row in range(rows):
            env = int(planner_state[row, 0])
            step = int(planner_state[row, 1])
            for frame in range(HORIZON):
                out[row, frame, :] = _frame_value(env, step, frame)
        return out

    def gr00t_stats(self) -> dict[str, Any]:
        return {"head_calls": self.head_calls}

    def gr00t_assert_goal_matches(self, env_ids, names) -> None:  # pragma: no cover
        del env_ids, names


class Clock:
    """Mutable per-env episode step counter standing in for episode_length_buf."""

    def __init__(self) -> None:
        self.steps = torch.zeros(NUM_ENVS, dtype=torch.long)

    def __call__(self) -> torch.Tensor:
        return self.steps.clone()


def identity_anchor() -> tuple[torch.Tensor, torch.Tensor]:
    pos = torch.zeros(NUM_ENVS, 3)
    quat = torch.zeros(NUM_ENVS, 4)
    quat[:, 0] = 1.0
    return pos, quat


class FakeChunkService:
    """In-thread REP echoing the stub head's value convention, with delay."""

    def __init__(self, *, delay_s: float = 0.0) -> None:
        self.endpoint = f"ipc:///tmp/gr00t-chunk-fake-{id(self)}.ipc"
        self.context = zmq.Context()
        self.delay_s = delay_s
        self.requests = 0
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait(2.0)

    def _run(self) -> None:
        import json

        socket = self.context.socket(zmq.REP)
        socket.bind(self.endpoint)
        self._ready.set()
        while True:
            parts = socket.recv_multipart()
            if json.loads(parts[0]).get("kind") == "stop":
                socket.send_multipart(encode_control("stopped"))
                break
            request = decode_act_request(parts)
            if self.delay_s:
                time.sleep(self.delay_s)
            rows = request.states.shape[0]
            self.requests += rows
            chunks = np.zeros((rows, HORIZON, WIDTH), dtype=np.float32)
            for row in range(rows):
                env = int(request.states[row, 0])
                step = int(request.states[row, 1])
                for frame in range(HORIZON):
                    chunks[row, frame, :] = _frame_value(env, step, frame)
            socket.send_multipart(
                encode_chunks_reply(chunks, request.request_ids, head_ms=1.0)
            )
        socket.close(0)

    def stop(self) -> None:
        socket = self.context.socket(zmq.REQ)
        socket.connect(self.endpoint)
        socket.send_multipart([b'{"kind": "stop"}'])
        socket.recv_multipart()
        socket.close(0)


class RecordingTerm:
    """ChunkActorCommand stand-in: records publishes and pinned anchors."""

    def __init__(self) -> None:
        self.window_steps = HORIZON
        self.published: list[tuple[int, dict[str, torch.Tensor]]] = []
        self.pinned: list[tuple[int, torch.Tensor, torch.Tensor]] = []

    def publish(self, env_ids: torch.Tensor, payload: dict[str, torch.Tensor]) -> None:
        for row, env in enumerate(env_ids.reshape(-1).tolist()):
            self.published.append(
                (int(env), {k: v[row].clone() for k, v in payload.items()})
            )

    def pin_anchor_pose(
        self, env_ids: torch.Tensor, pos_w: torch.Tensor, quat_w: torch.Tensor
    ) -> None:
        for row, env in enumerate(env_ids.reshape(-1).tolist()):
            self.pinned.append((int(env), pos_w[row].clone(), quat_w[row].clone()))

    def published_frame0(self, name: str = "joint_qpos") -> torch.Tensor:
        env, payload = self.published[-1]
        del env
        return payload[name].reshape(HORIZON, -1)[0]


# --------------------------------------------------------------------------- #
# Frame math
# --------------------------------------------------------------------------- #


def _random_unit_quat(n: int) -> torch.Tensor:
    q = torch.randn(n, 4)
    return q / q.norm(dim=-1, keepdim=True)


def test_quat_matrix_and_apply_agree() -> None:
    torch.manual_seed(0)
    q = _random_unit_quat(16)
    v = torch.randn(16, 3)
    rotated = torch.matmul(matrix_from_quat(q), v.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(rotated, quat_apply(q, v), atol=1e-5, rtol=1e-5)
    # inverse composes to identity rotation
    identity = quat_mul(q, quat_inv(q))
    torch.testing.assert_close(identity[:, 0].abs(), torch.ones(16), atol=1e-5, rtol=0)
    torch.testing.assert_close(identity[:, 1:], torch.zeros(16, 3), atol=1e-5, rtol=0)


def test_align_packet_frames_shifts_and_holds_tail() -> None:
    frames = torch.arange(HORIZON, dtype=torch.float32).reshape(HORIZON, 1)
    shifted = align_packet_frames(frames.expand(HORIZON, WIDTH), 7)
    assert float(shifted[0, 0]) == 7.0
    assert float(shifted[HORIZON - 8, 0]) == HORIZON - 1
    assert float(shifted[HORIZON - 1, 0]) == HORIZON - 1
    same = align_packet_frames(frames, 0)
    torch.testing.assert_close(same, frames)


def test_reexpress_identity_and_roundtrip() -> None:
    torch.manual_seed(1)
    frames = torch.randn(HORIZON, WIDTH)
    # rot6d blocks must be valid-ish columns; any values are fine for a rigid
    # transform round-trip check.
    same_anchor = (torch.zeros(3), torch.tensor([1.0, 0.0, 0.0, 0.0]))
    out = reexpress_root_qpos_frames(
        frames,
        request_anchor=same_anchor,
        current_anchor=same_anchor,
        components=COMPONENTS,
    )
    torch.testing.assert_close(out, frames, atol=1e-6, rtol=0)
    anchor_a = (torch.randn(3), _random_unit_quat(1)[0])
    anchor_b = (torch.randn(3), _random_unit_quat(1)[0])
    there = reexpress_root_qpos_frames(
        frames, request_anchor=anchor_a, current_anchor=anchor_b, components=COMPONENTS
    )
    back = reexpress_root_qpos_frames(
        there, request_anchor=anchor_b, current_anchor=anchor_a, components=COMPONENTS
    )
    torch.testing.assert_close(back, frames, atol=1e-4, rtol=1e-4)
    # joints never change frame
    torch.testing.assert_close(there[:, :2], frames[:, :2], atol=0.0, rtol=0.0)


def test_frames_to_term_major_layout() -> None:
    frames = torch.randn(2, HORIZON, WIDTH)
    packet = frames_to_term_major_rows(frames, COMPONENTS)
    assert packet.shape == (2, HORIZON * WIDTH)
    joint_block = packet[:, : HORIZON * 2].reshape(2, HORIZON, 2)
    torch.testing.assert_close(joint_block, frames[:, :, :2])


# --------------------------------------------------------------------------- #
# Cursor planner (sync-matched encoded route)
# --------------------------------------------------------------------------- #


def _make_cursor(clock: Clock) -> tuple[Gr00tCursorPacketPlanner, StubPublisher]:
    publisher = StubPublisher(clock)
    planner = Gr00tCursorPacketPlanner(
        publisher,
        consume_frames=CONSUME,
        encoder_frames=ENCODER_FRAMES,
        anchor_state_fn=identity_anchor,
        episode_step_fn=clock,
        num_envs=NUM_ENVS,
        components=COMPONENTS,
    )
    return planner, publisher


def _cursor_step(planner: Gr00tCursorPacketPlanner, clock: Clock) -> torch.Tensor:
    env_ids = torch.arange(NUM_ENVS, dtype=torch.long)
    planner.note_env_ids(env_ids)
    state = torch.zeros(NUM_ENVS, STATE_VALUES)
    for env in range(NUM_ENVS):
        state[env, 0] = float(env)
        state[env, 1] = float(clock.steps[env])
    return planner(state)


def _leading_frame_value(packet: torch.Tensor, env: int) -> float:
    # term-major: the joint block's first frame is the packet's frame 0.
    return float(packet[env, 0])


def test_cursor_replans_on_cadence_and_serves_offsets() -> None:
    clock = Clock()
    planner, publisher = _make_cursor(clock)
    for step in range(2 * CONSUME + 3):
        clock.steps.fill_(step)
        packet = _cursor_step(planner, clock)
        request_step = (step // CONSUME) * CONSUME
        offset = step - request_step
        for env in range(NUM_ENVS):
            assert _leading_frame_value(packet, env) == _frame_value(
                env, request_step, offset
            )
    # one head call per env per boundary: steps 0, 10, 20 -> 3 calls (batched)
    assert publisher.head_calls == 3


def test_cursor_epoch_reset_forces_fresh_plan() -> None:
    clock = Clock()
    planner, _ = _make_cursor(clock)
    clock.steps.fill_(0)
    _cursor_step(planner, clock)
    clock.steps.fill_(5)
    _cursor_step(planner, clock)
    # env 1 resets: its episode counter goes backwards.
    clock.steps[1] = 0
    packet = _cursor_step(planner, clock)
    assert _leading_frame_value(packet, 1) == _frame_value(1, 0, 0)
    assert _leading_frame_value(packet, 0) == _frame_value(0, 0, 5)


# --------------------------------------------------------------------------- #
# Async encoded route
# --------------------------------------------------------------------------- #


def _make_async_cursor(
    clock: Clock, service: FakeChunkService, *, lead: int = 5
) -> Gr00tAsyncPacketPlanner:
    publisher = StubPublisher(clock)
    return Gr00tAsyncPacketPlanner(
        publisher,
        service_endpoint=service.endpoint,
        lead_steps=lead,
        consume_frames=CONSUME,
        encoder_frames=ENCODER_FRAMES,
        anchor_state_fn=identity_anchor,
        episode_step_fn=clock,
        num_envs=NUM_ENVS,
        components=COMPONENTS,
    )


def test_async_cursor_startup_swap_and_boundary_swap() -> None:
    service = FakeChunkService(delay_s=0.0)
    clock = Clock()
    planner = _make_async_cursor(clock, service)
    try:
        for step in range(CONSUME + 2):
            clock.steps.fill_(step)
            packet = _cursor_step(planner, clock)
            if step < CONSUME:
                # startup packet requested at step 0, consumed at its age
                for env in range(NUM_ENVS):
                    assert _leading_frame_value(packet, env) == _frame_value(
                        env, 0, step
                    )
            if step == CONSUME - 1:
                # give the lead request (fired at consume - lead) time to land
                time.sleep(0.2)
        assert planner.startup_syncs == NUM_ENVS
        assert planner.deadline_misses == 0
        # after the boundary the swapped packet is the lead-time one
        clock.steps.fill_(CONSUME + 2)
        packet = _cursor_step(planner, clock)
        request_step = CONSUME - 5  # lead 5 before the boundary
        for env in range(NUM_ENVS):
            assert _leading_frame_value(packet, env) == _frame_value(
                env, request_step, CONSUME + 2 - request_step
            )
    finally:
        planner.close()
        service.stop()


def test_async_cursor_miss_slides_tail_then_holds_window() -> None:
    service = FakeChunkService(delay_s=10.0)  # replies never arrive in time
    clock = Clock()
    planner = _make_async_cursor(clock, service)
    try:
        clock.steps.fill_(0)
        # startup blocks once (10 s delay would fail the test budget), so
        # pre-seed the live packet through the store instead.
        store = planner._store
        store.last_episode_step = clock().clone()
        for env in range(NUM_ENVS):
            frames = torch.zeros(HORIZON, WIDTH)
            for frame in range(HORIZON):
                frames[frame] = _frame_value(env, 0, frame)
            store.pending_request_step[env] = 0
            store.pending_anchor_pos[env] = torch.zeros(3)
            store.pending_anchor_quat[env] = torch.tensor([1.0, 0.0, 0.0, 0.0])
            store.accept_reply(env, frames)
            store.swap(env)
        max_offset = HORIZON - ENCODER_FRAMES
        for step in range(1, 3 * CONSUME + 1):
            clock.steps.fill_(step)
            packet = _cursor_step(planner, clock)
            expected_offset = min(step, max_offset)
            for env in range(NUM_ENVS):
                assert _leading_frame_value(packet, env) == _frame_value(
                    env, 0, expected_offset
                )
        # one miss per env per crossed boundary (steps 10, 20, 30)
        assert planner.deadline_misses == 3 * NUM_ENVS
    finally:
        planner.close()
        service.stop()


# --------------------------------------------------------------------------- #
# Async native route
# --------------------------------------------------------------------------- #


def test_async_native_publishes_aligned_and_pins_request_anchor() -> None:
    service = FakeChunkService(delay_s=0.0)
    clock = Clock()
    term = RecordingTerm()
    publisher = StubPublisher(clock)
    native = Gr00tAsyncChunkPublisher(
        publisher=publisher,
        chunk_term=term,
        hold_steps=CONSUME,
        service_endpoint=service.endpoint,
        lead_steps=5,
        anchor_state_fn=identity_anchor,
        episode_step_fn=clock,
        num_envs=NUM_ENVS,
    )
    try:
        env_ids = torch.arange(NUM_ENVS, dtype=torch.long)
        clock.steps.fill_(0)
        native.note_step()
        native.publish(env_ids)  # startup: blocking, frame 0 aligned
        assert native.startup_syncs == NUM_ENVS
        assert term.published_frame0()[0] == _frame_value(NUM_ENVS - 1, 0, 0)
        assert len(term.pinned) == NUM_ENVS
        # steps 1..9: lead request fires at step 5 (until_boundary == 5)
        for step in range(1, CONSUME):
            clock.steps.fill_(step)
            native.note_step()
        time.sleep(0.3)
        clock.steps.fill_(CONSUME)
        native.note_step()
        native.publish(env_ids)
        assert native.deadline_misses == 0
        # swapped packet was requested at step 5, published at 10: frame 5 leads
        assert term.published_frame0()[0] == _frame_value(NUM_ENVS - 1, 5, 5)
    finally:
        native.close()
        service.stop()


def test_async_native_miss_republishes_shifted_tail() -> None:
    service = FakeChunkService(delay_s=10.0)
    clock = Clock()
    term = RecordingTerm()
    publisher = StubPublisher(clock)
    native = Gr00tAsyncChunkPublisher(
        publisher=publisher,
        chunk_term=term,
        hold_steps=CONSUME,
        service_endpoint=service.endpoint,
        lead_steps=5,
        anchor_state_fn=identity_anchor,
        episode_step_fn=clock,
        num_envs=NUM_ENVS,
    )
    try:
        store = native._store
        clock.steps.fill_(0)
        store.last_episode_step = clock().clone()
        frames = torch.zeros(HORIZON, WIDTH)
        for frame in range(HORIZON):
            frames[frame] = _frame_value(0, 0, frame)
        store.pending_request_step[0] = 0
        store.pending_anchor_quat[0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        store.accept_reply(0, frames)
        store.swap(0)
        env_ids = torch.tensor([0], dtype=torch.long)
        clock.steps.fill_(CONSUME)
        native.note_step()
        native.publish(env_ids)
        assert native.deadline_misses == 1
        # the republished packet leads with the live packet's frame 10
        assert term.published_frame0()[0] == _frame_value(0, 0, CONSUME)
    finally:
        native.close()
        service.stop()
