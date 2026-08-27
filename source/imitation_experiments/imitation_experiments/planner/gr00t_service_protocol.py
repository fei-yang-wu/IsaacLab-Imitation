"""Wire protocol for the batched GR00T chunk service (phase D1).

One request/reply pair over a ZeroMQ REQ/REP socket, as a two-part message:
a JSON header and one raw little-endian float32 buffer. Raw bytes rather
than pickled tensors because the two ends deliberately run different torch
versions (the head trains under the upstream 2.9 pin, Isaac Lab owns 2.11);
nothing torch-specific may cross the socket.

Request::

    part 0 (json): {"kind": "act", "rows": B, "goal_ids": [B ints],
                    "request_ids": [B ints]}
    part 1 (f32):  states, B x 930 (10 frames x 93 values, oldest first)

Reply::

    part 0 (json): {"kind": "chunks", "rows": B, "horizon": H, "dim": D,
                    "request_ids": [B ints], "head_ms": float}
    part 1 (f32):  chunks, B x H x D

Control requests carry an empty payload part::

    {"kind": "describe"} -> {"kind": "described", "action_horizon": H,
                             "action_dim": D, "goals": [names...],
                             "checkpoint": ..., "update": ...}
    {"kind": "stop"}     -> {"kind": "stopped"}

`request_ids` are opaque to the service and echoed verbatim: the client uses
them to match replies to per-environment requests across the asynchronous
boundary. A reply with a wrong shape, a non-finite value, or unknown ids is
a client-side fault, never something to repair silently.

This module is importable from BOTH Pixi environments, so it must not import
torch, zmq, or anything heavier than numpy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

PROTOCOL_VERSION = 1
STATE_HISTORY = 10
STATE_WIDTH = 93
STATE_VALUES = STATE_HISTORY * STATE_WIDTH


@dataclass(frozen=True)
class ActRequest:
    goal_ids: np.ndarray  # [B] int64
    request_ids: np.ndarray  # [B] int64
    states: np.ndarray  # [B, STATE_VALUES] float32


def encode_act_request(
    states: np.ndarray, goal_ids: np.ndarray, request_ids: np.ndarray
) -> list[bytes]:
    states = np.ascontiguousarray(states, dtype=np.float32)
    if states.ndim != 2 or states.shape[1] != STATE_VALUES:
        msg = f"states must be [B, {STATE_VALUES}], got {states.shape}"
        raise ValueError(msg)
    rows = int(states.shape[0])
    header = {
        "kind": "act",
        "version": PROTOCOL_VERSION,
        "rows": rows,
        "goal_ids": [int(g) for g in np.asarray(goal_ids).reshape(-1)],
        "request_ids": [int(r) for r in np.asarray(request_ids).reshape(-1)],
    }
    if len(header["goal_ids"]) != rows or len(header["request_ids"]) != rows:
        msg = "goal_ids/request_ids row count disagrees with states"
        raise ValueError(msg)
    return [json.dumps(header).encode(), states.tobytes()]


def decode_act_request(parts: list[bytes]) -> ActRequest:
    header = json.loads(parts[0])
    if header.get("kind") != "act":
        msg = f"expected an act request, got {header.get('kind')!r}"
        raise ValueError(msg)
    rows = int(header["rows"])
    states = np.frombuffer(parts[1], dtype=np.float32).reshape(rows, STATE_VALUES)
    return ActRequest(
        goal_ids=np.asarray(header["goal_ids"], dtype=np.int64),
        request_ids=np.asarray(header["request_ids"], dtype=np.int64),
        states=states,
    )


def encode_chunks_reply(
    chunks: np.ndarray, request_ids: np.ndarray, head_ms: float
) -> list[bytes]:
    chunks = np.ascontiguousarray(chunks, dtype=np.float32)
    if chunks.ndim != 3:
        msg = f"chunks must be [B, H, D], got {chunks.shape}"
        raise ValueError(msg)
    header = {
        "kind": "chunks",
        "version": PROTOCOL_VERSION,
        "rows": int(chunks.shape[0]),
        "horizon": int(chunks.shape[1]),
        "dim": int(chunks.shape[2]),
        "request_ids": [int(r) for r in np.asarray(request_ids).reshape(-1)],
        "head_ms": float(head_ms),
    }
    return [json.dumps(header).encode(), chunks.tobytes()]


def decode_chunks_reply(parts: list[bytes]) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (chunks [B, H, D], request_ids [B], head_ms); fault on bad data."""
    header = json.loads(parts[0])
    if header.get("kind") != "chunks":
        msg = f"service fault: {header}"
        raise RuntimeError(msg)
    rows, horizon, dim = (
        int(header["rows"]),
        int(header["horizon"]),
        int(header["dim"]),
    )
    chunks = np.frombuffer(parts[1], dtype=np.float32).reshape(rows, horizon, dim)
    if not np.isfinite(chunks).all():
        msg = "service returned a non-finite chunk"
        raise RuntimeError(msg)
    request_ids = np.asarray(header["request_ids"], dtype=np.int64)
    if request_ids.shape[0] != rows:
        msg = "service reply request_ids disagree with row count"
        raise RuntimeError(msg)
    return chunks, request_ids, float(header.get("head_ms", 0.0))


def encode_control(kind: str) -> list[bytes]:
    return [json.dumps({"kind": kind, "version": PROTOCOL_VERSION}).encode(), b""]


def decode_header(parts: list[bytes]) -> dict:
    return json.loads(parts[0])


__all__ = [
    "PROTOCOL_VERSION",
    "STATE_HISTORY",
    "STATE_VALUES",
    "STATE_WIDTH",
    "ActRequest",
    "decode_act_request",
    "decode_chunks_reply",
    "decode_header",
    "encode_act_request",
    "encode_chunks_reply",
    "encode_control",
]
