"""Batched GR00T chunk service over ZeroMQ (phase D1).

One process owns the trained action head in the ``gr00t`` Pixi environment
and answers batched ``act`` requests from the Isaac evaluator's asynchronous
sampler. The wire format lives in
:mod:`imitation_experiments.planner.gr00t_service_protocol`; nothing
torch-specific crosses the socket, so the two ends keep their own torch pins
(2.9 here, 2.11 in Isaac).

The service is stateless per request: every request carries its rows'
states and goal ids, every reply echoes the caller's ``request_ids``. All
goal features are preloaded to the device — the whole table is megabytes.

Run::

    pixi run -e gr00t python -m imitation_experiments.planner.gr00t_batch_service \
        --checkpoint outputs/planner_10b/arms/fsq64_10b/checkpoints/update_0012000.pt \
        --goal-features outputs/gr00t_language30/goal_features/goal_features.pt \
        --endpoint ipc:///tmp/gr00t_batch_service.ipc

The first stdout line after the model is warm is a JSON ready record; the
launcher waits for it before starting the evaluator.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from imitation_experiments.planner.gr00t_service_protocol import (
    STATE_HISTORY,
    STATE_WIDTH,
    decode_act_request,
    decode_header,
    encode_chunks_reply,
    encode_control,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--goal-features", type=Path, required=True)
    parser.add_argument("--endpoint", default="ipc:///tmp/gr00t_batch_service.ipc")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--num-inference-timesteps", type=int, default=4)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Seed the flow-matching sampler once at startup. The reply for a "
            "given request then still depends on request order — an async row "
            "is never bit-comparable with a sync one; pin the seed to make "
            "the SERVICE run reproducible, not to match D0."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    import torch
    import zmq

    from imitation_experiments.planner.gr00t_head import (
        build_batch,
        build_g1_head_config,
        denormalize_minmax,
        import_head_classes,
        normalize_minmax,
    )

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    if args.seed is not None:
        torch.manual_seed(int(args.seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(args.seed))

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    goal_table = torch.load(args.goal_features, map_location="cpu", weights_only=False)
    goal_names = list(goal_table["goal_names"])
    features_all = goal_table["features"].to(device, dtype)
    feature_mask_all = goal_table["attention_mask"].to(device)

    config = build_g1_head_config(
        trunk=checkpoint["trunk_config"],
        action_horizon=int(checkpoint["action_horizon"]),
        max_action_dim=int(checkpoint.get("action_dim", 38)),
        state_dropout_prob=0.0,
    )
    config.num_inference_timesteps = int(args.num_inference_timesteps)
    _, head_cls = import_head_classes()
    with redirect_stdout(sys.stderr):
        head = head_cls(config)
        head.load_state_dict(checkpoint["head_state_dict"], strict=True)
        head.to(device, dtype).eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)

    norm = checkpoint["normalization"]
    state_q01, state_q99 = norm["state_q01"], norm["state_q99"]
    action_q01, action_q99 = norm["action_q01"], norm["action_q99"]
    horizon = int(checkpoint["action_horizon"])
    action_dim = int(checkpoint.get("action_dim", 38))

    def forward(states: np.ndarray, goal_ids: np.ndarray) -> tuple[np.ndarray, float]:
        state = torch.from_numpy(np.ascontiguousarray(states)).reshape(
            -1, STATE_HISTORY, STATE_WIDTH
        )
        state = normalize_minmax(state, state_q01, state_q99).to(device, dtype)
        index = torch.from_numpy(np.ascontiguousarray(goal_ids)).to(device)
        backbone_output, action_input = build_batch(
            state=state,
            action=None,
            action_mask=None,
            language_features=features_all.index_select(0, index),
            language_attention_mask=feature_mask_all.index_select(0, index),
        )
        started = time.perf_counter()
        with redirect_stdout(sys.stderr), torch.inference_mode():
            result = head.get_action(backbone_output, action_input, None)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        head_ms = (time.perf_counter() - started) * 1000.0
        chunks = denormalize_minmax(
            result["action_pred"].float().cpu(), action_q01, action_q99
        )
        return chunks.numpy(), head_ms

    # Warm before declaring readiness so the first real request does not pay
    # allocator/compile latency, and so a broken checkpoint fails HERE.
    warm, _ = forward(
        np.zeros((2, STATE_HISTORY * STATE_WIDTH), dtype=np.float32),
        np.zeros(2, dtype=np.int64),
    )
    if not np.isfinite(warm).all():
        raise RuntimeError("warmup produced a non-finite chunk")

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(args.endpoint)

    print(
        json.dumps(
            {
                "ready": True,
                "endpoint": args.endpoint,
                "action_horizon": horizon,
                "action_dim": action_dim,
                "goals": len(goal_names),
                "checkpoint": str(args.checkpoint),
                "update": int(checkpoint.get("update", -1)),
                "target_mode": checkpoint.get("target_mode"),
                "dtype": args.dtype,
                "num_inference_timesteps": int(args.num_inference_timesteps),
                "seed": args.seed,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )

    served = 0
    while True:
        parts = socket.recv_multipart()
        header = decode_header(parts)
        kind = header.get("kind")
        if kind == "stop":
            socket.send_multipart(encode_control("stopped"))
            break
        if kind == "describe":
            socket.send_multipart(
                [
                    json.dumps(
                        {
                            "kind": "described",
                            "action_horizon": horizon,
                            "action_dim": action_dim,
                            "goals": goal_names,
                            "checkpoint": str(args.checkpoint),
                            "update": int(checkpoint.get("update", -1)),
                        }
                    ).encode(),
                    b"",
                ]
            )
            continue
        request = decode_act_request(parts)
        chunks, head_ms = forward(request.states, request.goal_ids)
        socket.send_multipart(encode_chunks_reply(chunks, request.request_ids, head_ms))
        served += int(request.states.shape[0])
        print(
            json.dumps(
                {
                    "served_rows": served,
                    "batch": int(request.states.shape[0]),
                    "head_ms": round(head_ms, 2),
                },
                separators=(",", ":"),
            ),
            file=sys.stderr,
            flush=True,
        )

    socket.close(0)
    context.term()


if __name__ == "__main__":
    main()
