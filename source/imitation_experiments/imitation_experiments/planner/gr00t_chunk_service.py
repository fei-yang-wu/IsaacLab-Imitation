"""Serve GR00T-head root_qpos window chunks over a line-based stdio protocol.

One JSON object per line on stdin, one per line on stdout. The parent process
(the Embodied-Control tracker harness) sends the causal planner state history
and receives the full predicted root_qpos chunk. The tracker selects the
time-aligned encoder window when it consumes the reply. The head stays in the
`gr00t` Pixi environment; no simulator or tracker import happens here.

Protocol:
    -> {"state": [930 floats]}            # 10x93, oldest->newest, frame-major
    <- {"chunk": [H*38 floats], "head_ms": 12.3}
    -> {"stop": true}

The first stdout line is {"ready": true, "action_horizon": H, ...}.

Run:
    pixi run -e gr00t python -m imitation_experiments.planner.gr00t_chunk_service \
        --checkpoint logs/gr00t_planner_local_debug/gr00t_head_stage_a/checkpoints/update_0002000.pt \
        --goal-features logs/gr00t_planner_local_debug/goal_features/goal_features.pt \
        --goal walk_arc_cw_start_R_slow_001_A443
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import torch

from imitation_experiments.planner.gr00t_head import (
    build_batch,
    build_g1_head_config,
    denormalize_minmax,
    import_head_classes,
    normalize_minmax,
)

STATE_HISTORY = 10
STATE_WIDTH = 93
WINDOW_FRAMES = 10


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--goal-features", type=Path, required=True)
    parser.add_argument("--goal", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--window-frames", type=int, default=WINDOW_FRAMES)
    parser.add_argument("--include-window", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    goal_table = torch.load(args.goal_features, map_location="cpu", weights_only=False)
    goal_names = list(goal_table["goal_names"])
    if args.goal not in goal_names:
        raise SystemExit(f"goal {args.goal!r} not in {goal_names}")
    goal_index = goal_names.index(args.goal)
    features = goal_table["features"][goal_index : goal_index + 1].to(device, dtype)
    feature_mask = goal_table["attention_mask"][goal_index : goal_index + 1].to(device)

    config = build_g1_head_config(
        trunk=checkpoint["trunk_config"],
        action_horizon=int(checkpoint["action_horizon"]),
        state_dropout_prob=0.0,
    )
    _, head_cls = import_head_classes()
    # Upstream constructors print parameter counts. Stdout is reserved for
    # protocol records, so diagnostic output must stay on stderr.
    with redirect_stdout(sys.stderr):
        head = head_cls(config)
        head.load_state_dict(checkpoint["head_state_dict"], strict=True)
        head.to(device, dtype).eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)

    norm = checkpoint["normalization"]
    state_q01, state_q99 = norm["state_q01"], norm["state_q99"]
    action_q01, action_q99 = norm["action_q01"], norm["action_q99"]

    # Warm both inference paths (plain and RTC) before declaring readiness so
    # the first real request does not pay compile/allocator latency.
    horizon = int(checkpoint["action_horizon"])
    if not 0 < args.window_frames <= horizon:
        raise SystemExit(
            f"window frames must be in [1, {horizon}], got {args.window_frames}"
        )
    warm_state = torch.zeros(1, STATE_HISTORY, STATE_WIDTH, device=device, dtype=dtype)
    for warm_rtc in (False, True):
        warm_action = None
        warm_mask = None
        warm_options = None
        if warm_rtc:
            warm_action = torch.zeros(1, horizon, 38, device=device, dtype=dtype)
            warm_mask = torch.ones(1, horizon, dtype=torch.bool, device=device)
            warm_options = {
                "action_horizon": horizon,
                "rtc_overlap_steps": horizon - WINDOW_FRAMES - 1,
                "rtc_frozen_steps": 4,
                "rtc_ramp_rate": 5.0,
            }
        warm_backbone, warm_input = build_batch(
            state=warm_state,
            action=warm_action,
            action_mask=warm_mask,
            language_features=features,
            language_attention_mask=feature_mask,
        )
        with redirect_stdout(sys.stderr), torch.inference_mode():
            head.get_action(warm_backbone, warm_input, warm_options)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    print(
        json.dumps(
            {
                "ready": True,
                "action_horizon": int(checkpoint["action_horizon"]),
                "goal": args.goal,
                "state_history": STATE_HISTORY,
                "state_width": STATE_WIDTH,
                "action_width": 38,
                "window_frames": args.window_frames,
                "params_m": round(
                    sum(v.numel() for v in checkpoint["head_state_dict"].values()) / 1e6
                ),
            },
            separators=(",", ":"),
        ),
        flush=True,
    )

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request = json.loads(line)
        if request.get("stop"):
            break
        state = torch.tensor(request["state"], dtype=torch.float32).reshape(
            1, STATE_HISTORY, STATE_WIDTH
        )
        state = normalize_minmax(state, state_q01, state_q99).to(device, dtype)
        action = None
        action_mask = None
        options = None
        horizon = int(checkpoint["action_horizon"])
        if request.get("prev_chunk") is not None:
            # Real-time chunking: seed the overlap with the previous chunk's
            # tail, freeze the slots covering the inference latency, ramp the
            # rest (gr00t_n1d7.py:356-395 semantics). Chunks start one frame
            # AFTER their request state, so with renewal every `hold` ticks
            # the new chunk's frame 0 aligns with the previous chunk's frame
            # `hold + 1`: overlap = horizon - hold - 1, seeded from the tail.
            prev = torch.tensor(request["prev_chunk"], dtype=torch.float32).reshape(
                1, horizon, -1
            )
            action = normalize_minmax(prev, action_q01, action_q99).to(device, dtype)
            action_mask = torch.ones(1, horizon, dtype=torch.bool, device=device)
            hold = int(request.get("hold_steps", 10))
            options = {
                "action_horizon": horizon,
                "rtc_overlap_steps": horizon - hold - 1,
                "rtc_frozen_steps": int(request.get("freeze_steps", 0)),
                "rtc_ramp_rate": float(request.get("rtc_ramp_rate", 5.0)),
            }
        backbone_output, action_input = build_batch(
            state=state,
            action=action,
            action_mask=action_mask,
            language_features=features,
            language_attention_mask=feature_mask,
        )
        started = time.perf_counter()
        with redirect_stdout(sys.stderr), torch.inference_mode():
            result = head.get_action(backbone_output, action_input, options)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        head_ms = (time.perf_counter() - started) * 1000.0
        chunk = denormalize_minmax(
            result["action_pred"][0].float().cpu(), action_q01, action_q99
        )
        if not torch.isfinite(chunk).all():
            raise RuntimeError("GR00T action head produced a non-finite chunk")
        response = {
            "chunk": [float(v) for v in chunk.reshape(-1)],
            "head_ms": round(head_ms, 2),
        }
        if args.include_window:
            response["window"] = [
                float(v) for v in chunk[: args.window_frames].reshape(-1)
            ]
        print(
            json.dumps(response, separators=(",", ":")),
            flush=True,
        )


if __name__ == "__main__":
    main()
