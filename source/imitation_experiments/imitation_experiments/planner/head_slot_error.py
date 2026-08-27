"""Per-slot open-loop accuracy of a trained GR00T head, off its own table.

A chunk of ``H`` predicted slots is consumed over ``H`` control steps, so a
deployed row's quality depends on how accurate slot ``k`` is, not on the
chunk average: re-planning every ``consume_frames`` steps means the tracker
only ever sees slots ``0..consume_frames-1``. This module measures accuracy
as a function of ``k`` so the consumption cadence can be chosen from evidence
instead of from the training cadence.

Two numbers per slot, both against the table's own target:

* ``cosine`` — direction agreement, the quantity the closed-loop rows report
  as ``published_vs_oracle_z_cosine_mean``;
* ``rmse`` — scale-aware error in target units.

Run (GPU, the head's own environment)::

    pixi run -e gr00t python -m imitation_experiments.planner.head_slot_error \
        --checkpoint outputs/planner_10b/arms/ln_hold1_10b/checkpoints/update_0012000.pt \
        --table outputs/planner_10b/tables/ln_hold1_10b_table.pt \
        --rows 4096

Rows are drawn evenly across the table rather than from its head, because a
table is written in collection order and its first rows are all episode
starts.
"""

from __future__ import annotations

import argparse
import json

import torch
from torch import Tensor


def per_slot_stats(
    prediction: Tensor, target: Tensor, valid: Tensor | None = None
) -> list[dict[str, float]]:
    """Per-slot cosine and RMSE between ``[B, H, D]`` prediction and target.

    ``valid`` is an optional ``[B, H]`` mask; a slot with no valid row reports
    ``count`` 0 and NaN statistics rather than silently averaging nothing.
    """
    if prediction.shape != target.shape:
        msg = f"shape mismatch: prediction {tuple(prediction.shape)} vs target {tuple(target.shape)}."
        raise ValueError(msg)
    if prediction.ndim != 3:
        msg = f"expected [B, H, D], got {tuple(prediction.shape)}."
        raise ValueError(msg)
    prediction = prediction.float()
    target = target.float()
    horizon = int(prediction.shape[1])
    out: list[dict[str, float]] = []
    for slot in range(horizon):
        pred_k = prediction[:, slot]
        target_k = target[:, slot]
        if valid is not None:
            keep = valid[:, slot].bool()
            pred_k = pred_k[keep]
            target_k = target_k[keep]
        count = int(pred_k.shape[0])
        if count == 0:
            out.append(
                {"slot": slot, "count": 0, "cosine": float("nan"), "rmse": float("nan")}
            )
            continue
        cosine = torch.nn.functional.cosine_similarity(pred_k, target_k, dim=-1)
        rmse = (pred_k - target_k).pow(2).mean(dim=-1).sqrt()
        out.append(
            {
                "slot": slot,
                "count": count,
                "cosine": float(cosine.mean()),
                "rmse": float(rmse.mean()),
            }
        )
    return out


def _load_head(checkpoint_path: str, device: torch.device, dtype: torch.dtype):
    from imitation_experiments.planner.gr00t_head import (
        build_g1_head_config,
        import_head_classes,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = build_g1_head_config(
        trunk=checkpoint["trunk_config"],
        action_horizon=int(checkpoint["action_horizon"]),
        max_action_dim=int(checkpoint.get("action_dim", 38)),
        state_dropout_prob=0.0,
    )
    config.num_inference_timesteps = 4
    _, head_cls = import_head_classes()
    head = head_cls(config)
    head.load_state_dict(checkpoint["head_state_dict"], strict=True)
    head.to(device, dtype).eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    return head, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument("--goal-features", default=None)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--inference-steps", type=int, default=4)
    parser.add_argument("--state-field", default="oracle_rollout_state_history")
    parser.add_argument("--target", default="latent", choices=("latent", "chunk"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    from imitation_experiments.planner.gr00t_head import (
        PLANNER_STATE_HISTORY,
        PLANNER_STATE_WIDTH,
        build_batch,
        denormalize_minmax,
        normalize_minmax,
    )

    device = torch.device(args.device)
    dtype = torch.float32
    head, checkpoint = _load_head(args.checkpoint, device, dtype)
    head.config.num_inference_timesteps = int(args.inference_steps)

    table = torch.load(args.table, map_location="cpu", weights_only=False)
    states = table["states"][args.state_field]
    target_key = "latent_target" if args.target == "latent" else "chunk_target"
    valid_key = "latent_valid" if args.target == "latent" else "chunk_valid"
    targets = table[target_key]
    valid = table.get(valid_key)
    goal_id = table["goal_id"]

    goal_features_path = args.goal_features or checkpoint.get("goal_features_path")
    if goal_features_path is None:
        msg = "pass --goal-features; the checkpoint records none."
        raise ValueError(msg)
    goal_table = torch.load(goal_features_path, map_location="cpu", weights_only=False)
    features_all = goal_table["features"].to(device, dtype)
    mask_all = goal_table["attention_mask"].to(device)

    rows = min(int(args.rows), int(states.shape[0]))
    index = torch.linspace(0, int(states.shape[0]) - 1, rows).long()

    norm = checkpoint["normalization"]
    horizon = int(checkpoint["action_horizon"])
    action_dim = int(checkpoint.get("action_dim", 38))
    state_frames = PLANNER_STATE_HISTORY
    if states.ndim != 2 or int(states.shape[1]) != state_frames * PLANNER_STATE_WIDTH:
        msg = (
            f"state field {args.state_field!r} has shape {tuple(states.shape)}; "
            f"expected [rows, {state_frames * PLANNER_STATE_WIDTH}]."
        )
        raise ValueError(msg)

    predictions: list[Tensor] = []
    for start in range(0, rows, int(args.batch_size)):
        rows_index = index[start : start + int(args.batch_size)]
        state = states.index_select(0, rows_index).reshape(
            len(rows_index), state_frames, PLANNER_STATE_WIDTH
        )
        state = normalize_minmax(state, norm["state_q01"], norm["state_q99"]).to(
            device, dtype
        )
        goal = goal_id.index_select(0, rows_index).to(device)
        backbone_output, action_input = build_batch(
            state=state,
            action=None,
            action_mask=None,
            language_features=features_all.index_select(0, goal),
            language_attention_mask=mask_all.index_select(0, goal),
        )
        with torch.inference_mode():
            result = head.get_action(backbone_output, action_input, None)
        # `get_action` returns a BatchFeature; the tensor lives under
        # "action_pred", exactly as the batched service reads it.
        chunk = denormalize_minmax(
            result["action_pred"].float().cpu(),
            norm["action_q01"],
            norm["action_q99"],
        )[:, :horizon, :action_dim]
        predictions.append(chunk)

    prediction = torch.cat(predictions, dim=0)
    target = targets.index_select(0, index).float()
    mask = None if valid is None else valid.index_select(0, index)
    stats = per_slot_stats(prediction, target, mask)

    record = {
        "checkpoint": args.checkpoint,
        "table": args.table,
        "target": args.target,
        "rows": rows,
        "inference_steps": int(args.inference_steps),
        "per_slot": stats,
    }
    print(json.dumps(record, indent=1))
    if args.output_json:
        with open(args.output_json, "w") as handle:
            json.dump(record, handle, indent=1)


if __name__ == "__main__":
    main()
