"""Train the verbatim GR00T N1.7 action head on G1 planner samples.

Runs in the ``gr00t`` Pixi environment with our training loop; the model class
comes untouched from ``external/Isaac-GR00T``. Two arms:

- from scratch (control): fresh head, no load;
- warm start: filtered load of the embodiment-independent trunk from the
  bundle exported by ``cache_gr00t_goal_features.py``, fresh G1 projectors,
  stage A (projectors only) then stage B (unfreeze DiT).

``--synthetic_rows N`` replaces sample files and the goal-feature cache with
random tensors of the correct shapes — the local pipeline smoke.

Run from the repository root, e.g.:

    pixi run -e gr00t python -m imitation_experiments.planner.train_gr00t_head \
        --output_dir outputs/gr00t_head_smoke --preset debug \
        --synthetic_rows 256 --num_updates 20
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch

from imitation_experiments.planner.gr00t_head import (
    BACKBONE_EMBEDDING_DIM,
    DEBUG_TRUNK_CONFIG,
    N17_TRUNK_CONFIG,
    PLANNER_STATE_HISTORY,
    PLANNER_STATE_WIDTH,
    ROOT_QPOS_WIDTH,
    build_batch,
    build_g1_head_config,
    compute_quantile_stats,
    filtered_pretrained_load,
    gr00t_submodule_commit,
    import_head_classes,
    normalize_minmax,
    save_provenance,
)

TRUNK_PRESETS = {"debug": DEBUG_TRUNK_CONFIG, "n17": N17_TRUNK_CONFIG}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--preset", choices=sorted(TRUNK_PRESETS), default="n17")
    parser.add_argument("--action_horizon", type=int, default=30)
    # Optimization defaults mirror Isaac-GR00T's finetune recipe at the pinned
    # commit (gr00t/configs/{finetune_config,training/training_config}.py):
    # AdamW(fused) lr 1e-4, wd 1e-5, cosine schedule with warmup_ratio 0.05,
    # grad clip 1.0, batch 64, bf16 + tf32, no EMA, and — their default —
    # projectors and DiT trained together (stage_a_updates=0).
    parser.add_argument("--num_updates", type=int, default=10_000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight_decay", type=float, default=1.0e-5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument(
        "--lr_schedule", choices=("cosine", "constant"), default="cosine"
    )
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable DiT gradient checkpointing (their trainer's flag).",
    )
    parser.add_argument(
        "--state_dropout_prob",
        type=float,
        default=0.2,
        help=(
            "GR00T's finetune default (0.2; 0.8 is pretrain-only). Also forces "
            "reliance on the language tokens. 0.0 disables."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--bf16", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--checkpoint_interval", type=int, default=2000)
    # Data: either real sample files + goal features, or synthetic.
    parser.add_argument("--samples", type=Path, nargs="*", default=[])
    parser.add_argument("--goal_features", type=Path, default=None)
    parser.add_argument("--synthetic_rows", type=int, default=0)
    # Defaults match the rollout-training-sample schema written by
    # scripts/rlopt/eval_skill_commander_closed_loop.py.
    parser.add_argument("--state_key", type=str, default="oracle_rollout_state_history")
    parser.add_argument("--future_key", type=str, default="expert_root_qpos_future")
    parser.add_argument(
        "--future_valid_key", type=str, default="expert_root_qpos_future_valid"
    )
    parser.add_argument("--goal_key", type=str, default="motion_name")
    # Warm start.
    parser.add_argument("--pretrained_bundle", type=Path, default=None)
    parser.add_argument(
        "--stage_a_updates",
        type=int,
        default=0,
        help=(
            "With --pretrained_bundle: updates with the DiT trunk frozen "
            "(projectors only) before unfreezing everything."
        ),
    )
    args = parser.parse_args()
    if bool(args.samples) == bool(args.synthetic_rows > 0):
        parser.error("Provide exactly one of --samples or --synthetic_rows.")
    if args.samples and args.goal_features is None:
        parser.error("--goal_features is required with real --samples.")
    if args.stage_a_updates > 0 and args.pretrained_bundle is None:
        parser.error("--stage_a_updates requires --pretrained_bundle.")
    return args


def _load_rows(args: argparse.Namespace) -> dict[str, torch.Tensor]:
    """Concatenate sample files into row-aligned tensors (fail-loud schema)."""
    horizon = int(args.action_horizon)
    states, actions, valids, goal_ids = [], [], [], []
    goal_table = torch.load(args.goal_features, map_location="cpu", weights_only=False)
    name_to_index = {name: i for i, name in enumerate(goal_table["goal_names"])}
    for path in args.samples:
        sample = torch.load(path, map_location="cpu", weights_only=False)
        for key in (args.state_key, args.future_key, args.future_valid_key, args.goal_key):
            if key not in sample:
                msg = f"{path} lacks key {key!r}; available: {sorted(sample)[:20]}"
                raise KeyError(msg)
        future = sample[args.future_key].float()
        valid = sample[args.future_valid_key].bool()
        state = sample[args.state_key].float()
        rows = int(future.shape[0])
        if future.shape[1] < horizon:
            msg = (
                f"{path}: stored lookahead {tuple(future.shape)} is shorter than "
                f"--action_horizon {horizon}."
            )
            raise ValueError(msg)
        if tuple(future.shape[2:]) != (ROOT_QPOS_WIDTH,) or state.shape[0] != rows:
            msg = f"{path}: schema mismatch future={tuple(future.shape)} state={tuple(state.shape)}"
            raise ValueError(msg)
        state = state.reshape(rows, PLANNER_STATE_HISTORY, PLANNER_STATE_WIDTH)
        names = sample[args.goal_key]
        ids = []
        for name in names:
            if str(name) not in name_to_index:
                msg = f"{path}: goal {name!r} missing from the goal-feature cache."
                raise KeyError(msg)
            ids.append(name_to_index[str(name)])
        keep = valid[:, :horizon].any(dim=1)
        states.append(state[keep])
        actions.append(future[keep, :horizon])
        valids.append(valid[keep, :horizon])
        goal_ids.append(torch.as_tensor(ids, dtype=torch.long)[keep])
    return {
        "state": torch.cat(states),
        "action": torch.cat(actions),
        "valid": torch.cat(valids),
        "goal_id": torch.cat(goal_ids),
        "features": goal_table["features"].float(),
        "feature_mask": goal_table["attention_mask"].bool(),
    }


def _synthetic_rows(args: argparse.Namespace) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(int(args.seed))
    rows = int(args.synthetic_rows)
    horizon = int(args.action_horizon)
    num_goals, seq = 4, 12
    return {
        "state": torch.randn(rows, PLANNER_STATE_HISTORY, PLANNER_STATE_WIDTH, generator=generator),
        "action": torch.randn(rows, horizon, ROOT_QPOS_WIDTH, generator=generator),
        "valid": torch.ones(rows, horizon, dtype=torch.bool),
        "goal_id": torch.randint(0, num_goals, (rows,), generator=generator),
        "features": torch.randn(num_goals, seq, BACKBONE_EMBEDDING_DIM, generator=generator),
        "feature_mask": torch.ones(num_goals, seq, dtype=torch.bool),
    }


def main() -> None:
    args = _parse_args()
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device)
    data = _synthetic_rows(args) if args.synthetic_rows else _load_rows(args)
    rows = int(data["state"].shape[0])
    print(f"[train] {rows} rows, horizon {args.action_horizon}, preset {args.preset}")

    # Quantile normalization (GR00T use_percentiles convention), stored with
    # the checkpoint; state uses the same scheme.
    action_q01, action_q99 = compute_quantile_stats(data["action"], data["valid"])
    state_q01, state_q99 = compute_quantile_stats(data["state"])
    data["action"] = normalize_minmax(data["action"], action_q01, action_q99)
    data["state"] = normalize_minmax(data["state"], state_q01, state_q99)

    trunk = dict(TRUNK_PRESETS[args.preset])
    bundle = None
    if args.pretrained_bundle is not None:
        bundle = torch.load(args.pretrained_bundle, map_location="cpu", weights_only=False)
        source_config = bundle.get("source_config", {})
        for key in ("hidden_size", "input_embedding_dim", "backbone_embedding_dim",
                    "diffusion_model_cfg", "vl_self_attention_cfg"):
            if key in source_config:
                trunk[key] = source_config[key]
        print("[train] trunk architecture taken from the pretrained bundle")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    config = build_g1_head_config(
        trunk=trunk,
        action_horizon=int(args.action_horizon),
        state_dropout_prob=float(args.state_dropout_prob),
    )
    _, head_cls = import_head_classes()
    head = head_cls(config).to(device)
    if bool(args.gradient_checkpointing):
        head.model.enable_gradient_checkpointing()

    load_manifest: dict | None = None
    if bundle is not None:
        load_manifest = filtered_pretrained_load(head, bundle["trunk_state_dict"])
        print(
            f"[train] warm start: kept {load_manifest['num_kept_params']/1e6:.1f}M "
            f"params, fresh {load_manifest['num_fresh_params']/1e6:.1f}M"
        )

    total_updates = int(args.num_updates)
    warmup_updates = max(1, int(total_updates * float(args.warmup_ratio)))

    def _make_optimizer() -> torch.optim.AdamW:
        params = [p for p in head.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            params,
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            fused=device.type == "cuda",
        )

    def _lr_lambda(update_index: int) -> float:
        if args.lr_schedule == "constant":
            return 1.0
        if update_index < warmup_updates:
            return (update_index + 1) / warmup_updates
        progress = (update_index - warmup_updates) / max(
            total_updates - warmup_updates, 1
        )
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    def _make_scheduler(
        opt: torch.optim.AdamW, last_update: int
    ) -> torch.optim.lr_scheduler.LambdaLR:
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
        for _ in range(last_update):
            scheduler.step()
        return scheduler

    stage_a = int(args.stage_a_updates)
    if stage_a > 0:
        head.set_trainable_parameters(
            tune_projector=True, tune_diffusion_model=False, tune_vlln=False
        )
    optimizer = _make_optimizer()
    scheduler = _make_scheduler(optimizer, 0)

    generator = torch.Generator().manual_seed(int(args.seed) + 1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = args.output_dir / "checkpoints"
    autocast_dtype = torch.bfloat16 if bool(args.bf16) else None
    start = time.time()
    for update in range(1, int(args.num_updates) + 1):
        if stage_a > 0 and update == stage_a + 1:
            head.set_trainable_parameters(
                tune_projector=True, tune_diffusion_model=True, tune_vlln=True
            )
            optimizer = _make_optimizer()
            scheduler = _make_scheduler(optimizer, update - 1)
            print(f"[train] update {update}: stage B — trunk unfrozen")
        index = torch.randint(0, rows, (int(args.batch_size),), generator=generator)
        goal = data["goal_id"][index]
        backbone_output, action_input = build_batch(
            state=data["state"][index].to(device),
            action=data["action"][index].to(device),
            action_mask=data["valid"][index].to(device),
            language_features=data["features"][goal].to(device),
            language_attention_mask=data["feature_mask"][goal].to(device),
        )
        head.train()
        optimizer.zero_grad(set_to_none=True)
        if autocast_dtype is not None:
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                out = head(backbone_output, action_input)
                loss = out["loss"]
        else:
            out = head(backbone_output, action_input)
            loss = out["loss"]
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in head.parameters() if p.requires_grad], float(args.grad_clip)
        )
        optimizer.step()
        scheduler.step()
        if update % int(args.log_interval) == 0 or update == 1:
            rate = update / max(time.time() - start, 1.0e-9)
            print(
                f"[train] update {update}/{args.num_updates} "
                f"loss={float(loss.detach()):.6f} grad={float(grad_norm):.3f} "
                f"lr={scheduler.get_last_lr()[0]:.2e} ({rate:.2f} up/s)",
                flush=True,
            )
        if update % int(args.checkpoint_interval) == 0 or update == int(args.num_updates):
            checkpoints.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "head_state_dict": head.state_dict(),
                    "trunk_config": trunk,
                    "action_horizon": int(args.action_horizon),
                    "normalization": {
                        "action_q01": action_q01, "action_q99": action_q99,
                        "state_q01": state_q01, "state_q99": state_q99,
                    },
                    "pretrained_load_manifest": load_manifest,
                    "update": update,
                },
                checkpoints / f"update_{update:07d}.pt",
            )
    save_provenance(
        args.output_dir / "training_provenance.json",
        {
            "gr00t_submodule_commit": gr00t_submodule_commit(),
            "pretrained_bundle": str(args.pretrained_bundle) if args.pretrained_bundle else None,
            "stage_a_updates": stage_a,
            "num_updates": int(args.num_updates),
            "lr_schedule": str(args.lr_schedule),
            "warmup_ratio": float(args.warmup_ratio),
            "state_dropout_prob": float(args.state_dropout_prob),
            "gradient_checkpointing": bool(args.gradient_checkpointing),
            "rows": rows,
            "preset": args.preset,
            "synthetic": bool(args.synthetic_rows),
            "seed": int(args.seed),
        },
    )
    print(f"[PASS] training complete -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
