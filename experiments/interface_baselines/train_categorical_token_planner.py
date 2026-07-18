#!/usr/bin/env python3
"""Train the shared causal Transformer to predict per-step token packets."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from rlopt.agent.causal_interface_planner import (
    CausalInterfaceTransformerCategoricalPlanner,
)

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))

from interface_planner_common import (  # noqa: E402
    InterfaceTargetSpec,
    load_planner_checkpoint,
    load_rollout_samples,
    paired_target_key,
    parameter_counts,
    save_planner_checkpoint,
    write_jsonl,
)


MODEL_PRESETS: dict[str, dict[str, int | float]] = {
    "tiny": {
        "d_model": 64,
        "num_layers": 1,
        "num_heads": 4,
        "feedforward_dim": 128,
        "num_state_tokens": 1,
        "dropout": 0.0,
    },
    "small": {
        "d_model": 256,
        "num_layers": 4,
        "num_heads": 4,
        "feedforward_dim": 1024,
        "num_state_tokens": 2,
        "dropout": 0.0,
    },
    "medium": {
        "d_model": 512,
        "num_layers": 6,
        "num_heads": 8,
        "feedforward_dim": 2048,
        "num_state_tokens": 4,
        "dropout": 0.0,
    },
    "large": {
        "d_model": 768,
        "num_layers": 8,
        "num_heads": 12,
        "feedforward_dim": 3072,
        "num_state_tokens": 4,
        "dropout": 0.0,
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument(
        "--state_key",
        choices=("expert_planner_state", "planner_state"),
        default="planner_state",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_updates", type=int, default=2000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight_decay", type=float, default=1.0e-4)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--model_size", choices=tuple(MODEL_PRESETS), default="medium")
    return parser.parse_args()


def _device(raw: str) -> torch.device:
    if str(raw).strip().lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def _select_rows(
    data: dict[str, torch.Tensor], *, max_samples: int, seed: int
) -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:
    rows = int(data["causal_target"].shape[0])
    if int(max_samples) <= 0 or int(max_samples) >= rows:
        return data, None
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    indices = torch.randperm(rows, generator=generator)[: int(max_samples)]
    indices, _ = torch.sort(indices)
    selected = {
        key: value.index_select(0, indices) if value.shape[0] == rows else value
        for key, value in data.items()
    }
    return selected, indices


@torch.no_grad()
def _evaluate(
    planner: CausalInterfaceTransformerCategoricalPlanner,
    state: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    planner.eval()
    logits = planner.logits(state)
    prediction = logits.argmax(dim=-1)
    token_accuracy = (prediction == target).to(torch.float32).mean()
    packet_accuracy = (prediction == target).all(dim=-1).to(torch.float32).mean()
    loss = F.cross_entropy(
        logits.reshape(-1, planner.codebook_size), target.reshape(-1)
    )
    return {
        "eval/cross_entropy": float(loss.item()),
        "eval/token_accuracy": float(token_accuracy.item()),
        "eval/packet_accuracy": float(packet_accuracy.item()),
        "eval/sample_count": float(state.shape[0]),
    }


def main() -> None:
    args = _parse_args()
    if args.batch_size <= 0 or args.num_updates <= 0 or args.log_interval <= 0:
        raise ValueError("batch_size, num_updates, and log_interval must be positive.")
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    device = _device(str(args.device))

    data, sample_metadata = load_rollout_samples(args.samples_dir.expanduser())
    target_key = paired_target_key(args.state_key, sample_metadata)
    source_sample_count = int(data[target_key].shape[0])
    data, selected_indices = _select_rows(
        data, max_samples=int(args.max_samples), seed=int(args.seed)
    )
    if args.state_key not in data:
        raise KeyError(f"Samples do not contain state key {args.state_key!r}.")
    if str(sample_metadata.get("interface")) != "per_step_token_sequence":
        raise ValueError(
            "Categorical trainer requires per_step_token_sequence samples."
        )
    encoding = sample_metadata.get("target_encoding")
    if not isinstance(encoding, dict) or encoding.get("kind") != "categorical_sequence":
        raise ValueError("Samples lack categorical_sequence target_encoding metadata.")
    horizon = int(encoding["horizon"])
    codebook_size = int(encoding["codebook_size"])
    target_spec = InterfaceTargetSpec.from_dict(sample_metadata["target_spec"])

    state = data[args.state_key].to(device=device, dtype=torch.float32)
    target = data[target_key].to(device=device, dtype=torch.long)
    if tuple(target.shape[1:]) != (horizon,):
        raise ValueError(f"Token target shape {tuple(target.shape)} != [N, {horizon}].")

    checkpoint_metadata: dict = {}
    if args.checkpoint is not None:
        loaded, checkpoint_spec, checkpoint_metadata = load_planner_checkpoint(
            args.checkpoint.expanduser(), map_location=device
        )
        if checkpoint_spec != target_spec:
            raise ValueError("Checkpoint and sample target specifications differ.")
        if not isinstance(loaded, CausalInterfaceTransformerCategoricalPlanner):
            raise TypeError("Checkpoint is not a categorical causal planner.")
        planner = loaded.to(device)
        init_checkpoint = str(args.checkpoint.expanduser().resolve())
    else:
        planner = CausalInterfaceTransformerCategoricalPlanner(
            state_dim=int(state.shape[-1]),
            token_horizon=horizon,
            codebook_size=codebook_size,
            **MODEL_PRESETS[str(args.model_size)],
        ).to(device)
        planner.set_state_normalization(
            state_mean=state.mean(dim=0),
            state_std=state.std(dim=0, unbiased=False),
        )
        init_checkpoint = None
    if planner.token_horizon != horizon or planner.codebook_size != codebook_size:
        raise ValueError("Planner token shape does not match sample encoding.")

    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = Path("logs", "interface_planner", f"{timestamp}_tokens")
    run_dir = output_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "checkpoints" / "latest.pt"
    metrics_path = run_dir / "metrics.jsonl"
    selected_count = int(state.shape[0])
    old_pretrain_updates = checkpoint_metadata.get(
        "pretrain_num_updates", checkpoint_metadata.get("num_updates")
    )
    finetune = args.state_key == "planner_state"
    metadata = {
        "interface": "per_step_token_sequence",
        "planner_type": planner.planner_type,
        "state_key": args.state_key,
        "samples_dir": str(args.samples_dir.expanduser().resolve()),
        "source_sample_count": source_sample_count,
        "selected_sample_count": selected_count,
        "heldout_sample_count": source_sample_count - selected_count,
        "selected_indices": (
            selected_indices.tolist() if selected_indices is not None else None
        ),
        "selection_seed": int(args.seed),
        "num_updates": int(args.num_updates),
        "pretrain_num_updates": (
            old_pretrain_updates
            if old_pretrain_updates is not None
            else (None if finetune else int(args.num_updates))
        ),
        "finetune_num_updates": int(args.num_updates) if finetune else None,
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "model_size": str(args.model_size),
        "planner_config": planner.config_dict(),
        "planner_observation_spec": sample_metadata.get("planner_observation_spec"),
        "target_encoding": encoding,
        "init_checkpoint": init_checkpoint,
        "sample_metadata": sample_metadata,
        **parameter_counts(planner),
    }
    (run_dir / "config.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8"
    )
    optimizer = torch.optim.AdamW(
        planner.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)
    )
    for update in range(1, int(args.num_updates) + 1):
        indices = torch.randint(
            0, selected_count, (int(args.batch_size),), device=device
        )
        planner.train()
        loss = planner.categorical_loss(
            state.index_select(0, indices), target.index_select(0, indices)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            planner.parameters(), float(args.grad_clip_norm)
        )
        optimizer.step()
        if (
            update == 1
            or update % int(args.log_interval) == 0
            or update == int(args.num_updates)
        ):
            metrics = {
                "update": update,
                "train/loss": float(loss.detach().item()),
                "train/grad_norm": float(grad_norm.detach().item()),
                **_evaluate(planner, state, target),
            }
            write_jsonl(metrics_path, metrics)
            save_planner_checkpoint(
                checkpoint_path,
                planner=planner,
                optimizer=optimizer,
                target_spec=target_spec,
                metadata=metadata | {"last_metrics": metrics},
            )
            print(
                f"[METRIC] update={update} loss={metrics['train/loss']:.6f} "
                f"token_acc={metrics['eval/token_accuracy']:.4f}",
                flush=True,
            )
    print(f"[INFO] Wrote categorical planner checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
