#!/usr/bin/env python3
"""Train or finetune a flow planner for hand-designed command interfaces."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))

from interface_planner_common import (  # noqa: E402
    InterfaceFlowPlanner,
    InterfaceTargetSpec,
    cosine_mean,
    load_planner_checkpoint,
    load_rollout_samples,
    mean_std,
    paired_target_key,
    parameter_counts,
    rmse_per_row,
    save_planner_checkpoint,
    supported_interfaces,
    write_jsonl,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument(
        "--interface",
        choices=supported_interfaces(),
        default=None,
        help="Interface name. Defaults to the sample metadata value.",
    )
    parser.add_argument(
        "--state_key",
        choices=("expert_planner_state", "planner_state"),
        default="planner_state",
        help="Use expert states for offline pretrain or achieved states for rollout finetune.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional planner checkpoint used to initialize finetuning.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_updates", type=int, default=2000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight_decay", type=float, default=1.0e-4)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--flow_loss_coeff", type=float, default=1.0)
    parser.add_argument("--endpoint_loss_coeff", type=float, default=1.0)
    parser.add_argument("--endpoint_cosine_coeff", type=float, default=1.0)
    parser.add_argument("--hidden_dims", type=str, default="512,512,256")
    parser.add_argument("--activation", type=str, default="mish")
    parser.add_argument("--flow_num_inference_steps", type=int, default=16)
    parser.add_argument("--flow_inference_noise_std", type=float, default=0.0)
    return parser.parse_args()


def _resolve_device(device: str) -> torch.device:
    if device.strip().lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _run_dir(output_dir: Path | None, *, interface: str, state_key: str) -> Path:
    if output_dir is not None:
        return output_dir.expanduser().resolve()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return Path(
        "logs",
        "interface_planner",
        f"{timestamp}_{interface}_{state_key}",
    ).resolve()


def _target_spec_from_metadata(
    metadata: dict[str, Any], interface: str | None
) -> InterfaceTargetSpec:
    if "target_spec" not in metadata:
        raise KeyError("Rollout samples are missing metadata.target_spec.")
    spec = InterfaceTargetSpec.from_dict(metadata["target_spec"])
    if interface is not None and spec.interface != interface:
        raise ValueError(
            f"Sample interface {spec.interface!r} does not match --interface={interface!r}."
        )
    return spec


def _hidden_dims(raw: str) -> tuple[int, ...]:
    dims = tuple(int(chunk.strip()) for chunk in raw.split(",") if chunk.strip())
    if not dims:
        raise ValueError("--hidden_dims must contain at least one integer.")
    return dims


@torch.no_grad()
def _evaluate(
    planner: InterfaceFlowPlanner,
    state: torch.Tensor,
    target: torch.Tensor,
    *,
    flow_num_inference_steps: int,
    flow_inference_noise_std: float,
) -> dict[str, float]:
    planner.eval()
    prediction = planner(
        state,
        num_inference_steps=flow_num_inference_steps,
        inference_noise_std=flow_inference_noise_std,
    )
    rmse_mean, rmse_std = mean_std(rmse_per_row(prediction, target))
    return {
        "eval/target_rmse_mean": rmse_mean,
        "eval/target_rmse_std": rmse_std,
        "eval/target_mse": float(F.mse_loss(prediction, target).item()),
        "eval/target_cosine": cosine_mean(prediction, target),
        "eval/prediction_rms": float(prediction.pow(2).mean().sqrt().item()),
        "eval/target_rms": float(target.pow(2).mean().sqrt().item()),
    }


def main() -> None:
    args = _parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be > 0.")
    if args.num_updates <= 0:
        raise ValueError("--num_updates must be > 0.")
    if args.log_interval <= 0:
        raise ValueError("--log_interval must be > 0.")

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    device = _resolve_device(str(args.device))

    data_cpu, sample_metadata = load_rollout_samples(args.samples_dir.expanduser())
    target_key = paired_target_key(args.state_key, sample_metadata)
    source_sample_count = int(data_cpu[target_key].shape[0])
    if args.state_key not in data_cpu:
        raise KeyError(f"Sample data does not contain state key {args.state_key!r}.")
    target_spec = _target_spec_from_metadata(sample_metadata, args.interface)
    planner_observation_spec = sample_metadata.get("planner_observation_spec")
    if not isinstance(planner_observation_spec, dict):
        raise ValueError(
            "Rollout samples have no causal planner_observation_spec. "
            "Recollect them with robot-only planner observations."
        )
    interface = target_spec.interface
    run_dir = _run_dir(args.output_dir, interface=interface, state_key=args.state_key)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    checkpoint_path = run_dir / "checkpoints" / "latest.pt"

    state = data_cpu[args.state_key].to(device=device, dtype=torch.float32)
    target = data_cpu[target_key].to(device=device, dtype=torch.float32)
    if int(target.shape[-1]) != target_spec.target_dim:
        raise ValueError(
            f"Target width mismatch: sample target has {target.shape[-1]}, "
            f"metadata target_spec has {target_spec.target_dim}."
        )
    if int(state.shape[-1]) != int(planner_observation_spec.get("flat_dim", -1)):
        raise ValueError(
            f"Sample state width {state.shape[-1]} does not match causal planner "
            f"spec {planner_observation_spec.get('flat_dim')}."
        )

    if args.checkpoint is not None:
        planner, checkpoint_spec, checkpoint_metadata = load_planner_checkpoint(
            args.checkpoint.expanduser(), map_location=device
        )
        if checkpoint_spec != target_spec:
            raise ValueError(
                "Checkpoint target spec does not match sample target spec: "
                f"{checkpoint_spec.to_dict()} vs {target_spec.to_dict()}."
            )
        checkpoint_observation_spec = checkpoint_metadata.get(
            "planner_observation_spec"
        )
        if checkpoint_observation_spec is None:
            checkpoint_sample_metadata = checkpoint_metadata.get("sample_metadata", {})
            if isinstance(checkpoint_sample_metadata, dict):
                checkpoint_observation_spec = checkpoint_sample_metadata.get(
                    "planner_observation_spec"
                )
        if checkpoint_observation_spec != planner_observation_spec:
            raise ValueError(
                "Checkpoint and sample planner observation specifications differ: "
                f"{checkpoint_observation_spec} != {planner_observation_spec}."
            )
        planner = planner.to(device)
        init_checkpoint = str(args.checkpoint.expanduser().resolve())
    else:
        checkpoint_metadata = {}
        planner = InterfaceFlowPlanner(
            state_dim=int(state.shape[-1]),
            target_dim=target_spec.target_dim,
            hidden_dims=_hidden_dims(args.hidden_dims),
            activation=str(args.activation),
        ).to(device)
        init_checkpoint = None
    if int(planner.state_dim) != int(state.shape[-1]):
        raise ValueError(
            f"Planner state_dim={planner.state_dim} does not match samples {state.shape[-1]}."
        )
    if int(planner.target_dim) != target_spec.target_dim:
        raise ValueError(
            f"Planner target_dim={planner.target_dim} does not match target spec {target_spec.target_dim}."
        )

    optimizer = torch.optim.AdamW(
        planner.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    pretrain_num_updates = None
    if isinstance(checkpoint_metadata, dict):
        pretrain_num_updates = checkpoint_metadata.get(
            "pretrain_num_updates", checkpoint_metadata.get("num_updates")
        )
    is_finetune_stage = str(args.state_key) == "planner_state"
    metadata = {
        "interface": interface,
        "planner_type": "mlp_flow",
        "state_key": args.state_key,
        "samples_dir": str(args.samples_dir.expanduser().resolve()),
        "source_sample_count": source_sample_count,
        "num_samples": int(state.shape[0]),
        "selected_sample_count": int(state.shape[0]),
        "heldout_sample_count": 0,
        "batch_size": int(args.batch_size),
        "num_updates": int(args.num_updates),
        "pretrain_num_updates": pretrain_num_updates
        if pretrain_num_updates not in (None, "")
        else (None if is_finetune_stage else int(args.num_updates)),
        "finetune_num_updates": int(args.num_updates) if is_finetune_stage else None,
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "flow_num_inference_steps": int(args.flow_num_inference_steps),
        "flow_inference_noise_std": float(args.flow_inference_noise_std),
        "state_dim": int(state.shape[-1]),
        "target_dim": int(target.shape[-1]),
        "planner_config": planner.config_dict(),
        "planner_observation_spec": planner_observation_spec,
        **parameter_counts(planner),
        "init_checkpoint": init_checkpoint,
        "checkpoint_metadata": checkpoint_metadata,
        "args": vars(args),
        "sample_metadata": sample_metadata,
    }
    (run_dir / "config.yaml").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(f"[INFO] Loaded {state.shape[0]} samples from {args.samples_dir}.")
    print(f"[INFO] Training {interface} planner with state_key={args.state_key}.")
    print(f"[INFO] Output: {run_dir}")

    num_samples = int(state.shape[0])
    for update in range(1, int(args.num_updates) + 1):
        indices = torch.randint(
            low=0,
            high=num_samples,
            size=(int(args.batch_size),),
            device=device,
        )
        batch_state = state.index_select(0, indices)
        batch_target = target.index_select(0, indices)
        planner.train()
        flow_loss = planner.flow_matching_loss(batch_state, batch_target)
        endpoint_prediction = planner(
            batch_state,
            num_inference_steps=int(args.flow_num_inference_steps),
            inference_noise_std=float(args.flow_inference_noise_std),
        )
        endpoint_mse = F.mse_loss(endpoint_prediction, batch_target)
        endpoint_cosine = (
            1.0 - F.cosine_similarity(endpoint_prediction, batch_target, dim=-1).mean()
        )
        loss = float(args.flow_loss_coeff) * flow_loss + float(
            args.endpoint_loss_coeff
        ) * (endpoint_mse + float(args.endpoint_cosine_coeff) * endpoint_cosine)
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
            eval_metrics = _evaluate(
                planner,
                state,
                target,
                flow_num_inference_steps=int(args.flow_num_inference_steps),
                flow_inference_noise_std=float(args.flow_inference_noise_std),
            )
            row = {
                "update": update,
                "train/loss": float(loss.detach().item()),
                "train/flow_loss": float(flow_loss.detach().item()),
                "train/endpoint_mse": float(endpoint_mse.detach().item()),
                "train/endpoint_cosine_loss": float(endpoint_cosine.detach().item()),
                "train/grad_norm": float(grad_norm.detach().item()),
                **eval_metrics,
            }
            write_jsonl(metrics_path, row)
            print(
                "[METRIC] "
                f"update={update} loss={row['train/loss']:.6f} "
                f"eval_rmse={row['eval/target_rmse_mean']:.6f} "
                f"eval_cos={row['eval/target_cosine']:.6f}"
            )
            save_planner_checkpoint(
                checkpoint_path,
                planner=planner,
                optimizer=optimizer,
                target_spec=target_spec,
                metadata=metadata | {"last_metrics": row},
            )

    print(f"[INFO] Wrote planner checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
