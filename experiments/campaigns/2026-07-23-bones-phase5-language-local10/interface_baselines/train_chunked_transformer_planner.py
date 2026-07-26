#!/usr/bin/env python3
"""Train a strong chunked Transformer flow planner for interface baselines."""

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
    CausalInterfaceTransformerDeterministicPlanner,
    CausalInterfaceTransformerDiffusionPlanner,
    ChunkedTransformerFlowPlanner,
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


MODEL_PRESETS: dict[str, dict[str, int | float]] = {
    "tiny": {
        "d_model": 64,
        "num_layers": 1,
        "num_heads": 4,
        "feedforward_dim": 128,
        "patch_dim": 16,
        "num_state_tokens": 1,
        "dropout": 0.0,
    },
    "small": {
        "d_model": 256,
        "num_layers": 4,
        "num_heads": 4,
        "feedforward_dim": 1024,
        "patch_dim": 32,
        "num_state_tokens": 2,
        "dropout": 0.0,
    },
    "medium": {
        "d_model": 512,
        "num_layers": 6,
        "num_heads": 8,
        "feedforward_dim": 2048,
        "patch_dim": 32,
        "num_state_tokens": 4,
        "dropout": 0.0,
    },
    "large": {
        "d_model": 768,
        "num_layers": 8,
        "num_heads": 12,
        "feedforward_dim": 3072,
        "patch_dim": 32,
        "num_state_tokens": 4,
        "dropout": 0.0,
    },
}

PLANNER_FAMILIES = {
    "flow": ChunkedTransformerFlowPlanner,
    "diffusion": CausalInterfaceTransformerDiffusionPlanner,
    "deterministic": CausalInterfaceTransformerDeterministicPlanner,
}


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
        help="Optional matching planner-family checkpoint used to initialize finetuning.",
    )
    parser.add_argument(
        "--planner_family",
        choices=("flow", "diffusion", "deterministic"),
        default="flow",
        help="Continuous chunk prediction method; the Transformer backbone is shared.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--micro_batch_size",
        type=int,
        default=0,
        help=(
            "Per-forward microbatch used for gradient accumulation. "
            "<=0 uses --batch_size."
        ),
    )
    parser.add_argument("--num_updates", type=int, default=2000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument(
        "--eval_max_samples",
        type=int,
        default=4096,
        help="Maximum samples used for periodic evaluation; <=0 means all.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Randomly subsample this many rollout rows for sample-budget sweeps; <=0 means all.",
    )
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight_decay", type=float, default=1.0e-4)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--flow_loss_coeff", type=float, default=1.0)
    parser.add_argument("--endpoint_loss_coeff", type=float, default=1.0)
    parser.add_argument("--endpoint_cosine_coeff", type=float, default=1.0)
    parser.add_argument(
        "--normalization",
        choices=("per_dim", "none"),
        default="per_dim",
        help="Normalize state and target dimensions before flow training.",
    )
    parser.add_argument(
        "--use_checkpoint_normalization",
        action="store_true",
        default=False,
        help="Keep normalization buffers from --checkpoint instead of recomputing them from samples.",
    )
    parser.add_argument(
        "--model_size",
        choices=tuple(MODEL_PRESETS),
        default="medium",
        help="Architecture preset for new checkpoints.",
    )
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--feedforward_dim", type=int, default=None)
    parser.add_argument("--patch_dim", type=int, default=None)
    parser.add_argument("--num_state_tokens", type=int, default=None)
    parser.add_argument(
        "--num_language_tokens",
        type=int,
        default=1,
        help="Transformer tokens used for the optional language embedding.",
    )
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--flow_num_inference_steps", type=int, default=16)
    parser.add_argument(
        "--endpoint_num_inference_steps",
        type=int,
        default=0,
        help=(
            "Number of flow integration steps used for the training endpoint loss. "
            "<=0 uses --flow_num_inference_steps. Evaluation still uses "
            "--flow_num_inference_steps."
        ),
    )
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
        f"{timestamp}_{interface}_{state_key}_chunked_transformer",
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


def _model_kwargs(args: argparse.Namespace) -> dict[str, int | float]:
    kwargs = dict(MODEL_PRESETS[str(args.model_size)])
    for key in (
        "d_model",
        "num_layers",
        "num_heads",
        "feedforward_dim",
        "patch_dim",
        "num_state_tokens",
        "dropout",
    ):
        value = getattr(args, key)
        if value is not None:
            kwargs[key] = value
    return kwargs


def _select_rows(
    data: dict[str, torch.Tensor], *, max_samples: int, seed: int
) -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:
    num_rows = int(data["causal_target"].shape[0])
    if int(max_samples) <= 0 or int(max_samples) >= num_rows:
        return data, None
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    indices = torch.randperm(num_rows, generator=generator)[: int(max_samples)]
    indices, _ = torch.sort(indices)
    selected = {
        key: value.index_select(0, indices) if value.ndim > 0 else value
        for key, value in data.items()
    }
    return selected, indices


def _normalization_stats(
    state: torch.Tensor, target: torch.Tensor, *, mode: str
) -> dict[str, torch.Tensor]:
    if mode == "none":
        return {
            "state_mean": torch.zeros(state.shape[-1], dtype=torch.float32),
            "state_std": torch.ones(state.shape[-1], dtype=torch.float32),
            "target_mean": torch.zeros(target.shape[-1], dtype=torch.float32),
            "target_std": torch.ones(target.shape[-1], dtype=torch.float32),
        }
    if mode != "per_dim":
        raise ValueError(f"Unsupported normalization mode={mode!r}.")
    return {
        "state_mean": state.float().mean(dim=0).cpu(),
        "state_std": state.float().std(dim=0, unbiased=False).cpu(),
        "target_mean": target.float().mean(dim=0).cpu(),
        "target_std": target.float().std(dim=0, unbiased=False).cpu(),
    }


@torch.no_grad()
def _evaluate(
    planner: ChunkedTransformerFlowPlanner,
    state: torch.Tensor,
    target: torch.Tensor,
    language: torch.Tensor | None,
    *,
    flow_num_inference_steps: int,
    flow_inference_noise_std: float,
    batch_size: int,
    max_samples: int,
) -> dict[str, float]:
    planner.eval()
    if int(max_samples) > 0 and int(max_samples) < int(state.shape[0]):
        eval_state = state[: int(max_samples)]
        eval_target = target[: int(max_samples)]
    else:
        eval_state = state
        eval_target = target
    predictions: list[torch.Tensor] = []
    for start in range(0, int(eval_state.shape[0]), int(batch_size)):
        stop = min(start + int(batch_size), int(eval_state.shape[0]))
        predictions.append(
            planner(
                eval_state[start:stop],
                num_inference_steps=flow_num_inference_steps,
                inference_noise_std=flow_inference_noise_std,
                language=(None if language is None else language[start:stop]),
            )
        )
    prediction = torch.cat(predictions, dim=0)
    rmse_mean, rmse_std = mean_std(rmse_per_row(prediction, eval_target))
    prediction_norm = planner.normalize_target(prediction)
    target_norm = planner.normalize_target(eval_target)
    normalized_rmse_mean, normalized_rmse_std = mean_std(
        rmse_per_row(prediction_norm, target_norm)
    )
    return {
        "eval/target_rmse_mean": rmse_mean,
        "eval/target_rmse_std": rmse_std,
        "eval/target_mse": float(F.mse_loss(prediction, eval_target).item()),
        "eval/target_cosine": cosine_mean(prediction, eval_target),
        "eval/normalized_target_rmse_mean": normalized_rmse_mean,
        "eval/normalized_target_rmse_std": normalized_rmse_std,
        "eval/normalized_target_mse": float(
            F.mse_loss(prediction_norm, target_norm).item()
        ),
        "eval/normalized_target_cosine": cosine_mean(prediction_norm, target_norm),
        "eval/prediction_rms": float(prediction.pow(2).mean().sqrt().item()),
        "eval/target_rms": float(eval_target.pow(2).mean().sqrt().item()),
        "eval/sample_count": int(eval_state.shape[0]),
    }


def main() -> None:
    args = _parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be > 0.")
    if args.micro_batch_size < 0:
        raise ValueError("--micro_batch_size must be >= 0.")
    if args.num_updates <= 0:
        raise ValueError("--num_updates must be > 0.")
    if args.log_interval <= 0:
        raise ValueError("--log_interval must be > 0.")
    if args.eval_batch_size <= 0:
        raise ValueError("--eval_batch_size must be > 0.")
    if args.flow_num_inference_steps <= 0:
        raise ValueError("--flow_num_inference_steps must be > 0.")
    if args.endpoint_num_inference_steps < 0:
        raise ValueError("--endpoint_num_inference_steps must be >= 0.")
    if args.num_language_tokens <= 0:
        raise ValueError("--num_language_tokens must be > 0.")

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    device = _resolve_device(str(args.device))

    data_cpu, sample_metadata = load_rollout_samples(args.samples_dir.expanduser())
    target_key = paired_target_key(args.state_key, sample_metadata)
    source_sample_count = int(data_cpu[target_key].shape[0])
    data_cpu, selected_indices = _select_rows(
        data_cpu, max_samples=int(args.max_samples), seed=int(args.seed)
    )
    if args.state_key not in data_cpu:
        raise KeyError(f"Sample data does not contain state key {args.state_key!r}.")
    target_spec = _target_spec_from_metadata(sample_metadata, args.interface)
    interface = target_spec.interface
    run_dir = _run_dir(args.output_dir, interface=interface, state_key=args.state_key)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    checkpoint_path = run_dir / "checkpoints" / "latest.pt"

    state = data_cpu[args.state_key].to(device=device, dtype=torch.float32)
    target = data_cpu[target_key].to(device=device, dtype=torch.float32)
    language_cpu = data_cpu.get("language_embedding")
    language = (
        None
        if language_cpu is None
        else language_cpu.to(device=device, dtype=torch.float32)
    )
    language_dim = 0 if language is None else int(language.shape[-1])
    if int(target.shape[-1]) != target_spec.target_dim:
        raise ValueError(
            f"Target width mismatch: sample target has {target.shape[-1]}, "
            f"metadata target_spec has {target_spec.target_dim}."
        )

    model_kwargs = _model_kwargs(args)
    planner_class = PLANNER_FAMILIES[str(args.planner_family)]
    if args.checkpoint is not None:
        loaded_planner, checkpoint_spec, checkpoint_metadata = load_planner_checkpoint(
            args.checkpoint.expanduser(), map_location=device
        )
        if checkpoint_spec != target_spec:
            raise ValueError(
                "Checkpoint target spec does not match sample target spec: "
                f"{checkpoint_spec.to_dict()} vs {target_spec.to_dict()}."
            )
        if type(loaded_planner) is not planner_class:
            raise TypeError(
                "--checkpoint planner family does not match --planner_family: "
                f"{loaded_planner.config_dict().get('planner_type')} vs "
                f"{args.planner_family}."
            )
        planner = loaded_planner.to(device)
        init_checkpoint = str(args.checkpoint.expanduser().resolve())
    else:
        checkpoint_metadata = {}
        planner = planner_class(
            state_dim=int(state.shape[-1]),
            target_dim=target_spec.target_dim,
            term_widths=target_spec.term_widths,
            language_dim=language_dim,
            num_language_tokens=int(args.num_language_tokens),
            **model_kwargs,
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
    if int(planner.language_dim) != language_dim:
        raise ValueError(
            f"Planner language_dim={planner.language_dim} does not match samples "
            f"{language_dim}."
        )
    if not args.use_checkpoint_normalization:
        stats = _normalization_stats(
            state.detach().cpu(), target.detach().cpu(), mode=args.normalization
        )
        planner.set_normalization(**stats)

    optimizer = torch.optim.AdamW(
        planner.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    selected_index_list = (
        selected_indices.detach().cpu().tolist()
        if selected_indices is not None
        else None
    )
    pretrain_num_updates = None
    if isinstance(checkpoint_metadata, dict):
        pretrain_num_updates = checkpoint_metadata.get(
            "pretrain_num_updates", checkpoint_metadata.get("num_updates")
        )
    is_finetune_stage = str(args.state_key) == "planner_state"
    selected_sample_count = int(state.shape[0])
    micro_batch_size = (
        int(args.batch_size)
        if int(args.micro_batch_size) <= 0
        else min(int(args.micro_batch_size), int(args.batch_size))
    )
    endpoint_num_inference_steps = (
        int(args.flow_num_inference_steps)
        if int(args.endpoint_num_inference_steps) <= 0
        else int(args.endpoint_num_inference_steps)
    )
    heldout_sample_count = (
        source_sample_count - selected_sample_count
        if selected_indices is not None
        else 0
    )
    metadata = {
        "interface": interface,
        "planner_type": str(planner.config_dict()["planner_type"]),
        "planner_family": str(args.planner_family),
        "state_key": args.state_key,
        "samples_dir": str(args.samples_dir.expanduser().resolve()),
        "source_sample_count": source_sample_count,
        "num_samples": selected_sample_count,
        "selected_sample_count": selected_sample_count,
        "heldout_sample_count": heldout_sample_count,
        "max_samples": int(args.max_samples),
        "batch_size": int(args.batch_size),
        "micro_batch_size": micro_batch_size,
        "gradient_accumulation_steps": int(
            (int(args.batch_size) + micro_batch_size - 1) // micro_batch_size
        ),
        "num_updates": int(args.num_updates),
        "pretrain_num_updates": pretrain_num_updates
        if pretrain_num_updates not in (None, "")
        else (None if is_finetune_stage else int(args.num_updates)),
        "finetune_num_updates": int(args.num_updates) if is_finetune_stage else None,
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "flow_num_inference_steps": int(args.flow_num_inference_steps),
        "flow_inference_noise_std": float(args.flow_inference_noise_std),
        "endpoint_num_inference_steps": endpoint_num_inference_steps,
        "selected_indices": selected_index_list,
        "selection_seed": int(args.seed),
        "selection_fraction": selected_sample_count / max(source_sample_count, 1),
        "state_dim": int(state.shape[-1]),
        "target_dim": int(target.shape[-1]),
        "language_dim": language_dim,
        "num_language_tokens": int(planner.num_language_tokens),
        "model_size": str(args.model_size),
        "model_kwargs": model_kwargs,
        "planner_config": planner.config_dict(),
        **parameter_counts(planner),
        "init_checkpoint": init_checkpoint,
        "checkpoint_metadata": checkpoint_metadata,
        "args": vars(args),
        "sample_metadata": sample_metadata,
    }
    (run_dir / "config.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(
        f"[INFO] Loaded {state.shape[0]} samples from {args.samples_dir}.", flush=True
    )
    print(
        f"[INFO] Training {interface} {args.planner_family} Transformer planner "
        f"with state_key={args.state_key}.",
        flush=True,
    )
    print(f"[INFO] Output: {run_dir}", flush=True)
    print(f"[INFO] Planner config: {planner.config_dict()}", flush=True)
    if micro_batch_size != int(args.batch_size):
        print(
            "[INFO] Using gradient accumulation: "
            f"batch_size={args.batch_size}, micro_batch_size={micro_batch_size}",
            flush=True,
        )
    if endpoint_num_inference_steps != int(args.flow_num_inference_steps):
        print(
            "[INFO] Using shorter training endpoint integration: "
            f"endpoint_num_inference_steps={endpoint_num_inference_steps}, "
            f"eval_flow_num_inference_steps={args.flow_num_inference_steps}",
            flush=True,
        )

    num_samples = int(state.shape[0])
    for update in range(1, int(args.num_updates) + 1):
        indices = torch.randint(
            low=0,
            high=num_samples,
            size=(int(args.batch_size),),
            device=device,
        )
        planner.train()
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        flow_loss_value = 0.0
        endpoint_mse_value = 0.0
        endpoint_cosine_value = 0.0
        for start in range(0, int(args.batch_size), micro_batch_size):
            stop = min(start + micro_batch_size, int(args.batch_size))
            micro_indices = indices[start:stop]
            batch_state = state.index_select(0, micro_indices)
            batch_target = target.index_select(0, micro_indices)
            batch_language = (
                None if language is None else language.index_select(0, micro_indices)
            )
            if args.planner_family == "flow":
                objective_loss = planner.flow_matching_loss(
                    batch_state,
                    batch_target,
                    language=batch_language,
                )
            elif args.planner_family == "diffusion":
                assert isinstance(planner, CausalInterfaceTransformerDiffusionPlanner)
                objective_loss = planner.diffusion_loss(
                    batch_state,
                    batch_target,
                    language=batch_language,
                )
            else:
                assert isinstance(
                    planner, CausalInterfaceTransformerDeterministicPlanner
                )
                objective_loss = planner.deterministic_loss(
                    batch_state,
                    batch_target,
                    language=batch_language,
                )
            endpoint_prediction = planner(
                batch_state,
                num_inference_steps=endpoint_num_inference_steps,
                inference_noise_std=float(args.flow_inference_noise_std),
                language=batch_language,
            )
            endpoint_mse, endpoint_cosine = planner.normalized_endpoint_loss(
                endpoint_prediction, batch_target
            )
            micro_loss = float(args.flow_loss_coeff) * objective_loss + float(
                args.endpoint_loss_coeff
            ) * (endpoint_mse + float(args.endpoint_cosine_coeff) * endpoint_cosine)
            scale = float(stop - start) / float(args.batch_size)
            (micro_loss * scale).backward()
            loss_value += float(micro_loss.detach().item()) * scale
            flow_loss_value += float(objective_loss.detach().item()) * scale
            endpoint_mse_value += float(endpoint_mse.detach().item()) * scale
            endpoint_cosine_value += float(endpoint_cosine.detach().item()) * scale
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
                language,
                flow_num_inference_steps=int(args.flow_num_inference_steps),
                flow_inference_noise_std=float(args.flow_inference_noise_std),
                batch_size=int(args.eval_batch_size),
                max_samples=int(args.eval_max_samples),
            )
            row = {
                "update": update,
                "train/loss": loss_value,
                "train/objective_loss": flow_loss_value,
                "train/flow_loss": flow_loss_value
                if args.planner_family == "flow"
                else None,
                "train/normalized_endpoint_mse": endpoint_mse_value,
                "train/normalized_endpoint_cosine_loss": endpoint_cosine_value,
                "train/grad_norm": float(grad_norm.detach().item()),
                **eval_metrics,
            }
            write_jsonl(metrics_path, row)
            print(
                "[METRIC] "
                f"update={update} loss={row['train/loss']:.6f} "
                f"eval_rmse={row['eval/target_rmse_mean']:.6f} "
                f"eval_norm_rmse={row['eval/normalized_target_rmse_mean']:.6f} "
                f"eval_cos={row['eval/target_cosine']:.6f}",
                flush=True,
            )
            save_planner_checkpoint(
                checkpoint_path,
                planner=planner,
                optimizer=optimizer,
                target_spec=target_spec,
                metadata=metadata | {"last_metrics": row},
            )

    print(f"[INFO] Wrote planner checkpoint: {checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()
