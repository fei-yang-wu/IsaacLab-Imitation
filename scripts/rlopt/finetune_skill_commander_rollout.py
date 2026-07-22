#!/usr/bin/env python3
"""Finetune a SkillCommander on saved achieved-state rollout samples."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from rlopt.agent.skill_commander import (
    DiffusionSkillCommander,
    FlowMatchingSkillCommander,
    _build_skill_commander_generator_from_checkpoint,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finetune a SkillCommander from saved closed-loop rollout tensors."
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Base commander checkpoint."
    )
    parser.add_argument(
        "--samples_dir",
        required=True,
        help="Directory containing sample_step_*.pt tensors from closed-loop eval.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory. Defaults to logs/skill_commander_rollout_finetune/<timestamp>.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_updates", type=int, default=2000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument(
        "--flow_loss_coeff",
        type=float,
        default=1.0,
        help="Weight for flow/diffusion training loss when applicable.",
    )
    parser.add_argument(
        "--endpoint_loss_coeff",
        type=float,
        default=1.0,
        help="Weight for deterministic endpoint MSE/cosine loss.",
    )
    parser.add_argument(
        "--endpoint_cosine_coeff",
        type=float,
        default=1.0,
        help="Weight on 1-cosine inside endpoint loss.",
    )
    parser.add_argument(
        "--flow_num_inference_steps",
        type=int,
        default=None,
        help="Override flow inference steps for endpoint loss/eval.",
    )
    parser.add_argument(
        "--flow_inference_noise_std",
        type=float,
        default=0.0,
        help="Override flow inference noise std for endpoint loss/eval.",
    )
    return parser.parse_args()


def _run_dir(output_dir: str | None) -> Path:
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return Path("logs", "skill_commander_rollout_finetune", timestamp).resolve()


def _resolve_device(device: str) -> torch.device:
    if device.strip().lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _load_samples(samples_dir: Path) -> dict[str, torch.Tensor]:
    samples_dir = samples_dir.expanduser()
    paths = sorted(samples_dir.glob("sample_step_*.pt")) + sorted(
        samples_dir.glob("sample_chunk_*.pt")
    )
    if not paths:
        raise FileNotFoundError(
            f"No sample_step_*.pt or sample_chunk_*.pt files found in {samples_dir}."
        )
    rows: dict[str, list[torch.Tensor]] = {
        "planner_state": [],
        "expert_planner_state": [],
        "lang": [],
        "z_target": [],
        "traj_rank": [],
    }
    steps: list[int] = []
    for path in paths:
        sample = torch.load(path, map_location="cpu", weights_only=False)
        sample_values: dict[str, Any] = {
            "planner_state": sample.get("planner_state"),
            "expert_planner_state": sample.get("expert_planner_state"),
            "lang": sample.get("lang", sample.get("language_embedding")),
            "z_target": sample.get("z_target", sample.get("demonstration_target")),
            "traj_rank": sample.get("traj_rank", sample.get("trajectory_rank")),
        }
        for key in (
            "planner_state",
            "expert_planner_state",
            "z_target",
            "traj_rank",
        ):
            if sample_values[key] is None:
                raise KeyError(f"Sample {path} is missing required key {key!r}.")

        sample_rows: dict[str, torch.Tensor] = {}
        for key, value in sample_values.items():
            if value is None and key == "lang":
                row_count = int(sample_values["planner_state"].shape[0])
                value = torch.zeros((row_count, 0), dtype=torch.float32)
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"Sample {path} key {key!r} must be a tensor, got {type(value).__name__}."
                )
            if key == "traj_rank":
                value = value.reshape(-1)
            elif value.ndim == 1:
                value = value.unsqueeze(0)
            if key != "traj_rank" and value.ndim != 2:
                raise ValueError(
                    f"Sample {path} key {key!r} must be rank-2, got {tuple(value.shape)}."
                )
            dtype = torch.long if key == "traj_rank" else torch.float32
            sample_rows[key] = value.to(dtype=dtype)
        row_count = int(sample_rows["planner_state"].shape[0])
        for key, value in sample_rows.items():
            if int(value.shape[0]) != row_count:
                raise ValueError(
                    f"Sample {path} key {key!r} row count {value.shape[0]} "
                    f"does not match planner_state row count {row_count}."
                )
            rows[key].append(value)
        step = sample.get("step")
        if step is None:
            raise KeyError(f"Sample {path} is missing required key 'step'.")
        if isinstance(step, torch.Tensor):
            if step.numel() == 1:
                step_value = int(step.item())
                steps.extend([step_value] * row_count)
            elif int(step.numel()) == row_count:
                steps.extend([int(value) for value in step.reshape(-1).tolist()])
            else:
                raise ValueError(
                    f"Sample {path} step tensor must contain one value or {row_count} "
                    f"values, got {tuple(step.shape)}."
                )
        else:
            steps.extend([int(step)] * row_count)
    data = {key: torch.cat(value, dim=0).contiguous() for key, value in rows.items()}
    data["step"] = torch.as_tensor(steps, dtype=torch.long)
    return data


def _write_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


@torch.no_grad()
def _evaluate(
    generator: torch.nn.Module, data: dict[str, torch.Tensor]
) -> dict[str, float]:
    generator.eval()
    z_hat = generator(data["planner_state"], data["lang"])
    z_target = data["z_target"]
    return {
        "eval/z_cosine": float(
            F.cosine_similarity(z_hat, z_target, dim=-1).mean().item()
        ),
        "eval/z_mse": float(F.mse_loss(z_hat, z_target).item()),
        "eval/z_hat_rms": float(z_hat.pow(2).mean().sqrt().item()),
        "eval/z_target_rms": float(z_target.pow(2).mean().sqrt().item()),
    }


def _loss(
    generator: torch.nn.Module,
    state: torch.Tensor,
    lang: torch.Tensor,
    z_target: torch.Tensor,
    *,
    flow_loss_coeff: float,
    endpoint_loss_coeff: float,
    endpoint_cosine_coeff: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    loss = z_target.new_zeros(())
    metrics: dict[str, float] = {}
    if isinstance(generator, FlowMatchingSkillCommander):
        flow_loss, flow_metrics = generator.flow_matching_loss(state, lang, z_target)
        loss = loss + float(flow_loss_coeff) * flow_loss
        metrics.update({f"train/{key}": value for key, value in flow_metrics.items()})
        metrics["train/flow_loss"] = float(flow_loss.detach().item())
    elif isinstance(generator, DiffusionSkillCommander):
        diffusion_loss, diffusion_metrics = generator.diffusion_loss(
            state, lang, z_target
        )
        loss = loss + float(flow_loss_coeff) * diffusion_loss
        metrics.update(
            {f"train/{key}": value for key, value in diffusion_metrics.items()}
        )
        metrics["train/diffusion_loss"] = float(diffusion_loss.detach().item())

    z_hat = generator(state, lang)
    mse_loss = F.mse_loss(z_hat, z_target)
    cosine_loss = 1.0 - F.cosine_similarity(z_hat, z_target, dim=-1).mean()
    endpoint_loss = mse_loss + float(endpoint_cosine_coeff) * cosine_loss
    loss = loss + float(endpoint_loss_coeff) * endpoint_loss
    metrics.update(
        {
            "train/endpoint_loss": float(endpoint_loss.detach().item()),
            "train/endpoint_mse_loss": float(mse_loss.detach().item()),
            "train/endpoint_cosine_loss": float(cosine_loss.detach().item()),
            "train/z_cosine": float(
                F.cosine_similarity(z_hat, z_target, dim=-1).mean().detach().item()
            ),
            "train/z_mse": float(mse_loss.detach().item()),
        }
    )
    return loss, metrics


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
    log_dir = _run_dir(args.output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = log_dir / "metrics.jsonl"
    checkpoint_path = log_dir / "checkpoints" / "latest.pt"

    checkpoint = torch.load(
        Path(args.checkpoint).expanduser(),
        map_location="cpu",
        weights_only=False,
    )
    samples = _load_samples(Path(args.samples_dir))
    data = {key: value.to(device=device) for key, value in samples.items()}
    config_overrides = {
        "flow_num_inference_steps": args.flow_num_inference_steps,
        "flow_inference_noise_std": args.flow_inference_noise_std,
    }
    generator = _build_skill_commander_generator_from_checkpoint(
        checkpoint,
        state_dim=int(data["planner_state"].shape[-1]),
        lang_embed_dim=int(data["lang"].shape[-1]),
        z_dim=int(data["z_target"].shape[-1]),
        config_overrides=config_overrides,
    ).to(device)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.train()

    optimizer = torch.optim.AdamW(
        generator.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    config_payload = {
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "samples_dir": str(Path(args.samples_dir).expanduser().resolve()),
        "num_samples": int(data["planner_state"].shape[0]),
        "state_dim": int(data["planner_state"].shape[-1]),
        "lang_embed_dim": int(data["lang"].shape[-1]),
        "z_dim": int(data["z_target"].shape[-1]),
        "args": vars(args),
    }
    (log_dir / "config.yaml").write_text(
        yaml.safe_dump(config_payload, sort_keys=True), encoding="utf-8"
    )
    print(f"[INFO] Loaded {config_payload['num_samples']} rollout samples.")
    print(f"[INFO] Logging rollout finetune to: {log_dir}")

    num_samples = int(data["planner_state"].shape[0])
    for update in range(1, int(args.num_updates) + 1):
        indices = torch.randint(
            low=0,
            high=num_samples,
            size=(int(args.batch_size),),
            device=device,
        )
        state = data["planner_state"].index_select(0, indices)
        lang = data["lang"].index_select(0, indices)
        z_target = data["z_target"].index_select(0, indices)
        generator.train()
        loss, metrics = _loss(
            generator,
            state,
            lang,
            z_target,
            flow_loss_coeff=float(args.flow_loss_coeff),
            endpoint_loss_coeff=float(args.endpoint_loss_coeff),
            endpoint_cosine_coeff=float(args.endpoint_cosine_coeff),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if float(args.grad_clip_norm) > 0.0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                generator.parameters(),
                max_norm=float(args.grad_clip_norm),
            )
            metrics["train/grad_norm"] = float(grad_norm.item())
        optimizer.step()

        if (
            update == 1
            or update == int(args.num_updates)
            or update % int(args.log_interval) == 0
        ):
            row = {
                "update": int(update),
                "train/loss": float(loss.detach().item()),
                **metrics,
                **_evaluate(generator, data),
            }
            _write_jsonl(metrics_path, row)
            print(json.dumps(row, sort_keys=True))

    output_checkpoint = dict(checkpoint)
    config = dict(output_checkpoint.get("config", {}))
    if args.flow_num_inference_steps is not None:
        config["flow_num_inference_steps"] = int(args.flow_num_inference_steps)
    config["flow_inference_noise_std"] = float(args.flow_inference_noise_std)
    output_checkpoint["config"] = config
    output_checkpoint["generator_state_dict"] = generator.state_dict()
    output_checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    output_checkpoint["update"] = int(checkpoint.get("update", 0)) + int(
        args.num_updates
    )
    output_checkpoint["rollout_finetune"] = config_payload
    checkpoint_metadata = output_checkpoint.get("metadata")
    if not isinstance(checkpoint_metadata, dict):
        checkpoint_metadata = {}
    pretrain_num_updates = checkpoint_metadata.get("pretrain_num_updates")
    if pretrain_num_updates in (None, ""):
        pretrain_num_updates = checkpoint_metadata.get("num_updates")
    if pretrain_num_updates in (None, ""):
        pretrain_num_updates = int(checkpoint.get("update", 0))
    checkpoint_metadata.update(
        {
            "interface": "latent_skill",
            "planner_type": config.get("planner_type", "skill_commander"),
            "state_key": "planner_state",
            "source_sample_count": int(num_samples),
            "num_samples": int(num_samples),
            "selected_sample_count": int(num_samples),
            "heldout_sample_count": 0,
            "batch_size": int(args.batch_size),
            "num_updates": int(args.num_updates),
            "pretrain_num_updates": int(pretrain_num_updates),
            "finetune_num_updates": int(args.num_updates),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "flow_inference_noise_std": float(args.flow_inference_noise_std),
            "state_dim": int(data["planner_state"].shape[-1]),
            "target_dim": int(data["z_target"].shape[-1]),
        }
    )
    if args.flow_num_inference_steps is not None:
        checkpoint_metadata["flow_num_inference_steps"] = int(
            args.flow_num_inference_steps
        )
    output_checkpoint["metadata"] = checkpoint_metadata
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_checkpoint, checkpoint_path)
    summary = {
        **config_payload,
        "output_checkpoint": str(checkpoint_path),
        "final_metrics": _evaluate(generator, data),
    }
    (log_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[INFO] Saved checkpoint: {checkpoint_path}")
    print(json.dumps(summary["final_metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
