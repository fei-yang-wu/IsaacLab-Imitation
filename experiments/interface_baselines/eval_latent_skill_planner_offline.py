#!/usr/bin/env python3
"""Evaluate a SkillCommander planner on saved latent rollout samples."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from rlopt.agent.skill_commander import _build_skill_commander_generator_from_checkpoint

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))

from interface_planner_common import mean_std, rmse_per_row  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples_dir", type=Path, required=True)
    parser.add_argument("--planner_checkpoint", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, default=None)
    parser.add_argument(
        "--state_key",
        choices=("expert_planner_state", "planner_state"),
        default="expert_planner_state",
    )
    parser.add_argument("--setting", type=str, default="")
    parser.add_argument("--label", type=str, default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--flow_num_inference_steps", type=int, default=None)
    parser.add_argument("--flow_inference_noise_std", type=float, default=None)
    return parser.parse_args()


def _resolve_device(device: str) -> torch.device:
    if device.strip().lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _load_samples(samples_dir: Path) -> dict[str, torch.Tensor]:
    paths = sorted(samples_dir.expanduser().glob("sample_step_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No sample_step_*.pt files found in {samples_dir}.")
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
        sample_rows: dict[str, torch.Tensor] = {}
        for key in rows:
            if key not in sample:
                raise KeyError(f"Sample {path} is missing required key {key!r}.")
            value = sample[key]
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
            if step.numel() != 1:
                raise ValueError(
                    f"Sample {path} step tensor must contain one value, got {tuple(step.shape)}."
                )
            step = step.item()
        steps.extend([int(step)] * row_count)
    data = {key: torch.cat(value, dim=0).contiguous() for key, value in rows.items()}
    data["step"] = torch.as_tensor(steps, dtype=torch.long)
    return data


def _sample_indices(*, num_rows: int, max_samples: int, seed: int) -> torch.Tensor:
    indices = torch.arange(int(num_rows), dtype=torch.long)
    if int(max_samples) > 0 and int(max_samples) < int(num_rows):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        indices = indices[
            torch.randperm(int(num_rows), generator=generator)[: int(max_samples)]
        ]
    indices, _ = torch.sort(indices)
    return indices


@torch.no_grad()
def _predict_batches(
    generator: torch.nn.Module,
    state: torch.Tensor,
    lang: torch.Tensor,
    *,
    batch_size: int,
) -> torch.Tensor:
    predictions: list[torch.Tensor] = []
    for start in range(0, int(state.shape[0]), int(batch_size)):
        stop = min(start + int(batch_size), int(state.shape[0]))
        predictions.append(generator(state[start:stop], lang[start:stop]).detach())
    return torch.cat(predictions, dim=0)


def _metric_stats(values: torch.Tensor) -> dict[str, float | int]:
    mean, std = mean_std(values)
    return {"mean": mean, "std": std, "count": int(values.numel())}


def _metrics(
    prediction: torch.Tensor, target: torch.Tensor
) -> dict[str, dict[str, float | int]]:
    return {
        "planner_target_rmse": _metric_stats(rmse_per_row(prediction, target)),
        "planner_target_mse": {
            "mean": float(F.mse_loss(prediction, target).item()),
            "std": 0.0,
            "count": int(prediction.shape[0]),
        },
        "planner_target_cosine": {
            "mean": float(
                F.cosine_similarity(prediction, target, dim=-1).mean().item()
            ),
            "std": 0.0,
            "count": int(prediction.shape[0]),
        },
        "planner_prediction_rms": {
            "mean": float(prediction.pow(2).mean().sqrt().item()),
            "std": 0.0,
            "count": int(prediction.shape[0]),
        },
        "planner_target_rms": {
            "mean": float(target.pow(2).mean().sqrt().item()),
            "std": 0.0,
            "count": int(prediction.shape[0]),
        },
    }


def _write_csv(summary: dict[str, Any], output_csv: Path) -> None:
    row: dict[str, Any] = {}
    row.update(summary["metadata"])
    row.update(summary["aggregate"])
    for metric_name, stats in summary["metrics"].items():
        for stat_name, value in stats.items():
            row[f"{metric_name}_{stat_name}"] = value
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def evaluate_latent_skill_planner_checkpoint(
    *,
    samples_dir: Path,
    planner_checkpoint: Path,
    output_json: Path,
    output_csv: Path | None = None,
    state_key: str = "expert_planner_state",
    setting: str = "",
    label: str = "",
    device: str = "auto",
    batch_size: int = 512,
    max_samples: int = 0,
    seed: int = 0,
    flow_num_inference_steps: int | None = None,
    flow_inference_noise_std: float | None = None,
) -> dict[str, Any]:
    if int(batch_size) <= 0:
        raise ValueError("--batch_size must be positive.")
    device_obj = _resolve_device(str(device))
    samples = _load_samples(samples_dir)
    if state_key not in samples:
        raise KeyError(f"Sample data does not contain state key {state_key!r}.")
    checkpoint = torch.load(
        planner_checkpoint.expanduser(), map_location="cpu", weights_only=False
    )
    indices = _sample_indices(
        num_rows=int(samples["z_target"].shape[0]),
        max_samples=int(max_samples),
        seed=int(seed),
    )
    state = (
        samples[state_key]
        .index_select(0, indices)
        .to(device=device_obj, dtype=torch.float32)
    )
    lang = (
        samples["lang"]
        .index_select(0, indices)
        .to(device=device_obj, dtype=torch.float32)
    )
    target = (
        samples["z_target"]
        .index_select(0, indices)
        .to(device=device_obj, dtype=torch.float32)
    )
    config_overrides = {
        "flow_num_inference_steps": flow_num_inference_steps,
        "flow_inference_noise_std": flow_inference_noise_std,
    }
    generator = _build_skill_commander_generator_from_checkpoint(
        checkpoint,
        state_dim=int(state.shape[-1]),
        lang_embed_dim=int(lang.shape[-1]),
        z_dim=int(target.shape[-1]),
        config_overrides=config_overrides,
    ).to(device_obj)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()
    with torch.inference_mode():
        prediction = _predict_batches(
            generator, state, lang, batch_size=int(batch_size)
        )
    summary = {
        "metadata": {
            "setting": setting or f"eval_{state_key}",
            "label": label or setting or f"eval_{state_key}",
            "interface": "latent_skill",
            "state_key": state_key,
            "samples_dir": str(samples_dir.expanduser().resolve()),
            "planner_checkpoint": str(planner_checkpoint.expanduser().resolve()),
            "planner_target_dim": int(target.shape[-1]),
            "planner_metadata": {
                "update": int(checkpoint.get("update", 0)),
                "config": checkpoint.get("config", {}),
                "rollout_finetune": checkpoint.get("rollout_finetune"),
            },
            "max_samples": int(max_samples),
            "seed": int(seed),
            "flow_num_inference_steps": flow_num_inference_steps,
            "flow_inference_noise_std": flow_inference_noise_std,
        },
        "aggregate": {
            "sample_count": int(indices.numel()),
            "source_sample_count": int(samples["z_target"].shape[0]),
            "target_dim": int(target.shape[-1]),
            "state_dim": int(state.shape[-1]),
            "lang_embed_dim": int(lang.shape[-1]),
        },
        "metrics": _metrics(prediction, target),
    }
    output_json = output_json.expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    if output_csv is not None:
        _write_csv(summary, output_csv.expanduser().resolve())
    return summary


def main() -> None:
    args = _parse_args()
    summary = evaluate_latent_skill_planner_checkpoint(
        samples_dir=args.samples_dir,
        planner_checkpoint=args.planner_checkpoint,
        output_json=args.output_json,
        output_csv=args.output_csv,
        state_key=str(args.state_key),
        setting=str(args.setting),
        label=str(args.label),
        device=str(args.device),
        batch_size=int(args.batch_size),
        max_samples=int(args.max_samples),
        seed=int(args.seed),
        flow_num_inference_steps=args.flow_num_inference_steps,
        flow_inference_noise_std=args.flow_inference_noise_std,
    )
    print(f"[INFO] Wrote latent offline planner eval: {args.output_json}")
    print(
        "[INFO] "
        f"state_key={summary['metadata']['state_key']} "
        f"samples={summary['aggregate']['sample_count']} "
        f"target_rmse={summary['metrics']['planner_target_rmse']['mean']:.6f}"
    )


if __name__ == "__main__":
    main()
