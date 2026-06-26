#!/usr/bin/env python3
"""Evaluate an interface planner on saved oracle-drive samples without Isaac."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))

from interface_planner_common import (  # noqa: E402
    InterfaceTargetSpec,
    cosine_mean,
    load_planner_checkpoint,
    load_rollout_samples,
    mean_std,
    rmse_per_row,
    supported_interfaces,
    unflatten_command_target,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples_dir", type=Path, required=True)
    parser.add_argument("--planner_checkpoint", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, default=None)
    parser.add_argument(
        "--interface",
        choices=supported_interfaces(),
        default=None,
        help="Interface name. Defaults to the checkpoint target spec.",
    )
    parser.add_argument(
        "--state_key",
        choices=("expert_planner_state", "planner_state"),
        default="expert_planner_state",
        help="Expert-state pretrain eval or achieved-state finetune eval.",
    )
    parser.add_argument("--setting", type=str, default="")
    parser.add_argument("--label", type=str, default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--exclude_checkpoint_selected_indices",
        action="store_true",
        default=False,
        help=(
            "For sample-budget sweeps, evaluate on rows not selected by the "
            "planner checkpoint metadata when that metadata is available."
        ),
    )
    parser.add_argument("--flow_num_inference_steps", type=int, default=16)
    parser.add_argument("--flow_inference_noise_std", type=float, default=0.0)
    return parser.parse_args()


def _resolve_device(device: str) -> torch.device:
    if device.strip().lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _sample_indices(
    *,
    num_rows: int,
    max_samples: int,
    seed: int,
    planner_metadata: dict[str, Any],
    exclude_checkpoint_selected_indices: bool,
) -> torch.Tensor:
    indices = torch.arange(int(num_rows), dtype=torch.long)
    selected_indices = planner_metadata.get("selected_indices")
    if exclude_checkpoint_selected_indices and selected_indices:
        selected = torch.as_tensor(selected_indices, dtype=torch.long)
        keep = torch.ones(int(num_rows), dtype=torch.bool)
        selected = selected[(selected >= 0) & (selected < int(num_rows))]
        keep[selected] = False
        indices = indices[keep]
        if int(indices.numel()) == 0:
            raise ValueError(
                "No evaluation rows remain after excluding checkpoint selected_indices."
            )
    if int(max_samples) > 0 and int(max_samples) < int(indices.numel()):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        perm = torch.randperm(int(indices.numel()), generator=generator)[
            : int(max_samples)
        ]
        indices = indices[perm]
    indices, _ = torch.sort(indices)
    return indices


@torch.no_grad()
def _predict_batches(
    planner: torch.nn.Module,
    state: torch.Tensor,
    *,
    batch_size: int,
    flow_num_inference_steps: int,
    flow_inference_noise_std: float,
) -> torch.Tensor:
    predictions: list[torch.Tensor] = []
    for start in range(0, int(state.shape[0]), int(batch_size)):
        stop = min(start + int(batch_size), int(state.shape[0]))
        predictions.append(
            planner(
                state[start:stop],
                num_inference_steps=int(flow_num_inference_steps),
                inference_noise_std=float(flow_inference_noise_std),
            ).detach()
        )
    return torch.cat(predictions, dim=0)


def _metric_stats(values: torch.Tensor) -> dict[str, float | int]:
    mean, std = mean_std(values)
    return {"mean": mean, "std": std, "count": int(values.numel())}


def _target_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_spec: InterfaceTargetSpec,
    planner: torch.nn.Module,
) -> dict[str, dict[str, float | int]]:
    metrics: dict[str, dict[str, float | int]] = {
        "planner_target_rmse": _metric_stats(rmse_per_row(prediction, target)),
        "planner_target_mse": {
            "mean": float(F.mse_loss(prediction, target).item()),
            "std": 0.0,
            "count": int(prediction.shape[0]),
        },
        "planner_target_cosine": {
            "mean": cosine_mean(prediction, target),
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
    normalize_target = getattr(planner, "normalize_target", None)
    if callable(normalize_target):
        prediction_norm = normalize_target(prediction)
        target_norm = normalize_target(target)
        metrics["planner_normalized_target_rmse"] = _metric_stats(
            rmse_per_row(prediction_norm, target_norm)
        )
        metrics["planner_normalized_target_mse"] = {
            "mean": float(F.mse_loss(prediction_norm, target_norm).item()),
            "std": 0.0,
            "count": int(prediction.shape[0]),
        }
        metrics["planner_normalized_target_cosine"] = {
            "mean": cosine_mean(prediction_norm, target_norm),
            "std": 0.0,
            "count": int(prediction.shape[0]),
        }
    prediction_terms = unflatten_command_target(prediction.cpu(), target_spec)
    target_terms = unflatten_command_target(target.cpu(), target_spec)
    for term_name in target_spec.term_names:
        metrics[f"{term_name}_rmse"] = _metric_stats(
            rmse_per_row(prediction_terms[term_name], target_terms[term_name])
        )
    return metrics


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


def evaluate_planner_checkpoint(
    *,
    samples_dir: Path,
    planner_checkpoint: Path,
    output_json: Path,
    output_csv: Path | None = None,
    interface: str | None = None,
    state_key: str = "expert_planner_state",
    setting: str = "",
    label: str = "",
    device: str = "auto",
    batch_size: int = 512,
    max_samples: int = 0,
    seed: int = 0,
    exclude_checkpoint_selected_indices: bool = False,
    flow_num_inference_steps: int = 16,
    flow_inference_noise_std: float = 0.0,
) -> dict[str, Any]:
    if int(batch_size) <= 0:
        raise ValueError("--batch_size must be positive.")
    device_obj = _resolve_device(str(device))
    data_cpu, sample_metadata = load_rollout_samples(samples_dir.expanduser())
    if state_key not in data_cpu:
        raise KeyError(f"Sample data does not contain state key {state_key!r}.")
    planner, target_spec, planner_metadata = load_planner_checkpoint(
        planner_checkpoint.expanduser(), map_location=device_obj
    )
    if interface is not None and str(interface) != target_spec.interface:
        raise ValueError(
            f"Checkpoint interface {target_spec.interface!r} does not match --interface={interface!r}."
        )
    sample_spec = InterfaceTargetSpec.from_dict(sample_metadata["target_spec"])
    if sample_spec != target_spec:
        raise ValueError(
            "Sample target spec does not match checkpoint target spec: "
            f"{sample_spec.to_dict()} vs {target_spec.to_dict()}."
        )

    indices = _sample_indices(
        num_rows=int(data_cpu["target"].shape[0]),
        max_samples=int(max_samples),
        seed=int(seed),
        planner_metadata=planner_metadata,
        exclude_checkpoint_selected_indices=bool(exclude_checkpoint_selected_indices),
    )
    state = (
        data_cpu[state_key]
        .index_select(0, indices)
        .to(device=device_obj, dtype=torch.float32)
    )
    target = (
        data_cpu["target"]
        .index_select(0, indices)
        .to(device=device_obj, dtype=torch.float32)
    )
    if int(planner.state_dim) != int(state.shape[-1]):
        raise ValueError(
            f"Planner state_dim={planner.state_dim} does not match samples {state.shape[-1]}."
        )
    if int(planner.target_dim) != int(target.shape[-1]):
        raise ValueError(
            f"Planner target_dim={planner.target_dim} does not match samples {target.shape[-1]}."
        )

    planner = planner.to(device_obj)
    planner.eval()
    with torch.inference_mode():
        prediction = _predict_batches(
            planner,
            state,
            batch_size=int(batch_size),
            flow_num_inference_steps=int(flow_num_inference_steps),
            flow_inference_noise_std=float(flow_inference_noise_std),
        )
    metrics = _target_metrics(prediction, target, target_spec, planner)
    summary = {
        "metadata": {
            "setting": setting or f"eval_{state_key}",
            "label": label or setting or f"eval_{state_key}",
            "interface": target_spec.interface,
            "state_key": state_key,
            "samples_dir": str(samples_dir.expanduser().resolve()),
            "planner_checkpoint": str(planner_checkpoint.expanduser().resolve()),
            "planner_target_dim": int(target_spec.target_dim),
            "planner_metadata": planner_metadata,
            "sample_metadata": sample_metadata,
            "max_samples": int(max_samples),
            "seed": int(seed),
            "exclude_checkpoint_selected_indices": bool(
                exclude_checkpoint_selected_indices
            ),
            "flow_num_inference_steps": int(flow_num_inference_steps),
            "flow_inference_noise_std": float(flow_inference_noise_std),
        },
        "aggregate": {
            "sample_count": int(indices.numel()),
            "source_sample_count": int(data_cpu["target"].shape[0]),
            "target_dim": int(target.shape[-1]),
            "state_dim": int(state.shape[-1]),
        },
        "metrics": metrics,
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
    summary = evaluate_planner_checkpoint(
        samples_dir=args.samples_dir,
        planner_checkpoint=args.planner_checkpoint,
        output_json=args.output_json,
        output_csv=args.output_csv,
        interface=args.interface,
        state_key=str(args.state_key),
        setting=str(args.setting),
        label=str(args.label),
        device=str(args.device),
        batch_size=int(args.batch_size),
        max_samples=int(args.max_samples),
        seed=int(args.seed),
        exclude_checkpoint_selected_indices=bool(
            args.exclude_checkpoint_selected_indices
        ),
        flow_num_inference_steps=int(args.flow_num_inference_steps),
        flow_inference_noise_std=float(args.flow_inference_noise_std),
    )
    print(f"[INFO] Wrote offline planner eval: {args.output_json}")
    print(
        "[INFO] "
        f"interface={summary['metadata']['interface']} "
        f"state_key={summary['metadata']['state_key']} "
        f"samples={summary['aggregate']['sample_count']} "
        f"target_rmse={summary['metrics']['planner_target_rmse']['mean']:.6f}"
    )


if __name__ == "__main__":
    main()
