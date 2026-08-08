#!/usr/bin/env python3
"""Materialize three ordered H10 latent forecasts from trajectory-keyed samples.

Two target frames are intentionally supported:

``future_publication``
    Token k is the oracle latent saved at planner publication n+k. This is the
    transport-aware target: every forecast is supervised in the robot pelvis
    frame in which it will eventually be consumed.

``current_publication``
    Split the stored 30-frame root_qpos window into three H10 packets and encode
    all three in the current publication frame. This is the deliberately stale-
    frame diagnostic discussed for raw latent temporal ensembling.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from imitation_experiments.capacity.measure_encoder_noise_contraction import (
    SkillEncoder,
)


TOKENS = 3
TOKEN_WIDTH = 256
FRAMES_PER_TOKEN = 10
ROOT_QPOS_WIDTH = 38


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples_dir", type=Path, required=True)
    parser.add_argument("--skill_checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--target_frame",
        choices=("future_publication", "current_publication"),
        required=True,
    )
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--validation_tolerance", type=float, default=2.0e-5)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _device(value: str) -> torch.device:
    if value.strip().lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _row_key(sample: dict[str, Any], index: int) -> tuple[int, int, int]:
    return (
        int(sample["env_id"][index]),
        int(sample["episode_id"][index]),
        int(sample["planner_step"][index]),
    )


def _select_rows(
    sample: dict[str, Any], *, selected: list[int], rows: int
) -> dict[str, Any]:
    indices = torch.as_tensor(selected, dtype=torch.long)
    result: dict[str, Any] = {}
    for key, value in sample.items():
        if (
            isinstance(value, torch.Tensor)
            and value.ndim > 0
            and int(value.shape[0]) == rows
        ):
            result[key] = value.index_select(0, indices)
        elif isinstance(value, list) and len(value) == rows:
            result[key] = [value[index] for index in selected]
        else:
            result[key] = copy.deepcopy(value)
    return result


def _encode_windows(
    encoder: SkillEncoder,
    windows: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if tuple(windows.shape[1:]) != (TOKENS * FRAMES_PER_TOKEN, ROOT_QPOS_WIDTH):
        raise ValueError(
            f"Expected root_qpos lookahead [N,30,38], got {tuple(windows.shape)}."
        )
    outputs: list[torch.Tensor] = []
    encoder = encoder.to(device)
    for start in range(0, int(windows.shape[0]), int(batch_size)):
        batch = windows[start : start + int(batch_size)].to(
            device=device, dtype=torch.float32
        )
        token_windows = batch.reshape(-1, FRAMES_PER_TOKEN, ROOT_QPOS_WIDTH)
        with torch.no_grad():
            encoded = encoder(token_windows.reshape(token_windows.shape[0], -1))
        outputs.append(encoded.reshape(-1, TOKENS, TOKEN_WIDTH).cpu())
    return torch.cat(outputs, dim=0)


def main() -> None:
    args = _parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if args.validation_tolerance <= 0:
        raise ValueError("--validation_tolerance must be positive.")
    samples_dir = args.samples_dir.expanduser().resolve()
    skill_checkpoint = args.skill_checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    source_paths = sorted(samples_dir.glob("sample_step_*.pt"))
    if not source_paths:
        raise FileNotFoundError(f"No sample shards under {samples_dir}.")
    if not skill_checkpoint.is_file():
        raise FileNotFoundError(skill_checkpoint)
    if output_dir.exists():
        raise FileExistsError(f"Refusing existing output directory: {output_dir}")

    checkpoint = torch.load(skill_checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("skill_encoder_state_dict")
    if not isinstance(state, dict):
        raise ValueError("Skill checkpoint has no skill_encoder_state_dict.")
    encoder = SkillEncoder(state)
    first_weight = state.get("net.0.weight")
    if not isinstance(first_weight, torch.Tensor) or int(first_weight.shape[1]) != 380:
        raise ValueError(
            "H3 materialization requires the frozen root_qpos H10 encoder "
            f"(input 380), got {getattr(first_weight, 'shape', None)}."
        )

    # Only compact identity + latent records are retained across shards. The
    # 1.7-GB causal/root_qpos payload is streamed again during materialization.
    latent_by_key: dict[tuple[int, int, int], torch.Tensor] = {}
    for path in source_paths:
        sample = torch.load(path, map_location="cpu", weights_only=False)
        latent = sample.get("latent_skill_target")
        if not isinstance(latent, torch.Tensor) or tuple(latent.shape[1:]) != (
            TOKEN_WIDTH,
        ):
            raise ValueError(f"{path} has no [N,256] latent_skill_target.")
        for index in range(int(latent.shape[0])):
            key = _row_key(sample, index)
            if key in latent_by_key:
                raise ValueError(f"Duplicate trajectory publication key {key}.")
            latent_by_key[key] = latent[index].detach().cpu().float().contiguous()
        del sample

    output_dir.mkdir(parents=True)
    manifest: dict[str, Any] = {
        "format": "latent_receding_horizon_materialization",
        "version": 1,
        "target_frame": str(args.target_frame),
        "tokens": TOKENS,
        "frames_per_token": FRAMES_PER_TOKEN,
        "token_width": TOKEN_WIDTH,
        "source_samples_dir": str(samples_dir),
        "skill_checkpoint": str(skill_checkpoint),
        "skill_checkpoint_sha256": _sha256(skill_checkpoint),
        "source_rows": len(latent_by_key),
        "retained_rows": 0,
        "dropped_tail_rows": 0,
        "z0_reconstruction_max_abs": 0.0,
        "output_files": [],
    }
    target_spec = {
        "interface": "latent_skill",
        "term_names": ["z_current", "z_plus_10", "z_plus_20"],
        "term_widths": [TOKEN_WIDTH] * TOKENS,
        "target_dim": TOKEN_WIDTH * TOKENS,
    }
    for source_path in source_paths:
        sample = torch.load(source_path, map_location="cpu", weights_only=False)
        latent = sample["latent_skill_target"].float()
        windows = sample.get("expert_root_qpos_future")
        valid = sample.get("expert_root_qpos_future_valid")
        rows = int(latent.shape[0])
        if (
            not isinstance(windows, torch.Tensor)
            or tuple(windows.shape) != (rows, 30, ROOT_QPOS_WIDTH)
            or not isinstance(valid, torch.Tensor)
            or tuple(valid.shape) != (rows, 30)
        ):
            raise ValueError(f"{source_path} lacks the [N,30,38] lookahead contract.")

        selected: list[int] = []
        aligned_targets: list[torch.Tensor] = []
        for index in range(rows):
            key = _row_key(sample, index)
            future_keys = [(key[0], key[1], key[2] + age) for age in range(TOKENS)]
            if bool(valid[index].all()) and all(
                k in latent_by_key for k in future_keys
            ):
                selected.append(index)
                aligned_targets.append(
                    torch.stack([latent_by_key[k] for k in future_keys])
                )
        manifest["dropped_tail_rows"] += rows - len(selected)
        if not selected:
            del sample
            continue

        selected_tensor = torch.as_tensor(selected, dtype=torch.long)
        selected_windows = windows.index_select(0, selected_tensor).float()
        current_frame_targets = _encode_windows(
            encoder,
            selected_windows,
            batch_size=int(args.batch_size),
            device=_device(str(args.device)),
        )
        source_z0 = latent.index_select(0, selected_tensor)
        z0_error = float((current_frame_targets[:, 0] - source_z0).abs().max().item())
        if z0_error > float(args.validation_tolerance):
            raise ValueError(
                "Stored root_qpos window does not reproduce the frozen encoder z0: "
                f"{z0_error:.6g} > {args.validation_tolerance:.6g}."
            )
        manifest["z0_reconstruction_max_abs"] = max(
            float(manifest["z0_reconstruction_max_abs"]), z0_error
        )
        if args.target_frame == "future_publication":
            token_target = torch.stack(aligned_targets)
        else:
            token_target = current_frame_targets
        flat_target = token_target.reshape(len(selected), TOKENS * TOKEN_WIDTH)

        result = _select_rows(sample, selected=selected, rows=rows)
        result["source_h1_latent_target"] = source_z0.contiguous()
        result["latent_skill_target"] = flat_target.contiguous()
        result["z_target"] = flat_target.contiguous()
        result["causal_target"] = flat_target.contiguous()
        result["demonstration_target"] = flat_target.contiguous()
        metadata = result.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"{source_path} has no metadata mapping.")
        metadata["interface"] = "latent_skill"
        metadata["target_spec"] = target_spec
        metadata["command_future_steps"] = TOKENS * FRAMES_PER_TOKEN
        metadata["latent_receding_horizon"] = {
            "tokens": TOKENS,
            "frames_per_token": FRAMES_PER_TOKEN,
            "target_frame": str(args.target_frame),
            "skill_checkpoint": str(skill_checkpoint),
            "skill_checkpoint_sha256": manifest["skill_checkpoint_sha256"],
            "z0_reconstruction_max_abs": z0_error,
        }
        result["metadata"] = metadata
        output_path = output_dir / source_path.name
        torch.save(result, output_path)
        retained = len(selected)
        manifest["retained_rows"] += retained
        manifest["output_files"].append(
            {"name": output_path.name, "rows": retained, "sha256": _sha256(output_path)}
        )
        del sample, result

    manifest_path = output_dir / "materialization_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"[PASS] H3 {args.target_frame}: {manifest['retained_rows']} rows, "
        f"dropped {manifest['dropped_tail_rows']} trajectory-tail rows, "
        f"z0 max_abs={manifest['z0_reconstruction_max_abs']:.3e}.",
        flush=True,
    )
    print(f"[PASS] {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
