#!/usr/bin/env python3
"""Certify that a one-motion interface planner uses causal robot history only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))

from interface_planner_common import (  # noqa: E402
    load_planner_checkpoint,
    load_rollout_samples,
)


EXPECTED_FEATURE_NAMES = [
    "joint_pos_rel",
    "joint_vel_rel",
    "base_ang_vel",
    "projected_gravity",
    "last_action",
]
EXPECTED_FEATURE_WIDTHS = [29, 29, 3, 3, 29]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples_dir", type=Path, required=True)
    parser.add_argument("--planner_checkpoint", type=Path, required=True)
    parser.add_argument("--low_level_checkpoint", type=Path, required=True)
    parser.add_argument(
        "--interface",
        choices=("latent_skill", "full_body_trajectory"),
        default="latent_skill",
    )
    parser.add_argument("--skill_checkpoint", type=Path, default=None)
    parser.add_argument("--motion_name", required=True)
    parser.add_argument("--expected_rows", type=int, default=1000)
    parser.add_argument("--output_json", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = _parse_args()
    samples_dir = args.samples_dir.expanduser().resolve()
    planner_checkpoint = args.planner_checkpoint.expanduser().resolve()
    low_level_checkpoint = args.low_level_checkpoint.expanduser().resolve()
    skill_checkpoint = (
        args.skill_checkpoint.expanduser().resolve()
        if args.skill_checkpoint is not None
        else None
    )

    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    tensors, sample_metadata = load_rollout_samples(samples_dir)
    checkpoint = torch.load(
        planner_checkpoint, map_location="cpu", weights_only=False
    )
    planner, target_spec, planner_metadata = load_planner_checkpoint(
        planner_checkpoint, map_location="cpu"
    )

    rows = int(tensors["planner_state"].shape[0])
    require(rows == int(args.expected_rows), "sample row count mismatch")
    require(
        tuple(tensors["planner_state"].shape) == (rows, 930),
        "causal planner state must have shape [rows, 930]",
    )
    require(
        tuple(tensors["expert_planner_state"].shape) == (rows, 930),
        "expert planner state must have shape [rows, 930]",
    )
    state_delta = float(
        (tensors["planner_state"] - tensors["expert_planner_state"])
        .abs()
        .mean()
        .item()
    )
    require(state_delta > 0.0, "causal state unexpectedly equals expert state")
    if args.interface == "latent_skill":
        require(
            torch.equal(tensors["causal_target"], tensors["demonstration_target"]),
            "latent oracle causal and demonstration targets differ",
        )
    require(
        "language_embedding" not in tensors,
        "state-only samples unexpectedly contain language embeddings",
    )

    observation_spec = sample_metadata.get("planner_observation_spec", {})
    require(
        observation_spec.get("feature_names") == EXPECTED_FEATURE_NAMES,
        "causal feature names do not match the frozen robot-history contract",
    )
    require(
        observation_spec.get("feature_widths") == EXPECTED_FEATURE_WIDTHS,
        "causal feature widths do not match the frozen robot-history contract",
    )
    require(
        observation_spec.get("reference_features") == [],
        "planner observation contains reference features",
    )
    require(
        int(observation_spec.get("history_frames", -1)) == 10,
        "planner history must contain ten frames",
    )
    require(
        int(observation_spec.get("flat_dim", -1)) == 930,
        "planner history flat width must be 930",
    )
    language_metadata = sample_metadata.get("language_conditioning", {})
    require(
        language_metadata == {"enabled": False, "embedding_dim": 0},
        "one-motion gate must disable language conditioning",
    )
    recorded_motion = sample_metadata.get("motion_name")
    recorded_motions = sample_metadata.get("motion_names")
    require(
        recorded_motion == args.motion_name
        or recorded_motions == [args.motion_name],
        "sample metadata is not restricted to the expected motion",
    )

    expected_target_dim = 256 if args.interface == "latent_skill" else 670
    require(target_spec.interface == args.interface, "planner interface mismatch")
    require(
        int(target_spec.target_dim) == expected_target_dim,
        "planner target width mismatch",
    )
    require(
        int(tensors["causal_target"].shape[-1]) == expected_target_dim,
        "sample target width mismatch",
    )
    if args.interface == "latent_skill":
        require(skill_checkpoint is not None, "latent audit requires skill checkpoint")
    require(
        planner_metadata.get("state_key") == "planner_state",
        "planner was not trained from causal planner_state",
    )
    require(
        int(planner_metadata.get("language_dim", -1)) == 0,
        "planner checkpoint unexpectedly enables language",
    )
    require(
        int(planner_metadata.get("state_dim", -1)) == 930,
        "planner checkpoint state width mismatch",
    )
    require(
        checkpoint.get("metadata", {}).get("sample_metadata", {}).get(
            "planner_observation_spec"
        )
        == observation_spec,
        "checkpoint and sample observation specifications differ",
    )

    planner.eval()
    state = tensors["planner_state"][: min(rows, 32)]
    with torch.inference_mode():
        original_prediction = planner(
            state,
            num_inference_steps=16,
            inference_noise_std=0.0,
            language=None,
        )
        # These are the forbidden expert/reference-side fields. Perturbing them
        # leaves the deployable planner input unchanged by construction.
        counterfactual_expert = -tensors["expert_planner_state"][: state.shape[0]]
        counterfactual_target = torch.flip(
            tensors["demonstration_target"][: state.shape[0]], dims=(0,)
        )
        require(
            not torch.equal(
                counterfactual_expert,
                tensors["expert_planner_state"][: state.shape[0]],
            ),
            "expert-state counterfactual did not change",
        )
        require(
            not torch.equal(
                counterfactual_target,
                tensors["demonstration_target"][: state.shape[0]],
            ),
            "target counterfactual did not change",
        )
        counterfactual_prediction = planner(
            state,
            num_inference_steps=16,
            inference_noise_std=0.0,
            language=None,
        )
    max_prediction_delta = float(
        (original_prediction - counterfactual_prediction).abs().max().item()
    )
    require(
        max_prediction_delta == 0.0,
        "planner output changed after perturbing forbidden expert-side fields",
    )

    result: dict[str, Any] = {
        "passed": not errors,
        "errors": errors,
        "interface": args.interface,
        "motion_name": args.motion_name,
        "sample_rows": rows,
        "state_key": planner_metadata.get("state_key"),
        "state_dim": int(tensors["planner_state"].shape[-1]),
        "target_dim": int(tensors["causal_target"].shape[-1]),
        "language_dim": int(planner_metadata.get("language_dim", -1)),
        "causal_vs_expert_state_mean_abs_delta": state_delta,
        "reference_features": observation_spec.get("reference_features"),
        "feature_names": observation_spec.get("feature_names"),
        "forbidden_field_counterfactual_max_prediction_delta": max_prediction_delta,
        "samples_dir": str(samples_dir),
        "planner_checkpoint": str(planner_checkpoint),
        "planner_checkpoint_sha256": _sha256(planner_checkpoint),
        "low_level_checkpoint": str(low_level_checkpoint),
        "low_level_checkpoint_sha256": _sha256(low_level_checkpoint),
        "skill_checkpoint": str(skill_checkpoint) if skill_checkpoint else None,
        "skill_checkpoint_sha256": (
            _sha256(skill_checkpoint) if skill_checkpoint else None
        ),
    }
    args.output_json.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output_json.expanduser().resolve().write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if errors:
        raise SystemExit("Causal planner gate audit failed: " + "; ".join(errors))
    print(
        "[PASS] One-motion planner uses causal achieved-state history only."
    )


if __name__ == "__main__":
    main()
