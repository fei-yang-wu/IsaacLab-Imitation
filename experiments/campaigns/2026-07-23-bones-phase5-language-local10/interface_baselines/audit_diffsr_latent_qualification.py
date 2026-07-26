#!/usr/bin/env python3
"""Audit a local corrected-data DiffSR latent qualification run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from validate_latent_skill_checkpoint_binding import validate_binding


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check(name: str, passed: bool, actual: Any, expected: Any) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "actual": actual,
        "expected": expected,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--low_level_checkpoint", type=Path, required=True)
    parser.add_argument("--skill_checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected_dataset_path", type=Path, default=None)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--expected_num_envs", type=int, default=40)
    parser.add_argument("--expected_steps", type=int, default=1000)
    parser.add_argument("--expected_seed", type=int, default=0)
    parser.add_argument("--success_threshold", type=float, default=0.8)
    parser.add_argument("--require_pass", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary_path = args.summary.expanduser().resolve()
    low_level_checkpoint = args.low_level_checkpoint.expanduser().resolve()
    skill_checkpoint = args.skill_checkpoint.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metadata = summary.get("metadata", {})
    aggregate = summary.get("aggregate", {})
    dataset_path = Path(str(metadata.get("dataset_path", ""))).expanduser().resolve()
    tracking_success = float(aggregate.get("tracking_success_rate", float("nan")))
    tracking_failure = float(aggregate.get("tracking_failure_rate", float("nan")))
    skill_binding = validate_binding(low_level_checkpoint, skill_checkpoint)
    checks = [
        _check(
            "task",
            metadata.get("task") == "Isaac-Imitation-G1-Latent-v0",
            metadata.get("task"),
            "Isaac-Imitation-G1-Latent-v0",
        ),
        _check(
            "algorithm",
            metadata.get("algorithm") == "IPMD",
            metadata.get("algorithm"),
            "IPMD",
        ),
        _check(
            "interface",
            metadata.get("interface") == "latent_skill",
            metadata.get("interface"),
            "latent_skill",
        ),
        _check(
            "planner_target_dim",
            int(metadata.get("planner_target_dim", -1)) == 256,
            metadata.get("planner_target_dim"),
            256,
        ),
        _check(
            "num_envs",
            int(metadata.get("num_envs", -1)) == args.expected_num_envs,
            metadata.get("num_envs"),
            args.expected_num_envs,
        ),
        _check(
            "seed",
            int(metadata.get("seed", -1)) == args.expected_seed,
            metadata.get("seed"),
            args.expected_seed,
        ),
        _check(
            "reset_schedule",
            metadata.get("reset_schedule") == "sequential",
            metadata.get("reset_schedule"),
            "sequential",
        ),
        _check(
            "wrap_steps",
            metadata.get("wrap_steps") is False,
            metadata.get("wrap_steps"),
            False,
        ),
        _check(
            "observation_corruption",
            metadata.get("policy_observation_corruption_enabled") is False,
            metadata.get("policy_observation_corruption_enabled"),
            False,
        ),
        _check(
            "early_terminations",
            metadata.get("early_terminations_enabled") is True,
            metadata.get("early_terminations_enabled"),
            True,
        ),
        _check(
            "time_out",
            metadata.get("time_out_enabled") is True,
            metadata.get("time_out_enabled"),
            True,
        ),
        _check(
            "episode_length_extension",
            metadata.get("episode_length_extension_enabled") is True,
            metadata.get("episode_length_extension_enabled"),
            True,
        ),
        _check(
            "reward_clipping",
            metadata.get("reward_clipping_enabled") is False,
            metadata.get("reward_clipping_enabled"),
            False,
        ),
        _check(
            "low_level_checkpoint",
            Path(str(metadata.get("checkpoint", ""))).resolve() == low_level_checkpoint,
            metadata.get("checkpoint"),
            str(low_level_checkpoint),
        ),
        _check(
            "manifest",
            Path(str(metadata.get("motion_manifest", ""))).resolve() == manifest,
            metadata.get("motion_manifest"),
            str(manifest),
        ),
        _check(
            "dataset_path",
            args.expected_dataset_path is None
            or dataset_path == args.expected_dataset_path.expanduser().resolve(),
            str(dataset_path),
            str(args.expected_dataset_path.expanduser().resolve())
            if args.expected_dataset_path is not None
            else "recorded dataset path",
        ),
        _check(
            "skill_checkpoint",
            Path(str(summary.get("skill_checkpoint_override", ""))).resolve()
            == skill_checkpoint,
            summary.get("skill_checkpoint_override"),
            str(skill_checkpoint),
        ),
        _check(
            "low_level_skill_binding",
            skill_binding.get("passed") is True,
            skill_binding.get("passed"),
            True,
        ),
        _check(
            "multi_env_stop",
            summary.get("stop_reason") in {"all_envs_done", "max_steps"},
            summary.get("stop_reason"),
            "all_envs_done or max_steps",
        ),
        _check(
            "max_steps",
            int(summary.get("max_steps", -1)) == args.expected_steps,
            summary.get("max_steps"),
            args.expected_steps,
        ),
        _check(
            "step_budget",
            0 < int(summary.get("steps_run", 0)) <= args.expected_steps,
            summary.get("steps_run"),
            f"1..{args.expected_steps}",
        ),
        _check(
            "explicit_tracking_success",
            math.isfinite(tracking_success),
            aggregate.get("tracking_success_rate"),
            "finite explicit metric",
        ),
        _check(
            "strict_success_consistency",
            math.isfinite(tracking_failure)
            and math.isclose(tracking_success, 1.0 - tracking_failure, abs_tol=1.0e-6),
            tracking_success,
            (
                1.0 - tracking_failure
                if math.isfinite(tracking_failure)
                else "finite 1 - tracking_failure_rate"
            ),
        ),
    ]
    protocol_passed = all(item["passed"] for item in checks)
    oracle_passed = tracking_success >= float(args.success_threshold)
    result = {
        "protocol_passed": protocol_passed,
        "oracle_passed": oracle_passed,
        "summary": str(summary_path),
        "low_level_checkpoint": str(low_level_checkpoint),
        "low_level_checkpoint_sha256": _sha256(low_level_checkpoint),
        "skill_checkpoint": str(skill_checkpoint),
        "skill_checkpoint_sha256": _sha256(skill_checkpoint),
        "low_level_skill_binding": skill_binding,
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "dataset_path": str(dataset_path),
        "tracking_success_rate": tracking_success,
        "tracking_failure_rate": tracking_failure,
        "success_threshold": float(args.success_threshold),
        "survival_steps_mean": aggregate.get("survival_steps_mean"),
        "checks": checks,
    }
    output = args.output_json.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"[RESULT] protocol_passed={protocol_passed} "
        f"oracle_passed={oracle_passed} success_rate={tracking_success:.3f}"
    )
    if not protocol_passed or (args.require_pass and not oracle_passed):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
