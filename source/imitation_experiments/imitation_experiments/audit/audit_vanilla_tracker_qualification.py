#!/usr/bin/env python3
"""Audit one strict direct-vanilla tracker qualification result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


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
    parser.add_argument("--checkpoint", type=Path, required=True)
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
    checkpoint = args.checkpoint.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})
    aggregate = payload.get("aggregate", {})
    tracker = metadata.get("low_level_tracker", {})
    dataset_path = Path(str(metadata.get("dataset_path", ""))).expanduser().resolve()

    done_rate = float(aggregate.get("done_rate", float("nan")))
    success_rate = float(aggregate.get("tracking_success_rate", float("nan")))
    checks = [
        _check(
            "task",
            metadata.get("task") == "Isaac-Imitation-G1-v0",
            metadata.get("task"),
            "Isaac-Imitation-G1-v0",
        ),
        _check(
            "algorithm",
            metadata.get("algorithm") == "IPMD",
            metadata.get("algorithm"),
            "IPMD",
        ),
        _check(
            "command_space",
            metadata.get("command_space") == "single_frame_full_body",
            metadata.get("command_space"),
            "single_frame_full_body",
        ),
        _check(
            "command_mode",
            metadata.get("low_level_command_mode") == "native",
            metadata.get("low_level_command_mode"),
            "native",
        ),
        _check(
            "command_source",
            metadata.get("command_observation_source") == "reference",
            metadata.get("command_observation_source"),
            "reference",
        ),
        _check(
            "command_past_steps",
            int(metadata.get("command_past_steps", -1)) == 0,
            metadata.get("command_past_steps"),
            0,
        ),
        _check(
            "command_future_steps",
            int(metadata.get("command_future_steps", -1)) == 0,
            metadata.get("command_future_steps"),
            0,
        ),
        _check(
            "planner_update_interval",
            int(metadata.get("planner_update_interval", -1)) == 1,
            metadata.get("planner_update_interval"),
            1,
        ),
        _check(
            "policy_only_checkpoint",
            metadata.get("policy_only_checkpoint") is True,
            metadata.get("policy_only_checkpoint"),
            True,
        ),
        _check(
            "strict_policy_restore",
            tracker.get("strict_policy_restore") is True,
            tracker.get("strict_policy_restore"),
            True,
        ),
        _check(
            "policy_frozen",
            tracker.get("policy_frozen") is True,
            tracker.get("policy_frozen"),
            True,
        ),
        _check(
            "num_envs",
            int(metadata.get("num_envs", -1)) == args.expected_num_envs,
            metadata.get("num_envs"),
            args.expected_num_envs,
        ),
        _check(
            "steps",
            int(metadata.get("steps_requested", -1)) == args.expected_steps,
            metadata.get("steps_requested"),
            args.expected_steps,
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
            "reference_start_frame",
            int(metadata.get("reference_start_frame", -1)) == 0,
            metadata.get("reference_start_frame"),
            0,
        ),
        _check(
            "stop_after_done",
            metadata.get("keep_after_done") is False,
            metadata.get("keep_after_done"),
            False,
        ),
        _check(
            "observation_corruption",
            metadata.get("observation_corruption_enabled") is False,
            metadata.get("observation_corruption_enabled"),
            False,
        ),
        _check(
            "wrap_steps",
            metadata.get("wrap_steps") is False,
            metadata.get("wrap_steps"),
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
            "max_steps",
            int(payload.get("max_steps", -1)) == args.expected_steps,
            payload.get("max_steps"),
            args.expected_steps,
        ),
        _check(
            "steps_run",
            0 < int(payload.get("steps_run", 0)) <= args.expected_steps,
            payload.get("steps_run"),
            f"1..{args.expected_steps}",
        ),
        _check(
            "checkpoint_path",
            Path(str(metadata.get("checkpoint", ""))).resolve() == checkpoint,
            metadata.get("checkpoint"),
            str(checkpoint),
        ),
        _check(
            "checkpoint_sha256",
            tracker.get("checkpoint_sha256") == _sha256(checkpoint),
            tracker.get("checkpoint_sha256"),
            _sha256(checkpoint),
        ),
        _check(
            "manifest_path",
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
            "tracking_success_rate_present",
            "tracking_success_rate" in aggregate,
            aggregate.get("tracking_success_rate"),
            "explicit metric",
        ),
        _check(
            "per_environment_present",
            len(payload.get("per_environment", [])) == args.expected_num_envs,
            len(payload.get("per_environment", [])),
            args.expected_num_envs,
        ),
    ]
    protocol_passed = all(item["passed"] for item in checks)
    oracle_passed = success_rate >= float(args.success_threshold)
    result = {
        "summary": str(summary_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "dataset_path": str(dataset_path),
        "protocol_passed": protocol_passed,
        "oracle_passed": oracle_passed,
        "success_rate": success_rate,
        "success_threshold": float(args.success_threshold),
        "survival_steps_mean": aggregate.get("survival_steps_mean"),
        "done_rate": done_rate,
        "checks": checks,
    }
    output = args.output_json.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"[RESULT] protocol_passed={protocol_passed} oracle_passed={oracle_passed} "
        f"success_rate={success_rate:.3f}"
    )
    if not protocol_passed or (args.require_pass and not oracle_passed):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
