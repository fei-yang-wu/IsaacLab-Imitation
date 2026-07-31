#!/usr/bin/env python3
"""Audit the expert-packet -> frozen encoder -> latent command pin test."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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


def _resolved(value: Any) -> Path:
    return Path(str(value)).expanduser().resolve()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--planner_checkpoint", type=Path, required=True)
    parser.add_argument("--low_level_checkpoint", type=Path, required=True)
    parser.add_argument("--skill_checkpoint", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--mse_tolerance", type=float, default=1.0e-10)
    parser.add_argument("--require_pass", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary_path = args.summary.expanduser().resolve()
    planner_path = args.planner_checkpoint.expanduser().resolve()
    tracker_path = args.low_level_checkpoint.expanduser().resolve()
    encoder_path = args.skill_checkpoint.expanduser().resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metadata = summary.get("metadata", {})
    packet = metadata.get("packet_encoder_command") or {}
    z_mse = float(packet.get("expert_pin_latent_mse", float("nan")))
    pin_max_abs = float(packet.get("expert_pin_latent_max_abs", float("nan")))
    pin_value_count = int(packet.get("expert_pin_latent_value_count", 0))

    expected_terms = [
        ["expert_motion_qpos", 29],
        ["expert_anchor_pos_b", 3],
        ["expert_anchor_ori_b", 6],
    ]
    planner_sha256 = _sha256(planner_path)
    checks = [
        _check(
            "packet_source",
            packet.get("packet_source") == "expert",
            packet.get("packet_source"),
            "expert",
        ),
        _check(
            "packet_interface",
            packet.get("packet_interface") == "root_qpos",
            packet.get("packet_interface"),
            "root_qpos",
        ),
        _check(
            "packet_target_dim",
            int(packet.get("packet_target_dim", -1)) == 380,
            packet.get("packet_target_dim"),
            380,
        ),
        _check(
            "encoder_input_width",
            int(packet.get("encoder_input_width", -1)) == 380,
            packet.get("encoder_input_width"),
            380,
        ),
        _check(
            "packet_width",
            int(packet.get("packet_width", -1)) == 380,
            packet.get("packet_width"),
            380,
        ),
        _check(
            "packet_frames",
            int(packet.get("packet_frames", -1)) == 10,
            packet.get("packet_frames"),
            10,
        ),
        _check(
            "packet_frame_width",
            int(packet.get("packet_frame_width", -1)) == 38,
            packet.get("packet_frame_width"),
            38,
        ),
        _check(
            "packet_terms",
            packet.get("packet_term_widths") == expected_terms,
            packet.get("packet_term_widths"),
            expected_terms,
        ),
        _check(
            "layout_verified",
            packet.get("layout_verified") is True,
            packet.get("layout_verified"),
            True,
        ),
        _check(
            "publishes",
            int(packet.get("publishes", 0)) > 0,
            packet.get("publishes"),
            "> 0",
        ),
        _check(
            "packet_noise",
            float(packet.get("packet_noise_alpha", float("nan"))) == 0.0,
            packet.get("packet_noise_alpha"),
            0.0,
        ),
        _check(
            "latent_noise",
            float(packet.get("z_noise_alpha", float("nan"))) == 0.0,
            packet.get("z_noise_alpha"),
            0.0,
        ),
        _check(
            "planner_checkpoint",
            _resolved(packet.get("packet_planner_checkpoint", "")) == planner_path,
            packet.get("packet_planner_checkpoint"),
            str(planner_path),
        ),
        _check(
            "planner_sha256",
            packet.get("packet_planner_sha256") == planner_sha256,
            packet.get("packet_planner_sha256"),
            planner_sha256,
        ),
        _check(
            "tracker_checkpoint",
            _resolved(metadata.get("checkpoint", "")) == tracker_path,
            metadata.get("checkpoint"),
            str(tracker_path),
        ),
        _check(
            "encoder_checkpoint",
            _resolved(summary.get("skill_checkpoint_override", "")) == encoder_path,
            summary.get("skill_checkpoint_override"),
            str(encoder_path),
        ),
        _check(
            "expert_pin_value_count",
            pin_value_count > 0,
            pin_value_count,
            "> 0",
        ),
        _check(
            "expert_pin_latent_mse",
            math.isfinite(z_mse) and z_mse <= args.mse_tolerance,
            z_mse,
            f"<= {args.mse_tolerance}",
        ),
        _check(
            "expert_pin_latent_max_abs",
            math.isfinite(pin_max_abs) and pin_max_abs <= math.sqrt(args.mse_tolerance),
            pin_max_abs,
            f"<= {math.sqrt(args.mse_tolerance)}",
        ),
    ]
    passed = all(check["passed"] for check in checks)
    result = {
        "passed": passed,
        "summary": str(summary_path),
        "planner_checkpoint": str(planner_path),
        "planner_checkpoint_sha256": planner_sha256,
        "low_level_checkpoint": str(tracker_path),
        "low_level_checkpoint_sha256": _sha256(tracker_path),
        "skill_checkpoint": str(encoder_path),
        "skill_checkpoint_sha256": _sha256(encoder_path),
        "expert_pin_latent_mse": z_mse,
        "expert_pin_latent_max_abs": pin_max_abs,
        "expert_pin_latent_value_count": pin_value_count,
        "checks": checks,
    }
    output_path = args.output_json.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.require_pass and not passed:
        raise SystemExit("Expert packet encoder pin audit failed.")


if __name__ == "__main__":
    main()
