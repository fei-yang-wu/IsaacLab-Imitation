#!/usr/bin/env python3
"""Validate the cross-segment enc380 5B completion record and artifact hashes."""

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


def audit(
    record_path: Path,
    *,
    low_level_checkpoint: Path,
    skill_checkpoint: Path,
) -> dict[str, Any]:
    record_path = record_path.expanduser().resolve()
    low_level_checkpoint = low_level_checkpoint.expanduser().resolve()
    skill_checkpoint = skill_checkpoint.expanduser().resolve()
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if (
        record.get("format") != "enc380_tracker_completion"
        or int(record.get("version", -1)) != 1
    ):
        raise ValueError("Unsupported enc380 tracker completion record.")
    before = int(record.get("credited_frames_before_segment", -1))
    segment = int(record.get("continuation_segment_frames", -1))
    cumulative = int(record.get("cumulative_credited_frames", -1))
    cap = int(record.get("frame_cap", -1))
    if before <= 0 or segment <= 0 or cumulative != before + segment:
        raise ValueError("Completion record has inconsistent frame accounting.")
    if cap < 5_000_000_000 or cumulative < cap:
        raise ValueError("enc380 tracker did not reach its >=5B frame cap.")
    if record.get("continuation_job_state") != "COMPLETED":
        raise ValueError("Continuation Slurm job did not complete successfully.")
    if str(record.get("continuation_exit_code")) != "0:0":
        raise ValueError("Continuation Slurm exit code is not 0:0.")
    recorded_low_level = (
        Path(str(record.get("final_checkpoint", ""))).expanduser().resolve()
    )
    recorded_skill = (
        Path(str(record.get("skill_checkpoint", ""))).expanduser().resolve()
    )
    low_level_sha = _sha256(low_level_checkpoint)
    skill_sha = _sha256(skill_checkpoint)
    if record.get("final_checkpoint_sha256") != low_level_sha:
        raise ValueError(
            "Final tracker checkpoint SHA-256 does not match completion record."
        )
    if record.get("skill_checkpoint_sha256") != skill_sha:
        raise ValueError(
            "Frozen skill checkpoint SHA-256 does not match completion record."
        )
    return {
        "format": "enc380_tracker_completion_audit",
        "version": 1,
        "passed": True,
        "source_record": str(record_path),
        "source_record_sha256": _sha256(record_path),
        "frame_cap": cap,
        "credited_frames_before_segment": before,
        "continuation_segment_frames": segment,
        "cumulative_credited_frames": cumulative,
        "continuation_job_id": int(record["continuation_job_id"]),
        "continuation_job_state": record["continuation_job_state"],
        "continuation_exit_code": record["continuation_exit_code"],
        "recorded_final_checkpoint": str(recorded_low_level),
        "final_checkpoint": str(low_level_checkpoint),
        "final_checkpoint_relocated": recorded_low_level != low_level_checkpoint,
        "final_checkpoint_sha256": low_level_sha,
        "recorded_skill_checkpoint": str(recorded_skill),
        "skill_checkpoint": str(skill_checkpoint),
        "skill_checkpoint_relocated": recorded_skill != skill_checkpoint,
        "skill_checkpoint_sha256": skill_sha,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completion_record", type=Path, required=True)
    parser.add_argument("--low_level_checkpoint", type=Path, required=True)
    parser.add_argument("--skill_checkpoint", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        args.completion_record,
        low_level_checkpoint=args.low_level_checkpoint,
        skill_checkpoint=args.skill_checkpoint,
    )
    output = args.output_json.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing existing completion audit: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        f"[PASS] enc380 cumulative frames={result['cumulative_credited_frames']} "
        f"-> {output}"
    )


if __name__ == "__main__":
    main()
