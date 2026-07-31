from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from audit_enc380_tracker_completion import audit


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    tracker = tmp_path / "tracker.pt"
    skill = tmp_path / "skill.pt"
    tracker.write_bytes(b"tracker")
    skill.write_bytes(b"skill")
    record = tmp_path / "completion.json"
    record.write_text(
        json.dumps(
            {
                "format": "enc380_tracker_completion",
                "version": 1,
                "frame_cap": 5_000_000_000,
                "credited_frames_before_segment": 4_300_111_872,
                "continuation_segment_frames": 699_973_632,
                "cumulative_credited_frames": 5_000_085_504,
                "continuation_job_id": 5549446,
                "continuation_job_state": "COMPLETED",
                "continuation_exit_code": "0:0",
                "final_checkpoint": str(tracker),
                "final_checkpoint_sha256": _sha(tracker),
                "skill_checkpoint": str(skill),
                "skill_checkpoint_sha256": _sha(skill),
            }
        ),
        encoding="utf-8",
    )
    return record, tracker, skill


def test_accepts_completed_cross_segment_tracker(tmp_path: Path) -> None:
    record, tracker, skill = _fixture(tmp_path)
    result = audit(record, low_level_checkpoint=tracker, skill_checkpoint=skill)
    assert result["passed"] is True
    assert result["cumulative_credited_frames"] == 5_000_085_504


def test_accepts_hash_identical_relocated_artifacts(tmp_path: Path) -> None:
    record, tracker, skill = _fixture(tmp_path)
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    relocated_tracker = relocated / "model_5b.pt"
    relocated_skill = relocated / "latest.pt"
    relocated_tracker.write_bytes(tracker.read_bytes())
    relocated_skill.write_bytes(skill.read_bytes())

    result = audit(
        record,
        low_level_checkpoint=relocated_tracker,
        skill_checkpoint=relocated_skill,
    )

    assert result["passed"] is True
    assert result["final_checkpoint_relocated"] is True
    assert result["skill_checkpoint_relocated"] is True
    assert result["recorded_final_checkpoint"] == str(tracker.resolve())
    assert result["recorded_skill_checkpoint"] == str(skill.resolve())


def test_rejects_running_job(tmp_path: Path) -> None:
    record, tracker, skill = _fixture(tmp_path)
    payload = json.loads(record.read_text())
    payload["continuation_job_state"] = "RUNNING"
    record.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="did not complete"):
        audit(record, low_level_checkpoint=tracker, skill_checkpoint=skill)


def test_rejects_tracker_hash_mismatch(tmp_path: Path) -> None:
    record, tracker, skill = _fixture(tmp_path)
    tracker.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        audit(record, low_level_checkpoint=tracker, skill_checkpoint=skill)
