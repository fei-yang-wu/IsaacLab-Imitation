from __future__ import annotations

import json
from pathlib import Path

import pytest

from audit_enc380_motion_selection import audit


def _manifest(path: Path, names: list[str]) -> Path:
    path.write_text(
        json.dumps(
            {
                "dataset": {
                    "trajectories": {
                        "lafan1_csv": [
                            {"name": name, "path": f"{name}.npz", "input_fps": 50.0}
                            for name in names
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def _names() -> list[str]:
    names = [f"motion_{index:02d}" for index in range(1, 41)]
    names[28] = "walk1_subject1"
    return names


def test_accepts_frozen_manifest_positions(tmp_path: Path) -> None:
    result = audit(_manifest(tmp_path / "manifest.json", _names()))
    assert result["passed"] is True
    assert result["positions_one_based"] == [29]
    assert result["performance_data_used"] is True
    assert result["paper_representative_motion_selection"] is False


def test_rejects_reordered_manifest(tmp_path: Path) -> None:
    names = _names()
    names[28], names[29] = names[29], names[28]
    with pytest.raises(ValueError, match="does not match"):
        audit(_manifest(tmp_path / "manifest.json", names))
