from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).with_name("audit_bones_seed_phase5.py")


def _write_motion(path: Path, *, offset: tuple[float, float, float]) -> None:
    frames = 4
    root_pos = np.stack(
        (
            np.linspace(0.0, 0.3, frames),
            np.zeros(frames),
            np.ones(frames),
        ),
        axis=-1,
    ).astype(np.float32)
    body_pos = np.zeros((frames, 2, 3), dtype=np.float32)
    body_pos[:, 0] = root_pos + np.asarray(offset, dtype=np.float32)
    body_pos[:, 1] = root_pos + np.asarray((0.0, 0.2, 0.3), dtype=np.float32)
    np.savez(
        path,
        fps=np.asarray([50.0], dtype=np.float32),
        root_pos=root_pos,
        body_pos_w=body_pos,
        body_names=np.asarray(("pelvis", "right_hand"), dtype=np.str_),
    )


def _write_inputs(
    root: Path,
    *,
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    omit_language_name: str | None = None,
    with_events: bool = True,
) -> tuple[Path, Path]:
    motions_dir = root / "npz"
    motions_dir.mkdir(parents=True)
    names = ("walk", "wave")
    for name in names:
        _write_motion(motions_dir / f"{name}.npz", offset=offset)

    sidecar_path = root / "language.json"
    language_motions = []
    for name in reversed(names):
        if name == omit_language_name:
            continue
        events = (
            [{"start_time": 0.0, "end_time": 0.08, "description": f"begin {name}"}]
            if with_events
            else []
        )
        language_motions.append(
            {
                "name": name,
                "language_goal": f"perform {name}",
                "natural_descriptions": [f"a person will {name}"],
                "short_descriptions": [name],
                "events": events,
                "num_events": len(events),
            }
        )
    sidecar_path.write_text(
        json.dumps({"dataset_name": "bones_seed_test", "motions": language_motions}),
        encoding="utf-8",
    )

    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_name": "bones_seed_test",
                "dataset": {
                    "trajectories": {
                        "lafan1_csv": [
                            {
                                "name": name,
                                "path": f"npz/{name}.npz",
                                "input_fps": 50.0,
                            }
                            for name in names
                        ]
                    }
                },
                "metadata": {"language_annotations_path": "language.json"},
            }
        ),
        encoding="utf-8",
    )
    return manifest_path, sidecar_path


def _run(manifest: Path, report: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest",
            str(manifest),
            "--report",
            str(report),
            *extra,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_preflight_accepts_exact_coverage_with_different_order(tmp_path: Path) -> None:
    manifest, _ = _write_inputs(tmp_path)
    report_path = tmp_path / "report.json"

    completed = _run(manifest, report_path, "--require-temporal-events")

    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["summary"]["manifest_motion_count"] == 2
    assert report["summary"]["language_goal_count"] == 2
    assert report["summary"]["body_frame_passed_motion_count"] == 2
    assert report["language_sidecar"]["order_matches_manifest"] is False
    assert {item["code"] for item in report["warnings"]} == {"language_order_differs"}


def test_preflight_rejects_scene_grid_body_offset(tmp_path: Path) -> None:
    manifest, _ = _write_inputs(tmp_path, offset=(2.0, -3.0, 0.0))
    report_path = tmp_path / "report.json"

    completed = _run(manifest, report_path)

    assert completed.returncode == 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["passed"] is False
    body_failures = [
        item for item in report["failures"] if item["code"] == "body_frame_offset"
    ]
    assert len(body_failures) == 2
    assert all(item["scene_grid_like_offset"] is True for item in body_failures)
    assert report["body_frames"]["passed_motion_count"] == 0


def test_preflight_rejects_missing_language_and_events(tmp_path: Path) -> None:
    manifest, _ = _write_inputs(
        tmp_path,
        omit_language_name="wave",
        with_events=False,
    )
    report_path = tmp_path / "report.json"

    completed = _run(manifest, report_path, "--require-temporal-events")

    assert completed.returncode == 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    codes = {item["code"] for item in report["failures"]}
    assert "language_coverage_missing" in codes
    assert "temporal_events_missing" in codes
    assert report["language_sidecar"]["missing_manifest_names"] == ["wave"]
