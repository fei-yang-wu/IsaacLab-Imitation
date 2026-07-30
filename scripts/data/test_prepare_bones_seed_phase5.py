from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).with_name("prepare_bones_seed_phase5.py")
PREFLIGHT = Path(__file__).with_name("audit_bones_seed_phase5.py")


def _write_inputs(root: Path, *, omit_language: str | None = None) -> tuple[Path, Path]:
    csv_dir = root / "raw"
    csv_dir.mkdir()
    names = ("pick_up_001__A001", "walk_001__A002")
    for name in names:
        (csv_dir / f"{name}.csv").write_text("0,0,1\n", encoding="utf-8")

    motions = []
    for name in names:
        normalized = name.replace("__", "_")
        if normalized == omit_language:
            continue
        motions.append(
            {
                "name": normalized,
                "language_goal": f"perform {normalized}",
                "events": [
                    {"start_time": 0.0, "end_time": 0.1, "description": "start"}
                ],
                "num_events": 1,
            }
        )
    sidecar = root / "language.json"
    sidecar.write_text(json.dumps({"motions": motions}), encoding="utf-8")
    return csv_dir, sidecar


def _write_fake_exporter(path: Path) -> None:
    path.write_text(
        """\
import argparse
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--jobs_json", required=True)
parser.add_argument("--output_fps", type=float, required=True)
args, _ = parser.parse_known_args()
for job in json.loads(Path(args.jobs_json).read_text(encoding="utf-8")):
    output = Path(job["output_name"])
    output.parent.mkdir(parents=True, exist_ok=True)
    root = np.asarray([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32)
    body = np.zeros((2, 2, 3), dtype=np.float32)
    body[:, 0] = root
    body[:, 1] = root + np.asarray([0.0, 0.2, 0.3], dtype=np.float32)
    np.savez(
        output,
        fps=np.asarray([args.output_fps], dtype=np.float32),
        root_pos=root,
        body_pos_w=body,
        body_names=np.asarray(["pelvis", "right_hand"], dtype=np.str_),
    )
""",
        encoding="utf-8",
    )


def _run(
    csv_dir: Path,
    sidecar: Path,
    output_root: Path,
    exporter: Path,
    *extra: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--csv-dir",
            str(csv_dir),
            "--language-sidecar",
            str(sidecar),
            "--output-root",
            str(output_root),
            "--dataset-name",
            "bones_seed_test",
            "--batch-converter-script",
            str(exporter),
            "--preflight-script",
            str(PREFLIGHT),
            *extra,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_dry_run_validates_and_does_not_write(tmp_path: Path) -> None:
    csv_dir, sidecar = _write_inputs(tmp_path)
    exporter = tmp_path / "fake_exporter.py"
    _write_fake_exporter(exporter)
    output_root = tmp_path / "corrected"

    completed = _run(csv_dir, sidecar, output_root, exporter, "--dry-run")

    assert completed.returncode == 0, completed.stdout + completed.stderr
    plan = json.loads(completed.stdout)
    assert plan["motion_count"] == 2
    assert plan["requirements"]["fresh_output_root"] is True
    assert {motion["name"] for motion in plan["motions"]} == {
        "pick_up_001_A001",
        "walk_001_A002",
    }
    assert not output_root.exists()


def test_fresh_export_writes_audited_self_contained_tree(tmp_path: Path) -> None:
    csv_dir, sidecar = _write_inputs(tmp_path)
    exporter = tmp_path / "fake_exporter.py"
    _write_fake_exporter(exporter)
    output_root = tmp_path / "corrected"

    completed = _run(
        csv_dir,
        sidecar,
        output_root,
        exporter,
        "--require-temporal-events",
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    manifest_path = output_root / "manifests" / "g1_bones_seed_test_manifest.json"
    report_path = output_root / "reports" / "phase5_preflight.json"
    record_path = output_root / "preparation" / "preparation.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    language_path = output_root / "language" / sidecar.name
    language = json.loads(language_path.read_text(encoding="utf-8"))

    assert report["passed"] is True
    assert report["summary"]["body_frame_passed_motion_count"] == 2
    assert record["status"] == "complete"
    assert len(record["artifacts"]["npz_files"]) == 2
    assert (
        manifest["metadata"]["language_annotations_path"] == "../language/language.json"
    )
    assert language["manifest"] == "../manifests/g1_bones_seed_test_manifest.json"
    assert [motion["name"] for motion in language["motions"]] == [
        "pick_up_001_A001",
        "walk_001_A002",
    ]
    assert not Path(
        manifest["dataset"]["trajectories"]["lafan1_csv"][0]["path"]
    ).is_absolute()


def test_language_coverage_mismatch_fails_before_writing(tmp_path: Path) -> None:
    csv_dir, sidecar = _write_inputs(tmp_path, omit_language="walk_001_A002")
    exporter = tmp_path / "fake_exporter.py"
    _write_fake_exporter(exporter)
    output_root = tmp_path / "corrected"

    completed = _run(csv_dir, sidecar, output_root, exporter)

    assert completed.returncode != 0
    assert "CSV/language coverage differs" in completed.stderr
    assert not output_root.exists()


def test_existing_output_root_is_never_reused(tmp_path: Path) -> None:
    csv_dir, sidecar = _write_inputs(tmp_path)
    exporter = tmp_path / "fake_exporter.py"
    _write_fake_exporter(exporter)
    output_root = tmp_path / "corrected"
    output_root.mkdir()

    completed = _run(csv_dir, sidecar, output_root, exporter)

    assert completed.returncode != 0
    assert "never overwrites or resumes" in completed.stderr
