from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch


SCRIPT = Path(__file__).with_name("compare_bones_seed_exports.py")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(root: Path, names: list[str]) -> Path:
    manifest = root / "manifests" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "dataset": {
                    "trajectories": {
                        "lafan1_csv": [
                            {"name": name, "path": f"../npz/{name}.npz"}
                            for name in names
                        ]
                    }
                },
                "metadata": {"paths_are_relative_to_manifest": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def _tree(root: Path, names: list[str]) -> Path:
    npz = root / "npz"
    npz.mkdir(parents=True)
    for index, name in enumerate(names):
        (npz / f"{name}.npz").write_bytes(f"motion-{index}".encode())
    return _manifest(root, names)


def test_compare_bones_seed_exports(tmp_path: Path) -> None:
    names = ["goal_a", "goal_b"]
    reference = _tree(tmp_path / "reference", names)
    candidate = _tree(tmp_path / "candidate", names)
    language = tmp_path / "language.pt"
    torch.save({"names": names, "embeddings": torch.zeros(2, 4)}, language)
    candidate_entries = json.loads(candidate.read_text())["dataset"]["trajectories"][
        "lafan1_csv"
    ]
    prepared = []
    for entry in candidate_entries:
        path = (candidate.parent / entry["path"]).resolve()
        prepared.append({"name": entry["name"], "npz_sha256": _sha256(path)})
    preparation = tmp_path / "preparation.json"
    preparation.write_text(
        json.dumps(
            {
                "status": "complete",
                "artifacts": {
                    "manifest_sha256": _sha256(candidate),
                    "npz_files": prepared,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "comparison.json"
    command = [
        sys.executable,
        str(SCRIPT),
        "--reference-manifest",
        str(reference),
        "--candidate-manifest",
        str(candidate),
        "--language-embeddings",
        str(language),
        "--preparation-record",
        str(preparation),
        "--output-json",
        str(output),
    ]
    subprocess.run(command, check=True)
    assert json.loads(output.read_text())["passed"] is True

    (candidate.parent.parent / "npz" / "goal_b.npz").write_bytes(b"changed")
    failed = subprocess.run(command, check=False, capture_output=True, text=True)
    assert failed.returncode != 0
    assert "one or more NPZ files differ" in failed.stderr
