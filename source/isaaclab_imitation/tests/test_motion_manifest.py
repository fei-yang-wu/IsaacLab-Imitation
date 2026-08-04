from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

# Load by file path (motion_manifest is stdlib-only by design) so this test
# runs in any environment without importing the isaaclab_imitation package.
_MODULE_PATH = (
    Path(__file__).parent.parent
    / "isaaclab_imitation"
    / "tasks"
    / "manager_based"
    / "imitation"
    / "motion_manifest.py"
)
_MODULE_SPEC = importlib.util.spec_from_file_location("motion_manifest", _MODULE_PATH)
assert _MODULE_SPEC is not None and _MODULE_SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_MODULE_SPEC)
_MODULE_SPEC.loader.exec_module(_MODULE)

REPO_ROOT = Path(__file__).resolve().parents[3]
REAL_MANIFESTS = (
    REPO_ROOT / "data" / "unitree" / "manifests" / "g1_unitree_dance102_manifest.json",
    REPO_ROOT / "data" / "lafan1" / "manifests" / "g1_lafan1_manifest.json",
)

# Keys the read path may emit per normalized entry: the required loader keys
# plus the optional frame range and preserved provenance keys.
ALLOWED_ENTRY_KEYS = {
    "name",
    "path",
    "input_fps",
    "frame_range",
    "source_dataset",
    "source_motion_name",
}


@pytest.mark.parametrize(
    "manifest_path", REAL_MANIFESTS, ids=[path.stem for path in REAL_MANIFESTS]
)
def test_real_manifest_read_path_structural_invariants(manifest_path: Path) -> None:
    if not manifest_path.is_file():
        pytest.skip(f"manifest not present: {manifest_path}")

    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_entries = raw["dataset"]["trajectories"]["lafan1_csv"]

    resolved_path, entries = _MODULE.load_clip_manifest(manifest_path)

    assert resolved_path == manifest_path.resolve()
    assert len(entries) == len(raw_entries)
    for raw_entry, entry in zip(raw_entries, entries, strict=True):
        assert set(entry.keys()) <= ALLOWED_ENTRY_KEYS
        assert {"name", "path", "input_fps"} <= set(entry.keys())
        assert isinstance(entry["name"], str) and entry["name"]
        assert isinstance(entry["input_fps"], float)
        assert entry["input_fps"] == float(raw_entry["input_fps"])
        resolved = Path(entry["path"])
        assert resolved.is_absolute()
        # Relative manifest paths resolve against the manifest directory.
        assert resolved == (manifest_path.parent / raw_entry["path"]).resolve()
        assert ("frame_range" in entry) == ("frame_range" in raw_entry)
        if "frame_range" in raw_entry:
            assert entry["frame_range"] == raw_entry["frame_range"]

    # read_manifest accepts legacy blobs without family/role.
    payload = _MODULE.read_manifest(manifest_path)
    assert payload["dataset"]["trajectories"]["lafan1_csv"] == raw_entries
    assert _MODULE.manifest_family(payload) is None or isinstance(
        _MODULE.manifest_family(payload), str
    )


def test_write_manifest_read_manifest_round_trip(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifests" / "g1_test_manifest.json"
    entries = [
        {"name": "walk1", "path": "../npz/walk1.npz", "input_fps": 50.0},
        {
            "name": "dance1",
            "path": "../npz/dance1.npz",
            "input_fps": 50.0,
            "frame_range": [1, 100],
        },
    ]
    metadata = {"num_motions": 2, "custom_extra": {"nested": True}}

    payload = _MODULE.write_manifest(
        manifest_path,
        dataset_name="bones_seed",
        entries=entries,
        metadata=metadata,
        family="bones_seed",
        role="headline",
    )

    text = manifest_path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert json.dumps(json.loads(text), indent=2, sort_keys=True) + "\n" == text

    read_back = _MODULE.read_manifest(manifest_path)
    assert read_back == payload
    assert read_back["dataset_name"] == "bones_seed"
    assert read_back["dataset"]["trajectories"]["lafan1_csv"] == entries
    assert read_back["metadata"]["num_motions"] == 2
    assert read_back["metadata"]["custom_extra"] == {"nested": True}
    assert _MODULE.manifest_family(read_back) == "bones_seed"
    assert _MODULE.manifest_role(read_back) == "headline"
    # The caller's metadata dict is not mutated by the family/role merge.
    assert "family" not in metadata and "role" not in metadata


def test_manifest_family_and_role_are_free_form_labels(tmp_path: Path) -> None:
    """A new dataset needs no code change to describe itself.

    family/role name the *data* -- which lineage a manifest describes and what
    it is for -- and nothing branches on them, so they are carried through
    verbatim rather than checked against a closed set that a new dataset would
    have to be added to first.
    """
    entries = [{"name": "m", "path": "m.npz", "input_fps": 50.0}]
    payload = _MODULE.write_manifest(
        tmp_path / "novel.json",
        dataset_name="novel_capture",
        entries=entries,
        metadata={},
        family="novel_capture",
        role="pilot",
    )
    assert _MODULE.manifest_family(payload) == "novel_capture"
    assert _MODULE.manifest_role(payload) == "pilot"
    assert _MODULE.read_manifest(tmp_path / "novel.json") == payload

    # Role is optional; omitting it records no role rather than inventing one.
    payload = _MODULE.write_manifest(
        tmp_path / "no_role.json",
        dataset_name="novel_capture",
        entries=entries,
        metadata={},
        family="novel_capture",
    )
    assert _MODULE.manifest_role(payload) is None

    with pytest.raises(ValueError, match="family"):
        _MODULE.write_manifest(
            tmp_path / "blank.json",
            dataset_name="x",
            entries=entries,
            metadata={},
            family="   ",
        )


def test_normalize_preserves_provenance_keys_and_drops_other_extras() -> None:
    entries = _MODULE.normalize_clip_entries(
        [
            {
                "name": "walk1",
                "path": "/tmp/does/not/need/to/exist/walk1.npz",
                "input_fps": 50,
                "source_dataset": "lafan1",
                "source_motion_name": "walk1_subject1",
                "unrelated_extra": "dropped",
            }
        ]
    )

    assert len(entries) == 1
    entry = entries[0]
    assert entry["source_dataset"] == "lafan1"
    assert entry["source_motion_name"] == "walk1_subject1"
    assert "unrelated_extra" not in entry
    assert entry["input_fps"] == 50.0


def test_provenance_keys_survive_loader_kwargs_build() -> None:
    loader_kwargs = _MODULE.build_clip_loader_kwargs(
        entries=[
            {
                "name": "walk1",
                "path": "/tmp/x/walk1.npz",
                "input_fps": 50.0,
                "source_dataset": "bones_seed",
                "source_motion_name": "orig_walk",
            }
        ],
        sim_dt=0.005,
        decimation=4,
        joint_names=["j0"],
    )

    entry = loader_kwargs["dataset"]["trajectories"]["lafan1_csv"][0]
    assert entry["source_dataset"] == "bones_seed"
    assert entry["source_motion_name"] == "orig_walk"


def test_cache_dir_from_entries_family_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ISAACLAB_IMITATION_MOTION_CACHE_ROOT", str(tmp_path))
    entries = [{"name": "m", "path": "/tmp/x/m.npz", "input_fps": 50.0}]
    manifest_path = tmp_path / "g1_bones_manifest.json"

    default_path = Path(
        _MODULE.cache_dir_from_entries(entries, manifest_path=manifest_path)
    )
    bones_path = Path(
        _MODULE.cache_dir_from_entries(
            entries, manifest_path=manifest_path, family="bones_seed"
        )
    )

    assert default_path.name.startswith("iltools_g1_lafan1_tracking_")
    assert bones_path.name.startswith("iltools_g1_bones_seed_tracking_")
    # The content digest is unchanged by the family prefix.
    assert default_path.name.split("_")[-1] == bones_path.name.split("_")[-1]
    assert default_path.parent == bones_path.parent == tmp_path


def test_load_manifest_family_reads_written_family(tmp_path: Path) -> None:
    manifest_path = tmp_path / "fam.json"
    _MODULE.write_manifest(
        manifest_path,
        dataset_name="unified",
        entries=[{"name": "m", "path": "m.npz", "input_fps": 50.0}],
        metadata={},
        family="unified",
        role="testing",
    )
    assert _MODULE.load_manifest_family(manifest_path) == "unified"
    assert _MODULE.load_manifest_family(tmp_path / "missing.json") is None
