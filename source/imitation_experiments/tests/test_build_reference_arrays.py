"""Tests for the NPZ -> training-shaped reference-array builder.

The two properties that matter at 129,785 trajectories are that worker count
cannot change the bytes on disk, and that a shard cannot silently write the
wrong span. Both are exercised here on a synthetic tree small enough to compare
exhaustively.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from imitation_experiments.data.build_reference_arrays import (
    SIDECAR_NAME,
    WXYZ_TO_XYZW,
    build_reference_arrays,
    load_npz_paths,
    npz_member_shape,
    open_reference_arrays,
    validation_errors,
    verify_against_source,
)


BODY_NAMES = ("pelvis", "torso_link", "left_knee_link")
ANCHOR = "torso_link"
NUM_SOURCE_BODIES = 5
SOURCE_BODY_NAMES = (
    "pelvis",
    "unused_a",
    "torso_link",
    "unused_b",
    "left_knee_link",
)
PERSIST_ID = "synthetic@deadbeef"


def _write_npz(
    path: Path, *, frames: int, seed: int, drop: tuple[str, ...] = ()
) -> None:
    rng = np.random.default_rng(seed)
    payload = {
        "fps": np.asarray([50.0], dtype=np.float32),
        "qpos": rng.standard_normal((frames, 36), dtype=np.float32),
        "qvel": rng.standard_normal((frames, 35), dtype=np.float32),
        "root_pos": rng.standard_normal((frames, 3), dtype=np.float32),
        "root_quat": rng.standard_normal((frames, 4), dtype=np.float32),
        "root_lin_vel": rng.standard_normal((frames, 3), dtype=np.float32),
        "root_ang_vel": rng.standard_normal((frames, 3), dtype=np.float32),
        "joint_pos": rng.standard_normal((frames, 29), dtype=np.float32),
        "joint_vel": rng.standard_normal((frames, 29), dtype=np.float32),
        "body_pos_w": rng.standard_normal(
            (frames, NUM_SOURCE_BODIES, 3), dtype=np.float32
        ),
        "body_quat_w": rng.standard_normal(
            (frames, NUM_SOURCE_BODIES, 4), dtype=np.float32
        ),
        "body_lin_vel_w": rng.standard_normal(
            (frames, NUM_SOURCE_BODIES, 3), dtype=np.float32
        ),
        "body_ang_vel_w": rng.standard_normal(
            (frames, NUM_SOURCE_BODIES, 3), dtype=np.float32
        ),
        "joint_names": np.asarray([f"joint_{i}" for i in range(29)]),
        "body_names": np.asarray(SOURCE_BODY_NAMES),
    }
    for name in drop:
        payload.pop(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    # savez, not savez_compressed: the real tree is STORED, and the builder's
    # header-only shape reads depend on that.
    np.savez(path, **payload)


def _write_tree(
    tmp_path: Path, lengths: dict[str, int], *, drop: tuple[str, ...] = ()
) -> Path:
    npz_dir = tmp_path / "npz"
    entries = []
    for index, (name, frames) in enumerate(lengths.items()):
        _write_npz(npz_dir / f"{name}.npz", frames=frames, seed=index, drop=drop)
        entries.append({"name": name, "path": f"../npz/{name}.npz", "input_fps": 50.0})
    manifest = tmp_path / "manifests" / "synthetic.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "dataset_name": "synthetic",
                "dataset": {"trajectories": {"lafan1_csv": entries}},
                "metadata": {
                    "num_motions": len(entries),
                    "paths_are_relative_to_manifest": True,
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest


@pytest.fixture
def source_tree(tmp_path: Path) -> tuple[Path, dict[str, int]]:
    """A three-motion NPZ tree with uneven lengths, plus its manifest."""
    lengths = {"walk": 11, "wave": 7, "jump": 23}
    return _write_tree(tmp_path, lengths), lengths


def _build(manifest: Path, output_dir: Path, workers: int) -> dict:
    return build_reference_arrays(
        manifest=manifest,
        output_dir=output_dir,
        persist_id=PERSIST_ID,
        body_names=list(BODY_NAMES),
        anchor_body=ANCHOR,
        workers=workers,
    )


def test_worker_count_does_not_change_the_bytes(
    source_tree: tuple[Path, dict[str, int]], tmp_path: Path
) -> None:
    manifest, lengths = source_tree
    serial = tmp_path / "serial"
    parallel = tmp_path / "parallel"

    sidecar_serial = _build(manifest, serial, workers=1)
    sidecar_parallel = _build(manifest, parallel, workers=3)

    assert sidecar_serial == sidecar_parallel
    for path in sorted(serial.glob("*.memmap")):
        assert path.read_bytes() == (parallel / path.name).read_bytes(), path.name

    expected_rows = sum(frames - 1 for frames in lengths.values())
    assert sidecar_serial["traj_info"]["written"] == expected_rows
    assert sidecar_serial["traj_info"]["end_index"][-1] == expected_rows


def test_spans_are_contiguous_and_match_source_lengths(
    source_tree: tuple[Path, dict[str, int]], tmp_path: Path
) -> None:
    manifest, lengths = source_tree
    output_dir = tmp_path / "arrays"
    sidecar = _build(manifest, output_dir, workers=3)

    traj_info = sidecar["traj_info"]
    starts = traj_info["start_index"]
    ends = traj_info["end_index"]
    ordered = [entry[1] for entry in traj_info["ordered_traj_list"]]

    assert ordered == list(lengths)
    assert starts[0] == 0
    for previous_end, start in zip(ends, starts[1:]):
        assert start == previous_end
    for motion, start, end in zip(ordered, starts, ends):
        assert end - start == lengths[motion] - 1


def test_arrays_match_the_source_including_quaternion_conventions(
    source_tree: tuple[Path, dict[str, int]], tmp_path: Path
) -> None:
    manifest, _lengths = source_tree
    output_dir = tmp_path / "arrays"
    sidecar = _build(manifest, output_dir, workers=2)

    arrays, _ = open_reference_arrays(output_dir)
    npz_paths = load_npz_paths(manifest)
    body_ids = [SOURCE_BODY_NAMES.index(name) for name in BODY_NAMES]
    anchor_id = SOURCE_BODY_NAMES.index(ANCHOR)

    for entry, start, end in zip(
        sidecar["traj_info"]["ordered_traj_list"],
        sidecar["traj_info"]["start_index"],
        sidecar["traj_info"]["end_index"],
    ):
        motion = entry[1]
        with np.load(npz_paths[motion]) as npz:
            rows = end - start
            assert np.array_equal(arrays["qpos"][start:end], npz["qpos"][:rows])
            assert np.array_equal(arrays["qvel"][start:end], npz["qvel"][:rows])
            for name in ("body_pos_w", "body_lin_vel_w", "body_ang_vel_w"):
                assert np.array_equal(
                    arrays[name][start:end], npz[name][:rows, body_ids]
                ), name
            # body_quat_w keeps the dataset's WXYZ order ...
            assert np.array_equal(
                arrays["body_quat_w"][start:end], npz["body_quat_w"][:rows, body_ids]
            )
            # ... while anchor_quat_w is pre-swizzled to XYZW.
            assert np.array_equal(
                arrays["anchor_quat_w"][start:end],
                npz["body_quat_w"][:rows, anchor_id][:, WXYZ_TO_XYZW],
            )
            assert np.array_equal(
                arrays["anchor_pos_w"][start:end], npz["body_pos_w"][:rows, anchor_id]
            )

    assert sidecar["key"]["arrays"]["anchor_quat_w"]["quaternion_order"] == "xyzw"
    assert sidecar["key"]["arrays"]["body_quat_w"]["quaternion_order"] == "wxyz"


def test_validation_and_verification_pass_on_a_fresh_build(
    source_tree: tuple[Path, dict[str, int]], tmp_path: Path
) -> None:
    manifest, lengths = source_tree
    output_dir = tmp_path / "arrays"
    _build(manifest, output_dir, workers=2)

    assert (
        validation_errors(
            output_dir,
            persist_id=PERSIST_ID,
            body_names=list(BODY_NAMES),
            anchor_body=ANCHOR,
            expected_motions=len(lengths),
            expected_transitions=sum(frames - 1 for frames in lengths.values()),
        )
        == []
    )
    verify_against_source(output_dir, manifest=manifest, samples=3)


def test_validation_rejects_a_different_body_set_or_anchor(
    source_tree: tuple[Path, dict[str, int]], tmp_path: Path
) -> None:
    manifest, _lengths = source_tree
    output_dir = tmp_path / "arrays"
    _build(manifest, output_dir, workers=1)

    wrong_bodies = validation_errors(
        output_dir,
        persist_id=PERSIST_ID,
        body_names=["pelvis", "torso_link"],
        anchor_body=ANCHOR,
    )
    assert any("body_names" in error for error in wrong_bodies)

    wrong_anchor = validation_errors(
        output_dir,
        persist_id=PERSIST_ID,
        body_names=list(BODY_NAMES),
        anchor_body="pelvis",
    )
    assert any("anchor_body" in error for error in wrong_anchor)

    wrong_id = validation_errors(
        output_dir,
        persist_id="other@0000",
        body_names=list(BODY_NAMES),
        anchor_body=ANCHOR,
    )
    assert any("persist_id" in error for error in wrong_id)


def test_verification_catches_a_corrupted_row(
    source_tree: tuple[Path, dict[str, int]], tmp_path: Path
) -> None:
    manifest, _lengths = source_tree
    output_dir = tmp_path / "arrays"
    _build(manifest, output_dir, workers=1)

    # Sizes and the sidecar stay valid, so only a content check can see this.
    qpos = np.memmap(
        output_dir / "qpos.memmap",
        dtype=np.float32,
        mode="r+",
        shape=tuple(sidecar_shape(output_dir, "qpos")),
    )
    qpos[0, 0] += np.float32(1.0)
    qpos.flush()
    del qpos

    assert (
        validation_errors(
            output_dir,
            persist_id=PERSIST_ID,
            body_names=list(BODY_NAMES),
            anchor_body=ANCHOR,
        )
        == []
    )
    with pytest.raises(RuntimeError, match="does not match the source NPZ"):
        verify_against_source(output_dir, manifest=manifest, samples=3)


def sidecar_shape(output_dir: Path, name: str) -> list[int]:
    sidecar = json.loads((output_dir / SIDECAR_NAME).read_text(encoding="utf-8"))
    return [int(value) for value in sidecar["key"]["arrays"][name]["shape"]]


def test_build_refuses_to_overwrite_a_complete_cache(
    source_tree: tuple[Path, dict[str, int]], tmp_path: Path
) -> None:
    manifest, _lengths = source_tree
    output_dir = tmp_path / "arrays"
    _build(manifest, output_dir, workers=1)

    with pytest.raises(RuntimeError, match="fresh, versioned directory"):
        _build(manifest, output_dir, workers=1)


def test_traj_info_sidecar_fixes_the_order_and_offsets(
    source_tree: tuple[Path, dict[str, int]], tmp_path: Path
) -> None:
    """A replay sidecar's order wins over manifest order, so ranks stay stable."""
    manifest, lengths = source_tree
    reversed_names = list(reversed(list(lengths)))
    starts, ends, cursor = [], [], 0
    for name in reversed_names:
        starts.append(cursor)
        cursor += lengths[name] - 1
        ends.append(cursor)

    traj_info_path = tmp_path / "iltools_rb_manifest.json"
    traj_info_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "key": {"source": {"persist_id": PERSIST_ID}},
                "traj_info": {
                    "capacity": cursor,
                    "written": cursor,
                    "start_index": starts,
                    "end_index": ends,
                    "ordered_traj_list": [
                        ["lafan1", name, "trajectory_0"] for name in reversed_names
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "arrays"
    sidecar = build_reference_arrays(
        manifest=manifest,
        output_dir=output_dir,
        persist_id=PERSIST_ID,
        body_names=list(BODY_NAMES),
        anchor_body=ANCHOR,
        traj_info=traj_info_path,
        workers=2,
    )

    assert [
        entry[1] for entry in sidecar["traj_info"]["ordered_traj_list"]
    ] == reversed_names
    verify_against_source(output_dir, manifest=manifest, samples=3)


def test_npz_member_shape_reads_only_the_header(
    source_tree: tuple[Path, dict[str, int]],
) -> None:
    manifest, lengths = source_tree
    paths = load_npz_paths(manifest)
    assert npz_member_shape(paths["walk"], "qpos") == (lengths["walk"], 36)
    assert npz_member_shape(paths["jump"], "body_quat_w") == (
        lengths["jump"],
        NUM_SOURCE_BODIES,
        4,
    )


def test_defaults_keep_every_body_and_write_no_anchor(tmp_path: Path) -> None:
    """The dataset-agnostic default: all bodies, no anchor, no assumed names."""
    manifest = _write_tree(tmp_path, {"walk": 9})
    output_dir = tmp_path / "arrays"
    sidecar = build_reference_arrays(
        manifest=manifest,
        output_dir=output_dir,
        persist_id=PERSIST_ID,
        workers=1,
    )

    assert sidecar["key"]["body_names"] == list(SOURCE_BODY_NAMES)
    assert sidecar["key"]["anchor_body"] is None
    assert "anchor_pos_w" not in sidecar["key"]["arrays"]
    assert "anchor_quat_w" not in sidecar["key"]["arrays"]
    assert sidecar["key"]["arrays"]["body_pos_w"]["shape"][1] == NUM_SOURCE_BODIES
    verify_against_source(output_dir, manifest=manifest, samples=1)


def test_a_source_without_body_velocities_yields_a_smaller_array_set(
    tmp_path: Path,
) -> None:
    """Not every CSV-derived tree carries every field; that is not an error."""
    manifest = _write_tree(
        tmp_path, {"walk": 9}, drop=("body_lin_vel_w", "body_ang_vel_w")
    )
    output_dir = tmp_path / "arrays"
    sidecar = build_reference_arrays(
        manifest=manifest,
        output_dir=output_dir,
        persist_id=PERSIST_ID,
        body_names=list(BODY_NAMES),
        anchor_body=ANCHOR,
        workers=1,
    )

    assert set(sidecar["key"]["arrays"]) == {
        "qpos",
        "qvel",
        "body_pos_w",
        "body_quat_w",
        "anchor_pos_w",
        "anchor_quat_w",
    }
    verify_against_source(output_dir, manifest=manifest, samples=1)


def test_widths_come_from_the_data_not_from_a_constant(tmp_path: Path) -> None:
    """A different robot -- other qpos width, other body count -- just works."""
    rng = np.random.default_rng(7)
    frames, joints, bodies = 12, 6, 3
    npz_dir = tmp_path / "npz"
    npz_dir.mkdir(parents=True)
    np.savez(
        npz_dir / "clip.npz",
        qpos=rng.standard_normal((frames, 7 + joints), dtype=np.float32),
        qvel=rng.standard_normal((frames, 6 + joints), dtype=np.float32),
        body_pos_w=rng.standard_normal((frames, bodies, 3), dtype=np.float32),
        body_quat_w=rng.standard_normal((frames, bodies, 4), dtype=np.float32),
        joint_names=np.asarray([f"j{i}" for i in range(joints)]),
        body_names=np.asarray(["base", "arm", "tool"]),
    )
    manifest = tmp_path / "m.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset": {
                    "trajectories": {
                        "lafan1_csv": [
                            {"name": "clip", "path": "npz/clip.npz", "input_fps": 50.0}
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "arrays"
    sidecar = build_reference_arrays(
        manifest=manifest,
        output_dir=output_dir,
        persist_id="other-robot@0001",
        anchor_body="base",
        dataset_name="somedataset",
        workers=1,
    )

    arrays = sidecar["key"]["arrays"]
    assert arrays["qpos"]["shape"] == [frames - 1, 7 + joints]
    assert arrays["qvel"]["shape"] == [frames - 1, 6 + joints]
    assert arrays["body_pos_w"]["shape"] == [frames - 1, bodies, 3]
    assert sidecar["key"]["joint_names"] == [f"j{i}" for i in range(joints)]
    assert sidecar["traj_info"]["ordered_traj_list"][0][0] == "somedataset"
    verify_against_source(output_dir, manifest=manifest, samples=1)


def test_multi_take_motions_are_refused_rather_than_duplicated(
    source_tree: tuple[Path, dict[str, int]], tmp_path: Path
) -> None:
    manifest, lengths = source_tree
    traj_info_path = tmp_path / "iltools_rb_manifest.json"
    rows = lengths["walk"] - 1
    traj_info_path.write_text(
        json.dumps(
            {
                "traj_info": {
                    "start_index": [0, rows],
                    "end_index": [rows, 2 * rows],
                    "ordered_traj_list": [
                        ["lafan1", "walk", "trajectory_0"],
                        ["lafan1", "walk", "trajectory_1"],
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="appears more than once"):
        build_reference_arrays(
            manifest=manifest,
            output_dir=tmp_path / "arrays",
            persist_id=PERSIST_ID,
            traj_info=traj_info_path,
            workers=1,
        )


def test_anchor_body_absent_from_the_dataset_fails_loudly(
    source_tree: tuple[Path, dict[str, int]], tmp_path: Path
) -> None:
    manifest, _lengths = source_tree
    with pytest.raises(RuntimeError, match="not in the dataset's bodies"):
        build_reference_arrays(
            manifest=manifest,
            output_dir=tmp_path / "arrays",
            persist_id=PERSIST_ID,
            anchor_body="no_such_link",
            workers=1,
        )


def test_npz_root_override_relocates_the_tree(
    source_tree: tuple[Path, dict[str, int]], tmp_path: Path
) -> None:
    manifest, lengths = source_tree
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    for name in lengths:
        (relocated / f"{name}.npz").write_bytes(
            (manifest.parent.parent / "npz" / f"{name}.npz").read_bytes()
        )

    paths = load_npz_paths(manifest, npz_root=relocated)
    assert all(path.parent == relocated for path in paths.values())

    output_dir = tmp_path / "arrays"
    build_reference_arrays(
        manifest=manifest,
        output_dir=output_dir,
        persist_id=PERSIST_ID,
        body_names=list(BODY_NAMES),
        anchor_body=ANCHOR,
        npz_root=relocated,
        workers=2,
    )
    verify_against_source(output_dir, manifest=manifest, npz_root=relocated, samples=3)
