"""The environment side of the prebuilt reference arrays.

The builder's own tests prove the bytes on disk are right. What matters here is
that loading them produces exactly what the two derived caches would have
produced from a replay -- same columns, same order, same quaternion conventions
-- and that a directory built for different content is refused rather than read.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from imitation_experiments.data.build_reference_arrays import build_reference_arrays
from isaaclab_imitation.envs.expert_data_plane import ExpertDataPlane
from isaaclab_imitation.envs.reference_arrays import ReferenceArrayStore
from isaaclab_imitation.tasks.manager_based.imitation.motion_data import MotionDataCfg


SOURCE_BODY_NAMES = ("pelvis", "spare_link", "torso_link", "left_knee_link")
TRACKED = ["pelvis", "torso_link", "left_knee_link"]
ANCHOR = "torso_link"
PERSIST_ID = "synthetic@feedface"
LENGTHS = {"walk": 9, "wave": 6}
NUM_JOINTS = 5


def _write_tree(tmp_path: Path) -> Path:
    npz_dir = tmp_path / "npz"
    npz_dir.mkdir(parents=True)
    entries = []
    for index, (name, frames) in enumerate(LENGTHS.items()):
        rng = np.random.default_rng(index)
        bodies = len(SOURCE_BODY_NAMES)
        np.savez(
            npz_dir / f"{name}.npz",
            qpos=rng.standard_normal((frames, 7 + NUM_JOINTS), dtype=np.float32),
            qvel=rng.standard_normal((frames, 6 + NUM_JOINTS), dtype=np.float32),
            body_pos_w=rng.standard_normal((frames, bodies, 3), dtype=np.float32),
            body_quat_w=rng.standard_normal((frames, bodies, 4), dtype=np.float32),
            body_lin_vel_w=rng.standard_normal((frames, bodies, 3), dtype=np.float32),
            body_ang_vel_w=rng.standard_normal((frames, bodies, 3), dtype=np.float32),
            joint_names=np.asarray([f"j{i}" for i in range(NUM_JOINTS)]),
            body_names=np.asarray(SOURCE_BODY_NAMES),
        )
        entries.append({"name": name, "path": f"npz/{name}.npz", "input_fps": 50.0})

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"dataset": {"trajectories": {"lafan1_csv": entries}}}),
        encoding="utf-8",
    )
    return manifest


@pytest.fixture
def built(tmp_path: Path) -> tuple[Path, Path]:
    manifest = _write_tree(tmp_path)
    output_dir = tmp_path / "arrays"
    build_reference_arrays(
        manifest=manifest,
        output_dir=output_dir,
        persist_id=PERSIST_ID,
        body_names=list(TRACKED),
        anchor_body=ANCHOR,
        workers=1,
    )
    return manifest, output_dir


def _plane(output_dir: Path, *, warm_workers: int = 2) -> ExpertDataPlane:
    plane = ExpertDataPlane.__new__(ExpertDataPlane)
    plane._expert_anchor_body_name = ANCHOR
    data_cfg = SimpleNamespace(
        runtime_cache_body_names=list(TRACKED),
        runtime_cache_device="cpu",
        runtime_cache_chunk_size=4,
        storage_device="cpu",
        persist_id=PERSIST_ID,
        reference_arrays_warm_workers=warm_workers,
        macro_cache_chunk_size=4,
        macro_cache_device="cpu",
    )
    plane._env = SimpleNamespace(
        cfg=SimpleNamespace(
            data=data_cfg,
            expert_macro_state_terms=[
                "expert_motion_qpos",
                "expert_anchor_pos_b",
                "expert_anchor_ori_b",
            ],
        )
    )
    return plane


def _load(plane: ExpertDataPlane, output_dir: Path):
    cfg = SimpleNamespace(
        mpjpe_metric_body_names=list(TRACKED),
        command_ee_body_names=[],
        command_keypoint_body_names=[],
    )
    return plane._open_reference_arrays(
        data_cfg=plane._env.cfg.data,
        cfg=cfg,
        resolved=SimpleNamespace(reference_arrays_dir=str(output_dir)),
    )


def test_loading_reproduces_the_runtime_cache_columns(built) -> None:
    manifest, output_dir = built
    rb, traj_info, body_names, site_names, joint_names = _load(
        _plane(output_dir), output_dir
    )

    assert body_names == TRACKED
    assert site_names == []
    assert joint_names == [f"j{i}" for i in range(NUM_JOINTS)]
    assert [entry[1] for entry in traj_info["ordered_traj_list"]] == list(LENGTHS)
    assert traj_info["written"] == sum(frames - 1 for frames in LENGTHS.values())

    source = rb._storage._storage
    body_ids = [SOURCE_BODY_NAMES.index(name) for name in TRACKED]
    starts, ends = traj_info["start_index"], traj_info["end_index"]
    for motion, start, end in zip(LENGTHS, starts, ends):
        with np.load(manifest.parent / "npz" / f"{motion}.npz") as npz:
            rows = end - start
            for key in ("qpos", "qvel"):
                assert torch.equal(
                    source[key][start:end], torch.from_numpy(npz[key][:rows])
                ), key
            for key in (
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ):
                assert torch.equal(
                    source[key][start:end],
                    torch.from_numpy(npz[key][:rows, body_ids]),
                ), key


def test_macro_cache_matches_a_direct_derivation(built) -> None:
    """The prebuilt anchor arrays must equal what the replay path computes.

    The replay path reads all bodies and swizzles WXYZ -> XYZW at materialize
    time; the builder did that once, offline. Same numbers or the encoder sees a
    different macro state than it was trained on.
    """
    manifest, output_dir = built
    plane = _plane(output_dir)
    rb, traj_info, _bodies, _sites, _joints = _load(plane, output_dir)
    total = traj_info["written"]
    plane.trajectory_manager = SimpleNamespace(
        rb=rb, end=torch.tensor(traj_info["end_index"], dtype=torch.long)
    )

    cache = plane._ensure_root_qpos_macro_cache()
    assert cache is not None

    starts, ends = traj_info["start_index"], traj_info["end_index"]
    anchor_id = SOURCE_BODY_NAMES.index(ANCHOR)
    for motion, start, end in zip(LENGTHS, starts, ends):
        with np.load(manifest.parent / "npz" / f"{motion}.npz") as npz:
            rows = end - start
            assert torch.equal(
                cache["joint_pos"][start:end],
                torch.from_numpy(npz["qpos"][:rows, 7:]),
            )
            assert torch.equal(
                cache["anchor_pos_w"][start:end],
                torch.from_numpy(npz["body_pos_w"][:rows, anchor_id]),
            )
            assert torch.equal(
                cache["anchor_quat_w"][start:end],
                torch.from_numpy(npz["body_quat_w"][:rows, anchor_id][:, [1, 2, 3, 0]]),
            )
    assert cache["joint_pos"].shape == (total, NUM_JOINTS)


def test_worker_count_does_not_change_what_is_loaded(built) -> None:
    _manifest, output_dir = built
    serial = _load(_plane(output_dir, warm_workers=1), output_dir)[0]._storage._storage
    parallel = _load(_plane(output_dir, warm_workers=4), output_dir)[
        0
    ]._storage._storage
    for key in serial.keys():
        assert torch.equal(serial[key], parallel[key]), key


def test_a_different_anchor_inside_the_body_set_is_derived_not_refused(
    built,
) -> None:
    """A published artifact must serve every anchor it retains.

    The anchor pose is also in body_pos_w/body_quat_w, so baking one anchor
    must not force a second 49.4 GB upload for the other anchor in use.
    """
    manifest, output_dir = built
    other = "pelvis"  # built with ANCHOR == "torso_link"
    assert other in TRACKED and other != ANCHOR

    store = ReferenceArrayStore.open(
        output_dir, body_names=list(TRACKED), anchor_body=other
    )
    assert store.anchor_source(other) == TRACKED.index(other)
    assert store.anchor_source(ANCHOR) is None

    plane = _plane(output_dir)
    plane._expert_anchor_body_name = other
    rb, traj_info, _b, _s, _j = _load(plane, output_dir)
    plane.trajectory_manager = SimpleNamespace(
        rb=rb, end=torch.tensor(traj_info["end_index"], dtype=torch.long)
    )
    cache = plane._ensure_root_qpos_macro_cache()

    source_id = SOURCE_BODY_NAMES.index(other)
    for motion, start, end in zip(
        LENGTHS, traj_info["start_index"], traj_info["end_index"]
    ):
        with np.load(manifest.parent / "npz" / f"{motion}.npz") as npz:
            rows = end - start
            assert torch.equal(
                cache["anchor_pos_w"][start:end],
                torch.from_numpy(npz["body_pos_w"][:rows, source_id]),
            )
            assert torch.equal(
                cache["anchor_quat_w"][start:end],
                torch.from_numpy(npz["body_quat_w"][:rows, source_id][:, [1, 2, 3, 0]]),
            )


def test_an_anchor_outside_the_body_set_is_still_refused(built) -> None:
    _manifest, output_dir = built
    with pytest.raises(RuntimeError, match="cannot serve an environment anchored"):
        ReferenceArrayStore.open(
            output_dir, body_names=list(TRACKED), anchor_body="spare_link"
        )


def test_a_directory_built_for_other_content_is_refused(built) -> None:
    _manifest, output_dir = built

    with pytest.raises(RuntimeError, match="column positions"):
        ReferenceArrayStore.open(
            output_dir, body_names=["pelvis", "torso_link"], anchor_body=ANCHOR
        )
    with pytest.raises(RuntimeError, match="env.data.persist_id"):
        ReferenceArrayStore.open(
            output_dir,
            body_names=list(TRACKED),
            anchor_body=ANCHOR,
            persist_id="something-else@0000",
        )
    # Body order is a column mapping, so a permutation is a different artifact.
    with pytest.raises(RuntimeError, match="column positions"):
        ReferenceArrayStore.open(
            output_dir, body_names=list(reversed(TRACKED)), anchor_body=ANCHOR
        )


def test_an_interrupted_build_is_not_loadable(built, tmp_path: Path) -> None:
    _manifest, output_dir = built
    (output_dir / "reference_arrays_manifest.json").unlink()
    with pytest.raises(FileNotFoundError, match="quarantine"):
        ReferenceArrayStore.open(
            output_dir, body_names=list(TRACKED), anchor_body=ANCHOR
        )


def _resolve(**kwargs):
    return MotionDataCfg(**kwargs).resolve(
        sim_dt=0.005, decimation=4, joint_names=[], canonical_joint_names=[]
    )


def test_reference_arrays_win_over_a_cache_dir_but_never_over_a_manifest(
    tmp_path: Path,
) -> None:
    """A task default that names a manifest must not silently win.

    Resolution would otherwise take the Zarr branch and ignore the arrays, which
    reads like a slow run rather than a misconfiguration.
    """
    resolved = _resolve(reference_arrays_dir=str(tmp_path), cache_dir=str(tmp_path))
    assert resolved is not None
    assert resolved.reference_arrays_dir == str(tmp_path.resolve())
    assert resolved.can_build is False

    with pytest.raises(ValueError, match="env.data.manifest=null"):
        _resolve(reference_arrays_dir=str(tmp_path), manifest="/some/manifest.json")

    # And with nothing declared at all, resolution still says "no data".
    assert _resolve() is None


def test_a_truncated_array_is_caught_before_training(built) -> None:
    _manifest, output_dir = built
    path = output_dir / "body_quat_w.memmap"
    with path.open("r+b") as handle:
        handle.truncate(path.stat().st_size - 4)
    with pytest.raises(RuntimeError, match="bytes but its sidecar declares"):
        ReferenceArrayStore.open(
            output_dir, body_names=list(TRACKED), anchor_body=ANCHOR
        )
