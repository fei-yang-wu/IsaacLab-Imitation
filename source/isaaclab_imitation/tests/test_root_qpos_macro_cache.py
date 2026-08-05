import json
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from torchrl.data.replay_buffers import TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import TensorStorage

from isaaclab_imitation.envs.expert_data_plane import (
    ExpertDataPlane,
    _require_matching_persisted_replay,
)


def _persist_manifest(*, motions: list[str] | None) -> dict:
    return {
        "format_version": 1,
        "key": {
            "source": {"persist_id": "full@abc123"},
            "datasets": None,
            "motions": motions,
            "trajectories": None,
            "keys": ["qpos", "qvel"],
        },
        "traj_info": {},
    }


def test_persisted_replay_guard_accepts_exact_identity(tmp_path) -> None:
    persist_dir = tmp_path / "full"
    persist_dir.mkdir()
    (persist_dir / "iltools_rb_manifest.json").write_text(
        json.dumps(_persist_manifest(motions=None)), encoding="utf-8"
    )

    _require_matching_persisted_replay(
        zarr_path=tmp_path / "source.zarr",
        persist_dir=str(persist_dir),
        persist_id="full@abc123",
        persist_rebuild=False,
        motions=None,
        keys=["qpos", "qvel"],
    )


def test_persisted_replay_guard_refuses_subset_overwrite(tmp_path) -> None:
    persist_dir = tmp_path / "full"
    persist_dir.mkdir()
    manifest_path = persist_dir / "iltools_rb_manifest.json"
    original = json.dumps(_persist_manifest(motions=None))
    manifest_path.write_text(original, encoding="utf-8")

    with pytest.raises(RuntimeError, match="different content or selection"):
        _require_matching_persisted_replay(
            zarr_path=tmp_path / "source.zarr",
            persist_dir=str(persist_dir),
            persist_id="full@abc123",
            persist_rebuild=False,
            motions=["one_motion"],
            keys=["qpos", "qvel"],
        )

    assert manifest_path.read_text(encoding="utf-8") == original


def test_persisted_replay_guard_refuses_nonempty_partial_build(tmp_path) -> None:
    persist_dir = tmp_path / "partial"
    persist_dir.mkdir()
    (persist_dir / "qpos.memmap").write_bytes(b"partial")

    with pytest.raises(RuntimeError, match="nonempty replay persist_dir"):
        _require_matching_persisted_replay(
            zarr_path=tmp_path / "source.zarr",
            persist_dir=str(persist_dir),
            persist_id="full@abc123",
            persist_rebuild=False,
            motions=None,
            keys=["qpos", "qvel"],
        )


def test_persisted_replay_guard_allows_explicit_rebuild(tmp_path) -> None:
    persist_dir = tmp_path / "owned"
    persist_dir.mkdir()
    (persist_dir / "sentinel").write_text("owned", encoding="utf-8")

    _require_matching_persisted_replay(
        zarr_path=tmp_path / "source.zarr",
        persist_dir=str(persist_dir),
        persist_id="full@abc123",
        persist_rebuild=True,
        motions=["replacement"],
        keys=["qpos"],
    )


def test_expert_macro_split_ranks_are_cached() -> None:
    plane = object.__new__(ExpertDataPlane)
    plane.trajectory_manager = SimpleNamespace(
        _length=torch.ones(20, dtype=torch.long),
        state_device=torch.device("cpu"),
    )
    plane._expert_macro_split_rank_cache = {}

    first = plane._expert_macro_split_trajectory_ranks(
        split="train", eval_fraction=0.25, split_seed=7
    )
    second = plane._expert_macro_split_trajectory_ranks(
        split="train", eval_fraction=0.25, split_seed=7
    )

    assert first.data_ptr() == second.data_ptr()
    assert first.numel() == 15


def test_compact_root_qpos_cache_clamps_windows_at_trajectory_bounds() -> None:
    terms = [
        "expert_motion_qpos",
        "expert_anchor_pos_b",
        "expert_anchor_ori_b",
    ]
    env = SimpleNamespace(
        cfg=SimpleNamespace(
            data=SimpleNamespace(
                macro_cache_device="cpu",
                macro_cache_chunk_size=3,
            ),
            expert_macro_state_terms=terms,
        ),
        device="cpu",
    )
    total = 9
    joint_pos = torch.arange(total * 2, dtype=torch.float32).reshape(total, 2)
    qpos = torch.zeros(total, 9)
    qpos[:, 7:] = joint_pos
    body_pos = torch.zeros(total, 2, 3)
    body_pos[:, 0, 0] = torch.arange(total)
    body_quat = torch.zeros(total, 2, 4)
    body_quat[..., 0] = 1.0  # Dataset WXYZ identity.
    source = TensorDict(
        {
            "qpos": qpos,
            "body_pos_w": body_pos,
            "body_quat_w": body_quat,
        },
        batch_size=[total],
    )
    trajectory_manager = SimpleNamespace(
        rb=SimpleNamespace(_storage=SimpleNamespace(_storage=source)),
        start=torch.tensor([0, 5]),
        end=torch.tensor([5, 9]),
        length=torch.tensor([5, 4]),
        state_device=torch.device("cpu"),
    )

    plane = object.__new__(ExpertDataPlane)
    plane._env = env
    plane.trajectory_manager = trajectory_manager
    plane._expert_anchor_body_name = "pelvis"
    plane.reference_body_names = ["pelvis", "other"]
    plane._root_qpos_macro_cache = None

    window = plane._sample_root_qpos_macro_window_for_trajectory_ranks(
        torch.tensor([0, 1]),
        torch.tensor([3, 0]),
        past_steps=1,
        future_steps=2,
    )

    assert window is not None
    assert window.batch_size == torch.Size([2, 4])
    assert tuple(window["joint_pos"].shape) == (2, 4, 2)
    assert window["_macro_anchor_pos_w"][0, :, 0].tolist() == [2.0, 3.0, 4.0, 4.0]
    assert window["_macro_anchor_pos_w"][1, :, 0].tolist() == [5.0, 5.0, 6.0, 7.0]
    assert window["_macro_anchor_quat_w"][0, 0].tolist() == [0.0, 0.0, 0.0, 1.0]


def test_runtime_reference_cache_keeps_only_selected_bodies() -> None:
    total = 7
    body_count = 3
    source = TensorDict(
        {
            "qpos": torch.arange(total * 9, dtype=torch.float32).reshape(total, 9),
            "qvel": torch.arange(total * 8, dtype=torch.float32).reshape(total, 8),
            "body_pos_w": torch.arange(
                total * body_count * 3, dtype=torch.float32
            ).reshape(total, body_count, 3),
            "body_quat_w": torch.arange(
                total * body_count * 4, dtype=torch.float32
            ).reshape(total, body_count, 4),
            "body_lin_vel_w": torch.arange(
                total * body_count * 3, dtype=torch.float32
            ).reshape(total, body_count, 3),
            "body_ang_vel_w": torch.arange(
                total * body_count * 3, dtype=torch.float32
            ).reshape(total, body_count, 3),
        },
        batch_size=[total],
    )
    rb = TensorDictReplayBuffer(
        storage=TensorStorage(source, device="cpu"), batch_size=1
    )
    cfg = SimpleNamespace(
        mpjpe_metric_body_names=["pelvis", "wrist"],
        command_ee_body_names=["wrist"],
        command_keypoint_body_names=["pelvis", "wrist"],
    )
    data_cfg = SimpleNamespace(
        runtime_cache_device="cpu",
        runtime_cache_body_names=["pelvis", "wrist"],
        runtime_cache_chunk_size=3,
    )
    plane = object.__new__(ExpertDataPlane)
    plane._expert_anchor_body_name = "pelvis"

    compact_rb, body_names = plane._materialize_runtime_reference_cache(
        rb=rb,
        traj_info=None,
        data_cfg=data_cfg,
        cfg=cfg,
        dataset_body_names=["pelvis", "middle", "wrist"],
    )

    compact = compact_rb._storage._storage
    assert body_names == ["pelvis", "wrist"]
    assert set(compact.keys()) == {
        "qpos",
        "qvel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    }
    assert compact["body_pos_w"].shape == (total, 2, 3)
    torch.testing.assert_close(compact["qpos"], source["qpos"])
    torch.testing.assert_close(compact["qvel"], source["qvel"])
    torch.testing.assert_close(compact["body_pos_w"][:, 1], source["body_pos_w"][:, 2])
