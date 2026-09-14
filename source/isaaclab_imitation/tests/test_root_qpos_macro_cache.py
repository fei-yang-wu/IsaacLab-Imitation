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
    plane._compact_macro_cache = None

    window = plane._sample_compact_macro_window_for_trajectory_ranks(
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


def _compact_macro_plane(*, frame_stride: int = 1, total: int = 20):
    """A one-trajectory compact-cache plane, long enough for a strided window."""
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
            expert_macro_frame_stride=frame_stride,
        ),
        device="cpu",
    )
    joint_pos = torch.arange(total * 2, dtype=torch.float32).reshape(total, 2)
    qpos = torch.zeros(total, 9)
    qpos[:, 7:] = joint_pos
    body_pos = torch.zeros(total, 2, 3)
    body_pos[:, 0, 0] = torch.arange(total)
    body_quat = torch.zeros(total, 2, 4)
    body_quat[..., 0] = 1.0  # Dataset WXYZ identity.
    source = TensorDict(
        {"qpos": qpos, "body_pos_w": body_pos, "body_quat_w": body_quat},
        batch_size=[total],
    )
    trajectory_manager = SimpleNamespace(
        rb=SimpleNamespace(_storage=SimpleNamespace(_storage=source)),
        start=torch.tensor([0]),
        end=torch.tensor([total]),
        length=torch.tensor([total]),
        state_device=torch.device("cpu"),
    )
    plane = object.__new__(ExpertDataPlane)
    plane._env = env
    plane.trajectory_manager = trajectory_manager
    plane._expert_anchor_body_name = "pelvis"
    plane.reference_body_names = ["pelvis", "other"]
    plane._compact_macro_cache = None
    return plane


def test_macro_frame_stride_spaces_the_window_and_still_clamps() -> None:
    """SONIC's cadence: 10 slots 5 reference frames apart, 0.9 s at 50 Hz.

    The window WIDTH is identical to the consecutive one, which is exactly why
    a mismatch cannot be caught downstream by a shape.
    """
    plane = _compact_macro_plane(frame_stride=5, total=20)

    window = plane._sample_expert_macro_window_for_trajectory_ranks(
        torch.tensor([0, 0]),
        torch.tensor([0, 10]),
        past_steps=0,
        future_steps=3,
    )

    assert window.batch_size == torch.Size([2, 4])
    # Row 0 starts at frame 0: slots land on 0, 5, 10, 15.
    assert window["_macro_anchor_pos_w"][0, :, 0].tolist() == [0.0, 5.0, 10.0, 15.0]
    # Row 1 starts at 10 and runs past the end: 10, 15, then clamped to 19.
    assert window["_macro_anchor_pos_w"][1, :, 0].tolist() == [10.0, 15.0, 19.0, 19.0]


def test_macro_frame_stride_one_is_the_historical_window() -> None:
    plane = _compact_macro_plane(frame_stride=1, total=20)

    window = plane._sample_expert_macro_window_for_trajectory_ranks(
        torch.tensor([0]),
        torch.tensor([2]),
        past_steps=1,
        future_steps=2,
    )

    assert window["_macro_anchor_pos_w"][0, :, 0].tolist() == [1.0, 2.0, 3.0, 4.0]


def test_macro_frame_stride_rejects_zero() -> None:
    plane = _compact_macro_plane(frame_stride=0)

    with pytest.raises(ValueError, match="expert_macro_frame_stride"):
        plane._expert_macro_frame_stride()


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


def _compact_ee_plane(*, total: int = 12, terms: list[str] | None = None):
    """A compact-cache plane whose macro terms add the command end effectors."""
    terms = terms or [
        "expert_motion_qpos",
        "expert_anchor_pos_b",
        "expert_anchor_ori_b",
        "expert_ee_pos_b",
    ]
    env = SimpleNamespace(
        cfg=SimpleNamespace(
            data=SimpleNamespace(
                macro_cache_device="cpu",
                macro_cache_chunk_size=5,
            ),
            expert_macro_state_terms=terms,
            expert_macro_frame_stride=1,
        ),
        device="cpu",
        _command_ee_body_names=("left_foot", "right_hand"),
    )
    joint_pos = torch.arange(total * 2, dtype=torch.float32).reshape(total, 2)
    qpos = torch.zeros(total, 9)
    qpos[:, 7:] = joint_pos
    # Bodies: pelvis, right_hand, left_foot, other. Each body's x is
    # frame + 10 * body index so a wrong column or order is visible.
    body_pos = torch.zeros(total, 4, 3)
    for body in range(4):
        body_pos[:, body, 0] = torch.arange(total) + 10.0 * body
    body_quat = torch.zeros(total, 4, 4)
    body_quat[..., 0] = 1.0  # Dataset WXYZ identity.
    body_quat[:, 2, 0] = 0.0
    body_quat[:, 2, 1] = 1.0  # left_foot: WXYZ (0, 1, 0, 0) -> XYZW (1, 0, 0, 0)
    source = TensorDict(
        {"qpos": qpos, "body_pos_w": body_pos, "body_quat_w": body_quat},
        batch_size=[total],
    )
    trajectory_manager = SimpleNamespace(
        rb=SimpleNamespace(_storage=SimpleNamespace(_storage=source)),
        start=torch.tensor([0]),
        end=torch.tensor([total]),
        length=torch.tensor([total]),
        state_device=torch.device("cpu"),
    )
    plane = object.__new__(ExpertDataPlane)
    plane._env = env
    plane.trajectory_manager = trajectory_manager
    plane._expert_anchor_body_name = "pelvis"
    plane.reference_body_names = ["pelvis", "right_hand", "left_foot", "other"]
    plane._compact_macro_cache = None
    return plane


def test_compact_cache_carries_command_end_effectors_in_command_order() -> None:
    plane = _compact_ee_plane()

    window = plane._sample_expert_macro_window_for_trajectory_ranks(
        torch.tensor([0]),
        torch.tensor([3]),
        past_steps=1,
        future_steps=1,
    )

    assert window.batch_size == torch.Size([1, 3])
    assert tuple(window["_macro_ee_pos_w"].shape) == (1, 3, 2, 3)
    # Command order (left_foot, right_hand) = dataset columns (2, 1).
    assert window["_macro_ee_pos_w"][0, :, 0, 0].tolist() == [22.0, 23.0, 24.0]
    assert window["_macro_ee_pos_w"][0, :, 1, 0].tolist() == [12.0, 13.0, 14.0]
    # Quaternions are swizzled to XYZW like the anchor.
    assert window["_macro_ee_quat_w"][0, 0, 0].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert window["_macro_ee_quat_w"][0, 0, 1].tolist() == [0.0, 0.0, 0.0, 1.0]
    assert window["_macro_anchor_pos_w"][0, :, 0].tolist() == [2.0, 3.0, 4.0]


def test_compact_cache_without_ee_term_carries_no_bodies() -> None:
    plane = _compact_ee_plane(
        terms=["expert_motion_qpos", "expert_anchor_pos_b", "expert_anchor_ori_b"]
    )

    window = plane._sample_expert_macro_window_for_trajectory_ranks(
        torch.tensor([0]), torch.tensor([3]), past_steps=1, future_steps=1
    )

    assert "_macro_ee_pos_w" not in window.keys()


def test_compact_cache_refuses_unsupported_macro_terms() -> None:
    plane = _compact_ee_plane(
        terms=[
            "expert_motion_qpos",
            "expert_anchor_pos_b",
            "expert_anchor_ori_b",
            "expert_ee_ori_b",
        ]
    )

    with pytest.raises(ValueError, match="macro_cache_device supports"):
        plane._compact_macro_cache_device()


def test_compact_ee_window_terms_match_the_replay_window_terms() -> None:
    """The compact ee window must give byte-identical expert_ee_pos_b."""
    plane = _compact_ee_plane()
    env_ids = torch.tensor([0])
    compact = plane._sample_expert_macro_window_for_trajectory_ranks(
        torch.tensor([0]), torch.tensor([4]), past_steps=2, future_steps=2
    )
    # The replay-window path reads the full body block under body_pos_w.
    source = plane.trajectory_manager.rb._storage._storage
    replay = TensorDict(
        {
            "joint_pos": source["qpos"][2:7, 7:].unsqueeze(0),
            "body_pos_w": source["body_pos_w"][2:7].unsqueeze(0),
            "body_quat_w": source["body_quat_w"][2:7][..., [1, 2, 3, 0]].unsqueeze(0),
        },
        batch_size=[1, 5],
    )
    plane._get_joint_ids_tensor_fast = lambda ids: torch.arange(2)
    plane._get_reference_body_ids_fast = lambda names: torch.tensor(
        [plane.reference_body_names.index(n) for n in names]
    )
    kwargs = dict(
        context="expert_heading",
        past_steps=2,
        anchor_body_name="pelvis",
        reference_body_names=("left_foot", "right_hand"),
    )
    compact_terms = plane._build_expert_window_terms(compact, env_ids, **kwargs)
    replay_terms = plane._build_expert_window_terms(replay, env_ids, **kwargs)

    for name in ("expert_motion_qpos", "expert_anchor_pos_b", "expert_ee_pos_b"):
        assert torch.equal(compact_terms[name], replay_terms[name]), name
    assert tuple(compact_terms["expert_ee_pos_b"].shape) == (1, 5 * 2 * 3)


def test_compact_ee_window_refuses_other_body_sets() -> None:
    plane = _compact_ee_plane()
    compact = plane._sample_expert_macro_window_for_trajectory_ranks(
        torch.tensor([0]), torch.tensor([4]), past_steps=1, future_steps=1
    )
    plane._get_joint_ids_tensor_fast = lambda ids: torch.arange(2)

    with pytest.raises(ValueError, match="command end-effector bodies"):
        plane._build_expert_window_terms(
            compact,
            torch.tensor([0]),
            context="expert_heading",
            past_steps=1,
            anchor_body_name="pelvis",
            reference_body_names=("other",),
        )


class _FakeArrayStore:
    """The slice of ReferenceArrayStore the compact cache reads (baked anchor)."""

    def __init__(self, arrays: dict[str, torch.Tensor], body_names: list[str]):
        self._arrays = arrays
        self.body_names = body_names
        self.directory = "fake-store"
        self.available_arrays = frozenset(arrays)
        self.num_rows = int(next(iter(arrays.values())).shape[0])

    def anchor_source(self, anchor_body: str) -> int | None:
        return None

    def array(self, name: str) -> torch.Tensor:
        return self._arrays[name]


def test_compact_cache_from_arrays_selects_ee_columns_and_swizzles() -> None:
    plane = _compact_ee_plane(total=6)
    total = 6
    body_names = ["pelvis", "right_hand", "left_foot", "other"]
    body_pos = torch.zeros(total, 4, 3)
    for body in range(4):
        body_pos[:, body, 0] = torch.arange(total) + 10.0 * body
    body_quat = torch.zeros(total, 4, 4)
    body_quat[..., 0] = 1.0
    body_quat[:, 2, 0] = 0.0
    body_quat[:, 2, 1] = 1.0
    store = _FakeArrayStore(
        {
            "qpos": torch.arange(total * 9, dtype=torch.float32).reshape(total, 9),
            "anchor_pos_w": body_pos[:, 0],
            "anchor_quat_w": body_quat[:, 0][..., [1, 2, 3, 0]],
            "body_pos_w": body_pos,
            "body_quat_w": body_quat,
        },
        body_names,
    )

    cache = plane._compact_macro_cache_from_arrays(store, torch.device("cpu"))

    assert tuple(cache["ee_pos_w"].shape) == (total, 2, 3)
    assert cache["ee_pos_w"][:, 0, 0].tolist() == [20.0, 21.0, 22.0, 23.0, 24.0, 25.0]
    assert cache["ee_pos_w"][:, 1, 0].tolist() == [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    assert cache["ee_quat_w"][0, 0].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert cache["ee_quat_w"][0, 1].tolist() == [0.0, 0.0, 0.0, 1.0]
    assert torch.equal(cache["joint_pos"], store.array("qpos")[:, 7:])


def test_compact_cache_from_arrays_refuses_missing_ee_body() -> None:
    plane = _compact_ee_plane(total=4)
    total = 4
    body_pos = torch.zeros(total, 2, 3)
    body_quat = torch.zeros(total, 2, 4)
    body_quat[..., 0] = 1.0
    store = _FakeArrayStore(
        {
            "qpos": torch.zeros(total, 9),
            "anchor_pos_w": body_pos[:, 0],
            "anchor_quat_w": body_quat[:, 0],
            "body_pos_w": body_pos,
            "body_quat_w": body_quat,
        },
        ["pelvis", "other"],
    )

    with pytest.raises(KeyError, match="does not retain end-effector bodies"):
        plane._compact_macro_cache_from_arrays(store, torch.device("cpu"))
