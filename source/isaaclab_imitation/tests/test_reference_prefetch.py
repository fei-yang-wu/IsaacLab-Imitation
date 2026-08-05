from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from iltools.datasets.manager import ParallelTrajectoryManager, ResetSchedule
from iltools.datasets.reset_sampling import SonicAdaptiveResetSampler
from isaaclab_imitation.envs.expert_data_plane import (
    ExpertDataPlane,
    _convert_reference_quats_to_xyzw,
)
from isaaclab_imitation.envs.reference_arrays import (
    RUNTIME_FIELDS,
    pack_cpu_fields_parallel,
)
from tensordict import TensorDict
from torchrl.data.replay_buffers import TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import TensorStorage
from isaaclab_imitation.tasks.manager_based.imitation.mdp.commands.reference import (
    ReferenceCommandTerm,
)


def _manager(device: torch.device) -> tuple[ParallelTrajectoryManager, TensorDict]:
    rows = 8
    qpos = torch.zeros((rows, 9), dtype=torch.float32)
    qpos[:, 0] = torch.arange(rows, dtype=torch.float32)
    qpos[:, 3] = 1.0  # identity in dataset WXYZ order
    qvel = torch.arange(rows * 8, dtype=torch.float32).reshape(rows, 8)
    body_pos = torch.arange(rows * 6, dtype=torch.float32).reshape(rows, 2, 3)
    body_quat = torch.zeros((rows, 2, 4), dtype=torch.float32)
    body_quat[..., 0] = 1.0
    body_vel = torch.arange(rows * 6, dtype=torch.float32).reshape(rows, 2, 3)
    source = TensorDict(
        {
            "qpos": qpos,
            "qvel": qvel,
            "body_pos_w": body_pos,
            "body_quat_w": body_quat,
            "body_lin_vel_w": body_vel,
            "body_ang_vel_w": -body_vel,
        },
        batch_size=[rows],
        device="cpu",
    )
    rb = TensorDictReplayBuffer(
        storage=TensorStorage(source, device="cpu"), batch_size=1
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(5)
    manager = ParallelTrajectoryManager(
        rb=rb,
        traj_info={
            "start_index": [0, 4],
            "end_index": [4, 8],
            "ordered_traj_list": [("ds", "m", "a"), ("ds", "m", "b")],
        },
        num_envs=2,
        reset_schedule=ResetSchedule.SEQUENTIAL,
        device=device,
        reset_generator=generator,
        target_joint_names=["j0", "j1"],
        reference_joint_names=["j0", "j1"],
    )
    return manager, source


def _plane(device: torch.device, mode: str, *, packed: bool = False) -> ExpertDataPlane:
    manager, _source = _manager(device)
    packed_source = None
    if packed:
        source = manager.rb._storage._storage
        packed_source, fields = pack_cpu_fields_parallel(
            {key: source[key] for key in RUNTIME_FIELDS},
            names=RUNTIME_FIELDS,
            workers=1,
            chunk_rows=4,
        )
        packed_td = TensorDict(fields, batch_size=source.batch_size, device="cpu")
        manager.rb = TensorDictReplayBuffer(
            storage=TensorStorage(packed_td, device="cpu"), batch_size=1
        )
    plane = ExpertDataPlane.__new__(ExpertDataPlane)
    plane._env = SimpleNamespace(
        num_envs=2,
        device=device,
        cfg=SimpleNamespace(data=SimpleNamespace(reference_prefetch_reset_pool_size=2)),
    )
    plane.trajectory_manager = manager
    plane.current_expert_frame = _convert_reference_quats_to_xyzw(
        manager.sample(advance=False)
    )
    plane._current_reference_local_step = manager.env_step.to(device).clone()
    plane._invalidate_mdp_cache = lambda: None
    plane._reference_prefetch_mode = mode
    plane._reference_prefetch_source = None
    plane._reference_prefetch_packed_source = packed_source
    plane._reference_prefetch_executor = None
    plane._reference_prefetch_stream = None
    plane._reference_prefetch_slots = []
    plane._reference_prefetch_slot_index = 0
    plane._reference_prefetch_pending = None
    plane._reference_reset_prefetch_slot = None
    plane._reference_reset_prefetch_pending = False
    plane._reference_reset_prefetch_metrics = {}
    plane._reference_prefetch_metrics = {}
    if mode != "off":
        plane._initialize_reference_prefetch()
    return plane


def test_single_fetch_path_does_not_reread_the_reward_frame() -> None:
    plane = _plane(torch.device("cpu"), "off")
    manager = plane.trajectory_manager
    original_sample = manager.sample
    manager.sample = Mock(wraps=original_sample)

    plane.begin_next_reference()
    assert manager.sample.call_count == 0
    torch.testing.assert_close(
        plane.current_expert_frame["qpos"][:, 0], torch.tensor([0.0, 4.0])
    )

    # Simulate one asynchronous reset inside `_step_core`: its row is required
    # immediately by reset events, then the one full post-step fetch supplies
    # both the ordinary next row and that newly selected reset row.
    reset_ids = torch.tensor([1])
    manager.reset_envs(reset_ids, ranks=torch.tensor([0]), steps=torch.tensor([2]))
    plane._refresh_current_expert_frame(reset_ids, advance=False)
    plane.finish_next_reference(reset_ids)

    assert manager.sample.call_count == 2  # one reset subset + one full fetch
    torch.testing.assert_close(
        plane.current_expert_frame["qpos"][:, 0], torch.tensor([1.0, 2.0])
    )
    assert plane._current_reference_local_step.tolist() == [1, 2]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("packed", [False, True])
def test_async_next_prefetch_matches_current_cursors_and_patches_resets(
    packed: bool,
) -> None:
    device = torch.device("cuda:0")
    plane = _plane(device, "next", packed=packed)
    manager = plane.trajectory_manager
    original_sample = manager.sample
    manager.sample = Mock(wraps=original_sample)
    try:
        plane.begin_next_reference()
        assert manager.sample.call_count == 0

        reset_ids = torch.tensor([1], device=device)
        manager.reset_envs(
            reset_ids,
            ranks=torch.tensor([0], device=device),
            steps=torch.tensor([2], device=device),
        )
        plane._refresh_current_expert_frame(reset_ids, advance=False)
        plane.finish_next_reference(reset_ids)

        # The full batch came through the persistent staging slots; only the
        # reset row needed a synchronous manager sample.
        assert manager.sample.call_count == 1
        torch.testing.assert_close(
            plane.current_expert_frame["qpos"][:, 0],
            torch.tensor([1.0, 2.0], device=device),
        )
        assert plane._current_reference_local_step.tolist() == [1, 2]
        # Both root and body quaternions cross the same WXYZ -> XYZW boundary.
        expected_identity = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
        torch.testing.assert_close(
            plane.current_expert_frame["root_quat"],
            expected_identity.expand(2, 4),
        )
        torch.testing.assert_close(
            plane.current_expert_frame["body_quat_w"],
            expected_identity.expand(2, 2, 4),
        )

        plane.begin_next_reference()
        plane.finish_next_reference(torch.empty(0, device=device, dtype=torch.long))
        torch.testing.assert_close(
            plane.current_expert_frame["qpos"][:, 0],
            torch.tensor([2.0, 3.0], device=device),
        )
        assert plane._current_reference_local_step.tolist() == [2, 3]
        assert plane.reference_prefetch_metrics()["ReferencePrefetch/gather_ms"] >= 0
    finally:
        plane.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_predictive_reset_pool_installs_rows_without_storage_sampling() -> None:
    device = torch.device("cuda:0")
    plane = _plane(device, "next_and_reset")
    manager = plane.trajectory_manager
    original_sample = manager.sample
    manager.sample = Mock(wraps=original_sample)
    try:
        plane.begin_next_reference()
        candidate_ranks = torch.tensor([0, 1], device=device)
        candidate_steps = torch.tensor([3, 2], device=device)
        plane.begin_predicted_reset_reference(candidate_ranks, candidate_steps)

        reset_ids = torch.tensor([0, 1], device=device)
        manager.reset_envs(reset_ids, ranks=candidate_ranks, steps=candidate_steps)
        plane.consume_predicted_reset_reference(reset_ids, prefetched_count=2)
        assert manager.sample.call_count == 0
        torch.testing.assert_close(
            plane.current_expert_frame["qpos"][:, 0],
            torch.tensor([3.0, 6.0], device=device),
        )
        assert plane._current_reference_local_step.tolist() == [3, 2]

        # The ordinary sequential batch was planned before the resets. Passing
        # both ids as overrides must preserve the predicted reset frames.
        plane.finish_next_reference(reset_ids)
        torch.testing.assert_close(
            plane.current_expert_frame["qpos"][:, 0],
            torch.tensor([3.0, 6.0], device=device),
        )
        metrics = plane.reference_prefetch_metrics()
        assert metrics["ReferencePrefetch/reset_pool_hits"] == 2.0
        assert metrics["ReferencePrefetch/reset_pool_overflow"] == 0.0

        # Force the overflow branch: only the first row is declared prefetched;
        # the second must come from one synchronous subset sample.
        plane.begin_next_reference()
        plane.begin_predicted_reset_reference(candidate_ranks, candidate_steps)
        manager.reset_envs(reset_ids, ranks=candidate_ranks, steps=candidate_steps)
        plane.consume_predicted_reset_reference(reset_ids, prefetched_count=1)
        assert manager.sample.call_count == 1
        plane.finish_next_reference(reset_ids)
        metrics = plane.reference_prefetch_metrics()
        assert metrics["ReferencePrefetch/reset_pool_hits"] == 1.0
        assert metrics["ReferencePrefetch/reset_pool_overflow"] == 1.0
    finally:
        plane.close()


def test_reference_term_uses_the_prephysics_sonic_snapshot() -> None:
    manager, _source = _manager(torch.device("cpu"))
    sampler = SonicAdaptiveResetSampler(
        manager.length,
        pre_failure_sample_window=0,
        generator=manager.reset_generator,
    )

    class _Plane:
        reference_prefetch_mode = "next_and_reset"

        def __init__(self) -> None:
            self.staged: tuple[torch.Tensor, torch.Tensor] | None = None

        def begin_predicted_reset_reference(
            self, ranks: torch.Tensor, steps: torch.Tensor
        ) -> None:
            self.staged = (ranks.clone(), steps.clone())

    plane = _Plane()
    env = SimpleNamespace(
        trajectory_manager=manager,
        expert_data_plane=plane,
        cfg=SimpleNamespace(data=SimpleNamespace(reference_prefetch_reset_pool_size=2)),
    )
    term = ReferenceCommandTerm.__new__(ReferenceCommandTerm)
    term.cfg = SimpleNamespace(selection=SimpleNamespace(full_trajectory=True))
    term._adaptive_failure_reset_sampler = sampler
    term._predicted_reset_ranks = None
    term._predicted_reset_steps = None
    term._predicted_reset_probabilities = None
    term._imitation_env = lambda: env

    term.prepare_predicted_resets()
    assert plane.staged is not None
    staged_ranks, staged_steps = plane.staged

    # A current failure changes the live distribution, but the reset already
    # staged for this step must continue using the pre-physics snapshot.
    sampler.num_visits.fill_(100.0)
    sampler.num_failures.fill_(1.0)
    sampler.num_failures[-1] = 100.0
    prefetched = term.resample_reference(torch.tensor([0, 1]))

    assert prefetched == 2
    torch.testing.assert_close(manager.env_traj_rank, staged_ranks)
    torch.testing.assert_close(manager.env_step, staged_steps)
    assert term._predicted_reset_ranks is None
