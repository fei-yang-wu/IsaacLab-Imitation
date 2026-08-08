from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from imitation_experiments.planner.interface_planner_common import load_rollout_samples
from imitation_experiments.data.planner_sample_schema import (
    CompletedTrajectorySampleWriter,
    PlannerSampleWriter,
    add_sample_format_metadata,
    build_planner_sample,
    concatenate_planner_samples,
)


def _metadata() -> dict:
    return add_sample_format_metadata(
        {
            "interface": "latent_skill",
            "planner_observation_spec": {
                "history_frames": 10,
                "frame_dim": 93,
                "flat_dim": 930,
            },
        },
        collection_stage="demonstration",
        planner_interval_steps=10,
    )


def test_common_sample_keeps_unflattened_history_and_compatibility_aliases() -> None:
    causal = torch.randn(2, 10, 93)
    demonstration = torch.randn(2, 10, 93)
    causal_target = torch.randn(2, 256)
    demonstration_target = torch.randn(2, 256)
    sample = build_planner_sample(
        causal_state_history=causal,
        demonstration_state_history=demonstration,
        causal_target=causal_target,
        demonstration_target=demonstration_target,
        trajectory_rank=torch.tensor([3, 4]),
        episode_id=torch.tensor([7, 8]),
        control_step=torch.tensor([20, 30]),
        planner_step=torch.tensor([2, 3]),
        motion_names=["walk", "dance"],
        metadata=_metadata(),
    )
    assert sample["causal_state_history"].shape == (2, 10, 93)
    assert sample["planner_state"].shape == (2, 930)
    assert torch.equal(sample["causal_target"], causal_target)
    assert torch.equal(sample["demonstration_target"], demonstration_target)
    assert "target" not in sample
    assert sample["metadata"]["planner_rate_hz"] == 5.0


def test_sample_writer_chunks_rows_without_changing_values(tmp_path: Path) -> None:
    first = build_planner_sample(
        causal_state_history=torch.arange(24, dtype=torch.float32).reshape(2, 3, 4),
        demonstration_state_history=torch.ones(2, 3, 4),
        causal_target=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        demonstration_target=torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        trajectory_rank=torch.tensor([0, 1]),
        episode_id=torch.tensor([0, 0]),
        control_step=torch.tensor([0, 10]),
        planner_step=torch.tensor([0, 1]),
        motion_names=["a", "b"],
        metadata=add_sample_format_metadata(
            {
                "interface": "latent_skill",
                "planner_observation_spec": {
                    "history_frames": 3,
                    "frame_dim": 4,
                    "flat_dim": 12,
                },
            },
            collection_stage="planner_rollout",
            planner_interval_steps=10,
        ),
    )
    second = build_planner_sample(
        causal_state_history=torch.full((1, 3, 4), 9.0),
        demonstration_state_history=torch.full((1, 3, 4), 8.0),
        causal_target=torch.tensor([[9.0, 10.0]]),
        demonstration_target=torch.tensor([[11.0, 12.0]]),
        trajectory_rank=torch.tensor([2]),
        episode_id=torch.tensor([1]),
        control_step=torch.tensor([20]),
        planner_step=torch.tensor([2]),
        motion_names=["c"],
        metadata=first["metadata"],
    )

    joined = concatenate_planner_samples([first, second])
    assert joined["motion_name"] == ["a", "b", "c"]
    assert joined["causal_target"].tolist() == [
        [1.0, 2.0],
        [3.0, 4.0],
        [9.0, 10.0],
    ]

    writer = PlannerSampleWriter(tmp_path / "samples", rows_per_file=3)
    writer.add(first)
    assert writer.file_count == 0
    writer.add(second)
    writer.flush()
    assert writer.file_count == 1
    assert writer.row_count == 3
    saved = torch.load(
        tmp_path / "samples/sample_step_000000.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert saved["motion_name"] == ["a", "b", "c"]
    assert torch.equal(saved["causal_target"], joined["causal_target"])


def test_completed_trajectory_writer_keeps_variable_lengths_and_exact_budget(
    tmp_path: Path,
) -> None:
    metadata = add_sample_format_metadata(
        {
            "interface": "latent_skill",
            "planner_observation_spec": {
                "history_frames": 2,
                "frame_dim": 3,
                "flat_dim": 6,
            },
        },
        collection_stage="demonstration",
        planner_interval_steps=10,
    )

    def sample(
        env_ids: list[int],
        episode_ids: list[int],
        motions: list[str],
        value: float,
    ) -> dict:
        rows = len(env_ids)
        return build_planner_sample(
            causal_state_history=torch.full((rows, 2, 3), value),
            demonstration_state_history=torch.full((rows, 2, 3), value),
            causal_target=torch.full((rows, 2), value),
            demonstration_target=torch.full((rows, 2), value),
            trajectory_rank=torch.arange(rows),
            episode_id=torch.tensor(episode_ids),
            env_id=torch.tensor(env_ids),
            control_step=torch.zeros(rows, dtype=torch.long),
            planner_step=torch.zeros(rows, dtype=torch.long),
            motion_names=motions,
            metadata=metadata,
        )

    writer = CompletedTrajectorySampleWriter(
        tmp_path / "samples",
        motion_names=("a", "b"),
        trajectories_per_motion=1,
        rows_per_file=10,
    )
    writer.add(sample([0, 1], [0, 0], ["a", "b"], 1.0))
    writer.add(sample([0, 1], [0, 0], ["a", "b"], 2.0))
    writer.complete(env_ids=[0], episode_ids=[0], motion_names=["a"])
    writer.add(sample([0, 1], [1, 0], ["a", "b"], 3.0))
    writer.complete(
        env_ids=[0, 1],
        episode_ids=[1, 0],
        motion_names=["a", "b"],
        termination_reasons=[["reference_finished"], ["anchor_pos"]],
        tracking_success=[True, False],
    )
    # This unfinished segment must not leak through the global cutoff.
    writer.add(sample([2], [0], ["a"], 4.0))
    writer.flush()

    assert writer.complete_budget
    assert writer.counts() == {"a": 1, "b": 1}
    assert writer.completed_trajectory_count == 2
    assert writer.records()[-1]["termination_reasons"] == ["anchor_pos"]
    assert writer.records()[-1]["tracking_success"] is False
    assert writer.buffered_trajectory_count == 1
    assert writer.row_count == 5
    saved = torch.load(
        tmp_path / "samples/sample_step_000000.pt",
        map_location="cpu",
        weights_only=False,
    )
    keys = torch.stack([saved["env_id"], saved["episode_id"]], dim=1)
    assert torch.equal(torch.unique(keys, dim=0), torch.tensor([[0, 0], [1, 0]]))
    assert saved["motion_name"] == ["a", "a", "b", "b", "b"]


def test_common_sample_rejects_wrong_history_width() -> None:
    with pytest.raises(ValueError, match="flat width"):
        build_planner_sample(
            causal_state_history=torch.randn(1, 929),
            demonstration_state_history=torch.randn(1, 930),
            causal_target=torch.randn(1, 4),
            demonstration_target=torch.randn(1, 4),
            trajectory_rank=torch.tensor([0]),
            episode_id=0,
            control_step=0,
            planner_step=0,
            motion_names=["walk"],
            metadata=_metadata(),
        )


def test_common_sample_preserves_categorical_token_targets() -> None:
    metadata = _metadata()
    metadata["interface"] = "per_step_token_sequence"
    metadata["target_encoding"] = {
        "kind": "categorical_sequence",
        "horizon": 10,
        "codebook_size": 512,
    }
    tokens = torch.randint(0, 512, (2, 10))
    sample = build_planner_sample(
        causal_state_history=torch.randn(2, 10, 93),
        demonstration_state_history=torch.randn(2, 10, 93),
        causal_target=tokens,
        demonstration_target=tokens,
        trajectory_rank=torch.tensor([0, 1]),
        episode_id=torch.tensor([0, 0]),
        control_step=torch.tensor([0, 10]),
        planner_step=torch.tensor([0, 1]),
        motion_names=["walk", "dance"],
        metadata=metadata,
    )

    assert sample["causal_target"].dtype == torch.long
    assert sample["demonstration_target"].dtype == torch.long
    assert torch.equal(sample["causal_target"], tokens)


def test_language_conditioned_sample_roundtrip(tmp_path: Path) -> None:
    metadata = _metadata()
    metadata["target_spec"] = {
        "interface": "latent_skill",
        "term_names": ["z"],
        "term_widths": [4],
        "target_dim": 4,
    }
    metadata["language_conditioning"] = {
        "enabled": True,
        "embedding_dim": 5,
        "embedding_path": "/tmp/test_language.pt",
    }
    language = torch.randn(2, 5)
    sample = build_planner_sample(
        causal_state_history=torch.randn(2, 10, 93),
        demonstration_state_history=torch.randn(2, 10, 93),
        causal_target=torch.randn(2, 4),
        demonstration_target=torch.randn(2, 4),
        trajectory_rank=torch.tensor([0, 1]),
        episode_id=torch.tensor([0, 0]),
        control_step=torch.tensor([0, 10]),
        planner_step=torch.tensor([0, 1]),
        motion_names=["walk", "dance"],
        metadata=metadata,
        language_embedding=language,
    )
    samples_dir = tmp_path / "language_samples"
    samples_dir.mkdir()
    torch.save(sample, samples_dir / "sample_step_000000.pt")

    data, loaded_metadata = load_rollout_samples(samples_dir)

    assert torch.equal(data["language_embedding"], language)
    assert loaded_metadata["language_conditioning"]["embedding_dim"] == 5


def test_state_only_sample_rejects_zero_width_language_tensor() -> None:
    metadata = _metadata()
    metadata["target_spec"] = {
        "interface": "latent_skill",
        "term_names": ["z"],
        "term_widths": [4],
        "target_dim": 4,
    }
    metadata["language_conditioning"] = {
        "enabled": False,
        "embedding_dim": 0,
    }
    with pytest.raises(ValueError, match="omit it for a state-only"):
        build_planner_sample(
            causal_state_history=torch.randn(2, 10, 93),
            demonstration_state_history=torch.randn(2, 10, 93),
            causal_target=torch.randn(2, 4),
            demonstration_target=torch.randn(2, 4),
            trajectory_rank=torch.tensor([0, 0]),
            episode_id=torch.tensor([0, 0]),
            control_step=torch.tensor([0, 10]),
            planner_step=torch.tensor([0, 1]),
            motion_names=["walk", "walk"],
            metadata=metadata,
            language_embedding=torch.empty(2, 0),
        )


def test_language_samples_merge_across_explicit_goals(tmp_path: Path) -> None:
    base_metadata = _metadata()
    base_metadata["target_spec"] = {
        "interface": "latent_skill",
        "term_names": ["z"],
        "term_widths": [4],
        "target_dim": 4,
    }
    base_metadata["language_conditioning"] = {
        "enabled": True,
        "embedding_dim": 5,
        "embedding_path": "/tmp/shared_language.pt",
        "embedding_sha256": "a" * 64,
        "backend": "test",
        "model": "test",
    }
    samples_dir = tmp_path / "multi_goal_samples"
    samples_dir.mkdir()
    for index, goal_name in enumerate(("walk", "kick")):
        metadata = copy.deepcopy(base_metadata)
        metadata["language_conditioning"].update(
            {
                "goal_name": goal_name,
                "goal_phrase": f"do {goal_name}",
                "motion_count": index + 1,
            }
        )
        sample = build_planner_sample(
            causal_state_history=torch.randn(1, 10, 93),
            demonstration_state_history=torch.randn(1, 10, 93),
            causal_target=torch.randn(1, 4),
            demonstration_target=torch.randn(1, 4),
            trajectory_rank=torch.tensor([index]),
            episode_id=index,
            control_step=index * 10,
            planner_step=index,
            motion_names=[goal_name],
            metadata=metadata,
            language_embedding=torch.randn(1, 5),
        )
        torch.save(sample, samples_dir / f"sample_step_{index:06d}.pt")

    data, _ = load_rollout_samples(samples_dir)
    assert data["planner_state"].shape[0] == 2
    assert data["language_embedding"].shape == (2, 5)

    incompatible = torch.load(
        samples_dir / "sample_step_000001.pt", map_location="cpu", weights_only=False
    )
    incompatible["metadata"]["language_conditioning"]["embedding_sha256"] = "b" * 64
    torch.save(incompatible, samples_dir / "sample_step_000001.pt")
    with pytest.raises(ValueError, match="metadata does not match"):
        load_rollout_samples(samples_dir)


def test_loader_preserves_env_id_for_trajectory_split(tmp_path: Path) -> None:
    metadata = _metadata()
    metadata["target_spec"] = {
        "interface": "latent_skill",
        "term_names": ["z"],
        "term_widths": [2],
        "target_dim": 2,
    }
    sample = build_planner_sample(
        causal_state_history=torch.zeros(2, 10, 93),
        demonstration_state_history=torch.zeros(2, 10, 93),
        causal_target=torch.zeros(2, 2),
        demonstration_target=torch.zeros(2, 2),
        trajectory_rank=torch.zeros(2, dtype=torch.long),
        episode_id=torch.tensor([0, 0]),
        env_id=torch.tensor([4, 9]),
        control_step=torch.tensor([0, 0]),
        planner_step=torch.tensor([0, 0]),
        motion_names=["a", "a"],
        metadata=metadata,
    )
    torch.save(sample, tmp_path / "sample_step_000000.pt")
    loaded, _ = load_rollout_samples(tmp_path)
    assert torch.equal(loaded["env_id"], torch.tensor([4, 9]))
