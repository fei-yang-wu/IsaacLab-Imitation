from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from imitation_experiments.evaluation.analyze_collected_latent_space import (
    SEMANTIC_AXES,
    CollectedLatents,
    SemanticPhaseAssignments,
)
from imitation_experiments.evaluation.analyze_cross_motion_latent_structure import (
    AggregatedPublications,
    aggregate_publications,
    analyze_cross_motion_retrieval,
    load_current_collected_latents,
    load_reference_kinematics,
)


def _collected_fixture() -> tuple[CollectedLatents, SemanticPhaseAssignments]:
    motion_names = np.asarray(["a", "a", "a", "a", "b", "b", "b", "b"], dtype=object)
    reference_steps = np.asarray([0, 0, 10, 10, 0, 0, 10, 10])
    current = np.asarray(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [10.0, 0.0],
            [12.0, 0.0],
            [0.2, 0.0],
            [2.2, 0.0],
            [10.2, 0.0],
            [12.2, 0.0],
        ],
        dtype=np.float32,
    )
    data = CollectedLatents(
        current=current,
        future=np.zeros((8, 3, 2), dtype=np.float32),
        language=np.zeros((8, 3), dtype=np.float32),
        motion_names=motion_names,
        env_ids=np.arange(8),
        planner_steps=np.tile([0, 0, 1, 1], 2),
        reference_steps=reference_steps,
        reference_lengths=np.full(8, 40),
        shard_paths=(),
    )
    activities = np.asarray(
        ["walk", "walk", "reach", "reach", "walk", "walk", "reach", "reach"],
        dtype=object,
    )
    assignments = SemanticPhaseAssignments(
        labels=activities.copy(),
        activities=activities,
        phase_indices=np.tile([0, 0, 1, 1], 2),
        axes={
            axis: np.asarray([1, 1, 0, 0, 1, 1, 0, 0], dtype=np.int8)
            if axis == "locomoting"
            else np.asarray([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int8)
            for axis in SEMANTIC_AXES
        },
    )
    return data, assignments


def test_aggregate_publications_averages_replicates_and_retains_spread() -> None:
    data, assignments = _collected_fixture()

    aggregated = aggregate_publications(data, assignments)

    assert aggregated.motion_names.tolist() == ["a", "a", "b", "b"]
    assert aggregated.reference_steps.tolist() == [0, 10, 0, 10]
    assert aggregated.replicate_counts.tolist() == [2, 2, 2, 2]
    np.testing.assert_allclose(
        aggregated.latent_mean,
        [[1.0, 0.0], [11.0, 0.0], [1.2, 0.0], [11.2, 0.0]],
    )
    np.testing.assert_allclose(aggregated.latent_rms_spread, [0.70710677] * 4)
    assert aggregated.activities.tolist() == ["walk", "reach", "walk", "reach"]


def test_cross_motion_retrieval_excludes_identity_and_recovers_activity() -> None:
    data, assignments = _collected_fixture()
    publications = aggregate_publications(data, assignments)
    latent = publications.latent_mean
    kinematic = latent.copy()

    metrics, rows = analyze_cross_motion_retrieval(
        publications,
        latent,
        kinematic,
        neighbor_counts=(1,),
        bootstrap_samples=100,
    )

    assert all(row["query_motion"] != row["neighbor_motion"] for row in rows)
    assert all(row["activity_match"] == 1 for row in rows)
    assert metrics["semantic"]["activity"]["1"]["agreement"] == 1.0
    assert metrics["kinematic"]["1"]["retrieved_to_random_distance_ratio_mean"] < 0.1


def test_load_current_collected_latents_accepts_raw_collection_shards(
    tmp_path: Path,
) -> None:
    sample = {
        "latent_skill_target": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "language_embedding": torch.ones(3, 2),
        "motion_name": ["a", "a", "b"],
        "env_id": torch.tensor([0, 0, 1]),
        "planner_step": torch.tensor([0, 1, 0]),
        "reference_local_step": torch.tensor([0, 10, 0]),
        "reference_trajectory_length": torch.tensor([40, 40, 50]),
    }
    torch.save(sample, tmp_path / "sample_step_000000.pt")

    loaded = load_current_collected_latents(tmp_path)

    assert loaded.current.shape == (3, 4)
    assert loaded.future.shape == (3, 0, 4)
    assert loaded.motion_names.tolist() == ["a", "a", "b"]


def _write_memmap(path: Path, values: np.ndarray) -> None:
    array = np.memmap(path, mode="w+", dtype=np.float32, shape=values.shape)
    array[:] = values
    array.flush()


def test_load_reference_kinematics_builds_yaw_normalized_windows(
    tmp_path: Path,
) -> None:
    bodies = [
        "pelvis",
        "torso_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    ]
    rows = 80
    qpos = np.zeros((rows, 36), dtype=np.float32)
    qpos[:, 3] = 1.0
    qvel = np.zeros((rows, 35), dtype=np.float32)
    body_pos = np.zeros((rows, len(bodies), 3), dtype=np.float32)
    body_lin = np.zeros_like(body_pos)
    body_ang = np.zeros_like(body_pos)
    body_pos[:, 0, 0] = np.arange(rows) * 0.01
    body_pos[:, 0, 2] = 0.8
    body_pos[:, 1, 0] = body_pos[:, 0, 0]
    body_pos[:, 1, 2] = 1.1
    body_lin[:, 0, 0] = 0.5
    body_lin[:, 2:, 0] = 0.2
    arrays = {
        "qpos": qpos,
        "qvel": qvel,
        "body_pos_w": body_pos,
        "body_lin_vel_w": body_lin,
        "body_ang_vel_w": body_ang,
    }
    for name, values in arrays.items():
        _write_memmap(tmp_path / f"{name}.memmap", values)
    sidecar = {
        "format_version": 1,
        "key": {
            "body_names": bodies,
            "arrays": {
                name: {"shape": list(values.shape), "dtype": "float32"}
                for name, values in arrays.items()
            },
        },
        "traj_info": {
            "start_index": [0, 40],
            "end_index": [40, 80],
            "ordered_traj_list": [
                ["test", "a", "trajectory_0"],
                ["test", "b", "trajectory_0"],
            ],
        },
    }
    (tmp_path / "reference_arrays_manifest.json").write_text(json.dumps(sidecar))
    publications = AggregatedPublications(
        latent_mean=np.zeros((2, 2), dtype=np.float32),
        latent_rms_spread=np.zeros(2, dtype=np.float32),
        motion_names=np.asarray(["a", "b"], dtype=object),
        reference_steps=np.zeros(2, dtype=np.int64),
        reference_lengths=np.full(2, 40, dtype=np.int64),
        replicate_counts=np.ones(2, dtype=np.int64),
        activities=None,
        phase_labels=None,
        axes={},
    )

    result = load_reference_kinematics(tmp_path, publications)

    assert result.descriptor.shape == (2, 920)
    assert result.summary.shape == (2, 8)
    np.testing.assert_allclose(result.summary[:, 0], 0.5)
    np.testing.assert_allclose(result.summary[:, 2], 0.8)
    assert np.isfinite(result.descriptor).all()
