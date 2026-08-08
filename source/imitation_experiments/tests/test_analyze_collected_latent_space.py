from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from imitation_experiments.evaluation.analyze_collected_latent_space import (
    _balanced_indices,
    assign_semantic_phases,
    load_collected_latents,
    load_semantic_phase_annotations,
)


def test_load_collected_latents_reads_only_row_aligned_analysis_fields(
    tmp_path: Path,
) -> None:
    rows = 6
    width = 4
    sample = {
        "source_h1_latent_target": torch.arange(rows * width).reshape(rows, width),
        "z_target": torch.arange(rows * 3 * width).reshape(rows, 3 * width),
        "language_embedding": torch.ones(rows, 3),
        "motion_name": ["a", "a", "a", "b", "b", "b"],
        "env_id": torch.tensor([0, 0, 1, 2, 2, 3]),
        "planner_step": torch.tensor([0, 1, 0, 0, 1, 0]),
        "reference_local_step": torch.tensor([0, 10, 0, 0, 10, 0]),
        "reference_trajectory_length": torch.full((rows,), 21),
        "large_unused_field": torch.zeros(rows, 10, 93),
    }
    torch.save(sample, tmp_path / "sample_step_000000.pt")

    loaded = load_collected_latents(tmp_path)

    assert loaded.current.shape == (rows, width)
    assert loaded.future.shape == (rows, 3, width)
    assert loaded.language.shape == (rows, 3)
    assert loaded.phase.tolist() == [0.0, 0.5, 0.0, 0.0, 0.5, 0.0]


def test_balanced_indices_caps_each_motion_without_replacement() -> None:
    labels = np.asarray([0] * 8 + [1] * 3 + [2] * 5)

    selected = _balanced_indices(labels, max_points_per_motion=4, seed=7)
    selected_labels = labels[selected]

    assert len(np.unique(selected)) == len(selected)
    assert np.sum(selected_labels == 0) == 4
    assert np.sum(selected_labels == 1) == 3
    assert np.sum(selected_labels == 2) == 4


def test_semantic_phase_annotations_are_end_exclusive_and_row_aligned(
    tmp_path: Path,
) -> None:
    rows = 6
    width = 2
    sample = {
        "source_h1_latent_target": torch.zeros(rows, width),
        "z_target": torch.zeros(rows, 3 * width),
        "language_embedding": torch.ones(rows, 3),
        "motion_name": ["a"] * 3 + ["b"] * 3,
        "env_id": torch.arange(rows),
        "planner_step": torch.zeros(rows, dtype=torch.long),
        "reference_local_step": torch.tensor([0, 9, 10, 0, 5, 19]),
        "reference_trajectory_length": torch.full((rows,), 20),
    }
    torch.save(sample, tmp_path / "sample_step_000000.pt")
    annotations = {
        "schema": "semantic_phase_annotations_v1",
        "phase_definition": "test",
        "motions": [
            {
                "motion_name": name,
                "reference_frames": 20,
                "phases": [
                    {
                        "start_step": 0,
                        "end_step": 10,
                        "label": "first",
                        "activity": "locomotion",
                        "semantics": {
                            "locomoting": True,
                            "manipulating": False,
                            "object_loaded": False,
                            "torso_lowered": False,
                            "turning": False,
                        },
                    },
                    {
                        "start_step": 10,
                        "end_step": 20,
                        "label": "second",
                        "activity": "manipulation",
                        "semantics": {
                            "locomoting": False,
                            "manipulating": True,
                            "object_loaded": True,
                            "torso_lowered": False,
                            "turning": False,
                        },
                    },
                ],
            }
            for name in ("a", "b")
        ],
    }
    annotation_path = tmp_path / "phases.json"
    annotation_path.write_text(json.dumps(annotations))

    loaded = load_collected_latents(tmp_path)
    parsed = load_semantic_phase_annotations(annotation_path)
    assigned = assign_semantic_phases(loaded, parsed)

    assert assigned.labels.tolist() == [
        "first",
        "first",
        "second",
        "first",
        "first",
        "second",
    ]
    assert assigned.axes["manipulating"].tolist() == [0, 0, 1, 0, 0, 1]
