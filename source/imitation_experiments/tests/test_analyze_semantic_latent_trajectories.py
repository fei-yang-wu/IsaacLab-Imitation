from __future__ import annotations

from pathlib import Path

import numpy as np

from imitation_experiments.evaluation.analyze_semantic_latent_trajectories import (
    PhaseUnits,
    analyze_region_separation,
    analyze_return_excursions,
    analyze_semantic_neighbors,
    _short_motion_name,
    assign_semantic_region,
    load_semantic_region_taxonomy,
)


def test_short_motion_name_removes_take_suffix() -> None:
    assert _short_motion_name("Neutral_stoop_down_001_A057") == "Neutral stoop down"
    assert (
        _short_motion_name("inside_door_handle_right_side_R_001_A514")
        == "inside door handle right side"
    )


def _taxonomy(tmp_path: Path) -> dict[str, object]:
    path = tmp_path / "taxonomy.json"
    path.write_text(
        """{
          "schema": "semantic_region_taxonomy_v1",
          "regions": [
            {"name":"loaded","display_name":"Loaded","description":"loaded locomotion","all_true":["locomoting"],"any_true":["manipulating"],"color":"#112233"},
            {"name":"walk","display_name":"Walk","description":"walking","all_true":["locomoting"],"color":"#223344"},
            {"name":"other","display_name":"Other","description":"fallback","default":true,"color":"#334455"}
          ]
        }"""
    )
    return load_semantic_region_taxonomy(path)


def test_region_rules_are_ordered_and_movement_first(tmp_path: Path) -> None:
    taxonomy = _taxonomy(tmp_path)
    assert (
        assign_semantic_region(
            "transition", {"locomoting": True, "manipulating": True}, taxonomy
        )
        == "loaded"
    )
    assert (
        assign_semantic_region(
            "transition", {"locomoting": True, "manipulating": False}, taxonomy
        )
        == "walk"
    )
    assert assign_semantic_region("stationary", {}, taxonomy) == "other"


def _phase_units() -> PhaseUnits:
    # Motion a deliberately leaves and returns to region walk: [0,0] -> [5,0] -> [1,0].
    return PhaseUnits(
        motion_names=np.asarray(["a", "a", "a", "b", "b", "c", "c"], dtype=object),
        phase_indices=np.asarray([0, 1, 2, 0, 1, 0, 1]),
        phase_labels=np.asarray(
            ["a0", "a1", "a2", "b0", "b1", "c0", "c1"], dtype=object
        ),
        activities=np.asarray(
            ["walk", "reach", "walk", "walk", "reach", "walk", "reach"], dtype=object
        ),
        regions=np.asarray(
            ["walk", "reach", "walk", "walk", "reach", "walk", "reach"], dtype=object
        ),
        start_steps=np.arange(7) * 10,
        end_steps=np.arange(1, 8) * 10,
        publication_counts=np.ones(7, dtype=np.int64),
        centroid_features=np.asarray(
            [
                [0.0, 0.0],
                [5.0, 0.0],
                [1.0, 0.0],
                [0.1, 0.0],
                [5.1, 0.0],
                [0.2, 0.0],
                [5.2, 0.0],
            ]
        ),
        pca2=np.zeros((7, 2)),
        tsne2=np.zeros((7, 2)),
    )


def test_cross_motion_neighbor_and_separation_recover_shared_regions() -> None:
    phases = _phase_units()
    neighbors = analyze_semantic_neighbors(
        phases.centroid_features,
        phases.motion_names,
        phases.regions,
        neighbor_counts=(1,),
        seed=0,
        bootstrap_samples=100,
    )
    separation, ratios = analyze_region_separation(
        phases, seed=0, bootstrap_samples=100
    )
    assert neighbors["by_k"]["1"]["agreement"] == 1.0
    assert neighbors["by_k"]["1"]["agreement_improvement"] > 0.0
    assert separation["fraction_below_one"] == 1.0
    assert np.all(ratios < 1.0)


def test_return_excursion_detects_a_to_other_to_a() -> None:
    metrics, rows = analyze_return_excursions(_phase_units())
    assert metrics["sequence_count"] == 1
    assert metrics["fraction_below_one"] == 1.0
    assert rows[0]["motion_name"] == "a"
    assert rows[0]["return_region"] == "walk"
    assert rows[0]["return_ratio"] < 1.0
