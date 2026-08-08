from __future__ import annotations

from imitation_experiments.evaluation.build_latent_neighbor_gallery import (
    phase_for_publication,
    select_median_queries,
)


def test_phase_for_publication_uses_end_exclusive_intervals() -> None:
    phases = {
        "motion": [
            {"start_step": 0, "end_step": 10, "label": "first"},
            {"start_step": 10, "end_step": 20, "label": "second"},
        ]
    }
    assert phase_for_publication(phases, "motion", 9)["label"] == "first"
    assert phase_for_publication(phases, "motion", 10)["label"] == "second"


def test_select_median_queries_does_not_choose_best_case() -> None:
    publications = [
        {
            "publication_index": str(index),
            "motion_name": f"motion{index}",
            "reference_step": "0",
            "semantic_activity": "walk",
            "locomoting": "1",
        }
        for index in range(3)
    ]
    neighbors = {
        0: [
            {
                "neighbor_motion": "n0a",
                "locomoting_neighbor": "0",
                "activity_match": "0",
            },
            {
                "neighbor_motion": "n0b",
                "locomoting_neighbor": "0",
                "activity_match": "0",
            },
        ],
        1: [
            {
                "neighbor_motion": "n1a",
                "locomoting_neighbor": "1",
                "activity_match": "1",
            },
            {
                "neighbor_motion": "n1b",
                "locomoting_neighbor": "0",
                "activity_match": "0",
            },
        ],
        2: [
            {
                "neighbor_motion": "n2a",
                "locomoting_neighbor": "1",
                "activity_match": "1",
            },
            {
                "neighbor_motion": "n2b",
                "locomoting_neighbor": "1",
                "activity_match": "1",
            },
        ],
    }

    selected = select_median_queries(
        publications,
        neighbors,
        targets=["locomoting"],
        neighbor_count=2,
        excluded_motion_names=set(),
    )

    assert selected[0]["publication"]["motion_name"] == "motion1"
    assert selected[0]["match_rate"] == 0.5
