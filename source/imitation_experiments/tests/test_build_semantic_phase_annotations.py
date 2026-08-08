from __future__ import annotations

from imitation_experiments.evaluation.build_semantic_phase_annotations import (
    _event_end_steps,
    build_annotations,
)


def test_event_end_steps_fill_initial_gap_and_exact_final_length() -> None:
    events = [
        {"start_time": 0.3, "end_time": 1.0},
        {"start_time": 1.0, "end_time": 2.03},
    ]

    assert _event_end_steps(events, reference_frames=103, fps=50.0) == [50, 103]


def test_build_annotations_joins_events_and_manual_traits() -> None:
    selection = {
        "motions": [
            {
                "motion_name": "walk",
                "trajectory_rank": 4,
                "reference_frames": 100,
                "language_goal": "walk and wave",
                "category": "test",
            }
        ]
    }
    language = {
        "motions": [
            {
                "name": "walk",
                "events": [
                    {
                        "start_time": 0.0,
                        "end_time": 1.0,
                        "description": "A person walks forward.",
                    },
                    {
                        "start_time": 1.0,
                        "end_time": 2.0,
                        "description": "A person waves their right hand.",
                    },
                ],
            }
        ]
    }
    traits = {
        "semantic_axes": {
            "locomoting": "root translates",
            "manipulating": "operates object",
        },
        "motions": {
            "walk": [
                {"activity": "locomotion", "true_axes": ["locomoting"]},
                {"activity": "gesture", "true_axes": []},
            ]
        },
    }

    result = build_annotations(
        selection_payload=selection,
        language_payload=language,
        trait_payload=traits,
        output_fps=50.0,
        video_root=None,
    )

    phases = result["motions"][0]["phases"]
    assert [(row["start_step"], row["end_step"]) for row in phases] == [
        (0, 50),
        (50, 100),
    ]
    assert phases[0]["semantics"] == {"locomoting": True, "manipulating": False}
    assert phases[1]["activity"] == "gesture"
