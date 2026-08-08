from __future__ import annotations

import json
from pathlib import Path

import pytest

from imitation_experiments.data.prepare_language_motion_selection import (
    _validate_hold_screen,
    prepare_language_motion_selection,
)


def test_prepare_language_motion_selection_freezes_text_and_order(
    tmp_path: Path,
) -> None:
    clips = tmp_path / "clips"
    clips.mkdir()
    for name in ("walk", "wave"):
        (clips / f"{name}.npz").write_bytes(b"npz")
    sidecar = tmp_path / "language.json"
    sidecar.write_text(
        json.dumps(
            {
                "motions": [
                    {"name": "walk", "language_goal": "old walk"},
                    {"name": "wave", "language_goal": "old wave"},
                ]
            }
        ),
        encoding="utf-8",
    )
    source = tmp_path / "manifest.json"
    source.write_text(
        json.dumps(
            {
                "dataset": {
                    "trajectories": {
                        "lafan1_csv": [
                            {"name": name, "path": str(clips / f"{name}.npz")}
                            for name in ("walk", "wave")
                        ]
                    }
                },
                "metadata": {"language_annotations_path": str(sidecar)},
            }
        ),
        encoding="utf-8",
    )
    selection = tmp_path / "selected.json"
    selection.write_text(
        json.dumps(
            {
                "motions": [
                    {
                        "motion_name": "wave",
                        "language_goal": "Raise your right hand.",
                        "trajectory_rank": 7,
                    },
                    {
                        "motion_name": "walk",
                        "language_goal": "Walk forward.",
                        "trajectory_rank": 3,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "derived" / "selected_manifest.json"

    record = prepare_language_motion_selection(
        source_manifest=source, selection=selection, output_manifest=output
    )

    manifest = json.loads(output.read_text(encoding="utf-8"))
    entries = manifest["dataset"]["trajectories"]["lafan1_csv"]
    assert [entry["name"] for entry in entries] == ["wave", "walk"]
    derived_sidecar = Path(record["language_sidecar"])
    language = json.loads(derived_sidecar.read_text(encoding="utf-8"))
    assert [row["language_goal"] for row in language["motions"]] == [
        "Raise your right hand.",
        "Walk forward.",
    ]
    assert language["motions"][0]["selection_metrics"]["trajectory_rank"] == 7
    assert Path(manifest["metadata"]["preparation_record"]).is_file()


def test_hold_screen_accepts_active_motions() -> None:
    _validate_hold_screen(
        {
            "hold_screen": {
                "max_hold_fraction": 0.2,
                "max_longest_hold_s": 1.5,
            }
        },
        [
            {
                "motion_name": "walk",
                "hold_fraction": 0.05,
                "longest_hold_s": 0.3,
            }
        ],
    )


@pytest.mark.parametrize(
    ("hold_fraction", "longest_hold_s"),
    [(0.21, 0.3), (0.05, 1.51)],
)
def test_hold_screen_rejects_long_holding_motions(
    hold_fraction: float, longest_hold_s: float
) -> None:
    with pytest.raises(ValueError, match="fails the hold screen"):
        _validate_hold_screen(
            {
                "hold_screen": {
                    "max_hold_fraction": 0.2,
                    "max_longest_hold_s": 1.5,
                }
            },
            [
                {
                    "motion_name": "idle",
                    "hold_fraction": hold_fraction,
                    "longest_hold_s": longest_hold_s,
                }
            ],
        )
