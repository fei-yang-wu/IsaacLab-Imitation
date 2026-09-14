"""Do not present mismatched or partial evaluations as a controlled result."""

import json

import pytest

from imitation_experiments.evaluation.vega_sharpa_comparison import collect_comparison


def _report():
    return {
        "complete": True,
        "expected_result_count": 3,
        "reference_manifest_sha256": "manifest",
        "reference_sha256": "data",
        "seed": 42,
        "num_envs": 2,
        "protocol": {
            "name": "object-task",
            "position_tolerance_m": 0.05,
            "orientation_tolerance_rad_secondary": 0.35,
            "observation_corruption": False,
        },
        "results": [
            {
                "policy": "model_step_3072000",
                "start_mode": "first",
                "policy_and_normalizer_unchanged": True,
                "assistance_scale": scale,
                "checkpoint": "model_step_3072000.pt",
                "start_frames": [0, 0],
                "episode_control_steps": [503, 503],
                "object_endpoint": {
                    "reached_reference_end": [True, True],
                    "trial_count": 2,
                    "position_error_m": {"mean": 0.1, "per_trial": [0.1, 0.1]},
                    "orientation_error_rad": {"mean": 0.0, "per_trial": [0.0, 0.0]},
                    "position_success_count": 0,
                },
            }
            for scale in [1.0, 0.75, 0.0]
        ],
    }


def test_matched_reports_preserve_every_case_and_source_identity(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(_report()))
    result = collect_comparison([("Original", path), ("Changed", path)])
    assert len(result["rows"]) == 6
    assert result["labels"] == ["Original", "Changed"]
    assert result["sources"][0]["sha256"] == result["sources"][1]["sha256"]


@pytest.mark.parametrize(
    "bad_case", ["incomplete", "reference", "missing_scale", "start", "unfrozen"]
)
def test_rejects_comparisons_that_do_not_support_the_claim(tmp_path, bad_case):
    first = tmp_path / "first.json"
    first.write_text(json.dumps(_report()))
    report = _report()
    if bad_case == "incomplete":
        report["complete"] = False
    elif bad_case == "reference":
        report["reference_sha256"] = "different"
    elif bad_case == "missing_scale":
        report["results"].pop()
        report["expected_result_count"] = 2
    elif bad_case == "start":
        report["results"][0]["start_frames"] = [1, 1]
    else:
        report["results"][0]["policy_and_normalizer_unchanged"] = False
    second = tmp_path / "second.json"
    second.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        collect_comparison([("Original", first), ("Changed", second)])
