"""Static guards for the SONIC release evaluator CLI.

The evaluator launches Isaac Sim at import time, so these tests inspect the
entrypoint source. They protect the release-surface safety gates without
starting a simulator in the default Pixi test environment.
"""

from __future__ import annotations

from imitation_experiments.paths import REPO_ROOT


EVALUATOR = (
    REPO_ROOT
    / "source/imitation_experiments/imitation_experiments/lowlevel/evaluate_sonic_release.py"
)


def _source() -> str:
    return EVALUATOR.read_text(encoding="utf-8")


def test_release_evaluator_bakes_sonic_termination_contract() -> None:
    source = _source()
    assert '"anchor_pos": {"threshold": 0.25, "down_threshold": 0.25}' in source
    assert '"anchor_ori": {"threshold": 1.0}' in source
    assert '"ee_body_pos": {"threshold": 0.25, "down_threshold": 0.25}' in source
    assert (
        'SONIC_SUCCESS_DISABLED_TERMINATIONS = ("foot_pos_xyz", "base_too_low")'
        in source
    )
    assert "--termination_contract" in source
    assert 'default="sonic"' in source


def test_release_evaluator_has_episode_and_completion_gates() -> None:
    source = _source()
    assert "_extend_episode_length_for_steps" in source
    assert "--preserve_episode_length" in source
    assert "--allow_incomplete_release" in source
    assert "_require_reportable_release(summary)" in source
    assert 'summary.get("stop_reason") != "all_envs_done"' in source
    assert '"done_rate"' in source
    assert '"time_out_rate"' in source


def test_release_evaluator_aliases_mpjpe_l_to_tracking_mpjpe() -> None:
    source = _source()
    assert "def _with_tracking_mpjpe_alias" in source
    assert '"tracking_mpjpe_mm" not in metrics and "mpjpe_l_mm" in metrics' in source
    assert 'metrics["tracking_mpjpe_mm"] = metrics["mpjpe_l_mm"]' in source


def test_release_evaluator_defaults_to_checkpoint_schedule() -> None:
    source = _source()
    assert '"--reset_schedule"' in source
    assert 'default="sequential"' in source
