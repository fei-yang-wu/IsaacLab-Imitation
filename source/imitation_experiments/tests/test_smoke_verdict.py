"""The wiring gate for an interface-design-study arm.

These tests pin what the gate is allowed to accept. The failure it exists to
catch is an arm whose checkpoint trained a DIFFERENT design from the one the
campaign declares -- that would silently turn an ablation cell into a duplicate
of another.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from imitation_experiments.lowlevel.smoke_verdict import verdict_for


def _arm(
    tmp_path: Path,
    *,
    objective: str = "endpoint",
    latent_mode: str = "deterministic",
    z_dim: int = 256,
    rows: list[dict] | None = None,
) -> Path:
    encoder = tmp_path / "encoder"
    checkpoints = encoder / "checkpoints"
    checkpoints.mkdir(parents=True)
    checkpoint = checkpoints / "latest.pt"
    torch.save(
        {
            "skill_encoder_state_dict": {},
            "config": {
                "transition_objective": objective,
                "latent_mode": latent_mode,
                "z_dim": z_dim,
            },
        },
        checkpoint,
    )
    if rows is None:
        rows = [
            {
                "update": 1,
                "train/loss": 38.6,
                "train/z_effective_rank": 87.8,
                "train/z_dim_std_mean": 0.139,
                "train/z_dim_std_min": 0.081,
            }
        ]
    log_dir = encoder / "2026-08-19_12-00-00"
    log_dir.mkdir()
    (log_dir / "metrics.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    return checkpoint


def test_a_clean_arm_passes(tmp_path: Path) -> None:
    verdict = verdict_for(
        arm="ctrl",
        checkpoint=_arm(tmp_path),
        lowlevel_exit=0,
        expected_command_dim=258,
        expected_objective="endpoint",
        expected_latent_mode="deterministic",
        expected_z_dim=256,
    )
    assert verdict.status == "pass"
    assert verdict.checks["z_effective_rank"] == pytest.approx(87.8)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("expected_objective", "reconstruction"),
        ("expected_latent_mode", "sonic_fsq"),
        ("expected_z_dim", 64),
    ),
)
def test_a_checkpoint_that_trained_another_design_fails(
    tmp_path: Path, field: str, value: object
) -> None:
    """The gate's main job: an ablation cell must not silently be a copy of a
    different cell."""
    kwargs = {
        "expected_objective": "endpoint",
        "expected_latent_mode": "deterministic",
        "expected_z_dim": 256,
    }
    kwargs[field] = value
    verdict = verdict_for(
        arm="ctrl",
        checkpoint=_arm(tmp_path),
        lowlevel_exit=0,
        expected_command_dim=258,
        **kwargs,
    )
    assert verdict.status == "fail"
    assert "campaign says" in (verdict.reason or "")


def test_a_quantized_arm_is_judged_on_codebook_perplexity(tmp_path: Path) -> None:
    """Perplexity 1.0 is a single code. Measured on a real FSQ smoke at four
    updates: perplexity 27.1 at code usage 0.9375."""
    checkpoint = _arm(
        tmp_path,
        latent_mode="sonic_fsq",
        z_dim=64,
        rows=[
            {
                "update": 1,
                "train/loss": 1.0,
                "train/z_effective_rank": 34.5,
                "train/z_dim_std_mean": 0.091,
                "train/z_dim_std_min": 0.023,
                "train/diversity/code_perplexity": 27.14,
                "train/diversity/code_usage_frac": 0.9375,
            }
        ],
    )
    verdict = verdict_for(
        arm="bn_sonic_fsq64",
        checkpoint=checkpoint,
        lowlevel_exit=0,
        expected_command_dim=66,
        expected_latent_mode="sonic_fsq",
        expected_z_dim=64,
    )
    assert verdict.status == "pass"
    assert verdict.checks["code_perplexity"] == pytest.approx(27.14)


def test_a_codebook_on_one_level_fails(tmp_path: Path) -> None:
    checkpoint = _arm(
        tmp_path,
        latent_mode="vq",
        rows=[
            {
                "update": 1,
                "train/loss": 1.0,
                "train/z_effective_rank": 11.57,
                "train/z_dim_std_mean": 0.0,
                "train/diversity/code_perplexity": 1.0,
            }
        ],
    )
    verdict = verdict_for(
        arm="bn_vq_ema",
        checkpoint=checkpoint,
        lowlevel_exit=0,
        expected_command_dim=258,
    )
    assert verdict.status == "fail"
    assert "one level" in (verdict.reason or "")


def test_a_continuous_arm_falls_back_to_the_code_spread(tmp_path: Path) -> None:
    """A continuous bottleneck has no codebook, so perplexity is absent and the
    spread of the code is the only signal available."""
    checkpoint = _arm(
        tmp_path,
        rows=[
            {
                "update": 1,
                "train/loss": 1.0,
                "train/z_effective_rank": 11.57,
                "train/z_dim_std_mean": 0.0,
                "train/z_dim_std_min": 0.0,
            }
        ],
    )
    verdict = verdict_for(
        arm="ctrl", checkpoint=checkpoint, lowlevel_exit=0, expected_command_dim=258
    )
    assert verdict.status == "fail"
    assert "collapsed" in (verdict.reason or "")


def test_dead_dimensions_in_a_working_codebook_pass(tmp_path: Path) -> None:
    """The real `bn_gumbel_multicat` smoke: `z_dim_std_min` 0.0 at a healthy
    `z_dim_std_mean` 0.317. Gating on the MINIMUM would fail a working
    quantizer."""
    checkpoint = _arm(
        tmp_path,
        latent_mode="gumbel_multicat",
        z_dim=64,
        rows=[
            {
                "update": 1,
                "train/loss": 1.0,
                "train/z_effective_rank": 35.6,
                "train/z_dim_std_mean": 0.317,
                "train/z_dim_std_min": 0.0,
            }
        ],
    )
    verdict = verdict_for(
        arm="bn_gumbel_multicat",
        checkpoint=checkpoint,
        lowlevel_exit=0,
        expected_command_dim=66,
        expected_latent_mode="gumbel_multicat",
        expected_z_dim=64,
    )
    assert verdict.status == "pass"


def test_non_finite_metrics_fail(tmp_path: Path) -> None:
    checkpoint = _arm(
        tmp_path,
        rows=[
            {
                "update": 1,
                "train/loss": float("nan"),
                "train/z_effective_rank": 8.0,
                "train/z_dim_std_mean": 0.1,
                "train/z_dim_std_min": 0.1,
            }
        ],
    )
    verdict = verdict_for(
        arm="ctrl", checkpoint=checkpoint, lowlevel_exit=0, expected_command_dim=258
    )
    assert verdict.status == "fail"
    assert "non-finite" in (verdict.reason or "")


def test_a_failed_low_level_iteration_fails(tmp_path: Path) -> None:
    verdict = verdict_for(
        arm="ctrl", checkpoint=_arm(tmp_path), lowlevel_exit=1, expected_command_dim=258
    )
    assert verdict.status == "fail"
    assert "exited 1" in (verdict.reason or "")


def test_a_missing_checkpoint_fails(tmp_path: Path) -> None:
    verdict = verdict_for(
        arm="ctrl",
        checkpoint=tmp_path / "nope.pt",
        lowlevel_exit=0,
        expected_command_dim=258,
    )
    assert verdict.status == "fail"
    assert "no encoder checkpoint" in (verdict.reason or "")


def test_the_gate_does_not_judge_learning(tmp_path: Path) -> None:
    """At a few updates the code is expected to be WORSE than a zero code, so
    gating on the learning signal would fail every arm. Measured on the real
    control arm: loss_real_z_eval 39.25 against loss_zero_z_eval 38.17."""
    checkpoint = _arm(
        tmp_path,
        rows=[
            {
                "update": 1,
                "train/loss": 38.6,
                "train/loss_real_z_eval": 39.25,
                "train/loss_zero_z_eval": 38.17,
                "train/z_effective_rank": 87.8,
                "train/z_dim_std_mean": 0.139,
                "train/z_dim_std_min": 0.081,
            }
        ],
    )
    verdict = verdict_for(
        arm="ctrl", checkpoint=checkpoint, lowlevel_exit=0, expected_command_dim=258
    )
    assert verdict.status == "pass"
