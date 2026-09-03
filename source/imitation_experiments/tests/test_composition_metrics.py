"""Contracts for the composition-probe metrics and pair selection."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from imitation_experiments.evaluation import composition_metrics as cm
from imitation_experiments.evaluation import composition_pairs as cp
from imitation_experiments.evaluation.composition_probe import (
    ArmConfig,
    Setting,
    evaluator_args,
    make_plan,
)


def _gait(
    steps: int, hz: float, amp: float = 0.4, arm: float = 0.2
) -> list[list[float]]:
    t = np.arange(steps) * cm.CONTROL_DT
    rows = np.zeros((steps, 29))
    rows[:, 0] = amp * np.sin(2 * np.pi * hz * t)
    rows[:, 1] = -amp * np.sin(2 * np.pi * hz * t)
    rows[:, 11] = arm * np.sin(2 * np.pi * hz * t)
    rows[:, 12] = -arm * np.sin(2 * np.pi * hz * t)
    return rows.tolist()


def test_stride_frequency_recovers_the_hip_frequency():
    hz, frac = cm.stride_frequency_hz(_gait(300, 1.5), 0, 300)
    assert hz == pytest.approx(1.5, abs=0.1)
    assert frac > 0.9
    hz_stand, frac_stand = cm.stride_frequency_hz([[0.0] * 29] * 300, 0, 300)
    assert math.isnan(hz_stand) or frac_stand == 0.0


def test_arm_swing_amplitude_is_the_percentile_range():
    amp = cm.arm_swing_amplitude(_gait(300, 1.0, arm=0.3), 0, 300)
    assert amp == pytest.approx(0.6, abs=0.06)


def test_settling_time_and_peak_action_delta():
    speed = [0.8] * 100 + [1.0, 1.2, 1.4, 1.6, 1.8] + [2.0] * 100
    # 1.8 is inside the 0.3 tolerance of 2.0, so settling starts at index 104.
    assert cm.settling_time_steps(speed, 2.0, 100, hold_steps=10) == 4
    assert cm.settling_time_steps(speed, 5.0, 100) is None
    assert cm.peak_action_delta([0.1, 0.5, float("nan"), 3.0, 0.2], 1, window=3) == 3.0


def test_episode_metrics_and_monotone_fraction():
    target = {
        "root_speed": [0.8] * 150 + [1.5] * 150,
        "upright": [1.0] * 300,
        "action_delta": [0.5] * 300,
        "joint_pos": _gait(300, 1.2),
        "code_distance": [5.0] * 300,
    }
    source = {
        "root_speed": [1.5] * 300,
        "upright": [1.0] * 300,
        "action_delta": [0.5] * 300,
        "joint_pos": _gait(300, 2.0),
        "code_distance": [],
    }
    m = cm.episode_metrics(target, source, start_step=150, ramp_steps=0)
    assert m["fall_free"] is True
    assert m["speed_pre"] == pytest.approx(0.8)
    assert m["speed_post"] == pytest.approx(1.5)
    assert m["settling_steps"] == 0
    assert m["stride_hz_post"] == pytest.approx(1.2, abs=0.15)
    assert m["source_stride_hz_post"] == pytest.approx(2.0, abs=0.15)
    assert m["joint_gait_distance_post"] > 0.0
    assert cm.monotone_fraction([(0, 0.8), (0.5, 1.1), (1, 1.5)]) == 1.0
    assert cm.monotone_fraction([(0, 0.8), (0.5, 0.2), (1, 1.5)]) == 0.5
    assert cm.monotone_fraction([(0, 1.0), (1, 1.0)]) is None


def test_aggregate_groups_and_rates():
    rows = [
        {
            "label": "a",
            "final_alpha": 0.5,
            "fall_free": True,
            "settling_steps": 3,
            "speed_post": 1.0,
        },
        {
            "label": "a",
            "final_alpha": 0.5,
            "fall_free": False,
            "settling_steps": None,
            "speed_post": float("nan"),
        },
    ]
    for r in rows:
        for k in (
            "speed_pre",
            "speed_ramp",
            "source_speed_post",
            "stride_hz_post",
            "source_stride_hz_post",
            "arm_swing_post",
            "source_arm_swing_post",
            "action_delta_pre",
            "action_delta_ramp",
            "action_delta_post",
            "peak_action_delta_after_switch",
            "joint_gait_distance_post",
            "code_distance_post",
        ):
            r.setdefault(k, float("nan"))
    agg = cm.aggregate(rows, ["label", "final_alpha"])
    assert (
        agg[0]["n"] == 2
        and agg[0]["fall_free_rate"] == 0.5
        and agg[0]["settled_rate"] == 0.5
    )
    assert agg[0]["speed_post"] == pytest.approx(1.0)
    assert "| label |" in cm.markdown_table(agg, ["label", "n", "fall_free_rate"])


def test_pair_selection_respects_categories_survivors_and_length(tmp_path):
    manifest = {
        "traj_info": {
            "ordered_traj_list": [
                ["x", "walk_forward_a", "t"],
                ["x", "walk_forward_b", "t"],
                ["x", "jog_forward_a", "t"],
                ["x", "jog_forward_short", "t"],
                ["x", "walk_backward_a", "t"],
                ["x", "idle_a", "t"],
            ],
            "start_index": [0, 500, 1000, 1500, 1600, 2100],
            "end_index": [500, 1000, 1500, 1600, 2100, 2600],
        }
    }
    mp = tmp_path / "m.json"
    mp.write_text(json.dumps(manifest))
    surv = tmp_path / "s.json"
    surv.write_text(
        json.dumps(
            {
                "per_environment": [
                    {"trajectory_rank": r, "tracking_success": r != 1} for r in range(6)
                ]
            }
        )
    )
    clips = cp.load_clips(mp)
    by_kind = cp.categorize(clips, 320, cp.load_survivors([surv]))
    assert [c["rank"] for c in by_kind["walk"]] == [
        0
    ]  # rank 1 not a survivor, rank 4 backward
    assert [c["rank"] for c in by_kind["jog"]] == [2]  # rank 3 too short
    assert [c["rank"] for c in by_kind["stand"]] == [5]
    pairs = cp.draw_pairs(
        by_kind,
        [("walk", "jog"), ("walk", "stand"), ("stand", "wave")],
        pairs_per_kind=3,
        seed=1,
    )
    assert {p["kind"] for p in pairs} == {
        "walk->jog",
        "walk->stand",
    }  # no wave clips: kind dropped
    assert all(p["a"] != p["b"] for p in pairs)


def test_probe_plan_and_evaluator_args(tmp_path):
    plan = make_plan(
        "held_alpha", steps=300, alphas=[0.0, 0.5], ramps=[], switch_steps=[]
    )
    assert [s.label for s in plan.settings] == ["alpha p0.00", "alpha p0.50"] or [
        s.label for s in plan.settings
    ] == ["alphap0.00", "alphap0.50"]
    plan = make_plan(
        "handover", steps=300, alphas=[], ramps=[0, 50], switch_steps=[150]
    )
    assert [(s.start_step, s.ramp_steps, s.final_alpha) for s in plan.settings] == [
        (150, 0, 1.0),
        (150, 50, 1.0),
    ]
    arm = ArmConfig(
        "lstm_affine",
        tmp_path / "ckpt.pt",
        tmp_path / "enc.pt",
        extra_overrides=("agent.ppo.rnn_hidden_size=256",),
    )
    args = evaluator_args(
        arm,
        [{"a": 10, "b": 20}, {"a": 30, "b": 40}],
        Setting("held_alpha", "alphap0.50", 0, 0, 0.5, 300),
        out_json=tmp_path / "o.json",
        reference_arrays="/ref",
        persist_id="pid",
        physics="newton_mjwarp",
        seed=0,
    )
    i = args.index("--trajectory_ranks")
    assert args[i + 1 : i + 5] == ["10", "20", "30", "40"]
    assert args[args.index("--num_envs") + 1] == "4"
    assert (
        "--latent_blend_layout" in args
        and args[args.index("--latent_blend_layout") + 1] == "pairs"
    )
    assert args[args.index("--latent_blend_final_alpha") + 1] == "0.5"
    assert "env.terminations.ee_body_pos=null" in args
    assert args[-1] == "agent.ppo.rnn_hidden_size=256"
