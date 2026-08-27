"""Unit tests for push-to-termination attribution (torch-only, no Isaac)."""

import torch

from imitation_experiments.evaluation.push_attribution import (
    PushAttributionTracker,
    attach_push_tracker,
    histogram_since_push,
)


def _mask(num_envs, ids):
    mask = torch.zeros(num_envs, dtype=torch.bool)
    mask[list(ids)] = True
    return mask


def test_distance_and_per_term_accounting():
    tracker = PushAttributionTracker(4)
    tracker.record_push([1, 2], step=100)
    tracker.on_terminal(
        terminated_mask=_mask(4, [1]),
        done_mask=_mask(4, [1]),
        step=112,
        term_masks={"ee_body_pos": _mask(4, [1]), "anchor_pos": _mask(4, [])},
    )
    assert tracker.distances == [12]
    assert tracker.per_term_distances == {"ee_body_pos": [12]}
    assert tracker.no_push_seen == 0


def test_termination_without_prior_push_goes_to_no_push_bucket():
    tracker = PushAttributionTracker(2)
    tracker.on_terminal(_mask(2, [0]), _mask(2, [0]), step=50)
    assert tracker.distances == []
    assert tracker.no_push_seen == 1


def test_done_clears_push_so_next_episode_cannot_attribute():
    tracker = PushAttributionTracker(2)
    tracker.record_push([0], step=10)
    # Truncation (done without termination) must clear the push.
    tracker.on_terminal(_mask(2, []), _mask(2, [0]), step=20)
    tracker.on_terminal(_mask(2, [0]), _mask(2, [0]), step=90)
    assert tracker.distances == []
    assert tracker.no_push_seen == 1


def test_histogram_buckets_and_open_tail():
    counts = histogram_since_push([3, 10, 26, 999])
    assert counts["<=5"] == 1
    assert counts["<=10"] == 1
    assert counts["<=25"] == 0
    assert counts["<=50"] == 1
    assert counts[">250"] == 1


def test_completion_terms_stay_out_of_overall_buckets():
    tracker = PushAttributionTracker(2)
    tracker.record_push([0, 1], step=0)
    tracker.on_terminal(
        _mask(2, [0, 1]),
        _mask(2, [0, 1]),
        step=40,
        term_masks={
            "reference_finished": _mask(2, [0]),
            "ee_body_pos": _mask(2, [1]),
        },
    )
    assert tracker.distances == [40]  # failure only
    assert tracker.per_term_distances["reference_finished"] == [40]
    assert tracker.per_term_distances["ee_body_pos"] == [40]


def test_summary_shape():
    tracker = PushAttributionTracker(2)
    tracker.record_push([0], step=0)
    tracker.on_terminal(
        _mask(2, [0]), _mask(2, [0]), step=30, term_masks={"ee_body_pos": _mask(2, [0])}
    )
    out = tracker.summary()
    assert out["push_events"] == 1
    assert out["terminations_with_push"] == 1
    assert out["frac_within"]["50"] == 1.0
    assert out["per_term"]["ee_body_pos"]["count"] == 1


class _FakeEventManager:
    def __init__(self, cfg):
        self._cfg = cfg

    def get_term_cfg(self, name):
        if name != "push_robot":
            raise ValueError(name)
        return self._cfg


class _FakeCfg:
    def __init__(self, func):
        self.func = func


class _FakeEnv:
    common_step_counter = 7


def test_attach_wraps_and_preserves_call():
    calls = []

    def push(env, env_ids, velocity_range=None):
        calls.append((env_ids, velocity_range))
        return "pushed"

    cfg = _FakeCfg(push)
    tracker = attach_push_tracker(_FakeEventManager(cfg), num_envs=4)
    assert tracker is not None
    result = cfg.func(_FakeEnv(), torch.tensor([2]), velocity_range={"x": (0, 1)})
    assert result == "pushed"
    assert calls and calls[0][1] == {"x": (0, 1)}
    assert tracker.push_events == 1
    assert int(tracker._last_push_step[2]) == 7


def test_attach_returns_none_without_manager_or_term():
    assert attach_push_tracker(None, 4) is None
    cfg = _FakeCfg(None)
    assert attach_push_tracker(_FakeEventManager(cfg), 4) is None
