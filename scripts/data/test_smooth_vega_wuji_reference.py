"""Filtering contract for the Vega-Wuji Reference joint trajectory."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = Path(__file__).with_name("smooth_vega_wuji_reference.py")
MODULE_SPEC = importlib.util.spec_from_file_location("_smooth_reference", MODULE_PATH)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
MODULE = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(MODULE)

savitzky_golay = MODULE.savitzky_golay


def _jittery_trajectory(frames: int = 32, joints: int = 4) -> np.ndarray:
    """A smooth ramp with a one-frame IK flip on a single joint."""

    time = np.linspace(0.0, 1.0, frames)
    values = np.stack([np.sin(2.0 * np.pi * time) for _ in range(joints)], axis=1)
    values[frames // 2, 0] += 0.9
    return values


def test_filtering_reduces_the_worst_per_frame_step() -> None:
    raw = _jittery_trajectory()
    before = np.abs(np.diff(raw, axis=0)).max()
    after = np.abs(np.diff(savitzky_golay(raw, 9, 2), axis=0)).max()
    assert after < before / 2.0


def test_filtering_preserves_shape_and_stays_finite() -> None:
    raw = _jittery_trajectory()
    smoothed = savitzky_golay(raw, 9, 2)
    assert smoothed.shape == raw.shape
    assert np.isfinite(smoothed).all()


def test_a_window_of_one_is_a_passthrough() -> None:
    raw = _jittery_trajectory()
    assert np.array_equal(savitzky_golay(raw, 1, 2), raw)


def test_a_window_longer_than_the_clip_is_clamped_not_an_error() -> None:
    raw = _jittery_trajectory(frames=9)
    smoothed = savitzky_golay(raw, 99, 2)
    assert smoothed.shape == raw.shape
    assert np.isfinite(smoothed).all()


def test_a_clip_too_short_to_filter_is_returned_unchanged() -> None:
    raw = _jittery_trajectory(frames=3)
    assert np.array_equal(savitzky_golay(raw, 3, 2), raw)


def test_velocity_recomputed_from_positions_is_self_consistent() -> None:
    """The invariant the tool exists to restore: qvel is d(qpos)/dt."""

    fps = 20.0
    smoothed = savitzky_golay(_jittery_trajectory(), 9, 2)
    qvel = np.gradient(smoothed, 1.0 / fps, axis=0)
    assert np.abs(qvel - np.gradient(smoothed, 1.0 / fps, axis=0)).max() == 0.0
    # A ramp of known slope recovers that slope away from the endpoints.
    ramp = np.linspace(0.0, 1.0, 40)[:, None] * np.ones((1, 3))
    slope = np.gradient(ramp, 1.0 / fps, axis=0)
    assert slope[5:-5] == pytest.approx((1.0 / 39.0) * fps, rel=1e-6)
