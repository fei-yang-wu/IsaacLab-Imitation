import numpy as np
import pytest

from imitation_experiments.evaluation.vega_sharpa_policy import (
    EpisodeMeasurements,
    fixed_start_frames,
)


def test_finished_world_does_not_add_reset_measurements():
    measured = EpisodeMeasurements(2)
    measured.add({"error": np.array([0.1, 0.2])}, np.array([True, True]))
    # World zero finished in the preceding step; its post-reset state must
    # not dilute its error or count toward a second episode.
    measured.add({"error": np.array([np.nan, 0.4])}, np.array([False, True]))
    result = measured.summary()["error"]
    np.testing.assert_allclose(result["per_episode_mean"], [0.1, 0.3])
    np.testing.assert_array_equal(measured.counts, [1, 2])
    assert result["maximum"] == 0.4


def test_active_nonfinite_measurement_is_rejected():
    with pytest.raises(ValueError, match="Invalid active-episode metric"):
        EpisodeMeasurements(1).add({"error": np.array([np.nan])}, np.array([True]))


def test_grid_uses_the_same_eligible_reset_pool_including_endpoints():
    allowed = [0, 5, 6, 10, 12, 20, 25]
    np.testing.assert_array_equal(fixed_start_frames(allowed, 3, "grid"), [0, 10, 25])
    np.testing.assert_array_equal(fixed_start_frames(allowed, 3, "first"), [0, 0, 0])
