"""Tests for the EC rehearsal sensor-noise model.

The worker keeps every ``embodied_control`` import inside a function, so the
noise math is importable and testable in the default environment without the
Embodied-Control ``lowlevel-sim`` Pixi env.

What these pin is the property that makes the rehearsal number trustworthy:
noise reaches the controller's view of the robot and nothing else. If noise
ever leaked into the anchor fields, MPJPE would be scoring the noise we
injected rather than the tracking error, and the number would fail upward —
plausible, stable, and wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from imitation_experiments.evaluation.ec_sidecar_worker import (
    DEFAULT_NOISE,
    _episode_seed,
    _sensed,
)
from imitation_experiments.evaluation.protocol import (
    PROTOCOLS,
    SONIC_OBSERVATION_NOISE,
)


@dataclass(frozen=True)
class FakeRobotState:
    """Stands in for ``embodied_control.lowlevel.contracts.RobotState``.

    ``frozen=True`` is load-bearing, not decoration: the real ``RobotState`` is
    frozen, and an unfrozen fake let a ``setattr``-based implementation pass
    every unit test here while failing instantly against the real rig.
    """

    stamp: float = 0.0
    joint_pos: np.ndarray = field(default_factory=lambda: np.zeros(29, np.float32))
    joint_vel: np.ndarray = field(default_factory=lambda: np.zeros(29, np.float32))
    projected_gravity: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, -1.0], np.float32)
    )
    base_ang_vel: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    anchor_pos_w: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 2.0, 0.76], np.float32)
    )
    anchor_quat_w: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0], np.float32)
    )


def test_noise_never_touches_the_fields_metrics_are_computed_from() -> None:
    """MPJPE reads anchor pose and FK-replayed joints from the TRUE state."""
    state = FakeRobotState()
    sensed = _sensed(state, DEFAULT_NOISE, np.random.default_rng(0))

    assert np.array_equal(sensed.anchor_pos_w, state.anchor_pos_w)
    assert np.array_equal(sensed.anchor_quat_w, state.anchor_quat_w)


def test_sensed_state_is_a_copy_and_leaves_the_true_state_intact() -> None:
    state = FakeRobotState(joint_pos=np.full(29, 0.3, np.float32))
    before = state.joint_pos.copy()

    sensed = _sensed(state, DEFAULT_NOISE, np.random.default_rng(0))

    assert np.array_equal(state.joint_pos, before), "true state was mutated"
    assert not np.array_equal(sensed.joint_pos, before), "noise was not applied"


def test_every_noised_channel_stays_inside_its_declared_half_width() -> None:
    state = FakeRobotState()
    rng = np.random.default_rng(7)
    for _ in range(200):
        sensed = _sensed(state, DEFAULT_NOISE, rng)
        assert np.abs(sensed.joint_pos - state.joint_pos).max() <= 0.01
        assert np.abs(sensed.joint_vel - state.joint_vel).max() <= 0.5
        assert np.abs(sensed.projected_gravity - state.projected_gravity).max() <= 0.05
        assert np.abs(sensed.base_ang_vel - state.base_ang_vel).max() <= 0.2


def test_noise_is_reproducible_from_the_seed_alone() -> None:
    """Sync lockstep survives noise only because the stream is seeded."""
    state = FakeRobotState()
    first = _sensed(state, DEFAULT_NOISE, np.random.default_rng(11))
    second = _sensed(state, DEFAULT_NOISE, np.random.default_rng(11))

    assert np.array_equal(first.joint_pos, second.joint_pos)
    assert np.array_equal(first.base_ang_vel, second.base_ang_vel)


def test_empty_noise_is_an_exact_passthrough() -> None:
    """The deterministic path must be untouched, or the drift guard is void."""
    state = FakeRobotState()
    assert _sensed(state, {}, np.random.default_rng(0)) is state


def test_zero_magnitude_channel_is_left_alone() -> None:
    state = FakeRobotState()
    sensed = _sensed(
        state, {"joint_pos": 0.0, "joint_vel": 0.5}, np.random.default_rng(0)
    )

    assert np.array_equal(sensed.joint_pos, state.joint_pos)
    assert not np.array_equal(sensed.joint_vel, state.joint_vel)


def test_projected_gravity_is_not_renormalized() -> None:
    """Isaac adds noise to the observation without restoring the unit norm.

    Renormalizing here would hand the policy a cleaner gravity vector than it
    ever saw in training, quietly making the rehearsal optimistic.
    """
    state = FakeRobotState()
    norms = [
        float(
            np.linalg.norm(
                _sensed(
                    state, DEFAULT_NOISE, np.random.default_rng(s)
                ).projected_gravity
            )
        )
        for s in range(50)
    ]
    assert not all(abs(norm - 1.0) < 1e-6 for norm in norms)


def test_episode_seed_is_distinct_across_motion_seed_and_repeat() -> None:
    cases = [
        {"trajectory_rank": rank, "env_seed": seed, "repeat_index": repeat}
        for rank in range(10)
        for seed in (0, 1, 2)
        for repeat in range(5)
    ]
    seeds = [_episode_seed(case) for case in cases]

    assert len(set(seeds)) == len(cases), "two episodes would share a noise stream"


def test_repeats_of_one_motion_draw_different_noise() -> None:
    """Repeats are samples now, so they must not be identical rollouts."""
    state = FakeRobotState()
    draws = [
        _sensed(
            state,
            DEFAULT_NOISE,
            np.random.default_rng(
                _episode_seed(
                    {"trajectory_rank": 3, "env_seed": 0, "repeat_index": repeat}
                )
            ),
        ).joint_vel
        for repeat in range(5)
    ]
    for index in range(1, len(draws)):
        assert not np.array_equal(draws[0], draws[index])


def test_worker_default_matches_the_protocol_registry() -> None:
    """The worker fallback and the hashed protocol must not disagree."""
    assert DEFAULT_NOISE == dict(SONIC_OBSERVATION_NOISE)
    assert DEFAULT_NOISE == dict(PROTOCOLS["ec_latent_rehearsal_v1"].observation_noise)


def test_ec_rehearsal_carries_no_physics_randomization() -> None:
    """Physics DR is a training device.

    The rehearsal is the last mile before hardware, and the real robot has one
    mass, one centre of mass, and one friction coefficient. Randomizing them
    here would measure a distribution of robots we are not deploying to.
    Sensor noise is the opposite case: it is present on every real tick.
    """
    protocol = PROTOCOLS["ec_latent_rehearsal_v1"]

    assert protocol.randomization_profile == "none"
    assert dict(protocol.randomization_kept) == {
        "startup": False,
        "reset": False,
        "push": False,
    }
    assert protocol.observation_corruption is True


@pytest.mark.parametrize(
    "protocol_id", ["ec_latent_rehearsal_v1", "cross_backend_isaac_v1"]
)
def test_promotion_gate_compares_like_with_like(protocol_id: str) -> None:
    """Both sides of the cross-backend certificate carry the same noise."""
    protocol = PROTOCOLS[protocol_id]

    assert protocol.observation_corruption is True
    assert protocol.observation_noise == SONIC_OBSERVATION_NOISE
