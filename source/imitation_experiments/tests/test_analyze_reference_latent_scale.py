from __future__ import annotations

import numpy as np

from imitation_experiments.evaluation.analyze_reference_latent_scale import (
    normalized_action_family,
    root_qpos_expert_windows,
)


def test_normalized_action_family_removes_take_actor_and_mirror() -> None:
    assert normalized_action_family("walk_forward_003_A216_M") == "walk_forward"
    assert normalized_action_family("walk_forward_003_A216") == "walk_forward"


def test_root_qpos_expert_windows_use_first_anchor_frame() -> None:
    joint = np.zeros((1, 10, 29), dtype=np.float32)
    position = np.zeros((1, 10, 3), dtype=np.float32)
    position[0, :, 0] = np.arange(10)
    # Frame zero faces +Y (90 degree yaw), so world +X is local -Y.
    half = np.sqrt(0.5)
    quaternion = np.tile(
        np.asarray([0.0, 0.0, half, half], dtype=np.float32), (1, 10, 1)
    )

    frames = root_qpos_expert_windows(joint, position, quaternion)

    assert frames.shape == (1, 10, 38)
    np.testing.assert_allclose(frames[0, :, 29], 0.0, atol=1.0e-6)
    np.testing.assert_allclose(frames[0, :, 30], -np.arange(10), atol=1.0e-6)
    np.testing.assert_allclose(
        frames[0, :, 32:],
        np.tile([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], (10, 1)),
        atol=1.0e-6,
    )
