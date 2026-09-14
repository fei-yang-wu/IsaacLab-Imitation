"""The Newton wrench-frame correction rotates body-frame buffers to world."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

wp = pytest.importorskip("warp")

from isaaclab_imitation import newton_wrench_frame as fix  # noqa: E402


def _composer(quat_xyzw, force, torque, device="cpu"):
    quat = wp.array([[wp.quatf(*quat_xyzw)]], dtype=wp.quatf, device=device)
    return SimpleNamespace(
        num_envs=1,
        num_bodies=1,
        device=device,
        _get_link_quat_fn=lambda: quat,
        _out_force_b=wp.array([[wp.vec3f(*force)]], dtype=wp.vec3f, device=device),
        _out_torque_b=wp.array([[wp.vec3f(*torque)]], dtype=wp.vec3f, device=device),
    )


def test_rotation_maps_body_x_to_world_y_for_a_90_degree_yaw() -> None:
    wp.init()
    half = math.sqrt(0.5)
    composer = _composer((0.0, 0.0, half, half), (1.0, 0.0, 0.0), (0.0, 2.0, 0.0))
    fix.rotate_composer_output_to_world(composer)
    force = composer._out_force_b.numpy()[0, 0]
    torque = composer._out_torque_b.numpy()[0, 0]
    assert force == pytest.approx((0.0, 1.0, 0.0), abs=1e-6)
    assert torque == pytest.approx((-2.0, 0.0, 0.0), abs=1e-6)


def test_identity_orientation_leaves_the_wrench_unchanged() -> None:
    wp.init()
    composer = _composer((0.0, 0.0, 0.0, 1.0), (3.0, -1.0, 0.5), (0.1, 0.2, 0.3))
    fix.rotate_composer_output_to_world(composer)
    assert composer._out_force_b.numpy()[0, 0] == pytest.approx((3.0, -1.0, 0.5))
    assert composer._out_torque_b.numpy()[0, 0] == pytest.approx((0.1, 0.2, 0.3))


def test_only_newton_assets_are_recognised() -> None:
    class Fake:  # module is this test file, not isaaclab_newton
        pass

    assert not fix.is_newton_asset(Fake())
    Fake.__module__ = "isaaclab_newton.assets.rigid_object.rigid_object"
    assert fix.is_newton_asset(Fake())


def test_install_is_gated_on_the_affected_release(monkeypatch) -> None:
    monkeypatch.setattr(fix, "_installed", False)
    monkeypatch.setattr(fix, "isaaclab_version", lambda: "9.9.9")
    assert fix.install() is False
    monkeypatch.setenv(fix.ENV_VAR, "0")
    monkeypatch.setattr(fix, "isaaclab_version", lambda: fix.AFFECTED_VERSIONS[0])
    assert fix.install() is False
