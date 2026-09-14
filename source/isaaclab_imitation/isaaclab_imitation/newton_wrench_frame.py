"""Correct the frame of external wrenches on the Newton backend.

Isaac Lab composes every external wrench into the **body** frame
(``WrenchComposer.compose_to_body_frame`` writes ``out_force_b`` and
``out_torque_b``). PhysX consumes that body-frame wrench correctly. The Newton
assets in Isaac Lab ``3.0.0b2.post1`` copy the same two buffers straight into
Newton's ``state.body_f``, which Newton documents as an external wrench **in
world frame** at the body's center of mass. Nothing rotates in between, so on
Newton every external wrench lands rotated by the inverse of the body's
orientation: a world-frame force passed with ``is_global=True`` comes out
rotated, and a body-frame force passed with ``is_global=False`` comes out
as if it were already world-frame.

Measured on the Sharpa task with a contact-free reset: a world +Z force equal
to the object's weight, passed as global, leaves the object falling at
``-0.488 m/s`` while pushing it sideways at ``0.49 m/s``; the same vector
passed as *local* holds it at exactly zero. The task's floating-hand and
virtual-object controllers both compute body-frame wrenches and pass them as
local, which is why under Newton the hands drift off the Reference and the
object leaves at 1.9 m/s with nothing touching it.

This module rotates the composer's two output buffers from body frame to
world frame after composition, for composers attached to Newton assets only.
The Newton write path then hands a genuinely world-frame wrench to
``body_f``. It is gated on the exact Isaac Lab version that has the defect, so
a fixed release is not double-rotated, and it can be switched off with
``ISAACLAB_IMITATION_NEWTON_WRENCH_FIX=0``.
"""

from __future__ import annotations

import logging
import os
from typing import Any

AFFECTED_VERSIONS = ("3.0.0b2.post1",)
"""Isaac Lab releases known to write body-frame wrenches into Newton's body_f."""

ENV_VAR = "ISAACLAB_IMITATION_NEWTON_WRENCH_FIX"

_log = logging.getLogger(__name__)
_installed = False
_kernel: Any = None


def _rotate_kernel():
    """Return the warp kernel, compiled once."""

    global _kernel
    if _kernel is None:
        import warp as wp

        @wp.kernel
        def rotate_body_to_world(
            link_quat_w: wp.array2d(dtype=wp.quatf),
            force: wp.array2d(dtype=wp.vec3f),
            torque: wp.array2d(dtype=wp.vec3f),
        ):
            env, body = wp.tid()
            quat = link_quat_w[env, body]
            force[env, body] = wp.quat_rotate(quat, force[env, body])
            torque[env, body] = wp.quat_rotate(quat, torque[env, body])

        _kernel = rotate_body_to_world
    return _kernel


def is_newton_asset(asset: Any) -> bool:
    """True when ``asset`` is one of Isaac Lab's Newton-backed assets."""

    return type(asset).__module__.startswith("isaaclab_newton")


def rotate_composer_output_to_world(composer: Any) -> None:
    """Rotate ``composer`` outputs from the body frame to the world frame, in place."""

    import warp as wp

    link_quat_w = composer._get_link_quat_fn()
    wp.launch(
        _rotate_kernel(),
        dim=(composer.num_envs, composer.num_bodies),
        inputs=[link_quat_w, composer._out_force_b, composer._out_torque_b],
        device=composer.device,
    )


def isaaclab_version() -> str:
    import importlib.metadata as metadata

    try:
        return metadata.version("isaaclab")
    except metadata.PackageNotFoundError:  # pragma: no cover - not an Isaac env
        return ""


def is_enabled() -> bool:
    return os.environ.get(ENV_VAR, "1") not in ("0", "false", "False", "no")


def install() -> bool:
    """Install the correction once. Returns True when it is active.

    Safe to call repeatedly and safe to call outside an Isaac environment,
    where it does nothing.
    """

    global _installed
    if _installed:
        return True
    if not is_enabled():
        _log.info("%s=0: Newton external wrench frame correction disabled.", ENV_VAR)
        return False
    version = isaaclab_version()
    if version not in AFFECTED_VERSIONS:
        _log.info(
            "Isaac Lab %s is not a release known to misframe Newton wrenches; "
            "leaving it alone.",
            version or "(not installed)",
        )
        return False
    try:
        from isaaclab.utils.wrench_composer import WrenchComposer
    except ImportError:  # pragma: no cover - not an Isaac env
        return False

    original = WrenchComposer.compose_to_body_frame

    def compose_to_body_frame(self: Any) -> None:
        original(self)
        if is_newton_asset(self._asset):
            rotate_composer_output_to_world(self)

    compose_to_body_frame.__wrapped_by_isaaclab_imitation__ = True  # type: ignore[attr-defined]
    WrenchComposer.compose_to_body_frame = compose_to_body_frame  # type: ignore[method-assign]
    _installed = True
    print(
        "[INFO] Newton external wrench frame correction installed "
        f"(Isaac Lab {version}); set {ENV_VAR}=0 to disable."
    )
    return True


def is_installed() -> bool:
    return _installed


__all__ = [
    "AFFECTED_VERSIONS",
    "ENV_VAR",
    "install",
    "is_enabled",
    "is_installed",
    "is_newton_asset",
    "rotate_composer_output_to_world",
]
