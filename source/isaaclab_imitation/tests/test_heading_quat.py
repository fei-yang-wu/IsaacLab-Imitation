"""Contract for :func:`heading_quat`, the SONIC-matched heading extraction.

The reroot helpers use this to align a reference onto the robot anchor, so it
feeds ``motion_body_pos``, ``motion_body_ori`` and the ``foot_pos_xyz``
termination. Three things are pinned:

1. It is SONIC's ``get_heading_q`` -- zero the horizontal components, then
   renormalize -- transcribed for Isaac Lab's ``(x, y, z, w)`` layout rather
   than SONIC's scalar-first ``(w, x, y, z)``. Getting that index mapping wrong
   yields a *roll* quaternion that runs and looks plausible, which is why the
   layout has a test of its own.
2. It equals the twist of a swing-twist decomposition about world Z.
3. It is continuous at pitch = 90 degrees, where the ZYX Euler yaw this
   replaced is degenerate. That singularity is the reason for the change: a
   pelvis reaches 90 degrees of pitch when it falls forward, and there the old
   form swung the rerooted reference by half a turn.
"""

from __future__ import annotations

import math

import torch

from isaaclab_imitation.tasks.manager_based.imitation.mdp._compiled import heading_quat

_ATOL = 1e-6


def _quat_xyzw(roll: float, pitch: float, yaw: float) -> torch.Tensor:
    """Build an Isaac Lab (x, y, z, w) quaternion from ZYX Euler angles."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return torch.tensor(
        [
            sr * cp * cy - cr * sp * sy,  # x
            cr * sp * cy + sr * cp * sy,  # y
            cr * cp * sy - sr * sp * cy,  # z
            cr * cp * cy + sr * sp * sy,  # w
        ],
        dtype=torch.float64,
    )


def _sonic_get_heading_q_wxyz(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """SONIC's implementation verbatim, on its own scalar-first layout.

    ``gear_sonic/trl/utils/torch_transform.py::get_heading_q``: zero indices 1
    and 2 -- x and y under (w, x, y, z) -- then renormalize.
    """
    out = quat_wxyz.clone()
    out[..., 1] = 0.0
    out[..., 2] = 0.0
    return out / out.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-9)


def _to_wxyz(quat_xyzw: torch.Tensor) -> torch.Tensor:
    return quat_xyzw[..., [3, 0, 1, 2]]


def _angle_about_z(quat_xyzw: torch.Tensor) -> float:
    return 2.0 * math.atan2(float(quat_xyzw[..., 2]), float(quat_xyzw[..., 3]))


_ATTITUDES = [
    (0.0, 0.0, 0.0),
    (0.1, 0.2, 0.3),
    (-0.4, 0.6, -1.2),
    (1.0, -0.5, 2.0),
    (-0.9, 1.2, -2.5),
]


def test_matches_sonic_get_heading_q_across_layouts():
    """Ours on (x,y,z,w) == SONIC's on (w,x,y,z), for the same rotation."""
    for angles in _ATTITUDES:
        quat = _quat_xyzw(*angles)
        ours = heading_quat(quat)
        theirs = _sonic_get_heading_q_wxyz(_to_wxyz(quat))
        assert torch.allclose(_to_wxyz(ours), theirs, atol=_ATOL), angles


def test_zeroes_x_and_y_not_y_and_z():
    """The layout trap: SONIC's literal indices would keep roll, not yaw.

    A pure-roll rotation has no heading, so it must map to identity. Zeroing
    indices 1 and 2 of an (x, y, z, w) quaternion -- a literal transcription of
    SONIC's code -- would keep x and w and return the roll unchanged.
    """
    roll_only = _quat_xyzw(0.7, 0.0, 0.0)
    result = heading_quat(roll_only)
    identity = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)
    assert torch.allclose(result, identity, atol=_ATOL)
    assert not torch.allclose(result, roll_only, atol=1e-3)


def test_equals_swing_twist_twist_about_z():
    for angles in _ATTITUDES:
        quat = _quat_xyzw(*angles)
        w, z = float(quat[3]), float(quat[2])
        norm = math.hypot(w, z)
        expected = 2.0 * math.atan2(z / norm, w / norm)
        assert abs(_angle_about_z(heading_quat(quat)) - expected) < _ATOL, angles


def test_pure_yaw_is_preserved():
    for yaw in (0.0, 0.5, -1.7, 3.0):
        quat = _quat_xyzw(0.0, 0.0, yaw)
        assert abs(_angle_about_z(heading_quat(quat)) - yaw) < _ATOL


def test_continuous_through_the_zyx_yaw_singularity():
    """Pitch sweeps through 90 degrees; heading must not jump.

    ``yaw_quat`` returns 30 -> 90 -> -150 degrees across this sweep. Anything
    that moves by more than a degree here would put a half-turn spin into the
    reward exactly when the robot pitches over.
    """
    headings = [
        math.degrees(
            _angle_about_z(
                heading_quat(_quat_xyzw(0.0, math.radians(p), math.radians(30.0)))
            )
        )
        for p in (85.0, 89.0, 90.0, 91.0, 95.0)
    ]
    for value in headings:
        assert abs(value - 30.0) < 1e-3, headings


def test_stays_finite_at_full_inversion():
    """Upside down must not produce NaN; the clamp is what guarantees it."""
    for angles in [(math.pi, 0.0, 0.5), (0.0, math.pi, 0.5), (math.pi, math.pi, 0.0)]:
        result = heading_quat(_quat_xyzw(*angles))
        assert torch.isfinite(result).all(), angles
        assert abs(float(result.norm()) - 1.0) < 1e-3 or float(result.norm()) < 1e-6


def test_batched_shapes_are_preserved():
    quats = torch.stack([_quat_xyzw(*a) for a in _ATTITUDES]).reshape(-1, 4)
    batched = heading_quat(quats)
    assert batched.shape == quats.shape
    for index in range(quats.shape[0]):
        assert torch.allclose(batched[index], heading_quat(quats[index]), atol=_ATOL)

    nested = quats.reshape(-1, 1, 4).expand(-1, 3, -1).contiguous()
    assert heading_quat(nested).shape == nested.shape
