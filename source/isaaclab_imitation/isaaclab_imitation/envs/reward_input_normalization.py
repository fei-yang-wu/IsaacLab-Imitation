"""Shared 0-1 normalization for the ``reward_input`` observation group.

The IPMD reward estimator scores rollout states against expert states, so the
policy-side observation terms (computed from the live robot) and the
expert-side values (served from the data plane's reference cache) must apply
the exact same feature map. Both sides import these helpers; nothing about the
normalization may live in only one of them.

The canonical reward input is three blocks, each mapped into ``[0, 1]``:

- joint positions, normalized per joint by the soft joint position limits;
- the expert anchor position expressed in the robot anchor frame ("relative
  root position"), mapped from ``[-REWARD_INPUT_ANCHOR_POS_RANGE_M, +...]``
  so perfect tracking sits at 0.5;
- the expert anchor orientation in the robot anchor frame as flattened rot6d
  ("relative root orientation"), whose components live in ``[-1, 1]`` and map
  affinely so the identity rotation sits at ``(1, 0.5, 0.5, 0.5, 1, 0.5)``.
"""

from __future__ import annotations

import torch

#: Half-range, in metres, of the anchor-position error mapped onto [0, 1].
#: Errors beyond this saturate; at 1 m the episode is unrecoverable anyway.
REWARD_INPUT_ANCHOR_POS_RANGE_M = 1.0


def normalize_joint_pos_unit(
    joint_pos: torch.Tensor,
    lower_limits: torch.Tensor,
    upper_limits: torch.Tensor,
) -> torch.Tensor:
    """Map joint positions into [0, 1] with per-joint soft limits.

    Reference (mocap) frames may exceed the robot's limits; values clamp.
    """
    span = (upper_limits - lower_limits).clamp_min(1e-6)
    return ((joint_pos - lower_limits) / span).clamp(0.0, 1.0)


def normalize_anchor_pos_unit(anchor_pos_b: torch.Tensor) -> torch.Tensor:
    """Map a relative anchor position (metres) into [0, 1]; zero error -> 0.5."""
    scaled = anchor_pos_b / REWARD_INPUT_ANCHOR_POS_RANGE_M
    return scaled.clamp(-1.0, 1.0) * 0.5 + 0.5


def normalize_rot6d_unit(rot6d: torch.Tensor) -> torch.Tensor:
    """Map flattened rot6d components from [-1, 1] into [0, 1]."""
    return rot6d.clamp(-1.0, 1.0) * 0.5 + 0.5


def identity_rot6d_unit(device: torch.device) -> torch.Tensor:
    """The expert-side 'zero orientation error' value after normalization."""
    identity = torch.zeros(6, device=device)
    identity[0] = 1.0
    identity[4] = 1.0
    return normalize_rot6d_unit(identity)
