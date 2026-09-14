"""Position-only reward at a declared box-center error scale."""

from __future__ import annotations

import math

import torch


def box_center_tracking_reward(actual, target, position_scale_m=0.05):
    """Score current COM tracking without adding an orientation condition."""
    if not math.isfinite(position_scale_m) or position_scale_m <= 0:
        raise ValueError("Box position scale must be finite and positive.")
    return torch.exp(-(actual - target).square().sum(dim=-1) / position_scale_m**2)
