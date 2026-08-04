"""Shared schema and tensor builders for causal high-level planner inputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


CAUSAL_PLANNER_OBSERVATION_VERSION = 1
CAUSAL_PLANNER_FEATURE_WIDTHS: tuple[tuple[str, int], ...] = (
    ("joint_pos_rel", 29),
    ("joint_vel_rel", 29),
    ("base_ang_vel", 3),
    ("projected_gravity", 3),
    ("last_action", 29),
)
CAUSAL_PLANNER_FRAME_DIM = sum(width for _, width in CAUSAL_PLANNER_FEATURE_WIDTHS)


class CausalPlannerHistory:
    """Oldest-to-newest fixed history with repeat-first reset padding."""

    def __init__(self, initial_frame: torch.Tensor, *, history_steps: int) -> None:
        history_steps = int(history_steps)
        if history_steps < 0:
            raise ValueError("history_steps must be >= 0.")
        if initial_frame.ndim != 2:
            raise ValueError(
                f"initial_frame must be rank 2, got {tuple(initial_frame.shape)}."
            )
        if int(initial_frame.shape[-1]) != CAUSAL_PLANNER_FRAME_DIM:
            raise ValueError(
                f"initial_frame must have width {CAUSAL_PLANNER_FRAME_DIM}."
            )
        self.history_steps = history_steps
        self._history = initial_frame.unsqueeze(1).repeat(1, history_steps + 1, 1)

    @property
    def tensor(self) -> torch.Tensor:
        return self._history

    def append(self, frame: torch.Tensor) -> None:
        if tuple(frame.shape) != (
            int(self._history.shape[0]),
            CAUSAL_PLANNER_FRAME_DIM,
        ):
            raise ValueError(
                "Appended causal planner frame shape mismatch: "
                f"got {tuple(frame.shape)}."
            )
        self._history[:, :-1].copy_(self._history[:, 1:].clone())
        self._history[:, -1].copy_(frame)

    def reset(self, env_ids: torch.Tensor, frame: torch.Tensor) -> None:
        env_ids = torch.as_tensor(
            env_ids, device=self._history.device, dtype=torch.long
        ).reshape(-1)
        if tuple(frame.shape) != (int(env_ids.numel()), CAUSAL_PLANNER_FRAME_DIM):
            raise ValueError(
                "Reset causal planner frame shape mismatch: "
                f"got {tuple(frame.shape)} for {int(env_ids.numel())} rows."
            )
        padded = frame.unsqueeze(1).expand(-1, self.history_steps + 1, -1)
        self._history.index_copy_(0, env_ids, padded)

    def select(self, env_ids: torch.Tensor, *, history_steps: int) -> torch.Tensor:
        history_steps = int(history_steps)
        if history_steps < 0 or history_steps > self.history_steps:
            raise ValueError(
                f"history_steps must be in [0, {self.history_steps}], got {history_steps}."
            )
        env_ids = torch.as_tensor(
            env_ids, device=self._history.device, dtype=torch.long
        ).reshape(-1)
        return self._history.index_select(0, env_ids)[
            :, -(history_steps + 1) :
        ].contiguous()


def causal_planner_observation_spec(*, history_steps: int) -> dict[str, Any]:
    """Return the JSON-serializable contract for one planner observation."""
    history_steps = int(history_steps)
    if history_steps < 0:
        raise ValueError("history_steps must be >= 0.")
    history_frames = history_steps + 1
    return {
        "name": "g1_causal_robot_history",
        "version": CAUSAL_PLANNER_OBSERVATION_VERSION,
        "feature_names": [name for name, _ in CAUSAL_PLANNER_FEATURE_WIDTHS],
        "feature_widths": [width for _, width in CAUSAL_PLANNER_FEATURE_WIDTHS],
        "frame_dim": CAUSAL_PLANNER_FRAME_DIM,
        "history_steps": history_steps,
        "history_frames": history_frames,
        "flat_dim": history_frames * CAUSAL_PLANNER_FRAME_DIM,
        "history_order": "oldest_to_newest",
        "reset_padding": "repeat_initial_observation",
        "reference_features": [],
    }


def build_causal_planner_frame(
    features: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Validate and concatenate one batch of robot-observable planner features."""
    missing = [
        name for name, _ in CAUSAL_PLANNER_FEATURE_WIDTHS if name not in features
    ]
    if missing:
        raise KeyError(f"Causal planner features are missing: {missing}.")

    tensors: list[torch.Tensor] = []
    batch_shape: tuple[int, ...] | None = None
    for name, width in CAUSAL_PLANNER_FEATURE_WIDTHS:
        value = features[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Causal planner feature {name!r} must be a tensor.")
        if value.ndim < 2:
            raise ValueError(
                f"Causal planner feature {name!r} must be rank >= 2, "
                f"got {tuple(value.shape)}."
            )
        if int(value.shape[-1]) != width:
            raise ValueError(
                f"Causal planner feature {name!r} must have width {width}, "
                f"got {int(value.shape[-1])}."
            )
        current_batch_shape = tuple(int(dim) for dim in value.shape[:-1])
        if batch_shape is None:
            batch_shape = current_batch_shape
        elif current_batch_shape != batch_shape:
            raise ValueError(
                "Causal planner features must have matching leading shapes: "
                f"expected {batch_shape}, got {current_batch_shape} for {name!r}."
            )
        tensors.append(value)

    frame = torch.cat(tensors, dim=-1).to(dtype=torch.float32)
    if int(frame.shape[-1]) != CAUSAL_PLANNER_FRAME_DIM:
        raise RuntimeError(
            "Causal planner frame width changed unexpectedly: "
            f"expected {CAUSAL_PLANNER_FRAME_DIM}, got {int(frame.shape[-1])}."
        )
    return frame.contiguous()
