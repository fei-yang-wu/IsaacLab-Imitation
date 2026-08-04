"""Live oracle action: the training-target action derived from the reference.

Replaces the deleted reconstructed-reference-action subsystem. The oracle
action is what the low-level tracker would output to follow the reference
perfectly: the reference's next-frame joint targets, inverse-processed
through the live action term's affine transform (offset/scale/clip) back
into the raw action space. It is a training-target diagnostic only -- never
a deployable policy result -- and it uses only live env surfaces
(``current_expert_frame``, ``action_manager``, ``reference_joint_names``),
so it works on both the legacy env and ``ImitationRLEnvV2``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv


def live_oracle_action(
    env: ManagerBasedRLEnv,
    *,
    env_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the raw action that tracks the reference's next frame exactly.

    Args:
        env: The unwrapped imitation env (legacy or v2).
        env_ids: Optional environment selection; None uses all envs.

    Raises:
        RuntimeError: When the live reference frame has no aligned next joint
            targets (the data lacks ``next_qpos``), mirroring the deleted
            reconstruction path's loud failure.
    """
    action_term = env.action_manager.get_term("joint_pos")
    reference = env.current_expert_frame
    if reference is None:
        raise RuntimeError("Oracle action requested before expert data is ready.")

    next_joint_pos = reference.get(("next", "joint_pos"))
    if next_joint_pos is None:
        raise RuntimeError(
            "Live oracle action requires reference data with aligned next "
            "joint targets (`next_qpos`); the loaded dataset does not provide "
            "them."
        )

    # Map the reference joint order onto the action term's pinned order.
    reference_names = list(env.reference_joint_names)
    action_names = list(action_term._joint_names)
    index = torch.as_tensor(
        [reference_names.index(name) for name in action_names],
        dtype=torch.long,
        device=env.device,
    )
    processed = next_joint_pos.index_select(-1, index).to(
        device=env.device, dtype=torch.float32
    )

    # Clamp in processed space (the action term's clip bounds are defined
    # there), then invert the affine transform: raw = (processed - offset) /
    # scale, with zeros where the scale is degenerate.
    clip = getattr(getattr(action_term, "cfg", None), "clip", None)
    if clip is not None:
        low = torch.as_tensor(clip[0], device=env.device, dtype=torch.float32)
        high = torch.as_tensor(clip[1], device=env.device, dtype=torch.float32)
        processed = torch.clamp(processed, min=low, max=high)

    offset = action_term._offset
    scale = action_term._scale
    if isinstance(offset, torch.Tensor):
        offset = offset.to(device=env.device, dtype=torch.float32)
    else:
        offset = torch.full_like(processed, float(offset))
    if isinstance(scale, torch.Tensor):
        scale = scale.to(device=env.device, dtype=torch.float32)
    else:
        scale = torch.full_like(processed, float(scale))
    safe_scale = torch.where(scale.abs() > 1.0e-8, scale, torch.ones_like(scale))
    raw = torch.where(
        scale.abs() > 1.0e-8,
        (processed - offset) / safe_scale,
        torch.zeros_like(processed),
    )

    if env_ids is None:
        return raw
    env_ids_t = torch.as_tensor(
        env_ids, device=env.device, dtype=torch.long
    ).reshape(-1)
    return raw.index_select(0, env_ids_t)
