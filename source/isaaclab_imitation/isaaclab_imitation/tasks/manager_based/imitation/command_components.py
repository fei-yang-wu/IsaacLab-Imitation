# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The command component vocabulary shared by the terms and the interface.

A *component* is one atomic piece of a tracking command (the joint state, the
root pose, an end-effector pose, a keypoint set). Every explicit command space
in this project is a selection of components, and every component has exactly
one observation term name -- the stable contract every trained checkpoint's
input keys are expressed in.

This module is a leaf on purpose: the command terms
(``mdp/commands/reference.py``, ``mdp/commands/actor.py``) and the declared
interface (``command_interface.py``) all import it, so it must import nothing
from either.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import MISSING

# Atomic command components are canonicalized in this order. A config selects a
# set, not a concatenation order, so spelling the same ablation in a different
# YAML order cannot silently change a checkpoint's actor contract.
COMMAND_COMPONENT_ORDER: tuple[str, ...] = (
    "joint_qpos_qvel",
    "joint_qpos",
    "keypoint_pos",
    "keypoint_ori",
    "ee_pos",
    "ee_ori",
    "root_pos",
    "root_ori",
)

# Component -> observation term name.
COMMAND_COMPONENT_TERM_NAMES: dict[str, str] = {
    "joint_qpos_qvel": "expert_motion",
    "joint_qpos": "expert_motion_qpos",
    "keypoint_pos": "expert_keypoint_pos_b",
    "keypoint_ori": "expert_keypoint_ori_b",
    "ee_pos": "expert_ee_pos_b",
    "ee_ori": "expert_ee_ori_b",
    "root_pos": "expert_anchor_pos_b",
    "root_ori": "expert_anchor_ori_b",
}

COMMAND_TERM_NAME_COMPONENTS: dict[str, str] = {
    term_name: component
    for component, term_name in COMMAND_COMPONENT_TERM_NAMES.items()
}

COMMAND_COMPONENT_ALIASES: dict[str, str] = {
    **{name: name for name in COMMAND_COMPONENT_ORDER},
    "qpos_qvel": "joint_qpos_qvel",
    "full_joint_state": "joint_qpos_qvel",
    "qpos": "joint_qpos",
    "keypoint_position": "keypoint_pos",
    "keypoint_orientation": "keypoint_ori",
    "ee_position": "ee_pos",
    "ee_orientation": "ee_ori",
    "root_position": "root_pos",
    "root_orientation": "root_ori",
}

# Named component sets: labels for common selections, not a second mechanism.
COMMAND_SPACE_COMPONENTS: dict[str, tuple[str, ...]] = {
    "full_body": ("joint_qpos_qvel", "root_pos", "root_ori"),
    "root_qpos": ("joint_qpos", "root_pos", "root_ori"),
    "root_points5": ("keypoint_pos", "root_pos", "root_ori"),
    "root_points5_pose": ("keypoint_pos", "keypoint_ori", "root_pos", "root_ori"),
    "ee": ("ee_pos", "ee_ori"),
}

# The full-body trio: the 67-D reference command, the historical latent-recipe
# encoder input, and the default critic command view.
FULL_BODY_COMPONENTS: tuple[str, ...] = COMMAND_SPACE_COMPONENTS["full_body"]

# The agent-published latent command term (z + phase). Not a component: it has
# no reference-side counterpart and is never mixed with explicit components.
LATENT_COMMAND_TERM_NAME = "latent_command"

# Components built from a configured body set, and the reference-channel field
# each one needs. Selecting one without its bodies is a config error.
COMPONENT_BODY_SET_FIELDS: dict[str, str] = {
    "ee_pos": "ee_body_names",
    "ee_ori": "ee_body_names",
    "keypoint_pos": "keypoint_body_names",
    "keypoint_ori": "keypoint_body_names",
}

_MISSING_TYPE = type(MISSING)


def is_missing(value: object) -> bool:
    """Whether a required config field is still unset.

    ``configclass`` deep-copies field defaults, so a required field holds a
    fresh ``_MISSING_TYPE`` instance rather than the ``MISSING`` singleton and
    an ``is MISSING`` check silently reports "provided".
    """
    return isinstance(value, _MISSING_TYPE)


def normalize_command_components(
    command_components: Iterable[str] | str,
) -> tuple[str, ...]:
    """Validate and canonically order a command component set.

    Accepts a ``"[a,b,c]"`` / ``"a,b,c"`` string form as well: Isaac Lab's strict
    config updater passes a Hydra CLI override for a ``None``-default field
    through as the raw string.
    """
    if isinstance(command_components, str):
        command_components = [
            part
            for part in command_components.strip().strip("[]").split(",")
            if part.strip()
        ]
    normalized: list[str] = []
    for raw_name in command_components:
        name = str(raw_name).strip().lower().replace("-", "_")
        try:
            normalized.append(COMMAND_COMPONENT_ALIASES[name])
        except KeyError as err:
            raise ValueError(
                f"Unsupported command component {raw_name!r}. Expected a subset "
                f"of {list(COMMAND_COMPONENT_ORDER)}."
            ) from err
    if not normalized:
        raise ValueError("command components must select at least one component.")
    duplicates = sorted({name for name in normalized if normalized.count(name) > 1})
    if duplicates:
        raise ValueError(f"command components contain duplicates: {duplicates}.")
    if {"joint_qpos_qvel", "joint_qpos"}.issubset(normalized):
        raise ValueError(
            "joint_qpos_qvel and joint_qpos are mutually exclusive command components."
        )
    selected = set(normalized)
    return tuple(name for name in COMMAND_COMPONENT_ORDER if name in selected)


def command_space_components(command_space: str) -> tuple[str, ...]:
    """Component tuple of a named command space (a label, not a mechanism)."""
    name = str(command_space).strip().lower().replace("-", "_")
    try:
        return COMMAND_SPACE_COMPONENTS[name]
    except KeyError as err:
        raise ValueError(
            f"Unsupported command space {command_space!r}. Expected one of "
            f"{sorted(COMMAND_SPACE_COMPONENTS)}."
        ) from err


def component_term_names(components: Sequence[str]) -> tuple[str, ...]:
    """Observation term names of a canonically ordered component set."""
    return tuple(COMMAND_COMPONENT_TERM_NAMES[name] for name in components)


__all__ = [
    "COMMAND_COMPONENT_ALIASES",
    "COMMAND_COMPONENT_ORDER",
    "COMMAND_COMPONENT_TERM_NAMES",
    "COMMAND_SPACE_COMPONENTS",
    "COMMAND_TERM_NAME_COMPONENTS",
    "COMPONENT_BODY_SET_FIELDS",
    "FULL_BODY_COMPONENTS",
    "LATENT_COMMAND_TERM_NAME",
    "command_space_components",
    "component_term_names",
    "is_missing",
    "normalize_command_components",
]
