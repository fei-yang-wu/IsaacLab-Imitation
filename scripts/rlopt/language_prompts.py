"""Prompt templates for language-conditioned G1 imitation goals.

The SkillCommander runtime consumes precomputed vectors, not text. This module
keeps the text-generation side deterministic and offline so embedding tables can
be rebuilt and audited reproducibly.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROMPT_TIERS = (
    "raw_name",
    "category",
    "robot_instruction",
    "kinematic_description",
    "event_level",
    "attribute_text",
)


LAFAN1_CATEGORY_PROMPTS: dict[str, dict[str, str]] = {
    "dance": {
        "robot_instruction": "A humanoid robot performs a dance motion.",
        "kinematic_description": (
            "A humanoid performs a rhythmic whole-body dance with alternating "
            "foot contacts, expressive arm motion, torso turns, and controlled "
            "weight shifts."
        ),
        "event_level": (
            "The humanoid shifts weight between the feet, steps rhythmically, "
            "moves the arms expressively, turns the torso, and maintains an "
            "upright dance posture."
        ),
        "attribute_text": (
            "action: dance; locomotion: rhythmic stepping; speed: variable; "
            "contact: alternating feet; arms: expressive; torso: turning; "
            "energy: medium-high; balance: upright."
        ),
    },
    "fall and get up": {
        "robot_instruction": ("A humanoid robot falls to the ground and gets back up."),
        "kinematic_description": (
            "A humanoid loses balance, lowers the body to the ground, makes "
            "body contact, then pushes through the limbs to return to standing."
        ),
        "event_level": (
            "The humanoid starts upright, collapses or descends to the ground, "
            "stabilizes on the floor, plants the limbs, pushes the body upward, "
            "and finishes standing."
        ),
        "attribute_text": (
            "action: fall and get up; locomotion: recovery; speed: variable; "
            "contact: ground body contact; posture: upright to ground to "
            "upright; arms: support; energy: high; balance: lost then recovered."
        ),
    },
    "fight": {
        "robot_instruction": "A humanoid robot performs a fighting motion.",
        "kinematic_description": (
            "A humanoid makes aggressive whole-body fighting gestures with "
            "quick arm strikes, defensive posture changes, and short stance "
            "adjustments."
        ),
        "event_level": (
            "The humanoid sets a fighting stance, shifts weight, throws arm "
            "strikes or blocks, adjusts the feet, and keeps the torso guarded."
        ),
        "attribute_text": (
            "action: fight; locomotion: short stance changes; speed: quick; "
            "contact: feet planted or stepping; arms: striking and guarding; "
            "torso: guarded; energy: high; balance: braced."
        ),
    },
    "fight and sports": {
        "robot_instruction": (
            "A humanoid robot performs a sport-like fighting motion."
        ),
        "kinematic_description": (
            "A humanoid combines fighting gestures with athletic sports "
            "movement, using quick arm actions, stance changes, and dynamic "
            "whole-body balance."
        ),
        "event_level": (
            "The humanoid enters an athletic stance, shifts weight rapidly, "
            "performs sport-like strikes or gestures, steps to reposition, and "
            "recovers balance."
        ),
        "attribute_text": (
            "action: fight and sports; locomotion: athletic repositioning; "
            "speed: quick; contact: alternating stance; arms: sport-like "
            "strikes; torso: dynamic; energy: high; balance: reactive."
        ),
    },
    "jumps": {
        "robot_instruction": "A humanoid robot performs jumping motions.",
        "kinematic_description": (
            "A humanoid bends the legs, pushes off the ground, enters an aerial "
            "phase, lands through the feet, and absorbs impact with the knees "
            "and hips."
        ),
        "event_level": (
            "The humanoid crouches, drives upward through the legs, leaves the "
            "ground, lands on the feet, and stabilizes after impact."
        ),
        "attribute_text": (
            "action: jumps; locomotion: vertical impulse; speed: explosive; "
            "contact: takeoff and landing; arms: balance aid; legs: crouch and "
            "extend; energy: high; balance: landing recovery."
        ),
    },
    "run": {
        "robot_instruction": "A humanoid robot runs forward.",
        "kinematic_description": (
            "A humanoid uses a forward running gait with alternating foot "
            "contacts, brief flight phases, arm swing, and moderate-to-fast "
            "center-of-mass motion."
        ),
        "event_level": (
            "The humanoid cycles left and right foot contacts, swings the arms "
            "opposite the legs, moves forward quickly, and repeats the running "
            "gait."
        ),
        "attribute_text": (
            "action: run; locomotion: forward gait; speed: fast; contact: "
            "alternating feet with flight; arms: reciprocal swing; torso: "
            "forward; energy: high; balance: dynamic."
        ),
    },
    "sprint": {
        "robot_instruction": "A humanoid robot sprints forward at high speed.",
        "kinematic_description": (
            "A humanoid uses a very fast forward running gait with powerful leg "
            "drive, pronounced arm swing, short contact time, and strong "
            "dynamic balance."
        ),
        "event_level": (
            "The humanoid leans forward, drives one leg after the other, swings "
            "the arms strongly, makes brief foot contacts, and accelerates in a "
            "sprinting gait."
        ),
        "attribute_text": (
            "action: sprint; locomotion: forward gait; speed: very fast; "
            "contact: brief alternating feet with flight; arms: strong swing; "
            "torso: forward lean; energy: very high; balance: dynamic."
        ),
    },
    "walk": {
        "robot_instruction": "A humanoid robot walks forward.",
        "kinematic_description": (
            "A humanoid uses a controlled forward walking gait with alternating "
            "foot contacts, no flight phase, mild arm swing, and steady upright "
            "balance."
        ),
        "event_level": (
            "The humanoid places one foot, transfers weight, swings the other "
            "leg forward, places the next foot, and repeats a stable walking "
            "cycle."
        ),
        "attribute_text": (
            "action: walk; locomotion: forward gait; speed: slow-medium; "
            "contact: alternating feet without flight; arms: mild swing; "
            "torso: upright; energy: low-medium; balance: steady."
        ),
    },
}


def normalize_prompt_tier(tier: str) -> str:
    """Validate and normalize a prompt tier name."""
    normalized = str(tier).strip().lower()
    if normalized not in PROMPT_TIERS:
        choices = ", ".join(PROMPT_TIERS)
        raise ValueError(f"Unknown prompt tier {tier!r}. Expected one of: {choices}.")
    return normalized


def humanize_motion_name(name: str) -> str:
    """Convert a manifest motion name into a coarse natural-language category."""
    base = re.sub(r"_subject\d+$", "", str(name))
    base = re.sub(r"\d+$", "", base)
    base = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", base)
    base = base.replace("_", " ").replace("-", " ")
    base = re.sub(r"\s+", " ", base).strip().lower()
    return base or str(name).strip().lower()


def _fallback_prompt(category: str, tier: str) -> str:
    if tier == "robot_instruction":
        return f"A humanoid robot performs a {category} motion."
    if tier == "kinematic_description":
        return (
            f"A humanoid performs a {category} motion with coordinated "
            "whole-body balance, foot contacts, arm motion, and torso control."
        )
    if tier == "event_level":
        return (
            f"The humanoid begins a {category} motion, coordinates the limbs "
            "and torso through the action, and returns to a balanced state."
        )
    if tier == "attribute_text":
        return (
            f"action: {category}; body: humanoid; motion: full-body; "
            "balance: controlled; contact: task-dependent; energy: variable."
        )
    raise ValueError(f"Unsupported fallback prompt tier: {tier!r}.")


def prompt_for_motion(name: str, tier: str) -> str:
    """Return the deterministic built-in prompt for a motion and prompt tier."""
    tier = normalize_prompt_tier(tier)
    if tier == "raw_name":
        return str(name)
    category = humanize_motion_name(name)
    if tier == "category":
        return category
    return LAFAN1_CATEGORY_PROMPTS.get(category, {}).get(
        tier, _fallback_prompt(category, tier)
    )


def _coerce_prompt_value(value: Any, tier: str) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, Mapping):
        language = value.get("language")
        if isinstance(language, Mapping):
            prompt = _coerce_prompt_value(language, tier)
            if prompt is not None:
                return prompt
        for key in (tier, "prompt", "text"):
            nested = value.get(key)
            if isinstance(nested, str) and nested.strip():
                return nested.strip()
    return None


def _extract_named_entries(data: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    entries: Any = data.get("dataset", {}).get("trajectories", {}).get("lafan1_csv")
    if entries is None:
        entries = data.get("lafan1_csv", data.get("motions"))
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, Mapping)]


def _name_for_entry(entry: Mapping[str, Any]) -> str | None:
    name = entry.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    path_value = entry.get("path") or entry.get("file")
    if isinstance(path_value, str) and path_value.strip():
        return Path(path_value).stem
    return None


def _extract_manifest_prompt_overrides(
    data: Mapping[str, Any],
    tier: str,
) -> dict[str, str]:
    entries = _extract_named_entries(data)
    if not entries:
        return {}
    overrides: dict[str, str] = {}
    for entry in entries:
        name = _name_for_entry(entry)
        if name is None:
            continue
        prompt = _coerce_prompt_value(entry, tier)
        if prompt is not None:
            overrides[name] = prompt
    return overrides


def load_prompt_overrides(path: str | Path | None, tier: str) -> dict[str, str]:
    """Load optional prompt overrides keyed by raw motion name or category.

    Supported JSON shapes:

    * ``{"dance1_subject1": "custom prompt"}``
    * ``{"dance": "custom prompt"}``
    * ``{"prompts": {"dance": "custom prompt"}}``
    * ``{"kinematic_description": {"dance": "custom prompt"}}``
    * ``{"dance": {"prompt": "custom prompt"}}``
    * A manifest whose entries contain ``{"language": {"attribute_text": ...}}``.
    """
    if path is None or not str(path).strip():
        return {}
    tier = normalize_prompt_tier(tier)
    prompt_path = Path(path).expanduser().resolve()
    data = json.loads(prompt_path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"Prompt override JSON must be an object: {prompt_path}")

    if _extract_named_entries(data):
        return _extract_manifest_prompt_overrides(data, tier)

    raw_mapping: Any
    if isinstance(data.get("prompts"), Mapping):
        raw_mapping = data["prompts"]
    elif isinstance(data.get(tier), Mapping):
        raw_mapping = data[tier]
    else:
        raw_mapping = data
    if not isinstance(raw_mapping, Mapping):
        raise ValueError(f"Prompt override mapping is invalid: {prompt_path}")

    overrides: dict[str, str] = {}
    for key, value in raw_mapping.items():
        prompt = _coerce_prompt_value(value, tier)
        if prompt is not None:
            overrides[str(key)] = prompt
    return overrides


def resolve_prompt_for_motion(
    name: str,
    tier: str,
    overrides: Mapping[str, str] | None = None,
) -> str:
    """Resolve a prompt, applying raw-name overrides before category overrides."""
    if overrides:
        category = humanize_motion_name(name)
        for key in (str(name), category):
            override = overrides.get(key)
            if isinstance(override, str) and override.strip():
                return override.strip()
    return prompt_for_motion(name, tier)
