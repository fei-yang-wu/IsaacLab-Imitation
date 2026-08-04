# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generic config-resolution helpers shared by the imitation environments.

Each helper here replaces a hand-maintained table or a per-field special case
with a rule that reads the config tree itself. That is deliberate: a table
listing which terms take an anchor body, or which fields need coercing, is a
second source of truth that silently rots when a term is added. A rule cannot.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from isaaclab_tasks.utils import PresetCfg
from isaaclab_tasks.utils.hydra import resolve_presets

_ANCHOR_PARAM = "anchor_body_name"


def resolve_remaining_presets(cfg: Any) -> None:
    """Collapse any :class:`PresetCfg` still standing in a config tree.

    Isaac Lab resolves presets inside ``register_task``, so a launch through
    the CLI hands the environment a fully concrete tree. A config built
    directly -- a test, an audit script, an evaluation driver that never went
    through Hydra -- does not, and would otherwise reach the managers holding
    preset objects where configs belong.

    Calling this once at the top of resolution makes both paths converge, which
    is why no consumer downstream has to accept "a config, or the preset that
    would have produced it". On an already-resolved tree it is a walk that
    finds nothing.
    """
    resolve_presets(cfg)


def iter_terms(*containers: Any) -> Iterator[Any]:
    """Yield every term config held by the given groups or term containers.

    Walks one level into group objects (an observation config holding groups,
    a reward config holding terms), which is the shape every manager config in
    this package has.
    """
    for container in containers:
        if container is None:
            continue
        for name in dir(container):
            if name.startswith("_"):
                continue
            value = getattr(container, name, None)
            if value is None or callable(value):
                continue
            if hasattr(value, "params"):
                yield value
            elif hasattr(value, "__dataclass_fields__"):
                yield from iter_terms(value)


def stamp_anchor_body(anchor_body_name: str, *containers: Any) -> None:
    """Point every anchor-relative term at one body.

    Membership is read off the terms themselves -- a term takes an anchor body
    exactly when it declares the parameter -- rather than from a per-surface
    list of term names that has to be updated whenever the surface changes.
    """
    for term in iter_terms(*containers):
        params = getattr(term, "params", None)
        if isinstance(params, dict) and _ANCHOR_PARAM in params:
            params[_ANCHOR_PARAM] = anchor_body_name


def coerce_declared_types(cfg: Any, _seen: set[int] | None = None) -> None:
    """Parse CLI values that arrived as strings into the shape their field declares.

    Isaac Lab parses ``env.foo=bar`` with :func:`ast.literal_eval` and falls
    back to the raw string when that fails, so an unquoted sequence override --
    ``env.data.clips=[walk1,walk2]``, the form every job script writes --
    arrives as the literal string ``"[walk1,walk2]"``. Iterating it then yields
    single characters and dies somewhere with no visible connection to the
    override that caused it.

    Doing this once, driven by the field annotations, is why no individual
    field has to learn to accept its own string form, and why a field added
    later is covered without anyone remembering to handle it.
    """
    seen = _seen if _seen is not None else set()
    if id(cfg) in seen or not hasattr(cfg, "__dataclass_fields__"):
        return
    seen.add(id(cfg))
    for name, spec in cfg.__dataclass_fields__.items():
        value = getattr(cfg, name, None)
        if isinstance(value, str):
            coerced = _coerce_string(value, spec.type)
            if coerced is not value:
                setattr(cfg, name, coerced)
        elif hasattr(value, "__dataclass_fields__"):
            coerce_declared_types(value, seen)


def _coerce_string(value: str, annotation: Any) -> Any:
    text = value.strip()
    parts = tuple(part.strip() for part in str(annotation).split("|"))
    is_sequence = any(
        part.startswith(("list", "tuple", "Sequence", "Iterable")) for part in parts
    )
    if is_sequence and text.startswith("[") and text.endswith("]"):
        items = [item.strip().strip("\"'") for item in text[1:-1].split(",")]
        items = [item for item in items if item]
        return tuple(items) if any(p.startswith("tuple") for p in parts) else items
    return value


__all__ = [
    "PresetCfg",
    "coerce_declared_types",
    "iter_terms",
    "resolve_remaining_presets",
    "stamp_anchor_body",
]
