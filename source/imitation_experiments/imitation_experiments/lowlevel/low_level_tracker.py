"""Load a low-level tracker policy without restoring training-only state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, TypeAlias

import torch
from torch import nn


NestedKey: TypeAlias = str | tuple[str, ...]


@dataclass(frozen=True)
class FrozenLowLevelTracker:
    """A frozen collector policy and its checkpoint provenance."""

    policy: nn.Module
    provenance: dict[str, Any]


def _canonical_key(key: object) -> NestedKey:
    if isinstance(key, str):
        return key
    if isinstance(key, (tuple, list)) and all(
        isinstance(component, str) for component in key
    ):
        return tuple(key)
    raise TypeError(
        f"Policy input keys must be strings or sequences of strings, got {key!r}."
    )


def _policy_input_keys(agent: object) -> tuple[NestedKey, ...]:
    keys = getattr(agent, "_policy_obs_keys", None)
    if keys is None:
        config = getattr(agent, "config", None)
        policy_config = getattr(config, "policy", None)
        get_input_keys = getattr(policy_config, "get_input_keys", None)
        if callable(get_input_keys):
            keys = get_input_keys()
    if keys is None or isinstance(keys, (str, bytes)):
        raise TypeError(
            "The low-level agent must expose policy input keys through "
            "_policy_obs_keys or config.policy.get_input_keys()."
        )
    return tuple(_canonical_key(key) for key in keys)


def _json_key(key: NestedKey) -> str | list[str]:
    return list(key) if isinstance(key, tuple) else key


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_low_level_tracker(
    agent: object,
    checkpoint_path: str | Path,
    *,
    expected_input_keys: Sequence[NestedKey],
    map_location: str | torch.device = "cpu",
) -> FrozenLowLevelTracker:
    """Strict-load only ``policy_state_dict`` and return a frozen policy.

    This deliberately does not call ``agent.load_model``. Value functions,
    optimizers, reward models, and other training state may use a different
    configuration from the inference-only tracker and must not affect it.

    Checkpoints containing observation-normalization state are rejected: exact
    inference would require rebuilding and restoring that transform rather than
    silently running the policy on unnormalized observations.
    """

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Low-level tracker checkpoint not found: {path}")

    expected_keys = tuple(_canonical_key(key) for key in expected_input_keys)
    if not expected_keys:
        raise ValueError("expected_input_keys must contain at least one policy key.")
    actual_keys = _policy_input_keys(agent)
    if actual_keys != expected_keys:
        raise ValueError(
            "Low-level tracker policy input keys do not match the expected "
            f"ordered contract: actual={actual_keys!r}, expected={expected_keys!r}."
        )

    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            "Low-level tracker checkpoint must contain a mapping, "
            f"got {type(checkpoint).__name__}."
        )
    if "vec_norm_msg" in checkpoint:
        raise ValueError(
            "Low-level tracker checkpoint contains vec_norm_msg. Exact policy-only "
            "inference cannot ignore observation-normalization state."
        )
    policy_state = checkpoint.get("policy_state_dict")
    if not isinstance(policy_state, Mapping):
        raise KeyError(
            "Low-level tracker checkpoint is missing mapping 'policy_state_dict'."
        )

    policy_module = getattr(agent, "policy", None)
    if not isinstance(policy_module, nn.Module):
        raise TypeError("The low-level agent does not expose an nn.Module policy.")
    incompatible = policy_module.load_state_dict(policy_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Strict low-level policy restore reported incompatible keys: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}."
        )

    collector_policy = getattr(agent, "collector_policy", None)
    if not isinstance(collector_policy, nn.Module):
        raise TypeError(
            "The low-level agent collector_policy must be an nn.Module after loading."
        )
    policy_module.requires_grad_(False)
    collector_policy.requires_grad_(False)
    policy_module.eval()
    collector_policy.eval()
    if any(parameter.requires_grad for parameter in collector_policy.parameters()):
        raise RuntimeError("Frozen low-level tracker still has trainable parameters.")

    checkpoint_keys = sorted(str(key) for key in checkpoint)
    ignored_keys = [key for key in checkpoint_keys if key != "policy_state_dict"]
    parameter_count = sum(
        int(parameter.numel()) for parameter in collector_policy.parameters()
    )
    provenance: dict[str, Any] = {
        "checkpoint_path": str(path),
        "checkpoint_sha256": _sha256(path),
        "loaded_components": ["policy_state_dict"],
        "ignored_checkpoint_keys": ignored_keys,
        "strict_policy_restore": True,
        "policy_frozen": True,
        "policy_training": bool(collector_policy.training),
        "policy_parameter_count": int(parameter_count),
        "policy_trainable_parameter_count": 0,
        "policy_input_keys": [_json_key(key) for key in actual_keys],
    }
    return FrozenLowLevelTracker(policy=collector_policy, provenance=provenance)


__all__ = ["FrozenLowLevelTracker", "load_frozen_low_level_tracker"]
