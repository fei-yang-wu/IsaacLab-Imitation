"""Verbatim GR00T N1.7 action-head integration.

The model code is NEVER copied: it is imported from the pinned submodule at
``external/Isaac-GR00T``. This module owns only the glue:

- import plumbing (``sys.path`` + optional ``gr00t.model`` package stub so the
  Isaac environment does not need the GR00T dataset stack),
- the G1 embodiment head configuration,
- the filtered pretrained load (keep the embodiment-independent trunk, fresh
  projectors) with a recorded kept/fresh manifest,
- quantile normalization and ``BatchFeature`` collation helpers.

Everything imports lazily so this module is importable (and its pure logic
testable) in environments without the GR00T dependencies.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Mapping

import torch

from imitation_experiments.paths import REPO_ROOT

GR00T_SUBMODULE = REPO_ROOT / "external" / "Isaac-GR00T"

ROOT_QPOS_WIDTH = 38
PLANNER_STATE_WIDTH = 93
PLANNER_STATE_HISTORY = 10
BACKBONE_EMBEDDING_DIM = 2048  # Cosmos-Reason2-2B hidden size

# Embodiment-independent trunk of ``Gr00tN1d7ActionHead`` — transferable from
# the released checkpoint. Everything else is embodiment-specific
# (CategorySpecificMLP banks) and must be freshly initialized for G1.
PRETRAINED_KEEP_PREFIXES: tuple[str, ...] = (
    "model.",  # DiT (incl. its timestep encoder)
    "vlln.",
    "vl_self_attention.",
    "position_embedding.",
)
FRESH_PREFIXES: tuple[str, ...] = (
    "state_encoder.",
    "action_encoder.",
    "action_decoder.",
)


def gr00t_submodule_commit() -> str:
    """Pinned commit of the Isaac-GR00T submodule, for provenance records."""
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=GR00T_SUBMODULE, text=True
    ).strip()


def ensure_gr00t_importable(*, stub_model_package: bool = False) -> None:
    """Make ``gr00t`` importable from the pinned submodule.

    ``stub_model_package=True`` registers a stub for ``gr00t.model`` whose
    ``__init__`` only exports the training pipeline and drags in the dataset
    stack (pandas/lmdb/msgpack). The stub lets leaf modules (config, DiT,
    action head) import without those dependencies — required inside the
    ``isaaclab`` environment, harmless elsewhere. All other package
    ``__init__`` files on the leaf path are empty.
    """
    if not (GR00T_SUBMODULE / "gr00t" / "__init__.py").is_file():
        msg = (
            f"Isaac-GR00T submodule not found at {GR00T_SUBMODULE}. "
            "Run: git submodule update --init external/Isaac-GR00T"
        )
        raise FileNotFoundError(msg)
    root = str(GR00T_SUBMODULE)
    if root not in sys.path:
        sys.path.insert(0, root)
    if stub_model_package and "gr00t.model" not in sys.modules:
        import gr00t  # noqa: PLC0415  (root __init__ is dependency-free)

        stub = types.ModuleType("gr00t.model")
        stub.__path__ = [str(GR00T_SUBMODULE / "gr00t" / "model")]
        stub.__package__ = "gr00t.model"
        sys.modules["gr00t.model"] = stub
        gr00t.model = stub  # type: ignore[attr-defined]


def import_head_classes(*, stub_model_package: bool = False) -> tuple[Any, Any]:
    """Return ``(Gr00tN1d7Config, Gr00tN1d7ActionHead)`` from the submodule."""
    ensure_gr00t_importable(stub_model_package=stub_model_package)
    from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config  # noqa: PLC0415
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import (  # noqa: PLC0415
        Gr00tN1d7ActionHead,
    )

    return Gr00tN1d7Config, Gr00tN1d7ActionHead


# N1.7-3B trunk architecture (must match the released checkpoint for the
# warm start; overridden by the exported bundle's recorded config when
# loading pretrained weights).
N17_TRUNK_CONFIG: dict[str, Any] = {
    "hidden_size": 1024,
    "input_embedding_dim": 1536,
    "backbone_embedding_dim": BACKBONE_EMBEDDING_DIM,
    "diffusion_model_cfg": {
        "positional_embeddings": None,
        "num_layers": 16,
        "num_attention_heads": 32,
        "attention_head_dim": 48,
        "norm_type": "ada_norm",
        "dropout": 0.2,
        "final_dropout": True,
        "output_dim": 1024,
        "interleave_self_attention": True,
    },
}

# Small trunk for local pipeline debugging (no warm start possible).
DEBUG_TRUNK_CONFIG: dict[str, Any] = {
    "hidden_size": 256,
    "input_embedding_dim": 256,
    "backbone_embedding_dim": BACKBONE_EMBEDDING_DIM,
    "diffusion_model_cfg": {
        "positional_embeddings": None,
        "num_layers": 4,
        "num_attention_heads": 8,
        "attention_head_dim": 32,
        "norm_type": "ada_norm",
        "dropout": 0.0,
        "final_dropout": True,
        "output_dim": 256,
        "interleave_self_attention": True,
    },
}


def build_g1_head_config(
    *,
    trunk: Mapping[str, Any],
    action_horizon: int = 30,
    max_action_dim: int = ROOT_QPOS_WIDTH,
    max_state_dim: int = PLANNER_STATE_WIDTH,
    state_history_length: int = PLANNER_STATE_HISTORY,
    state_dropout_prob: float = 0.0,
    num_inference_timesteps: int = 4,
) -> Any:
    """G1 embodiment config on a given trunk.

    Text-only conditioning uses ``use_alternate_vl_dit=True`` with
    ``attend_text_every_n_blocks=1``: every cross-attention block then attends
    the non-image (text) tokens under the real padding mask. The plain ``DiT``
    forward ignores ``encoder_attention_mask`` entirely, and any image-block
    schedule would softmax over an empty token set (NaN) — so this is the only
    padding-correct, weight-compatible configuration without images.
    """
    config_cls, _ = import_head_classes()
    return config_cls(
        max_state_dim=int(max_state_dim),
        state_history_length=int(state_history_length),
        max_action_dim=int(max_action_dim),
        action_horizon=int(action_horizon),
        max_num_embodiments=1,
        state_dropout_prob=float(state_dropout_prob),
        use_alternate_vl_dit=True,
        attend_text_every_n_blocks=1,
        num_inference_timesteps=int(num_inference_timesteps),
        vl_self_attention_cfg=dict(
            trunk.get("vl_self_attention_cfg", {"num_layers": 0})
        ),
        **{
            key: (dict(value) if isinstance(value, Mapping) else value)
            for key, value in trunk.items()
            if key != "vl_self_attention_cfg"
        },
    )


def classify_head_keys(keys: list[str]) -> dict[str, str]:
    """Map each action-head state-dict key to ``keep``/``fresh``/``other``."""
    result: dict[str, str] = {}
    for key in keys:
        if key.startswith(PRETRAINED_KEEP_PREFIXES):
            result[key] = "keep"
        elif key.startswith(FRESH_PREFIXES):
            result[key] = "fresh"
        else:
            result[key] = "other"
    return result


def _tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().float().numpy().tobytes()
    ).hexdigest()


def filtered_pretrained_load(
    head: torch.nn.Module,
    source_state: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Load the transferable trunk from ``source_state`` into ``head``.

    Only ``keep``-classified keys load; ``fresh`` keys retain their new
    initialization. A ``keep`` key that is missing from the source or has a
    mismatched shape is an error, never a silent skip. Returns the kept/fresh
    manifest with per-tensor hashes for provenance.
    """
    target_state = head.state_dict()
    classes = classify_head_keys(list(target_state.keys()))
    unexpected = sorted(key for key, cls in classes.items() if cls == "other")
    if unexpected:
        msg = f"Unclassified action-head keys (update prefix lists): {unexpected[:8]}"
        raise ValueError(msg)

    kept: dict[str, torch.Tensor] = {}
    problems: list[str] = []
    for key, cls in classes.items():
        if cls != "keep":
            continue
        if key not in source_state:
            problems.append(f"missing in source: {key}")
            continue
        if tuple(source_state[key].shape) != tuple(target_state[key].shape):
            problems.append(
                f"shape mismatch {key}: source {tuple(source_state[key].shape)} "
                f"vs target {tuple(target_state[key].shape)}"
            )
            continue
        kept[key] = source_state[key]
    if problems:
        msg = (
            "Filtered pretrained load failed — trunk config does not match the "
            f"checkpoint: {problems[:8]} ({len(problems)} total)"
        )
        raise ValueError(msg)

    missing, extra = head.load_state_dict(kept, strict=False)
    unexpected_loaded = sorted(extra)
    if unexpected_loaded:
        msg = f"Unexpected keys during filtered load: {unexpected_loaded[:8]}"
        raise ValueError(msg)
    fresh = sorted(key for key, cls in classes.items() if cls == "fresh")
    if sorted(missing) != fresh:
        msg = (
            "Filtered load consistency failure: missing keys "
            f"{sorted(set(missing) ^ set(fresh))[:8]} do not equal the fresh set."
        )
        raise ValueError(msg)
    return {
        "kept": {key: _tensor_sha256(kept[key]) for key in sorted(kept)},
        "fresh": fresh,
        "num_kept_params": int(sum(kept[k].numel() for k in kept)),
        "num_fresh_params": int(
            sum(target_state[k].numel() for k in fresh if k in target_state)
        ),
    }


QUANTILE_ROW_LIMIT = 8_000_000


def compute_quantile_stats(
    values: torch.Tensor, valid: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-dimension q01/q99 over valid frames (GR00T's ``use_percentiles``)."""
    flat = values.reshape(-1, values.shape[-1]).float()
    if valid is not None:
        mask = valid.reshape(-1).bool()
        if int(mask.sum()) == 0:
            msg = "No valid frames to compute quantile statistics from."
            raise ValueError(msg)
        flat = flat[mask]
    # `torch.quantile` refuses an input above roughly 16M elements. A hold-1
    # collection is one row per control step, so the flattened frame count runs
    # past that. Estimate from a deterministic random subsample instead: q01 and
    # q99 of millions of frames are unchanged by sampling a few million of them,
    # and the alternative (a full sort) would cost far more memory than the
    # statistic is worth.
    limit = QUANTILE_ROW_LIMIT
    if int(flat.shape[0]) > limit:
        generator = torch.Generator(device="cpu").manual_seed(0)
        index = torch.randperm(int(flat.shape[0]), generator=generator)[:limit]
        flat = flat[index]
    q01 = torch.quantile(flat, 0.01, dim=0)
    q99 = torch.quantile(flat, 0.99, dim=0)
    return q01, q99


def normalize_minmax(
    values: torch.Tensor, q01: torch.Tensor, q99: torch.Tensor
) -> torch.Tensor:
    """Map values to [-1, 1] by the q01/q99 range (GR00T convention)."""
    return (values - q01) / (q99 - q01 + 1.0e-6) * 2.0 - 1.0


def denormalize_minmax(
    values: torch.Tensor, q01: torch.Tensor, q99: torch.Tensor
) -> torch.Tensor:
    return (values + 1.0) / 2.0 * (q99 - q01 + 1.0e-6) + q01


def build_batch(
    *,
    state: torch.Tensor,
    action: torch.Tensor | None,
    action_mask: torch.Tensor | None,
    language_features: torch.Tensor,
    language_attention_mask: torch.Tensor,
) -> tuple[Any, Any]:
    """Collate tensors into the ``(backbone_output, action_input)`` pair.

    Shapes: state ``[B, T_state, state_dim]``; action ``[B, H, action_dim]``
    (normalized; ``None`` at inference); action_mask ``[B, H]``; language
    features ``[B, S, 2048]`` with mask ``[B, S]``. ``image_mask`` is all
    False — there are no image tokens in this integration.
    """
    from transformers.feature_extraction_utils import BatchFeature  # noqa: PLC0415

    if state.ndim != 3:
        msg = f"state must be [B, T, D], got {tuple(state.shape)}."
        raise ValueError(msg)
    if (
        language_features.ndim != 3
        or language_features.shape[-1] != BACKBONE_EMBEDDING_DIM
    ):
        msg = (
            f"language_features must be [B, S, {BACKBONE_EMBEDDING_DIM}], "
            f"got {tuple(language_features.shape)}."
        )
        raise ValueError(msg)
    backbone_output = BatchFeature(
        data={
            "backbone_features": language_features,
            "backbone_attention_mask": language_attention_mask.bool(),
            "image_mask": torch.zeros_like(language_attention_mask, dtype=torch.bool),
        }
    )
    data: dict[str, torch.Tensor] = {
        "state": state,
        "embodiment_id": torch.zeros(
            state.shape[0], dtype=torch.long, device=state.device
        ),
    }
    if action is not None:
        if action_mask is None:
            msg = "action_mask is required whenever action is provided."
            raise ValueError(msg)
        if action.ndim != 3 or action_mask.shape != action.shape[:2]:
            msg = (
                f"action [B,H,D] / action_mask [B,H] mismatch: "
                f"{tuple(action.shape)} vs {tuple(action_mask.shape)}."
            )
            raise ValueError(msg)
        data["action"] = action
        # Loss is [B, H, D] * mask — broadcast over the action dimension.
        data["action_mask"] = action_mask.to(action.dtype).unsqueeze(-1)
    return backbone_output, BatchFeature(data=data)


def save_provenance(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(record), indent=2, sort_keys=True) + "\n")
