#!/usr/bin/env python3
"""Shared utilities for fair interface-planner baselines."""

from __future__ import annotations

import json
import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
import torch.nn.functional as F
import yaml
from rlopt.agent.causal_interface_planner import (
    CausalInterfaceTransformerCategoricalPlanner,
    CausalInterfaceTransformerFlowPlanner,
)
from rlopt.agent.skill_commander import (
    build_rank_embedding_lookup,
    load_language_embedding_table,
)
from torch import nn

from planner_sample_schema import PLANNER_SAMPLE_FORMAT, PLANNER_SAMPLE_VERSION


INTERFACE_TERMS: dict[str, tuple[str, ...]] = {
    "latent_skill": ("z",),
    "full_body_trajectory": (
        "expert_motion",
        "expert_anchor_pos_b",
        "expert_anchor_ori_b",
    ),
    "ee_trajectory": ("expert_ee_pos_b", "expert_ee_ori_b"),
    "future_cvae": ("z",),
    "per_step_token_sequence": ("token_ids",),
}
ROLLOUT_SAMPLE_TENSOR_KEYS = (
    "planner_state",
    "expert_planner_state",
    "causal_target",
    "demonstration_target",
    "traj_rank",
    "episode_id",
    "control_step",
    "planner_step",
)
LANGUAGE_SAMPLE_KEY = "language_embedding"

STATE_TARGET_KEYS = {
    "planner_state": "causal_target",
    "causal_state_history": "causal_target",
    "expert_planner_state": "demonstration_target",
    "demonstration_state_history": "demonstration_target",
}


def paired_target_key(state_key: str, metadata: dict[str, Any]) -> str:
    """Return the target expressed for the selected planner state."""
    key = str(state_key)
    if key not in STATE_TARGET_KEYS:
        raise ValueError(
            f"Unsupported planner state key {key!r}; expected one of "
            f"{sorted(STATE_TARGET_KEYS)}."
        )
    contract = metadata.get("paired_target_contract")
    if not isinstance(contract, dict):
        raise ValueError("Planner samples have no paired_target_contract metadata.")
    target_key = STATE_TARGET_KEYS[key]
    declared_targets = {
        str(item.get("target")) for item in contract.values() if isinstance(item, dict)
    }
    if target_key not in declared_targets:
        raise ValueError(
            f"Planner sample contract does not declare target {target_key!r}."
        )
    return target_key


@dataclass(frozen=True)
class InterfaceTargetSpec:
    interface: str
    term_names: tuple[str, ...]
    term_widths: tuple[int, ...]

    @property
    def target_dim(self) -> int:
        return int(sum(self.term_widths))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["term_names"] = list(self.term_names)
        payload["term_widths"] = list(self.term_widths)
        payload["target_dim"] = self.target_dim
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "InterfaceTargetSpec":
        return cls(
            interface=str(payload["interface"]),
            term_names=tuple(str(name) for name in payload["term_names"]),
            term_widths=tuple(int(width) for width in payload["term_widths"]),
        )


def supported_interfaces() -> tuple[str, ...]:
    return tuple(INTERFACE_TERMS)


def empty_language(
    batch_size: int, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    return torch.empty((int(batch_size), 0), device=device, dtype=dtype)


def file_sha256(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rank_language_embeddings(
    path: str | Path,
    *,
    motion_names: list[str],
    device: torch.device | str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Load a manifest-aligned language lookup and serializable provenance."""
    resolved = Path(path).expanduser().resolve()
    table = load_language_embedding_table(resolved)
    lookup = build_rank_embedding_lookup(table, motion_names, device)
    metadata = {
        "enabled": True,
        "embedding_dim": int(table["embed_dim"]),
        "embedding_path": str(resolved),
        "embedding_sha256": file_sha256(resolved),
        "backend": table.get("backend"),
        "model": table.get("model"),
        "motion_count": len(motion_names),
    }
    if tuple(lookup.shape) != (len(motion_names), int(table["embed_dim"])):
        raise ValueError(
            "Rank-indexed language lookup shape does not match the active motions: "
            f"{tuple(lookup.shape)}."
        )
    return lookup, metadata


def load_language_goal_embedding(
    path: str | Path,
    *,
    goal_name: str,
    device: torch.device | str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Load one explicit deployable language goal without reading a motion cursor."""
    resolved = Path(path).expanduser().resolve()
    table = load_language_embedding_table(resolved)
    normalized_name = str(goal_name).strip()
    index = table["name_to_index"].get(normalized_name)
    if index is None:
        raise ValueError(
            f"Language embedding table has no explicit goal {normalized_name!r}."
        )
    embedding = table["embeddings"][int(index)].to(
        device=device,
        dtype=torch.float32,
    )
    metadata = {
        "enabled": True,
        "embedding_dim": int(table["embed_dim"]),
        "embedding_path": str(resolved),
        "embedding_sha256": file_sha256(resolved),
        "backend": table.get("backend"),
        "model": table.get("model"),
        "goal_name": normalized_name,
        "goal_phrase": (
            table.get("phrases", [])[int(index)]
            if int(index) < len(table.get("phrases", []))
            else normalized_name
        ),
    }
    return embedding.reshape(1, -1).contiguous(), metadata


def flatten_command_terms(
    interface: str,
    command_terms: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, InterfaceTargetSpec]:
    interface = str(interface)
    if interface not in INTERFACE_TERMS:
        raise ValueError(
            f"Unsupported interface={interface!r}; expected one of {sorted(INTERFACE_TERMS)}."
        )
    term_names = INTERFACE_TERMS[interface]
    missing = [name for name in term_names if name not in command_terms]
    if missing:
        raise KeyError(f"Missing command target terms for {interface}: {missing}")
    tensors = [command_terms[name] for name in term_names]
    batch_size = int(tensors[0].shape[0])
    for name, tensor in zip(term_names, tensors):
        if tensor.ndim != 2:
            raise ValueError(
                f"Command term {name!r} must be rank-2, got {tuple(tensor.shape)}."
            )
        if int(tensor.shape[0]) != batch_size:
            raise ValueError(
                f"Command term {name!r} batch mismatch: expected {batch_size}, got {tensor.shape[0]}."
            )
    target = torch.cat(tensors, dim=-1).contiguous()
    spec = InterfaceTargetSpec(
        interface=interface,
        term_names=term_names,
        term_widths=tuple(int(tensor.shape[-1]) for tensor in tensors),
    )
    return target, spec


def unflatten_command_target(
    target: torch.Tensor,
    spec: InterfaceTargetSpec | dict[str, Any],
) -> dict[str, torch.Tensor]:
    if isinstance(spec, dict):
        spec = InterfaceTargetSpec.from_dict(spec)
    if target.ndim != 2:
        raise ValueError(
            f"Flat command target must be rank-2, got {tuple(target.shape)}."
        )
    if int(target.shape[-1]) != spec.target_dim:
        raise ValueError(
            f"Target width mismatch for {spec.interface}: expected {spec.target_dim}, got {target.shape[-1]}."
        )
    terms: dict[str, torch.Tensor] = {}
    offset = 0
    for name, width in zip(spec.term_names, spec.term_widths):
        next_offset = offset + int(width)
        terms[name] = target[:, offset:next_offset].contiguous()
        offset = next_offset
    return terms


def planner_state_from_batch(batch: Any, *, state_history_steps: int) -> torch.Tensor:
    group = "planner" if ("planner", "state") in batch.keys(True) else "hl"
    history_key = (group, "state_history")
    if int(state_history_steps) > 0 and history_key in batch.keys(True):
        return batch.get(history_key).reshape(batch.batch_size[0], -1)
    state = batch.get((group, "state"))
    if state is None:
        raise KeyError(f"Planner batch is missing {group}/state.")
    return state.reshape(batch.batch_size[0], -1)


def write_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def parameter_counts(module: nn.Module) -> dict[str, int]:
    parameters = list(module.parameters())
    return {
        "parameter_count": int(sum(parameter.numel() for parameter in parameters)),
        "trainable_parameter_count": int(
            sum(
                parameter.numel() for parameter in parameters if parameter.requires_grad
            )
        ),
    }


def _to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_to_serializable(item) for item in value]
    if isinstance(value, list):
        return [_to_serializable(item) for item in value]
    return value


def _metadata_signature(metadata: dict[str, Any]) -> str:
    provenance = metadata.get("provenance", {})
    stable_provenance = {}
    if isinstance(provenance, dict):
        stable_provenance = {
            key: provenance.get(key)
            for key in (
                "low_level_checkpoint",
                "low_level_tracker",
                "motion_manifest",
                "dataset_path",
                "skill_checkpoint",
            )
            if key in provenance
        }
    language = metadata.get("language_conditioning", {})
    stable_language = {}
    if isinstance(language, dict):
        stable_language = {
            key: language.get(key)
            for key in (
                "enabled",
                "embedding_dim",
                "embedding_path",
                "embedding_sha256",
                "backend",
                "model",
            )
            if key in language
        }
    contract = {
        key: metadata.get(key)
        for key in (
            "sample_format",
            "interface",
            "low_level_command_mode",
            "low_level_command_space",
            "policy_command_mode",
            "target_spec",
            "state_history_steps",
            "command_past_steps",
            "command_future_steps",
            "task",
            "algorithm",
            "seed",
            "planner_observation_spec",
            "control_rate_hz",
            "planner_interval_steps",
            "planner_rate_hz",
            "reset_schedule",
            "wrap_steps",
            "policy_observation_corruption_enabled",
            "early_terminations_enabled",
            "time_out_enabled",
            "episode_length_extension_enabled",
            "reward_clipping_enabled",
            "target_encoding",
        )
    }
    contract["language_conditioning"] = stable_language
    contract["provenance"] = stable_provenance
    return json.dumps(_to_serializable(contract), sort_keys=True)


def _sample_step(value: Any, *, sample_path: Path) -> int:
    if value is None:
        raise KeyError(f"Sample {sample_path} is missing required key 'step'.")
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(
                f"Sample {sample_path} step tensor must contain one value, got {tuple(value.shape)}."
            )
        value = value.item()
    return int(value)


def _sample_tensor(
    sample: dict[str, Any],
    key: str,
    *,
    sample_path: Path,
    target_kind: str = "continuous",
) -> torch.Tensor:
    if key not in sample:
        raise KeyError(f"Sample {sample_path} is missing required key {key!r}.")
    value = sample[key]
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            f"Sample {sample_path} key {key!r} must be a tensor, got {type(value).__name__}."
        )
    if key in {"traj_rank", "episode_id", "control_step", "planner_step"}:
        value = value.reshape(-1)
        return value.to(dtype=torch.long)
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim != 2:
        raise ValueError(
            f"Sample {sample_path} key {key!r} must be rank-2, got {tuple(value.shape)}."
        )
    dtype = (
        torch.long
        if key in {"causal_target", "demonstration_target"}
        and target_kind == "categorical_sequence"
        else torch.float32
    )
    return value.to(dtype=dtype)


def _sample_metadata(sample: dict[str, Any], *, sample_path: Path) -> dict[str, Any]:
    if "metadata" not in sample:
        raise KeyError(f"Sample {sample_path} is missing required key 'metadata'.")
    metadata = dict(sample["metadata"])
    if "interface" not in metadata:
        raise KeyError(f"Sample {sample_path} metadata is missing 'interface'.")
    if "target_spec" not in metadata:
        raise KeyError(f"Sample {sample_path} metadata is missing 'target_spec'.")
    expected_format = {
        "name": PLANNER_SAMPLE_FORMAT,
        "version": PLANNER_SAMPLE_VERSION,
    }
    if metadata.get("sample_format") != expected_format:
        raise ValueError(
            f"Sample {sample_path} does not use the Phase 2 format: "
            f"{metadata.get('sample_format')} != {expected_format}."
        )
    return metadata


class InterfacePlanner(Protocol):
    state_dim: int
    target_dim: int

    def flow_matching_loss(
        self,
        state: torch.Tensor,
        target: torch.Tensor,
        *,
        language: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

    def forward(
        self,
        state: torch.Tensor,
        *,
        num_inference_steps: int = 16,
        inference_noise_std: float = 0.0,
        language: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

    def config_dict(self) -> dict[str, Any]: ...


class InterfaceFlowPlanner(nn.Module):
    """Small conditional flow-matching planner for command-interface targets."""

    def __init__(
        self,
        *,
        state_dim: int,
        target_dim: int,
        hidden_dims: tuple[int, ...] = (512, 512, 256),
        activation: str = "mish",
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.target_dim = int(target_dim)
        self.hidden_dims = tuple(int(dim) for dim in hidden_dims)
        self.activation = str(activation)
        input_dim = self.state_dim + self.target_dim + 1
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(_activation(self.activation))
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, self.target_dim))
        self.net = nn.Sequential(*layers)

    def velocity(
        self, state: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        return self.net(torch.cat([state, x_t, t.to(dtype=state.dtype)], dim=-1))

    def flow_matching_loss(
        self,
        state: torch.Tensor,
        target: torch.Tensor,
        *,
        language: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if language is not None and int(language.shape[-1]) != 0:
            raise ValueError("Legacy MLP planner does not accept language input.")
        noise = torch.randn_like(target)
        t = torch.rand((target.shape[0], 1), device=target.device, dtype=target.dtype)
        x_t = (1.0 - t) * noise + t * target
        target_velocity = target - noise
        return F.mse_loss(self.velocity(state, x_t, t), target_velocity)

    def forward(
        self,
        state: torch.Tensor,
        *,
        num_inference_steps: int = 16,
        inference_noise_std: float = 0.0,
        language: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if language is not None and int(language.shape[-1]) != 0:
            raise ValueError("Legacy MLP planner does not accept language input.")
        steps = max(1, int(num_inference_steps))
        if float(inference_noise_std) > 0.0:
            x_t = torch.randn(
                (state.shape[0], self.target_dim),
                device=state.device,
                dtype=state.dtype,
            ) * float(inference_noise_std)
        else:
            x_t = torch.zeros(
                (state.shape[0], self.target_dim),
                device=state.device,
                dtype=state.dtype,
            )
        dt = 1.0 / float(steps)
        for step in range(steps):
            t = torch.full(
                (state.shape[0], 1),
                float(step) / float(steps),
                device=state.device,
                dtype=state.dtype,
            )
            x_t = x_t + dt * self.velocity(state, x_t, t)
        return x_t

    def config_dict(self) -> dict[str, Any]:
        return {
            "planner_type": "mlp_flow",
            "state_dim": self.state_dim,
            "target_dim": self.target_dim,
            "hidden_dims": list(self.hidden_dims),
            "activation": self.activation,
        }


class _LegacyChunkedTransformerFlowPlanner(nn.Module):
    """Conditional Transformer flow planner over fixed-width command chunks.

    This is the stronger internal baseline: the command vector is padded into
    chunk tokens, conditioned on learned state tokens, and denoised by a
    Transformer encoder. Flow dynamics run in normalized target space while the
    public forward path returns commands in the original interface units.
    """

    def __init__(
        self,
        *,
        state_dim: int,
        target_dim: int,
        term_widths: tuple[int, ...] = (),
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        feedforward_dim: int = 2048,
        patch_dim: int = 32,
        num_state_tokens: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.target_dim = int(target_dim)
        self.term_widths = tuple(int(width) for width in term_widths)
        self.d_model = int(d_model)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.feedforward_dim = int(feedforward_dim)
        self.patch_dim = int(patch_dim)
        self.num_state_tokens = int(num_state_tokens)
        self.dropout = float(dropout)
        if self.state_dim <= 0 or self.target_dim <= 0:
            raise ValueError("state_dim and target_dim must be positive.")
        if self.d_model <= 0 or self.patch_dim <= 0:
            raise ValueError("d_model and patch_dim must be positive.")
        if self.num_layers <= 0 or self.num_heads <= 0:
            raise ValueError("num_layers and num_heads must be positive.")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        if self.num_state_tokens <= 0:
            raise ValueError("num_state_tokens must be positive.")
        if self.term_widths and sum(self.term_widths) != self.target_dim:
            raise ValueError(
                f"term_widths sum to {sum(self.term_widths)}, expected {self.target_dim}."
            )

        self.num_patches = int(math.ceil(self.target_dim / self.patch_dim))
        self.padded_target_dim = self.num_patches * self.patch_dim
        term_ids = _patch_term_ids(
            target_dim=self.target_dim,
            patch_dim=self.patch_dim,
            term_widths=self.term_widths,
        )

        self.register_buffer("patch_term_ids", term_ids, persistent=False)
        self.register_buffer("state_mean", torch.zeros(self.state_dim))
        self.register_buffer("state_std", torch.ones(self.state_dim))
        self.register_buffer("target_mean", torch.zeros(self.target_dim))
        self.register_buffer("target_std", torch.ones(self.target_dim))

        self.state_proj = nn.Sequential(
            nn.Linear(self.state_dim, self.d_model * self.num_state_tokens),
            nn.Mish(),
            nn.LayerNorm(self.d_model * self.num_state_tokens),
        )
        self.patch_proj = nn.Linear(self.patch_dim, self.d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 4),
            nn.Mish(),
            nn.Linear(self.d_model * 4, self.d_model),
        )
        self.state_token_embed = nn.Parameter(
            torch.zeros(1, self.num_state_tokens, self.d_model)
        )
        self.patch_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, self.d_model)
        )
        self.term_embed = nn.Embedding(max(1, len(self.term_widths)), self.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.feedforward_dim,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.output_norm = nn.LayerNorm(self.d_model)
        self.patch_head = nn.Linear(self.d_model, self.patch_dim)
        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.normal_(self.state_token_embed, std=0.02)
        nn.init.normal_(self.patch_pos_embed, std=0.02)

    @torch.no_grad()
    def set_normalization(
        self,
        *,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        target_mean: torch.Tensor,
        target_std: torch.Tensor,
        min_std: float = 1.0e-4,
    ) -> None:
        if tuple(state_mean.shape) != (self.state_dim,):
            raise ValueError(
                f"state_mean shape {tuple(state_mean.shape)} does not match {(self.state_dim,)}."
            )
        if tuple(state_std.shape) != (self.state_dim,):
            raise ValueError(
                f"state_std shape {tuple(state_std.shape)} does not match {(self.state_dim,)}."
            )
        if tuple(target_mean.shape) != (self.target_dim,):
            raise ValueError(
                f"target_mean shape {tuple(target_mean.shape)} does not match {(self.target_dim,)}."
            )
        if tuple(target_std.shape) != (self.target_dim,):
            raise ValueError(
                f"target_std shape {tuple(target_std.shape)} does not match {(self.target_dim,)}."
            )
        device = self.state_mean.device
        self.state_mean.copy_(state_mean.to(device=device, dtype=torch.float32))
        self.state_std.copy_(
            state_std.to(device=device, dtype=torch.float32).clamp_min(float(min_std))
        )
        self.target_mean.copy_(target_mean.to(device=device, dtype=torch.float32))
        self.target_std.copy_(
            target_std.to(device=device, dtype=torch.float32).clamp_min(float(min_std))
        )

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        return (state - self.state_mean.to(state.device)) / self.state_std.to(
            state.device
        )

    def normalize_target(self, target: torch.Tensor) -> torch.Tensor:
        return (target - self.target_mean.to(target.device)) / self.target_std.to(
            target.device
        )

    def denormalize_target(self, target: torch.Tensor) -> torch.Tensor:
        return target * self.target_std.to(target.device) + self.target_mean.to(
            target.device
        )

    def _patchify(self, flat_target: torch.Tensor) -> torch.Tensor:
        if int(flat_target.shape[-1]) != self.target_dim:
            raise ValueError(
                f"Target width mismatch: expected {self.target_dim}, got {flat_target.shape[-1]}."
            )
        if self.padded_target_dim == self.target_dim:
            padded = flat_target
        else:
            padded = F.pad(flat_target, (0, self.padded_target_dim - self.target_dim))
        return padded.reshape(flat_target.shape[0], self.num_patches, self.patch_dim)

    def _unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        flat = patches.reshape(patches.shape[0], self.padded_target_dim)
        return flat[:, : self.target_dim].contiguous()

    def velocity(
        self, state: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        state_norm = self.normalize_state(state)
        batch_size = int(state.shape[0])
        time_embed = self.time_mlp(_sinusoidal_embedding(t.reshape(-1), self.d_model))
        state_tokens = self.state_proj(state_norm).reshape(
            batch_size, self.num_state_tokens, self.d_model
        )
        state_tokens = (
            state_tokens
            + self.state_token_embed
            + time_embed.unsqueeze(1).to(dtype=state_tokens.dtype)
        )
        patch_tokens = self.patch_proj(self._patchify(x_t))
        term_embed = self.term_embed(self.patch_term_ids.to(patch_tokens.device))
        patch_tokens = (
            patch_tokens
            + self.patch_pos_embed
            + term_embed.unsqueeze(0)
            + time_embed.unsqueeze(1).to(dtype=patch_tokens.dtype)
        )
        encoded = self.encoder(torch.cat([state_tokens, patch_tokens], dim=1))
        patch_output = encoded[:, self.num_state_tokens :, :]
        patch_velocity = self.patch_head(self.output_norm(patch_output))
        return self._unpatchify(patch_velocity)

    def flow_matching_loss(
        self, state: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        target_norm = self.normalize_target(target)
        noise = torch.randn_like(target_norm)
        t = torch.rand(
            (target_norm.shape[0], 1),
            device=target_norm.device,
            dtype=target_norm.dtype,
        )
        x_t = (1.0 - t) * noise + t * target_norm
        target_velocity = target_norm - noise
        return F.mse_loss(self.velocity(state, x_t, t), target_velocity)

    def normalized_endpoint_loss(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prediction_norm = self.normalize_target(prediction)
        target_norm = self.normalize_target(target)
        mse = F.mse_loss(prediction_norm, target_norm)
        cosine = 1.0 - F.cosine_similarity(prediction_norm, target_norm, dim=-1).mean()
        return mse, cosine

    def forward(
        self,
        state: torch.Tensor,
        *,
        num_inference_steps: int = 16,
        inference_noise_std: float = 0.0,
    ) -> torch.Tensor:
        steps = max(1, int(num_inference_steps))
        if float(inference_noise_std) > 0.0:
            x_t = torch.randn(
                (state.shape[0], self.target_dim),
                device=state.device,
                dtype=state.dtype,
            ) * float(inference_noise_std)
        else:
            x_t = torch.zeros(
                (state.shape[0], self.target_dim),
                device=state.device,
                dtype=state.dtype,
            )
        dt = 1.0 / float(steps)
        for step in range(steps):
            t = torch.full(
                (state.shape[0], 1),
                float(step) / float(steps),
                device=state.device,
                dtype=state.dtype,
            )
            x_t = x_t + dt * self.velocity(state, x_t, t)
        return self.denormalize_target(x_t)

    def config_dict(self) -> dict[str, Any]:
        return {
            "planner_type": "chunked_transformer_flow",
            "state_dim": self.state_dim,
            "target_dim": self.target_dim,
            "term_widths": list(self.term_widths),
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "feedforward_dim": self.feedforward_dim,
            "patch_dim": self.patch_dim,
            "num_state_tokens": self.num_state_tokens,
            "dropout": self.dropout,
        }


# Keep the public name stable for existing scripts and old checkpoints while
# making every new continuous-interface run use the exact RLOpt runtime class.
ChunkedTransformerFlowPlanner = CausalInterfaceTransformerFlowPlanner


class CausalInterfaceTransformerDiffusionPlanner(CausalInterfaceTransformerFlowPlanner):
    """DDIM-style chunk planner with the same Transformer parameters as flow."""

    planner_type = "causal_interface_transformer_diffusion"

    @staticmethod
    def _alpha_bar(t: torch.Tensor) -> torch.Tensor:
        # The small offset avoids a perfectly clean endpoint during training.
        offset = 0.008
        angle = (t + offset) / (1.0 + offset) * (math.pi / 2.0)
        return torch.cos(angle).square().clamp(1.0e-5, 1.0)

    def diffusion_loss(
        self,
        state: torch.Tensor,
        target: torch.Tensor,
        *,
        language: torch.Tensor | None = None,
    ) -> torch.Tensor:
        target_norm = self.normalize_target(target)
        noise = torch.randn_like(target_norm)
        t = torch.rand(
            (target_norm.shape[0], 1),
            device=target_norm.device,
            dtype=target_norm.dtype,
        ).clamp_min(1.0e-4)
        alpha_bar = self._alpha_bar(t)
        x_t = alpha_bar.sqrt() * target_norm + (1.0 - alpha_bar).sqrt() * noise
        predicted_clean = self.velocity(state, x_t, t, language=language)
        return F.mse_loss(predicted_clean, target_norm)

    def forward(
        self,
        state: torch.Tensor,
        *,
        num_inference_steps: int = 16,
        inference_noise_std: float = 0.0,
        language: torch.Tensor | None = None,
    ) -> torch.Tensor:
        steps = max(1, int(num_inference_steps))
        x_t = torch.zeros(
            (state.shape[0], self.target_dim),
            device=state.device,
            dtype=state.dtype,
        )
        if float(inference_noise_std) > 0.0:
            x_t.normal_().mul_(float(inference_noise_std))
        x0 = x_t
        for step in range(steps, 0, -1):
            t = torch.full(
                (state.shape[0], 1),
                float(step) / float(steps),
                device=state.device,
                dtype=state.dtype,
            )
            previous_t = torch.full_like(t, float(step - 1) / float(steps))
            alpha_bar = self._alpha_bar(t)
            previous_alpha_bar = self._alpha_bar(previous_t)
            x0 = self.velocity(state, x_t, t, language=language).clamp(-20.0, 20.0)
            predicted_noise = (x_t - alpha_bar.sqrt() * x0) / (
                1.0 - alpha_bar
            ).sqrt().clamp_min(1.0e-4)
            x_t = (
                previous_alpha_bar.sqrt() * x0
                + (1.0 - previous_alpha_bar).sqrt() * predicted_noise
            )
        return self.denormalize_target(x0)


class CausalInterfaceTransformerDeterministicPlanner(
    CausalInterfaceTransformerFlowPlanner
):
    """Single-pass chunk predictor with the same Transformer parameters as flow."""

    planner_type = "causal_interface_transformer_deterministic"

    def forward(
        self,
        state: torch.Tensor,
        *,
        num_inference_steps: int = 1,
        inference_noise_std: float = 0.0,
        language: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del num_inference_steps, inference_noise_std
        query = torch.zeros(
            (state.shape[0], self.target_dim),
            device=state.device,
            dtype=state.dtype,
        )
        t = torch.zeros((state.shape[0], 1), device=state.device, dtype=state.dtype)
        prediction = self.velocity(state, query, t, language=language)
        return self.denormalize_target(prediction)

    def deterministic_loss(
        self,
        state: torch.Tensor,
        target: torch.Tensor,
        *,
        language: torch.Tensor | None = None,
    ) -> torch.Tensor:
        prediction = self(
            state,
            num_inference_steps=1,
            inference_noise_std=0.0,
            language=language,
        )
        prediction_norm = self.normalize_target(prediction)
        return F.mse_loss(prediction_norm, self.normalize_target(target))


def _patch_term_ids(
    *,
    target_dim: int,
    patch_dim: int,
    term_widths: tuple[int, ...],
) -> torch.Tensor:
    num_patches = int(math.ceil(int(target_dim) / int(patch_dim)))
    if not term_widths:
        return torch.zeros(num_patches, dtype=torch.long)
    term_ends: list[int] = []
    running = 0
    for width in term_widths:
        running += int(width)
        term_ends.append(running)
    ids: list[int] = []
    for patch_idx in range(num_patches):
        patch_start = patch_idx * int(patch_dim)
        term_idx = 0
        while term_idx + 1 < len(term_ends) and patch_start >= term_ends[term_idx]:
            term_idx += 1
        ids.append(term_idx)
    return torch.as_tensor(ids, dtype=torch.long)


def _sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = int(dim) // 2
    if half <= 0:
        return t.unsqueeze(-1)
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=t.device, dtype=t.dtype)
        / max(half - 1, 1)
    )
    args = t.unsqueeze(-1) * frequencies.unsqueeze(0)
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if int(dim) % 2 == 1:
        embedding = F.pad(embedding, (0, 1))
    return embedding


def _activation(name: str) -> nn.Module:
    normalized = name.strip().lower()
    if normalized == "mish":
        return nn.Mish()
    if normalized == "elu":
        return nn.ELU()
    if normalized == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation={name!r}.")


def load_rollout_samples(
    samples_dir: Path,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    sample_paths = sorted(samples_dir.expanduser().glob("sample_step_*.pt"))
    if not sample_paths:
        raise FileNotFoundError(f"No sample_step_*.pt files found in {samples_dir}.")
    tensor_rows: dict[str, list[torch.Tensor]] = {
        key: [] for key in ROLLOUT_SAMPLE_TENSOR_KEYS
    }
    has_language: bool | None = None
    steps: list[int] = []
    metadata: dict[str, Any] | None = None
    metadata_signature: str | None = None
    for sample_path in sample_paths:
        sample = torch.load(sample_path, map_location="cpu", weights_only=False)
        sample_metadata = _sample_metadata(sample, sample_path=sample_path)
        sample_has_language = LANGUAGE_SAMPLE_KEY in sample
        if has_language is None:
            has_language = sample_has_language
            if has_language:
                tensor_rows[LANGUAGE_SAMPLE_KEY] = []
        elif sample_has_language != has_language:
            raise ValueError(
                "Planner sample files mix language-conditioned and state-only rows."
            )
        if metadata is None:
            metadata = sample_metadata
            metadata_signature = _metadata_signature(sample_metadata)
        elif _metadata_signature(sample_metadata) != metadata_signature:
            raise ValueError(
                f"Sample {sample_path} metadata does not match earlier sample metadata."
            )

        target_encoding = sample_metadata.get("target_encoding", {"kind": "continuous"})
        target_kind = (
            str(target_encoding.get("kind", "continuous"))
            if isinstance(target_encoding, dict)
            else "continuous"
        )
        sample_tensors = {
            key: _sample_tensor(
                sample,
                key,
                sample_path=sample_path,
                target_kind=target_kind,
            )
            for key in tensor_rows
        }
        row_count = int(sample_tensors["planner_state"].shape[0])
        for key, value in sample_tensors.items():
            if int(value.shape[0]) != row_count:
                raise ValueError(
                    f"Sample {sample_path} key {key!r} row count {value.shape[0]} "
                    f"does not match planner_state row count {row_count}."
                )
            tensor_rows[key].append(value)

        step = _sample_step(sample.get("step"), sample_path=sample_path)
        steps.extend([step] * row_count)
    data = {
        key: torch.cat(values, dim=0).contiguous()
        for key, values in tensor_rows.items()
    }
    if metadata is None:
        raise RuntimeError(f"No sample metadata loaded from {samples_dir}.")
    language_metadata = metadata.get("language_conditioning", {})
    language_enabled = bool(
        isinstance(language_metadata, dict) and language_metadata.get("enabled", False)
    )
    if language_enabled != bool(has_language):
        raise ValueError(
            "Planner language metadata does not match the saved sample tensors."
        )
    if language_enabled:
        language_dim = int(language_metadata.get("embedding_dim", -1))
        if int(data[LANGUAGE_SAMPLE_KEY].shape[-1]) != language_dim:
            raise ValueError(
                "Saved language width does not match metadata: "
                f"{data[LANGUAGE_SAMPLE_KEY].shape[-1]} != {language_dim}."
            )
    target_spec = InterfaceTargetSpec.from_dict(metadata["target_spec"])
    for target_key in ("causal_target", "demonstration_target"):
        target_dim = int(data[target_key].shape[-1])
        if target_dim != target_spec.target_dim:
            raise ValueError(
                f"Sample {target_key} width mismatch: tensor has {target_dim}, "
                f"metadata target_spec has {target_spec.target_dim}."
            )
    if str(metadata["interface"]) != target_spec.interface:
        raise ValueError(
            "Sample metadata interface does not match target_spec interface: "
            f"{metadata['interface']!r} vs {target_spec.interface!r}."
        )
    data["step"] = torch.as_tensor(steps, dtype=torch.long)
    return data, metadata or {}


def save_planner_checkpoint(
    path: Path,
    *,
    planner: InterfacePlanner,
    optimizer: torch.optim.Optimizer | None,
    target_spec: InterfaceTargetSpec,
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable_metadata = _to_serializable(metadata)
    payload = {
        "planner_state_dict": planner.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
        if optimizer is not None
        else None,
        "planner_config": planner.config_dict(),
        "target_spec": target_spec.to_dict(),
        "metadata": serializable_metadata,
    }
    torch.save(payload, path)
    config_path = path.parent.parent / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "planner_config": planner.config_dict(),
                "target_spec": target_spec.to_dict(),
                "metadata": serializable_metadata,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def load_planner_checkpoint(
    path: Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[nn.Module, InterfaceTargetSpec, dict[str, Any]]:
    checkpoint = torch.load(
        path.expanduser(), map_location=map_location, weights_only=False
    )
    config = checkpoint["planner_config"]
    target_spec = InterfaceTargetSpec.from_dict(checkpoint["target_spec"])
    planner_type = str(config.get("planner_type", "mlp_flow"))
    if planner_type == "mlp_flow":
        planner = InterfaceFlowPlanner(
            state_dim=int(config["state_dim"]),
            target_dim=int(config["target_dim"]),
            hidden_dims=tuple(
                int(dim) for dim in config.get("hidden_dims", (512, 512, 256))
            ),
            activation=str(config.get("activation", "mish")),
        )
    elif planner_type in {
        "chunked_transformer_flow",
        "causal_interface_transformer_flow",
    }:
        planner = ChunkedTransformerFlowPlanner(
            state_dim=int(config["state_dim"]),
            target_dim=int(config["target_dim"]),
            term_widths=tuple(
                int(width)
                for width in config.get("term_widths", target_spec.term_widths)
            ),
            d_model=int(config.get("d_model", 512)),
            num_layers=int(config.get("num_layers", 6)),
            num_heads=int(config.get("num_heads", 8)),
            feedforward_dim=int(config.get("feedforward_dim", 2048)),
            patch_dim=int(config.get("patch_dim", 32)),
            num_state_tokens=int(config.get("num_state_tokens", 4)),
            language_dim=int(config.get("language_dim", 0)),
            num_language_tokens=int(config.get("num_language_tokens", 1)),
            dropout=float(config.get("dropout", 0.0)),
        )
    elif planner_type == "causal_interface_transformer_diffusion":
        planner = CausalInterfaceTransformerDiffusionPlanner.from_config(config)
    elif planner_type == "causal_interface_transformer_deterministic":
        planner = CausalInterfaceTransformerDeterministicPlanner.from_config(config)
    elif planner_type == "causal_interface_transformer_categorical":
        planner = CausalInterfaceTransformerCategoricalPlanner.from_config(config)
    else:
        raise ValueError(f"Unsupported planner_type={planner_type!r}.")
    planner.load_state_dict(checkpoint["planner_state_dict"])
    return planner, target_spec, dict(checkpoint.get("metadata", {}))


def mean_std(values: torch.Tensor) -> tuple[float, float]:
    values = values.detach().float()
    if values.numel() == 0:
        return float("nan"), float("nan")
    mean = float(values.mean().item())
    std = float(values.std(unbiased=False).item()) if values.numel() > 1 else 0.0
    return mean, std


def cosine_mean(prediction: torch.Tensor, target: torch.Tensor) -> float:
    if prediction.shape[-1] == 0:
        return float("nan")
    return float(F.cosine_similarity(prediction, target, dim=-1).mean().item())


def rmse_per_row(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((prediction - target).square(), dim=-1))


def finite_float(value: float) -> float | None:
    return value if math.isfinite(float(value)) else None
