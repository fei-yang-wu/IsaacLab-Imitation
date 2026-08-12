"""Prepare a GR00T-head training table from a planner-sample collection.

Reads the ``rollout_training_samples/*.pt`` files written by
``scripts/rlopt/eval_skill_commander_closed_loop.py`` and emits one
consolidated ``.pt`` table with every target the GR00T action-head arms
train on:

- ``chunk``: the stored 30-frame expert ``root_qpos`` lookahead;
- ``latent``: ``slots`` consecutive per-publication latents, each the exact
  latent the oracle published ``hold_steps`` control steps apart. Slot 0 is
  the row's own stored target; later slots come from joining the rows saved
  at ``control_step + hold_steps * k`` in the same environment episode —
  never from re-encoding a shifted window, because window frames are
  re-expressed against the query-time anchor.
- ``fsq_prequant``: like ``latent`` but the per-row value is recomputed as
  the encoder's PRE-quantization lattice-scaled output (``bound(z) /
  half_levels``). The recompute is parity-gated: its rounded value must
  reproduce the stored post-quantization ``z_target`` exactly.

Every input path lives in the Hydra config — no pinned paths in code. Runs
in the default Pixi environment:

    pixi run python -m imitation_experiments.planner.prepare_gr00t_dataset \
        --config-path <campaign>/conf --config-name prepare_z256
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from imitation_experiments.paths import REPO_ROOT

ROOT_QPOS_WIDTH = 38
ENCODER_WINDOW_FRAMES = 10  # state frame + 9 future frames (intermediate mode)


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_sample_files(collection_dir: Path) -> list[Path]:
    files = sorted((collection_dir / "rollout_training_samples").glob("*.pt"))
    if not files:
        msg = f"No sample files under {collection_dir}/rollout_training_samples."
        raise FileNotFoundError(msg)
    return files


def _encoder_flat_input(future: torch.Tensor) -> torch.Tensor:
    """Encoder input layout, verified against stored targets to 8e-7:
    ``[frame 0 (state); frames 1..9 flattened frame-major]``."""
    if future.shape[1] < ENCODER_WINDOW_FRAMES:
        msg = f"window has {future.shape[1]} frames, need {ENCODER_WINDOW_FRAMES}."
        raise ValueError(msg)
    window = future[:, 1:ENCODER_WINDOW_FRAMES]
    return torch.cat([future[:, 0], window.reshape(future.shape[0], -1)], dim=1)


def _validate_window_layout(
    reference_checkpoint: Path, future: torch.Tensor, stored_z: torch.Tensor
) -> dict[str, Any]:
    """Prove the encoder-input layout on a collection's OWN encoder.

    Cross-encoder re-encoding (targets from encoder B on a collection whose
    stored `z_target` came from encoder A) cannot parity-check against the
    stored latent — different encoder, different values. Recomputing A's
    latent from the same window and matching the stored one validates exactly
    the thing the parity gate protects: that `_encoder_flat_input` reproduces
    the layout this collection was written with.
    """
    from imitation_experiments.lowlevel.export_policy_bundle import (  # noqa: PLC0415
        _encoder_trunk_from_state,
    )

    checkpoint = torch.load(
        reference_checkpoint, map_location="cpu", weights_only=False
    )
    state = checkpoint["skill_encoder_state_dict"]
    config = checkpoint["config"]
    if "_half_levels" in state:
        msg = (
            "the layout-check encoder must be the collection's own encoder; "
            f"{reference_checkpoint} is an FSQ encoder."
        )
        raise ValueError(msg)
    trunk = _encoder_trunk_from_state(
        {key: value for key, value in state.items() if key.startswith("net.")},
        str(config.get("encoder_activation", "mish")),
    )
    with torch.no_grad():
        recomputed = trunk(_encoder_flat_input(future))
    if recomputed.shape != stored_z.shape:
        msg = (
            f"layout check produced {tuple(recomputed.shape)} but the "
            f"collection stores {tuple(stored_z.shape)}."
        )
        raise ValueError(msg)
    error = float((recomputed - stored_z).abs().max())
    if error > 1.0e-4:
        msg = (
            f"encoder-input layout check failed: max abs error {error} against "
            "the collection's own stored latent."
        )
        raise ValueError(msg)
    return {
        "layout_check_encoder": str(reference_checkpoint),
        "layout_check_encoder_sha256": _sha256_file(reference_checkpoint),
        "layout_check_max_abs_error": error,
    }


def _fsq_prequant(
    encoder_checkpoint: Path,
    future: torch.Tensor,
    stored_z: torch.Tensor,
    *,
    parity_gate: bool = True,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Recompute the pre-quantization FSQ command; parity-gate vs stored."""
    from imitation_experiments.lowlevel.export_policy_bundle import (  # noqa: PLC0415
        _encoder_trunk_from_state,
    )

    checkpoint = torch.load(
        encoder_checkpoint, map_location="cpu", weights_only=False
    )
    state = checkpoint["skill_encoder_state_dict"]
    config = checkpoint["config"]
    activation = str(config.get("encoder_activation", "mish"))
    levels = [int(level) for level in config["sonic_fsq_levels"]]
    trunk = _encoder_trunk_from_state(
        {key: value for key, value in state.items() if key.startswith("net.")},
        activation,
    )
    levels_t = torch.tensor(levels, dtype=torch.float32)
    bound_half = (levels_t - 1.0) * 0.5
    offset = (levels_t.long() % 2 == 0).float() * 0.5
    shift = torch.atanh(offset / bound_half.clamp(min=1.0))
    half_levels = state["_half_levels"].float()
    with torch.no_grad():
        raw = trunk(_encoder_flat_input(future))
        bounded = torch.tanh(raw + shift) * bound_half - offset
        prequant = bounded / half_levels
        rounded = torch.round(bounded) / half_levels
    record = {
        "encoder_checkpoint": str(encoder_checkpoint),
        "encoder_checkpoint_sha256": _sha256_file(encoder_checkpoint),
        "encoder_activation": activation,
        "fsq_levels_first": levels[0],
        "fsq_dims": len(levels),
    }
    if not parity_gate:
        # Cross-encoder re-encoding: the stored latent came from a different
        # encoder, so there is nothing to compare against here. The caller
        # validates the window layout separately.
        record["parity_gate"] = "skipped_cross_encoder"
        return prequant, record
    # Parity gate. CPU-recomputed borderline values may round one lattice step
    # away from the online GPU result, so tolerate a tiny fraction of one-step
    # diffs; anything larger means the encoder or layout is wrong.
    step = float((1.0 / half_levels).min())
    diff = (rounded - stored_z).abs()
    mismatch_rows = int((diff.max(dim=1).values > 0).sum())
    mismatch_fraction = mismatch_rows / max(int(stored_z.shape[0]), 1)
    parity = float(diff.max())
    if parity > step * 1.5 or mismatch_fraction > 1.0e-4:
        msg = (
            f"FSQ parity gate failed: quantize(recomputed) differs from the "
            f"stored z_target (max {parity}, {mismatch_rows} rows) — encoder "
            "checkpoint or window layout does not match this collection."
        )
        raise ValueError(msg)
    record["parity_max_abs"] = parity
    record["parity_mismatch_rows"] = mismatch_rows
    return prequant, record


def _join_slots(
    per_row: torch.Tensor,
    env_id: torch.Tensor,
    episode_id: torch.Tensor,
    control_step: torch.Tensor,
    *,
    slots: int,
    hold_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack each row's value with the rows ``hold_steps * k`` later."""
    index: dict[tuple[int, int, int], int] = {}
    for row in range(int(env_id.numel())):
        key = (int(env_id[row]), int(episode_id[row]), int(control_step[row]))
        if key in index:
            msg = f"Duplicate row key {key}: collection is not join-safe."
            raise ValueError(msg)
        index[key] = row
    rows = int(env_id.numel())
    target = per_row.new_zeros((rows, slots, per_row.shape[-1]))
    valid = torch.zeros((rows, slots), dtype=torch.bool)
    for row in range(rows):
        env = int(env_id[row])
        episode = int(episode_id[row])
        step = int(control_step[row])
        for slot in range(slots):
            other = index.get((env, episode, step + hold_steps * slot))
            if other is None:
                continue
            target[row, slot] = per_row[other]
            valid[row, slot] = True
    if not bool(valid[:, 0].all()):
        msg = "Slot 0 must always be valid (it is the row itself)."
        raise ValueError(msg)
    return target, valid


@hydra.main(
    version_base="1.3", config_path="conf_gr00t", config_name="base_prepare"
)
def main(cfg: DictConfig) -> None:
    collection_dir = _resolve(cfg.collection_dir)
    output = _resolve(cfg.output)
    slots = int(cfg.latent.slots)
    hold_steps = int(cfg.latent.hold_steps)
    latent_source = str(cfg.latent.source)
    if latent_source not in {"stored", "fsq_prequant"}:
        msg = f"latent.source must be stored|fsq_prequant, got {latent_source!r}."
        raise ValueError(msg)
    chunk_horizon = int(cfg.chunk.horizon)
    state_fields = [str(field) for field in cfg.state_fields]

    files = _load_sample_files(collection_dir)
    columns: dict[str, list[torch.Tensor]] = {}
    motion_names: list[str] = []
    for path in files:
        sample = torch.load(path, map_location="cpu", weights_only=False)
        required = [
            "env_id",
            "episode_id",
            "control_step",
            "z_target",
            "expert_root_qpos_future",
            "expert_root_qpos_future_valid",
            *state_fields,
        ]
        for key in required:
            if key not in sample:
                msg = f"{path} lacks key {key!r}."
                raise KeyError(msg)
        motion_names.extend(str(name) for name in sample["motion_name"])
        for key in required:
            columns.setdefault(key, []).append(torch.as_tensor(sample[key]))
    data = {key: torch.cat(values) for key, values in columns.items()}
    rows = int(data["env_id"].numel())
    if len(motion_names) != rows:
        msg = f"motion_name count {len(motion_names)} != rows {rows}."
        raise ValueError(msg)

    future = data["expert_root_qpos_future"].float()
    future_valid = data["expert_root_qpos_future_valid"].bool()
    if future.shape[1] < chunk_horizon or future.shape[2] != ROOT_QPOS_WIDTH:
        msg = f"chunk window {tuple(future.shape)} incompatible with horizon {chunk_horizon}."
        raise ValueError(msg)

    encoder_record: dict[str, Any] | None = None
    if latent_source == "fsq_prequant":
        encoder_checkpoint = _resolve(cfg.latent.encoder_checkpoint)
        layout_encoder = cfg.latent.get("layout_check_encoder")
        per_row_latent, encoder_record = _fsq_prequant(
            encoder_checkpoint,
            future,
            data["z_target"].float(),
            parity_gate=layout_encoder is None,
        )
        if layout_encoder is not None:
            encoder_record.update(
                _validate_window_layout(
                    _resolve(layout_encoder), future, data["z_target"].float()
                )
            )
            encoder_record["cross_encoded_from"] = str(collection_dir)
    else:
        per_row_latent = data["z_target"].float()
    latent_target, latent_valid = _join_slots(
        per_row_latent,
        data["env_id"],
        data["episode_id"],
        data["control_step"],
        slots=slots,
        hold_steps=hold_steps,
    )

    goal_names = sorted(set(motion_names))
    name_to_id = {name: i for i, name in enumerate(goal_names)}
    goal_id = torch.tensor([name_to_id[name] for name in motion_names])

    states: dict[str, torch.Tensor] = {}
    for field in state_fields:
        tensor = data[field].float()
        states[field] = tensor.reshape(rows, -1)
    if bool(cfg.get("require_distinct_demonstration", False)):
        pair = [states[field] for field in state_fields]
        if len(pair) == 2 and torch.equal(pair[0], pair[1]):
            msg = (
                "require_distinct_demonstration: the two state fields are "
                "byte-identical — this collection has no expert-state rows."
            )
            raise ValueError(msg)

    table: dict[str, Any] = {
        "states": states,
        "chunk_target": future[:, :chunk_horizon].contiguous(),
        "chunk_valid": future_valid[:, :chunk_horizon].contiguous(),
        "latent_target": latent_target.contiguous(),
        "latent_valid": latent_valid.contiguous(),
        "goal_id": goal_id,
        "goal_names": goal_names,
        "provenance": {
            "collection_dir": str(collection_dir),
            "sample_files": {
                str(path.name): _sha256_file(path) for path in files
            },
            "collection_summary_sha256": _sha256_file(
                collection_dir / "summary.json"
            )
            if (collection_dir / "summary.json").is_file()
            else None,
            "latent_source": latent_source,
            "latent_slots": slots,
            "latent_hold_steps": hold_steps,
            "latent_slot_valid_fraction": [
                float(latent_valid[:, slot].float().mean())
                for slot in range(slots)
            ],
            "chunk_horizon": chunk_horizon,
            "encoder": encoder_record,
            "rows": rows,
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(table, output)
    fractions = table["provenance"]["latent_slot_valid_fraction"]
    print(
        f"[PASS] {rows} rows, {len(goal_names)} goals -> {output}\n"
        f"       latent slot valid fractions: {fractions}",
        flush=True,
    )


if __name__ == "__main__":
    main()
