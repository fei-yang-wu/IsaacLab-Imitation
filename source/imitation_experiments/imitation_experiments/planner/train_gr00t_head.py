"""Train the verbatim GR00T N1.7 action head on G1 planner samples.

Runs in the ``gr00t`` Pixi environment with our training loop; the model class
comes untouched from ``external/Isaac-GR00T``. All paths and choices live in
the Hydra config — no pinned paths in code. Target modes:

- ``chunk``: regress the 30-frame expert ``root_qpos`` lookahead
  (``[H=30, D=38]``);
- ``latent``: regress consecutive per-publication latents
  (``[H=slots, D=z]``), e.g. 3 x z256 or 3 x FSQ-prequant-64. The table's
  ``latent_target``/``latent_valid`` come from
  ``prepare_gr00t_dataset.py``.

Data modes select the table state field: the achieved-robot history
(``oracle_rollout_state_history``) or the expert/mocap history
(``demonstration_state_history`` from a collection that stored true
expert-state rows).

Run from the repository root, e.g.:

    pixi run -e gr00t python -m imitation_experiments.planner.train_gr00t_head \
        --config-path <campaign>/conf --config-name train_z256_rollout
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from imitation_experiments.paths import REPO_ROOT
from imitation_experiments.planner.gr00t_head import (
    BACKBONE_EMBEDDING_DIM,
    DEBUG_TRUNK_CONFIG,
    N17_TRUNK_CONFIG,
    PLANNER_STATE_HISTORY,
    PLANNER_STATE_WIDTH,
    build_batch,
    build_g1_head_config,
    compute_quantile_stats,
    filtered_pretrained_load,
    gr00t_submodule_commit,
    import_head_classes,
    normalize_minmax,
    save_provenance,
)

TRUNK_PRESETS = {"debug": DEBUG_TRUNK_CONFIG, "n17": N17_TRUNK_CONFIG}


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def _load_table(cfg: DictConfig) -> dict[str, torch.Tensor | Any]:
    """Assemble training rows from a prepared table + goal-feature cache."""
    table = torch.load(
        _resolve(cfg.data.table), map_location="cpu", weights_only=False
    )
    state_field = str(cfg.data.state_field)
    if state_field not in table["states"]:
        msg = (
            f"state field {state_field!r} not in table "
            f"(has {sorted(table['states'])})."
        )
        raise KeyError(msg)
    state = table["states"][state_field].float()
    rows = int(state.shape[0])
    state = state.reshape(rows, PLANNER_STATE_HISTORY, PLANNER_STATE_WIDTH)

    target_mode = str(cfg.target)
    if target_mode == "chunk":
        action = table["chunk_target"].float()
        valid = table["chunk_valid"].bool()
    elif target_mode == "latent":
        action = table["latent_target"].float()
        valid = table["latent_valid"].bool()
    else:
        msg = f"target must be chunk|latent, got {target_mode!r}."
        raise ValueError(msg)

    goal_table = torch.load(
        _resolve(cfg.goal_features), map_location="cpu", weights_only=False
    )
    name_to_index = {name: i for i, name in enumerate(goal_table["goal_names"])}
    missing = [
        name for name in table["goal_names"] if name not in name_to_index
    ]
    if missing:
        msg = f"goals missing from the feature cache: {missing}"
        raise KeyError(msg)
    remap = torch.tensor(
        [name_to_index[name] for name in table["goal_names"]], dtype=torch.long
    )
    goal_id = remap[table["goal_id"]]

    keep = valid.any(dim=1)
    return {
        "state": state[keep],
        "action": action[keep],
        "valid": valid[keep],
        "goal_id": goal_id[keep],
        "features": goal_table["features"].float(),
        "feature_mask": goal_table["attention_mask"].bool(),
        "table_provenance": table.get("provenance"),
    }


def _synthetic_rows(cfg: DictConfig) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(int(cfg.run.seed))
    rows = int(cfg.data.synthetic_rows)
    horizon = int(cfg.data.synthetic_horizon)
    action_dim = int(cfg.data.synthetic_action_dim)
    num_goals, seq = 4, 12
    return {
        "state": torch.randn(
            rows, PLANNER_STATE_HISTORY, PLANNER_STATE_WIDTH, generator=generator
        ),
        "action": torch.randn(rows, horizon, action_dim, generator=generator),
        "valid": torch.ones(rows, horizon, dtype=torch.bool),
        "goal_id": torch.randint(0, num_goals, (rows,), generator=generator),
        "features": torch.randn(
            num_goals, seq, BACKBONE_EMBEDDING_DIM, generator=generator
        ),
        "feature_mask": torch.ones(num_goals, seq, dtype=torch.bool),
        "table_provenance": None,
    }


def _init_wandb(cfg: DictConfig, extra: dict[str, Any]) -> Any | None:
    if not bool(cfg.wandb.enabled):
        return None
    import wandb  # noqa: PLC0415

    config = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(config, dict)
    config.update(extra)
    return wandb.init(
        project=str(cfg.wandb.project),
        group=str(cfg.wandb.group),
        name=None if cfg.wandb.name is None else str(cfg.wandb.name),
        tags=[str(tag) for tag in cfg.wandb.tags],
        config=config,
        dir=str(_resolve(cfg.output_dir)),
    )


@hydra.main(version_base="1.3", config_path="conf_gr00t", config_name="base_train")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(int(cfg.run.seed))
    device = torch.device(str(cfg.run.device))
    output_dir = _resolve(cfg.output_dir)
    synthetic = int(cfg.data.get("synthetic_rows", 0)) > 0
    data = _synthetic_rows(cfg) if synthetic else _load_table(cfg)
    rows = int(data["state"].shape[0])
    horizon = int(data["action"].shape[1])
    action_dim = int(data["action"].shape[2])
    print(
        f"[train] {rows} rows, target {cfg.target if not synthetic else 'synthetic'} "
        f"[H={horizon}, D={action_dim}], preset {cfg.preset}"
    )

    # Quantile normalization (GR00T use_percentiles convention), stored with
    # the checkpoint; state uses the same scheme.
    action_q01, action_q99 = compute_quantile_stats(data["action"], data["valid"])
    state_q01, state_q99 = compute_quantile_stats(data["state"])
    data["action"] = normalize_minmax(data["action"], action_q01, action_q99)
    data["state"] = normalize_minmax(data["state"], state_q01, state_q99)

    trunk = dict(TRUNK_PRESETS[str(cfg.preset)])
    bundle = None
    if cfg.pretrained_bundle is not None:
        bundle = torch.load(
            _resolve(cfg.pretrained_bundle), map_location="cpu", weights_only=False
        )
        source_config = bundle.get("source_config", {})
        for key in (
            "hidden_size",
            "input_embedding_dim",
            "backbone_embedding_dim",
            "diffusion_model_cfg",
            "vl_self_attention_cfg",
        ):
            if key in source_config:
                trunk[key] = source_config[key]
        print("[train] trunk architecture taken from the pretrained bundle")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    config = build_g1_head_config(
        trunk=trunk,
        action_horizon=horizon,
        max_action_dim=action_dim,
        state_dropout_prob=float(cfg.model.state_dropout_prob),
        num_inference_timesteps=int(cfg.model.num_inference_timesteps),
    )
    _, head_cls = import_head_classes()
    head = head_cls(config).to(device)
    if bool(cfg.model.gradient_checkpointing):
        head.model.enable_gradient_checkpointing()

    load_manifest: dict | None = None
    if bundle is not None:
        load_manifest = filtered_pretrained_load(head, bundle["trunk_state_dict"])
        print(
            f"[train] warm start: kept {load_manifest['num_kept_params'] / 1e6:.1f}M "
            f"params, fresh {load_manifest['num_fresh_params'] / 1e6:.1f}M"
        )

    total_updates = int(cfg.optim.num_updates)
    warmup_updates = max(1, int(total_updates * float(cfg.optim.warmup_ratio)))

    def _make_optimizer() -> torch.optim.AdamW:
        params = [p for p in head.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            params,
            lr=float(cfg.optim.lr),
            weight_decay=float(cfg.optim.weight_decay),
            fused=device.type == "cuda",
        )

    def _lr_lambda(update_index: int) -> float:
        if str(cfg.optim.lr_schedule) == "constant":
            return 1.0
        if update_index < warmup_updates:
            return (update_index + 1) / warmup_updates
        progress = (update_index - warmup_updates) / max(
            total_updates - warmup_updates, 1
        )
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    def _make_scheduler(
        opt: torch.optim.AdamW, last_update: int
    ) -> torch.optim.lr_scheduler.LambdaLR:
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
        for _ in range(last_update):
            scheduler.step()
        return scheduler

    stage_a = int(cfg.stage_a_updates)
    if stage_a > 0 and bundle is None:
        msg = "stage_a_updates requires pretrained_bundle."
        raise ValueError(msg)
    if stage_a > 0:
        head.set_trainable_parameters(
            tune_projector=True, tune_diffusion_model=False, tune_vlln=False
        )
    optimizer = _make_optimizer()
    scheduler = _make_scheduler(optimizer, 0)

    run = _init_wandb(
        cfg,
        {
            "rows": rows,
            "action_horizon": horizon,
            "action_dim": action_dim,
            "gr00t_submodule_commit": gr00t_submodule_commit(),
        },
    )

    generator = torch.Generator().manual_seed(int(cfg.run.seed) + 1)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = output_dir / "checkpoints"
    autocast_dtype = torch.bfloat16 if bool(cfg.run.bf16) else None
    batch_size = int(cfg.optim.batch_size)
    start = time.time()
    for update in range(1, total_updates + 1):
        if stage_a > 0 and update == stage_a + 1:
            head.set_trainable_parameters(
                tune_projector=True, tune_diffusion_model=True, tune_vlln=True
            )
            optimizer = _make_optimizer()
            scheduler = _make_scheduler(optimizer, update - 1)
            print(f"[train] update {update}: stage B — trunk unfrozen")
        index = torch.randint(0, rows, (batch_size,), generator=generator)
        goal = data["goal_id"][index]
        backbone_output, action_input = build_batch(
            state=data["state"][index].to(device),
            action=data["action"][index].to(device),
            action_mask=data["valid"][index].to(device),
            language_features=data["features"][goal].to(device),
            language_attention_mask=data["feature_mask"][goal].to(device),
        )
        head.train()
        optimizer.zero_grad(set_to_none=True)
        if autocast_dtype is not None:
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                out = head(backbone_output, action_input)
                loss = out["loss"]
        else:
            out = head(backbone_output, action_input)
            loss = out["loss"]
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in head.parameters() if p.requires_grad],
            float(cfg.optim.grad_clip),
        )
        optimizer.step()
        scheduler.step()
        if update % int(cfg.run.log_interval) == 0 or update == 1:
            rate = update / max(time.time() - start, 1.0e-9)
            lr_now = scheduler.get_last_lr()[0]
            print(
                f"[train] update {update}/{total_updates} "
                f"loss={float(loss.detach()):.6f} grad={float(grad_norm):.3f} "
                f"lr={lr_now:.2e} ({rate:.2f} up/s)",
                flush=True,
            )
            if run is not None:
                run.log(
                    {
                        "loss": float(loss.detach()),
                        "grad_norm": float(grad_norm),
                        "lr": lr_now,
                        "updates_per_s": rate,
                    },
                    step=update,
                )
        if update % int(cfg.run.checkpoint_interval) == 0 or update == total_updates:
            checkpoints.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "head_state_dict": head.state_dict(),
                    "trunk_config": trunk,
                    "action_horizon": horizon,
                    "action_dim": action_dim,
                    "target_mode": None if synthetic else str(cfg.target),
                    "state_field": None if synthetic else str(cfg.data.state_field),
                    "normalization": {
                        "action_q01": action_q01,
                        "action_q99": action_q99,
                        "state_q01": state_q01,
                        "state_q99": state_q99,
                    },
                    "pretrained_load_manifest": load_manifest,
                    "table_provenance": data.get("table_provenance"),
                    "config": OmegaConf.to_container(cfg, resolve=True),
                    "update": update,
                },
                checkpoints / f"update_{update:07d}.pt",
            )
    save_provenance(
        output_dir / "training_provenance.json",
        {
            "gr00t_submodule_commit": gr00t_submodule_commit(),
            "config": OmegaConf.to_container(cfg, resolve=True),
            "rows": rows,
            "action_horizon": horizon,
            "action_dim": action_dim,
            "pretrained_load_kept_params": None
            if load_manifest is None
            else load_manifest["num_kept_params"],
            "synthetic": synthetic,
        },
    )
    if run is not None:
        run.finish()
    print(f"[PASS] training complete -> {output_dir}", flush=True)


if __name__ == "__main__":
    main()
