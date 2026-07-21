# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from runtime_bootstrap import (
    assert_kit_not_loaded,
    config_contains_type_name,
    install_kit_import_guard,
)


strict_kitless = "--assert-kitless" in sys.argv
if strict_kitless:
    install_kit_import_guard()

from isaaclab_tasks.utils import add_launcher_args

parser = argparse.ArgumentParser(
    description="Train an offline high-level skill encoder with DiffSR."
)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Imitation-G1-Latent-v0",
    help="Isaac Lab task used only for expert trajectory sampling.",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of envs.")
parser.add_argument("--seed", type=int, default=None, help="Random seed.")
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Output directory. Defaults to logs/hl_skill_diffsr/<timestamp>.",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Optional checkpoint to load before training or eval-only diagnostics.",
)
parser.add_argument(
    "--eval_only",
    action="store_true",
    default=False,
    help="Load --checkpoint and run held-out diagnostics without updates.",
)
parser.add_argument("--horizon_steps", type=int, default=25, help="Macro horizon W.")
parser.add_argument(
    "--encoder_window_mode",
    type=str,
    default="full",
    choices=("full", "intermediate"),
    help=(
        "Future window visible to E_skill. 'full' keeps legacy s_{t+1:t+W}; "
        "'intermediate' hides target s_{t+W} and uses s_{t+1:t+W-1}."
    ),
)
parser.add_argument("--z_dim", type=int, default=256, help="Skill latent dimension.")
parser.add_argument(
    "--latent_mode",
    type=str,
    default="deterministic",
    choices=(
        "deterministic",
        "gaussian",
        "categorical",
        "gumbel_multicat",
        "gumbel",
        "fsq",
        "vq",
    ),
    help=(
        "Latent bottleneck design: 'deterministic' (z_norm), 'gaussian' (beta-VAE), "
        "'categorical' (multi-categorical straight-through codebook), 'gumbel' "
        "(Gumbel-softmax, annealed tau), 'fsq' (finite scalar quant), 'vq' (EMA VQ-VAE)."
    ),
)
parser.add_argument(
    "--reg_coeff",
    type=float,
    default=1.0e-3,
    help="Latent regularizer weight: L2 on z for 'deterministic', KL bottleneck "
    "for 'gaussian'/'categorical'.",
)
parser.add_argument(
    "--categorical_groups",
    type=int,
    default=8,
    help="Number of categorical groups G for --latent_mode categorical (must divide "
    "z_dim; per-group code dim = z_dim // G).",
)
parser.add_argument(
    "--categorical_categories",
    type=int,
    default=32,
    help="Categories per group (vocab size) for --latent_mode categorical. Independent "
    "of z_dim; larger = more capacity per group.",
)
parser.add_argument(
    "--gumbel_codebook_size", type=int, default=512, help="Codebook size for gumbel/."
)
parser.add_argument("--gumbel_tau_start", type=float, default=2.0)
parser.add_argument("--gumbel_tau_end", type=float, default=0.5)
parser.add_argument("--gumbel_tau_anneal_iters", type=int, default=2000)
parser.add_argument(
    "--gumbel_hard", action=argparse.BooleanOptionalAction, default=True
)
parser.add_argument(
    "--fsq_levels",
    type=int,
    nargs="+",
    default=[8, 8, 8, 5, 5],
    help="Per-dim levels for --latent_mode fsq (codebook size = product).",
)
parser.add_argument("--vq_codebook_size", type=int, default=512)
parser.add_argument("--vq_ema_decay", type=float, default=0.99)
parser.add_argument(
    "--vq_dead_code_reset_iters",
    type=int,
    default=0,
    help="Revive unused VQ codes every N updates (0 disables).",
)
parser.add_argument(
    "--diffsr_feature_dim",
    type=int,
    default=128,
    help="DiffSR spectral feature dimension.",
)
parser.add_argument(
    "--diffsr_embed_dim",
    type=int,
    default=512,
    help="DiffSR bilinear embedding dimension.",
)
parser.add_argument("--batch_size", type=int, default=8192, help="Training batch size.")
parser.add_argument("--num_updates", type=int, default=2000, help="Training updates.")
parser.add_argument("--log_interval", type=int, default=100, help="Log cadence.")
parser.add_argument(
    "--eval_batches",
    type=int,
    default=4,
    help="Held-out batches per diagnostic evaluation.",
)
parser.add_argument(
    "--eval_batch_size",
    type=int,
    default=None,
    help="Held-out diagnostic batch size. Defaults to --batch_size.",
)
parser.add_argument(
    "--train_split",
    type=str,
    default="train",
    choices=("all", "train", "eval"),
    help="Expert trajectory split used for optimizer updates.",
)
parser.add_argument(
    "--eval_split",
    type=str,
    default="eval",
    choices=("all", "train", "eval"),
    help="Expert trajectory split used for diagnostics.",
)
parser.add_argument(
    "--eval_trajectory_fraction",
    type=float,
    default=0.1,
    help="Fraction of nonempty expert trajectories reserved for eval split.",
)
parser.add_argument(
    "--trajectory_split_seed",
    type=int,
    default=0,
    help="Deterministic seed for train/eval trajectory split.",
)
parser.add_argument(
    "--grad_clip_norm",
    type=float,
    default=1.0,
    help="Global gradient max norm. Use <=0 to disable.",
)
parser.add_argument(
    "--reconstruction_norm_eps",
    type=float,
    default=1.0e-6,
    help="Variance floor for normalized reconstruction diagnostics.",
)
parser.add_argument(
    "--reconstruction_eval",
    action="store_true",
    default=False,
    help="Include stochastic DiffSR sampled reconstruction errors in eval metrics.",
)
parser.add_argument(
    "--window_probe_eval",
    action="store_true",
    default=False,
    help="Fit eval-only linear probes from frozen z/state to held-out future windows.",
)
parser.add_argument(
    "--window_probe_train_batches",
    type=int,
    default=4,
    help="Train batches for eval-only closed-form window probes.",
)
parser.add_argument(
    "--window_probe_eval_batches",
    type=int,
    default=None,
    help="Held-out batches for eval-only window probes. Defaults to --eval_batches.",
)
parser.add_argument(
    "--window_probe_ridge",
    type=float,
    default=1.0e-3,
    help="Ridge coefficient for eval-only window probes.",
)
parser.add_argument(
    "--cotrain_commander",
    action="store_true",
    default=False,
    help="Co-train a skill commander (BC to encoder z from state + language goal).",
)
parser.add_argument(
    "--commander_language_embeddings",
    type=str,
    default=None,
    help="Language-goal embedding table (.pt); required with --cotrain_commander.",
)
parser.add_argument(
    "--commander_lr", type=float, default=3.0e-4, help="Commander BC learning rate."
)
parser.add_argument(
    "--commander_hidden_dims",
    type=int,
    nargs="+",
    default=[1024, 512, 512],
    help="Commander MLP hidden widths.",
)
parser.add_argument(
    "--commander_cosine_loss_coeff",
    type=float,
    default=1.0,
    help="Weight on the (1 - cosine) commander BC term.",
)
parser.add_argument(
    "--commander_state_noise_std",
    type=float,
    default=0.0,
    help="Per-dim Gaussian noise on the commander state input (M3 robustness).",
)
parser.add_argument(
    "--logger_backend",
    type=str,
    default="none",
    help="Metrics backend for pretraining. Use 'wandb' to log to Weights & Biases, "
    "or 'none' to keep local JSONL/stdout logging only.",
)
parser.add_argument(
    "--wandb_project",
    type=str,
    default="G1-Imitation-HL-Skill-DiffSR",
    help="W&B project name when --logger_backend=wandb.",
)
parser.add_argument(
    "--wandb_entity",
    type=str,
    default=None,
    help="Optional W&B entity (team/user) when --logger_backend=wandb.",
)
parser.add_argument(
    "--wandb_group",
    type=str,
    default=None,
    help="Optional W&B group when --logger_backend=wandb.",
)
parser.add_argument(
    "--wandb_run_name",
    type=str,
    default=None,
    help="Optional W&B run name. Defaults to the run output directory name.",
)
parser.add_argument(
    "--wandb_mode",
    type=str,
    default="online",
    choices=("online", "offline", "disabled"),
    help="W&B mode passed to wandb.init when --logger_backend=wandb.",
)
parser.add_argument(
    "--assert-kitless",
    action="store_true",
    help="Require a Newton configuration that never imports or starts Kit.",
)
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

import gymnasium as gym
import isaaclab_imitation.tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401
import torch
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
)
from isaaclab.utils.io import dump_yaml
from isaaclab_imitation.envs.rlopt import IsaacLabWrapper
from isaaclab_tasks.utils import compute_kit_requirements, launch_simulation
from isaaclab_tasks.utils.hydra import hydra_task_config
from rlopt.agent import HighLevelSkillDiffSRConfig, HighLevelSkillDiffSRTrainer

AGENT_ENTRY_POINT = "rlopt_ipmd_bilinear_cfg_entry_point"


def _write_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def _wandb_enabled() -> bool:
    return str(args_cli.logger_backend).strip().lower() == "wandb"


def _init_wandb(log_dir: Path, config_payload: dict[str, Any]) -> Any:
    """Initialize a W&B run for pretraining, or return None if disabled."""
    if not _wandb_enabled():
        return None
    import wandb  # local import so the dependency is only needed for wandb runs

    run = wandb.init(
        project=args_cli.wandb_project,
        entity=args_cli.wandb_entity,
        group=args_cli.wandb_group,
        name=args_cli.wandb_run_name or log_dir.name,
        dir=str(log_dir),
        mode=args_cli.wandb_mode,
        config=config_payload,
    )
    print(f"[INFO] W&B logging enabled: {run.url if run else '(no run)'}")
    return run


def _wandb_log(run: Any, row: dict[str, Any]) -> None:
    """Log the numeric fields of a metrics row to W&B, stepped by 'update'."""
    if run is None:
        return
    step = int(row["update"]) if "update" in row else None
    payload = {
        key: value
        for key, value in row.items()
        if key != "update" and isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    if payload:
        run.log(payload, step=step)


def _build_trainer_config() -> HighLevelSkillDiffSRConfig:
    grad_clip_norm = None if args_cli.grad_clip_norm <= 0 else args_cli.grad_clip_norm
    config = HighLevelSkillDiffSRConfig(
        horizon_steps=args_cli.horizon_steps,
        encoder_window_mode=args_cli.encoder_window_mode,
        z_dim=args_cli.z_dim,
        latent_mode=args_cli.latent_mode,
        reg_coeff=args_cli.reg_coeff,
        categorical_groups=args_cli.categorical_groups,
        categorical_categories=args_cli.categorical_categories,
        gumbel_codebook_size=args_cli.gumbel_codebook_size,
        gumbel_tau_start=args_cli.gumbel_tau_start,
        gumbel_tau_end=args_cli.gumbel_tau_end,
        gumbel_tau_anneal_iters=args_cli.gumbel_tau_anneal_iters,
        gumbel_hard=args_cli.gumbel_hard,
        fsq_levels=tuple(args_cli.fsq_levels),
        vq_codebook_size=args_cli.vq_codebook_size,
        vq_ema_decay=args_cli.vq_ema_decay,
        vq_dead_code_reset_iters=args_cli.vq_dead_code_reset_iters,
        diffsr_feature_dim=args_cli.diffsr_feature_dim,
        diffsr_embed_dim=args_cli.diffsr_embed_dim,
        batch_size=args_cli.batch_size,
        num_updates=args_cli.num_updates,
        log_interval=args_cli.log_interval,
        eval_batches=args_cli.eval_batches,
        eval_batch_size=args_cli.eval_batch_size,
        train_split=args_cli.train_split,
        eval_split=args_cli.eval_split,
        eval_trajectory_fraction=args_cli.eval_trajectory_fraction,
        trajectory_split_seed=args_cli.trajectory_split_seed,
        grad_clip_norm=grad_clip_norm,
        reconstruction_norm_eps=args_cli.reconstruction_norm_eps,
        device=args_cli.device if args_cli.device is not None else "auto",
        cotrain_commander=bool(args_cli.cotrain_commander),
        commander_language_embeddings_path=(
            str(Path(args_cli.commander_language_embeddings).expanduser())
            if args_cli.commander_language_embeddings
            else ""
        ),
        commander_hidden_dims=tuple(int(d) for d in args_cli.commander_hidden_dims),
        commander_lr=args_cli.commander_lr,
        commander_cosine_loss_coeff=args_cli.commander_cosine_loss_coeff,
        commander_state_noise_std=args_cli.commander_state_noise_std,
    )
    config.validate()
    return config


def _run_dir() -> Path:
    if args_cli.output_dir is not None:
        return Path(args_cli.output_dir).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return Path("logs", "hl_skill_diffsr", timestamp).resolve()


def _run_training(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    _agent_cfg: object,
) -> None:
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
        random.seed(int(args_cli.seed))
        torch.manual_seed(int(args_cli.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args_cli.seed))
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    torch.set_float32_matmul_precision("high")
    log_dir = _run_dir()
    checkpoints_dir = log_dir / "checkpoints"
    metrics_path = log_dir / "metrics.jsonl"
    checkpoint_path = checkpoints_dir / "latest.pt"
    log_dir.mkdir(parents=True, exist_ok=True)

    trainer_config = _build_trainer_config()
    config_payload = {
        "task": args_cli.task,
        "num_envs": int(env_cfg.scene.num_envs),
        "seed": args_cli.seed,
        "checkpoint": args_cli.checkpoint,
        "eval_only": bool(args_cli.eval_only),
        "reconstruction_eval": bool(args_cli.reconstruction_eval),
        "window_probe_eval": bool(args_cli.window_probe_eval),
        "window_probe_train_batches": args_cli.window_probe_train_batches,
        "window_probe_eval_batches": args_cli.window_probe_eval_batches,
        "window_probe_ridge": args_cli.window_probe_ridge,
        "trainer": trainer_config.to_dict(),
        "command": " ".join(sys.orig_argv),
    }
    (log_dir / "config.yaml").write_text(
        yaml.safe_dump(config_payload, sort_keys=True),
        encoding="utf-8",
    )
    dump_yaml(str(log_dir / "env.yaml"), env_cfg)
    print(f"[INFO] Logging high-level DiffSR run to: {log_dir}")

    wandb_run = _init_wandb(log_dir, config_payload)

    env_cfg.log_dir = str(log_dir)
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        raise NotImplementedError("DirectMARLEnv is not supported by this script.")

    wrapped_env = IsaacLabWrapper(env)
    trainer = HighLevelSkillDiffSRTrainer(config=trainer_config, env=wrapped_env)
    try:
        if args_cli.checkpoint is not None:
            trainer.load_checkpoint(args_cli.checkpoint)
            print(f"[INFO] Loaded checkpoint: {Path(args_cli.checkpoint).resolve()}")

        if args_cli.eval_only:
            if args_cli.checkpoint is None:
                raise ValueError("--eval_only requires --checkpoint.")
            metrics = trainer.evaluate(
                prefix="train",
                include_reconstruction=bool(args_cli.reconstruction_eval),
            )
            if args_cli.window_probe_eval:
                metrics.update(
                    trainer.evaluate_window_probe(
                        train_batches=args_cli.window_probe_train_batches,
                        eval_batches=args_cli.window_probe_eval_batches,
                        prefix="train",
                        ridge=args_cli.window_probe_ridge,
                    )
                )
            row: dict[str, Any] = {
                "update": int(trainer.update),
                "eval_only": True,
                **metrics,
            }
            _write_jsonl(metrics_path, row)
            _wandb_log(wandb_run, row)
            print(json.dumps(row, indent=2, sort_keys=True))
            return

        def _log(row: dict[str, float | int]) -> None:
            _write_jsonl(metrics_path, dict(row))
            _wandb_log(wandb_run, dict(row))
            print(json.dumps(row, sort_keys=True))

        trainer.train(
            log_callback=_log,
            checkpoint_path=checkpoint_path,
            reconstruction_eval=bool(args_cli.reconstruction_eval),
        )
        final_metrics = trainer.evaluate(
            prefix="train",
            include_reconstruction=bool(args_cli.reconstruction_eval),
        )
        if args_cli.window_probe_eval:
            final_metrics.update(
                trainer.evaluate_window_probe(
                    train_batches=args_cli.window_probe_train_batches,
                    eval_batches=args_cli.window_probe_eval_batches,
                    prefix="train",
                    ridge=args_cli.window_probe_ridge,
                )
            )
        final_row: dict[str, Any] = {
            "update": int(trainer.update),
            "post_train_eval": True,
            **final_metrics,
        }
        _write_jsonl(metrics_path, final_row)
        _wandb_log(wandb_run, final_row)
        trainer.save_checkpoint(checkpoint_path)
        print(f"[INFO] Saved checkpoint: {checkpoint_path}")
        if trainer.commander is not None:
            commander_path = checkpoints_dir / "commander.pt"
            trainer.save_commander_checkpoint(
                commander_path, skill_checkpoint_path=str(checkpoint_path)
            )
            print(f"[INFO] Saved commander checkpoint: {commander_path}")
        print(json.dumps(final_row, indent=2, sort_keys=True))
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        wrapped_env.close()


@hydra_task_config(args_cli.task, AGENT_ENTRY_POINT)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: object,
) -> None:
    needs_kit, _, _ = compute_kit_requirements(env_cfg, args_cli)
    if args_cli.assert_kitless:
        if needs_kit or not config_contains_type_name(env_cfg, "NewtonCfg"):
            raise RuntimeError(
                "--assert-kitless requires a resolved NewtonCfg with no Kit cameras or Kit visualizer."
            )
        assert_kit_not_loaded()
        print("[INFO] Strict kit-less Newton pretraining runtime validated.")
    if os.environ.get("ISAACLAB_SPLIT_RUNTIME") == "1" and needs_kit:
        raise RuntimeError(
            "The split runtime cannot start Kit from the offline skill-pretraining entrypoint. "
            "Use Newton with --assert-kitless on compute-only GPUs."
        )
    with launch_simulation(env_cfg, args_cli):
        _run_training(env_cfg, agent_cfg)


if __name__ == "__main__":
    main()
