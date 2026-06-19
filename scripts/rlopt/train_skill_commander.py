# ruff: noqa: E402
"""Train a language-conditioned skill generator (System 1) by distillation.

Loads a frozen high-level skill encoder (from ``train_hl_skill_diffsr.py``) and a
language-goal embedding table (from ``build_language_goal_embeddings.py``), then
trains a generator that maps ``(current_state, language_goal) -> z`` to match the
encoder's skill latent. Evaluation uses held-out trajectory names.

Example:
    pixi run -e isaaclab python scripts/rlopt/train_skill_commander.py \
        --task Isaac-Imitation-G1-Latent-v0 --num_envs 16 \
        --skill_checkpoint logs/hl_skill_diffsr/<run>/checkpoints/latest.pt \
        --language_embeddings data/lafan1/language/g1_lafan1_name_embeddings.pt
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Train a language-conditioned skill generator by distillation."
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
    help="Output directory. Defaults to logs/skill_commander/<timestamp>.",
)
parser.add_argument(
    "--skill_checkpoint",
    type=str,
    default=None,
    help="Frozen high-level skill encoder checkpoint to distill from.",
)
parser.add_argument(
    "--language_embeddings",
    type=str,
    default=None,
    help="Language goal embedding table (.pt) keyed by motion name.",
)
parser.add_argument(
    "--no_language",
    action="store_true",
    default=False,
    help="Train a state-only planner with a zero-width language condition.",
)
parser.add_argument(
    "--state_history_steps",
    type=int,
    default=None,
    help="Past expert macro states to flatten with the current state for planner input.",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Optional generator checkpoint to load before training or eval-only.",
)
parser.add_argument(
    "--eval_only",
    action="store_true",
    default=False,
    help="Load --checkpoint and run held-out diagnostics without updates.",
)
parser.add_argument(
    "--generator_hidden_dims",
    type=int,
    nargs="+",
    default=None,
    help="Generator MLP hidden layer widths.",
)
parser.add_argument(
    "--planner_type",
    type=str,
    default=None,
    choices=("mlp", "flow_matching", "diffusion_policy"),
    help="Planner architecture for (state, language) -> skill z.",
)
parser.add_argument(
    "--flow_num_inference_steps",
    type=int,
    default=None,
    help="Euler integration steps for flow-matching planner inference.",
)
parser.add_argument(
    "--flow_time_embed_dim",
    type=int,
    default=None,
    help="Sinusoidal time embedding width for flow-matching planner.",
)
parser.add_argument(
    "--flow_train_noise_std",
    type=float,
    default=None,
    help="Gaussian source std for flow-matching training.",
)
parser.add_argument(
    "--flow_inference_noise_std",
    type=float,
    default=None,
    help="Gaussian source std for flow-matching inference.",
)
parser.add_argument(
    "--diffusion_num_train_timesteps",
    type=int,
    default=None,
    help="DDPM training horizon for diffusion-policy planner.",
)
parser.add_argument(
    "--diffusion_num_inference_steps",
    type=int,
    default=None,
    help="DDPM reverse steps for diffusion-policy planner inference.",
)
parser.add_argument(
    "--diffusion_time_embed_dim",
    type=int,
    default=None,
    help="Sinusoidal timestep embedding width for diffusion-policy planner.",
)
parser.add_argument(
    "--diffusion_beta_schedule",
    type=str,
    default=None,
    choices=("linear", "scaled_linear", "squaredcos_cap_v2", "sigmoid"),
    help="Diffusers DDPMScheduler beta schedule for diffusion-policy planner.",
)
parser.add_argument(
    "--diffusion_prediction_type",
    type=str,
    default=None,
    choices=("epsilon",),
    help="DDPM prediction target for diffusion-policy planner.",
)
parser.add_argument(
    "--diffusion_inference_scheduler",
    type=str,
    default=None,
    choices=("ddpm", "ddim"),
    help="Reverse-process scheduler for diffusion-policy inference.",
)
parser.add_argument(
    "--diffusion_ddim_eta",
    type=float,
    default=None,
    help="DDIM eta for diffusion-policy inference; 0.0 is deterministic.",
)
parser.add_argument(
    "--diffusion_inference_noise_std",
    type=float,
    default=None,
    help="Gaussian source std for diffusion-policy inference.",
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
parser.add_argument("--lr", type=float, default=3.0e-4, help="Generator learning rate.")
parser.add_argument(
    "--weight_decay", type=float, default=0.0, help="AdamW weight decay."
)
parser.add_argument(
    "--grad_clip_norm",
    type=float,
    default=1.0,
    help="Global gradient max norm. Use <=0 to disable.",
)
parser.add_argument(
    "--cosine_loss_coeff",
    type=float,
    default=1.0,
    help="Weight on the (1 - cosine) distillation term.",
)
parser.add_argument(
    "--z_norm_coeff",
    type=float,
    default=1.0e-4,
    help="Small generated-z scale regularizer coefficient.",
)
parser.add_argument(
    "--state_noise_std",
    type=float,
    default=0.0,
    help="Per-dim-scaled Gaussian noise on the generator state input (M3 robustness).",
)
parser.add_argument(
    "--state_feature_dropout_prob",
    type=float,
    default=None,
    help=(
        "Row probability for corrupting selected macro-state features during "
        "commander training."
    ),
)
parser.add_argument(
    "--state_feature_dropout_terms",
    type=str,
    nargs="+",
    default=None,
    help=(
        "Macro-state feature names to corrupt, e.g. expert_motion. Use all to "
        "corrupt the full state."
    ),
)
parser.add_argument(
    "--state_feature_dropout_mode",
    type=str,
    default=None,
    choices=("shuffle", "zero", "batch_mean"),
    help="Replacement mode for --state_feature_dropout_prob.",
)
parser.add_argument(
    "--state_feature_dropout_warmup_updates",
    type=int,
    default=None,
    help="Delay state-feature dropout until this many trainer updates have completed.",
)
parser.add_argument(
    "--language_contrastive_coeff",
    type=float,
    default=None,
    help=(
        "Optional weight for the endpoint loss that makes a different language "
        "embedding score worse than the correct one."
    ),
)
parser.add_argument(
    "--language_contrastive_margin",
    type=float,
    default=None,
    help="Cosine margin for --language_contrastive_coeff.",
)
parser.add_argument(
    "--language_contrastive_warmup_updates",
    type=int,
    default=None,
    help=(
        "Delay endpoint language-contrastive loss until this many trainer "
        "updates have completed."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

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
from isaaclab_tasks.utils.hydra import hydra_task_config
from rlopt.agent import SkillCommanderConfig, SkillCommanderTrainer

AGENT_ENTRY_POINT = "rlopt_ipmd_bilinear_cfg_entry_point"


def _write_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def _build_trainer_config() -> SkillCommanderConfig:
    grad_clip_norm = None if args_cli.grad_clip_norm <= 0 else args_cli.grad_clip_norm
    values: dict[str, Any] = {}
    if args_cli.checkpoint is not None:
        checkpoint = torch.load(
            Path(args_cli.checkpoint).expanduser(),
            map_location="cpu",
            weights_only=False,
        )
        values.update(checkpoint.get("config", {}))
        values.setdefault(
            "skill_checkpoint_path", checkpoint.get("skill_checkpoint_path", "")
        )
        values.setdefault(
            "language_embeddings_path",
            checkpoint.get("language_embeddings_path", ""),
        )

    if args_cli.skill_checkpoint is not None:
        values["skill_checkpoint_path"] = str(
            Path(args_cli.skill_checkpoint).expanduser()
        )
    if args_cli.no_language:
        values["condition_on_language"] = False
    if args_cli.language_embeddings is not None:
        values["language_embeddings_path"] = str(
            Path(args_cli.language_embeddings).expanduser()
        )
    if args_cli.state_history_steps is not None:
        values["state_history_steps"] = int(args_cli.state_history_steps)
    if args_cli.planner_type is not None:
        values["planner_type"] = args_cli.planner_type
    if args_cli.generator_hidden_dims is not None:
        values["generator_hidden_dims"] = tuple(
            int(dim) for dim in args_cli.generator_hidden_dims
        )
    if args_cli.state_feature_dropout_terms is not None:
        values["state_feature_dropout_terms"] = tuple(
            str(term) for term in args_cli.state_feature_dropout_terms
        )
    for cli_name, config_name in (
        ("flow_num_inference_steps", "flow_num_inference_steps"),
        ("flow_time_embed_dim", "flow_time_embed_dim"),
        ("flow_train_noise_std", "flow_train_noise_std"),
        ("flow_inference_noise_std", "flow_inference_noise_std"),
        ("diffusion_num_train_timesteps", "diffusion_num_train_timesteps"),
        ("diffusion_num_inference_steps", "diffusion_num_inference_steps"),
        ("diffusion_time_embed_dim", "diffusion_time_embed_dim"),
        ("diffusion_beta_schedule", "diffusion_beta_schedule"),
        ("diffusion_prediction_type", "diffusion_prediction_type"),
        ("diffusion_inference_scheduler", "diffusion_inference_scheduler"),
        ("diffusion_ddim_eta", "diffusion_ddim_eta"),
        ("diffusion_inference_noise_std", "diffusion_inference_noise_std"),
        ("state_feature_dropout_prob", "state_feature_dropout_prob"),
        ("state_feature_dropout_mode", "state_feature_dropout_mode"),
        (
            "state_feature_dropout_warmup_updates",
            "state_feature_dropout_warmup_updates",
        ),
        ("language_contrastive_coeff", "language_contrastive_coeff"),
        ("language_contrastive_margin", "language_contrastive_margin"),
        (
            "language_contrastive_warmup_updates",
            "language_contrastive_warmup_updates",
        ),
    ):
        value = getattr(args_cli, cli_name)
        if value is not None:
            values[config_name] = value

    values.update(
        {
            "batch_size": args_cli.batch_size,
            "num_updates": args_cli.num_updates,
            "log_interval": args_cli.log_interval,
            "eval_batches": args_cli.eval_batches,
            "eval_batch_size": args_cli.eval_batch_size,
            "train_split": args_cli.train_split,
            "eval_split": args_cli.eval_split,
            "eval_trajectory_fraction": args_cli.eval_trajectory_fraction,
            "trajectory_split_seed": args_cli.trajectory_split_seed,
            "lr": args_cli.lr,
            "weight_decay": args_cli.weight_decay,
            "grad_clip_norm": grad_clip_norm,
            "cosine_loss_coeff": args_cli.cosine_loss_coeff,
            "z_norm_coeff": args_cli.z_norm_coeff,
            "state_noise_std": args_cli.state_noise_std,
            "device": args_cli.device if args_cli.device is not None else "auto",
        }
    )

    config = SkillCommanderConfig.from_dict(values)
    config.validate()
    return config


def _run_dir() -> Path:
    if args_cli.output_dir is not None:
        return Path(args_cli.output_dir).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return Path("logs", "skill_commander", timestamp).resolve()


@hydra_task_config(args_cli.task, AGENT_ENTRY_POINT)
def main(
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
        "trainer": trainer_config.to_dict(),
        "command": " ".join(sys.orig_argv),
    }
    (log_dir / "config.yaml").write_text(
        yaml.safe_dump(config_payload, sort_keys=True),
        encoding="utf-8",
    )
    dump_yaml(str(log_dir / "env.yaml"), env_cfg)
    print(f"[INFO] Logging language skill generator run to: {log_dir}")

    env_cfg.log_dir = str(log_dir)
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        raise NotImplementedError("DirectMARLEnv is not supported by this script.")

    wrapped_env = IsaacLabWrapper(env)
    trainer = SkillCommanderTrainer(config=trainer_config, env=wrapped_env)
    try:
        if args_cli.checkpoint is not None:
            state = torch.load(
                Path(args_cli.checkpoint).expanduser(),
                map_location=trainer.device,
                weights_only=False,
            )
            trainer.generator.load_state_dict(state["generator_state_dict"])
            trainer.update = int(state.get("update", 0))
            print(f"[INFO] Loaded generator checkpoint: {args_cli.checkpoint}")

        if args_cli.eval_only:
            if args_cli.checkpoint is None:
                raise ValueError("--eval_only requires --checkpoint.")
            metrics = trainer.evaluate(prefix="eval")
            row: dict[str, Any] = {
                "update": int(trainer.update),
                "eval_only": True,
                **metrics,
            }
            _write_jsonl(metrics_path, row)
            print(json.dumps(row, indent=2, sort_keys=True))
            return

        def _log(row: dict[str, float | int]) -> None:
            _write_jsonl(metrics_path, dict(row))
            print(json.dumps(row, sort_keys=True))

        trainer.train(log_callback=_log, checkpoint_path=checkpoint_path)
        final_metrics = trainer.evaluate(prefix="eval")
        final_row: dict[str, Any] = {
            "update": int(trainer.update),
            "post_train_eval": True,
            **final_metrics,
        }
        _write_jsonl(metrics_path, final_row)
        trainer.save_checkpoint(checkpoint_path)
        print(f"[INFO] Saved checkpoint: {checkpoint_path}")
        print(json.dumps(final_row, indent=2, sort_keys=True))
    finally:
        wrapped_env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
