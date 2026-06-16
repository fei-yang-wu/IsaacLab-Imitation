# ruff: noqa: E402
"""Train a language-conditioned skill generator (System 2) by distillation.

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
    required=True,
    help="Frozen high-level skill encoder checkpoint to distill from.",
)
parser.add_argument(
    "--language_embeddings",
    type=str,
    required=True,
    help="Language goal embedding table (.pt) keyed by motion name.",
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
    default=[1024, 512, 512],
    help="Generator MLP hidden layer widths.",
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
    config = SkillCommanderConfig(
        skill_checkpoint_path=str(Path(args_cli.skill_checkpoint).expanduser()),
        language_embeddings_path=str(Path(args_cli.language_embeddings).expanduser()),
        generator_hidden_dims=tuple(int(dim) for dim in args_cli.generator_hidden_dims),
        batch_size=args_cli.batch_size,
        num_updates=args_cli.num_updates,
        log_interval=args_cli.log_interval,
        eval_batches=args_cli.eval_batches,
        eval_batch_size=args_cli.eval_batch_size,
        train_split=args_cli.train_split,
        eval_split=args_cli.eval_split,
        eval_trajectory_fraction=args_cli.eval_trajectory_fraction,
        trajectory_split_seed=args_cli.trajectory_split_seed,
        lr=args_cli.lr,
        weight_decay=args_cli.weight_decay,
        grad_clip_norm=grad_clip_norm,
        cosine_loss_coeff=args_cli.cosine_loss_coeff,
        z_norm_coeff=args_cli.z_norm_coeff,
        state_noise_std=args_cli.state_noise_std,
        device=args_cli.device if args_cli.device is not None else "auto",
    )
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
