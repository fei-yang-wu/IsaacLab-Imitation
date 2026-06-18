#!/usr/bin/env python3
# ruff: noqa: E402
"""Evaluate the language commander on expert macro states (M1-only).

This diagnostic removes low-level policy dynamics from the loop. It samples
expert macro transitions from the Isaac task, computes target z with the frozen
high-level skill encoder, and compares the loaded language commander prediction
from ``(expert_state, language_embedding)`` against that target.
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
    description="Evaluate a trained SkillCommander on expert-state M1 inputs."
)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Imitation-G1-Latent-v0",
    help="Isaac Lab task used for expert trajectory sampling.",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of envs.")
parser.add_argument("--seed", type=int, default=0, help="Random seed.")
parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="SkillCommander checkpoint to evaluate.",
)
parser.add_argument(
    "--skill_checkpoint",
    type=str,
    default=None,
    help="Override source high-level skill checkpoint path from the commander ckpt.",
)
parser.add_argument(
    "--language_embeddings",
    type=str,
    default=None,
    help="Override language embedding table path from the commander ckpt.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Output directory. Defaults to logs/skill_commander_m1_eval/<timestamp>.",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=4096,
    help="Batch size for aggregate split metrics.",
)
parser.add_argument(
    "--eval_batches",
    type=int,
    default=8,
    help="Number of batches per aggregate split.",
)
parser.add_argument(
    "--splits",
    type=str,
    nargs="+",
    default=["all", "train", "eval"],
    choices=("all", "train", "eval"),
    help="Expert trajectory splits to evaluate.",
)
parser.add_argument(
    "--eval_trajectory_fraction",
    type=float,
    default=None,
    help="Override held-out trajectory fraction from the checkpoint config.",
)
parser.add_argument(
    "--trajectory_split_seed",
    type=int,
    default=None,
    help="Override train/eval split seed from the checkpoint config.",
)
parser.add_argument(
    "--per_trajectory",
    action="store_true",
    default=False,
    help="Also evaluate each selected trajectory rank separately.",
)
parser.add_argument(
    "--trajectory_ranks",
    type=str,
    default="all",
    help="Ranks for --per_trajectory, e.g. all, 0-9, or 0,3,7.",
)
parser.add_argument(
    "--per_trajectory_batch_size",
    type=int,
    default=512,
    help="Batch size for each per-trajectory metric row.",
)
parser.add_argument(
    "--per_trajectory_batches",
    type=int,
    default=4,
    help="Number of batches per trajectory rank.",
)
parser.add_argument(
    "--flow_inference_noise_std",
    type=float,
    default=None,
    help="Override flow-matching inference noise std from the checkpoint config.",
)
parser.add_argument(
    "--flow_num_inference_steps",
    type=int,
    default=None,
    help="Override flow-matching inference steps from the checkpoint config.",
)
parser.add_argument(
    "--diffusion_num_inference_steps",
    type=int,
    default=None,
    help="Override diffusion-policy inference steps from the checkpoint config.",
)
parser.add_argument(
    "--diffusion_inference_scheduler",
    type=str,
    default=None,
    choices=("ddpm", "ddim"),
    help="Override diffusion-policy inference scheduler from the checkpoint config.",
)
parser.add_argument(
    "--diffusion_ddim_eta",
    type=float,
    default=None,
    help="Override diffusion-policy DDIM eta from the checkpoint config.",
)
parser.add_argument(
    "--diffusion_inference_noise_std",
    type=float,
    default=None,
    help="Override diffusion-policy inference noise std from the checkpoint config.",
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
import torch.nn.functional as F
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


def _positive_int(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}.")
    return value


def _parse_rank_spec(spec: str, *, count: int) -> list[int]:
    spec = str(spec).strip().lower()
    if spec in ("", "all"):
        return list(range(count))
    ranks: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_s, end_s = chunk.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise ValueError(f"Invalid descending rank range: {chunk!r}.")
            ranks.extend(range(start, end + 1))
        else:
            ranks.append(int(chunk))
    unique = list(dict.fromkeys(ranks))
    bad = [rank for rank in unique if rank < 0 or rank >= count]
    if bad:
        raise ValueError(f"Ranks out of range [0, {count - 1}]: {bad}.")
    return unique


def _write_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def _run_dir() -> Path:
    if args_cli.output_dir is not None:
        return Path(args_cli.output_dir).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return Path("logs", "skill_commander_m1_eval", timestamp).resolve()


def _trainer_config_from_checkpoint(checkpoint: dict[str, Any]) -> SkillCommanderConfig:
    values = dict(checkpoint.get("config", {}))
    values.setdefault("skill_checkpoint_path", checkpoint.get("skill_checkpoint_path", ""))
    values.setdefault(
        "language_embeddings_path", checkpoint.get("language_embeddings_path", "")
    )
    if args_cli.skill_checkpoint is not None:
        values["skill_checkpoint_path"] = str(Path(args_cli.skill_checkpoint).expanduser())
    if args_cli.language_embeddings is not None:
        values["language_embeddings_path"] = str(
            Path(args_cli.language_embeddings).expanduser()
        )
    values["batch_size"] = _positive_int("batch_size", args_cli.batch_size)
    values["eval_batches"] = _positive_int("eval_batches", args_cli.eval_batches)
    values["eval_batch_size"] = _positive_int("batch_size", args_cli.batch_size)
    if args_cli.eval_trajectory_fraction is not None:
        values["eval_trajectory_fraction"] = float(args_cli.eval_trajectory_fraction)
    if args_cli.trajectory_split_seed is not None:
        values["trajectory_split_seed"] = int(args_cli.trajectory_split_seed)
    for cli_name in (
        "flow_inference_noise_std",
        "flow_num_inference_steps",
        "diffusion_num_inference_steps",
        "diffusion_inference_scheduler",
        "diffusion_ddim_eta",
        "diffusion_inference_noise_std",
    ):
        value = getattr(args_cli, cli_name)
        if value is not None:
            values[cli_name] = value
    if args_cli.device is not None:
        values["device"] = str(args_cli.device)
    config = SkillCommanderConfig.from_dict(values)
    config.validate()
    return config


def _phrase_map(table: dict[str, Any]) -> dict[str, str]:
    names = table.get("names")
    phrases = table.get("phrases")
    if not isinstance(names, list) or not isinstance(phrases, list):
        return {}
    return {str(name): str(phrase) for name, phrase in zip(names, phrases)}


@torch.no_grad()
def _evaluate_batches(
    trainer: SkillCommanderTrainer,
    sampler,
    *,
    batch_size: int,
    num_batches: int,
    prefix: str,
) -> dict[str, float]:
    batch_size = _positive_int("batch_size", batch_size)
    num_batches = _positive_int("num_batches", num_batches)
    was_training = trainer.generator.training
    trainer.generator.eval()
    accum: dict[str, float] = {}
    num_ranks = int(trainer.rank_embeddings.shape[0])
    for _ in range(num_batches):
        batch = sampler(batch_size)
        state, future_window, _, traj_rank = trainer._validate_macro_batch(
            batch, batch_size=batch_size
        )
        z_target = trainer._target_z(state, future_window)
        lang = trainer._lang_for_ranks(traj_rank)
        z_hat = trainer.generator(state, lang)
        if num_ranks <= 1:
            wrong_rank = traj_rank
        else:
            wrong_rank = _different_language_ranks(traj_rank, trainer.rank_embeddings)
        wrong_lang = trainer.rank_embeddings.index_select(0, wrong_rank)
        z_hat_wrong = trainer.generator(state, wrong_lang)
        batch_metrics = {
            f"{prefix}/z_cosine": float(
                F.cosine_similarity(z_hat, z_target, dim=-1).mean().item()
            ),
            f"{prefix}/z_mse": float(F.mse_loss(z_hat, z_target).item()),
            f"{prefix}/z_hat_rms": float(z_hat.pow(2).mean().sqrt().item()),
            f"{prefix}/z_target_rms": float(z_target.pow(2).mean().sqrt().item()),
            f"{prefix}/z_cosine_wrong_lang": float(
                F.cosine_similarity(z_hat_wrong, z_target, dim=-1).mean().item()
            ),
            f"{prefix}/z_mse_wrong_lang": float(
                F.mse_loss(z_hat_wrong, z_target).item()
            ),
        }
        batch_metrics[f"{prefix}/z_cosine_language_delta"] = (
            batch_metrics[f"{prefix}/z_cosine"]
            - batch_metrics[f"{prefix}/z_cosine_wrong_lang"]
        )
        for key, value in batch_metrics.items():
            accum[key] = accum.get(key, 0.0) + float(value)
    for key in accum:
        accum[key] /= float(num_batches)
    if was_training:
        trainer.generator.train()
    return accum


def _flatten_prefixed(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    marker = f"{prefix}/"
    return {
        key[len(marker) :]: float(value)
        for key, value in metrics.items()
        if key.startswith(marker)
    }


def _different_language_ranks(traj_rank: torch.Tensor, rank_embeddings: torch.Tensor) -> torch.Tensor:
    num_ranks = int(rank_embeddings.shape[0])
    if num_ranks <= 1:
        return traj_rank.clone()
    current = rank_embeddings.index_select(0, traj_rank)
    wrong_rank = (traj_rank + 1) % num_ranks
    unresolved = torch.ones_like(traj_rank, dtype=torch.bool)
    for offset in range(1, num_ranks):
        candidate = (traj_rank + offset) % num_ranks
        candidate_embedding = rank_embeddings.index_select(0, candidate)
        different = (candidate_embedding - current).abs().amax(dim=-1) > 1.0e-6
        take = unresolved & different
        wrong_rank = torch.where(take, candidate, wrong_rank)
        unresolved = unresolved & ~take
        if not bool(unresolved.any().item()):
            break
    return wrong_rank


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
    checkpoint_path = Path(args_cli.checkpoint).expanduser()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    trainer_config = _trainer_config_from_checkpoint(checkpoint)

    log_dir = _run_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = log_dir / "metrics.jsonl"
    per_rank_path = log_dir / "per_trajectory.jsonl"
    summary_path = log_dir / "summary.json"

    config_payload = {
        "task": args_cli.task,
        "num_envs": int(env_cfg.scene.num_envs),
        "seed": args_cli.seed,
        "checkpoint": str(checkpoint_path),
        "splits": list(args_cli.splits),
        "per_trajectory": bool(args_cli.per_trajectory),
        "trajectory_ranks": args_cli.trajectory_ranks,
        "trainer": trainer_config.to_dict(),
        "command": " ".join(sys.orig_argv),
    }
    (log_dir / "config.yaml").write_text(
        yaml.safe_dump(config_payload, sort_keys=True), encoding="utf-8"
    )
    dump_yaml(str(log_dir / "env.yaml"), env_cfg)
    print(f"[INFO] Logging M1 expert-state commander eval to: {log_dir}")

    env_cfg.log_dir = str(log_dir)
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        raise NotImplementedError("DirectMARLEnv is not supported by this script.")

    wrapped_env = IsaacLabWrapper(env)
    trainer = SkillCommanderTrainer(config=trainer_config, env=wrapped_env)
    try:
        trainer.generator.load_state_dict(checkpoint["generator_state_dict"])
        trainer.update = int(checkpoint.get("update", 0))
        names = [str(name) for name in wrapped_env.expert_trajectory_motion_names()]
        phrases = _phrase_map(trainer.language_table)
        print(
            "[INFO] Loaded commander checkpoint: "
            f"{checkpoint_path} (update={trainer.update}, state_dim={trainer.state_dim}, "
            f"z_dim={trainer.z_dim}, horizon={trainer.horizon_steps})"
        )
        print(
            "[INFO] Evaluating M1 on EXPERT macro state + language; "
            "no low-level policy or achieved-state rollout is used."
        )

        summary: dict[str, Any] = {
            "checkpoint": str(checkpoint_path),
            "update": int(trainer.update),
            "state_dim": int(trainer.state_dim),
            "z_dim": int(trainer.z_dim),
            "horizon_steps": int(trainer.horizon_steps),
            "num_trajectories": len(names),
            "aggregate": {},
            "per_trajectory": [],
        }

        for split in args_cli.splits:
            prefix = f"m1_expert_state/{split}"
            metrics = trainer.evaluate(
                num_batches=args_cli.eval_batches,
                batch_size=args_cli.batch_size,
                prefix=prefix,
                split=split,
            )
            row: dict[str, Any] = {
                "kind": "aggregate",
                "split": split,
                "update": int(trainer.update),
                **metrics,
            }
            _write_jsonl(metrics_path, row)
            summary["aggregate"][split] = metrics
            print(json.dumps(row, sort_keys=True))

        if args_cli.per_trajectory:
            ranks = _parse_rank_spec(args_cli.trajectory_ranks, count=len(names))
            for rank in ranks:
                motion = names[rank]
                phrase = phrases.get(motion)
                prefix = f"m1_expert_state/rank_{rank:04d}"

                def _sampler(batch_size: int, *, rank: int = rank):
                    return wrapped_env.sample_expert_macro_transition_batch(
                        batch_size=batch_size,
                        horizon_steps=trainer.horizon_steps,
                        split="all",
                        trajectory_ranks=[rank],
                    )

                metrics = _evaluate_batches(
                    trainer,
                    _sampler,
                    batch_size=args_cli.per_trajectory_batch_size,
                    num_batches=args_cli.per_trajectory_batches,
                    prefix=prefix,
                )
                flat = _flatten_prefixed(metrics, prefix)
                row = {
                    "kind": "per_trajectory",
                    "rank": int(rank),
                    "motion": motion,
                    "language_phrase": phrase,
                    "update": int(trainer.update),
                    **flat,
                }
                _write_jsonl(per_rank_path, row)
                summary["per_trajectory"].append(row)
                print(json.dumps(row, sort_keys=True))

        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"[INFO] Wrote summary: {summary_path}")
    finally:
        wrapped_env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
