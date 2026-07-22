# ruff: noqa: E402
"""Distill a causal state-to-z planner from an online IPMD VQ/FSQ checkpoint."""

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
    description="Distill SkillCommander from an online IPMD VQVAE/FSQ encoder."
)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Imitation-G1-Latent-VQVAE-v0",
    help="Latent VQVAE Isaac Lab task.",
)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument(
    "--ll_checkpoint",
    type=str,
    required=True,
    help="Trained IPMD VQVAE low-level checkpoint (contains latent_learner_state_dict).",
)
parser.add_argument("--batch_size", type=int, default=8192)
parser.add_argument("--num_updates", type=int, default=2000)
parser.add_argument("--log_interval", type=int, default=100)
parser.add_argument("--eval_batches", type=int, default=2)
parser.add_argument("--lr", type=float, default=3.0e-4)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--grad_clip_norm", type=float, default=1.0)
parser.add_argument("--cosine_loss_coeff", type=float, default=1.0)
parser.add_argument("--z_norm_coeff", type=float, default=1.0e-4)
parser.add_argument(
    "--horizon_steps",
    type=int,
    default=10,
    help="Stored in synthetic skill_config for FrozenSkillCommanderSampler.",
)
parser.add_argument(
    "--state_history_steps",
    type=int,
    default=9,
    help="Past causal robot frames to flatten with the current frame for planner input.",
)
parser.add_argument(
    "--z_dim",
    type=int,
    default=64,
    help="Commander output dim (must match IPMD code_latent_dim).",
)
parser.add_argument(
    "--planner_type",
    type=str,
    default="flow_matching",
    choices=("mlp", "flow_matching"),
)
parser.add_argument("--flow_num_inference_steps", type=int, default=16)
parser.add_argument("--flow_inference_noise_std", type=float, default=0.0)
parser.add_argument("--flow_train_noise_std", type=float, default=1.0)
parser.add_argument("--flow_time_embed_dim", type=int, default=64)
parser.add_argument(
    "--generator_hidden_dims",
    type=int,
    nargs="+",
    default=[1024, 512, 512],
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
from rlopt.agent.hl_skill_diffsr import HighLevelSkillDiffSRConfig
from rlopt.agent.hl_skill_encoder import build_skill_encoder
from rlopt.agent.ipmd import IPMD
from rlopt.agent.skill_commander import (
    SkillCommanderConfig,
    _build_skill_commander_generator_from_config,
)
from tensordict import TensorDict
from torchrl.envs import TransformedEnv
from torchrl.envs.transforms import Compose, RewardClipping, RewardSum, StepCounter

AGENT_ENTRY_POINT = "rlopt_ipmd_vqvae_cfg_entry_point"


def _write_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def _run_dir() -> Path:
    if args_cli.output_dir is not None:
        return Path(args_cli.output_dir).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return Path("logs", "skill_commander_ipmd_vqvae", timestamp).resolve()


def _planner_state_from_macro_batch(
    batch: TensorDict,
    *,
    state_history_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """Flatten the causal planner observation."""
    if int(state_history_steps) > 0:
        history = batch.get(("hl", "state_history"))
        if history is None:
            history = batch.get(("planner", "state_history"))
        if history is None:
            raise KeyError("Expected hl/state_history for history>0 distillation.")
        history = history.to(device=device, dtype=torch.float32)
        return history.reshape(int(history.shape[0]), -1).contiguous()
    state = batch.get(("hl", "state"))
    if state is None:
        state = batch.get(("planner", "state"))
    if state is None:
        raise KeyError("Expected hl/state for distillation.")
    return state.to(device=device, dtype=torch.float32).contiguous()


def _vqvae_z_target_at_cursor(
    *,
    isaac_env: Any,
    learner: Any,
    traj_rank: torch.Tensor,
    local_step: torch.Tensor,
    z_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Encode the online VQ/FSQ target at the demo cursor."""
    cfg = learner._config()
    past_steps = int(cfg.patch_past_steps)
    future_steps = int(cfg.patch_future_steps)
    keys = [
        tuple(key) if isinstance(key, list) else key
        for key in list(getattr(cfg, "posterior_input_keys", None) or [])
    ]
    if not keys:
        keys = [
            tuple(key) if isinstance(key, list) else key
            for key in learner.required_expert_batch_keys()
            if isinstance(key, (tuple, list)) and key[0] == "expert_window"
        ]
    expert_raw = isaac_env._sample_expert_window_slice_for_trajectory_ranks(
        traj_rank,
        local_step,
        past_steps=past_steps,
        future_steps=future_steps,
    )
    batch_size = int(traj_rank.reshape(-1).shape[0])
    env_ids = torch.arange(batch_size, device=isaac_env.device, dtype=torch.long)
    terms = isaac_env._build_expert_window_terms(
        expert_raw,
        env_ids,
        context="expert",
        past_steps=past_steps,
        joint_ids=slice(None),
        anchor_body_name=getattr(isaac_env, "_expert_anchor_body_name", "torso_link"),
    )
    batch: dict[Any, torch.Tensor] = {}
    for key in keys:
        if key == "expert_action":
            continue
        if isinstance(key, tuple) and len(key) >= 2 and key[0] == "expert_window":
            term = str(key[1])
            if term not in terms:
                raise KeyError(
                    f"Expert window term {term!r} missing for VQVAE distillation target."
                )
            batch[key] = terms[term]
            continue
        raise ValueError(f"Unsupported posterior key for VQVAE target: {key!r}.")
    if not batch:
        raise RuntimeError("No expert_window keys available for VQVAE z_target.")
    td = TensorDict(batch, batch_size=[batch_size])
    z_target = learner.infer_expert_latents(td, detach=True).to(
        device=device, dtype=torch.float32
    )
    if int(z_target.shape[-1]) > int(z_dim):
        z_target = z_target[:, : int(z_dim)]
    if int(z_target.shape[-1]) != int(z_dim):
        raise ValueError(
            f"VQVAE z_target width {int(z_target.shape[-1])} != --z_dim={int(z_dim)}."
        )
    return z_target


def _write_diffsr_stub_skill_checkpoint(
    path: Path,
    *,
    state_dim: int,
    z_dim: int,
    horizon_steps: int,
    window_steps: int,
) -> HighLevelSkillDiffSRConfig:
    skill_config = HighLevelSkillDiffSRConfig(
        horizon_steps=int(horizon_steps),
        z_dim=int(z_dim),
        encoder_window_mode="full",
        latent_mode="deterministic",
    )
    encoder = build_skill_encoder(
        state_dim=int(state_dim),
        window_steps=int(window_steps),
        z_dim=int(z_dim),
        hidden_dims=skill_config.encoder_hidden_dims,
        spec=skill_config.latent_spec(),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": skill_config.to_dict(),
            "skill_encoder_state_dict": encoder.state_dict(),
            "note": "DiffSR-compatible stub for SkillCommanderTrainer.",
        },
        path,
    )
    return skill_config


@hydra_task_config(args_cli.task, AGENT_ENTRY_POINT)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: object,
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
    if hasattr(agent_cfg, "logger"):
        agent_cfg.logger.backend = ""

    torch.set_float32_matmul_precision("high")
    log_dir = _run_dir()
    checkpoints_dir = log_dir / "checkpoints"
    metrics_path = log_dir / "metrics.jsonl"
    planner_ckpt_path = checkpoints_dir / "latest.pt"
    stub_skill_path = checkpoints_dir / "diffsr_compat_stub_skill.pt"
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    ll_checkpoint = Path(args_cli.ll_checkpoint).expanduser().resolve()
    if not ll_checkpoint.is_file():
        raise FileNotFoundError(f"ll_checkpoint not found: {ll_checkpoint}")

    config_payload = {
        "task": args_cli.task,
        "ll_checkpoint": str(ll_checkpoint),
        "num_envs": int(env_cfg.scene.num_envs),
        "seed": args_cli.seed,
        "horizon_steps": int(args_cli.horizon_steps),
        "z_dim": int(args_cli.z_dim),
        "planner_type": args_cli.planner_type,
        "command": " ".join(sys.orig_argv),
    }
    (log_dir / "config.yaml").write_text(
        yaml.safe_dump(config_payload, sort_keys=True), encoding="utf-8"
    )
    dump_yaml(str(log_dir / "env.yaml"), env_cfg)
    print(f"[INFO] Logging IPMD-VQVAE skill commander run to: {log_dir}")

    env_cfg.log_dir = str(log_dir)
    gym_env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(gym_env.unwrapped, DirectMARLEnv):
        raise NotImplementedError("DirectMARLEnv is not supported by this script.")

    wrapped_env = IsaacLabWrapper(gym_env)
    train_env = TransformedEnv(
        base_env=wrapped_env,
        transform=Compose(
            RewardSum(),
            StepCounter(2),
            RewardClipping(-10.0, 5.0),
        ),
    )
    agent = IPMD(env=train_env, config=agent_cfg)
    print(f"[INFO] Loading LL checkpoint: {ll_checkpoint}")
    agent.load_model(str(ll_checkpoint))
    learner = getattr(agent, "_latent_learner", None)
    if learner is None:
        raise RuntimeError("Loaded IPMD agent has no latent_learner.")
    # Latent learners are plain objects (not nn.Module). Freeze nested modules.
    for value in vars(learner).values():
        if isinstance(value, torch.nn.Module):
            value.eval()
            for parameter in value.parameters():
                parameter.requires_grad_(False)

    device = torch.device(agent.device)
    isaac_env = wrapped_env._base_isaac_env()
    state_history_steps = int(args_cli.state_history_steps)
    if state_history_steps < 0:
        raise ValueError("--state_history_steps must be >= 0.")
    horizon_steps = int(args_cli.horizon_steps)
    z_dim = int(args_cli.z_dim)

    with torch.no_grad():
        probe = wrapped_env.sample_causal_planner_training_batch(
            batch_size=min(8, int(args_cli.batch_size)),
            horizon_steps=horizon_steps,
            split="all",
            history_steps=state_history_steps,
        )
        probe_state = _planner_state_from_macro_batch(
            probe,
            state_history_steps=state_history_steps,
            device=device,
        )
        probe_z = _vqvae_z_target_at_cursor(
            isaac_env=isaac_env,
            learner=learner,
            traj_rank=probe.get(("hl", "traj_rank")),
            local_step=probe.get(("hl", "local_step")),
            z_dim=z_dim,
            device=device,
        )
        macro_state = probe.get(("hl", "state"))
        if macro_state is None:
            raise RuntimeError("Causal planner batch missing hl/state for stub skill.")
    planner_state_dim = int(probe_state.shape[-1])
    macro_state_dim = int(macro_state.shape[-1])
    try:
        feature_slices = wrapped_env.causal_planner_observation_spec(
            history_steps=state_history_steps
        )
    except Exception:
        feature_slices = {}
    planner_observation_spec = dict(feature_slices)
    print(
        "[INFO] Distillation dims: "
        f"planner_state_dim={planner_state_dim} "
        f"macro_state_dim={macro_state_dim} z_dim={int(probe_z.shape[-1])}"
    )

    skill_config = _write_diffsr_stub_skill_checkpoint(
        stub_skill_path,
        # Stub DiffSR encoder consumes hl/state (macro), not planner_state.
        state_dim=macro_state_dim,
        z_dim=z_dim,
        horizon_steps=horizon_steps,
        window_steps=max(1, horizon_steps),
    )
    print(f"[INFO] Wrote DiffSR-compat stub skill checkpoint: {stub_skill_path}")

    commander_config = SkillCommanderConfig(
        skill_checkpoint_path=str(stub_skill_path),
        language_embeddings_path="",
        condition_on_language=False,
        state_history_steps=state_history_steps,
        planner_type=str(args_cli.planner_type),
        generator_hidden_dims=tuple(int(x) for x in args_cli.generator_hidden_dims),
        flow_num_inference_steps=int(args_cli.flow_num_inference_steps),
        flow_time_embed_dim=int(args_cli.flow_time_embed_dim),
        flow_train_noise_std=float(args_cli.flow_train_noise_std),
        flow_inference_noise_std=float(args_cli.flow_inference_noise_std),
        batch_size=int(args_cli.batch_size),
        num_updates=int(args_cli.num_updates),
        log_interval=int(args_cli.log_interval),
        eval_batches=int(args_cli.eval_batches),
        train_split="all",
        eval_split="all",
        lr=float(args_cli.lr),
        weight_decay=float(args_cli.weight_decay),
        grad_clip_norm=(
            None if float(args_cli.grad_clip_norm) <= 0 else float(args_cli.grad_clip_norm)
        ),
        cosine_loss_coeff=float(args_cli.cosine_loss_coeff),
        z_norm_coeff=float(args_cli.z_norm_coeff),
        device=str(device),
    )
    commander_config.validate()

    generator = _build_skill_commander_generator_from_config(
        commander_config,
        state_dim=planner_state_dim,
        lang_embed_dim=0,
        z_dim=z_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        generator.parameters(),
        lr=float(args_cli.lr),
        weight_decay=float(args_cli.weight_decay),
    )

    def _empty_lang(batch_size: int) -> torch.Tensor:
        return torch.empty((int(batch_size), 0), device=device, dtype=torch.float32)

    def _sample_batch(batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        macro = wrapped_env.sample_causal_planner_training_batch(
            batch_size=int(batch_size),
            horizon_steps=horizon_steps,
            split="all",
            history_steps=state_history_steps,
        )
        with torch.no_grad():
            planner_state_b = _planner_state_from_macro_batch(
                macro,
                state_history_steps=state_history_steps,
                device=device,
            )
            z_target = _vqvae_z_target_at_cursor(
                isaac_env=isaac_env,
                learner=learner,
                traj_rank=macro.get(("hl", "traj_rank")),
                local_step=macro.get(("hl", "local_step")),
                z_dim=z_dim,
                device=device,
            )
        if int(planner_state_b.shape[-1]) != planner_state_dim:
            raise RuntimeError(
                "Planner state width drifted during training: "
                f"{int(planner_state_b.shape[-1])} != {planner_state_dim}."
            )
        return planner_state_b, z_target

    def _evaluate(prefix: str) -> dict[str, float]:
        generator.eval()
        cos_vals: list[float] = []
        mse_vals: list[float] = []
        with torch.no_grad():
            for _ in range(max(1, int(args_cli.eval_batches))):
                state_b, z_b = _sample_batch(
                    min(int(args_cli.batch_size), 1024),
                )
                z_hat = generator(state_b, _empty_lang(int(state_b.shape[0])))
                cos_vals.append(
                    float(F.cosine_similarity(z_hat, z_b, dim=-1).mean().item())
                )
                mse_vals.append(float(F.mse_loss(z_hat, z_b).item()))
        return {
            f"{prefix}/z_cosine": sum(cos_vals) / len(cos_vals),
            f"{prefix}/z_mse": sum(mse_vals) / len(mse_vals),
        }

    def _save_planner(update_i: int) -> None:
        torch.save(
            {
                "generator_state_dict": generator.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": commander_config.to_dict(),
                "skill_config": skill_config.to_dict(),
                "skill_checkpoint_path": str(stub_skill_path),
                "language_embeddings_path": "",
                "condition_on_language": False,
                "state_history_steps": int(state_history_steps),
                "state_dim": int(planner_state_dim),
                "macro_state_dim": int(macro_state_dim),
                "planner_state_dim": int(planner_state_dim),
                "planner_observation_spec": dict(planner_observation_spec),
                "lang_embed_dim": 0,
                "z_dim": int(z_dim),
                "horizon_steps": int(horizon_steps),
                "encoder_window_mode": "full",
                "update": int(update_i),
                "target_encoder": "ipmd_vqvae",
                "ll_checkpoint": str(ll_checkpoint),
            },
            planner_ckpt_path,
        )

    update = 0
    generator.train()
    while update < int(args_cli.num_updates):
        state_b, z_b = _sample_batch(int(args_cli.batch_size))
        lang_b = _empty_lang(int(state_b.shape[0]))
        optimizer.zero_grad(set_to_none=True)
        if args_cli.planner_type == "flow_matching":
            flow_loss, _flow_metrics = generator.flow_matching_loss(
                state_b, lang_b, z_b
            )
            loss = flow_loss
            endpoint = generator(state_b, lang_b)
            cosine_term = 1.0 - F.cosine_similarity(endpoint, z_b, dim=-1).mean()
            loss = loss + float(args_cli.cosine_loss_coeff) * cosine_term
            if float(args_cli.z_norm_coeff) != 0.0:
                loss = loss + float(args_cli.z_norm_coeff) * endpoint.pow(2).mean()
        else:
            endpoint = generator(state_b, lang_b)
            cosine_term = 1.0 - F.cosine_similarity(endpoint, z_b, dim=-1).mean()
            loss = float(args_cli.cosine_loss_coeff) * cosine_term + F.mse_loss(
                endpoint, z_b
            )
            if float(args_cli.z_norm_coeff) != 0.0:
                loss = loss + float(args_cli.z_norm_coeff) * endpoint.pow(2).mean()
        loss.backward()
        if commander_config.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                generator.parameters(), float(commander_config.grad_clip_norm)
            )
        optimizer.step()
        update += 1

        if update % int(args_cli.log_interval) == 0 or update == 1:
            row = {
                "update": int(update),
                "loss": float(loss.detach().item()),
                **_evaluate("train_batch"),
            }
            _write_jsonl(metrics_path, row)
            print(json.dumps(row, sort_keys=True))
            _save_planner(update)

    _save_planner(update)

    final_metrics = _evaluate("eval")
    final_row = {"update": int(update), "post_train_eval": True, **final_metrics}
    _write_jsonl(metrics_path, final_row)
    print(json.dumps(final_row, indent=2, sort_keys=True))
    print(f"[INFO] Saved planner checkpoint: {planner_ckpt_path}")
    print(f"[INFO] Stub skill checkpoint: {stub_skill_path}")
    train_env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
