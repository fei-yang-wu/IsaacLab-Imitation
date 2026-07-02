"""Run high-level skill pretraining, then low-level IPMD training.

This is a thin orchestration wrapper around the existing Isaac/RLOpt entrypoints:

1. ``scripts/rlopt/train_hl_skill_diffsr.py`` writes a frozen skill encoder
   checkpoint to ``<pretrain-output-dir>/checkpoints/latest.pt``.
2. ``scripts/rlopt/train.py`` consumes that checkpoint with
   ``agent.ipmd.command_source=hl_skill`` and trains the low-level policy.
"""

from __future__ import annotations

import argparse
import netrc as netrc_lib
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_PATH = "./data/lafan1/manifests/g1_lafan1_manifest.json"
DEFAULT_DATASET_PATH = "./data/lafan1/g1_hl_diffsr"


def _str_to_backend(value: str) -> str:
    backend = str(value).strip()
    if backend.lower() in {"", "none", "disabled", "off"}:
        return ""
    return backend


def _repo_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return resolved.resolve()


def _quote_cmd(cmd: list[str]) -> str:
    return shlex.join(cmd)


def _bool_str(value: bool) -> str:
    return "true" if value else "false"


def _normalize_hl_skill_command_mode(value: str) -> str:
    normalized = str(value).strip().lower()
    aliases = {"fz": "phi", "z_fz": "z_phi"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"z", "phi", "z_phi"}:
        raise ValueError(
            "--hl-skill-command-mode must be one of z, phi, or z_phi "
            f"(aliases: fz, z_fz), got {value!r}."
        )
    return normalized


def _hl_skill_command_code_dim(args: argparse.Namespace) -> int:
    mode = _normalize_hl_skill_command_mode(args.hl_skill_command_mode)
    if mode == "z":
        return int(args.z_dim)
    if mode == "phi":
        return int(args.diffsr_feature_dim)
    if mode == "z_phi":
        return int(args.z_dim) + int(args.diffsr_feature_dim)
    raise ValueError(f"Unsupported HL skill command mode: {mode!r}.")


def _default_pretrain_output_dir(args: argparse.Namespace) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed_label = f"seed{args.seed}" if args.seed is not None else "seedauto"
    name = (
        f"lafan1_w{args.horizon_steps}_z{args.z_dim}_{seed_label}_"
        f"{args.encoder_window_mode}_pipeline_{timestamp}"
    )
    return REPO_ROOT / "logs" / "hl_skill_diffsr" / name


def _default_train_exp_name(
    *,
    pretrain_output_dir: Path,
    phase_mode: str,
) -> str:
    return f"pipeline_{pretrain_output_dir.name}_{phase_mode}"


def _wandb_api_key_from_netrc(host: str = "api.wandb.ai") -> str | None:
    try:
        auth = netrc_lib.netrc().authenticators(host)
    except (FileNotFoundError, netrc_lib.NetrcParseError, OSError):
        return None
    if auth is None:
        return None
    _login, _account, password = auth
    return password or None


def _build_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("TERM", "xterm")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("HYDRA_FULL_ERROR", "1")
    env.setdefault("TORCHDYNAMO_DISABLE", "1")

    logger_backend = _str_to_backend(args.logger_backend)
    if logger_backend.lower() == "wandb":
        env.setdefault("WANDB_MODE", "online")
        if args.export_wandb_key and not env.get("WANDB_API_KEY"):
            api_key = _wandb_api_key_from_netrc(args.wandb_host)
            if api_key:
                env["WANDB_API_KEY"] = api_key
                print(
                    f"[INFO] Loaded W&B credentials from ~/.netrc for {args.wandb_host}."
                )
            else:
                print(
                    "[WARNING] W&B backend requested, but no WANDB_API_KEY or "
                    f"~/.netrc entry for {args.wandb_host} was found."
                )
    return env


def _add_common_app_args(cmd: list[str], args: argparse.Namespace) -> None:
    if args.headless:
        cmd.append("--headless")
    if args.device:
        cmd.extend(["--device", args.device])
    for app_arg in args.app_arg:
        cmd.append(app_arg)


def _pretrain_cmd(args: argparse.Namespace, output_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/rlopt/train_hl_skill_diffsr.py",
    ]
    _add_common_app_args(cmd, args)
    cmd.extend(
        [
            "--task",
            args.task,
            "--num_envs",
            str(args.pretrain_num_envs),
            "--seed",
            str(args.seed),
            "--output_dir",
            str(output_dir),
            "--horizon_steps",
            str(args.horizon_steps),
            "--encoder_window_mode",
            args.encoder_window_mode,
            "--z_dim",
            str(args.z_dim),
            "--latent_mode",
            args.latent_mode,
            "--reg_coeff",
            str(args.reg_coeff),
            "--categorical_groups",
            str(args.categorical_groups),
            "--categorical_categories",
            str(args.categorical_categories),
            "--gumbel_codebook_size",
            str(args.gumbel_codebook_size),
            "--gumbel_tau_start",
            str(args.gumbel_tau_start),
            "--gumbel_tau_end",
            str(args.gumbel_tau_end),
            "--gumbel_tau_anneal_iters",
            str(args.gumbel_tau_anneal_iters),
            "--gumbel_hard" if args.gumbel_hard else "--no-gumbel_hard",
            "--fsq_levels",
            *[str(level) for level in args.fsq_levels],
            "--vq_codebook_size",
            str(args.vq_codebook_size),
            "--vq_ema_decay",
            str(args.vq_ema_decay),
            "--vq_dead_code_reset_iters",
            str(args.vq_dead_code_reset_iters),
            "--diffsr_feature_dim",
            str(args.diffsr_feature_dim),
            "--diffsr_embed_dim",
            str(args.diffsr_embed_dim),
            "--diffsr_phi_parameterization",
            args.diffsr_phi_parameterization,
            "--batch_size",
            str(args.pretrain_batch_size),
            "--num_updates",
            str(args.pretrain_updates),
            "--log_interval",
            str(args.pretrain_log_interval),
            "--eval_batches",
            str(args.pretrain_eval_batches),
            "--window_probe_train_batches",
            str(args.window_probe_train_batches),
            "--window_probe_eval_batches",
            str(args.window_probe_eval_batches),
        ]
    )
    if args.pretrain_checkpoint:
        cmd.extend(["--checkpoint", str(_repo_path(args.pretrain_checkpoint))])
    if args.pretrain_reconstruction_eval:
        cmd.append("--reconstruction_eval")
    if args.pretrain_window_probe_eval:
        cmd.append("--window_probe_eval")
    logger_backend = _str_to_backend(args.logger_backend)
    cmd.extend(["--logger_backend", logger_backend or "none"])
    if logger_backend.lower() == "wandb":
        cmd.extend(["--wandb_project", args.wandb_project])
        if args.wandb_entity:
            cmd.extend(["--wandb_entity", args.wandb_entity])
        if args.wandb_group:
            cmd.extend(["--wandb_group", args.wandb_group])
        cmd.extend(["--wandb_run_name", f"{output_dir.name}_pretrain"])
    cmd.extend(
        [
            f"env.lafan1_manifest_path={_repo_path(args.manifest_path)}",
            f"env.dataset_path={_repo_path(args.dataset_path)}",
        ]
    )
    cmd.extend(args.pretrain_override)
    return cmd


def _train_cmd(
    args: argparse.Namespace,
    *,
    checkpoint_path: Path,
    pretrain_output_dir: Path,
) -> list[str]:
    phase_dim = 2 if args.phase_mode == "sin_cos" else 0
    command_mode = _normalize_hl_skill_command_mode(args.hl_skill_command_mode)
    command_code_dim = _hl_skill_command_code_dim(args)
    latent_dim = command_code_dim + phase_dim
    logger_backend = _str_to_backend(args.logger_backend)
    exp_name = args.exp_name or _default_train_exp_name(
        pretrain_output_dir=pretrain_output_dir,
        phase_mode=args.phase_mode,
    )

    cmd = [
        sys.executable,
        "scripts/rlopt/train.py",
    ]
    _add_common_app_args(cmd, args)
    if args.train_video:
        cmd.append("--video")
        cmd.extend(["--video_length", str(args.video_length)])
        cmd.extend(["--video_interval", str(args.video_interval)])
    cmd.extend(
        [
            "--num_envs",
            str(args.train_num_envs),
            "--task",
            args.task,
            "--algo",
            args.algo,
            "--seed",
            str(args.seed),
        ]
    )
    if args.train_max_iterations is not None:
        cmd.extend(["--max_iterations", str(args.train_max_iterations)])
    if args.train_log_interval is not None:
        cmd.extend(["--log_interval", str(args.train_log_interval)])

    hydra_overrides = [
        f"agent.logger.backend={logger_backend}",
        f"agent.logger.project_name={args.wandb_project}",
        f"agent.logger.exp_name={exp_name}",
        f"agent.logger.video={_bool_str(args.train_video)}",
        f"agent.save_interval={args.save_interval}",
        f"env.latent_command_dim={latent_dim}",
        f"agent.ipmd.latent_dim={latent_dim}",
        "agent.ipmd.command_source=hl_skill",
        f"agent.ipmd.hl_skill_checkpoint_path={checkpoint_path}",
        f"agent.ipmd.hl_skill_horizon_steps={args.horizon_steps}",
        f"agent.ipmd.hl_skill_command_mode={command_mode}",
        f"agent.ipmd.latent_steps_min={args.horizon_steps}",
        f"agent.ipmd.latent_steps_max={args.horizon_steps}",
        f"agent.ipmd.latent_learning.command_phase_mode={args.phase_mode}",
        f"agent.ipmd.latent_learning.code_period={args.horizon_steps}",
        f"agent.ipmd.latent_learning.code_latent_dim={command_code_dim}",
        f"agent.ipmd.hl_skill_finetune_enabled={_bool_str(args.finetune_hl_skill)}",
        f"agent.ipmd.hl_skill_pg_coeff={args.hl_skill_pg_coeff}",
        f"agent.ipmd.hl_skill_anchor_coeff={args.hl_skill_anchor_coeff}",
        f"agent.ipmd.hl_skill_offline_diffsr_coeff={args.hl_skill_offline_coeff}",
        f"agent.ipmd.hl_skill_lr={args.hl_skill_lr}",
        f"env.lafan1_manifest_path={_repo_path(args.manifest_path)}",
        f"env.dataset_path={_repo_path(args.dataset_path)}",
    ]
    if args.wandb_entity:
        hydra_overrides.append(f"agent.logger.entity={args.wandb_entity}")
    if args.wandb_group:
        hydra_overrides.append(f"agent.logger.group_name={args.wandb_group}")
    if not args.train_ipmd_reward_model:
        hydra_overrides.extend(
            [
                "agent.ipmd.reward_loss_coeff=0.0",
                "agent.ipmd.reward_l2_coeff=0.0",
                "agent.ipmd.reward_grad_penalty_coeff=0.0",
                "agent.ipmd.reward_logit_reg_coeff=0.0",
                "agent.ipmd.reward_param_weight_decay_coeff=0.0",
            ]
        )
    hydra_overrides.extend(args.train_override)
    cmd.extend(hydra_overrides)
    return cmd


def _run(cmd: list[str], *, env: dict[str, str], dry_run: bool) -> None:
    print(f"[INFO] Command: {_quote_cmd(cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sequentially pretrain a high-level DiffSR skill encoder and then "
            "train the low-level IPMD policy with the frozen encoder."
        )
    )
    parser.add_argument("--task", default="Isaac-Imitation-G1-Latent-v0")
    parser.add_argument("--algo", default="IPMD")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Forward headless mode to both Isaac entrypoints.",
    )
    parser.add_argument(
        "--app-arg",
        action="append",
        default=[],
        help="Extra Isaac AppLauncher argument forwarded to both child commands.",
    )
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)

    parser.add_argument("--pretrain-num-envs", type=int, default=16)
    parser.add_argument("--pretrain-output-dir", default=None)
    parser.add_argument("--pretrain-checkpoint", default=None)
    parser.add_argument("--pretrain-updates", type=int, default=50000)
    parser.add_argument("--pretrain-batch-size", type=int, default=8192)
    parser.add_argument("--pretrain-log-interval", type=int, default=100)
    parser.add_argument("--pretrain-eval-batches", type=int, default=4)
    parser.add_argument(
        "--pretrain-reconstruction-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--pretrain-window-probe-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--window-probe-train-batches", type=int, default=8)
    parser.add_argument("--window-probe-eval-batches", type=int, default=4)
    parser.add_argument("--horizon-steps", type=int, default=25)
    parser.add_argument(
        "--encoder-window-mode",
        choices=("full", "intermediate"),
        default="intermediate",
    )
    parser.add_argument("--z-dim", type=int, default=256)
    parser.add_argument(
        "--latent-mode",
        choices=(
            "deterministic",
            "gaussian",
            "categorical",
            "gumbel_multicat",
            "gumbel",
            "fsq",
            "vq",
        ),
        default="deterministic",
        help="Skill latent bottleneck design forwarded to pretraining.",
    )
    parser.add_argument("--reg-coeff", type=float, default=1.0e-3)
    parser.add_argument("--categorical-groups", type=int, default=64)
    parser.add_argument("--categorical-categories", type=int, default=128)
    parser.add_argument("--gumbel-codebook-size", type=int, default=512)
    parser.add_argument("--gumbel-tau-start", type=float, default=2.0)
    parser.add_argument("--gumbel-tau-end", type=float, default=0.5)
    parser.add_argument("--gumbel-tau-anneal-iters", type=int, default=2000)
    parser.add_argument(
        "--gumbel-hard", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--fsq-levels", type=int, nargs="+", default=[8, 8, 8, 5, 5])
    parser.add_argument("--vq-codebook-size", type=int, default=512)
    parser.add_argument("--vq-ema-decay", type=float, default=0.99)
    parser.add_argument("--vq-dead-code-reset-iters", type=int, default=0)
    parser.add_argument("--diffsr-feature-dim", type=int, default=256)
    parser.add_argument("--diffsr-embed-dim", type=int, default=512)
    parser.add_argument(
        "--diffsr-phi-parameterization",
        choices=("concat", "bilinear"),
        default="concat",
        help="DiffSR phi(s,z) parameterization forwarded to skill pretraining.",
    )
    parser.add_argument(
        "--pretrain-override",
        action="append",
        default=[],
        help="Extra Hydra override appended to the pretrain command.",
    )
    parser.add_argument(
        "--skip-pretrain",
        action="store_true",
        help="Use --pretrained-checkpoint directly and only run low-level training.",
    )
    parser.add_argument(
        "--pretrain-only",
        action="store_true",
        help="Run only high-level skill pretraining and stop after checkpointing.",
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        default=None,
        help="Existing high-level skill checkpoint used with --skip-pretrain.",
    )

    parser.add_argument("--train-num-envs", type=int, default=4096)
    parser.add_argument("--train-max-iterations", type=int, default=None)
    parser.add_argument("--train-log-interval", type=int, default=None)
    parser.add_argument(
        "--train-video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record and W&B-sync low-level training videos.",
    )
    parser.add_argument("--video-length", type=int, default=300)
    parser.add_argument("--video-interval", type=int, default=20_000)
    parser.add_argument(
        "--phase-mode",
        choices=("none", "sin_cos"),
        default="sin_cos",
        help="Phase features appended to the frozen skill latent command.",
    )
    parser.add_argument(
        "--hl-skill-command-mode",
        choices=("z", "phi", "z_phi", "fz", "z_fz"),
        default="z",
        help="HL skill command representation sent to the low-level policy.",
    )
    parser.add_argument(
        "--train-ipmd-reward-model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Keep IPMD reward-estimator updates enabled. By default the "
            "pipeline trains the policy from environment rewards with the "
            "frozen skill command."
        ),
    )
    parser.add_argument(
        "--finetune-hl-skill",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable online policy-gradient finetuning for the HL skill encoder.",
    )
    parser.add_argument(
        "--hl-skill-pg-coeff",
        type=float,
        default=0.05,
        help="Policy-gradient coefficient for online HL skill finetuning.",
    )
    parser.add_argument(
        "--hl-skill-anchor-coeff",
        type=float,
        default=0.01,
        help="Checkpoint-anchor coefficient for online HL skill finetuning.",
    )
    parser.add_argument(
        "--hl-skill-offline-coeff",
        type=float,
        default=1.0,
        help="Offline DiffSR coefficient for online HL skill finetuning.",
    )
    parser.add_argument(
        "--hl-skill-lr",
        type=float,
        default=3.0e-5,
        help="Learning rate for online HL skill finetuning.",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=100_000_000,
        help="Low-level checkpoint interval in environment frames.",
    )
    parser.add_argument("--logger-backend", default="wandb")
    parser.add_argument("--wandb-project", default="G1-Imitation-RLOpt-IPMD")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-host", default="api.wandb.ai")
    parser.add_argument(
        "--export-wandb-key",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Populate WANDB_API_KEY from ~/.netrc when using W&B.",
    )
    parser.add_argument("--exp-name", default=None)
    parser.add_argument(
        "--train-override",
        action="append",
        default=[],
        help="Extra Hydra override appended to the low-level train command.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print child commands without executing them.",
    )
    args = parser.parse_args()

    if args.skip_pretrain and not args.pretrained_checkpoint:
        parser.error("--skip-pretrain requires --pretrained-checkpoint.")
    if args.skip_pretrain and args.pretrain_only:
        parser.error("--skip-pretrain cannot be combined with --pretrain-only.")
    return args


def main() -> None:
    args = _parse_args()
    env = _build_env(args)

    if args.skip_pretrain:
        pretrain_output_dir = _repo_path(Path(args.pretrained_checkpoint).parent.parent)
        checkpoint_path = _repo_path(args.pretrained_checkpoint)
    else:
        pretrain_output_dir = (
            _repo_path(args.pretrain_output_dir)
            if args.pretrain_output_dir
            else _default_pretrain_output_dir(args)
        )
        checkpoint_path = pretrain_output_dir / "checkpoints" / "latest.pt"
        print(f"[INFO] High-level skill pretrain output: {pretrain_output_dir}")
        _run(_pretrain_cmd(args, pretrain_output_dir), env=env, dry_run=args.dry_run)
        if args.dry_run:
            print(f"[INFO] Expected high-level skill checkpoint: {checkpoint_path}")
        elif not checkpoint_path.is_file():
            raise FileNotFoundError(
                "High-level skill pretrain finished without expected checkpoint: "
                f"{checkpoint_path}"
            )

    if args.pretrain_only:
        print(f"[INFO] Pretrain-only mode finished at: {checkpoint_path}")
        return

    print(f"[INFO] Low-level training will use checkpoint: {checkpoint_path}")
    train_cmd = _train_cmd(
        args,
        checkpoint_path=checkpoint_path,
        pretrain_output_dir=pretrain_output_dir,
    )
    _run(train_cmd, env=env, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
