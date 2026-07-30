#!/usr/bin/env python3
"""Stage 1a: pretrain the DiffSR latent skill encoder.

Trains an offline high-level skill encoder against sampled expert macro
transitions, then freezes it for the low-level controller to consume. The
latent-learning strategy (deterministic, Gaussian, VQ, FSQ, SONIC-FSQ,
categorical, Gumbel) is a config choice; every arm shares the same DiffSR
objective, window definition, and held-out split so the arms stay comparable.

Two distinctions this stage enforces, because getting either wrong produces
numbers that look fine and mean nothing:

* **Offline versus online latent lineage.** Arms such as ``future_cvae`` train
  their encoder jointly with the policy and have no offline pretrain at all.
  Selecting one here is a config error with a pointer to the low-level stage,
  rather than an argparse rejection minutes into Isaac Sim startup.
* **Held-out data versus held command.** ``holdout.*`` reserves whole expert
  trajectories for evaluation. ``hold.*`` (in the low-level stage) is how long a
  published command is held by the tracker. They are unrelated, and the shared
  informal name "hold out period" has meant both in this project.

Usage
-----
    # paper default: deterministic bottleneck on LAFAN1
    pixi run python experiments/paper/pipeline/pretrain_latent_encoder.py

    # a reconstruction-strategy ablation arm
    pixi run python experiments/paper/pipeline/pretrain_latent_encoder.py \\
        latent=fsq dataset=bones_seed seed=1

    # no phase clock, shorter future window
    pixi run python experiments/paper/pipeline/pretrain_latent_encoder.py \\
        encoder.horizon_steps=10 encoder.window_mode=full

Every parameter lives in ``experiments/paper/conf/pretrain.yaml``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

_PAPER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PAPER_DIR))

from _paper_common import (  # noqa: E402
    PipelineError,
    REPO_ROOT,
    SCRIPTS_RLOPT,
    extend_overrides,
    file_provenance,
    hydra_override,
    latest_checkpoint,
    opt_path,
    opt_str,
    refuse_existing_output,
    require_file,
    resolve_interpreter,
    run_command,
    seed_environment,
    to_container,
    write_stage_record,
)
from _paper_specs import get_latent_mode  # noqa: E402

logger = logging.getLogger("pretrain_latent_encoder")

ENTRYPOINT = SCRIPTS_RLOPT / "train_hl_skill_diffsr.py"


def build_command(cfg: DictConfig, output_dir: Path) -> list[str]:
    """Render the argv for ``scripts/rlopt/train_hl_skill_diffsr.py``."""
    mode = get_latent_mode(str(cfg.latent.mode))
    if mode.lineage != "offline":
        raise PipelineError(
            f"Latent mode {mode.name!r} trains its encoder jointly with the "
            "policy, so it has no offline pretrain stage.\n"
            f"  {mode.notes}\n"
            "Run experiments/paper/pipeline/train_low_level.py directly for this "
            "arm, or set stages=[low_level,...] in the pipeline config."
        )

    manifest = require_file(cfg.dataset.manifest, "dataset.manifest")
    dataset_path = opt_path(cfg.dataset.latent_dataset_path)
    if dataset_path is None:
        raise PipelineError(
            "dataset.latent_dataset_path is required: the reference cache is "
            "content-bound, and an environment-default cache can silently pair a "
            "manifest with rows built from a different NPZ tree."
        )

    cmd = [
        *resolve_interpreter(cfg, "isaac"),
        str(ENTRYPOINT),
        "--task",
        str(cfg.task),
        "--num_envs",
        str(int(cfg.num_envs)),
        "--seed",
        str(int(cfg.seed)),
        "--output_dir",
        str(output_dir),
        "--headless",
        # Encoder window.
        "--horizon_steps",
        str(int(cfg.encoder.horizon_steps)),
        "--encoder_window_mode",
        str(cfg.encoder.window_mode),
        # Latent bottleneck.
        "--latent_mode",
        str(mode.latent_mode),
        "--z_dim",
        str(int(cfg.latent.z_dim)),
        "--reg_coeff",
        str(cfg.latent.reg_coeff),
        # DiffSR head.
        "--diffsr_feature_dim",
        str(int(cfg.diffsr.feature_dim)),
        "--diffsr_embed_dim",
        str(int(cfg.diffsr.embed_dim)),
        # Optimization.
        "--batch_size",
        str(int(cfg.optim.batch_size)),
        "--num_updates",
        str(int(cfg.optim.num_updates)),
        "--log_interval",
        str(int(cfg.optim.log_interval)),
        "--grad_clip_norm",
        str(cfg.optim.grad_clip_norm),
        # Held-out trajectory split.
        "--train_split",
        str(cfg.holdout.train_split),
        "--eval_split",
        str(cfg.holdout.eval_split),
        "--eval_trajectory_fraction",
        str(cfg.holdout.eval_trajectory_fraction),
        "--trajectory_split_seed",
        str(int(cfg.holdout.trajectory_split_seed)),
        "--eval_batches",
        str(int(cfg.holdout.eval_batches)),
    ]

    hidden_dims = to_container(cfg.encoder.hidden_dims)
    if hidden_dims:
        cmd += ["--encoder_hidden_dims", *[str(int(dim)) for dim in hidden_dims]]

    eval_batch_size = cfg.holdout.get("eval_batch_size")
    if eval_batch_size:
        cmd += ["--eval_batch_size", str(int(eval_batch_size))]

    # Strategy-specific hyperparameters. Only the selected arm's group is passed
    # so an unrelated default cannot appear in the recorded command.
    group = mode.hyperparameter_group
    if group == "vq":
        vq = cfg.latent.vq
        cmd += [
            "--vq_codebook_size",
            str(int(vq.codebook_size)),
            "--vq_ema_decay",
            str(vq.ema_decay),
            "--vq_dead_code_reset_iters",
            str(int(vq.dead_code_reset_iters)),
        ]
    elif group == "fsq":
        levels = [str(int(level)) for level in to_container(cfg.latent.fsq.levels)]
        cmd += ["--fsq_levels", *levels]
    elif group == "sonic_fsq":
        levels = [str(int(level)) for level in to_container(cfg.latent.sonic_fsq.levels)]
        if len(levels) != int(cfg.latent.z_dim):
            raise PipelineError(
                f"sonic_fsq requires z_dim == len(levels); got z_dim="
                f"{int(cfg.latent.z_dim)} and {len(levels)} levels."
            )
        cmd += ["--sonic_fsq_levels", *levels]
    elif group == "categorical":
        cat = cfg.latent.categorical
        cmd += [
            "--categorical_groups",
            str(int(cat.groups)),
            "--categorical_categories",
            str(int(cat.categories)),
        ]
    elif group == "gumbel":
        gumbel = cfg.latent.gumbel
        cmd += [
            "--gumbel_codebook_size",
            str(int(gumbel.codebook_size)),
            "--gumbel_tau_start",
            str(gumbel.tau_start),
            "--gumbel_tau_end",
            str(gumbel.tau_end),
            "--gumbel_tau_anneal_iters",
            str(int(gumbel.tau_anneal_iters)),
            "--gumbel_hard" if bool(gumbel.hard) else "--no-gumbel_hard",
        ]

    # Diagnostics. Reconstruction and window-probe evaluations are what make the
    # strategy ablation interpretable, so they are on by default.
    if bool(cfg.diagnostics.reconstruction_eval):
        cmd += [
            "--reconstruction_eval",
            "--reconstruction_norm_eps",
            str(cfg.diagnostics.reconstruction_norm_eps),
        ]
    if bool(cfg.diagnostics.window_probe_eval):
        cmd += [
            "--window_probe_eval",
            "--window_probe_train_batches",
            str(int(cfg.diagnostics.window_probe_train_batches)),
            "--window_probe_ridge",
            str(cfg.diagnostics.window_probe_ridge),
        ]

    if bool(cfg.get("assert_kitless", False)):
        cmd.append("--assert-kitless")

    checkpoint = opt_path(cfg.get("resume_checkpoint"))
    if checkpoint is not None:
        cmd += ["--checkpoint", str(require_file(checkpoint, "resume_checkpoint"))]

    # Logging.
    cmd += ["--logger_backend", str(cfg.logging.backend)]
    for flag, key in (
        ("--wandb_project", "project"),
        ("--wandb_entity", "entity"),
        ("--wandb_group", "group"),
        ("--wandb_run_name", "run_name"),
        ("--wandb_mode", "mode"),
    ):
        value = opt_str(cfg.logging.get(key))
        if value:
            cmd += [flag, value]

    # Hydra overrides consumed by the environment config.
    cmd.append(hydra_override("env.lafan1_manifest_path", manifest))
    cmd.append(hydra_override("env.dataset_path", dataset_path))
    if bool(cfg.dataset.get("refresh_zarr", False)):
        cmd.append(hydra_override("env.refresh_zarr_dataset", True))

    macro_terms = to_container(cfg.encoder.get("macro_state_terms"))
    if macro_terms:
        # Selects which expert terms form the encoder's macro state, i.e. which
        # command space the latent is a compression OF.
        cmd.append(hydra_override("env.expert_macro_state_terms", macro_terms))

    physics = opt_str(cfg.get("physics"))
    if physics:
        cmd.append(f"physics={physics}")

    return extend_overrides(cmd, cfg.get("extra_overrides"))


@hydra.main(version_base=None, config_path="../conf", config_name="pretrain")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    output_dir = refuse_existing_output(
        cfg.output_dir,
        allow_existing=bool(cfg.allow_existing_output),
        what="pretrain output_dir",
    )
    command = build_command(cfg, output_dir)

    run_command(
        command,
        dry_run=bool(cfg.dry_run),
        log_path=output_dir / "logs/pretrain.log",
        env=seed_environment(int(cfg.seed), deterministic=bool(cfg.deterministic)),
        cwd=REPO_ROOT,
        what="latent encoder pretraining",
    )

    outputs: dict[str, object] = {"run_dir": str(output_dir)}
    if not bool(cfg.dry_run):
        # The low-level stage must bind the exact checkpoint recorded here: a
        # later checkpoint with identical runtime weights is still a different
        # artifact, and the qualification gate compares hashes, not behaviour.
        encoder = latest_checkpoint(
            output_dir, patterns=("checkpoints/latest.pt", "checkpoints/best.pt")
        )
        outputs["skill_encoder_checkpoint"] = file_provenance(encoder)
        best = output_dir / "checkpoints/best.pt"
        if best.is_file():
            outputs["best_checkpoint"] = file_provenance(best)
        metrics = output_dir / "metrics.jsonl"
        if metrics.is_file():
            outputs["metrics"] = file_provenance(metrics)
        print(f"[INFO] skill encoder checkpoint: {encoder}", flush=True)

    write_stage_record(
        output_dir,
        stage="pretrain_latent_encoder",
        entrypoint=Path(__file__).resolve(),
        cfg=cfg,
        command=command,
        inputs={
            "manifest": opt_path(cfg.dataset.manifest),
            "dataset_path": {"path": str(opt_path(cfg.dataset.latent_dataset_path))},
        },
        outputs=outputs,
        extra={
            "latent_mode": to_container(cfg.latent),
            "seed": int(cfg.seed),
            "deterministic": bool(cfg.deterministic),
        },
        status="dry_run" if bool(cfg.dry_run) else "complete",
    )


if __name__ == "__main__":
    main()
