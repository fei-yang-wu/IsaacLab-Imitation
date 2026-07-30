#!/usr/bin/env python3
"""Stage 1b: train the low-level tracking controller.

One entrypoint covers every low-level row the paper needs, because they differ
only in which command the policy receives:

``latent_skill``
    A frozen DiffSR encoder publishes a latent command that the tracker holds
    for the publication interval. Requires the stage-1a encoder checkpoint.
``full_body_trajectory`` / ``root_qpos`` / ``root_points5`` / ``ee_trajectory``
    An explicit packet consumed one slot per control step. Each interface gets
    its own natively-trained controller: nothing is reconstructed, so a
    ``root_qpos`` controller simply never receives joint velocities.
online latent arms (``future_cvae``, ``patch_vqvae``, ``per_step_vq_sequence``)
    The encoder trains jointly with the policy, so there is no frozen checkpoint
    to bind and no stage-1a input.

Where checkpoints go matters more than it looks. On ICE a wall-clock TIMEOUT is
a hard SIGKILL that wipes node-local output, and an earlier ablation lost every
checkpoint of a multi-hour run exactly that way. ``logging.log_dir`` must point
at persistent storage for any cluster run; the cluster profiles set it to a
``/data`` store rather than the per-submission workspace.

Usage
-----
    # paper latent row, bound to a specific encoder
    pixi run python experiments/paper/pipeline/train_low_level.py \\
        interface=latent_skill \\
        encoder.checkpoint=logs/paper/pretrain/checkpoints/latest.pt

    # the explicit packet row
    pixi run python experiments/paper/pipeline/train_low_level.py \\
        interface=full_body_trajectory

    # a command-space ablation arm
    pixi run python experiments/paper/pipeline/train_low_level.py \\
        interface=root_qpos num_envs=4096 max_iterations=10173

Every parameter lives in ``experiments/paper/conf/low_level.yaml``.
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
from _paper_specs import get_interface, get_latent_mode  # noqa: E402

logger = logging.getLogger("train_low_level")

NEWTON_ENTRYPOINT = SCRIPTS_RLOPT / "train.py"
PHYSX_ENTRYPOINT = SCRIPTS_RLOPT / "train_physx.py"


def resolve_entrypoint(cfg: DictConfig) -> tuple[Path, bool]:
    """Pick the training entrypoint and whether ``--assert-kitless`` applies.

    The PhysX path must start Kit before importing anything, so it has its own
    bootstrap script and rejects ``--assert-kitless``; the Newton path asserts
    Kit was never loaded. Getting this pairing wrong fails at import time with a
    message that does not name the real cause.
    """
    requested = str(cfg.entrypoint)
    physics = opt_str(cfg.physics) or "newton_mjwarp"
    if requested == "auto":
        requested = "physx" if physics == "physx" else "standard"
    if requested == "physx":
        if physics != "physx":
            raise PipelineError(
                f"entrypoint=physx requires physics=physx, got physics={physics!r}."
            )
        return PHYSX_ENTRYPOINT, False
    if requested == "standard":
        if physics == "physx":
            raise PipelineError(
                "physics=physx requires entrypoint=physx; the standard entrypoint "
                "asserts that Kit was never loaded."
            )
        return NEWTON_ENTRYPOINT, bool(cfg.assert_kitless)
    raise PipelineError(
        f"Unknown entrypoint {requested!r}; expected auto, standard, or physx."
    )


def _latent_command_dim(cfg: DictConfig) -> int:
    """Total latent command width: code plus the optional phase clock."""
    phase_mode = str(cfg.hold.phase_mode)
    phase_dim = {"sin_cos": 2, "none": 0}.get(phase_mode)
    if phase_dim is None:
        raise PipelineError(
            f"Unknown hold.phase_mode {phase_mode!r}; expected sin_cos or none."
        )
    return int(cfg.encoder.z_dim) + phase_dim


def build_command(cfg: DictConfig, log_dir: Path) -> list[str]:
    """Render the argv for the RLOpt training entrypoint."""
    interface_name = str(cfg.interface.name)
    entrypoint, assert_kitless = resolve_entrypoint(cfg)

    latent_arm = opt_str(cfg.get("latent_arm"))
    online_mode = None
    if latent_arm is not None:
        mode = get_latent_mode(latent_arm)
        if mode.lineage == "online":
            online_mode = mode

    if online_mode is not None:
        task = opt_str(cfg.get("task")) or online_mode.task
    else:
        task = opt_str(cfg.get("task")) or get_interface(interface_name).default_task
    if task is None:
        raise PipelineError("task could not be resolved and was not set explicitly.")

    manifest = require_file(cfg.dataset.manifest, "dataset.manifest")
    dataset_path = opt_path(cfg.dataset.dataset_path)
    if dataset_path is None:
        raise PipelineError(
            "dataset.dataset_path is required. The latent and vanilla rows use "
            "separate content-bound caches; relying on the environment default "
            "has previously paired a manifest with rows from a different tree."
        )

    cmd = [
        *resolve_interpreter(cfg, "isaac"),
        str(entrypoint),
        "--task",
        task,
        "--algo",
        str(cfg.algo),
        "--num_envs",
        str(int(cfg.num_envs)),
        "--max_iterations",
        str(int(cfg.max_iterations)),
        "--seed",
        str(int(cfg.seed)),
        "--headless",
    ]
    if assert_kitless:
        cmd.append("--assert-kitless")

    agent_entry_point = opt_str(cfg.get("agent_entry_point"))
    if agent_entry_point:
        cmd += ["--agent", agent_entry_point]

    checkpoint = opt_path(cfg.get("resume_checkpoint"))
    if checkpoint is not None:
        cmd += ["--checkpoint", str(require_file(checkpoint, "resume_checkpoint"))]

    if bool(cfg.video.enabled):
        cmd += [
            "--video",
            "--video_length",
            str(int(cfg.video.length)),
            "--video_interval",
            str(int(cfg.video.interval)),
        ]

    kit_args = opt_str(cfg.get("kit_args"))
    if kit_args:
        cmd.append(f"--kit_args={kit_args}")

    # ---- Hydra overrides ---------------------------------------------------
    physics = opt_str(cfg.physics)
    if physics:
        cmd.append(f"physics={physics}")
        if physics == "newton_mjwarp":
            cmd.append(hydra_override("env.sim.physics.solver_cfg.njmax", int(cfg.njmax)))
            cmd.append(
                hydra_override("env.sim.physics.solver_cfg.nconmax", int(cfg.nconmax))
            )

    cmd.append(hydra_override("env.lafan1_manifest_path", manifest))
    cmd.append(hydra_override("env.dataset_path", dataset_path))
    cmd.append(hydra_override("env.refresh_zarr_dataset", bool(cfg.dataset.refresh_zarr)))

    motions = to_container(cfg.dataset.get("motions"))
    if motions:
        cmd.append(hydra_override("env.motions", motions))
    trajectories = to_container(cfg.dataset.get("trajectories"))
    if trajectories:
        cmd.append(hydra_override("env.trajectories", trajectories))

    cmd.append(hydra_override("env.reset_schedule", str(cfg.reset.schedule)))
    if cfg.reset.get("random_reset_step_min") is not None:
        cmd.append(
            hydra_override(
                "env.random_reset_step_min", int(cfg.reset.random_reset_step_min)
            )
        )
    if cfg.reset.get("random_reset_step_max") is not None:
        cmd.append(
            hydra_override(
                "env.random_reset_step_max", int(cfg.reset.random_reset_step_max)
            )
        )
    if cfg.reset.get("random_reset_full_trajectory") is not None:
        cmd.append(
            hydra_override(
                "env.random_reset_full_trajectory",
                bool(cfg.reset.random_reset_full_trajectory),
            )
        )

    interface = get_interface(interface_name)
    hold_steps = int(cfg.hold.steps)

    if interface.kind == "latent":
        latent_dim = _latent_command_dim(cfg)
        declared = cfg.encoder.get("latent_command_dim")
        if declared is not None and int(declared) != latent_dim:
            raise PipelineError(
                f"encoder.latent_command_dim={int(declared)} disagrees with "
                f"z_dim={int(cfg.encoder.z_dim)} plus the "
                f"{cfg.hold.phase_mode} phase clock ({latent_dim}). The command "
                "width must match the encoder that produced it."
            )
        if online_mode is None:
            encoder_ckpt = require_file(cfg.encoder.checkpoint, "encoder.checkpoint")
            cmd.append(hydra_override("agent.ipmd.command_source", "hl_skill"))
            cmd.append(
                hydra_override("agent.ipmd.hl_skill_checkpoint_path", encoder_ckpt)
            )
            cmd.append(hydra_override("agent.ipmd.hl_skill_horizon_steps", hold_steps))
            cmd.append(
                hydra_override(
                    "agent.ipmd.hl_skill_command_mode", str(cfg.encoder.command_mode)
                )
            )
            if bool(cfg.encoder.finetune):
                cmd.append(hydra_override("agent.ipmd.hl_skill_finetune_enabled", True))
                cmd.append(
                    hydra_override("agent.ipmd.hl_skill_lr", cfg.encoder.finetune_lr)
                )
        else:
            # Jointly-trained encoder: the posterior is the command source and
            # there is no frozen checkpoint to bind.
            cmd.append(hydra_override("agent.ipmd.command_source", "posterior"))
            cmd.append(
                hydra_override(
                    "agent.ipmd.latent_learning.method", online_mode.online_method
                )
            )

        cmd.append(hydra_override("agent.ipmd.use_latent_command", True))
        cmd.append(hydra_override("agent.ipmd.latent_steps_min", hold_steps))
        cmd.append(hydra_override("agent.ipmd.latent_steps_max", hold_steps))
        cmd.append(hydra_override("agent.ipmd.latent_learning.code_period", hold_steps))
        cmd.append(
            hydra_override(
                "agent.ipmd.latent_learning.code_latent_dim", int(cfg.encoder.z_dim)
            )
        )
        cmd.append(
            hydra_override(
                "agent.ipmd.latent_learning.command_phase_mode", str(cfg.hold.phase_mode)
            )
        )
        cmd.append(hydra_override("env.latent_command_dim", latent_dim))
        cmd.append(hydra_override("agent.ipmd.latent_dim", latent_dim))
    else:
        # Explicit packet: the tracker reads one slot per control step from a
        # command window held for the publication interval.
        cmd.append(hydra_override("agent.ipmd.use_latent_command", False))
        cmd.append(hydra_override("agent.command_space", interface.command_space))
        cmd.append(hydra_override("env.command_hold_steps", hold_steps))
        cmd.append(
            hydra_override(
                "env.command_observation_source", str(cfg.command.observation_source)
            )
        )
        cmd.append(hydra_override("env.latent_patch_past_steps", 0))
        terms = to_container(cfg.command.get("observation_terms")) or list(
            interface.command_observation_terms
        )
        if terms:
            cmd.append(hydra_override("env.command_observation_terms", terms))
        if cfg.command.get("future_steps") is not None:
            cmd.append(
                hydra_override(
                    "env.latent_patch_future_steps", int(cfg.command.future_steps)
                )
            )

    cmd.append(hydra_override("agent.logger.log_dir", log_dir))
    backend = opt_str(cfg.logging.backend)
    cmd.append(f"agent.logger.backend={backend or ''}")
    for key, override in (
        ("project", "agent.logger.project"),
        ("run_name", "agent.logger.run_name"),
        ("group", "agent.logger.group"),
    ):
        value = opt_str(cfg.logging.get(key))
        if value:
            cmd.append(hydra_override(override, value))

    if cfg.get("save_interval") is not None:
        cmd.append(hydra_override("agent.save_interval", int(cfg.save_interval)))

    return extend_overrides(cmd, cfg.get("extra_overrides"))


@hydra.main(version_base=None, config_path="../conf", config_name="low_level")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    output_dir = refuse_existing_output(
        cfg.output_dir,
        allow_existing=bool(cfg.allow_existing_output),
        what="low-level output_dir",
    )
    log_dir = opt_path(cfg.logging.log_dir) or output_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    if log_dir != output_dir:
        print(f"[INFO] training checkpoints go to {log_dir}", flush=True)

    command = build_command(cfg, log_dir)

    frames = int(cfg.num_envs) * int(cfg.max_iterations) * int(cfg.frames_per_env_batch)
    print(
        f"[INFO] interface={cfg.interface.name} envs={int(cfg.num_envs)} "
        f"iterations={int(cfg.max_iterations)} ~{frames / 1e9:.2f}B frames",
        flush=True,
    )

    run_command(
        command,
        dry_run=bool(cfg.dry_run),
        log_path=output_dir / "logs/low_level.log",
        env=seed_environment(int(cfg.seed), deterministic=bool(cfg.deterministic)),
        cwd=REPO_ROOT,
        what="low-level training",
    )

    outputs: dict[str, object] = {"run_dir": str(output_dir), "log_dir": str(log_dir)}
    if not bool(cfg.dry_run):
        policy = latest_checkpoint(
            log_dir, patterns=("**/model_step_*.pt", "**/model.pt")
        )
        outputs["policy_checkpoint"] = file_provenance(policy)
        print(f"[INFO] low-level checkpoint: {policy}", flush=True)

    encoder_input = None
    if get_interface(str(cfg.interface.name)).kind == "latent":
        encoder_input = opt_path(cfg.encoder.get("checkpoint"))

    write_stage_record(
        output_dir,
        stage="train_low_level",
        entrypoint=Path(__file__).resolve(),
        cfg=cfg,
        command=command,
        inputs={
            "manifest": opt_path(cfg.dataset.manifest),
            "dataset_path": {"path": str(opt_path(cfg.dataset.dataset_path))},
            "skill_encoder_checkpoint": encoder_input,
        },
        outputs=outputs,
        extra={
            "interface": str(cfg.interface.name),
            "latent_arm": opt_str(cfg.get("latent_arm")),
            "hold_steps": int(cfg.hold.steps),
            "seed": int(cfg.seed),
            "target_frames": frames,
        },
        status="dry_run" if bool(cfg.dry_run) else "complete",
    )


if __name__ == "__main__":
    main()
