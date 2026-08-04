#!/usr/bin/env python3
"""Audit every command configuration the -G1-v2 environment can be asked for.

One declared command interface now carries the whole command surface (see
``tasks/manager_based/imitation/command_interface.py``): the always-present
dataset-backed ``reference`` channel plus exactly one ``actor`` channel. This
audit walks the matrix a user might train or evaluate -- explicit command
spaces (FB / root+qpos / keypoints / keypoint-pose / EE) as direct commands and
as published packets at every hold period, and the latent schemes (a pretrained
DiffSR skill encoder at every spectral bottleneck, plus the in-loop
reconstruction encoders CVAE / VQVAE / VAE / AE / FSQ at every encoding
horizon) -- and checks that each one resolves coherently.

Per row it resolves the environment config, binds the agent config to the
environment's command interface (the same call the training entry point makes),
and checks:

- every actor / critic / encoder input key resolves to a present observation
  term after the interface narrows the groups,
- the actor's kind agrees with the agent's latent switch, and the published
  latent width matches on both sides,
- the encoder view carries the row's window wherever the agent encodes one,
- packet rows satisfy the hold invariants (future-only window, horizon covering
  the hold, an external or oracle publisher),
- the parked reward-estimation stack is coherent (``agent.reward_estimation``
  <-> the environment's ``reward_input`` group),
- the DiffSR spectral modes the latent rows request exist in RLOpt.

``--construct ROW[,ROW...]`` additionally boots the simulation headless, builds
each named row's environment, and steps it, asserting the agent's input keys
are present in the produced observations.

Examples (run from the repository root):

.. code-block:: bash

    pixi run -e isaaclab python scripts/audit/audit_g1_v2_command_matrix.py
    pixi run -e isaaclab python scripts/audit/audit_g1_v2_command_matrix.py \
        --construct fb_k10,hl_skill_deterministic,vqvae_fsq
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

_TASK = "Isaac-Imitation-G1-v2"
_MANIFEST = Path("./data/unitree/manifests/g1_unitree_dance102_manifest.json")

_AGENTS = "isaaclab_imitation.tasks.manager_based.imitation.config.g1.agents."
_IPMD = f"{_AGENTS}rlopt_ipmd_cfg"
_VQVAE = f"{_AGENTS}rlopt_ipmd_vqvae_cfg"


# ---------------------------------------------------------------------------
# Matrix data.
# ---------------------------------------------------------------------------

# Explicit command spaces, as component sets.
EXPLICIT_SPACES: dict[str, tuple[str, ...]] = {
    "fb": ("joint_qpos_qvel", "root_pos", "root_ori"),
    "root_qpos": ("joint_qpos", "root_pos", "root_ori"),
    "root_points5": ("keypoint_pos", "root_pos", "root_ori"),
    "root_points5_pose": ("keypoint_pos", "keypoint_ori", "root_pos", "root_ori"),
    "ee": ("ee_pos", "ee_ori"),
}

# Hold periods exercised per explicit space: k=1 is the direct command, k>1 is
# a published packet of k frames consumed one slot per control step.
EXPLICIT_PERIODS: tuple[int, ...] = (1, 5, 10)

# Latent spectral bottlenecks available to the DiffSR pretrain stage.
PRETRAIN_SPECTRAL_MODES: tuple[str, ...] = (
    "deterministic",
    "gaussian",
    "categorical",
    "gumbel_multicat",
    "gumbel",
    "fsq",
    "sonic_fsq",
    "vq",
)

_DUMMY_SKILL_CHECKPOINT = "agent.ipmd.hl_skill_checkpoint_path=/tmp/dummy_skill.pt"

# Latent rows: the actor publishes a skill latent; the encoder view is the
# windowed reference the encoder consumes. `latent_dim` is the published width.
LATENT_ROWS: dict[str, dict] = {
    "hl_skill_deterministic": {
        "agent": f"{_IPMD}:G1ImitationLatentSonicRLOptIPMDConfig",
        "latent_dim": 258,
        "encoder_window": (0, 9),
        "agent_overrides": [
            _DUMMY_SKILL_CHECKPOINT,
            "agent.ipmd.latent_learning.code_period=10",
        ],
        "spectral": "deterministic",
    },
    "hl_skill_deterministic_h25": {
        "agent": f"{_IPMD}:G1ImitationLatentSonicRLOptIPMDConfig",
        "latent_dim": 258,
        "encoder_window": (0, 24),
        "agent_overrides": [
            _DUMMY_SKILL_CHECKPOINT,
            "agent.ipmd.hl_skill_horizon_steps=25",
            "agent.ipmd.latent_learning.code_period=25",
        ],
        "spectral": "deterministic",
    },
    "hl_skill_vq": {
        "agent": f"{_IPMD}:G1ImitationLatentSonicRLOptIPMDConfig",
        "latent_dim": 258,
        "encoder_window": (0, 9),
        "agent_overrides": [
            _DUMMY_SKILL_CHECKPOINT,
            "agent.ipmd.latent_learning.code_period=10",
        ],
        "spectral": "vq",
    },
    "hl_skill_fsq": {
        "agent": f"{_IPMD}:G1ImitationLatentSonicRLOptIPMDConfig",
        "latent_dim": 258,
        "encoder_window": (0, 9),
        "agent_overrides": [
            _DUMMY_SKILL_CHECKPOINT,
            "agent.ipmd.latent_learning.code_period=10",
        ],
        "spectral": "fsq",
    },
    "hl_skill_sonic_fsq": {
        # sonic_fsq publishes the quantizer output directly: z (64) + sin/cos
        # phase (2) = 66.
        "agent": f"{_IPMD}:G1ImitationLatentSonicRLOptIPMDConfig",
        "latent_dim": 66,
        "encoder_window": (0, 9),
        "agent_overrides": [
            _DUMMY_SKILL_CHECKPOINT,
            "agent.ipmd.latent_dim=66",
            "agent.ipmd.latent_learning.code_period=10",
        ],
        "spectral": "sonic_fsq",
    },
    "future_cvae": {
        "agent": f"{_IPMD}:G1ImitationLatentFutureCVAERLOptIPMDConfig",
        "latent_dim": 256,
        "encoder_window": (0, 9),
        "recon_method": "future_cvae",
    },
    "per_step_vq": {
        "agent": f"{_IPMD}:G1ImitationLatentPerStepVQRLOptIPMDConfig",
        "latent_dim": 64,
        "encoder_window": (0, 9),
        "recon_method": "per_step_vq_sequence",
    },
    "vqvae_fsq": {
        "agent": f"{_VQVAE}:G1ImitationLatentRLOptIPMDVQVAEConfig",
        "latent_dim": 64,
        "encoder_window": (8, 0),
        "recon_method": "patch_vqvae",
    },
    "vqvae_vq": {
        "agent": f"{_VQVAE}:G1ImitationLatentRLOptIPMDVQVAEConfig",
        "latent_dim": 64,
        "encoder_window": (8, 0),
        "agent_overrides": ["agent.ipmd.latent_learning.quantizer=vq_ema"],
        "recon_method": "patch_vqvae",
    },
    "ae": {
        "agent": f"{_IPMD}:G1ImitationLatentSonicRLOptIPMDConfig",
        "latent_dim": 64,
        "encoder_window": (0, 9),
        "agent_overrides": [
            "agent.ipmd.command_source=posterior",
            "agent.ipmd.latent_dim=64",
            "agent.ipmd.latent_learning.method=patch_autoencoder",
            "agent.ipmd.latent_learning.patch_future_steps=9",
            "agent.ipmd.latent_learning.posterior_command_period=10",
        ],
        "recon_method": "patch_autoencoder",
    },
    "vae": {
        "agent": f"{_IPMD}:G1ImitationLatentSonicRLOptIPMDConfig",
        "latent_dim": 64,
        "encoder_window": (0, 9),
        "agent_overrides": [
            "agent.ipmd.command_source=posterior",
            "agent.ipmd.latent_dim=64",
            "agent.ipmd.latent_learning.method=patch_autoencoder",
            "agent.ipmd.latent_learning.patch_future_steps=9",
            "agent.ipmd.latent_learning.posterior_command_period=10",
            "agent.ipmd.latent_learning.kl_coeff=0.01",
        ],
        "recon_method": "patch_autoencoder",
    },
    "sonic_official_fsq": {
        "agent": f"{_IPMD}:G1ImitationLatentSonicOfficialFSQRLOptIPMDConfig",
        "latent_dim": 64,
        "encoder_window": (0, 9),
        "recon_method": "patch_vqvae",
    },
}


# ---------------------------------------------------------------------------
# Row construction.
# ---------------------------------------------------------------------------


def _build_rows() -> dict[str, dict]:
    """The full matrix: explicit spaces x hold periods, plus the latent rows."""
    rows: dict[str, dict] = {}
    for space_name, components in EXPLICIT_SPACES.items():
        for hold_steps in EXPLICIT_PERIODS:
            rows[f"{space_name}_k{hold_steps}"] = {
                "agent": f"{_IPMD}:G1ImitationRLOptIPMDConfig",
                "components": components,
                "hold_steps": hold_steps,
            }
    rows.update(LATENT_ROWS)
    return rows


def _env_cfg_for(row: dict):
    """The environment config a row asks for, resolved as the env would."""
    from isaaclab_imitation.tasks.manager_based.imitation.command_interface import (
        ChunkCommandCfg,
        EncoderViewCfg,
        ExplicitCommandCfg,
        LatentCommandCfg,
    )
    from isaaclab_imitation.tasks.manager_based.imitation.config.g1.imitation_g1_env_v2 import (  # noqa: E501
        ImitationG1V2EnvCfg,
    )

    env_cfg = ImitationG1V2EnvCfg()
    interface = env_cfg.command_interface
    components = row.get("components")
    if components is None:
        interface.actor = LatentCommandCfg(dim=int(row["latent_dim"]))
        past_steps, future_steps = row["encoder_window"]
        interface.encoder = EncoderViewCfg(
            past_steps=int(past_steps), future_steps=int(future_steps)
        )
        interface.critic_channels = ("actor", "reference")
    else:
        hold_steps = int(row["hold_steps"])
        if hold_steps <= 1:
            interface.actor = ExplicitCommandCfg(components=tuple(components))
        else:
            interface.actor = ChunkCommandCfg(
                source="reference",
                components=tuple(components),
                horizon=hold_steps,
                hold_steps=hold_steps,
            )
        interface.encoder = None
        interface.critic_channels = ("reference",)
    env_cfg.resolve_late_overrides()
    return env_cfg


def _agent_cfg_for(row: dict, env_cfg):
    """The row's agent config, bound to the environment's command interface."""
    import yaml

    from isaaclab_imitation.tasks.manager_based.imitation.command_interface import (
        bind_command_interface,
    )

    module_name, class_name = row["agent"].split(":")
    module = __import__(module_name, fromlist=[class_name])
    agent_cfg = getattr(module, class_name)()

    overrides: dict = {}
    for override in row.get("agent_overrides", []):
        key, value = override.split("=", 1)
        if not key.startswith("agent."):
            raise ValueError(f"agent override expected, got {override!r}")
        parts = key[len("agent.") :].split(".")
        target = overrides
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = yaml.safe_load(value)
    if overrides:
        agent_cfg.from_dict(overrides)
    bind_command_interface(agent_cfg, env_cfg)
    return agent_cfg


# ---------------------------------------------------------------------------
# Coherence checks.
# ---------------------------------------------------------------------------


def _group_terms(group) -> set[str]:
    return {
        field.name
        for field in dataclasses.fields(group)
        if getattr(group, field.name) is not None
        and hasattr(getattr(group, field.name), "func")
    }


def _obs_groups(env_cfg) -> dict[str, set[str]]:
    groups = {}
    for field in dataclasses.fields(env_cfg.observations):
        group = getattr(env_cfg.observations, field.name)
        if group is not None:
            groups[field.name] = _group_terms(group)
    return groups


def _missing_keys(keys, groups: dict[str, set[str]]) -> list[str]:
    return [
        f"{group}.{term}"
        for group, term in (keys or [])
        if term not in groups.get(group, set())
    ]


def check_row(row_name: str, row: dict, report: list[str]) -> None:
    from isaaclab_imitation.tasks.manager_based.imitation.command_interface import (
        encoder_command_keys,
    )

    try:
        env_cfg = _env_cfg_for(row)
    except Exception as exc:
        report.append(f"[FAIL] {row_name}: env resolution: {type(exc).__name__}: {exc}")
        return
    try:
        agent_cfg = _agent_cfg_for(row, env_cfg)
    except Exception as exc:
        report.append(
            f"[FAIL] {row_name}: agent resolution: {type(exc).__name__}: {exc}"
        )
        return

    errors: list[str] = []
    interface = env_cfg.command_interface
    groups = _obs_groups(env_cfg)

    # Every network's inputs must exist on the narrowed observation surface.
    for label, keys in (
        ("actor", agent_cfg.policy.input_keys),
        (
            "critic",
            agent_cfg.value_function.input_keys if agent_cfg.value_function else [],
        ),
        ("encoder", encoder_command_keys(interface)),
    ):
        missing = _missing_keys(keys, groups)
        if missing:
            errors.append(f"{label} input keys absent from the surface: {missing}")

    use_latent = bool(agent_cfg.ipmd.use_latent_command)
    if use_latent != interface.is_latent():
        errors.append(
            f"actor kind {interface.actor_kind()!r} disagrees with "
            f"agent.ipmd.use_latent_command={use_latent}"
        )

    if interface.is_latent():
        env_dim = int(interface.actor.dim)
        agent_dim = int(agent_cfg.ipmd.latent_dim)
        if env_dim != agent_dim:
            errors.append(f"latent width mismatch: env {env_dim} vs agent {agent_dim}")
        if interface.encoder is None:
            errors.append("a latent row needs an encoder view to encode")
        else:
            want = tuple(int(step) for step in row["encoder_window"])
            got = (
                int(interface.encoder.past_steps),
                int(interface.encoder.future_steps),
            )
            if got != want:
                errors.append(f"encoder window {got} != requested {want}")
    else:
        actor = interface.actor
        hold_steps = int(getattr(actor, "hold_steps", 1) or 1)
        if hold_steps > 1:
            if int(actor.past_steps) != 0:
                errors.append("a packet command requires a future-only window")
            if int(actor.horizon) < hold_steps:
                errors.append(
                    f"packet does not cover the hold: horizon={actor.horizon} "
                    f"< hold_steps={hold_steps}"
                )
            if actor.source not in {"external", "reference"}:
                errors.append(f"a packet cannot be published by {actor.source!r}")
        if tuple(interface.actor_components()) != tuple(row["components"]):
            errors.append(
                f"actor components {interface.actor_components()} != requested "
                f"{tuple(row['components'])}"
            )

    reward_estimation = bool(getattr(agent_cfg, "reward_estimation", False))
    reward_input_group = "reward_input" in groups
    if reward_estimation != reward_input_group:
        errors.append(
            "reward stack mismatch: "
            f"agent.reward_estimation={reward_estimation} but env reward_input "
            f"group present={reward_input_group}"
        )

    report.append(
        f"[{'OK  ' if not errors else 'FAIL'}] {row_name}: "
        + ("; ".join(errors) if errors else "coherent")
    )


def _construct_row(row_name: str, row: dict) -> int:
    """Build the row's environment in the running app and step it."""
    import gymnasium as gym
    import torch

    import isaaclab_imitation.tasks  # noqa: F401

    status = 1
    env = None
    try:
        env_cfg = _env_cfg_for(row)
        env_cfg.data.manifest = str(_MANIFEST.resolve())
        env = gym.make(_TASK, cfg=env_cfg)
        agent_cfg = _agent_cfg_for(row, env_cfg)
        keys = list(agent_cfg.policy.input_keys or [])
        action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        obs, _ = env.reset()
        for _ in range(3):
            obs, *_ = env.step(action)
        obs_groups = dict(obs) if hasattr(obs, "keys") else {}
        missing = [
            f"{group}.{term}"
            for group, term in keys
            if group not in obs_groups
            or obs_groups[group] is None
            or term not in obs_groups[group]
        ]
        if missing:
            print(f"[FAIL] {row_name}: constructed but missing obs keys: {missing}")
        else:
            print(
                f"[OK  ] {row_name}: env built, stepped 3x, "
                f"{len(obs_groups)} obs groups, agent keys present"
            )
            status = 0
    except Exception as exc:
        print(f"[FAIL] {row_name}: {type(exc).__name__}: {exc}")
    finally:
        if env is not None:
            env.close()
    return status


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--construct",
        default="",
        help="Comma-separated row names to additionally build headless.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args(sys.argv[1:])
    report: list[str] = []

    if args.construct:
        # The AppLauncher must start before any Isaac Lab module import (pxr
        # binding requirement), so boot it first and run the config checks
        # against the running app.
        from isaaclab.app import AppLauncher

        app_launcher = AppLauncher({"headless": True, "device": "cuda:0"})
        simulation_app = app_launcher.app

    from rlopt.agent.hl_skill_encoder import LATENT_MODES

    for mode in PRETRAIN_SPECTRAL_MODES:
        if mode not in LATENT_MODES:
            report.append(
                f"[FAIL] spectral mode {mode!r} missing from RLOpt LATENT_MODES"
            )

    rows = _build_rows()
    for row_name, row in rows.items():
        check_row(row_name, row, report)

    print("\n".join(report))
    failures = [line for line in report if line.startswith("[FAIL]")]
    print(f"\n{len(report)} checks; {len(failures)} failures")
    if failures:
        raise SystemExit(1)

    if args.construct:
        construct_status = 0
        for row_name in args.construct.split(","):
            row_name = row_name.strip()
            if row_name not in rows:
                print(f"[SKIP] unknown row {row_name!r}")
                continue
            construct_status |= _construct_row(row_name, rows[row_name])
        simulation_app.close()  # type: ignore[possibly-undefined]
        if construct_status:
            raise SystemExit(construct_status)


if __name__ == "__main__":
    main()
