#!/usr/bin/env python3
"""Audit every command-space x encoder x period combination on the -G1-v2 env.

The v2 fork made ``ImitationRLEnv`` / ``ImitationG1EnvCfg`` the default task.
This audit checks that every combination a user might train or evaluate is
coherently expressible on ``-G1-v2``: explicit command spaces (FB / root+qpos
/ ee+root / keypoint-pose / ee-only) at every command hold period k, and
latent schemes (pretrained DiffSR skill encoder with every spectral
bottleneck, plus in-loop reconstruction encoders CVAE / VQVAE / VAE / AE /
FSQ) at every encoding horizon h and command period k.

Each row resolves the env config through the real Hydra path
(``resolve_task_config`` on ``Isaac-Imitation-G1-v2`` with the row's
``env.*`` overrides) and instantiates the row's agent config class directly,
then checks:

- ``env.command_mode`` agrees with ``agent.ipmd.use_latent_command``.
- every actor/critic input key resolves to a present observation term after
  command-term pruning,
- ``env.latent_command_dim`` matches the agent's published latent width,
- expert-window terms carry the row's encoding horizon h (and past window)
  wherever the agent reads ``expert_window`` observations,
- hold invariants: ``command_hold_steps=k>0`` requires
  ``latent_patch_past_steps==0``, a chunk ``policy_command_mode``, a
  planner/planner_oracle ``command_observation_source``, and a window that
  covers the hold (h + 1 >= k),
- the parked reward-estimation stack is coherent
  (``agent.reward_estimation`` <-> the env's ``reward_input`` group),
- the DiffSR spectral modes requested by latent rows exist in RLOpt.

``--construct ROW[,ROW...]`` additionally boots the simulation headless,
builds the env, and steps it three times with zero actions, asserting the
agent's input keys are present in the produced observations.

Examples (run from the repository root):

.. code-block:: bash

    pixi run -e isaaclab python scripts/audit/audit_g1_v2_command_matrix.py
    pixi run -e isaaclab python scripts/audit/audit_g1_v2_command_matrix.py --construct fb_chunk_k10,root_qpos_chunk_k10
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

_TASK = "Isaac-Imitation-G1-v2"
_AGENT_KEY = "rlopt_ipmd_cfg_entry_point"


# ---------------------------------------------------------------------------
# Matrix data.
# ---------------------------------------------------------------------------

# Explicit command spaces (env.command_mode=explicit, agent use_latent=false).
# term: (name, command_space, command_observation_terms, chunk policy mode)
EXPLICIT_SPACES: dict[str, tuple[str, tuple[str, ...], str | None]] = {
    "fb": (
        "single_frame_full_body",
        ("expert_motion", "expert_anchor_pos_b", "expert_anchor_ori_b"),
        "full_body_chunk_current_slot",
    ),
    "root_qpos": (
        "root_qpos",
        ("expert_motion_qpos", "expert_anchor_pos_b", "expert_anchor_ori_b"),
        "full_body_chunk_current_slot",
    ),
    "root_points5": (
        "root_points5",
        ("expert_keypoint_pos_b", "expert_anchor_pos_b", "expert_anchor_ori_b"),
        "full_body_chunk_current_slot",
    ),
    "root_points5_pose": (
        "root_points5_pose",
        (
            "expert_keypoint_pos_b",
            "expert_keypoint_ori_b",
            "expert_anchor_pos_b",
            "expert_anchor_ori_b",
        ),
        "explicit_chunk_current_slot",
    ),
    "ee": (
        "ee_trajectory",
        ("expert_ee_pos_b", "expert_ee_ori_b"),
        "ee_chunk_current_slot",
    ),
}

# (h, k) pairs exercised for explicit chunk rows: encoding horizon h (frames
# in the packet beyond the current one) and hold period k (control steps the
# packet is consumed slot-by-slot). Window covers the hold: h + 1 >= k.
EXPLICIT_PERIODS: tuple[tuple[int, int], ...] = ((0, 1), (4, 5), (9, 10))

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

# Latent scheme rows: name, agent class import path, env overrides, agent
# overrides. (h, k) semantics: h = encoded future horizon, k = command period
# (hl_skill code_period / posterior_command_period / env command_hold_steps).
LATENT_ROWS: dict[str, dict] = {
    "hl_skill_deterministic": {
        "agent": "isaaclab_imitation.tasks.manager_based.imitation.config.g1.agents.rlopt_ipmd_cfg:G1ImitationLatentSonicRLOptIPMDConfig",
        "h": 9,
        "k": 10,
        "env_overrides": ["env.command_mode=latent", "env.latent_patch_future_steps=9"],
        "agent_overrides": [
            "agent.ipmd.hl_skill_checkpoint_path=/tmp/dummy_skill_encoder.pt",
            "agent.ipmd.latent_learning.code_period=10",
        ],
        "spectral": "deterministic",
    },
    "hl_skill_deterministic_h25k25": {
        "agent": "isaaclab_imitation.tasks.manager_based.imitation.config.g1.agents.rlopt_ipmd_cfg:G1ImitationLatentSonicRLOptIPMDConfig",
        "h": 25,
        "k": 25,
        "env_overrides": ["env.command_mode=latent"],
        "agent_overrides": [
            "agent.ipmd.hl_skill_checkpoint_path=/tmp/dummy_skill_encoder.pt",
            "agent.ipmd.hl_skill_horizon_steps=25",
            "agent.ipmd.latent_learning.code_period=25",
        ],
        "spectral": "deterministic",
    },
    "hl_skill_vq": {
        "agent": "isaaclab_imitation.tasks.manager_based.imitation.config.g1.agents.rlopt_ipmd_cfg:G1ImitationLatentSonicRLOptIPMDConfig",
        "h": 9,
        "k": 10,
        "env_overrides": ["env.command_mode=latent", "env.latent_patch_future_steps=9"],
        "agent_overrides": [
            "agent.ipmd.hl_skill_checkpoint_path=/tmp/dummy_skill_encoder.pt",
            "agent.ipmd.latent_learning.code_period=10",
        ],
        "spectral": "vq",
    },
    "hl_skill_fsq": {
        "agent": "isaaclab_imitation.tasks.manager_based.imitation.config.g1.agents.rlopt_ipmd_cfg:G1ImitationLatentSonicRLOptIPMDConfig",
        "h": 9,
        "k": 10,
        "env_overrides": ["env.command_mode=latent", "env.latent_patch_future_steps=9"],
        "agent_overrides": [
            "agent.ipmd.hl_skill_checkpoint_path=/tmp/dummy_skill_encoder.pt",
            "agent.ipmd.latent_learning.code_period=10",
        ],
        "spectral": "fsq",
    },
    "hl_skill_sonic_fsq": {
        "agent": "isaaclab_imitation.tasks.manager_based.imitation.config.g1.agents.rlopt_ipmd_cfg:G1ImitationLatentSonicRLOptIPMDConfig",
        "h": 9,
        "k": 10,
        # sonic_fsq publishes the 64-level quantizer output directly, so the
        # published command is z (64) + sin/cos phase (2) = 66.
        "env_overrides": [
            "env.command_mode=latent",
            "env.latent_patch_future_steps=9",
            "env.latent_command_dim=66",
        ],
        "agent_overrides": [
            "agent.ipmd.hl_skill_checkpoint_path=/tmp/dummy_skill_encoder.pt",
            "agent.ipmd.latent_dim=66",
            "agent.ipmd.latent_learning.code_period=10",
        ],
        "spectral": "sonic_fsq",
    },
    "future_cvae": {
        "agent": "isaaclab_imitation.tasks.manager_based.imitation.config.g1.agents.rlopt_ipmd_cfg:G1ImitationLatentFutureCVAERLOptIPMDConfig",
        "h": 9,
        "k": 10,
        "env_overrides": [
            "env.command_mode=latent",
            "env.latent_command_dim=256",
            "env.latent_patch_future_steps=9",
        ],
        "agent_overrides": [],
        "recon_method": "future_cvae",
    },
    "per_step_vq": {
        "agent": "isaaclab_imitation.tasks.manager_based.imitation.config.g1.agents.rlopt_ipmd_cfg:G1ImitationLatentPerStepVQRLOptIPMDConfig",
        "h": 9,
        "k": 1,
        "env_overrides": [
            "env.command_mode=latent",
            "env.latent_command_dim=64",
            "env.latent_patch_future_steps=9",
        ],
        "agent_overrides": [],
        "recon_method": "per_step_vq_sequence",
    },
    "vqvae_fsq": {
        "agent": "isaaclab_imitation.tasks.manager_based.imitation.config.g1.agents.rlopt_ipmd_vqvae_cfg:G1ImitationLatentRLOptIPMDVQVAEConfig",
        "h": 0,
        "k": 30,
        "env_overrides": [
            "env.command_mode=latent",
            "env.latent_command_dim=64",
            "env.latent_patch_past_steps=8",
        ],
        "agent_overrides": [],
        "recon_method": "patch_vqvae",
    },
    "vqvae_vq": {
        "agent": "isaaclab_imitation.tasks.manager_based.imitation.config.g1.agents.rlopt_ipmd_vqvae_cfg:G1ImitationLatentRLOptIPMDVQVAEConfig",
        "h": 0,
        "k": 30,
        "env_overrides": [
            "env.command_mode=latent",
            "env.latent_command_dim=64",
            "env.latent_patch_past_steps=8",
        ],
        "agent_overrides": ["agent.ipmd.latent_learning.quantizer=vq_ema"],
        "recon_method": "patch_vqvae",
    },
    "ae": {
        "agent": "isaaclab_imitation.tasks.manager_based.imitation.config.g1.agents.rlopt_ipmd_cfg:G1ImitationLatentSonicRLOptIPMDConfig",
        "h": 9,
        "k": 10,
        "env_overrides": [
            "env.command_mode=latent",
            "env.latent_command_dim=64",
            "env.latent_patch_future_steps=9",
        ],
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
        "agent": "isaaclab_imitation.tasks.manager_based.imitation.config.g1.agents.rlopt_ipmd_cfg:G1ImitationLatentSonicRLOptIPMDConfig",
        "h": 9,
        "k": 10,
        "env_overrides": [
            "env.command_mode=latent",
            "env.latent_command_dim=64",
            "env.latent_patch_future_steps=9",
        ],
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
        "agent": "isaaclab_imitation.tasks.manager_based.imitation.config.g1.agents.rlopt_ipmd_cfg:G1ImitationLatentSonicOfficialFSQRLOptIPMDConfig",
        "h": 9,
        "k": 1,
        "env_overrides": [
            "env.command_mode=latent",
            "env.latent_command_dim=64",
            "env.latent_patch_future_steps=9",
        ],
        "agent_overrides": [],
        "recon_method": "patch_vqvae",
    },
}


# ---------------------------------------------------------------------------
# Coherence checks.
# ---------------------------------------------------------------------------


def _resolve_env_cfg(overrides: list[str]):
    from isaaclab_tasks.utils import resolve_task_config

    original_argv = sys.argv
    sys.argv = ["audit_g1_v2_command_matrix"] + overrides
    try:
        env_cfg, _default_agent = resolve_task_config(_TASK, _AGENT_KEY)
        return env_cfg
    finally:
        sys.argv = original_argv


def _resolve_agent_cfg(import_path: str, overrides: list[str]):
    import yaml

    module_name, class_name = import_path.split(":")
    module = __import__(module_name, fromlist=[class_name])
    agent_cfg = getattr(module, class_name)()
    agent_overrides: dict = {}
    for override in overrides:
        key, value = override.split("=", 1)
        if not key.startswith("agent."):
            raise ValueError(f"agent override expected, got {override!r}")
        parts = key[len("agent.") :].split(".")
        target = agent_overrides
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        # Hydra parses override values as YAML scalars; configclass.from_dict
        # applies strict typing, so coerce the same way.
        target[parts[-1]] = yaml.safe_load(value)
    if agent_overrides:
        agent_cfg.from_dict(agent_overrides)
    sync = getattr(agent_cfg, "sync_input_keys", None)
    if callable(sync):
        sync()
    return agent_cfg


def _group_terms(group) -> list[str]:
    terms = []
    for field in dataclasses.fields(group):
        value = getattr(group, field.name)
        if value is not None and hasattr(value, "func"):
            terms.append(field.name)
    return terms


def _obs_groups(env_cfg) -> dict[str, set[str]]:
    groups = {}
    for field in dataclasses.fields(env_cfg.observations):
        group = getattr(env_cfg.observations, field.name)
        if group is not None:
            groups[field.name] = set(_group_terms(group))
    return groups


def _window_params(env_cfg, group_name: str, term_name: str) -> dict | None:
    group = getattr(env_cfg.observations, group_name, None)
    if group is None:
        return None
    term = getattr(group, term_name, None)
    if term is None:
        return None
    return dict(term.params)


def check_row(row_name: str, row: dict, report: list[str]) -> None:
    from isaaclab_imitation.tasks.manager_based.imitation.config.g1.agents.rlopt_ipmd_cfg import (  # noqa: F401
        normalize_command_space,
    )

    errors: list[str] = []
    env_overrides = list(row.get("env_overrides", []))
    agent_overrides = list(row.get("agent_overrides", []))
    try:
        env_cfg = _resolve_env_cfg(env_overrides)
    except Exception as exc:  # pragma: no cover - failure surfaced in report
        report.append(f"[FAIL] {row_name}: env resolution: {type(exc).__name__}: {exc}")
        return
    try:
        agent_cfg = _resolve_agent_cfg(row["agent"], agent_overrides)
    except Exception as exc:  # pragma: no cover
        report.append(
            f"[FAIL] {row_name}: agent resolution: {type(exc).__name__}: {exc}"
        )
        return

    use_latent = bool(agent_cfg.ipmd.use_latent_command)
    command_mode = str(getattr(env_cfg, "command_mode", "explicit"))
    if use_latent != (command_mode == "latent"):
        errors.append(
            f"mode mismatch: env.command_mode={command_mode!r} vs "
            f"agent.ipmd.use_latent_command={use_latent}"
        )

    groups = _obs_groups(env_cfg)
    for label, keys in (("policy", agent_cfg.policy.input_keys or []),):
        for grp, term in keys:
            if grp not in groups or term not in groups[grp]:
                errors.append(f"{label} key ({grp}, {term!r}) missing from obs groups")
    value_function = getattr(agent_cfg, "value_function", None)
    if value_function is not None:
        for grp, term in value_function.input_keys or []:
            if grp not in groups or term not in groups[grp]:
                errors.append(f"critic key ({grp}, {term!r}) missing from obs groups")

    if use_latent:
        env_latent_dim = int(getattr(env_cfg, "latent_command_dim", 0))
        agent_latent_dim = int(agent_cfg.ipmd.latent_dim)
        if env_latent_dim != agent_latent_dim:
            errors.append(
                f"latent dim mismatch: env.latent_command_dim={env_latent_dim} vs "
                f"agent.ipmd.latent_dim={agent_latent_dim}"
            )
        latent_learning = agent_cfg.ipmd.latent_learning
        agent_h = int(getattr(latent_learning, "patch_future_steps", 0))
        agent_past = int(getattr(latent_learning, "patch_past_steps", 0))
        env_h = int(getattr(env_cfg, "latent_patch_future_steps", 0))
        env_past = int(getattr(env_cfg, "latent_patch_past_steps", 0))
        # Rows that read expert_window observations must carry the matching
        # window horizon; the hl_skill scheme reads the policy group only.
        reads_window = any(
            grp == "expert_window" for grp, _term in (agent_cfg.policy.input_keys or [])
        ) or any(
            grp == "expert_window"
            for grp, _term in (value_function.input_keys or [])
            if value_function is not None
        )
        if reads_window:
            if env_h != agent_h or env_past != agent_past:
                errors.append(
                    f"window mismatch for expert_window reader: env (past={env_past}, "
                    f"future={env_h}) vs agent (past={agent_past}, future={agent_h})"
                )
        # Command period (k) lives agent-side for latent schemes; which field
        # owns it depends on the scheme (patch_vqvae and the pretrained
        # hl_skill path hold via code_period; in-loop encoders via
        # posterior_command_period).
        method = str(getattr(latent_learning, "method", ""))
        command_source = str(getattr(agent_cfg.ipmd, "command_source", ""))
        if method == "patch_vqvae" or command_source == "hl_skill":
            agent_k = int(getattr(latent_learning, "code_period", 0))
        else:
            agent_k = int(getattr(latent_learning, "posterior_command_period", 0))
        row_k = int(row.get("k", 0))
        if agent_k != row_k:
            errors.append(
                f"command period mismatch: expected k={row_k}, agent has {agent_k}"
            )

    else:
        command_space = str(getattr(agent_cfg, "command_space", ""))
        try:
            normalize_command_space(command_space)
        except ValueError as exc:
            errors.append(str(exc))
        # Explicit chunk rows: validate hold invariants.
        hold = int(getattr(env_cfg, "command_hold_steps", 0))
        env_past = int(getattr(env_cfg, "latent_patch_past_steps", 0))
        env_h = int(getattr(env_cfg, "latent_patch_future_steps", 0))
        policy_mode = str(getattr(env_cfg, "policy_command_mode", "reference"))
        source = str(getattr(env_cfg, "command_observation_source", "reference"))
        row_k = int(row.get("k", 0))
        if hold != row_k:
            errors.append(f"env.command_hold_steps={hold} != row k={row_k}")
        if hold > 0:
            if env_past != 0:
                errors.append(
                    "command_hold_steps>0 requires latent_patch_past_steps==0"
                )
            if policy_mode == "reference":
                errors.append(
                    "command_hold_steps>0 requires a *_chunk_current_slot "
                    "policy_command_mode"
                )
            if source not in {"planner", "planner_oracle"}:
                errors.append(
                    "chunk consumption requires command_observation_source in "
                    "{planner, planner_oracle}"
                )
            if env_h + 1 < hold:
                errors.append(
                    f"window does not cover the hold: h+1={env_h + 1} < k={hold}"
                )
        elif policy_mode != "reference":
            errors.append("chunk policy_command_mode requires command_hold_steps>0")

    reward_estimation = bool(getattr(agent_cfg, "reward_estimation", False))
    reward_input_group = (
        getattr(getattr(env_cfg, "observations", None), "reward_input", None)
        is not None
    )
    if reward_estimation != reward_input_group:
        errors.append(
            "reward stack mismatch: "
            f"agent.reward_estimation={reward_estimation} but "
            f"env reward_input group present={reward_input_group}"
        )

    report.append(
        f"[{'OK  ' if not errors else 'FAIL'}] {row_name}: "
        + ("; ".join(errors) if errors else "coherent")
    )


def _construct_row(row_name: str, row: dict) -> int:
    """Build the env for the row in the already-running app, step three times."""
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    import torch
    from isaaclab_tasks.utils import parse_env_cfg

    import isaaclab_imitation.tasks  # noqa: F401

    status = 1
    env = None
    try:
        # Row env knobs go through the real Hydra path (sys.argv overrides);
        # the manifest is applied by plain setattr AFTER parse_env_cfg because
        # the Hydra path swallows `env.lafan1_manifest_path=...` (the base cfg
        # defaults it to None). The env's construction-time manifest resolver
        # then fills loader_kwargs from it.
        original_argv = sys.argv
        sys.argv = ["audit_g1_v2_command_matrix"] + list(row.get("env_overrides", []))
        try:
            env_cfg = parse_env_cfg(_TASK, device="cuda:0", num_envs=4, use_fabric=True)
        finally:
            sys.argv = original_argv
        manifest = Path("./data/unitree/manifests/g1_unitree_dance102_manifest.json")
        setattr(env_cfg, "lafan1_manifest_path", str(manifest.resolve()))
        env = gym.make(_TASK, cfg=env_cfg)
        agent_cfg = _resolve_agent_cfg(row["agent"], row.get("agent_overrides", []))
        keys = list(agent_cfg.policy.input_keys or [])
        action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        obs, _ = env.reset()
        for _ in range(3):
            obs, *_ = env.step(action)
        obs_groups = dict(obs) if hasattr(obs, "keys") else {}
        missing = [
            f"{grp}.{term}"
            for grp, term in keys
            if grp not in obs_groups or obs_groups[grp] is None
        ]
        if missing:
            print(f"[FAIL] {row_name}: constructed but missing obs keys: {missing}")
        else:
            print(
                f"[OK  ] {row_name}: env built, stepped 3x, "
                f"{len(obs_groups)} obs groups, agent keys present"
            )
            status = 0
    except Exception as exc:  # pragma: no cover
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
        # binding requirement); with constructions requested, boot the app
        # first and run the config checks against the running app, then build
        # every row's env in the same process.
        from isaaclab.app import AppLauncher

        app_launcher = AppLauncher({"headless": True, "device": "cuda:0"})
        simulation_app = app_launcher.app

    # Spectral-mode availability (pretrain side, RLOpt owns the encoders).
    from rlopt.agent.hl_skill_encoder import LATENT_MODES

    for mode in PRETRAIN_SPECTRAL_MODES:
        if mode not in LATENT_MODES:
            report.append(
                f"[FAIL] spectral mode {mode!r} missing from RLOpt LATENT_MODES"
            )

    rows: dict[str, dict] = {}
    for space_name, (command_space, terms, chunk_mode) in EXPLICIT_SPACES.items():
        for h, k in EXPLICIT_PERIODS:
            name = f"{space_name}_k{k}"
            env_overrides = [
                "env.command_mode=explicit",
                f"env.command_observation_terms=[{','.join(terms)}]",
                "agent.ipmd.use_latent_command=false",
                f"agent.command_space={command_space}",
            ]
            if k > 0:
                env_overrides += [
                    f"env.policy_command_mode={chunk_mode}",
                    f"env.command_hold_steps={k}",
                    f"env.latent_patch_future_steps={h}",
                    "env.command_observation_source=planner_oracle",
                ]
            rows[name] = {
                "agent": (
                    "isaaclab_imitation.tasks.manager_based.imitation.config.g1."
                    "agents.rlopt_ipmd_cfg:G1ImitationRLOptIPMDConfig"
                ),
                "h": h,
                "k": k,
                "env_overrides": env_overrides,
                "agent_overrides": ["agent.ipmd.use_latent_command=false"],
            }
    rows.update(LATENT_ROWS)

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
