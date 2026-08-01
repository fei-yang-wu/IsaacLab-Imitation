"""G1 imitation task registrations: recipe x command-config pins.

Recipe (reward/termination/reset design) is the only axis with separate env
classes; the latent-vs-explicit command choice is pure configuration
(``env.command_mode`` plus the agent entry point). Three recipes exist:

- **Stable** (``imitation_g1_env_v2.ImitationG1EnvCfg`` -- the flagship
  class name follows the newest release): SONIC release recipe with this
  repo's legacy reset distribution -- the current default (``-G1-v2``,
  2026-08-01 onward; ``-G1-v1`` is frozen at its exact old kwargs).
- **Strict** (``common.tracking_env._apply_strict_recipe``): strict SONIC
  termination functions on the legacy scaffolding -- pins
  ``variants.strict.ImitationG1StrictTrackEnvCfg`` (explicit) and
  ``variants.strict.ImitationG1LatentStrictEnvCfg`` (latent).
- **LafanTrack** (``imitation_g1_env_v0.ImitationG1LafanTrackEnvCfg``): the
  original torso-anchored loose-termination tracking recipe.

Module layout: shared components and the base machinery in ``common/``; the
releases as standalone, fully spelled-out assemblies in
``imitation_g1_env_v0.py`` (LafanTrack), ``imitation_g1_env_v1.py`` (the
frozen v1, flagship name moved to v2) and ``imitation_g1_env_v2.py`` (the
flagship ``ImitationG1EnvCfg``); every non-default surface in ``variants/``
(one standalone file per family, composing from ``common`` only). The four
historical ``imitation_g1*_env_cfg`` modules remain as re-export shims for
old imports and serialized configs.

Registrations below are grouped: current defaults first, then the Strict
recipe pins, then deprecated/frozen pins kept for reproducibility.
"""

import gymnasium as gym

from . import (
    agents,
    common,
    imitation_g1_env_v0,
    imitation_g1_env_v1,
    imitation_g1_env_v2,
    variants,
)

__all__ = [
    "agents",
    "common",
    "imitation_g1_env_v0",
    "imitation_g1_env_v1",
    "imitation_g1_env_v2",
    "variants",
]

# ---------------------------------------------------------------------------
# Command-config pins (task kwargs): env entry point + agent entry points.
# ---------------------------------------------------------------------------

# LafanTrack recipe x explicit full-body command (vanilla observation surface).
_VANILLA_TASK_KWARGS = {
    "env_cfg_entry_point": f"{__name__}.imitation_g1_env_v0:ImitationG1LafanTrackEnvCfg",
    "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ImitationPPORunnerCfg",
    "rlopt_cfg_entry_point": f"{agents.__name__}.rlopt_ppo_cfg:G1ImitationRLOptPPOConfig",
    "rlopt_ppo_cfg_entry_point": f"{agents.__name__}.rlopt_ppo_cfg:G1ImitationRLOptPPOConfig",
    "rlopt_sac_cfg_entry_point": f"{agents.__name__}.rlopt_sac_cfg:G1ImitationRLOptSACConfig",
    "rlopt_fastsac_cfg_entry_point": f"{agents.__name__}.rlopt_fastsac_cfg:G1ImitationRLOptFastSACConfig",
    "rlopt_ipmd_cfg_entry_point": f"{agents.__name__}.rlopt_ipmd_cfg:G1ImitationRLOptIPMDConfig",
    "rlopt_ipmd_bilinear_cfg_entry_point": f"{agents.__name__}.rlopt_ipmd_bilinear_cfg:G1ImitationRLOptIPMDBilinearConfig",
    "rlopt_gail_cfg_entry_point": f"{agents.__name__}.rlopt_gail_cfg:G1ImitationRLOptGAILConfig",
    "rlopt_amp_cfg_entry_point": f"{agents.__name__}.rlopt_amp_cfg:G1ImitationRLOptAMPConfig",
}

# Strict recipe x explicit command (2026-07-21): the vanilla observation/agent
# contract on the same protocol deltas as the strict latent pin (pelvis anchor,
# strict SONIC terminations, [0, 200] reset starts). Built for the interface
# ablation so full-body-chunk / EE-chunk / single-frame trackers (selected via
# `agent.command_space`) train on the same env protocol as the latent tracker
# and differ only in the command space.
_VANILLA_STRICT_TASK_KWARGS = {
    **_VANILLA_TASK_KWARGS,
    "env_cfg_entry_point": (f"{__name__}.variants.strict:ImitationG1StrictTrackEnvCfg"),
}

# DEPRECATED (2026-07-19): the pre-migration beyondmimic-style latent surface
# (torso anchor, loose terminations, no proprio history, [0, 200] reset
# starts). Kept only for pre-migration checkpoints and frozen paper-protocol
# reproductions under `Isaac-Imitation-G1-Latent-Legacy-v0`. The pelvis-
# anchored strict-terminations surface built on this base
# (`_LATENT_STRICT_TASK_KWARGS` below) superseded it as the default.
_LATENT_LEGACY_TASK_KWARGS = {
    "env_cfg_entry_point": f"{__name__}.common.latent_env:ImitationG1LatentEnvCfg",
    "rlopt_cfg_entry_point": f"{agents.__name__}.rlopt_ase_cfg:G1ImitationRLOptASEConfig",
    "rlopt_ipmd_cfg_entry_point": f"{agents.__name__}.rlopt_ipmd_cfg:G1ImitationLatentRLOptIPMDConfig",
    "rlopt_ipmd_bilinear_cfg_entry_point": f"{agents.__name__}.rlopt_ipmd_bilinear_cfg:G1ImitationLatentRLOptIPMDBilinearConfig",
    "rlopt_ase_cfg_entry_point": f"{agents.__name__}.rlopt_ase_cfg:G1ImitationRLOptASEConfig",
}

# DEPRECATED as the task default (2026-07-21): the SONIC release environment
# (pelvis anchor, strict adaptive terminations, adaptive failure sampling,
# SONIC actuators, rewards, and 10-step histories) with the SONIC release
# optimizer contract (actor lr 2e-5, joint grad clip 0.1, init std 0.05,
# 6-layer SiLU MLPs with running input normalization). Briefly made the
# default on 2026-07-20 on the theory that single-GPU ICE H100's ~10B-frame
# budget (8192 envs x 12 steps x 100k iterations) matches the release's own
# convergence criterion; reverted the same week once W&B run bn931wny
# (the strict surface + the legacy/local optimizer contract, same 8192x12x12288
# scale) was found to reach episode/length=244 / episode/return=13.1 --
# far above anything the SONIC release-optimizer contract produced at matched
# scale in the concurrent VRAM ablation. Reachable only via the explicit
# `Isaac-Imitation-G1-Latent-Sonic-v0` id now; see
# wiki/isaaclab3-cu130-runtime-migration.md, "Training-gate resolution
# (2026-07-19)" and the 2026-07-21 reversal.
_LATENT_SONIC_TASK_KWARGS = {
    "env_cfg_entry_point": (f"{__name__}.variants.sonic:ImitationG1LatentSonicEnvCfg"),
    "rlopt_cfg_entry_point": (
        f"{agents.__name__}.rlopt_ipmd_cfg:G1ImitationLatentSonicRLOptIPMDConfig"
    ),
    "rlopt_ipmd_cfg_entry_point": (
        f"{agents.__name__}.rlopt_ipmd_cfg:G1ImitationLatentSonicRLOptIPMDConfig"
    ),
    # Exact public-release optimizer contract; needs cluster-scale compute.
    "rlopt_ipmd_sonic_release_cfg_entry_point": (
        f"{agents.__name__}.rlopt_ipmd_cfg:G1ImitationLatentSonicReleaseRLOptIPMDConfig"
    ),
    "rlopt_ipmd_bilinear_cfg_entry_point": (
        f"{agents.__name__}.rlopt_ipmd_bilinear_cfg:"
        "G1ImitationLatentRLOptIPMDBilinearConfig"
    ),
}

# Strict recipe x latent command. Default latent task surface from 2026-07-21
# to 2026-07-27: pelvis-anchored legacy scaffolding with strict-from-scratch
# terminations, using the legacy/local optimizer contract
# (`G1ImitationLatentRLOptIPMDConfig`: 512/256/128 ELU MLPs, actor lr 1e-3).
# This is the config behind W&B run bn931wny (episode/length=244,
# episode/return=13.1 at 8192 envs x 12 steps x minibatch 12288) -- the best
# validated result of its period, ahead of the SONIC release-optimizer
# contract at matched scale. See `_LATENT_SONIC_TASK_KWARGS` above for the
# deprecation history.
_LATENT_STRICT_TASK_KWARGS = {
    **_LATENT_LEGACY_TASK_KWARGS,
    "env_cfg_entry_point": (
        f"{__name__}.variants.strict:ImitationG1LatentStrictEnvCfg"
    ),
}

# Stable recipe x latent command (258-D skill code + phase). Default latent
# task surface (2026-07-27): the full SONIC release recipe with this repo's
# legacy reset distribution ([0, 200] starts,
# `failure_rate_max_over_mean=50`) and strict-from-scratch terminations. The
# 2026-07-27 reset-sampling screen showed SONIC's full-trajectory
# adaptive-failure sampler, not its rewards or actuators, was what cost ~5.6x
# episode length at 4096 environments; see `imitation_g1_env_v1.ImitationG1EnvV1Cfg`.
#
# The agent side stays the SONIC input-key contract
# (`G1ImitationLatentSonicRLOptIPMDConfig`, local optimizer contract), matching
# the arms this surface was validated against.
#
# The previous default (`_LATENT_STRICT_TASK_KWARGS`, the plain strict surface)
# is still reachable at `Isaac-Imitation-G1-Latent-Strict-v0`. Every result
# before 2026-07-27 that names `Isaac-Imitation-G1-Latent-v0` -- Study B, the
# Study C grouped-VQ arms and their continuation segments, and the low-level
# qualification artifacts -- was produced on that surface and must use the
# explicit id to stay reproducible.
_LATENT_STABLE_TASK_KWARGS = {
    **_LATENT_SONIC_TASK_KWARGS,
    "env_cfg_entry_point": (f"{__name__}.imitation_g1_env_v1:ImitationG1EnvCfg"),
}

_LATENT_GOAL_TASK_KWARGS = {
    "env_cfg_entry_point": (f"{__name__}.variants.goal:ImitationG1LatentGoalEnvCfg"),
    "rlopt_ipmd_bilinear_cfg_entry_point": (
        f"{agents.__name__}.rlopt_ipmd_bilinear_cfg:"
        "G1ImitationLatentGoalRLOptIPMDBilinearConfig"
    ),
}

_LATENT_FUTURE_CVAE_TASK_KWARGS = {
    "env_cfg_entry_point": (
        f"{__name__}.variants.future_cvae:ImitationG1LatentFutureCVAEEnvCfg"
    ),
    "rlopt_cfg_entry_point": (
        f"{agents.__name__}.rlopt_ipmd_cfg:G1ImitationLatentFutureCVAERLOptIPMDConfig"
    ),
    "rlopt_ipmd_cfg_entry_point": (
        f"{agents.__name__}.rlopt_ipmd_cfg:G1ImitationLatentFutureCVAERLOptIPMDConfig"
    ),
}

_LATENT_PER_STEP_VQ_TASK_KWARGS = {
    "env_cfg_entry_point": (
        f"{__name__}.variants.future_cvae:ImitationG1LatentPerStepVQEnvCfg"
    ),
    "rlopt_cfg_entry_point": (
        f"{agents.__name__}.rlopt_ipmd_cfg:G1ImitationLatentPerStepVQRLOptIPMDConfig"
    ),
    "rlopt_ipmd_cfg_entry_point": (
        f"{agents.__name__}.rlopt_ipmd_cfg:G1ImitationLatentPerStepVQRLOptIPMDConfig"
    ),
}

# Official-window SONIC adaptation: the complete 10-frame future window is
# compressed into one 64-D FSQ command and recomputed every control step. This
# combines the SONIC environment/history and release optimizer contract, unlike
# the removed strict-lineage cached-packet implementation.
_LATENT_SONIC_OFFICIAL_FSQ_TASK_KWARGS = {
    **_LATENT_SONIC_TASK_KWARGS,
    "env_cfg_entry_point": (
        f"{__name__}.variants.sonic:ImitationG1LatentSonicOfficialFSQEnvCfg"
    ),
    "rlopt_cfg_entry_point": (
        f"{agents.__name__}.rlopt_ipmd_cfg:"
        "G1ImitationLatentSonicOfficialFSQRLOptIPMDConfig"
    ),
    "rlopt_ipmd_cfg_entry_point": (
        f"{agents.__name__}.rlopt_ipmd_cfg:"
        "G1ImitationLatentSonicOfficialFSQRLOptIPMDConfig"
    ),
}

_LATENT_VQVAE_TASK_KWARGS = {
    "env_cfg_entry_point": (f"{__name__}.variants.vqvae:ImitationG1LatentVQVAEEnvCfg"),
    "rlopt_cfg_entry_point": (
        f"{agents.__name__}.rlopt_ipmd_vqvae_cfg:G1ImitationLatentRLOptIPMDVQVAEConfig"
    ),
    "rlopt_ipmd_cfg_entry_point": (
        f"{agents.__name__}.rlopt_ipmd_vqvae_cfg:G1ImitationLatentRLOptIPMDVQVAEConfig"
    ),
    "rlopt_ipmd_vqvae_cfg_entry_point": (
        f"{agents.__name__}.rlopt_ipmd_vqvae_cfg:G1ImitationLatentRLOptIPMDVQVAEConfig"
    ),
    "rlopt_ipmd_bilinear_cfg_entry_point": (
        f"{agents.__name__}.rlopt_ipmd_bilinear_cfg:"
        "G1ImitationLatentRLOptIPMDBilinearVQVAEConfig"
    ),
}

_LATENT_ABLATION_TASK_KWARGS = {
    **_LATENT_STRICT_TASK_KWARGS,
    "env_cfg_entry_point": (
        f"{__name__}.variants.ablation:ImitationG1LatentAblationEnvCfg"
    ),
    "rlopt_cfg_entry_point": (
        f"{agents.__name__}.rlopt_ipmd_latent_ablation_cfg:"
        "G1ImitationLatentAblationRLOptIPMDConfig"
    ),
    "rlopt_ipmd_cfg_entry_point": (
        f"{agents.__name__}.rlopt_ipmd_latent_ablation_cfg:"
        "G1ImitationLatentAblationRLOptIPMDConfig"
    ),
}

# ---------------------------------------------------------------------------
# Current defaults.
#
# Versioning convention (2026-07-31 onward): "the default" is always the
# highest-numbered `-G1-vN` latent id below. Bumping the config is bumping
# the number, never mutating an existing vN's kwargs -- once a vN is
# superseded by vN+1, it is frozen forever (still registered, still exactly
# reproducible) and simply stops being cited as the default. Do not reuse
# `-Latent-v0`'s "moving alias" pattern again; it predates this convention
# and is kept only for its own back-compat reasons (see its comment below).
# ---------------------------------------------------------------------------

# Stable recipe, command space configured -- THE DEFAULT latent task
# (2026-08-01 onward; supersedes `-G1-v1`, which stays frozen at its exact
# old kwargs below). No `Latent` tag: under the recipe x command-config
# architecture the command space is configuration, not identity. Defaults to
# the latent 258-D command (the class default; unlike explicit `-G1-v0`);
# select vanilla with `env.command_mode=explicit` +
# `env.command_observation_terms` + `agent.ipmd.use_latent_command=false` +
# `agent.command_components`. Cite this id in protocols, gates, and paper
# commands so recorded runs and checkpoints stay reproducible even after a
# newer vN becomes the default.
#
# The v2 surface: the v1 config plus the CommandManager increment -- a native
# `motion` command term (`mdp.MotionCommandCfg`) that exposes the 67-D
# explicit tracking command via `command_manager.get_command("motion")` and
# logs `Metrics/motion/...` natively, and a `skill` term serving the
# agent-latent buffer. The env entry point is the composed flagship
# `ImitationRLEnv` (ExpertDataPlane + command planes); behavior is verified
# identical to the legacy env under the same cfg by the fixed-seed A/B
# certificate (`scripts/audit/certify_v2_env_equivalence.py`).
gym.register(
    id="Isaac-Imitation-G1-v2",
    entry_point="isaaclab_imitation.envs:ImitationRLEnv",
    disable_env_checker=True,
    kwargs={
        **_LATENT_STABLE_TASK_KWARGS,
        "env_cfg_entry_point": (f"{__name__}.imitation_g1_env_v2:ImitationG1EnvCfg"),
    },
)

# Stable recipe, command space configured -- FROZEN (superseded as the
# default by `-G1-v2` on 2026-08-01; see the versioning convention above).
# Still registered with its exact old kwargs (including the
# `imitation_g1_env_v1:ImitationG1EnvCfg` env-cfg string, which resolves via
# the module's back-compat alias) and the legacy env entry point; do not
# cite this id for new work. Defaults to the latent 258-D command; select
# vanilla with `env.command_mode=explicit` + `env.command_observation_terms`
# + `agent.ipmd.use_latent_command=false` + `agent.command_components`.
gym.register(
    id="Isaac-Imitation-G1-v1",
    entry_point="isaaclab_imitation.envs:ImitationRLEnvLegacy",
    disable_env_checker=True,
    kwargs=_LATENT_STABLE_TASK_KWARGS,
)

# Stable recipe x latent 258-D command -- ordinary alias of the FROZEN
# `-G1-v1` above, kept for back-compat with commands and checkpoints
# recorded before 2026-07-31. Same kwargs; new work should cite `-G1-v2`
# instead.
gym.register(
    id="Isaac-Imitation-G1-Latent-v0",
    entry_point="isaaclab_imitation.envs:ImitationRLEnvLegacy",
    disable_env_checker=True,
    kwargs=_LATENT_STABLE_TASK_KWARGS,
)

# LafanTrack recipe x explicit full-body command -- the default vanilla task.
gym.register(
    id="Isaac-Imitation-G1-v0",
    entry_point="isaaclab_imitation.envs:ImitationRLEnvLegacy",
    disable_env_checker=True,
    kwargs=_VANILLA_TASK_KWARGS,
)

# LafanTrack recipe x explicit full-body command -- explicit-name alias of G1-v0.
gym.register(
    id="Isaac-Imitation-G1-LafanTrack-v0",
    entry_point="isaaclab_imitation.envs:ImitationRLEnvLegacy",
    disable_env_checker=True,
    kwargs=_VANILLA_TASK_KWARGS,
)

# ---------------------------------------------------------------------------
# Strict recipe pins.
# ---------------------------------------------------------------------------

# Strict recipe x explicit command; see _VANILLA_STRICT_TASK_KWARGS.
gym.register(
    id="Isaac-Imitation-G1-Strict-v0",
    entry_point="isaaclab_imitation.envs:ImitationRLEnvLegacy",
    disable_env_checker=True,
    kwargs=_VANILLA_STRICT_TASK_KWARGS,
)

# Strict recipe x latent command -- the 2026-07-21..27 default; frozen for all
# pre-2026-07-27 `Isaac-Imitation-G1-Latent-v0` results (Study B DiffSR
# bottlenecks, Study C grouped-VQ capacity arms, low-level qualification
# artifacts). Use this explicit id for any continuation segment or re-run of
# that work; the default id no longer reproduces it.
gym.register(
    id="Isaac-Imitation-G1-Latent-Strict-v0",
    entry_point="isaaclab_imitation.envs:ImitationRLEnvLegacy",
    disable_env_checker=True,
    kwargs=_LATENT_STRICT_TASK_KWARGS,
)

# ---------------------------------------------------------------------------
# Deprecated / frozen pins (kept for reproducibility; see each kwargs note).
# ---------------------------------------------------------------------------

# DEPRECATED: pre-migration LafanTrack-recipe latent surface; see
# _LATENT_LEGACY_TASK_KWARGS.
gym.register(
    id="Isaac-Imitation-G1-Latent-Legacy-v0",
    entry_point="isaaclab_imitation.envs:ImitationRLEnvLegacy",
    disable_env_checker=True,
    kwargs=_LATENT_LEGACY_TASK_KWARGS,
)

# Opt-in only (2026-07-21, no longer aliased as Isaac-Imitation-G1-Latent-v0):
# the SONIC release surface x latent command; see _LATENT_SONIC_TASK_KWARGS.
gym.register(
    id="Isaac-Imitation-G1-Latent-Sonic-v0",
    entry_point="isaaclab_imitation.envs:ImitationRLEnvLegacy",
    disable_env_checker=True,
    kwargs=_LATENT_SONIC_TASK_KWARGS,
)

# SONIC release environment (rewards, adaptive strict terminations, threshold
# curriculum, level0_4 randomization, SONIC actuators/robot, full-trajectory
# adaptive-failure resets) x latent command with this repo's single-frame
# observations instead of SONIC's 10-step proprioceptive histories -- the
# 2026-07-21 isolated history ablation found those histories buy little at our
# scale. Optimizer contract is the local/legacy one
# (`sonic_release_optimizer=False`), so the only axis moving against
# `Isaac-Imitation-G1-Latent-v0` is the environment.
gym.register(
    id="Isaac-Imitation-G1-Latent-Sonic-NoHist-v0",
    entry_point="isaaclab_imitation.envs:ImitationRLEnvLegacy",
    disable_env_checker=True,
    kwargs={
        **_LATENT_SONIC_TASK_KWARGS,
        "env_cfg_entry_point": (
            f"{__name__}.variants.sonic:ImitationG1LatentSonicNoHistoryEnvCfg"
        ),
    },
)

# SONIC env x renewed 10-frame 64-D FSQ window command; see
# _LATENT_SONIC_OFFICIAL_FSQ_TASK_KWARGS.
gym.register(
    id="Isaac-Imitation-G1-Latent-SonicOfficialFSQ-v0",
    entry_point="isaaclab_imitation.envs:ImitationRLEnvLegacy",
    disable_env_checker=True,
    kwargs=_LATENT_SONIC_OFFICIAL_FSQ_TASK_KWARGS,
)

# History ablation (2026-07-21): the Strict recipe x latent command with
# SONIC's 10-step proprioceptive history observations and input keys, on the
# local optimizer contract. Only the observation/history contract differs from
# the then-default strict surface.
gym.register(
    id="Isaac-Imitation-G1-Latent-History-v0",
    entry_point="isaaclab_imitation.envs:ImitationRLEnvLegacy",
    disable_env_checker=True,
    kwargs={
        **_LATENT_STRICT_TASK_KWARGS,
        "env_cfg_entry_point": (
            f"{__name__}.variants.strict:ImitationG1LatentStrictHistoryEnvCfg"
        ),
        "rlopt_cfg_entry_point": (
            f"{agents.__name__}.rlopt_ipmd_cfg:G1ImitationLatentSonicRLOptIPMDConfig"
        ),
        "rlopt_ipmd_cfg_entry_point": (
            f"{agents.__name__}.rlopt_ipmd_cfg:G1ImitationLatentSonicRLOptIPMDConfig"
        ),
    },
)

# LafanTrack-lineage latent surface x held 128-D future-goal command.
gym.register(
    id="Isaac-Imitation-G1-Latent-Goal-v0",
    entry_point="isaaclab_imitation.envs:ImitationRLEnvLegacy",
    disable_env_checker=True,
    kwargs=_LATENT_GOAL_TASK_KWARGS,
)

# LafanTrack-lineage latent surface x 256-D future-window CVAE command.
gym.register(
    id="Isaac-Imitation-G1-Latent-FutureCVAE-v0",
    entry_point="isaaclab_imitation.envs:ImitationRLEnvLegacy",
    disable_env_checker=True,
    kwargs=_LATENT_FUTURE_CVAE_TASK_KWARGS,
)

# LafanTrack-lineage latent surface x 64-D per-control-step VQ token packets.
gym.register(
    id="Isaac-Imitation-G1-Latent-PerStepVQ-v0",
    entry_point="isaaclab_imitation.envs:ImitationRLEnvLegacy",
    disable_env_checker=True,
    kwargs=_LATENT_PER_STEP_VQ_TASK_KWARGS,
)

# LafanTrack-lineage latent surface x causal 9-step-window VQ-VAE command.
gym.register(
    id="Isaac-Imitation-G1-Latent-VQVAE-v0",
    entry_point="isaaclab_imitation.envs:ImitationRLEnvLegacy",
    disable_env_checker=True,
    kwargs=_LATENT_VQVAE_TASK_KWARGS,
)

# Strict recipe x 66-D (64 code + 2 phase) reconstruction-ablation command.
gym.register(
    id="Isaac-Imitation-G1-Latent-Ablation-v0",
    entry_point="isaaclab_imitation.envs:ImitationRLEnvLegacy",
    disable_env_checker=True,
    kwargs=_LATENT_ABLATION_TASK_KWARGS,
)
