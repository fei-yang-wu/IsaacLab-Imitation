"""G1 imitation task registrations.

A task id is the stable, citable identity of a protocol: it names a config
class so a recorded command, checkpoint, or paper gate keeps resolving. It is
not a kind of environment. Everything from ``-G1-v2`` onward is ONE environment
(``imitation_g1_env_v2.ImitationG1V2EnvCfg``) at different points in its
configuration space, and the classes behind those ids (``surfaces.py``) select
presets and set scalars, nothing more. A new ablation is a CLI override::

    --task Isaac-Imitation-G1-v2 env.command_interface.actor=explicit

not a new class and not a new id. Register an id when a protocol needs to be
cited later; until then, override.

Module layout: ``imitation_g1_env_v2.py`` (the current environment, inheriting
only ``ImitationLearningEnvCfg``) plus ``surfaces.py`` (its registered
configuration points). Frozen behind them: ``imitation_g1_env_v0.py``
(LafanTrack) and ``imitation_g1_env_v1.py``, both on the DEPRECATED
``common/tracking_env.py`` base, with ``variants/`` holding the Strict pins and
the historical ``imitation_g1*_env_cfg`` modules kept as re-export shims. The
legacy configs keep their own flat dataset fields; v2 configures motion data
through ``env.data.*`` (see ``motion_data.MotionDataCfg``).

Registrations below are grouped: current defaults first, then the v2 surfaces,
then the Strict recipe pins, then deprecated/frozen pins kept for
reproducibility.
"""

import gymnasium as gym

from . import (
    agents,
    common,
    imitation_g1_env_v0,
    imitation_g1_env_v1,
    imitation_g1_env_v2,
    surfaces,
    variants,
)

__all__ = [
    "agents",
    "common",
    "imitation_g1_env_v0",
    "imitation_g1_env_v1",
    "imitation_g1_env_v2",
    "surfaces",
    "variants",
]

# ---------------------------------------------------------------------------
# Command-config pins (task kwargs): env entry point + agent entry points.
# ---------------------------------------------------------------------------

# LafanTrack recipe x explicit full-body command (vanilla observation surface).
# Dead agent references (ASE/AMP/GAIL/FastSAC/bilinear) were pruned on
# 2026-08-01 with the configs themselves; the env + live agent contracts are
# untouched.
_VANILLA_TASK_KWARGS = {
    "env_cfg_entry_point": f"{__name__}.imitation_g1_env_v0:ImitationG1LafanTrackEnvCfg",
    "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ImitationPPORunnerCfg",
    "rlopt_cfg_entry_point": f"{agents.__name__}.rlopt_ppo_cfg:G1ImitationRLOptPPOConfig",
    "rlopt_ppo_cfg_entry_point": f"{agents.__name__}.rlopt_ppo_cfg:G1ImitationRLOptPPOConfig",
    "rlopt_sac_cfg_entry_point": f"{agents.__name__}.rlopt_sac_cfg:G1ImitationRLOptSACConfig",
    "rlopt_ipmd_cfg_entry_point": f"{agents.__name__}.rlopt_ipmd_cfg:G1ImitationRLOptIPMDConfig",
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

# Strict recipe x latent command. Default latent task surface from 2026-07-21
# to 2026-07-27: pelvis-anchored legacy scaffolding with strict-from-scratch
# terminations, using the legacy/local optimizer contract
# (`G1ImitationLatentRLOptIPMDConfig`: 512/256/128 ELU MLPs, actor lr 1e-3).
# This is the config behind W&B run bn931wny (episode/length=244,
# episode/return=13.1 at 8192 envs x 12 steps x minibatch 12288) -- the best
# validated result of its period, ahead of the SONIC release-optimizer
# contract at matched scale. The old ``-G1-Latent-v0`` moving alias pointed
# here; that id now aliases the Stable v1 surface.
#
# FROZEN (2026-08-01): kept for pre-2026-07-27 ``Isaac-Imitation-G1-Latent-v0``
# results (Study B DiffSR bottlenecks, Study C grouped-VQ capacity arms,
# low-level qualification artifacts). Dead agent references (ASE/bilinear)
# were pruned with the configs.
_LATENT_STRICT_TASK_KWARGS = {
    "env_cfg_entry_point": (
        f"{__name__}.variants.strict:ImitationG1LatentStrictEnvCfg"
    ),
    "rlopt_cfg_entry_point": (
        f"{agents.__name__}.rlopt_ipmd_cfg:G1ImitationLatentRLOptIPMDConfig"
    ),
    "rlopt_ipmd_cfg_entry_point": (
        f"{agents.__name__}.rlopt_ipmd_cfg:G1ImitationLatentRLOptIPMDConfig"
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
# FROZEN (2026-08-01) under `-G1-v1` / `-G1-Latent-v0`; every result before
# 2026-07-27 that names ``Isaac-Imitation-G1-Latent-v0`` (Study B, the Study C
# grouped-VQ arms and their continuation segments, and the low-level
# qualification artifacts) must use the explicit Strict id
# ``-G1-Latent-Strict-v0`` to stay reproducible. Dead agent references
# (bilinear) were pruned with the config.
_LATENT_STABLE_TASK_KWARGS = {
    "env_cfg_entry_point": (f"{__name__}.imitation_g1_env_v1:ImitationG1EnvCfg"),
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
}

# Migrated v2 surfaces (2026-08-01): the vqvae / cvae / per-step-VQ / sonic
# families, re-registered on the flat v2 full surface base and the v2 env.
_LATENT_VQVAE_V2_TASK_KWARGS = {
    "env_cfg_entry_point": (f"{__name__}.surfaces:ImitationG1VQVAESurfaceEnvCfg"),
    "rlopt_cfg_entry_point": (
        f"{agents.__name__}.rlopt_ipmd_vqvae_cfg:G1ImitationLatentRLOptIPMDVQVAEConfig"
    ),
    "rlopt_ipmd_cfg_entry_point": (
        f"{agents.__name__}.rlopt_ipmd_vqvae_cfg:G1ImitationLatentRLOptIPMDVQVAEConfig"
    ),
    "rlopt_ipmd_vqvae_cfg_entry_point": (
        f"{agents.__name__}.rlopt_ipmd_vqvae_cfg:G1ImitationLatentRLOptIPMDVQVAEConfig"
    ),
}

_LATENT_FUTURE_CVAE_V2_TASK_KWARGS = {
    "env_cfg_entry_point": (f"{__name__}.surfaces:ImitationG1FutureCVAESurfaceEnvCfg"),
    "rlopt_cfg_entry_point": (
        f"{agents.__name__}.rlopt_ipmd_cfg:G1ImitationLatentFutureCVAERLOptIPMDConfig"
    ),
    "rlopt_ipmd_cfg_entry_point": (
        f"{agents.__name__}.rlopt_ipmd_cfg:G1ImitationLatentFutureCVAERLOptIPMDConfig"
    ),
}

_LATENT_PER_STEP_VQ_V2_TASK_KWARGS = {
    "env_cfg_entry_point": (f"{__name__}.surfaces:ImitationG1PerStepVQSurfaceEnvCfg"),
    "rlopt_cfg_entry_point": (
        f"{agents.__name__}.rlopt_ipmd_cfg:G1ImitationLatentPerStepVQRLOptIPMDConfig"
    ),
    "rlopt_ipmd_cfg_entry_point": (
        f"{agents.__name__}.rlopt_ipmd_cfg:G1ImitationLatentPerStepVQRLOptIPMDConfig"
    ),
}

_SONIC_V2_TASK_KWARGS = {
    "env_cfg_entry_point": (f"{__name__}.surfaces:ImitationG1SonicSurfaceEnvCfg"),
    "rlopt_cfg_entry_point": (
        f"{agents.__name__}.rlopt_ipmd_cfg:G1ImitationLatentSonicRLOptIPMDConfig"
    ),
    "rlopt_ipmd_cfg_entry_point": (
        f"{agents.__name__}.rlopt_ipmd_cfg:G1ImitationLatentSonicRLOptIPMDConfig"
    ),
    "rlopt_ipmd_sonic_release_cfg_entry_point": (
        f"{agents.__name__}.rlopt_ipmd_cfg:G1ImitationLatentSonicReleaseRLOptIPMDConfig"
    ),
}

_SONIC_NO_HISTORY_V2_TASK_KWARGS = {
    **_SONIC_V2_TASK_KWARGS,
    "env_cfg_entry_point": (
        f"{__name__}.surfaces:ImitationG1SonicNoHistorySurfaceEnvCfg"
    ),
}

# Official-window SONIC adaptation: the complete 10-frame future window is
# compressed into one 64-D FSQ command and recomputed every control step. This
# combines the SONIC environment/history and release optimizer contract.
_SONIC_OFFICIAL_FSQ_V2_TASK_KWARGS = {
    **_SONIC_V2_TASK_KWARGS,
    "env_cfg_entry_point": (
        f"{__name__}.surfaces:ImitationG1SonicOfficialFSQSurfaceEnvCfg"
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

# ---------------------------------------------------------------------------
# Current defaults.
#
# WHAT `-G1-v2` MEANS AS OF 2026-08-04 -- both changed IN PLACE, by explicit
# decision, rather than by registering a `-v3`:
#   * rewards: `G1V2TunedRewardsCfg` (motion_body_pos std 0.05 w2.0,
#     motion_global_anchor_pos std 0.1 w2.0, motion_global_anchor_ori std 0.15
#     w2.0). Measured -37.3% MPJPE-G / -34.7% EE-G over three seeds against two
#     control seeds, ranges disjoint.
#   * macro state: the `root_qpos` frame (qpos + root pose, 38/frame -> a
#     380-wide encoder input), replacing full-body (67/frame -> 670).
#
# THE COST OF DOING IT IN PLACE: a v2 checkpoint trained before this date no
# longer reproduces from the id alone. It needs
# `env.expert_macro_state_terms=[expert_motion,expert_anchor_pos_b,expert_anchor_ori_b]`
# and its original reward overrides. Pairing an old full-body encoder with the
# new default fails loudly at the first forward (`hl/state shape mismatch`),
# never silently. See the `g1-encoder-interface` skill.
#
# `ImitationG1SonicSurfaceEnvCfg` is pinned to the pre-2026-08-04 rewards and
# macro state because it is the published SONIC recipe; the other v2 subclasses
# (Explicit-v2, Chunk-v2, VQVAE/CVAE/PerStepVQ-v0) deliberately track the
# default, so the comparison rows stay matched.
#
# Versioning convention (2026-07-31 onward): "the default" is always the
# highest-numbered `-G1-vN` latent id below. Bumping the config is bumping
# the number, never mutating an existing vN's kwargs -- once a vN is
# superseded by vN+1, it is frozen forever (still registered, still exactly
# reproducible) and simply stops being cited as the default. Do not reuse
# `-Latent-v0`'s "moving alias" pattern again; it predates this convention
# and is kept only for its own back-compat reasons (see its comment below).
# ---------------------------------------------------------------------------
_CURRENT_V2_IPMD_TUNED_ENTRY_POINT = (
    f"{agents.__name__}.rlopt_ipmd_cfg:G1ImitationTunedRLOptIPMDConfig"
)

# The 2026-08-30 optimizer geometry (full batch, 3 epochs) as a SEPARATE entry
# point rather than a redirect of the tuned one: the tuned contract is frozen
# so the 46.5B/50B chains stay reproducible. New campaigns should pass
# `--agent rlopt_ipmd_tuned_fullbatch_cfg_entry_point`; see the class docstring
# for what is measured and what is not.
_CURRENT_V2_IPMD_TUNED_FULLBATCH_ENTRY_POINT = (
    f"{agents.__name__}.rlopt_ipmd_cfg:G1ImitationTunedFullBatchRLOptIPMDConfig"
)

# Full batch + LINEAR actor lr decay in place of the KL-adaptive rule
# (2026-09-01). Additive, like the full-batch entry point; see the class
# docstring for the 64-D no-phase stall that motivated it.
_CURRENT_V2_IPMD_TUNED_FULLBATCH_LINEARLR_ENTRY_POINT = (
    f"{agents.__name__}.rlopt_ipmd_cfg:G1ImitationTunedFullBatchLinearLRRLOptIPMDConfig"
)

_CURRENT_V2_IPMD_POSTERIOR_ROOT_QPOS_ENTRY_POINT = (
    f"{agents.__name__}.rlopt_ipmd_cfg:G1ImitationPosteriorRootQposRLOptIPMDConfig"
)

_CURRENT_V2_IPMD_L2T_ENTRY_POINT = (
    f"{agents.__name__}.rlopt_ipmd_l2t_cfg:G1ImitationRLOptIPMDL2TConfig"
)

# Stable recipe, command space configured -- THE DEFAULT latent task
# (2026-08-01 onward; supersedes `-G1-v1`, which stays frozen at its exact
# old kwargs below). No `Latent` tag: under the recipe x command-config
# architecture the command space is configuration, not identity. Defaults to
# the latent 258-D command (the class default; unlike explicit `-G1-v0`);
# the explicit and planner-packet rows are the `-Explicit-v2` / `-Chunk-v2`
# surfaces below. Cite this id in protocols, gates, and paper commands so
# recorded runs and checkpoints stay reproducible even after a newer vN
# becomes the default.
#
# The v2 surface: one declared command interface (see
# `tasks/manager_based/imitation/command_interface.py`) -- the always-present
# dataset-backed `reference` channel (selection, reset-start sampling,
# `Metrics/reference/...`, the privileged critic view) plus one `actor`
# channel. The env entry point is the composed flagship `ImitationRLEnv`
# (ExpertDataPlane + the two command terms).
gym.register(
    id="Isaac-Imitation-G1-v2",
    entry_point="isaaclab_imitation.envs:ImitationRLEnv",
    disable_env_checker=True,
    kwargs={
        **_LATENT_STABLE_TASK_KWARGS,
        "env_cfg_entry_point": (f"{__name__}.imitation_g1_env_v2:ImitationG1V2EnvCfg"),
        # PPO / SAC train on the explicit surfaces (vanilla input keys):
        # pair them with `env.command_interface.actor=explicit` plus
        # `agent.ipmd.use_latent_command=false`, or use the `-Explicit-v2`
        # pin below. The default stays IPMD on the latent command.
        "rlopt_ppo_cfg_entry_point": (
            f"{agents.__name__}.rlopt_ppo_cfg:G1ImitationRLOptPPOConfig"
        ),
        "rlopt_sac_cfg_entry_point": (
            f"{agents.__name__}.rlopt_sac_cfg:G1ImitationRLOptSACConfig"
        ),
        "rlopt_ipmd_l2t_cfg_entry_point": (_CURRENT_V2_IPMD_L2T_ENTRY_POINT),
        "rlopt_ipmd_tuned_cfg_entry_point": (_CURRENT_V2_IPMD_TUNED_ENTRY_POINT),
        "rlopt_ipmd_tuned_fullbatch_cfg_entry_point": (
            _CURRENT_V2_IPMD_TUNED_FULLBATCH_ENTRY_POINT
        ),
        "rlopt_ipmd_tuned_fullbatch_linearlr_cfg_entry_point": (
            _CURRENT_V2_IPMD_TUNED_FULLBATCH_LINEARLR_ENTRY_POINT
        ),
        "rlopt_ipmd_posterior_root_qpos_cfg_entry_point": (
            _CURRENT_V2_IPMD_POSTERIOR_ROOT_QPOS_ENTRY_POINT
        ),
    },
)

# The explicit / oracle row of the interface comparison: the same v2
# environment with `command_interface.actor` set to the explicit channel, so
# the actor reads the reference command directly. Select a command space by
# its components, e.g.
# `env.command_interface.actor.components='[joint_qpos,root_pos,root_ori]'`.
gym.register(
    id="Isaac-Imitation-G1-Explicit-v2",
    entry_point="isaaclab_imitation.envs:ImitationRLEnv",
    disable_env_checker=True,
    kwargs={
        **_LATENT_STABLE_TASK_KWARGS,
        "env_cfg_entry_point": (
            f"{__name__}.surfaces:ImitationG1ExplicitSurfaceEnvCfg"
        ),
        "rlopt_ppo_cfg_entry_point": (
            f"{agents.__name__}.rlopt_ppo_cfg:G1ImitationRLOptPPOConfig"
        ),
        "rlopt_sac_cfg_entry_point": (
            f"{agents.__name__}.rlopt_sac_cfg:G1ImitationRLOptSACConfig"
        ),
        "rlopt_ipmd_l2t_cfg_entry_point": (_CURRENT_V2_IPMD_L2T_ENTRY_POINT),
        "rlopt_ipmd_tuned_cfg_entry_point": (_CURRENT_V2_IPMD_TUNED_ENTRY_POINT),
        "rlopt_ipmd_tuned_fullbatch_cfg_entry_point": (
            _CURRENT_V2_IPMD_TUNED_FULLBATCH_ENTRY_POINT
        ),
        "rlopt_ipmd_tuned_fullbatch_linearlr_cfg_entry_point": (
            _CURRENT_V2_IPMD_TUNED_FULLBATCH_LINEARLR_ENTRY_POINT
        ),
    },
)

# The planner-packet row: one ten-frame packet per 5 Hz hold window, consumed
# one phase-aligned slot per 50 Hz control step. Defaults to the oracle
# self-fill (`source=reference`) used to train and certify the streamed
# interface; a planner run sets `env.command_interface.actor.source=external`.
gym.register(
    id="Isaac-Imitation-G1-Chunk-v2",
    entry_point="isaaclab_imitation.envs:ImitationRLEnv",
    disable_env_checker=True,
    kwargs={
        **_LATENT_STABLE_TASK_KWARGS,
        "env_cfg_entry_point": (f"{__name__}.surfaces:ImitationG1ChunkSurfaceEnvCfg"),
        "rlopt_ipmd_l2t_cfg_entry_point": (_CURRENT_V2_IPMD_L2T_ENTRY_POINT),
        "rlopt_ipmd_tuned_cfg_entry_point": (_CURRENT_V2_IPMD_TUNED_ENTRY_POINT),
        "rlopt_ipmd_tuned_fullbatch_cfg_entry_point": (
            _CURRENT_V2_IPMD_TUNED_FULLBATCH_ENTRY_POINT
        ),
        "rlopt_ipmd_tuned_fullbatch_linearlr_cfg_entry_point": (
            _CURRENT_V2_IPMD_TUNED_FULLBATCH_LINEARLR_ENTRY_POINT
        ),
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
# v2 surface pins: latent-command widths and encoder windows on the single v2
# environment (`surfaces.py`). Each is equivalent to the corresponding
# `env.command_interface.*` overrides on `-G1-v2`; the id exists so protocols
# and checkpoints can cite one. The legacy `-G1-Latent-*` ids for these
# families were deleted with their legacy variant configs on 2026-08-01.
# ---------------------------------------------------------------------------

# Flat v2 full surface x causal 9-step-window VQ-VAE command
# (in-loop `patch_vqvae` encoder).
gym.register(
    id="Isaac-Imitation-G1-VQVAE-v0",
    entry_point="isaaclab_imitation.envs:ImitationRLEnv",
    disable_env_checker=True,
    kwargs=_LATENT_VQVAE_V2_TASK_KWARGS,
)

# Flat v2 full surface x 256-D future-window CVAE command.
gym.register(
    id="Isaac-Imitation-G1-CVAE-v0",
    entry_point="isaaclab_imitation.envs:ImitationRLEnv",
    disable_env_checker=True,
    kwargs=_LATENT_FUTURE_CVAE_V2_TASK_KWARGS,
)

# Flat v2 full surface x 64-D per-control-step VQ token packets.
gym.register(
    id="Isaac-Imitation-G1-PerStepVQ-v0",
    entry_point="isaaclab_imitation.envs:ImitationRLEnv",
    disable_env_checker=True,
    kwargs=_LATENT_PER_STEP_VQ_V2_TASK_KWARGS,
)

# Flat v2 full surface x latent command with SONIC's 10-step histories,
# full-trajectory adaptive-failure resets, and the termination curriculum.
gym.register(
    id="Isaac-Imitation-G1-Sonic-v0",
    entry_point="isaaclab_imitation.envs:ImitationRLEnv",
    disable_env_checker=True,
    kwargs=_SONIC_V2_TASK_KWARGS,
)

# SONIC release environment with this repo's single-frame observations
# (the 2026-07-21 isolated history ablation found the h10 histories buy
# little at our scale).
gym.register(
    id="Isaac-Imitation-G1-Sonic-NoHist-v0",
    entry_point="isaaclab_imitation.envs:ImitationRLEnv",
    disable_env_checker=True,
    kwargs=_SONIC_NO_HISTORY_V2_TASK_KWARGS,
)

# SONIC env x renewed 10-frame 64-D FSQ window command.
gym.register(
    id="Isaac-Imitation-G1-SonicOfficialFSQ-v0",
    entry_point="isaaclab_imitation.envs:ImitationRLEnv",
    disable_env_checker=True,
    kwargs=_SONIC_OFFICIAL_FSQ_V2_TASK_KWARGS,
)

# ---------------------------------------------------------------------------
# Frozen legacy pins (kept for reproducibility; see each kwargs note).
# ---------------------------------------------------------------------------

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
