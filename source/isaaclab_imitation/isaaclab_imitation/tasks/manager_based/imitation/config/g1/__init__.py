import gymnasium as gym

from . import (
    agents,
    imitation_g1_env_cfg,
    imitation_g1_latent_ablation_env_cfg,
    imitation_g1_latent_env_cfg,
    imitation_g1_latent_vqvae_env_cfg,
)

__all__ = [
    "imitation_g1_env_cfg",
    "imitation_g1_latent_ablation_env_cfg",
    "imitation_g1_latent_env_cfg",
    "imitation_g1_latent_vqvae_env_cfg",
    "agents",
]

_VANILLA_TASK_KWARGS = {
    "env_cfg_entry_point": f"{__name__}.imitation_g1_env_cfg:ImitationG1LafanTrackEnvCfg",
    "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ImitationPPORunnerCfg",
    "rlopt_cfg_entry_point": f"{agents.__name__}.rlopt_ppo_cfg:G1ImitationRLOptPPOConfig",
    "rlopt_ppo_cfg_entry_point": f"{agents.__name__}.rlopt_ppo_cfg:G1ImitationRLOptPPOConfig",
    "rlopt_sac_cfg_entry_point": f"{agents.__name__}.rlopt_sac_cfg:G1ImitationRLOptSACConfig",
    "rlopt_fastsac_cfg_entry_point": f"{agents.__name__}.rlopt_fastsac_cfg:G1ImitationRLOptFastSACConfig",
    "rlopt_ipmd_cfg_entry_point": f"{agents.__name__}.rlopt_ipmd_cfg:G1ImitationRLOptIPMDConfig",
    "rlopt_ipmd_sr_cfg_entry_point": f"{agents.__name__}.rlopt_ipmd_sr_cfg:G1ImitationRLOptIPMDSRConfig",
    "rlopt_ipmd_bilinear_cfg_entry_point": f"{agents.__name__}.rlopt_ipmd_bilinear_cfg:G1ImitationRLOptIPMDBilinearConfig",
    "rlopt_gail_cfg_entry_point": f"{agents.__name__}.rlopt_gail_cfg:G1ImitationRLOptGAILConfig",
    "rlopt_amp_cfg_entry_point": f"{agents.__name__}.rlopt_amp_cfg:G1ImitationRLOptAMPConfig",
}

# Strict-protocol explicit-command surface (2026-07-21): the vanilla
# observation/agent contract on the same protocol deltas as the strict latent
# default (pelvis anchor, strict SONIC terminations, [0, 200] reset starts).
# Built for the interface ablation so full-body-chunk / EE-chunk / single-frame
# trackers (selected via `agent.command_space`) train on the same env protocol
# as the latent tracker and differ only in the command space.
_VANILLA_STRICT_TASK_KWARGS = {
    **_VANILLA_TASK_KWARGS,
    "env_cfg_entry_point": (
        f"{__name__}.imitation_g1_env_cfg:ImitationG1StrictTrackEnvCfg"
    ),
}

# DEPRECATED (2026-07-19): the pre-migration beyondmimic-style latent surface
# (torso anchor, loose terminations, no proprio history, [0, 200] reset
# starts). Kept only for pre-migration checkpoints and frozen paper-protocol
# reproductions under `Isaac-Imitation-G1-Latent-Legacy-v0`. The pelvis-
# anchored strict-terminations surface built on this base
# (`_LATENT_STRICT_TASK_KWARGS` below) is the current default instead.
_LATENT_LEGACY_TASK_KWARGS = {
    "env_cfg_entry_point": f"{__name__}.imitation_g1_latent_env_cfg:ImitationG1LatentEnvCfg",
    "rlopt_cfg_entry_point": f"{agents.__name__}.rlopt_ase_cfg:G1ImitationRLOptASEConfig",
    "rlopt_ipmd_cfg_entry_point": f"{agents.__name__}.rlopt_ipmd_cfg:G1ImitationLatentRLOptIPMDConfig",
    "rlopt_ipmd_sr_cfg_entry_point": f"{agents.__name__}.rlopt_ipmd_sr_cfg:G1ImitationLatentRLOptIPMDSRConfig",
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
# (the strict surface, now Latent-v0, + the legacy/local optimizer contract, same 8192x12x12288
# scale) was found to reach episode/length=244 / episode/return=13.1 --
# far above anything the SONIC release-optimizer contract produced at matched
# scale in the concurrent VRAM ablation. Reachable only via the explicit
# `Isaac-Imitation-G1-Latent-Sonic-v0` id now; see
# wiki/isaaclab3-cu130-runtime-migration.md, "Training-gate resolution
# (2026-07-19)" and the 2026-07-21 reversal.
_LATENT_SONIC_TASK_KWARGS = {
    "env_cfg_entry_point": (
        f"{__name__}.imitation_g1_latent_env_cfg:ImitationG1LatentSonicEnvCfg"
    ),
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

# Default latent task surface (2026-07-21): pelvis-anchored legacy scaffolding
# with annealed strict terminations, using the legacy/local optimizer contract
# (`G1ImitationLatentRLOptIPMDConfig`: 512/256/128 ELU MLPs, actor lr 1e-3).
# This is the config behind W&B run bn931wny (episode/length=244,
# episode/return=13.1 at 8192 envs x 12 steps x minibatch 12288) -- the best
# validated result to date, ahead of the SONIC release-optimizer contract at
# matched scale. See `_LATENT_SONIC_TASK_KWARGS` above for the deprecation
# history.
_LATENT_STRICT_TASK_KWARGS = {
    **_LATENT_LEGACY_TASK_KWARGS,
    "env_cfg_entry_point": (
        f"{__name__}.imitation_g1_latent_env_cfg:ImitationG1LatentStrictEnvCfg"
    ),
}

# Default latent task surface (2026-07-27): the full SONIC release recipe with
# this repo's legacy reset distribution ([0, 200] starts,
# `failure_rate_max_over_mean=50`) and strict-from-scratch terminations. The
# 2026-07-27 reset-sampling screen showed SONIC's full-trajectory
# adaptive-failure sampler, not its rewards or actuators, was what cost ~5.6x
# episode length at 4096 environments; see `ImitationG1LatentStableEnvCfg`.
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
    "env_cfg_entry_point": (
        f"{__name__}.imitation_g1_latent_env_cfg:ImitationG1LatentStableEnvCfg"
    ),
}

_LATENT_GOAL_TASK_KWARGS = {
    "env_cfg_entry_point": (
        f"{__name__}.imitation_g1_latent_env_cfg:ImitationG1LatentGoalEnvCfg"
    ),
    "rlopt_ipmd_bilinear_cfg_entry_point": (
        f"{agents.__name__}.rlopt_ipmd_bilinear_cfg:"
        "G1ImitationLatentGoalRLOptIPMDBilinearConfig"
    ),
}

_LATENT_FUTURE_CVAE_TASK_KWARGS = {
    "env_cfg_entry_point": (
        f"{__name__}.imitation_g1_latent_env_cfg:ImitationG1LatentFutureCVAEEnvCfg"
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
        f"{__name__}.imitation_g1_latent_env_cfg:ImitationG1LatentPerStepVQEnvCfg"
    ),
    "rlopt_cfg_entry_point": (
        f"{agents.__name__}.rlopt_ipmd_cfg:G1ImitationLatentPerStepVQRLOptIPMDConfig"
    ),
    "rlopt_ipmd_cfg_entry_point": (
        f"{agents.__name__}.rlopt_ipmd_cfg:G1ImitationLatentPerStepVQRLOptIPMDConfig"
    ),
}

# SONIC-motivated per-step FSQ token interface. Built on the *strict* bundle,
# not the legacy one `_LATENT_PER_STEP_VQ_TASK_KWARGS` inherits through
# `ImitationG1LatentFutureCVAEEnvCfg`: the qualified LAFAN1 oracles and the
# Study B/C arms all ran on the strict surface, so only this lineage is
# comparable with them.
_LATENT_SONIC_FSQ_TASK_KWARGS = {
    **_LATENT_STRICT_TASK_KWARGS,
    "env_cfg_entry_point": (
        f"{__name__}.imitation_g1_latent_env_cfg:ImitationG1LatentSonicFSQEnvCfg"
    ),
    "rlopt_cfg_entry_point": (
        f"{agents.__name__}.rlopt_ipmd_cfg:G1ImitationLatentSonicFSQRLOptIPMDConfig"
    ),
    "rlopt_ipmd_cfg_entry_point": (
        f"{agents.__name__}.rlopt_ipmd_cfg:G1ImitationLatentSonicFSQRLOptIPMDConfig"
    ),
}

_LATENT_VQVAE_TASK_KWARGS = {
    "env_cfg_entry_point": (
        f"{__name__}.imitation_g1_latent_vqvae_env_cfg:ImitationG1LatentVQVAEEnvCfg"
    ),
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
        f"{__name__}.imitation_g1_latent_ablation_env_cfg:"
        "ImitationG1LatentAblationEnvCfg"
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

gym.register(
    id="Isaac-Imitation-G1-v0",
    entry_point="isaaclab_imitation.envs:ImitationRLEnv",
    disable_env_checker=True,
    kwargs=_VANILLA_TASK_KWARGS,
)

gym.register(
    id="Isaac-Imitation-G1-LafanTrack-v0",
    entry_point="isaaclab_imitation.envs:ImitationRLEnv",
    disable_env_checker=True,
    kwargs=_VANILLA_TASK_KWARGS,
)

# Strict-protocol explicit-command surface; see _VANILLA_STRICT_TASK_KWARGS.
gym.register(
    id="Isaac-Imitation-G1-Strict-v0",
    entry_point="isaaclab_imitation.envs:ImitationRLEnv",
    disable_env_checker=True,
    kwargs=_VANILLA_STRICT_TASK_KWARGS,
)

# Default latent task (2026-07-27): SONIC release recipe with this repo's
# legacy reset distribution; see _LATENT_STABLE_TASK_KWARGS above.
gym.register(
    id="Isaac-Imitation-G1-Latent-v0",
    entry_point="isaaclab_imitation.envs:ImitationRLEnv",
    disable_env_checker=True,
    kwargs=_LATENT_STABLE_TASK_KWARGS,
)

# The previous default (2026-07-21 to 2026-07-27). Everything recorded against
# `Isaac-Imitation-G1-Latent-v0` before 2026-07-27 ran on this surface: Study B
# DiffSR bottlenecks, the Study C grouped-VQ capacity arms, and the low-level
# qualification artifacts. Use this explicit id for any continuation segment or
# re-run of that work; the default id no longer reproduces it.
gym.register(
    id="Isaac-Imitation-G1-Latent-Strict-v0",
    entry_point="isaaclab_imitation.envs:ImitationRLEnv",
    disable_env_checker=True,
    kwargs=_LATENT_STRICT_TASK_KWARGS,
)

# History ablation (2026-07-21): the strict default surface with SONIC's
# 10-step proprioceptive history observations and input keys, on the local
# optimizer contract. Only the observation/history contract differs from
# Isaac-Imitation-G1-Latent-v0.
gym.register(
    id="Isaac-Imitation-G1-Latent-History-v0",
    entry_point="isaaclab_imitation.envs:ImitationRLEnv",
    disable_env_checker=True,
    kwargs={
        **_LATENT_STRICT_TASK_KWARGS,
        "env_cfg_entry_point": (
            f"{__name__}.imitation_g1_latent_env_cfg:"
            "ImitationG1LatentStrictHistoryEnvCfg"
        ),
        "rlopt_cfg_entry_point": (
            f"{agents.__name__}.rlopt_ipmd_cfg:G1ImitationLatentSonicRLOptIPMDConfig"
        ),
        "rlopt_ipmd_cfg_entry_point": (
            f"{agents.__name__}.rlopt_ipmd_cfg:G1ImitationLatentSonicRLOptIPMDConfig"
        ),
    },
)

# DEPRECATED: pre-migration latent surface; see _LATENT_LEGACY_TASK_KWARGS.
gym.register(
    id="Isaac-Imitation-G1-Latent-Legacy-v0",
    entry_point="isaaclab_imitation.envs:ImitationRLEnv",
    disable_env_checker=True,
    kwargs=_LATENT_LEGACY_TASK_KWARGS,
)

# Opt-in only (2026-07-21, no longer aliased as Isaac-Imitation-G1-Latent-v0):
# the SONIC release surface; see _LATENT_SONIC_TASK_KWARGS above.
gym.register(
    id="Isaac-Imitation-G1-Latent-Sonic-v0",
    entry_point="isaaclab_imitation.envs:ImitationRLEnv",
    disable_env_checker=True,
    kwargs=_LATENT_SONIC_TASK_KWARGS,
)

# SONIC release environment (rewards, adaptive strict terminations, threshold
# curriculum, level0_4 randomization, SONIC actuators/robot, full-trajectory
# adaptive-failure resets) with this repo's single-frame observations instead
# of SONIC's 10-step proprioceptive histories -- the 2026-07-21 isolated
# history ablation found those histories buy little at our scale. Optimizer
# contract is the local/legacy one (`sonic_release_optimizer=False`), so the
# only axis moving against `Isaac-Imitation-G1-Latent-v0` is the environment.
gym.register(
    id="Isaac-Imitation-G1-Latent-Sonic-NoHist-v0",
    entry_point="isaaclab_imitation.envs:ImitationRLEnv",
    disable_env_checker=True,
    kwargs={
        **_LATENT_SONIC_TASK_KWARGS,
        "env_cfg_entry_point": (
            f"{__name__}.imitation_g1_latent_env_cfg:"
            "ImitationG1LatentSonicNoHistoryEnvCfg"
        ),
    },
)

gym.register(
    id="Isaac-Imitation-G1-Latent-Goal-v0",
    entry_point="isaaclab_imitation.envs:ImitationRLEnv",
    disable_env_checker=True,
    kwargs=_LATENT_GOAL_TASK_KWARGS,
)

gym.register(
    id="Isaac-Imitation-G1-Latent-FutureCVAE-v0",
    entry_point="isaaclab_imitation.envs:ImitationRLEnv",
    disable_env_checker=True,
    kwargs=_LATENT_FUTURE_CVAE_TASK_KWARGS,
)

gym.register(
    id="Isaac-Imitation-G1-Latent-PerStepVQ-v0",
    entry_point="isaaclab_imitation.envs:ImitationRLEnv",
    disable_env_checker=True,
    kwargs=_LATENT_PER_STEP_VQ_TASK_KWARGS,
)

# SONIC method replication (2026-07-28): per-step FSQ tokens, co-trained
# encoder, strict lineage. See the campaign under
# experiments/campaigns/2026-07-28-sonic-method-replication/.
gym.register(
    id="Isaac-Imitation-G1-Latent-SonicFSQ-v0",
    entry_point="isaaclab_imitation.envs:ImitationRLEnv",
    disable_env_checker=True,
    kwargs=_LATENT_SONIC_FSQ_TASK_KWARGS,
)

gym.register(
    id="Isaac-Imitation-G1-Latent-VQVAE-v0",
    entry_point="isaaclab_imitation.envs:ImitationRLEnv",
    disable_env_checker=True,
    kwargs=_LATENT_VQVAE_TASK_KWARGS,
)

gym.register(
    id="Isaac-Imitation-G1-Latent-Ablation-v0",
    entry_point="isaaclab_imitation.envs:ImitationRLEnv",
    disable_env_checker=True,
    kwargs=_LATENT_ABLATION_TASK_KWARGS,
)
