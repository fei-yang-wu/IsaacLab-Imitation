#!/usr/bin/env bash

# Scheme-specific deltas over one common BONES-129k controller recipe.
# Populated by configure_arm <name> into ARM_ENV_OVERRIDES and
# ARM_AGENT_OVERRIDES. The common launcher owns data, rewards, resets, PPO,
# rollout geometry, logging, and cluster resources.

LATENT_SCHEME_ARMS=(
    reset80_diffsr
    sonic_fsq32
    sonic_fsq32_v2
    vqvae_k32
    autoencoder
    root_qpos_explicit
)

# SONIC's released tokenizer body set (gear_sonic motion.yaml body_names):
# the same 14 bodies this campaign's runtime cache carries.
SONIC_TOKENIZER_KEYPOINT_BODIES="[pelvis,left_hip_roll_link,left_knee_link,left_ankle_roll_link,right_hip_roll_link,right_knee_link,right_ankle_roll_link,torso_link,left_shoulder_roll_link,left_elbow_link,left_wrist_yaw_link,right_shoulder_roll_link,right_elbow_link,right_wrist_yaw_link]"

configure_arm() {
    local arm="$1"
    ARM_ENV_OVERRIDES=()
    ARM_AGENT_OVERRIDES=()
    ARM_DESCRIPTION=""
    ARM_WANDB_TAGS=""
    ARM_NEEDS_ENCODER=0
    ARM_ONLINE_LATENT=0

    case "${arm}" in
        reset80_diffsr)
            ARM_DESCRIPTION="frozen root_qpos DiffSR, h10 encoder, hold10; reset sampler only"
            ARM_WANDB_TAGS="diffsr,root-qpos,z256,h10,hold10,frozen-encoder"
            ARM_NEEDS_ENCODER=1
            ARM_ENV_OVERRIDES+=(
                env.command_interface.actor=latent
                env.command_interface.actor.dim=258
                env.command_interface.encoder=single
            )
            ARM_AGENT_OVERRIDES+=(
                agent.ipmd.latent_dim=258
                agent.ipmd.command_source=hl_skill
                agent.ipmd.hl_skill_horizon_steps=10
                agent.ipmd.hl_skill_command_mode=z
                agent.ipmd.latent_steps_min=10
                agent.ipmd.latent_steps_max=10
                agent.ipmd.latent_learning.code_period=10
                agent.ipmd.latent_learning.command_phase_mode=sin_cos
                agent.ipmd.latent_learning.code_latent_dim=256
                agent.ipmd.hl_skill_finetune_enabled=false
            )
            ;;
        sonic_fsq32)
            ARM_DESCRIPTION="SONIC-style online FSQ, future10 window, hold1, 64 coordinates x 32 levels"
            ARM_WANDB_TAGS="sonic-style,fsq,fsq32x64,future10,hold1,online-pg,recon001"
            ARM_ONLINE_LATENT=1
            ARM_ENV_OVERRIDES+=(
                env.command_interface.actor=latent
                env.command_interface.actor.dim=64
                env.command_interface.encoder=future10
            )
            ARM_AGENT_OVERRIDES+=(
                agent.ipmd.latent_dim=64
                agent.ipmd.command_source=posterior
                agent.ipmd.latent_steps_min=1
                agent.ipmd.latent_steps_max=1
                agent.ipmd.latent_learning.method=patch_vqvae
                agent.ipmd.latent_learning.quantizer=fsq
                agent.ipmd.latent_learning.code_latent_dim=64
                agent.ipmd.latent_learning.command_phase_mode=none
                agent.ipmd.latent_learning.patch_past_steps=0
                agent.ipmd.latent_learning.patch_future_steps=9
                agent.ipmd.latent_learning.code_period=1
                agent.ipmd.latent_learning.posterior_command_period=1
                agent.ipmd.latent_learning.fsq_normalize_codes=true
                agent.ipmd.latent_learning.encoder_hidden_dims=[2048,1024,512,512]
                agent.ipmd.latent_learning.encoder_activation=silu
                agent.ipmd.latent_learning.decoder_hidden_dims=[2048,1024,512,512]
                agent.ipmd.latent_learning.decoder_activation=silu
                agent.ipmd.latent_learning.lr=2.0e-5
                agent.ipmd.latent_learning.freeze_encoder=false
                agent.ipmd.latent_learning.train_posterior_through_policy=true
                agent.ipmd.latent_learning.recon_coeff=0.01
                agent.ipmd.latent_learning.action_recon_coeff=0.0
            )
            ;;
        sonic_fsq32_v2)
            # SONIC-release-aligned tokenizer contract (2026-08-07 audit of
            # arXiv 2511.07820 + NVlabs/GR00T-WholeBodyControl
            # sonic_bones_seed.yaml). Deltas over sonic_fsq32:
            #   1. window: 10 future frames spaced 0.1 s (frame_stride=5,
            #      num_future_frames=10, dt_future_ref_frames=0.1), not 10
            #      consecutive 0.02 s frames;
            #   2. encoder input: 14-body keypoint positions in the robot's
            #      anchor frame + 6D root-orientation difference
            #      (command_multi_future_nonflat + motion_anchor_ori_b_mf,
            #      "noz" variant), not joint qpos/qvel + anchor pose.
            # Kept from sonic_fsq32 because they already match the release:
            # hold=1 re-encode each step, PG into the encoder, recon MSE with
            # coefficient 0.01 (aux_loss_coef.g1_recon), FSQ 64 coordinates x
            # 32 levels, encoder MLP [2048,1024,512,512] SiLU, no phase.
            ARM_DESCRIPTION="SONIC-release-aligned online FSQ: 14-keypoint + root-ori window, 10 frames at 0.1 s, hold1"
            ARM_WANDB_TAGS="sonic-style,fsq,fsq32x64,future10-dt0.1,keypoint14,hold1,online-pg,recon001"
            ARM_ONLINE_LATENT=1
            ARM_ENV_OVERRIDES+=(
                env.command_interface.actor=latent
                env.command_interface.actor.dim=64
                env.command_interface.encoder=future10_stride5
                "env.command_interface.encoder.components=[keypoint_pos,root_ori]"
                "env.command_interface.reference.keypoint_body_names=${SONIC_TOKENIZER_KEYPOINT_BODIES}"
            )
            ARM_AGENT_OVERRIDES+=(
                agent.ipmd.latent_dim=64
                agent.ipmd.command_source=posterior
                agent.ipmd.latent_steps_min=1
                agent.ipmd.latent_steps_max=1
                agent.ipmd.latent_learning.method=patch_vqvae
                agent.ipmd.latent_learning.quantizer=fsq
                agent.ipmd.latent_learning.code_latent_dim=64
                agent.ipmd.latent_learning.command_phase_mode=none
                agent.ipmd.latent_learning.patch_past_steps=0
                agent.ipmd.latent_learning.patch_future_steps=9
                agent.ipmd.latent_learning.code_period=1
                agent.ipmd.latent_learning.posterior_command_period=1
                agent.ipmd.latent_learning.fsq_normalize_codes=true
                agent.ipmd.latent_learning.encoder_hidden_dims=[2048,1024,512,512]
                agent.ipmd.latent_learning.encoder_activation=silu
                agent.ipmd.latent_learning.decoder_hidden_dims=[2048,1024,512,512]
                agent.ipmd.latent_learning.decoder_activation=silu
                agent.ipmd.latent_learning.lr=2.0e-5
                agent.ipmd.latent_learning.freeze_encoder=false
                agent.ipmd.latent_learning.train_posterior_through_policy=true
                agent.ipmd.latent_learning.recon_coeff=0.01
                agent.ipmd.latent_learning.action_recon_coeff=0.0
            )
            ;;
        vqvae_k32)
            ARM_DESCRIPTION="EMA VQ-VAE reconstruction, future10 window, hold10, flat K=32 codebook"
            ARM_WANDB_TAGS="vqvae,vq-ema,k32,future10,hold10,reconstruction"
            ARM_ONLINE_LATENT=1
            ARM_ENV_OVERRIDES+=(
                env.command_interface.actor=latent
                env.command_interface.actor.dim=66
                env.command_interface.encoder=future10
            )
            ARM_AGENT_OVERRIDES+=(
                agent.ipmd.latent_dim=66
                agent.ipmd.command_source=posterior
                agent.ipmd.latent_steps_min=10
                agent.ipmd.latent_steps_max=10
                agent.ipmd.latent_learning.method=patch_vqvae
                agent.ipmd.latent_learning.quantizer=vq_ema
                agent.ipmd.latent_learning.codebook_size=32
                agent.ipmd.latent_learning.codebook_embed_dim=64
                agent.ipmd.latent_learning.code_latent_dim=64
                agent.ipmd.latent_learning.command_phase_mode=sin_cos
                agent.ipmd.latent_learning.patch_past_steps=0
                agent.ipmd.latent_learning.patch_future_steps=9
                agent.ipmd.latent_learning.code_period=10
                agent.ipmd.latent_learning.posterior_command_period=10
                agent.ipmd.latent_learning.commitment_coeff=0.25
                agent.ipmd.latent_learning.ema_decay=0.99
                agent.ipmd.latent_learning.dead_code_reset_iters=1000
                agent.ipmd.latent_learning.lr=3.0e-4
                agent.ipmd.latent_learning.freeze_encoder=false
                agent.ipmd.latent_learning.train_posterior_through_policy=false
                agent.ipmd.latent_learning.recon_coeff=1.0
                agent.ipmd.latent_learning.action_recon_coeff=0.0
            )
            ;;
        autoencoder)
            ARM_DESCRIPTION="continuous reconstruction autoencoder, future10 window, hold10"
            ARM_WANDB_TAGS="autoencoder,continuous,future10,hold10,reconstruction"
            ARM_ONLINE_LATENT=1
            ARM_ENV_OVERRIDES+=(
                env.command_interface.actor=latent
                env.command_interface.actor.dim=66
                env.command_interface.encoder=future10
            )
            ARM_AGENT_OVERRIDES+=(
                agent.ipmd.latent_dim=66
                agent.ipmd.command_source=posterior
                agent.ipmd.latent_steps_min=10
                agent.ipmd.latent_steps_max=10
                agent.ipmd.latent_learning.method=patch_vqvae
                agent.ipmd.latent_learning.quantizer=identity
                agent.ipmd.latent_learning.code_latent_dim=64
                agent.ipmd.latent_learning.command_phase_mode=sin_cos
                agent.ipmd.latent_learning.patch_past_steps=0
                agent.ipmd.latent_learning.patch_future_steps=9
                agent.ipmd.latent_learning.code_period=10
                agent.ipmd.latent_learning.posterior_command_period=10
                agent.ipmd.latent_learning.lr=3.0e-4
                agent.ipmd.latent_learning.freeze_encoder=false
                agent.ipmd.latent_learning.train_posterior_through_policy=false
                agent.ipmd.latent_learning.recon_coeff=1.0
                agent.ipmd.latent_learning.action_recon_coeff=0.0
            )
            ;;
        root_qpos_explicit)
            ARM_DESCRIPTION="direct 38-D root_qpos command, single frame, renewed every control step"
            ARM_WANDB_TAGS="explicit,root-qpos,window1,hold1,command38"
            ARM_ENV_OVERRIDES+=(
                env.command_interface.actor=explicit
                env.command_interface.actor.components=[joint_qpos,root_pos,root_ori]
            )
            ARM_AGENT_OVERRIDES+=(
                agent.ipmd.use_latent_command=false
                agent.command_space=root_qpos
                agent.command_components=[joint_qpos,root_pos,root_ori]
                agent.ipmd.command_source=random
                agent.ipmd.hl_skill_checkpoint_path=null
            )
            ;;
        *)
            echo "[FATAL] Unknown arm: ${arm}" >&2
            return 2
            ;;
    esac
}
