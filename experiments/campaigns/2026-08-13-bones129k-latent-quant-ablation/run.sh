#!/usr/bin/env bash
# 2026-08-13 — latent-bottleneck ablation under the robot_heading anchor frame.
# Full BONES-SEED (129,785 motions), local workstation, 100M frames per arm.
#
# One variable across arms: the latent-learning method. Everything else —
# anchor frame, hold, encoder trunk, tracker capacity, data, budget — is
# shared. Hold is 1 (SONIC-style: the encoder re-encodes every control step),
# so there is NO phase channel and the actor command is exactly Z_DIM wide.
#
#   cont_det   deterministic 64-D continuous (z_norm regularized)
#   vq         EMA VQ-VAE, one codebook of 512
#   group_vq   multi-categorical straight-through: 8 groups x 32 categories
#   fsq64      finite scalar quantization, 64 scalars x 32 levels (SONIC's)
#   jepa_ntp   chunk-wise NTP + JEPA EMA target + spectral EBM (InfoNCE)
#   cont_det_ln  cont_det with hidden LayerNorm (the z256 lineage's recipe)
#
# 100M frames is a DIRECTION check per AGENTS.md's local-budget guidance
# (~50M serious check, 100M block ceiling), not a convergence claim. Winners
# go to the cluster for real budgets.
#
# Usage: run.sh <arm> <pretrain|lowlevel|both>
set -euo pipefail
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

ARM="${1:?usage: run.sh <cont_det|vq|group_vq|fsq64> <pretrain|lowlevel|both>}"
STAGE="${2:?usage: run.sh <arm> <pretrain|lowlevel|both>}"

ANCHOR_MODE="robot_heading"
TASK="${TASK:-Isaac-Imitation-G1-v2}"
AGENT_ENTRY_POINT="${AGENT_ENTRY_POINT:-rlopt_ipmd_tuned_cfg_entry_point}"
SEED="${SEED:-0}"
Z_DIM=64
HOLD=1

# LayerNorm is a PER-ARM axis; every arm except the *_ln arms matches the
# deployed fsq64 / released-SONIC recipe (no LN, silu). A 2026-08-14 edit made
# LN unconditional, which turned the *_ln arms into no-op duplicates of their
# siblings and gave every other arm LN too — so the README's "+15 survival
# points for LN" comparison could not have been produced by that revision.
# Restored 2026-08-15; encoders pretrained from the unconditional-LN revision
# have LN regardless of arm name.
LN_ARG=()
case "${ARM}" in
    cont_det) PRETRAIN_MODE_ARGS=(--latent_mode deterministic) ;;
    # Same bottleneck as cont_det; the ONLY difference is hidden LayerNorm.
    # Settles whether the z256 lineage's LN (vs the SONIC lineage's bare MLP)
    # matters at a matched trunk, bottleneck, and budget.
    cont_det_ln)
        PRETRAIN_MODE_ARGS=(--latent_mode deterministic)
        LN_ARG=(--encoder_layer_norm)
        ;;
    vq)       PRETRAIN_MODE_ARGS=(--latent_mode vq --vq_codebook_size 512) ;;
    group_vq) PRETRAIN_MODE_ARGS=(--latent_mode categorical
                                  --categorical_groups 8 --categorical_categories 32) ;;
    fsq64)    PRETRAIN_MODE_ARGS=(--latent_mode sonic_fsq) ;;
    # LN on the DEPLOYED bottleneck. cont_det_ln confounds LayerNorm with the
    # continuous bottleneck; this isolates LN as the single variable vs fsq64.
    fsq64_ln)
        PRETRAIN_MODE_ARGS=(--latent_mode sonic_fsq)
        LN_ARG=(--encoder_layer_norm)
        ;;
    # Chunk-wise next-token prediction with a JEPA-style EMA target encoder
    # and a bilinear (spectral) energy head trained by symmetric InfoNCE —
    # the report's future-work method. Continuous 64-D token; the objective,
    # not the bottleneck, is what changes.
    jepa_ntp) PRETRAIN_MODE_ARGS=(--latent_mode deterministic
                                  --transition_objective jepa_ntp
                                  --jepa_loss infonce) ;;
    # Ours: chunk NTP grounded by the DiffSR spectral EBM, SIGReg anti-collapse.
    jepa_sigreg_ebm) PRETRAIN_MODE_ARGS=(--latent_mode deterministic
                                  --transition_objective jepa_ntp
                                  --jepa_loss sigreg_ebm) ;;
    # Pure LeJEPA-style next-chunk prediction: MSE + SIGReg only.
    jepa_pure) PRETRAIN_MODE_ARGS=(--latent_mode deterministic
                                  --transition_objective jepa_ntp
                                  --jepa_loss sigreg) ;;
    # Stride-5 variants: SONIC's sparse cadence. The chunk samplers work in
    # SLOT space, so the env stride is the only change - a 10-slot chunk then
    # spans 0.9s and the next chunk starts 50 frames later. The stride is
    # recorded in the checkpoint and the low-level guard refuses a mismatch.
    # Prior record: our stride-5 encoders collapsed under the OLD recipe
    # (0.68 SR vs 0.90); fsq64_s5 is the control for whether the new frame,
    # hold 1, and recipe change that.
    fsq64_s5)           STRIDE=5; PRETRAIN_MODE_ARGS=(--latent_mode sonic_fsq) ;;
    jepa_pure_s5)       STRIDE=5; PRETRAIN_MODE_ARGS=(--latent_mode deterministic
                                  --transition_objective jepa_ntp
                                  --jepa_loss sigreg) ;;
    jepa_sigreg_ebm_s5) STRIDE=5; PRETRAIN_MODE_ARGS=(--latent_mode deterministic
                                  --transition_objective jepa_ntp
                                  --jepa_loss sigreg_ebm) ;;
    # Online-dynamics arms (obs-ring design, wiki/skill-encoder-jepa-plan.md as
    # amended 2026-08-14): reuse a completed arm's pretrained encoder, continue
    # training it DURING RL with the SAME DiffSR endpoint objective on ACHIEVED
    # windows from the raw-pose ring, mixed 1:1 with expert windows. NO policy
    # gradient (pg_coeff=0), per the design doc.
    fsq64_dyn)       BASE_ARM=fsq64;       DYN=1; PRETRAIN_MODE_ARGS=() ;;
    # SONIC-style adaptive-reset curriculum: uniform fraction 0.8 -> 0.2 over
    # the run, i.e. the failure-weighted share grows 20% -> 80%. Late training
    # concentrates environments on the trajectories the tracker keeps failing.
    # Reference-reset arm: reproduce the reset scheme of the 3.7B z256 hold-1
    # run (wandb 2zxhc8su), which at MATCHED 100M frames reached ep_len 120 vs
    # ~41 for this campaign's arms. That run used SONIC's full-trajectory
    # sampler (joint rank+frame, 0.8 random-trajectory ratio, no [0,200] start
    # window); our arms inherited the planner-thread eval convention instead.
    # One variable: the reset scheme.
    # Validation arm: the fsq64 recipe with the tuned contract, expecting the
    # reference trajectory (ep_len ~120 @100M) if the diagnosis is right.
    # Anchor-frame A/B: identical to fsq64 except the macro window frame is
    # the pre-2026-08-13 `robot` (full live pose) convention the reference
    # (2zxhc8su) trained under. Needs its own encoder in that frame.
    fsq64_robotanchor)
        PRETRAIN_MODE_ARGS=(--latent_mode sonic_fsq); ANCHOR_MODE=robot
        ;;
    fsq64_tuned)
        PRETRAIN_MODE_ARGS=(--latent_mode sonic_fsq)
        ;;
    fsq64_oldreset)
        PRETRAIN_MODE_ARGS=(--latent_mode sonic_fsq); OLDRESET=1
        ;;
    fsq64_curriculum)
        PRETRAIN_MODE_ARGS=(--latent_mode sonic_fsq); CURRICULUM=1
        ;;
    # Stacked training-side recipe: online dynamics + smoothing + curriculum.
    # Interpretable only against the single-factor arms; answers "do the wins
    # compose", not "which one works".
    fsq64_dyn_smooth_curriculum)
        PRETRAIN_MODE_ARGS=(--latent_mode sonic_fsq)
        DYN=1; SMOOTH=1; CURRICULUM=1
        ;;
    # Smoothing arm: 3x action-rate penalty + SONIC's 10-step proprio history.
    # The ablation's validated survival predictor is ACTION roughness, so this
    # penalises it directly and gives the policy temporal context to act on.
    fsq64_smooth)
        PRETRAIN_MODE_ARGS=(--latent_mode sonic_fsq)
        SMOOTH=1
        ;;
    # Stacked best-guess recipe: LN + online dynamics + smoothing, all on the
    # deployed fsq64 bottleneck. Only interpretable against the single-factor
    # arms above; it answers "do the wins compose", not "which one works".
    fsq64_ln_dyn_smooth)
        PRETRAIN_MODE_ARGS=(--latent_mode sonic_fsq)
        LN_ARG=(--encoder_layer_norm); DYN=1; SMOOTH=1
        ;;
    # LN_ARG here only matters if pretrain is (wrongly) invoked; the dyn arm
    # reuses the completed cont_det_ln encoder, whose LN lives in the checkpoint.
    cont_det_ln_dyn) BASE_ARM=cont_det_ln; DYN=1; PRETRAIN_MODE_ARGS=()
        LN_ARG=(--encoder_layer_norm) ;;
    *) echo "arm must be cont_det|cont_det_ln|vq|group_vq|fsq64|jepa_ntp|jepa_sigreg_ebm|jepa_pure ${ARM}" >&2; exit 1 ;;
esac

# Defined before CURRICULUM_ARGS, which interpolates it under `set -u`.
FRAME_CAP="${FRAME_CAP:-100000000}"
SMOOTH_ARGS=()
# Tuned hold-1 low-level contract (2026-08-09-bones129k-hold1 / wandb
# 2zxhc8su). Diagnosed 2026-08-15: omitting these cost ~3x ep_len at 100M.
# TUNED=0 reproduces the untuned recipe the original 12-arm table used.
TUNED_ARGS=()
if [ "${TUNED:-1}" = "1" ]; then
    TUNED_ARGS=(
        env.rewards.action_rate_l2.weight=0.0
        env.rewards.tracking_reward_points.weight=4.0
        env.enable_termination_curriculum=true
        env.termination_curriculum_start_frames=5000000
        env.termination_curriculum_end_frames=30000000
        env.command_interface.reference.selection=random80_adaptive20
        env.data.reference_prefetch_mode=next
    )
fi
OLDRESET_ARGS=()
if [ "${OLDRESET:-0}" = "1" ]; then
    OLDRESET_ARGS=(
        env.command_interface.reference.selection.full_trajectory=true
        env.command_interface.reference.selection.random_trajectory_sampling_ratio=0.8
        env.command_interface.reference.selection.random_step_max=0
        env.command_interface.reference.selection.start_mode=auto
    )
fi
CURRICULUM_ARGS=()
if [ "${CURRICULUM:-0}" = "1" ]; then
    # Ramp over the whole frame budget so the schedule scales with FRAME_CAP.
    CURRICULUM_ARGS=(
        env.command_interface.reference.selection.start_mode=adaptive
        env.command_interface.reference.selection.adaptive_uniform_ratio=0.8
        env.command_interface.reference.selection.adaptive_uniform_ratio_final=0.2
        "env.command_interface.reference.selection.adaptive_ratio_ramp_frames=${FRAME_CAP}"
    )
fi
if [ "${SMOOTH:-0}" = "1" ]; then
    # 3x the default -0.1 action-rate weight, plus 10-step proprio histories on
    # the four terms the actor reads (expert/latent terms stay single-frame).
    SMOOTH_ARGS=(
        env.rewards.action_rate_l2.weight=-0.3
        env.observations.policy.projected_gravity.history_length=10
        env.observations.policy.base_ang_vel.history_length=10
        env.observations.policy.joint_pos_rel.history_length=10
        env.observations.policy.joint_vel_rel.history_length=10
        env.observations.policy.last_action.history_length=10
    )
fi

REF_ARRAYS="${REF_ARRAYS:-/mnt/hsstorage/fwu91/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
PERSIST_ID="${PERSIST_ID:-bones_seed_sonic_full_129785@e714bbff}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/logs/bones129k_latent_quant_ablation}"

PRETRAIN_NUM_ENVS="${PRETRAIN_NUM_ENVS:-16}"
PRETRAIN_UPDATES="${PRETRAIN_UPDATES:-50000}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-8192}"

TRAIN_NUM_ENVS="${TRAIN_NUM_ENVS:-16384}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-24}"
FRAMES_PER_BATCH=$((TRAIN_NUM_ENVS * ROLLOUT_STEPS))
MAX_ITERATIONS=$(((FRAME_CAP + FRAMES_PER_BATCH - 1) / FRAMES_PER_BATCH))
MINIBATCH_SIZE="${MINIBATCH_SIZE:-$((FRAMES_PER_BATCH * 3 / 4))}"
ONLINE_EXPERT_BATCH_SIZE="${ONLINE_EXPERT_BATCH_SIZE:-24576}"
SAVE_INTERVAL="${SAVE_INTERVAL:-50000000}"

RUNTIME_BODY_NAMES=(
    pelvis
    left_hip_roll_link left_knee_link left_ankle_roll_link
    right_hip_roll_link right_knee_link right_ankle_roll_link
    torso_link
    left_shoulder_roll_link left_elbow_link left_wrist_yaw_link
    right_shoulder_roll_link right_elbow_link right_wrist_yaw_link
)
BODY_NAMES_OVERRIDE="env.data.runtime_cache_body_names=[$(IFS=,; echo "${RUNTIME_BODY_NAMES[*]}")]"

MACRO_INTERFACE_OVERRIDES=(
    'env.expert_macro_state_terms=[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b]'
    "env.expert_macro_frame_stride=${STRIDE:-1}"
    "env.expert_macro_anchor_mode=${ANCHOR_MODE}"
)
DATA_OVERRIDES=(
    physics=newton_mjwarp
    env.data.manifest=null
    "env.data.reference_arrays_dir=${REF_ARRAYS}"
    "env.data.persist_id=${PERSIST_ID}"
    env.data.reference_arrays_resident=false
    env.data.reference_arrays_warm_workers=2
    env.data.runtime_cache_device=cpu
    env.data.macro_cache_device=cuda:0
    "${BODY_NAMES_OVERRIDE}"
)
TRACKER_CAPACITY=(
    'agent.policy.num_cells=[2048,2048,1024,1024,512,512]'
    'agent.value_function.num_cells=[2048,2048,1024,1024,512,512]'
    agent.policy.activation_fn=silu
    agent.value_function.activation_fn=silu
)

ENCODER_DIR="${OUTPUT_ROOT}/${BASE_ARM:-${ARM}}/encoder"
TRACKER_DIR="${OUTPUT_ROOT}/${ARM}/tracker"
FINETUNE_ARGS=(agent.ipmd.hl_skill_finetune_enabled=false)
if [ "${DYN:-0}" = "1" ]; then
    # Capacity 128 holds the stride-1 chunk pair (span 21) with margin; the
    # ring records raw poses, windows are heading-anchored at sample time.
    FINETUNE_ARGS=(
        env.achieved_ring_capacity=128
        agent.ipmd.hl_skill_finetune_enabled=true
        agent.ipmd.hl_skill_pg_coeff=0
        agent.ipmd.hl_skill_offline_diffsr_coeff=1.0
        agent.ipmd.hl_skill_achieved_coeff=1.0
        agent.ipmd.hl_skill_anchor_coeff=0.01
    )
fi

run_pretrain() {
    [ -e "${ENCODER_DIR}" ] && { echo "Refusing to overwrite ${ENCODER_DIR}" >&2; exit 1; }
    pixi run -e isaaclab python -u scripts/rlopt/train_hl_skill_diffsr.py \
        --task "${TASK}" --num_envs "${PRETRAIN_NUM_ENVS}" --seed "${SEED}" \
        --device cuda:0 --headless --assert-kitless \
        --output_dir "${ENCODER_DIR}" --logger_backend none \
        --horizon_steps 10 \
        --encoder_window_mode intermediate --transition_objective endpoint \
        --z_dim "${Z_DIM}" "${PRETRAIN_MODE_ARGS[@]}" \
        --encoder_hidden_dims 2048 1024 512 512 --encoder_activation silu \
        "${LN_ARG[@]}" \
        --diffsr_feature_dim 256 --diffsr_embed_dim 1024 \
        --diffsr_g_hidden_dims 1024 1024 512 \
        --diffsr_mu_hidden_dims 1024 1024 512 \
        --batch_size "${PRETRAIN_BATCH_SIZE}" --num_updates "${PRETRAIN_UPDATES}" \
        --log_interval 1000 --eval_batches 4 \
        "${DATA_OVERRIDES[@]}" "${MACRO_INTERFACE_OVERRIDES[@]}"
}

run_lowlevel() {
    local encoder="${ENCODER_DIR}/checkpoints/latest.pt"
    [ -f "${encoder}" ] || { echo "missing encoder: ${encoder}" >&2; exit 1; }
    [ -e "${TRACKER_DIR}" ] && { echo "Refusing to overwrite ${TRACKER_DIR}" >&2; exit 1; }
    pixi run -e isaaclab python -u scripts/rlopt/train.py \
        --task "${TASK}" --algo IPMD --agent "${AGENT_ENTRY_POINT}" \
        --num_envs "${TRAIN_NUM_ENVS}" --seed "${SEED}" --headless --assert-kitless \
        --max_iterations "${MAX_ITERATIONS}" \
        "agent.logger.log_dir=${TRACKER_DIR}" \
        agent.logger.backend=csv agent.logger.video=false \
        "agent.logger.exp_name=quant_${ARM}" \
        env.command_interface.actor=latent \
        "env.command_interface.actor.dim=${Z_DIM}" \
        env.command_interface.encoder=single \
        "agent.ipmd.latent_dim=${Z_DIM}" \
        agent.ipmd.command_source=hl_skill \
        "agent.ipmd.hl_skill_checkpoint_path=${encoder}" \
        agent.ipmd.hl_skill_horizon_steps=10 \
        agent.ipmd.hl_skill_command_mode=z \
        "agent.ipmd.latent_steps_min=${HOLD}" \
        "agent.ipmd.latent_steps_max=${HOLD}" \
        "agent.ipmd.latent_learning.code_period=${HOLD}" \
        agent.ipmd.latent_learning.command_phase_mode=none \
        "agent.ipmd.latent_learning.code_latent_dim=${Z_DIM}" \
        "${FINETUNE_ARGS[@]}" \
        ${TUNED_ARGS[@]+"${TUNED_ARGS[@]}"} \
        ${SMOOTH_ARGS[@]+"${SMOOTH_ARGS[@]}"} \
        ${OLDRESET_ARGS[@]+"${OLDRESET_ARGS[@]}"} \
        ${CURRICULUM_ARGS[@]+"${CURRICULUM_ARGS[@]}"} \
        "agent.collector.frames_per_batch=${ROLLOUT_STEPS}" \
        "agent.loss.mini_batch_size=${MINIBATCH_SIZE}" \
        "agent.ipmd.expert_batch_size=${ONLINE_EXPERT_BATCH_SIZE}" \
        agent.loss.gamma=0.97 \
        "agent.save_interval=${SAVE_INTERVAL}" \
        env.sim.physics.solver_cfg.njmax=320 \
        env.sim.physics.solver_cfg.nconmax=200 \
        "${DATA_OVERRIDES[@]}" "${MACRO_INTERFACE_OVERRIDES[@]}" \
        "${TRACKER_CAPACITY[@]}"
}

case "${STAGE}" in
    pretrain) run_pretrain ;;
    lowlevel) run_lowlevel ;;
    both)     run_pretrain && run_lowlevel ;;
    *) echo "stage must be pretrain|lowlevel|both" >&2; exit 1 ;;
esac
echo "retained: ${OUTPUT_ROOT}/${ARM}"
