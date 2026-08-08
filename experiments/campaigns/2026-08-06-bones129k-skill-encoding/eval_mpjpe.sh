#!/usr/bin/env bash
set -euo pipefail

# MPJPE evaluation of the four completed BONES-129k skill-factorization
# controllers, on the workstation.
#
# The campaign's own comparison was training reward, which is not the paper
# metric. This scores the same four checkpoints under the evaluation protocol:
# every arm on the same motions, the same start frame, the same seed, and both
# mandated passes --
#
#   strict      every termination active; survival and success
#   fullhorizon every early termination off INCLUDING base_too_low, so MPJPE is
#               measured over the whole horizon rather than a truncated rollout
#
# The termination curriculum is switched OFF here even though training had it
# on. It anneals by cumulative frames, so an evaluation process starting at
# frame 0 would score against the LOOSE opening thresholds instead of the
# strict ones every arm finished training under.
#
# NOT matched frames: the endpoint control's only surviving checkpoint is
# 7.55B against the three arms' 5.00B (its 5B save was removed in the
# 2026-08-07 quota prune). The extra 2.55B favours the control, so treat a
# control win as an upper bound and an arm win as decisive.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
while [[ ! -d "${REPO_ROOT}/source/imitation_experiments" ]]; do
    [[ "${REPO_ROOT}" != "/" ]] || { echo "[FATAL] repository root not found" >&2; exit 2; }
    REPO_ROOT="$(dirname "${REPO_ROOT}")"
done
cd "${REPO_ROOT}"

fail() { echo "[FATAL] $*" >&2; exit 1; }

PULL_ROOT="${PULL_ROOT:-/mnt/hsstorage/fwu91/ice_eval_pull}"
REF_ARRAYS="${REF_ARRAYS:-/mnt/hsstorage/fwu91/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
PERSIST_ID="${PERSIST_ID:-bones_seed_sonic_full_129785@e714bbff}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/logs/skill_encoding_mpjpe_eval}"

NUM_ENVS="${NUM_ENVS:-512}"
STEPS="${STEPS:-1000}"
SEED="${SEED:-0}"
START_FRAME="${START_FRAME:-0}"
ARMS="${ARMS:-endpoint_control state_occupancy semimarkov_chain endpoint_delta}"

SKILL_ROOT="${PULL_ROOT}/bones129k_skill_encoding"
arm_checkpoint() {
    case "$1" in
        endpoint_control)
            printf '%s' "${PULL_ROOT}/bones129k_latent_sampler/bones129k_reset80_diffsr_reset80_e16384_r24_10b_seed0/rlopt_train/2026-08-05_21-07-16_wandb-r09s1pc7/models/model_step_7550140416.pt" ;;
        state_occupancy)
            printf '%s' "${SKILL_ROOT}/bones129k_skill_state_occupancy_h10_z256_seed0/rlopt_train/2026-08-06_14-35-25_wandb-8la4o48g/models/model_step_5000134656.pt" ;;
        semimarkov_chain)
            printf '%s' "${SKILL_ROOT}/bones129k_skill_semimarkov_chain_h10_z256_seed0/rlopt_train/2026-08-06_14-35-25_wandb-k7s6uha0/models/model_step_5000134656.pt" ;;
        endpoint_delta)
            printf '%s' "${SKILL_ROOT}/bones129k_skill_endpoint_delta_h10_z256_seed0/rlopt_train/2026-08-06_14-35-18_wandb-3296logf/models/model_step_5000134656.pt" ;;
        *) fail "unknown arm $1" ;;
    esac
}

arm_encoder() {
    case "$1" in
        endpoint_control)
            printf '%s' "${PULL_ROOT}/pretrain_store/bones129k_v2_root_qpos_det_sr_h10_z256_seed0/checkpoints/latest.pt" ;;
        *)
            printf '%s' "${SKILL_ROOT}/bones129k_skill_$1_h10_z256_seed0/encoder/checkpoints/latest.pt" ;;
    esac
}

RUNTIME_BODY_NAMES="[pelvis,left_hip_roll_link,left_knee_link,left_ankle_roll_link,right_hip_roll_link,right_knee_link,right_ankle_roll_link,torso_link,left_shoulder_roll_link,left_elbow_link,left_wrist_yaw_link,right_shoulder_roll_link,right_elbow_link,right_wrist_yaw_link]"

# Exactly the training contract, minus the frame-counted curriculum.
common_overrides() {
    local encoder="$1"
    COMMON_OVERRIDES=(
        physics=newton_mjwarp
        env.data.manifest=null
        "env.data.reference_arrays_dir=${REF_ARRAYS}"
        "env.data.persist_id=${PERSIST_ID}"
        env.data.reference_arrays_resident=false
        env.data.reference_arrays_warm_workers=8
        env.data.runtime_cache_device=cpu
        env.data.reference_prefetch_mode=off
        env.data.macro_cache_device=cuda:0
        "env.data.runtime_cache_body_names=${RUNTIME_BODY_NAMES}"
        env.expert_macro_state_terms=[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b]
        env.command_interface.actor=latent
        env.command_interface.actor.dim=258
        env.command_interface.encoder=single
        env.rewards.action_rate_l2.weight=0.0
        env.rewards.tracking_reward_points.weight=4.0
        env.enable_termination_curriculum=false
        env.sim.physics.solver_cfg.njmax=289
        env.sim.physics.solver_cfg.nconmax=200
        agent.ipmd.command_source=hl_skill
        "agent.ipmd.hl_skill_checkpoint_path=${encoder}"
        agent.ipmd.hl_skill_horizon_steps=10
        agent.ipmd.hl_skill_command_mode=z
        agent.ipmd.hl_skill_finetune_enabled=false
        agent.ipmd.latent_dim=258
        agent.ipmd.latent_steps_min=10
        agent.ipmd.latent_steps_max=10
        agent.ipmd.latent_learning.code_period=10
        agent.ipmd.latent_learning.command_phase_mode=sin_cos
        agent.ipmd.latent_learning.code_latent_dim=256
    )
}

mkdir -p "${OUTPUT_ROOT}"
echo "[INFO] arms=${ARMS}"
echo "[INFO] ${NUM_ENVS} envs x ${STEPS} steps, seed ${SEED}, start frame ${START_FRAME}"
echo "[INFO] output: ${OUTPUT_ROOT}"

for arm in ${ARMS}; do
    checkpoint="$(arm_checkpoint "${arm}")"
    encoder="$(arm_encoder "${arm}")"
    [[ -f "${checkpoint}" ]] || fail "missing checkpoint ${checkpoint}"
    [[ -f "${encoder}" ]] || fail "missing encoder ${encoder}"
    common_overrides "${encoder}"

    for pass in strict fullhorizon; do
        out_json="${OUTPUT_ROOT}/${arm}_${pass}.json"
        if [[ -f "${out_json}" ]]; then
            echo "[SKIP] ${arm}/${pass} already scored"
            continue
        fi
        cmd=(
            env TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
            HYDRA_FULL_ERROR=1 TORCHDYNAMO_DISABLE=1 WANDB_MODE=disabled
            pixi run -e isaaclab python -u
            -m imitation_experiments.lowlevel.evaluate_checkpoint
            --task Isaac-Imitation-G1-v2 --algo IPMD
            --agent_entry_point rlopt_ipmd_tuned_cfg_entry_point
            --checkpoint "${checkpoint}"
            --num_envs "${NUM_ENVS}" --steps "${STEPS}" --seed "${SEED}"
            --reference_start_frame "${START_FRAME}"
            --label "${arm}_${pass}"
            --output_json "${out_json}"
            --headless
        )
        [[ "${pass}" == "fullhorizon" ]] && cmd+=(--disable_early_terminations)
        cmd+=("${COMMON_OVERRIDES[@]}")
        echo "[RUN] ${arm}/${pass}"
        "${cmd[@]}" 2>&1 | tee "${OUTPUT_ROOT}/${arm}_${pass}.log" | tail -5
    done
done

echo "[DONE] ${OUTPUT_ROOT}"
