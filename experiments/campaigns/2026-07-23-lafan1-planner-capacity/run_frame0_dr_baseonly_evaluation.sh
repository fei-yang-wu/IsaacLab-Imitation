#!/usr/bin/env bash
set -euo pipefail

# User-requested diagnostic protocol:
#   - walk1_subject1, reference frame 0
#   - 100 parallel environments, one eval seed
#   - training-time physical domain randomization enabled
#   - base_too_low (torso < 0.40 m) is the only early termination
#   - MPJPE is accumulated only over valid pre-termination transitions

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

STUDY_ROOT="${STUDY_ROOT:-logs/interface_baselines/lafan1_enc380_route_capacity_5b_oracle100_progressive_b1024_20260730}"
MOTION_NAME="${MOTION_NAME:-walk1_subject1}"
MODEL_SIZES="${MODEL_SIZES:-tiny small medium large}"
PLANNER_SEED="${PLANNER_SEED:-0}"
EVAL_SEED="${EVAL_SEED:-0}"
EVAL_ENVS="${EVAL_ENVS:-100}"
EVAL_STEPS="${EVAL_STEPS:-500}"
FALL_HEIGHT_M="${FALL_HEIGHT_M:-0.4}"
ROUTES="${ROUTES:-root_qpos latent_skill h30_first10 h30_temporal pure_root_qpos}"
DEVICE="${DEVICE:-cuda:0}"
H30_ROOT="${H30_ROOT:-logs/interface_baselines/lafan1_enc380_h30_temporal_medium_seed0_20260730}"
H30_TEMPORAL_DECAY="${H30_TEMPORAL_DECAY:-0.5}"

MANIFEST="${MANIFEST:-data/lafan1/manifests/g1_lafan1_manifest.json}"
LATENT_DATASET_PATH="${LATENT_DATASET_PATH:-/tmp/iltools_g1_lafan1_tracking_corrected_8029acbce33a}"
EXPLICIT_DATASET_PATH="${EXPLICIT_DATASET_PATH:-/tmp/iltools_g1_lafan1_tracking_corrected_8029acbce33a}"
ENC380_TRACKER="${ENC380_TRACKER:-logs/downloaded_checkpoints/lafan1_enc380_rootqpos_h10_z256_seed0/model_5b.pt}"
ENC380_SKILL="${ENC380_SKILL:-logs/downloaded_checkpoints/lafan1_enc380_rootqpos_h10_z256_seed0/skill_encoder/latest.pt}"
PURE_ROOT_TRACKER="${PURE_ROOT_TRACKER:-logs/downloaded_checkpoints/lafan1_rootqpos_5b_seed0/model_step_4600037376.pt}"
PURE_ROOT_ROOT="${PURE_ROOT_ROOT:-${STUDY_ROOT}/pure_root_qpos_tracker}"

: "${ISAAC_PY:=pixi run -e isaaclab python}"
read -r -a ISAAC_PY_ARR <<<"${ISAAC_PY}"

LATENT_CFG=(
    env.latent_command_dim=258 agent.ipmd.latent_dim=258
    agent.ipmd.hl_skill_horizon_steps=10 agent.ipmd.hl_skill_command_mode=z
    agent.ipmd.latent_steps_min=10 agent.ipmd.latent_steps_max=10
    agent.ipmd.latent_learning.command_phase_mode=sin_cos
    agent.ipmd.latent_learning.code_latent_dim=256
    agent.ipmd.latent_learning.code_period=10
    agent.ipmd.reward_loss_coeff=0.0 agent.ipmd.reward_l2_coeff=0.0
    agent.ipmd.reward_grad_penalty_coeff=0.0
    agent.ipmd.reward_logit_reg_coeff=0.0
    agent.ipmd.reward_param_weight_decay_coeff=0.0
)
ENV_COMMON=(
    agent.logger.backend=
    env.refresh_zarr_dataset=false
    env.reset_schedule=sequential env.wrap_steps=false
    env.reference_start_frame=0
    env.random_reset_step_min=0 env.random_reset_step_max=0
    env.random_reset_full_trajectory=false
    env.observations.policy.enable_corruption=false
    physics=newton_mjwarp
    env.sim.physics.solver_cfg.njmax=320
    env.sim.physics.solver_cfg.nconmax=40
)

updates_for_size() {
    case "$1" in
        tiny) echo 10000 ;;
        small) echo 20000 ;;
        medium) echo 30000 ;;
        large) echo 50000 ;;
        *) echo "[ERROR] Unknown model size $1." >&2; return 2 ;;
    esac
}

run_if_missing() {
    local marker="$1"
    shift
    if [[ -f "${marker}" ]]; then
        echo "[SKIP] ${marker}"
        return 0
    fi
    mkdir -p "$(dirname "${marker}")"
    printf '[CMD]'; printf ' %q' "$@"; printf '\n'
    "$@"
    [[ -f "${marker}" ]] || {
        echo "[ERROR] Evaluation completed without ${marker}." >&2
        exit 2
    }
}

# Some transferred planner metadata retains the immutable container-side
# encoder path. Redirect only that exact read to the SHA-verified local copy.
redirect_dir="${STUDY_ROOT}/local_eval_runtime/path_redirect"
if [[ -f "${redirect_dir}/sitecustomize.py" ]]; then
    export PYTHONPATH="${redirect_dir}${PYTHONPATH:+:${PYTHONPATH}}"
    export ISAAC_LOCAL_RELOCATE_SOURCE="/data/enc380_store/lafan1_enc380_rootqpos_h10_z256_seed0/skill_encoder/checkpoints/latest.pt"
    export ISAAC_LOCAL_RELOCATE_TARGET="$(realpath "${ENC380_SKILL}")"
fi

eval_enc380() {
    local route="$1" size="$2" updates planner route_root output
    updates="$(updates_for_size "${size}")"
    route_root="${STUDY_ROOT}/motions/${MOTION_NAME}/capacity/${size}/seed${PLANNER_SEED}/matched/${route}"
    planner="${route_root}/planner_oracle_u${updates}_b1024/checkpoints/best.pt"
    output="${route_root}/eval_frame0_dr_baseonly_100env_seed${EVAL_SEED}"
    [[ -f "${planner}" ]] || {
        echo "[WAIT] Missing planner ${planner}"
        return 0
    }

    route_args=()
    if [[ "${route}" == "root_qpos" ]]; then
        route_args=(
            --packet_planner_checkpoint "${planner}"
            --packet_interface root_qpos
            --packet_source planner
            agent.ipmd.command_source=hl_skill
        )
    else
        route_args=(
            --planner_checkpoint "${planner}"
            agent.ipmd.command_source=skill_commander
            "agent.ipmd.skill_commander_checkpoint_path=${planner}"
            agent.ipmd.skill_commander_use_achieved_state=true
            agent.ipmd.skill_commander_flow_num_inference_steps=16
            agent.ipmd.skill_commander_flow_inference_noise_std=0.0
        )
    fi

    run_if_missing "${output}/summary.json" \
        "${ISAAC_PY_ARR[@]}" scripts/rlopt/eval_skill_commander_closed_loop.py \
        --headless --device "${DEVICE}" \
        --task Isaac-Imitation-G1-Latent-Strict-v0 --algorithm IPMD \
        --checkpoint "${ENC380_TRACKER}" --skill_checkpoint "${ENC380_SKILL}" \
        --state_history_steps 9 --output_dir "${output}" \
        --label "frame0_dr_baseonly_${route}_${size}" \
        --num_envs "${EVAL_ENVS}" --max_steps "${EVAL_STEPS}" \
        --seed "${EVAL_SEED}" --metric_interval 10 \
        --motion_name "${MOTION_NAME}" --base_only_termination \
        --fall_height_m "${FALL_HEIGHT_M}" --extend_episode_length_for_max_steps \
        --disable_reward_clipping --flow_num_inference_steps 16 \
        --flow_inference_noise_std 0.0 \
        --kit_args=--/app/extensions/fsWatcherEnabled=false \
        "agent.ipmd.hl_skill_checkpoint_path=${ENC380_SKILL}" \
        agent.ipmd.hl_skill_finetune_enabled=false \
        "env.lafan1_manifest_path=${MANIFEST}" \
        "env.dataset_path=${LATENT_DATASET_PATH}" \
        "env.expert_macro_state_terms=[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b]" \
        "${route_args[@]}" "${LATENT_CFG[@]}" "${ENV_COMMON[@]}"
}

eval_pure_root() {
    local size="$1" updates planner output
    updates="$(updates_for_size "${size}")"
    planner="${PURE_ROOT_ROOT}/${size}/seed0/planner_oracle_u${updates}_b1024/checkpoints/best.pt"
    output="${PURE_ROOT_ROOT}/${size}/seed0/eval_frame0_dr_baseonly_100env_seed${EVAL_SEED}"
    if [[ ! -f "${planner}" ]]; then
        echo "[WAIT] Missing direct root_qpos planner ${planner}"
        return 0
    fi
    run_if_missing "${output}/summary.json" \
        "${ISAAC_PY_ARR[@]}" \
        experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/eval_interface_planner_closed_loop.py \
        --headless --device "${DEVICE}" \
        --task Isaac-Imitation-G1-Strict-v0 --algorithm IPMD \
        --checkpoint "${PURE_ROOT_TRACKER}" \
        --low_level_command_mode streamed_vanilla \
        --planner_checkpoint "${planner}" \
        --output_json "${output}/summary.json" \
        --label "frame0_dr_baseonly_pure_root_qpos_${size}" \
        --motion_manifest "${MANIFEST}" --motion_name "${MOTION_NAME}" \
        --dataset_path "${EXPLICIT_DATASET_PATH}" \
        --num_envs "${EVAL_ENVS}" --steps "${EVAL_STEPS}" \
        --seed "${EVAL_SEED}" --state_history_steps 9 \
        --command_past_steps 0 --command_future_steps 9 \
        --planner_update_interval 10 --flow_num_inference_steps 16 \
        --flow_inference_noise_std 0.0 --reset_schedule sequential \
        --reference_start_frame 0 --base_only_termination \
        --fall_height_m "${FALL_HEIGHT_M}" \
        --kit_args=--/app/extensions/fsWatcherEnabled=false \
        "${ENV_COMMON[@]}"
}

eval_h30() {
    local route="$1" size="$2" updates planner ensemble output
    case "${route}" in
        h30_first10) ensemble=none ;;
        h30_temporal) ensemble=exponential ;;
        *)
            echo "[ERROR] Unknown H30 route ${route}." >&2
            exit 2
            ;;
    esac
    updates="$(updates_for_size "${size}")"
    planner="${H30_ROOT}/planner/${size}/seed0/planner_oracle_u${updates}_b1024/checkpoints/best.pt"
    output="${H30_ROOT}/evaluation_frame0_dr_baseonly_100env_seed${EVAL_SEED}/${size}/${route}"
    if [[ ! -f "${planner}" ]]; then
        echo "[WAIT] Missing H30 planner ${planner}"
        return 0
    fi
    run_if_missing "${output}/summary.json" \
        "${ISAAC_PY_ARR[@]}" scripts/rlopt/eval_skill_commander_closed_loop.py \
        --headless --device "${DEVICE}" \
        --task Isaac-Imitation-G1-Latent-Strict-v0 --algorithm IPMD \
        --checkpoint "${ENC380_TRACKER}" --skill_checkpoint "${ENC380_SKILL}" \
        --state_history_steps 9 --output_dir "${output}" \
        --label "frame0_dr_baseonly_${route}_${size}" \
        --num_envs "${EVAL_ENVS}" --max_steps "${EVAL_STEPS}" \
        --seed "${EVAL_SEED}" --metric_interval 10 \
        --motion_name "${MOTION_NAME}" --base_only_termination \
        --fall_height_m "${FALL_HEIGHT_M}" --extend_episode_length_for_max_steps \
        --disable_reward_clipping --flow_num_inference_steps 16 \
        --flow_inference_noise_std 0.0 \
        --packet_planner_checkpoint "${planner}" \
        --packet_interface root_qpos --packet_source planner \
        --packet_prediction_horizon_steps 30 \
        --packet_temporal_ensemble "${ensemble}" \
        --packet_temporal_ensemble_decay "${H30_TEMPORAL_DECAY}" \
        --kit_args=--/app/extensions/fsWatcherEnabled=false \
        "agent.ipmd.hl_skill_checkpoint_path=${ENC380_SKILL}" \
        agent.ipmd.hl_skill_finetune_enabled=false \
        "env.lafan1_manifest_path=${MANIFEST}" \
        "env.dataset_path=${LATENT_DATASET_PATH}" \
        "env.expert_macro_state_terms=[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b]" \
        agent.ipmd.command_source=hl_skill \
        "${LATENT_CFG[@]}" "${ENV_COMMON[@]}"
}

for route in ${ROUTES}; do
    case "${route}" in
        root_qpos|latent_skill) ;;
        h30_first10|h30_temporal)
            for size in ${MODEL_SIZES}; do
                eval_h30 "${route}" "${size}"
            done
            continue
            ;;
        pure_root_qpos)
            for size in ${MODEL_SIZES}; do
                eval_pure_root "${size}"
            done
            continue
            ;;
        *)
            echo "[ERROR] Unknown route ${route}." >&2
            exit 2
            ;;
    esac
    for size in ${MODEL_SIZES}; do
        eval_enc380 "${route}" "${size}"
    done
done

echo "[PASS] Frame-0 DR/base-only evaluation pass complete for available planners."
