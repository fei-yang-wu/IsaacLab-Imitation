#!/usr/bin/env bash
set -euo pipefail

# Focused strong-explicit diagnostic. Reuse the exact H10 causal rows, extend
# only their oracle labels to H30, fit one medium seed-0 planner, then evaluate
# that same checkpoint by (a) executing its first H10 and (b) overlapping H30
# predictions with temporal ensembling before the frozen enc380 encoder.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
while [[ ! -x "${REPO_ROOT}/docker/cluster/cluster_interface.sh" ]]; do
    if [[ "${REPO_ROOT}" == / ]]; then
        echo "[ERROR] Could not locate repository root above ${SCRIPT_DIR}." >&2
        exit 2
    fi
    REPO_ROOT="$(dirname "${REPO_ROOT}")"
done
cd "${REPO_ROOT}"

: "${LOW_LEVEL_CHECKPOINT:?Set LOW_LEVEL_CHECKPOINT to the enc380 latent tracker.}"
: "${SKILL_CHECKPOINT:?Set SKILL_CHECKPOINT to the frozen enc380 encoder.}"
: "${TRACKER_COMPLETION_RECORD:?Set TRACKER_COMPLETION_RECORD to its verified 5B record.}"
: "${SOURCE_STUDY_ROOT:?Set SOURCE_STUDY_ROOT to the matched H10 study.}"

MANIFEST="${MANIFEST:-/data/lafan1_corrected_8e95d557/manifests/g1_lafan1_manifest.json}"
DATASET_PATH="${DATASET_PATH:-/data/lafan1_corrected_8e95d557/g1_hl_diffsr}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/interface_baselines/lafan1_enc380_h30_temporal_screen}"
STAGES="${STAGES:-materialize train eval aggregate}"
DRY_RUN="${DRY_RUN:-0}"
DEVICE="${DEVICE:-cuda:0}"
TASK="${TASK:-Isaac-Imitation-G1-Latent-Strict-v0}"
MOTION_NAME="walk1_subject1"
SEED=0
BATCH_SIZE=1024
MODEL_SIZE="${MODEL_SIZE:-medium}"
FLOW_STEPS=16
EVAL_STEPS="${EVAL_STEPS:-500}"
EVAL_ENVS="${EVAL_ENVS:-10}"
TEMPORAL_DECAY="${TEMPORAL_DECAY:-0.5}"
EXPECTED_LOW_LEVEL_SHA256="${EXPECTED_LOW_LEVEL_SHA256:-}"
EXPECTED_SKILL_SHA256="${EXPECTED_SKILL_SHA256:-}"

: "${ISAAC_PY:=pixi run -e isaaclab python}"
: "${PLAIN_PY:=pixi run python}"
read -r -a ISAAC_PY_ARR <<<"${ISAAC_PY}"
read -r -a PLAIN_PY_ARR <<<"${PLAIN_PY}"

IFACE_DIR="${SCRIPT_DIR}/interface_baselines"
SHARED_DIR="experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines"
SOURCE_MOTION_ROOT="${SOURCE_STUDY_ROOT}/motions/${MOTION_NAME}"
SOURCE_H10="${SOURCE_MOTION_ROOT}/demonstrations/root_qpos"
SOURCE_DEMO_AUDIT="${SOURCE_MOTION_ROOT}/demonstrations/paired_demonstration_audit.json"
SOURCE_QUAL_AUDIT="${SOURCE_STUDY_ROOT}/qualification/latent_qualification_audit.json"
H30_DEMOS="${OUTPUT_ROOT}/demonstrations/root_qpos_h30"

case "${MODEL_SIZE}" in
    tiny)
        default_updates=10000
        default_micro_batch=1024
        ;;
    small)
        default_updates=20000
        default_micro_batch=512
        ;;
    medium)
        default_updates=30000
        default_micro_batch=256
        ;;
    large)
        default_updates=50000
        default_micro_batch=128
        ;;
    *)
        echo "[ERROR] Unknown model size: ${MODEL_SIZE}" >&2
        exit 2
        ;;
esac
TRAIN_UPDATES="${TRAIN_UPDATES:-${default_updates}}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-${default_micro_batch}}"
PLANNER="${OUTPUT_ROOT}/planner/${MODEL_SIZE}/seed0/planner_oracle_u${TRAIN_UPDATES}_b${BATCH_SIZE}"

case "${DRY_RUN}" in
    1|true|TRUE|yes|YES) DRY_RUN=1 ;;
    0|false|FALSE|no|NO) DRY_RUN=0 ;;
    *) echo "[ERROR] DRY_RUN must be boolean, got ${DRY_RUN}." >&2; exit 2 ;;
esac
has_stage() { [[ " ${STAGES} " == *" $1 "* ]]; }

require_file() {
    [[ "${DRY_RUN}" == "1" || -f "$1" ]] || {
        echo "[ERROR] required file missing: $1" >&2
        exit 2
    }
}
require_dir() {
    [[ "${DRY_RUN}" == "1" || -d "$1" ]] || {
        echo "[ERROR] required directory missing: $1" >&2
        exit 2
    }
}
run_if_missing() {
    local marker="$1"; shift
    if [[ -e "${marker}" ]]; then
        echo "[SKIP] ${marker}"
        return 0
    fi
    printf '[CMD]'; printf ' %q' "$@"; printf '\n'
    [[ "${DRY_RUN}" == "1" ]] && return 0
    "$@"
    [[ -e "${marker}" ]] || {
        echo "[ERROR] command completed without marker ${marker}." >&2
        exit 2
    }
}
verify_sha() {
    local path="$1" expected="$2" label="$3"
    [[ "${DRY_RUN}" == "1" || -z "${expected}" ]] && return 0
    local actual
    actual="$(sha256sum "${path}" | awk '{print $1}')"
    [[ "${actual}" == "${expected}" ]] || {
        echo "[ERROR] ${label} sha256 mismatch: expected ${expected}, got ${actual}." >&2
        exit 2
    }
}

require_file "${LOW_LEVEL_CHECKPOINT}"
require_file "${SKILL_CHECKPOINT}"
require_file "${TRACKER_COMPLETION_RECORD}"
require_file "${MANIFEST}"
require_dir "${DATASET_PATH}"
require_file "${SOURCE_QUAL_AUDIT}"
require_file "${SOURCE_DEMO_AUDIT}"
require_dir "${SOURCE_H10}"
verify_sha "${LOW_LEVEL_CHECKPOINT}" "${EXPECTED_LOW_LEVEL_SHA256}" tracker
verify_sha "${SKILL_CHECKPOINT}" "${EXPECTED_SKILL_SHA256}" encoder
if [[ "${DRY_RUN}" != "1" ]]; then
    grep -q '"protocol_passed": true' "${SOURCE_QUAL_AUDIT}"
    grep -q '"oracle_passed": true' "${SOURCE_QUAL_AUDIT}"
    grep -q '"passed": true' "${SOURCE_DEMO_AUDIT}"
fi

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
ENV_CFG=(
    agent.logger.backend=
    agent.ipmd.hl_skill_finetune_enabled=false
    "agent.ipmd.hl_skill_checkpoint_path=${SKILL_CHECKPOINT}"
    "env.lafan1_manifest_path=${MANIFEST}"
    "env.dataset_path=${DATASET_PATH}"
    env.refresh_zarr_dataset=false
    env.reset_schedule=sequential env.wrap_steps=false
    env.random_reset_step_min=0 env.random_reset_step_max=200
    env.random_reset_full_trajectory=false
    env.observations.policy.enable_corruption=false
    "env.expert_macro_state_terms=[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b]"
    physics=newton_mjwarp
    env.sim.physics.solver_cfg.njmax=320
    env.sim.physics.solver_cfg.nconmax=40
)
KITLESS_ARGS=()
[[ "${ASSERT_KITLESS:-0}" == "1" ]] && KITLESS_ARGS=(--assert-kitless)

if has_stage materialize; then
    run_if_missing "${H30_DEMOS}/materialization_manifest.json" \
        "${PLAIN_PY_ARR[@]}" -m imitation_experiments.capacity.materialize_long_horizon_root_qpos \
        --samples_dir "${SOURCE_H10}" --dataset_path "${DATASET_PATH}" \
        --output_dir "${H30_DEMOS}" --horizon_steps 30 \
        --anchor_body_name pelvis
fi

if has_stage train; then
    require_file "${H30_DEMOS}/materialization_manifest.json"
    run_if_missing "${PLANNER}/checkpoints/best.pt" \
        "${PLAIN_PY_ARR[@]}" -m imitation_experiments.planner.train_chunked_transformer_planner \
        --samples_dir "${H30_DEMOS}" --output_dir "${PLANNER}" \
        --interface root_qpos --planner_family flow --state_key planner_state \
        --training_stage oracle --device "${DEVICE}" --seed "${SEED}" \
        --model_size "${MODEL_SIZE}" --batch_size "${BATCH_SIZE}" \
        --micro_batch_size "${MICRO_BATCH_SIZE}" --num_updates "${TRAIN_UPDATES}" \
        --max_samples 0 --lr 0.0001 --weight_decay 0.0001 \
        --flow_num_inference_steps "${FLOW_STEPS}" \
        --endpoint_num_inference_steps 4 --flow_inference_noise_std 0.0
fi

run_eval() {
    local mode="$1" pass="$2"
    local ensemble="none" output="${OUTPUT_ROOT}/evaluation/${MODEL_SIZE}/${mode}/${pass}"
    [[ "${mode}" == "temporal_exponential" ]] && ensemble="exponential"
    local pass_args=()
    if [[ "${pass}" == "survival" ]]; then
        pass_args=(--keep_early_terminations --disable_tracking_terminations)
    else
        pass_args=(--extend_episode_length_for_max_steps)
        if [[ "${RENDER_VIDEO:-1}" == "1" && "${ASSERT_KITLESS:-0}" != "1" ]]; then
            pass_args+=(--video --video_length "${EVAL_STEPS}")
        fi
    fi
    run_if_missing "${output}/summary.json" \
        "${ISAAC_PY_ARR[@]}" scripts/rlopt/eval_skill_commander_closed_loop.py \
        "${KITLESS_ARGS[@]}" --headless --device "${DEVICE}" --task "${TASK}" \
        --algorithm IPMD --checkpoint "${LOW_LEVEL_CHECKPOINT}" \
        --skill_checkpoint "${SKILL_CHECKPOINT}" --state_history_steps 9 \
        --packet_planner_checkpoint "${PLANNER}/checkpoints/best.pt" \
        --packet_interface root_qpos --packet_source planner \
        --packet_prediction_horizon_steps 30 \
        --packet_temporal_ensemble "${ensemble}" \
        --packet_temporal_ensemble_decay "${TEMPORAL_DECAY}" \
        --output_dir "${output}" \
        --label "enc380_h30_${MODEL_SIZE}_${mode}_${pass}" \
        --num_envs "${EVAL_ENVS}" --max_steps "${EVAL_STEPS}" --seed "${SEED}" \
        --metric_interval 10 --motion_name "${MOTION_NAME}" \
        --allow_random_reset --keep_time_out --disable_reward_clipping \
        --flow_num_inference_steps "${FLOW_STEPS}" \
        --flow_inference_noise_std 0.0 \
        --kit_args=--/app/extensions/fsWatcherEnabled=false \
        "${pass_args[@]}" agent.ipmd.command_source=hl_skill \
        "${LATENT_CFG[@]}" "${ENV_CFG[@]}"
}

if has_stage eval; then
    require_file "${PLANNER}/checkpoints/best.pt"
    run_eval execute_first10 survival
    run_eval execute_first10 full_horizon
    run_eval temporal_exponential survival
    run_eval temporal_exponential full_horizon
fi

if has_stage aggregate; then
    run_if_missing "${OUTPUT_ROOT}/aggregate/results.json" \
        "${PLAIN_PY_ARR[@]}" -m imitation_experiments.capacity.aggregate_h30_temporal_ensemble \
        --output_root "${OUTPUT_ROOT}" --source_study_root "${SOURCE_STUDY_ROOT}"
fi

echo "[PASS] enc380 H30 temporal diagnostic stages complete: ${STAGES}"
