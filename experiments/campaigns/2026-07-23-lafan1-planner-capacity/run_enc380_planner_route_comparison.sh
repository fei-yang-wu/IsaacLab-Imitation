#!/usr/bin/env bash
set -euo pipefail

# Matched shared-tracker ablation:
#   root_qpos planner -> frozen enc380 encoder -> frozen enc380 latent tracker
#   latent planner ---------------------------> frozen enc380 latent tracker
# Both planners are trained once from the same oracle-collected causal rows and
# exact same expert windows. There is no learned-planner rollout or finetune.

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
: "${TRACKER_COMPLETION_RECORD:?Set TRACKER_COMPLETION_RECORD to the verified 5B record.}"

MANIFEST="${MANIFEST:-/data/lafan1_corrected_8e95d557/manifests/g1_lafan1_manifest.json}"
DATASET_PATH="${DATASET_PATH:-/data/lafan1_corrected_8e95d557/g1_hl_diffsr}"
TASK="${TASK:-Isaac-Imitation-G1-Latent-Strict-v0}"
MOTION_NAME="${MOTION_NAME:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/interface_baselines/lafan1_enc380_walk1_route_comparison}"
STAGES="${STAGES:-qualify demo train eval aggregate}"
DRY_RUN="${DRY_RUN:-0}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-0}"
MODEL_SIZE="${MODEL_SIZE:-medium}"
DEMO_TRAJECTORIES_PER_MOTION="${DEMO_TRAJECTORIES_PER_MOTION:-100}"
COLLECT_ENVS="${COLLECT_ENVS:-100}"
COLLECT_STEPS="${COLLECT_STEPS:-15000}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
case "${MODEL_SIZE}" in
    tiny)
        TRAIN_UPDATES="${TRAIN_UPDATES:-10000}"
        MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1024}"
        ;;
    small)
        TRAIN_UPDATES="${TRAIN_UPDATES:-20000}"
        MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-512}"
        ;;
    medium)
        TRAIN_UPDATES="${TRAIN_UPDATES:-30000}"
        MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-256}"
        ;;
    large)
        TRAIN_UPDATES="${TRAIN_UPDATES:-50000}"
        MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-128}"
        ;;
    *)
        echo "[ERROR] Unknown MODEL_SIZE=${MODEL_SIZE}." >&2
        exit 2
        ;;
esac
PLANNER_DIR_NAME="${PLANNER_DIR_NAME:-planner_oracle_u${TRAIN_UPDATES}_b${BATCH_SIZE}}"
FLOW_STEPS="${FLOW_STEPS:-16}"
EVAL_STEPS="${EVAL_STEPS:-500}"
EVAL_ENVS="${EVAL_ENVS:-10}"
QUALIFY_STEPS="${QUALIFY_STEPS:-500}"
QUALIFY_ENVS="${QUALIFY_ENVS:-100}"
MIN_ORACLE_SUCCESS="${MIN_ORACLE_SUCCESS:-0.8}"
EXPECTED_LOW_LEVEL_SHA256="${EXPECTED_LOW_LEVEL_SHA256:-}"
EXPECTED_SKILL_SHA256="${EXPECTED_SKILL_SHA256:-}"

: "${ISAAC_PY:=pixi run -e isaaclab python}"
: "${PLAIN_PY:=pixi run python}"
read -r -a ISAAC_PY_ARR <<<"${ISAAC_PY}"
read -r -a PLAIN_PY_ARR <<<"${PLAIN_PY}"

IFACE_DIR="${SCRIPT_DIR}/interface_baselines"
SHARED_DIR="experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines"
MOTIONS=(
    walk1_subject1
)
MOTION_ROOT="${OUTPUT_ROOT}/motions/${MOTION_NAME:-_all}"
RAW_DEMO_ROOT="${OUTPUT_ROOT}/demonstrations/paired_raw"
RAW_DEMOS="${RAW_DEMO_ROOT}/rollout_training_samples"
ROOT_DEMOS="${MOTION_ROOT}/demonstrations/root_qpos"
LATENT_DEMOS="${MOTION_ROOT}/demonstrations/latent_skill"
DEMO_AUDIT="${MOTION_ROOT}/demonstrations/paired_demonstration_audit.json"
POINT_ROOT="${MOTION_ROOT}/capacity/${MODEL_SIZE}/seed${SEED}/matched"
QUAL_ROOT="${OUTPUT_ROOT}/qualification"

case "${DRY_RUN}" in
    1|true|TRUE|yes|YES) DRY_RUN=1 ;;
    0|false|FALSE|no|NO) DRY_RUN=0 ;;
    *) echo "[ERROR] DRY_RUN must be boolean, got ${DRY_RUN}." >&2; exit 2 ;;
esac

has_stage() { [[ " ${STAGES} " == *" $1 "* ]]; }

if [[ "${DEMO_TRAJECTORIES_PER_MOTION}" -le 0 || "${COLLECT_ENVS}" -le 0 ]]; then
    echo "[ERROR] Oracle trajectory and collection-env counts must be positive." >&2
    exit 2
fi
if { has_stage train || has_stage eval; } && [[ -z "${MOTION_NAME}" ]]; then
    echo "[ERROR] train/eval stages require MOTION_NAME for the specialist cell." >&2
    exit 2
fi

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
verify_sha "${LOW_LEVEL_CHECKPOINT}" "${EXPECTED_LOW_LEVEL_SHA256}" tracker
verify_sha "${SKILL_CHECKPOINT}" "${EXPECTED_SKILL_SHA256}" encoder

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
QUAL_FULL_ARGS=()
if [[ "${RENDER_VIDEO:-1}" == "1" && "${ASSERT_KITLESS:-0}" != "1" ]]; then
    QUAL_FULL_ARGS=(--video --video_length "${QUALIFY_STEPS}")
fi

route_args() {
    local route="$1" planner="$2"
    if [[ "${route}" == "root_qpos" ]]; then
        ROUTE_ARGS=(
            --packet_planner_checkpoint "${planner}"
            --packet_interface root_qpos
            --packet_source planner
            agent.ipmd.command_source=hl_skill
        )
    else
        ROUTE_ARGS=(
            --planner_checkpoint "${planner}"
            agent.ipmd.command_source=skill_commander
            "agent.ipmd.skill_commander_checkpoint_path=${planner}"
            agent.ipmd.skill_commander_use_achieved_state=true
            agent.ipmd.skill_commander_flow_num_inference_steps="${FLOW_STEPS}"
            agent.ipmd.skill_commander_flow_inference_noise_std=0.0
        )
    fi
}

run_eval() {
    local route="$1" planner="$2" output="$3" label="$4" pass="$5"
    route_args "${route}" "${planner}"
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
        --output_dir "${output}" --label "${label}" --num_envs "${EVAL_ENVS}" \
        --max_steps "${EVAL_STEPS}" --seed "${SEED}" --metric_interval 10 \
        --motion_name "${MOTION_NAME}" --allow_random_reset --keep_time_out \
        --disable_reward_clipping --flow_num_inference_steps "${FLOW_STEPS}" \
        --flow_inference_noise_std 0.0 --kit_args=--/app/extensions/fsWatcherEnabled=false \
        "${pass_args[@]}" "${ROUTE_ARGS[@]}" "${LATENT_CFG[@]}" "${ENV_CFG[@]}"
}


run_packet_pin() {
    local planner="$1" output="${POINT_ROOT}/root_qpos/packet_encoder_pin"
    run_if_missing "${output}/summary.json" \
        "${ISAAC_PY_ARR[@]}" scripts/rlopt/eval_skill_commander_closed_loop.py \
        "${KITLESS_ARGS[@]}" --headless --device "${DEVICE}" --task "${TASK}" \
        --algorithm IPMD --checkpoint "${LOW_LEVEL_CHECKPOINT}" \
        --skill_checkpoint "${SKILL_CHECKPOINT}" --state_history_steps 9 \
        --packet_planner_checkpoint "${planner}" --packet_interface root_qpos \
        --packet_source expert --output_dir "${output}" \
        --label enc380_root_qpos_packet_encoder_pin --num_envs 4 --max_steps 30 \
        --seed "${SEED}" --metric_interval 1 --motion_name "${MOTION_NAME}" \
        --allow_random_reset --keep_time_out --extend_episode_length_for_max_steps \
        --disable_reward_clipping --flow_num_inference_steps "${FLOW_STEPS}" \
        --flow_inference_noise_std 0.0 \
        --kit_args=--/app/extensions/fsWatcherEnabled=false \
        agent.ipmd.command_source=hl_skill "${LATENT_CFG[@]}" "${ENV_CFG[@]}"
    run_if_missing "${output}/audit.json" \
        "${PLAIN_PY_ARR[@]}" "${IFACE_DIR}/audit_packet_encoder_pin.py" \
        --summary "${output}/summary.json" --planner_checkpoint "${planner}" \
        --low_level_checkpoint "${LOW_LEVEL_CHECKPOINT}" \
        --skill_checkpoint "${SKILL_CHECKPOINT}" --require_pass \
        --output_json "${output}/audit.json"
}

collect_oracle_trajectories() {
    local output="$1"
    run_if_missing "${output}/summary.json" \
        "${ISAAC_PY_ARR[@]}" scripts/rlopt/eval_skill_commander_closed_loop.py \
        "${KITLESS_ARGS[@]}" --headless --device "${DEVICE}" --task "${TASK}" \
        --algorithm IPMD --checkpoint "${LOW_LEVEL_CHECKPOINT}" \
        --skill_checkpoint "${SKILL_CHECKPOINT}" --state_history_steps 9 \
        --output_dir "${output}" --label enc380_paired_oracle_trajectory_collection \
        --num_envs "${COLLECT_ENVS}" \
        --max_steps "${COLLECT_STEPS}" --seed "${SEED}" --metric_interval 10 \
        --motion_names "${MOTIONS[@]}" --balanced_motion_names "${MOTIONS[@]}" \
        --balanced_trajectories_per_motion "${DEMO_TRAJECTORIES_PER_MOTION}" \
        --save_rollout_training_samples --sample_rows_per_file 1000 \
        --continue_after_reset --allow_random_reset \
        --keep_time_out --disable_tracking_terminations --disable_reward_clipping \
        --flow_num_inference_steps "${FLOW_STEPS}" --flow_inference_noise_std 0.0 \
        --kit_args=--/app/extensions/fsWatcherEnabled=false \
        --sample_target_interface root_qpos agent.ipmd.command_source=hl_skill \
        "${LATENT_CFG[@]}" "${ENV_CFG[@]}"
    if [[ "${DRY_RUN}" != "1" ]]; then
        grep -q '"stop_reason": "balanced_trajectories_complete"' \
            "${output}/summary.json" || {
            echo "[ERROR] oracle collection did not reach its trajectory budget." >&2
            exit 2
        }
    fi
}

train_planner() {
    local route="$1" samples="$2" output="$3"
    run_if_missing "${output}/checkpoints/best.pt" \
        "${PLAIN_PY_ARR[@]}" "${SHARED_DIR}/train_chunked_transformer_planner.py" \
        --samples_dir "${samples}" --output_dir "${output}" --interface "${route}" \
        --planner_family flow --state_key planner_state --training_stage oracle \
        --device "${DEVICE}" --seed "${SEED}" --model_size "${MODEL_SIZE}" --batch_size "${BATCH_SIZE}" \
        --micro_batch_size "${MICRO_BATCH_SIZE}" --num_updates "${TRAIN_UPDATES}" \
        --max_samples 0 --lr 0.0001 --weight_decay 0.0001 \
        --flow_num_inference_steps "${FLOW_STEPS}" --endpoint_num_inference_steps 4 \
        --flow_inference_noise_std 0.0
}

require_qualification() {
    local audit="${QUAL_ROOT}/latent_qualification_audit.json"
    require_file "${QUAL_ROOT}/tracker_completion.json"
    require_file "${QUAL_ROOT}/motion_selection.json"
    require_file "${QUAL_ROOT}/skill_binding.json"
    require_file "${audit}"
    [[ "${DRY_RUN}" == "1" ]] && return 0
    if [[ "${RENDER_VIDEO:-1}" == "1" ]]; then
        local video_dir="${QUAL_ROOT}/full_horizon_oracle/videos/play"
        local video_path
        video_path="$(find "${video_dir}" -type f -name '*.mp4' -print -quit 2>/dev/null || true)"
        [[ -n "${video_path}" ]] || {
            echo "[ERROR] retained full-horizon qualification video missing: ${video_dir}" >&2
            exit 2
        }
        echo "[PASS] retained qualification video: $(realpath "${video_path}")"
    fi
    grep -q '"protocol_passed": true' "${audit}" && \
        grep -q '"oracle_passed": true' "${audit}" || {
        echo "[ERROR] enc380 qualification did not pass: ${audit}" >&2
        exit 2
    }
}

require_demonstrations() {
    require_file "${DEMO_AUDIT}"
    [[ "${DRY_RUN}" == "1" ]] && return 0
    grep -q '"passed": true' "${DEMO_AUDIT}" || {
        echo "[ERROR] paired demonstrations did not pass: ${DEMO_AUDIT}" >&2
        exit 2
    }
}

if has_stage qualify; then
    run_if_missing "${QUAL_ROOT}/tracker_completion.json" \
        "${PLAIN_PY_ARR[@]}" "${IFACE_DIR}/audit_enc380_tracker_completion.py" \
        --completion_record "${TRACKER_COMPLETION_RECORD}" \
        --low_level_checkpoint "${LOW_LEVEL_CHECKPOINT}" \
        --skill_checkpoint "${SKILL_CHECKPOINT}" \
        --output_json "${QUAL_ROOT}/tracker_completion.json"
    run_if_missing "${QUAL_ROOT}/motion_selection.json" \
        "${PLAIN_PY_ARR[@]}" "${IFACE_DIR}/audit_enc380_motion_selection.py" \
        --manifest "${MANIFEST}" \
        --output_json "${QUAL_ROOT}/motion_selection.json"
    run_if_missing "${QUAL_ROOT}/skill_binding.json" \
        "${PLAIN_PY_ARR[@]}" "${SHARED_DIR}/validate_latent_skill_checkpoint_binding.py" \
        --low_level_checkpoint "${LOW_LEVEL_CHECKPOINT}" \
        --skill_checkpoint "${SKILL_CHECKPOINT}" \
        --output_json "${QUAL_ROOT}/skill_binding.json"
    run_if_missing "${QUAL_ROOT}/strict_oracle/summary.json" \
        "${ISAAC_PY_ARR[@]}" scripts/rlopt/eval_skill_commander_closed_loop.py \
        "${KITLESS_ARGS[@]}" --headless --device "${DEVICE}" --task "${TASK}" \
        --algorithm IPMD --checkpoint "${LOW_LEVEL_CHECKPOINT}" \
        --skill_checkpoint "${SKILL_CHECKPOINT}" --state_history_steps 9 \
        --output_dir "${QUAL_ROOT}/strict_oracle" --label enc380_strict_oracle \
        --num_envs "${QUALIFY_ENVS}" --max_steps "${QUALIFY_STEPS}" --seed "${SEED}" \
        --motion_names "${MOTIONS[@]}" \
        --metric_interval 1 --keep_time_out --allow_random_reset \
        --keep_early_terminations --disable_reward_clipping \
        --kit_args=--/app/extensions/fsWatcherEnabled=false \
        agent.ipmd.command_source=hl_skill "${LATENT_CFG[@]}" "${ENV_CFG[@]}"
    run_if_missing "${QUAL_ROOT}/full_horizon_oracle/summary.json" \
        "${ISAAC_PY_ARR[@]}" scripts/rlopt/eval_skill_commander_closed_loop.py \
        "${KITLESS_ARGS[@]}" --headless --device "${DEVICE}" --task "${TASK}" \
        --algorithm IPMD --checkpoint "${LOW_LEVEL_CHECKPOINT}" \
        --skill_checkpoint "${SKILL_CHECKPOINT}" --state_history_steps 9 \
        --output_dir "${QUAL_ROOT}/full_horizon_oracle" --label enc380_full_horizon_oracle \
        --num_envs "${QUALIFY_ENVS}" --max_steps "${QUALIFY_STEPS}" --seed "${SEED}" \
        --motion_names "${MOTIONS[@]}" \
        --metric_interval 10 --keep_time_out --allow_random_reset \
        --extend_episode_length_for_max_steps \
        --deterministic_tracking --disable_reward_clipping \
        --kit_args=--/app/extensions/fsWatcherEnabled=false \
        "${QUAL_FULL_ARGS[@]}" agent.ipmd.command_source=hl_skill "${LATENT_CFG[@]}" "${ENV_CFG[@]}"
    run_if_missing "${QUAL_ROOT}/latent_qualification_audit.json" \
        "${PLAIN_PY_ARR[@]}" "${SHARED_DIR}/audit_diffsr_latent_qualification.py" \
        --summary "${QUAL_ROOT}/strict_oracle/summary.json" \
        --low_level_checkpoint "${LOW_LEVEL_CHECKPOINT}" \
        --skill_checkpoint "${SKILL_CHECKPOINT}" --manifest "${MANIFEST}" \
        --expected_dataset_path "${DATASET_PATH}" --expected_num_envs "${QUALIFY_ENVS}" \
        --expected_steps "${QUALIFY_STEPS}" --expected_seed "${SEED}" \
        --expected_motion_names "${MOTIONS[@]}" \
        --expected_random_reset_step_max 200 --no-expect_episode_length_extension \
        --expected_task "${TASK}" --expected_planner_target_dim 256 \
        --success_threshold "${MIN_ORACLE_SUCCESS}" --require_pass \
        --output_json "${QUAL_ROOT}/latent_qualification_audit.json"
    require_qualification
fi

if has_stage demo; then
    require_qualification
    collect_oracle_trajectories "${RAW_DEMO_ROOT}"
    materialize_args=(
        "${PLAIN_PY_ARR[@]}"
        "${IFACE_DIR}/materialize_paired_interface_samples.py"
        --samples_dir "${RAW_DEMOS}"
    )
    for motion in "${MOTIONS[@]}"; do
        motion_demo_root="${OUTPUT_ROOT}/motions/${motion}/demonstrations"
        materialize_args+=(
            --target "encoder_input_packet_target:${motion}=${motion_demo_root}/root_qpos"
            --target "latent_skill_target:${motion}=${motion_demo_root}/latent_skill"
        )
    done
    final_materialization_marker="${OUTPUT_ROOT}/motions/${MOTIONS[0]}/demonstrations/latent_skill/materialization_manifest.json"
    run_if_missing "${final_materialization_marker}" "${materialize_args[@]}"
    for motion in "${MOTIONS[@]}"; do
        motion_demo_root="${OUTPUT_ROOT}/motions/${motion}/demonstrations"
        run_if_missing "${motion_demo_root}/paired_demonstration_audit.json" \
            "${PLAIN_PY_ARR[@]}" "${IFACE_DIR}/audit_enc380_paired_demonstrations.py" \
            --root_qpos_dir "${motion_demo_root}/root_qpos" \
            --latent_skill_dir "${motion_demo_root}/latent_skill" \
            --expected_trajectories "${DEMO_TRAJECTORIES_PER_MOTION}" \
            --min_trajectories "${DEMO_TRAJECTORIES_PER_MOTION}" \
            --expected_motion "${motion}" \
            --output_json "${motion_demo_root}/paired_demonstration_audit.json"
    done
fi

for route in root_qpos latent_skill; do
    route_root="${POINT_ROOT}/${route}"
    demos="${MOTION_ROOT}/demonstrations/${route}"
    planner="${route_root}/${PLANNER_DIR_NAME}"
    if has_stage train; then
        require_qualification
        require_demonstrations
        train_planner "${route}" "${demos}" "${planner}"
    fi
    if has_stage eval; then
        require_demonstrations
        if [[ "${route}" == root_qpos ]]; then
            run_packet_pin "${planner}/checkpoints/best.pt"
        fi
        run_eval "${route}" "${planner}/checkpoints/best.pt" \
            "${route_root}/eval_oracle_trained_survival" \
            "enc380_${route}_oracle_trained_survival" survival
        run_eval "${route}" "${planner}/checkpoints/best.pt" \
            "${route_root}/eval_oracle_trained_full_horizon" \
            "enc380_${route}_oracle_trained_full_horizon" full
    fi
done

if has_stage aggregate; then
    run_if_missing "${OUTPUT_ROOT}/aggregate/results.json" \
        "${PLAIN_PY_ARR[@]}" "${IFACE_DIR}/aggregate_enc380_route_comparison.py" \
        --study_root "${OUTPUT_ROOT}" --output_dir "${OUTPUT_ROOT}/aggregate"
fi

echo "[PASS] enc380 route comparison stages complete: ${STAGES}"
