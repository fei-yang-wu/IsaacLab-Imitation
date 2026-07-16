#!/usr/bin/env bash
set -euo pipefail

# Strict paper-facing BONES-SEED low-level gate. This evaluates existing final
# checkpoints only; it never trains or submits a high-level planner.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

: "${VANILLA_TRACKER_CHECKPOINT:?Set VANILLA_TRACKER_CHECKPOINT}"
: "${LATENT_LOW_LEVEL_CHECKPOINT:?Set LATENT_LOW_LEVEL_CHECKPOINT}"
: "${LATENT_SKILL_CHECKPOINT:?Set LATENT_SKILL_CHECKPOINT}"
: "${MANIFEST:?Set MANIFEST to the fresh BONES-SEED manifest}"
: "${VANILLA_DATASET_PATH:?Set VANILLA_DATASET_PATH}"
: "${LATENT_DATASET_PATH:?Set LATENT_DATASET_PATH}"

OUTPUT_ROOT="${OUTPUT_ROOT:-logs/interface_baselines/bones_seed_100_low_level_qualification_$(date +%Y%m%d_%H%M%S)}"
NUM_ENVS="${NUM_ENVS:-100}"
EVAL_STEPS="${EVAL_STEPS:-1000}"
SEED="${SEED:-0}"
MIN_ORACLE_SUCCESS="${MIN_ORACLE_SUCCESS:-0.8}"
EQUIVALENCE_NUM_ENVS="${EQUIVALENCE_NUM_ENVS:-2}"
EQUIVALENCE_STEPS="${EQUIVALENCE_STEPS:-20}"
DRY_RUN="${DRY_RUN:-0}"

read -r -a PYTHON_CMD <<< "${INTERFACE_BASELINE_PYTHON_CMD:-pixi run python}"
read -r -a ISAACLAB_PYTHON_CMD <<< "${INTERFACE_BASELINE_ISAACLAB_PYTHON_CMD:-pixi run -e isaaclab python}"

if [[ "${NUM_ENVS}" != "100" || "${EVAL_STEPS}" != "1000" || "${SEED}" != "0" ]]; then
    echo "[ERROR] Final BONES qualification requires NUM_ENVS=100, EVAL_STEPS=1000, and SEED=0." >&2
    exit 2
fi
if [[ "${EQUIVALENCE_NUM_ENVS}" != "2" || "${EQUIVALENCE_STEPS}" != "20" ]]; then
    echo "[ERROR] Equivalence qualification requires 2 environments and 20 steps." >&2
    exit 2
fi
if [[ "${MIN_ORACLE_SUCCESS}" != "0.8" ]]; then
    echo "[ERROR] Final BONES qualification requires MIN_ORACLE_SUCCESS=0.8." >&2
    exit 2
fi
if [[ -e "${OUTPUT_ROOT}" && "${DRY_RUN}" != "1" ]]; then
    echo "[ERROR] OUTPUT_ROOT already exists: ${OUTPUT_ROOT}" >&2
    exit 2
fi

for path in \
    "${VANILLA_TRACKER_CHECKPOINT}" \
    "${LATENT_LOW_LEVEL_CHECKPOINT}" \
    "${LATENT_SKILL_CHECKPOINT}" \
    "${MANIFEST}"; do
    if [[ ! -f "${path}" && "${DRY_RUN}" != "1" ]]; then
        echo "[ERROR] Required file does not exist: ${path}" >&2
        exit 2
    fi
done
for path in "${VANILLA_DATASET_PATH}" "${LATENT_DATASET_PATH}"; do
    if [[ ! -d "${path}" && "${DRY_RUN}" != "1" ]]; then
        echo "[ERROR] Required dataset cache does not exist: ${path}" >&2
        exit 2
    fi
done

run_cmd() {
    printf '[CMD]'
    printf ' %q' "$@"
    printf '\n'
    if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" ]]; then
        return 0
    fi
    "$@"
}

if [[ "${DRY_RUN}" != "1" && "${DRY_RUN}" != "true" ]]; then
    mkdir -p \
        "${OUTPUT_ROOT}/protocol_checks" \
        "${OUTPUT_ROOT}/vanilla_direct" \
        "${OUTPUT_ROOT}/latent_oracle"
fi

# Fail before launching Isaac if the supplied skill encoder is not the exact
# encoder embedded in the latent low-level checkpoint.
run_cmd "${PYTHON_CMD[@]}" \
    experiments/interface_baselines/validate_latent_skill_checkpoint_binding.py \
    --low_level_checkpoint "${LATENT_LOW_LEVEL_CHECKPOINT}" \
    --skill_checkpoint "${LATENT_SKILL_CHECKPOINT}" \
    --output_json "${OUTPUT_ROOT}/protocol_checks/latent_skill_binding.json"

run_cmd "${PYTHON_CMD[@]}" scripts/audit_bones_seed_phase5.py \
    --manifest "${MANIFEST}" \
    --report "${OUTPUT_ROOT}/protocol_checks/bones_seed_preflight.json" \
    --require-temporal-events \
    --require-body-names

# Direct 50 Hz vanilla ceiling and strict tracker qualification.
run_cmd "${ISAACLAB_PYTHON_CMD[@]}" \
    experiments/command_space_ablation/evaluate_checkpoint.py \
    --headless \
    --task Isaac-Imitation-G1-v0 \
    --algorithm IPMD \
    --checkpoint "${VANILLA_TRACKER_CHECKPOINT}" \
    --policy_only_checkpoint \
    --command_space single_frame_full_body \
    --low_level_command_mode native \
    --command_observation_source reference \
    --command_past_steps 0 \
    --command_future_steps 0 \
    --planner_update_interval 1 \
    --motion_manifest "${MANIFEST}" \
    --dataset_path "${VANILLA_DATASET_PATH}" \
    --num_envs "${NUM_ENVS}" \
    --steps "${EVAL_STEPS}" \
    --seed "${SEED}" \
    --reset_schedule sequential \
    --reference_start_frame 0 \
    --label bones_seed_100_direct_vanilla_50hz \
    --output_json "${OUTPUT_ROOT}/vanilla_direct/summary.json" \
    --output_csv "${OUTPUT_ROOT}/vanilla_direct/summary.csv" \
    --kit_args=--/app/extensions/fsWatcherEnabled=false \
    env.command_hold_steps=0

run_cmd "${PYTHON_CMD[@]}" \
    experiments/interface_baselines/audit_vanilla_tracker_qualification.py \
    --summary "${OUTPUT_ROOT}/vanilla_direct/summary.json" \
    --checkpoint "${VANILLA_TRACKER_CHECKPOINT}" \
    --manifest "${MANIFEST}" \
    --expected_dataset_path "${VANILLA_DATASET_PATH}" \
    --expected_num_envs "${NUM_ENVS}" \
    --expected_steps "${EVAL_STEPS}" \
    --expected_seed "${SEED}" \
    --success_threshold "${MIN_ORACLE_SUCCESS}" \
    --require_pass \
    --output_json "${OUTPUT_ROOT}/vanilla_qualification_audit.json"

# The exact vanilla packet must reproduce direct tracker inputs and actions for
# every phase, including asynchronous per-environment renewal.
run_cmd "${ISAACLAB_PYTHON_CMD[@]}" \
    experiments/command_space_ablation/evaluate_checkpoint.py \
    --headless \
    --task Isaac-Imitation-G1-v0 \
    --algorithm IPMD \
    --checkpoint "${VANILLA_TRACKER_CHECKPOINT}" \
    --low_level_command_mode streamed_vanilla \
    --command_space full_body_trajectory \
    --command_observation_source reference \
    --command_past_steps 0 \
    --command_future_steps 9 \
    --planner_update_interval 10 \
    --motion_manifest "${MANIFEST}" \
    --dataset_path "${VANILLA_DATASET_PATH}" \
    --num_envs "${EQUIVALENCE_NUM_ENVS}" \
    --steps "${EQUIVALENCE_STEPS}" \
    --seed "${SEED}" \
    --reset_schedule sequential \
    --reference_start_frame 0 \
    --certify_streamed_vanilla_equivalence \
    --equivalence_steps "${EQUIVALENCE_STEPS}" \
    --label bones_seed_100_streamed_vanilla_equivalence \
    --output_json "${OUTPUT_ROOT}/streamed_vanilla_equivalence.json" \
    --kit_args=--/app/extensions/fsWatcherEnabled=false

# DiffSR oracle under the same strict 100-motion, frame-0 evaluation protocol.
run_cmd "${ISAACLAB_PYTHON_CMD[@]}" scripts/rlopt/eval_skill_commander_closed_loop.py \
    --headless \
    --device cuda:0 \
    --task Isaac-Imitation-G1-Latent-v0 \
    --algorithm IPMD \
    --checkpoint "${LATENT_LOW_LEVEL_CHECKPOINT}" \
    --skill_checkpoint "${LATENT_SKILL_CHECKPOINT}" \
    --state_history_steps 9 \
    --output_dir "${OUTPUT_ROOT}/latent_oracle" \
    --label bones_seed_100_diffsr_latent_oracle \
    --num_envs "${NUM_ENVS}" \
    --max_steps "${EVAL_STEPS}" \
    --seed "${SEED}" \
    --metric_interval 1 \
    --keep_time_out \
    --extend_episode_length_for_max_steps \
    --keep_early_terminations \
    --disable_reward_clipping \
    --kit_args=--/app/extensions/fsWatcherEnabled=false \
    agent.ipmd.command_source=hl_skill \
    "agent.ipmd.hl_skill_checkpoint_path=${LATENT_SKILL_CHECKPOINT}" \
    agent.ipmd.hl_skill_finetune_enabled=false \
    "env.lafan1_manifest_path=${MANIFEST}" \
    "env.dataset_path=${LATENT_DATASET_PATH}" \
    env.refresh_zarr_dataset=false \
    env.reset_schedule=sequential \
    env.wrap_steps=false \
    env.observations.policy.enable_corruption=false \
    env.latent_command_dim=258 \
    agent.ipmd.latent_dim=258 \
    agent.ipmd.hl_skill_horizon_steps=10 \
    agent.ipmd.hl_skill_command_mode=z \
    agent.ipmd.latent_steps_min=10 \
    agent.ipmd.latent_steps_max=10 \
    agent.ipmd.latent_learning.command_phase_mode=sin_cos \
    agent.ipmd.latent_learning.code_latent_dim=256 \
    agent.ipmd.latent_learning.code_period=10 \
    agent.ipmd.reward_loss_coeff=0.0 \
    agent.ipmd.reward_l2_coeff=0.0 \
    agent.ipmd.reward_grad_penalty_coeff=0.0 \
    agent.ipmd.reward_logit_reg_coeff=0.0 \
    agent.ipmd.reward_param_weight_decay_coeff=0.0

run_cmd "${PYTHON_CMD[@]}" \
    experiments/interface_baselines/audit_diffsr_latent_qualification.py \
    --summary "${OUTPUT_ROOT}/latent_oracle/summary.json" \
    --low_level_checkpoint "${LATENT_LOW_LEVEL_CHECKPOINT}" \
    --skill_checkpoint "${LATENT_SKILL_CHECKPOINT}" \
    --manifest "${MANIFEST}" \
    --expected_dataset_path "${LATENT_DATASET_PATH}" \
    --expected_num_envs "${NUM_ENVS}" \
    --expected_steps "${EVAL_STEPS}" \
    --expected_seed "${SEED}" \
    --success_threshold "${MIN_ORACLE_SUCCESS}" \
    --require_pass \
    --output_json "${OUTPUT_ROOT}/latent_qualification_audit.json"

export \
    VANILLA_TRACKER_CHECKPOINT \
    LATENT_LOW_LEVEL_CHECKPOINT \
    LATENT_SKILL_CHECKPOINT \
    MANIFEST \
    VANILLA_DATASET_PATH \
    LATENT_DATASET_PATH \
    OUTPUT_ROOT \
    NUM_ENVS \
    EVAL_STEPS \
    SEED \
    MIN_ORACLE_SUCCESS
run_cmd "${PYTHON_CMD[@]}" experiments/interface_baselines/write_interface_run_provenance.py \
    --label bones-seed-100-low-level-qualification \
    --output_json "${OUTPUT_ROOT}/run_provenance.json" \
    --result_root "${OUTPUT_ROOT}" \
    --env_key VANILLA_TRACKER_CHECKPOINT \
    --env_key LATENT_LOW_LEVEL_CHECKPOINT \
    --env_key LATENT_SKILL_CHECKPOINT \
    --env_key MANIFEST \
    --env_key VANILLA_DATASET_PATH \
    --env_key LATENT_DATASET_PATH \
    --env_key MIN_ORACLE_SUCCESS \
    --note "strict 100-motion frame-0 oracle evaluation" \
    --note "planner submission remains blocked unless both audits and equivalence pass"

if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" ]]; then
    echo "[INFO] BONES-SEED low-level qualification dry run rendered."
else
    echo "[PASS] BONES-SEED low-level qualification passed: ${OUTPUT_ROOT}"
fi
