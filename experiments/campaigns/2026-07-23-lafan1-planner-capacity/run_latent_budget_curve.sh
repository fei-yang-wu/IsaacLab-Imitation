#!/usr/bin/env bash
set -euo pipefail

# Latent (DiffSR z256+phase) training-budget curve -- the matched partner of
# run_fb670_budget_curve.sh. Same motion, same demonstration budget, same
# planner sizes / updates / batch / holdout, same milestone cadence, and the
# same rigorous frame-0 + domain-randomization + base_too_low-only evaluation.
# Only the command interface and its frozen low-level tracker differ, so the
# two curves are directly comparable at every budget point.
#
# The latent arm uses a different executable on both ends: demonstrations and
# closed-loop evaluation run through scripts/rlopt/eval_skill_commander_closed_loop.py
# (command_source=hl_skill for the oracle, skill_commander for the planner),
# because the 258-value latent command is produced by the frozen encoder rather
# than assembled from an explicit packet.

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

STAGES="${STAGES:-collect train eval aggregate}"
STUDY_ROOT="${STUDY_ROOT:-logs/interface_baselines/lafan1_latent_budget_curve_20260730}"
MOTION_NAME="${MOTION_NAME:-walk1_subject1}"
MANIFEST="${MANIFEST:-data/lafan1/manifests/g1_lafan1_manifest.json}"
LATENT_DATASET_PATH="${LATENT_DATASET_PATH:-data/lafan1/zarr/latent_walk1_subject1_corrected_8e95d557}"
LATENT_TASK="${LATENT_TASK:-Isaac-Imitation-G1-Latent-Strict-v0}"
DEVICE="${DEVICE:-cuda:0}"

LATENT_LOW_LEVEL_CHECKPOINT="${LATENT_LOW_LEVEL_CHECKPOINT:-logs/downloaded_checkpoints/lafan1_latent_deterministic_5b_seed0/model_step_4525129728.pt}"
LATENT_SKILL_CHECKPOINT="${LATENT_SKILL_CHECKPOINT:-logs/downloaded_checkpoints/lafan1_latent_deterministic_5b_seed0/skill_encoder/latest.pt}"
EXPECTED_LATENT_SHA256="${EXPECTED_LATENT_SHA256:-785f5327f2356f4a301ac39fc435b78379e9c5a73293c450deb483dd7c188f7c}"
EXPECTED_SKILL_SHA256="${EXPECTED_SKILL_SHA256:-}"

# Demonstration collection -- mirrors the FB arm's budget exactly.
DEMO_ROWS="${DEMO_ROWS:-5000}"
DEMO_ENVS="${DEMO_ENVS:-100}"
DEMO_STEPS="${DEMO_STEPS:-15000}"
DEMO_EPISODE_LENGTH_S="${DEMO_EPISODE_LENGTH_S:-10.0}"
DEMO_SEED="${DEMO_SEED:-0}"

# Planner training -- identical contract to the FB arm.
MODEL_SIZE="${MODEL_SIZE:-medium}"
PLANNER_SEED="${PLANNER_SEED:-0}"
TRAIN_UPDATES="${TRAIN_UPDATES:-30000}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
MILESTONE_INTERVAL="${MILESTONE_INTERVAL:-1000}"
case "${MODEL_SIZE}" in
    tiny)   default_micro=1024 ;;
    small)  default_micro=512 ;;
    medium) default_micro=256 ;;
    large)  default_micro=128 ;;
    *) echo "[ERROR] Unknown MODEL_SIZE=${MODEL_SIZE}." >&2; exit 2 ;;
esac
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-${default_micro}}"

EVAL_ENVS="${EVAL_ENVS:-4096}"
EVAL_STEPS="${EVAL_STEPS:-500}"
EVAL_SEED="${EVAL_SEED:-0}"
EVAL_STRIDE="${EVAL_STRIDE:-2000}"
FALL_HEIGHT_M="${FALL_HEIGHT_M:-0.4}"
ASSERT_KITLESS="${ASSERT_KITLESS:-0}"

MODEL_SIZES_FOR_AGGREGATE="${MODEL_SIZES_FOR_AGGREGATE:-medium large}"

: "${ISAAC_PY:=pixi run -e isaaclab python}"
: "${PLAIN_PY:=pixi run python}"
read -r -a ISAAC_PY_ARR <<<"${ISAAC_PY}"
read -r -a PLAIN_PY_ARR <<<"${PLAIN_PY}"

SHARED_DIR="experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines"
DEMOS_DIR="${STUDY_ROOT}/oracle_demonstrations"
SAMPLES_DIR="${DEMOS_DIR}/rollout_training_samples"
RUN_TAG="planner_u${TRAIN_UPDATES}_b${BATCH_SIZE}"

KITLESS=()
[[ "${ASSERT_KITLESS}" == "1" ]] && KITLESS=(--assert-kitless)
KIT_QUIET=(--kit_args=--/app/extensions/fsWatcherEnabled=false)
NEWTON_ARGS=(
    physics=newton_mjwarp
    env.sim.physics.solver_cfg.njmax=320
    env.sim.physics.solver_cfg.nconmax=40
)
# Latent command contract: the deterministic DiffSR encoder is z256 + sin/cos
# phase = 258 values, held 10 control steps.
LATENT_CMD=(
    env.latent_command_dim=258 agent.ipmd.latent_dim=258
    agent.ipmd.hl_skill_horizon_steps=10 agent.ipmd.hl_skill_command_mode=z
    agent.ipmd.latent_steps_min=10 agent.ipmd.latent_steps_max=10
    agent.ipmd.latent_learning.command_phase_mode=sin_cos
    agent.ipmd.latent_learning.code_latent_dim=256
    agent.ipmd.latent_learning.code_period=10
)
LATENT_REWARD_ZEROS=(
    agent.ipmd.reward_loss_coeff=0.0 agent.ipmd.reward_l2_coeff=0.0
    agent.ipmd.reward_grad_penalty_coeff=0.0 agent.ipmd.reward_logit_reg_coeff=0.0
    agent.ipmd.reward_param_weight_decay_coeff=0.0
)
# Domain-randomization events are deliberately NOT overridden: the evaluation
# protocol keeps training-time physical randomization and interval pushes.
ENV_COMMON=(
    agent.logger.backend=
    env.refresh_zarr_dataset=false
    env.wrap_steps=false
    env.random_reset_full_trajectory=false
    env.reset_schedule=sequential
    env.observations.policy.enable_corruption=false
)

LATENT_COMMON=(
    "${ISAAC_PY_ARR[@]}" scripts/rlopt/eval_skill_commander_closed_loop.py
    "${KITLESS[@]}"
    --headless --device "${DEVICE}"
    --task "${LATENT_TASK}" --algorithm IPMD
    --checkpoint "${LATENT_LOW_LEVEL_CHECKPOINT}"
    --skill_checkpoint "${LATENT_SKILL_CHECKPOINT}"
    --motion_name "${MOTION_NAME}" --state_history_steps 9 --metric_interval 10
    --flow_num_inference_steps 16 --flow_inference_noise_std 0.0
    "${KIT_QUIET[@]}"
    "agent.ipmd.hl_skill_checkpoint_path=${LATENT_SKILL_CHECKPOINT}"
    agent.ipmd.hl_skill_finetune_enabled=false
    "env.lafan1_manifest_path=${MANIFEST}"
    "env.dataset_path=${LATENT_DATASET_PATH}"
)

verify_checkpoints() {
    local path sha
    for path in "${LATENT_LOW_LEVEL_CHECKPOINT}" "${LATENT_SKILL_CHECKPOINT}"; do
        [[ -f "${path}" ]] || {
            echo "[ERROR] Missing latent checkpoint: ${path}" >&2
            exit 2
        }
    done
    sha="$(sha256sum "${LATENT_LOW_LEVEL_CHECKPOINT}" | awk '{print $1}')"
    [[ "${sha}" == "${EXPECTED_LATENT_SHA256}" ]] || {
        echo "[ERROR] Latent tracker SHA mismatch: ${sha} != ${EXPECTED_LATENT_SHA256}" >&2
        exit 2
    }
    if [[ -n "${EXPECTED_SKILL_SHA256}" ]]; then
        sha="$(sha256sum "${LATENT_SKILL_CHECKPOINT}" | awk '{print $1}')"
        [[ "${sha}" == "${EXPECTED_SKILL_SHA256}" ]] || {
            echo "[ERROR] Skill encoder SHA mismatch: ${sha} != ${EXPECTED_SKILL_SHA256}" >&2
            exit 2
        }
    fi
    echo "[PASS] Latent tracker + encoder verified."
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
        echo "[ERROR] Stage completed without producing ${marker}." >&2
        exit 2
    }
}

stage_collect() {
    verify_checkpoints
    # --allow_random_reset is mandatory: without it the script forces
    # random_reset_step_min/max back to 0/0 AFTER Hydra applies them, so the
    # latent demonstrations would all start at frame 0 while the FB arm's start
    # uniformly in 0-200 -- an uncontrolled asymmetry (audit 2026-07-28).
    run_if_missing "${SAMPLES_DIR}/sample_step_000000.pt" \
        "${LATENT_COMMON[@]}" \
        --seed "${DEMO_SEED}" \
        --num_envs "${DEMO_ENVS}" --max_steps "${DEMO_STEPS}" \
        --disable_tracking_terminations \
        --save_rollout_training_samples --continue_after_reset \
        --allow_random_reset \
        --balanced_rows_per_motion "${DEMO_ROWS}" \
        --balanced_motion_names "${MOTION_NAME}" \
        --sample_rows_per_file "${DEMO_ROWS}" \
        --output_dir "${DEMOS_DIR}" --label latent_oracle_demonstrations \
        agent.ipmd.command_source=hl_skill \
        env.random_reset_step_min=0 env.random_reset_step_max=200 \
        "env.episode_length_s=${DEMO_EPISODE_LENGTH_S}" \
        "${LATENT_CMD[@]}" "${LATENT_REWARD_ZEROS[@]}" \
        "${ENV_COMMON[@]}" "${NEWTON_ARGS[@]}"
    sha256sum "${SAMPLES_DIR}/sample_step_000000.pt" \
        > "${DEMOS_DIR}/sample_sha256.txt"
    echo "[PASS] Demonstration rows: $(cat "${DEMOS_DIR}/sample_sha256.txt")"
}

stage_train() {
    local out="${STUDY_ROOT}/${MODEL_SIZE}/seed${PLANNER_SEED}/${RUN_TAG}"
    [[ -f "${SAMPLES_DIR}/sample_step_000000.pt" ]] || {
        echo "[ERROR] Missing demonstration samples: ${SAMPLES_DIR}" >&2
        exit 2
    }
    run_if_missing "${out}/checkpoints/best.pt" \
        "${PLAIN_PY_ARR[@]}" -m imitation_experiments.planner.train_chunked_transformer_planner \
        --samples_dir "${SAMPLES_DIR}" \
        --output_dir "${out}" \
        --interface latent_skill \
        --planner_family flow \
        --state_key planner_state \
        --training_stage oracle \
        --device "${DEVICE}" \
        --seed "${PLANNER_SEED}" \
        --model_size "${MODEL_SIZE}" \
        --batch_size "${BATCH_SIZE}" \
        --micro_batch_size "${MICRO_BATCH_SIZE}" \
        --num_updates "${TRAIN_UPDATES}" \
        --milestone_interval "${MILESTONE_INTERVAL}" \
        --max_samples 0 \
        --lr 0.0001 --weight_decay 0.0001 \
        --flow_num_inference_steps 16 \
        --endpoint_num_inference_steps 4 \
        --flow_inference_noise_std 0.0
    local expected=$(( TRAIN_UPDATES / MILESTONE_INTERVAL ))
    local actual
    actual="$(find "${out}/checkpoints" -name 'update_*.pt' | wc -l)"
    [[ "${actual}" -eq "${expected}" ]] || {
        echo "[ERROR] Expected ${expected} milestone snapshots, found ${actual}." >&2
        exit 2
    }
    echo "[PASS] Trained ${MODEL_SIZE} with ${actual} milestone snapshots: ${out}"
}

eval_one() {
    local planner="$1" tag="$2"
    local out="${STUDY_ROOT}/${MODEL_SIZE}/seed${PLANNER_SEED}/eval_frame0_dr_baseonly_${EVAL_ENVS}env_seed${EVAL_SEED}/${tag}"
    [[ -f "${planner}" ]] || {
        echo "[ERROR] Missing planner checkpoint: ${planner}" >&2
        exit 2
    }
    run_if_missing "${out}/summary.json" \
        "${LATENT_COMMON[@]}" \
        --seed "${EVAL_SEED}" \
        --num_envs "${EVAL_ENVS}" --max_steps "${EVAL_STEPS}" \
        --output_dir "${out}" \
        --label "latent_curve_${MODEL_SIZE}_${tag}" \
        --planner_checkpoint "${planner}" \
        --base_only_termination --fall_height_m "${FALL_HEIGHT_M}" \
        --extend_episode_length_for_max_steps --disable_reward_clipping \
        agent.ipmd.command_source=skill_commander \
        "agent.ipmd.skill_commander_checkpoint_path=${planner}" \
        agent.ipmd.skill_commander_use_achieved_state=true \
        agent.ipmd.skill_commander_flow_num_inference_steps=16 \
        agent.ipmd.skill_commander_flow_inference_noise_std=0.0 \
        env.reference_start_frame=0 \
        env.random_reset_step_min=0 env.random_reset_step_max=0 \
        "${LATENT_CMD[@]}" "${LATENT_REWARD_ZEROS[@]}" \
        "${ENV_COMMON[@]}" "${NEWTON_ARGS[@]}"
}

stage_eval() {
    verify_checkpoints
    local ckpt_dir="${STUDY_ROOT}/${MODEL_SIZE}/seed${PLANNER_SEED}/${RUN_TAG}/checkpoints"
    [[ -d "${ckpt_dir}" ]] || {
        echo "[ERROR] Missing checkpoint dir: ${ckpt_dir}" >&2
        exit 2
    }
    if (( EVAL_STRIDE % MILESTONE_INTERVAL != 0 )); then
        echo "[ERROR] EVAL_STRIDE (${EVAL_STRIDE}) must be a multiple of MILESTONE_INTERVAL (${MILESTONE_INTERVAL})." >&2
        exit 2
    fi
    local update
    for (( update = EVAL_STRIDE; update <= TRAIN_UPDATES; update += EVAL_STRIDE )); do
        eval_one "$(printf '%s/update_%07d.pt' "${ckpt_dir}" "${update}")" \
            "$(printf 'update_%07d' "${update}")"
    done
    eval_one "${ckpt_dir}/best.pt" "best"
    echo "[PASS] Evaluated $(( TRAIN_UPDATES / EVAL_STRIDE )) milestones + best for ${MODEL_SIZE}."
}

stage_aggregate() {
    local sizes=()
    read -r -a sizes <<<"${MODEL_SIZES_FOR_AGGREGATE}"
    "${PLAIN_PY_ARR[@]}" \
        -m imitation_experiments.capacity.aggregate_planner_budget_curve \
        --study_root "${STUDY_ROOT}" \
        --interface latent_skill \
        --sizes "${sizes[@]}" \
        --seed "${PLANNER_SEED}" \
        --run_tag "${RUN_TAG}" \
        --eval_dir_name "eval_frame0_dr_baseonly_${EVAL_ENVS}env_seed${EVAL_SEED}" \
        --expected_num_envs "${EVAL_ENVS}" \
        --expected_fall_height_m "${FALL_HEIGHT_M}" \
        --output_dir "${STUDY_ROOT}/budget_curve_summary"
}

for stage in ${STAGES}; do
    echo "[STAGE] ${stage} (MODEL_SIZE=${MODEL_SIZE})"
    case "${stage}" in
        collect)   stage_collect ;;
        train)     stage_train ;;
        eval)      stage_eval ;;
        aggregate) stage_aggregate ;;
        *) echo "[ERROR] Unknown stage '${stage}'." >&2; exit 2 ;;
    esac
done
echo "[DONE] ${STAGES}"
