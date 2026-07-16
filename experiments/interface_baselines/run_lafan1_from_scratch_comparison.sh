#!/usr/bin/env bash
set -euo pipefail

# From-scratch LAFAN1 interface comparison.
#
# Redoes the PR#19 LAFAN1 motion-tracking comparison with all three low-level
# policies trained from scratch on the same full LAFAN1 manifest, at a matched
# ~5B-frame budget, so the oracle ceilings are comparable before any planner
# claim is made:
#
#   latent_skill         IPMD on Isaac-Imitation-G1-Latent-v0, conditioned on
#                        hl_skill z (h=HORIZON_STEPS), encoder+planner trained
#                        in the same pipeline.
#   ee_trajectory        IPMD on Isaac-Imitation-G1-v0, reference EE window.
#   full_body_trajectory IPMD on Isaac-Imitation-G1-v0, reference full-body
#                        window.
#
# Cluster stages (submit and wait between them):
#   STAGE=submit-train   Submit the three ~5B-frame training jobs.
#   STAGE=submit-eval    Submit the three per-motion evaluation jobs. Requires
#                        EE_TRAJECTORY_CHECKPOINT and
#                        FULL_BODY_TRAJECTORY_CHECKPOINT.
# Local stages (sequential code qualification only, resumable per sub-stage):
#   STAGE=local-train    Train latent stack, then EE, then full-body, capped at
#                        approximately 50M frames per controller.
#   STAGE=local-eval     Run the three per-motion evaluations.
#   STAGE=local-all      local-train -> local-eval -> summarize.
# Shared:
#   STAGE=summarize      Build the combined table (oracle rows + gate).
#
# Use DRY_RUN=1 to print commands without running/submitting anything.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ! REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null)"; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${REPO_ROOT}"

STAGE="${STAGE:-}"
DRY_RUN="${DRY_RUN:-0}"

# Matched experiment knobs. Everything that defines the interface comparison
# hangs off HORIZON_STEPS and SEED.
SEED="${SEED:-0}"
HORIZON_STEPS="${HORIZON_STEPS:-10}"
STATE_HISTORY_STEPS="${STATE_HISTORY_STEPS:-9}"
Z_DIM="${Z_DIM:-256}"
LOW_LEVEL_ALGO="${LOW_LEVEL_ALGO:-IPMD}"
DEVICE="${DEVICE:-cuda:0}"

# Training budget: ~5B env frames per low-level policy.
# 50865 iterations x 4096 envs x 24 steps/iter ~= 5.0B frames
# (5x the repo's usual 10173-iteration ~1B convention).
TRAIN_NUM_ENVS="${TRAIN_NUM_ENVS:-4096}"
TRAIN_MAX_ITERATIONS="${TRAIN_MAX_ITERATIONS:-50865}"
TRAIN_SAVE_INTERVAL="${TRAIN_SAVE_INTERVAL:-250000000}"
TRAIN_WALLTIME="${TRAIN_WALLTIME:-7-00:00:00}"

# Local runs only need to show that the intended training path works. Keep
# serious local checks at or below 50M total environment frames per controller;
# convergence and paper results belong on Skynet. RLOpt currently collects 24
# environment steps per iteration.
LOCAL_MAX_ENV_FRAMES="${LOCAL_MAX_ENV_FRAMES:-50000000}"
ROLLOUT_STEPS_PER_ITERATION="${ROLLOUT_STEPS_PER_ITERATION:-24}"
LOCAL_TRAIN_MAX_ITERATIONS_DEFAULT=$((LOCAL_MAX_ENV_FRAMES / (TRAIN_NUM_ENVS * ROLLOUT_STEPS_PER_ITERATION)))
LOCAL_TRAIN_MAX_ITERATIONS="${LOCAL_TRAIN_MAX_ITERATIONS:-${LOCAL_TRAIN_MAX_ITERATIONS_DEFAULT}}"
LOCAL_TRAIN_SAVE_INTERVAL="${LOCAL_TRAIN_SAVE_INTERVAL:-10000000}"

# Skill encoder / base planner budgets (latent pipeline defaults).
SKILL_UPDATES="${SKILL_UPDATES:-5000}"
PLANNER_UPDATES="${PLANNER_UPDATES:-5000}"

# Evaluation knobs. EVAL_NUM_ENVS>1 gives within-seed variance per motion.
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-4}"
RANKS="${RANKS:-all}"
LIMIT="${LIMIT:-0}"
# Match the latent planner's pretrain budget (PLANNER_UPDATES) instead of the
# strong runner's 2000 default, and let the chunk planners use every collected
# sample like the latent finetune does.
BASELINE_PRETRAIN_UPDATES="${BASELINE_PRETRAIN_UPDATES:-${PLANNER_UPDATES}}"
BASELINE_SAMPLE_BUDGETS="${BASELINE_SAMPLE_BUDGETS:-all}"

MANIFEST_PATH="${MANIFEST_PATH:-data/lafan1/manifests/g1_lafan1_manifest.json}"
DATASET_PATH="${DATASET_PATH:-data/lafan1/g1_hl_diffsr}"

EXPERIMENT_TAG="${EXPERIMENT_TAG:-lafan1_fromscratch_h${HORIZON_STEPS}_ipmd_5b_seed${SEED}}"
RUN_ROOT_BASE="${RUN_ROOT_BASE:-logs/lafan1_motion_tracking_evaluation/${EXPERIMENT_TAG}}"
PROJECT_NAME="${PROJECT_NAME:-G1-Imitation-LAFAN1-FromScratch}"

# Local execution helpers. The latent pipeline defaults TMPDIR under /data,
# which does not exist on the workstation.
LOCAL_TMPDIR="${LOCAL_TMPDIR:-/tmp/isaaclab_pipeline_${USER}}"
LOCAL_USD_CACHE="${LOCAL_USD_CACHE:-${HOME}/.cache/isaaclab_imitation/unitree_usd}"

# Per-stage toggles for selective (re)runs.
SUBMIT_LATENT="${SUBMIT_LATENT:-1}"
SUBMIT_EE="${SUBMIT_EE:-1}"
SUBMIT_FB="${SUBMIT_FB:-1}"
ALLOW_LEGACY_THREE_INTERFACE="${ALLOW_LEGACY_THREE_INTERFACE:-0}"

# submit-eval inputs: cluster-side checkpoints from the finished train jobs.
# Local stages record and discover these automatically.
EE_TRAJECTORY_CHECKPOINT="${EE_TRAJECTORY_CHECKPOINT:-}"
FULL_BODY_TRAJECTORY_CHECKPOINT="${FULL_BODY_TRAJECTORY_CHECKPOINT:-}"
# Optional explicit latent low-level checkpoint; defaults to the path recorded
# by the base pipeline in ${RUN_ROOT_BASE}/latent/base_pipeline.
LATENT_LOW_LEVEL_CHECKPOINT="${LATENT_LOW_LEVEL_CHECKPOINT:-}"
# Optional resume checkpoints for interrupted local EE/FB training runs.
EE_RESUME_CHECKPOINT="${EE_RESUME_CHECKPOINT:-}"
FB_RESUME_CHECKPOINT="${FB_RESUME_CHECKPOINT:-}"

LATENT_STAGE_FLAGS_OFF=(
    "RUN_ORACLE_RECON_EVAL=0"
    "RUN_BASE_PLANNER_PREDICT_EVAL=0"
    "RUN_ORACLE_LL_EVAL=0"
    "RUN_BASE_PLANNER_LL_EVAL=0"
    "RUN_PLANNER_FT_SAMPLE_COLLECTION=0"
    "RUN_PLANNER_ROLLOUT_FINETUNE=0"
    "RUN_FINETUNED_PLANNER_PREDICT_EVAL=0"
    "RUN_FINETUNED_PLANNER_LL_EVAL=0"
)

LATENT_STAGE_FLAGS_ON=(
    "RUN_ORACLE_RECON_EVAL=1"
    "RUN_BASE_PLANNER_PREDICT_EVAL=1"
    "RUN_ORACLE_LL_EVAL=1"
    "RUN_BASE_PLANNER_LL_EVAL=1"
    "RUN_PLANNER_FT_SAMPLE_COLLECTION=1"
    "RUN_PLANNER_ROLLOUT_FINETUNE=1"
    "RUN_FINETUNED_PLANNER_PREDICT_EVAL=1"
    "RUN_FINETUNED_PLANNER_LL_EVAL=1"
)

run_cmd() {
    printf '[CMD]'
    printf ' %q' "$@"
    printf '\n'
    if [[ "${DRY_RUN}" == "1" ]]; then
        return 0
    fi
    "$@"
}

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

usage() {
    sed -n '3,31p' "${BASH_SOURCE[0]}"
    exit 1
}

latest_checkpoint_from_log_dir() {
    local log_dir="$1"
    find "${log_dir}/models" -maxdepth 1 -type f -name 'model_step_*.pt' -printf '%f %p\n' \
        | sort -V \
        | tail -n 1 \
        | cut -d' ' -f2-
}

checkpoint_file_valid() {
    local record="$1"
    [[ -f "${record}" ]] || return 1
    local path
    path="$(<"${record}")"
    [[ -n "${path}" && -f "${path}" ]]
}

validate_local_budget() {
    local requested_frames=$((TRAIN_NUM_ENVS * ROLLOUT_STEPS_PER_ITERATION * LOCAL_TRAIN_MAX_ITERATIONS))
    if ((LOCAL_TRAIN_MAX_ITERATIONS < 1)); then
        echo "[ERROR] LOCAL_TRAIN_MAX_ITERATIONS must be at least 1." >&2
        exit 1
    fi
    if ((requested_frames > LOCAL_MAX_ENV_FRAMES)); then
        echo "[ERROR] Local training requests ${requested_frames} frames, above the ${LOCAL_MAX_ENV_FRAMES}-frame local limit." >&2
        echo "[HINT] Reduce LOCAL_TRAIN_MAX_ITERATIONS or submit the long run to Skynet." >&2
        exit 1
    fi
}

audit_motion_body_frames() {
    local report="${RUN_ROOT_BASE}/body_frame_audit.json"
    run_cmd pixi run python scripts/audit_g1_lafan1_body_frames.py \
        --manifest "${MANIFEST_PATH}" \
        --report "${report}"
}

require_legacy_interface_opt_in() {
    if [[ ("${SUBMIT_EE}" == "1" || "${SUBMIT_FB}" == "1") \
        && "${ALLOW_LEGACY_THREE_INTERFACE}" != "1" ]]; then
        echo "[ERROR] EE/full-body variants are historical appendix diagnostics, not active paper rows." >&2
        echo "[HINT] Use the focused latent-versus-exact-vanilla launchers. Set ALLOW_LEGACY_THREE_INTERFACE=1 only for an explicitly requested historical diagnostic." >&2
        exit 2
    fi
}

# ---------------------------------------------------------------------------
# Cluster stages
# ---------------------------------------------------------------------------

submit_train() {
    if [[ "${SUBMIT_LATENT}" == "1" ]]; then
        log "Submitting latent-stack training job (encoder -> planner -> ${LOW_LEVEL_ALGO} low-level, ~5B frames)."
        run_cmd env \
            "CLUSTER_SLURM_TIME_LIMIT=${TRAIN_WALLTIME}" \
            "MODE=lafan1-motion-tracking" \
            "DRY_RUN=${DRY_RUN}" \
            "RUN_BASE_PIPELINE=1" \
            "${LATENT_STAGE_FLAGS_OFF[@]}" \
            "RUN_HAND_DESIGNED_BASELINES=0" \
            "LOW_LEVEL_ALGO=${LOW_LEVEL_ALGO}" \
            "LOW_LEVEL_MAX_ITERATIONS=${TRAIN_MAX_ITERATIONS}" \
            "NUM_ENVS=${TRAIN_NUM_ENVS}" \
            "HORIZON_STEPS=${HORIZON_STEPS}" \
            "STATE_HISTORY_STEPS=${STATE_HISTORY_STEPS}" \
            "Z_DIM=${Z_DIM}" \
            "SKILL_UPDATES=${SKILL_UPDATES}" \
            "PLANNER_UPDATES=${PLANNER_UPDATES}" \
            "SEED=${SEED}" \
            "MANIFEST_PATH=${MANIFEST_PATH}" \
            "DATASET_PATH=${DATASET_PATH}" \
            "RUN_ID=${EXPERIMENT_TAG}_latent_train" \
            "RUN_ROOT=${RUN_ROOT_BASE}/latent" \
            "RANKS=0" \
            "LIMIT=1" \
            experiments/interface_baselines/submit_cluster_interface_baselines.sh
    fi

    local command_space
    for command_space in ee_trajectory full_body_trajectory; do
        if [[ "${command_space}" == "ee_trajectory" && "${SUBMIT_EE}" != "1" ]]; then
            continue
        fi
        if [[ "${command_space}" == "full_body_trajectory" && "${SUBMIT_FB}" != "1" ]]; then
            continue
        fi
        log "Submitting ${command_space} low-level training job (IPMD oracle-command tracking, ~5B frames, held ${HORIZON_STEPS}-step command chunks)."
        run_cmd env \
            "CLUSTER_SLURM_TIME_LIMIT=${TRAIN_WALLTIME}" \
            "DRY_RUN=${DRY_RUN}" \
            "COMMAND_SPACES=${command_space}" \
            "SEEDS=${SEED}" \
            "NUM_ENVS=${TRAIN_NUM_ENVS}" \
            "MAX_ITERATIONS=${TRAIN_MAX_ITERATIONS}" \
            "COMMAND_FUTURE_STEPS=${HORIZON_STEPS}" \
            "EXTRA_OVERRIDES=env.command_hold_steps=${HORIZON_STEPS}" \
            "MANIFEST=${MANIFEST_PATH}" \
            "REFRESH_ZARR_DATASET=false" \
            "SAVE_INTERVAL=${TRAIN_SAVE_INTERVAL}" \
            "PROJECT_NAME=${PROJECT_NAME}" \
            "GROUP_NAME=${EXPERIMENT_TAG}" \
            "RUN_PREFIX=${EXPERIMENT_TAG}" \
            experiments/command_space_ablation/submit_cluster_oracle_ablation.sh
    done

    cat <<EOF

[INFO] Training jobs submitted. When they finish:
[INFO]   latent checkpoints land under ${RUN_ROOT_BASE}/latent/base_pipeline/
[INFO]   EE/full-body checkpoints land under logs/rlopt/ipmd/Isaac-Imitation-G1-v0/<timestamp>/models/
[INFO] Find the EE/full-body checkpoints on the cluster with:
[INFO]   RUN_PREFIX=${EXPERIMENT_TAG} experiments/command_space_ablation/list_cluster_checkpoints.sh
[INFO] Then run STAGE=submit-eval with EE_TRAJECTORY_CHECKPOINT=... FULL_BODY_TRAJECTORY_CHECKPOINT=...
EOF
}

submit_eval() {
    if [[ "${SUBMIT_LATENT}" == "1" ]]; then
        log "Submitting latent per-motion evaluation job (oracle + base planner + per-motion finetune)."
        latent_eval_env=(
            "MODE=lafan1-motion-tracking"
            "DRY_RUN=${DRY_RUN}"
            "RUN_BASE_PIPELINE=0"
            "BASE_ROOT=${RUN_ROOT_BASE}/latent/base_pipeline"
            "${LATENT_STAGE_FLAGS_ON[@]}"
            "RUN_HAND_DESIGNED_BASELINES=0"
            "LOW_LEVEL_ALGO=${LOW_LEVEL_ALGO}"
            "HORIZON_STEPS=${HORIZON_STEPS}"
            "STATE_HISTORY_STEPS=${STATE_HISTORY_STEPS}"
            "Z_DIM=${Z_DIM}"
            "SEED=${SEED}"
            "MANIFEST_PATH=${MANIFEST_PATH}"
            "DATASET_PATH=${DATASET_PATH}"
            "EVAL_NUM_ENVS=${EVAL_NUM_ENVS}"
            "RANKS=${RANKS}"
            "LIMIT=${LIMIT}"
            "RUN_ID=${EXPERIMENT_TAG}_latent_eval"
            "RUN_ROOT=${RUN_ROOT_BASE}/latent"
            "AUTO_SYNC_LOCAL_CHECKPOINTS=0"
        )
        if [[ -n "${LATENT_LOW_LEVEL_CHECKPOINT}" ]]; then
            latent_eval_env+=("LOW_LEVEL_CHECKPOINT=${LATENT_LOW_LEVEL_CHECKPOINT}")
        fi
        run_cmd env \
            "${latent_eval_env[@]}" \
            experiments/interface_baselines/submit_cluster_interface_baselines.sh
    fi

    local interface checkpoint run_suffix submit_flag
    for interface in ee_trajectory full_body_trajectory; do
        if [[ "${interface}" == "ee_trajectory" ]]; then
            checkpoint="${EE_TRAJECTORY_CHECKPOINT}"
            run_suffix="ee"
            submit_flag="${SUBMIT_EE}"
        else
            checkpoint="${FULL_BODY_TRAJECTORY_CHECKPOINT}"
            run_suffix="fb"
            submit_flag="${SUBMIT_FB}"
        fi
        if [[ "${submit_flag}" != "1" ]]; then
            continue
        fi
        if [[ -z "${checkpoint}" ]]; then
            echo "[ERROR] ${interface} eval requires its trained low-level checkpoint." >&2
            echo "[HINT] Set EE_TRAJECTORY_CHECKPOINT / FULL_BODY_TRAJECTORY_CHECKPOINT to cluster-side paths, or disable with SUBMIT_EE=0 / SUBMIT_FB=0." >&2
            exit 1
        fi
        log "Submitting ${interface} baseline evaluation job (oracle + chunk planner pretrain/finetune)."
        run_cmd env \
            "MODE=lafan1-motion-tracking" \
            "DRY_RUN=${DRY_RUN}" \
            "RUN_BASE_PIPELINE=0" \
            "${LATENT_STAGE_FLAGS_OFF[@]}" \
            "RUN_HAND_DESIGNED_BASELINES=1" \
            "BASELINE_INTERFACES=${interface}" \
            "EE_TRAJECTORY_CHECKPOINT=${EE_TRAJECTORY_CHECKPOINT}" \
            "FULL_BODY_TRAJECTORY_CHECKPOINT=${FULL_BODY_TRAJECTORY_CHECKPOINT}" \
            "BASELINE_PRETRAIN_UPDATES=${BASELINE_PRETRAIN_UPDATES}" \
            "BASELINE_SAMPLE_BUDGETS=${BASELINE_SAMPLE_BUDGETS}" \
            "BASELINE_COMMAND_FUTURE_STEPS=${HORIZON_STEPS}" \
            "HORIZON_STEPS=${HORIZON_STEPS}" \
            "STATE_HISTORY_STEPS=${STATE_HISTORY_STEPS}" \
            "SEED=${SEED}" \
            "MANIFEST_PATH=${MANIFEST_PATH}" \
            "DATASET_PATH=${DATASET_PATH}" \
            "EVAL_NUM_ENVS=${EVAL_NUM_ENVS}" \
            "RANKS=${RANKS}" \
            "LIMIT=${LIMIT}" \
            "RUN_ID=${EXPERIMENT_TAG}_${run_suffix}_eval" \
            "RUN_ROOT=${RUN_ROOT_BASE}/${run_suffix}" \
            "AUTO_SYNC_LOCAL_CHECKPOINTS=0" \
            experiments/interface_baselines/submit_cluster_interface_baselines.sh
    done

    cat <<EOF

[INFO] Evaluation jobs submitted. Result roots (cluster shared logs tree):
[INFO]   ${RUN_ROOT_BASE}/latent/per_trajectory/
[INFO]   ${RUN_ROOT_BASE}/ee/per_trajectory/
[INFO]   ${RUN_ROOT_BASE}/fb/per_trajectory/
[INFO] After syncing results locally, run STAGE=summarize.
EOF
}

# ---------------------------------------------------------------------------
# Local stages (sequential, single GPU)
# ---------------------------------------------------------------------------

local_train_latent() {
    local base_pipeline_root="${RUN_ROOT_BASE}/latent/base_pipeline"
    local record="${base_pipeline_root}/low_level_checkpoint.txt"
    if checkpoint_file_valid "${record}"; then
        log "Latent stack already trained ($(<"${record}")); skipping."
        return 0
    fi

    local pipeline_env=(
        "TASK=Isaac-Imitation-G1-Latent-v0"
        "LOW_LEVEL_ALGO=${LOW_LEVEL_ALGO}"
        "DEVICE=${DEVICE}"
        "SEED=${SEED}"
        "NUM_ENVS=${TRAIN_NUM_ENVS}"
        "MANIFEST_PATH=${MANIFEST_PATH}"
        "DATASET_PATH=${DATASET_PATH}"
        "HORIZON_STEPS=${HORIZON_STEPS}"
        "STATE_HISTORY_STEPS=${STATE_HISTORY_STEPS}"
        "Z_DIM=${Z_DIM}"
        "SKILL_UPDATES=${SKILL_UPDATES}"
        "PLANNER_UPDATES=${PLANNER_UPDATES}"
        "LOW_LEVEL_MAX_ITERATIONS=${LOCAL_TRAIN_MAX_ITERATIONS}"
        "SAVE_INTERVAL=${LOCAL_TRAIN_SAVE_INTERVAL}"
        "RUN_ID=${EXPERIMENT_TAG}_latent_train"
        "RUN_ROOT=${base_pipeline_root}"
        "SKIP_ROLLOUT_FINETUNE=1"
        "SKIP_EVAL=1"
        "RUN_M1_EVAL=0"
        "TMPDIR=${LOCAL_TMPDIR}"
        "ISAACLAB_IMITATION_UNITREE_USD_CACHE_ROOT=${LOCAL_USD_CACHE}"
    )
    # Resume support: reuse finished encoder/planner checkpoints if present.
    if [[ -f "${base_pipeline_root}/skill_encoder_h${HORIZON_STEPS}_z${Z_DIM}/checkpoints/latest.pt" ]]; then
        log "Reusing existing skill encoder checkpoint."
        pipeline_env+=("SKIP_SKILL=1")
    fi
    if [[ -f "${base_pipeline_root}/planner_flow_matching_no_language_hist$((STATE_HISTORY_STEPS + 1))/checkpoints/latest.pt" ]]; then
        log "Reusing existing base planner checkpoint."
        pipeline_env+=("SKIP_PLANNER=1")
    fi

    log "Training latent stack locally for code qualification (encoder ${SKILL_UPDATES} upd -> planner ${PLANNER_UPDATES} upd -> ${LOW_LEVEL_ALGO} ${LOCAL_TRAIN_MAX_ITERATIONS} iters, at most 50M frames)."
    run_cmd env "${pipeline_env[@]}" \
        pixi run -e isaaclab bash scripts/rlopt/run_lafan1_no_language_pipeline.sh
}

local_train_command_space() {
    local interface="$1"
    local resume_checkpoint="$2"
    local record="${RUN_ROOT_BASE}/${interface}_checkpoint.txt"
    if checkpoint_file_valid "${record}"; then
        log "${interface} low-level already trained ($(<"${record}")); skipping."
        return 0
    fi

    mkdir -p "${RUN_ROOT_BASE}"
    local marker="${RUN_ROOT_BASE}/${interface}_train_started.marker"
    touch "${marker}"

    local manifest_abs
    manifest_abs="$(realpath "${MANIFEST_PATH}")"
    local train_args=(
        --headless
        --device "${DEVICE}"
        --num_envs "${TRAIN_NUM_ENVS}"
        --task "Isaac-Imitation-G1-v0"
        --algo "IPMD"
        --max_iterations "${LOCAL_TRAIN_MAX_ITERATIONS}"
    )
    if [[ -n "${resume_checkpoint}" ]]; then
        train_args+=(--checkpoint "${resume_checkpoint}")
    fi
    train_args+=(
        "agent.seed=${SEED}"
        "agent.command_space=${interface}"
        "agent.logger.exp_name=${EXPERIMENT_TAG}_${interface}"
        "agent.logger.project_name=${PROJECT_NAME}"
        "agent.logger.group_name=${EXPERIMENT_TAG}"
        "agent.save_interval=${LOCAL_TRAIN_SAVE_INTERVAL}"
        "agent.ipmd.use_latent_command=false"
        "env.latent_patch_past_steps=0"
        "env.latent_patch_future_steps=${HORIZON_STEPS}"
        "env.command_hold_steps=${HORIZON_STEPS}"
        "env.command_observation_source=reference"
        "env.lafan1_manifest_path=${manifest_abs}"
        "env.refresh_zarr_dataset=false"
    )

    log "Training ${interface} low-level locally for code qualification (IPMD, ${LOCAL_TRAIN_MAX_ITERATIONS} iters, at most 50M frames)."
    run_cmd pixi run -e isaaclab python scripts/rlopt/train.py "${train_args[@]}"

    if [[ "${DRY_RUN}" == "1" ]]; then
        return 0
    fi
    local log_root="${REPO_ROOT}/logs/rlopt/ipmd/Isaac-Imitation-G1-v0"
    local train_log_dir
    train_log_dir="$(find "${log_root}" -mindepth 1 -maxdepth 1 -type d -newer "${marker}" -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
    if [[ -z "${train_log_dir}" ]]; then
        echo "[ERROR] Could not locate ${interface} training log dir under ${log_root}." >&2
        exit 1
    fi
    local checkpoint
    checkpoint="$(latest_checkpoint_from_log_dir "${train_log_dir}")"
    if [[ -z "${checkpoint}" || ! -f "${checkpoint}" ]]; then
        echo "[ERROR] No model_step_*.pt found in ${train_log_dir}/models." >&2
        exit 1
    fi
    printf '%s\n' "${checkpoint}" > "${record}"
    log "${interface} checkpoint recorded: ${checkpoint}"
}

local_train() {
    validate_local_budget
    mkdir -p "${LOCAL_TMPDIR}" "${LOCAL_USD_CACHE}" "${RUN_ROOT_BASE}"
    if [[ "${SUBMIT_LATENT}" == "1" ]]; then
        local_train_latent
    fi
    if [[ "${SUBMIT_EE}" == "1" ]]; then
        local_train_command_space ee_trajectory "${EE_RESUME_CHECKPOINT}"
    fi
    if [[ "${SUBMIT_FB}" == "1" ]]; then
        local_train_command_space full_body_trajectory "${FB_RESUME_CHECKPOINT}"
    fi
    log "Local training stage complete."
}

local_eval() {
    if [[ "${SUBMIT_LATENT}" == "1" ]]; then
        log "Running latent per-motion evaluation locally (oracle + base planner + per-motion finetune)."
        latent_eval_env=(
            "RUN_BASE_PIPELINE=0"
            "BASE_ROOT=${RUN_ROOT_BASE}/latent/base_pipeline"
            "${LATENT_STAGE_FLAGS_ON[@]}"
            "RUN_HAND_DESIGNED_BASELINES=0"
            "TASK=Isaac-Imitation-G1-Latent-v0"
            "LOW_LEVEL_ALGO=${LOW_LEVEL_ALGO}"
            "DEVICE=${DEVICE}"
            "HORIZON_STEPS=${HORIZON_STEPS}"
            "STATE_HISTORY_STEPS=${STATE_HISTORY_STEPS}"
            "Z_DIM=${Z_DIM}"
            "SEED=${SEED}"
            "MANIFEST_PATH=${MANIFEST_PATH}"
            "DATASET_PATH=${DATASET_PATH}"
            "EVAL_NUM_ENVS=${EVAL_NUM_ENVS}"
            "RANKS=${RANKS}"
            "LIMIT=${LIMIT}"
            "RUN_ID=${EXPERIMENT_TAG}_latent_eval"
            "RUN_ROOT=${RUN_ROOT_BASE}/latent"
            "DRY_RUN=${DRY_RUN}"
        )
        if [[ -n "${LATENT_LOW_LEVEL_CHECKPOINT}" ]]; then
            latent_eval_env+=("LOW_LEVEL_CHECKPOINT=${LATENT_LOW_LEVEL_CHECKPOINT}")
        fi
        run_cmd env "${latent_eval_env[@]}" \
            bash experiments/interface_baselines/run_lafan1_motion_tracking_evaluation.sh
    fi

    local interface checkpoint record run_suffix submit_flag
    for interface in ee_trajectory full_body_trajectory; do
        if [[ "${interface}" == "ee_trajectory" ]]; then
            checkpoint="${EE_TRAJECTORY_CHECKPOINT}"
            run_suffix="ee"
            submit_flag="${SUBMIT_EE}"
        else
            checkpoint="${FULL_BODY_TRAJECTORY_CHECKPOINT}"
            run_suffix="fb"
            submit_flag="${SUBMIT_FB}"
        fi
        if [[ "${submit_flag}" != "1" ]]; then
            continue
        fi
        record="${RUN_ROOT_BASE}/${interface}_checkpoint.txt"
        if [[ -z "${checkpoint}" ]] && checkpoint_file_valid "${record}"; then
            checkpoint="$(<"${record}")"
        fi
        if [[ -z "${checkpoint}" && "${DRY_RUN}" == "1" ]]; then
            checkpoint="${record%.txt}_DRY_RUN.pt"
        fi
        if [[ -z "${checkpoint}" ]]; then
            echo "[ERROR] ${interface} eval requires its trained low-level checkpoint (missing ${record})." >&2
            echo "[HINT] Run STAGE=local-train first, or set EE_TRAJECTORY_CHECKPOINT / FULL_BODY_TRAJECTORY_CHECKPOINT." >&2
            exit 1
        fi
        log "Running ${interface} baseline evaluation locally (oracle + chunk planner pretrain/finetune)."
        run_cmd env \
            "RUN_BASE_PIPELINE=0" \
            "${LATENT_STAGE_FLAGS_OFF[@]}" \
            "RUN_HAND_DESIGNED_BASELINES=1" \
            "BASELINE_INTERFACES=${interface}" \
            "EE_TRAJECTORY_CHECKPOINT=${checkpoint}" \
            "FULL_BODY_TRAJECTORY_CHECKPOINT=${checkpoint}" \
            "BASELINE_PRETRAIN_UPDATES=${BASELINE_PRETRAIN_UPDATES}" \
            "BASELINE_SAMPLE_BUDGETS=${BASELINE_SAMPLE_BUDGETS}" \
            "BASELINE_COMMAND_FUTURE_STEPS=${HORIZON_STEPS}" \
            "BASELINE_PLANNER_UPDATE_INTERVAL=${HORIZON_STEPS}" \
            "BASELINE_COMMAND_HOLD_STEPS=${HORIZON_STEPS}" \
            "DEVICE=${DEVICE}" \
            "HORIZON_STEPS=${HORIZON_STEPS}" \
            "STATE_HISTORY_STEPS=${STATE_HISTORY_STEPS}" \
            "SEED=${SEED}" \
            "MANIFEST_PATH=${MANIFEST_PATH}" \
            "DATASET_PATH=${DATASET_PATH}" \
            "EVAL_NUM_ENVS=${EVAL_NUM_ENVS}" \
            "RANKS=${RANKS}" \
            "LIMIT=${LIMIT}" \
            "RUN_ID=${EXPERIMENT_TAG}_${run_suffix}_eval" \
            "RUN_ROOT=${RUN_ROOT_BASE}/${run_suffix}" \
            "DRY_RUN=${DRY_RUN}" \
            bash experiments/interface_baselines/run_lafan1_motion_tracking_evaluation.sh
    done
    log "Local evaluation stage complete."
}

summarize() {
    run_cmd pixi run python experiments/interface_baselines/summarize_lafan1_motion_tracking.py \
        --run_root "${RUN_ROOT_BASE}" \
        --oracle_success_threshold 0.8 \
        --output_dir "${RUN_ROOT_BASE}/summary"
}

case "${STAGE}" in
    submit-train|submit-eval|local-train|local-eval|local-all)
        require_legacy_interface_opt_in
        audit_motion_body_frames
        ;;
esac

case "${STAGE}" in
    submit-train)
        submit_train
        ;;
    submit-eval)
        submit_eval
        ;;
    local-train)
        local_train
        ;;
    local-eval)
        local_eval
        ;;
    local-all)
        local_train
        local_eval
        summarize
        ;;
    summarize)
        summarize
        ;;
    *)
        echo "[ERROR] Set STAGE=submit-train | submit-eval | local-train | local-eval | local-all | summarize" >&2
        usage
        ;;
esac
