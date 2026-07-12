#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

MODE="${MODE:-dance102-strong}"
CLUSTER_PROFILE="${CLUSTER_PROFILE:-}"
DRY_RUN="${DRY_RUN:-0}"
EXTRA_LAUNCHER_ARGS_STR="${EXTRA_LAUNCHER_ARGS:-}"
AUTO_SYNC_LOCAL_CHECKPOINTS="${AUTO_SYNC_LOCAL_CHECKPOINTS:-1}"
AUTO_SYNC_EXTRA_AGGREGATE_ROOTS="${AUTO_SYNC_EXTRA_AGGREGATE_ROOTS:-1}"

if [[ "${MODE}" == lafan1-* ]]; then
    export CLUSTER_G1_MANIFEST_REFRESH_POLICY="${CLUSTER_G1_MANIFEST_REFRESH_POLICY:-auto}"
    export CLUSTER_G1_MANIFEST_PATH="${CLUSTER_G1_MANIFEST_PATH-}"
fi
export CLUSTER_APPEND_DEFAULT_G1_MANIFEST=0
export CLUSTER_PYTHON_EXECUTABLE=experiments/interface_baselines/run_interface_baseline_job.py
export CLUSTER_GIT_SYNC_FIRST="${CLUSTER_GIT_SYNC_FIRST:-0}"
export CLUSTER_EXTRA_RSYNC_EXCLUDES="${CLUSTER_EXTRA_RSYNC_EXCLUDES:-IsaacLab/ RLOpt/ ImitationLearningTools/}"
export CLUSTER_LINK_ISAACLAB_FROM_PREVIOUS="${CLUSTER_LINK_ISAACLAB_FROM_PREVIOUS:-1}"
export CLUSTER_SKIP_CACHE_COPY="${CLUSTER_SKIP_CACHE_COPY:-1}"
export CLUSTER_OVERLAY_SIZE_MB="${CLUSTER_OVERLAY_SIZE_MB:-8192}"
export CLUSTER_USE_SHARED_SIF="${CLUSTER_USE_SHARED_SIF:-1}"
if [[ -d "${REPO_ROOT}/RLOpt" ]]; then
    export CLUSTER_RLOPT_LOCAL_PATH="${CLUSTER_RLOPT_LOCAL_PATH:-${REPO_ROOT}/RLOpt}"
fi
if [[ -d "${REPO_ROOT}/ImitationLearningTools" ]]; then
    export CLUSTER_IMITATION_TOOLS_LOCAL_PATH="${CLUSTER_IMITATION_TOOLS_LOCAL_PATH:-${REPO_ROOT}/ImitationLearningTools}"
fi

read -r -a EXTRA_LAUNCHER_ARGS_LIST <<< "${EXTRA_LAUNCHER_ARGS_STR}"

append_extra_sync_spec() {
    local env_var_name="$1"
    local local_path="$2"
    local remote_subdir="$3"
    local spec="${local_path}:${remote_subdir}"

    if [[ " ${!env_var_name:-} " == *" ${spec} "* ]]; then
        return 0
    fi
    printf -v "${env_var_name}" '%s' "${!env_var_name:-}${!env_var_name:+ }${spec}"
    export "${env_var_name}"
}

is_truthy() {
    case "${1:-}" in
        1|true|TRUE|yes|YES|on|ON)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

aggregate_remote_dir_for_local_dir() {
    local local_dir="$1"

    if [[ "${local_dir}" == "${REPO_ROOT}/"* ]]; then
        printf '%s' "${local_dir#${REPO_ROOT}/}"
    else
        printf 'external/interface_baseline_results/%s' "$(basename "${local_dir}")"
    fi
}

maybe_sync_checkpoint_env() {
    local var_name="$1"
    local raw_path="${!var_name:-}"
    local local_file=""
    local local_parent=""
    local remote_file=""
    local remote_parent=""
    local basename=""

    if [[ -z "${raw_path}" ]]; then
        return 0
    fi

    if [[ "${raw_path}" = /* ]]; then
        if [[ ! -f "${raw_path}" ]]; then
            return 0
        fi
        local_file="$(realpath "${raw_path}")"
        if [[ "${local_file}" == "${REPO_ROOT}/"* ]]; then
            remote_file="${local_file#${REPO_ROOT}/}"
            remote_parent="$(dirname "${remote_file}")"
        else
            basename="$(basename "${local_file}")"
            remote_parent="external/interface_baseline_checkpoints/${var_name}"
            remote_file="${remote_parent}/${basename}"
        fi
    else
        if [[ ! -f "${REPO_ROOT}/${raw_path}" ]]; then
            return 0
        fi
        local_file="$(realpath "${REPO_ROOT}/${raw_path}")"
        remote_file="${raw_path}"
        remote_parent="$(dirname "${remote_file}")"
    fi

    local_parent="$(dirname "${local_file}")"
    append_extra_sync_spec CLUSTER_EXTRA_RSYNC_SPECS "${local_parent}" "${remote_parent}"
    printf -v "${var_name}" '%s' "${remote_file}"
    export "${var_name}"
    echo "[INFO] Auto-syncing ${var_name}: ${local_parent} -> ${remote_parent}; using ${remote_file}"
}

SYNCED_AGGREGATE_PATH=""

maybe_sync_extra_aggregate_root() {
    local raw_path="$1"
    local label="$2"
    local local_dir=""
    local remote_dir=""

    SYNCED_AGGREGATE_PATH="${raw_path}"
    if [[ -z "${raw_path}" ]]; then
        return 0
    fi

    if [[ "${raw_path}" = /* ]]; then
        if [[ ! -d "${raw_path}" ]]; then
            echo "[INFO] ${label} path is not local; leaving for cluster-side resolution: ${raw_path}"
            return 0
        fi
        local_dir="$(realpath "${raw_path}")"
    else
        if [[ ! -d "${REPO_ROOT}/${raw_path}" ]]; then
            echo "[INFO] ${label} path is not local; leaving for cluster-side resolution: ${raw_path}"
            return 0
        fi
        local_dir="$(realpath "${REPO_ROOT}/${raw_path}")"
    fi

    remote_dir="$(aggregate_remote_dir_for_local_dir "${local_dir}")"
    append_extra_sync_spec CLUSTER_EXTRA_RSYNC_SPECS "${local_dir}" "${remote_dir}"
    SYNCED_AGGREGATE_PATH="${remote_dir}"
    echo "[INFO] Auto-syncing ${label}: ${local_dir} -> ${remote_dir}; using ${remote_dir}"
}

extra_aggregate_glob_remote_pattern() {
    local pattern="$1"

    if [[ "${pattern}" = /* ]]; then
        if [[ "${pattern}" == "${REPO_ROOT}/"* ]]; then
            printf '%s' "${pattern#${REPO_ROOT}/}"
        else
            printf 'external/interface_baseline_results/%s' "$(basename "${pattern}")"
        fi
    else
        printf '%s' "${pattern}"
    fi
}

maybe_sync_extra_aggregate_glob() {
    local pattern="$1"
    local remote_pattern=""
    local match=""
    local matched_dir_count=0
    local -a matches=()

    SYNCED_AGGREGATE_PATH="${pattern}"
    if [[ -z "${pattern}" ]]; then
        return 0
    fi

    shopt -s nullglob
    if [[ "${pattern}" = /* ]]; then
        matches=(${pattern})
    else
        matches=(${REPO_ROOT}/${pattern})
    fi
    shopt -u nullglob

    for match in "${matches[@]}"; do
        if [[ ! -d "${match}" ]]; then
            continue
        fi
        matched_dir_count=$((matched_dir_count + 1))
        maybe_sync_extra_aggregate_root "${match}" "EXTRA_AGGREGATE_GLOBS match"
    done

    if [[ "${matched_dir_count}" -eq 0 ]]; then
        echo "[INFO] EXTRA_AGGREGATE_GLOBS pattern has no local directory matches; leaving for cluster-side resolution: ${pattern}"
        return 0
    fi

    remote_pattern="$(extra_aggregate_glob_remote_pattern "${pattern}")"
    SYNCED_AGGREGATE_PATH="${remote_pattern}"
}

maybe_sync_extra_aggregate_inputs() {
    local root=""
    local pattern=""
    local rewritten_roots=""
    local rewritten_globs=""
    local -a aggregate_roots=()
    local -a aggregate_globs=()

    read -r -a aggregate_roots <<< "${EXTRA_AGGREGATE_ROOTS:-}"
    for root in "${aggregate_roots[@]}"; do
        maybe_sync_extra_aggregate_root "${root}" "EXTRA_AGGREGATE_ROOTS"
        rewritten_roots="${rewritten_roots}${rewritten_roots:+ }${SYNCED_AGGREGATE_PATH}"
    done
    if [[ -n "${rewritten_roots}" ]]; then
        EXTRA_AGGREGATE_ROOTS="${rewritten_roots}"
        export EXTRA_AGGREGATE_ROOTS
    fi

    read -r -a aggregate_globs <<< "${EXTRA_AGGREGATE_GLOBS:-}"
    for pattern in "${aggregate_globs[@]}"; do
        maybe_sync_extra_aggregate_glob "${pattern}"
        rewritten_globs="${rewritten_globs}${rewritten_globs:+ }${SYNCED_AGGREGATE_PATH}"
    done
    if [[ -n "${rewritten_globs}" ]]; then
        EXTRA_AGGREGATE_GLOBS="${rewritten_globs}"
        export EXTRA_AGGREGATE_GLOBS
    fi
}

if is_truthy "${AUTO_SYNC_LOCAL_CHECKPOINTS}"; then
    for checkpoint_var in \
        LOW_LEVEL_CHECKPOINT \
        FULL_BODY_TRAJECTORY_CHECKPOINT \
        EE_TRAJECTORY_CHECKPOINT \
        LATENT_LOW_LEVEL_CHECKPOINT \
        LATENT_SKILL_CHECKPOINT \
        LATENT_PLANNER_CHECKPOINT \
        SKILL_CHECKPOINT \
        PLANNER_CHECKPOINT \
        LATENT_LANGUAGE_EMBEDDINGS; do
        maybe_sync_checkpoint_env "${checkpoint_var}"
    done
fi

if is_truthy "${AUTO_SYNC_EXTRA_AGGREGATE_ROOTS}"; then
    maybe_sync_extra_aggregate_inputs
fi

ENV_KEYS=(
    MODE
    TASK
    ALGORITHM
    LOW_LEVEL_ALGO
    DEVICE
    LOW_LEVEL_CHECKPOINT
    FULL_BODY_TRAJECTORY_CHECKPOINT
    EE_TRAJECTORY_CHECKPOINT
    SKILL_CHECKPOINT
    PLANNER_CHECKPOINT
    MANIFEST
    MANIFEST_PATH
    TRAIN_MANIFEST
    EVAL_MANIFEST
    FULL_MANIFEST
    DATASET_PATH
    SPLIT_OUTPUT_DIR
    SPLIT_PREFIX
    HELDOUT_NAMES
    HELDOUT_PATTERNS
    HELDOUT_COUNT
    HELDOUT_FRACTION
    OUTPUT_ROOT
    OUTPUT_PREFIX
    LATENT_OUTPUT_PREFIX
    AGGREGATE_OUTPUT_DIR
    EXTRA_AGGREGATE_ROOTS
    EXTRA_AGGREGATE_GLOBS
    INTERFACES
    BASELINE_INTERFACES
    BASELINE_TASK
    BASELINE_ALGO
    BASELINE_MODEL_SIZE
    BASELINE_SAMPLE_BUDGETS
    BASELINE_NUM_ENVS
    BASELINE_COLLECT_STEPS
    BASELINE_EVAL_STEPS
    BASELINE_COMMAND_PAST_STEPS
    BASELINE_COMMAND_FUTURE_STEPS
    BASELINE_USE_TRAJECTORY_STEPS
    RUN_ID
    RUN_ROOT
    BASE_ROOT
    RUN_LATENT
    RUN_LATENT_BASELINE
    RUN_BASE_PIPELINE
    RUN_ORACLE_RECON_EVAL
    RUN_BASE_PLANNER_PREDICT_EVAL
    RUN_ORACLE_LL_EVAL
    RUN_BASE_PLANNER_LL_EVAL
    RUN_PLANNER_FT_SAMPLE_COLLECTION
    RUN_PLANNER_ROLLOUT_FINETUNE
    RUN_FINETUNED_PLANNER_PREDICT_EVAL
    RUN_FINETUNED_PLANNER_LL_EVAL
    RUN_HAND_DESIGNED_BASELINES
    LATENT_TASK
    LATENT_ALGORITHM
    LATENT_LOW_LEVEL_CHECKPOINT
    LATENT_SKILL_CHECKPOINT
    LATENT_PLANNER_CHECKPOINT
    LATENT_LANGUAGE_EMBEDDINGS
    LATENT_MOTION_NAME
    LATENT_TRAJECTORY_NAME
    LATENT_DATASET_PATH
    LATENT_COMMAND_MODE
    LATENT_DIM
    LATENT_CODE_DIM
    LATENT_STEPS
    HORIZON_STEPS
    Z_DIM
    PLANNER_TYPE
    PLANNER_FLOW_STEPS
    PLANNER_EVAL_FLOW_NOISE_STD
    RANKS
    LIMIT
    SEED
    SEEDS
    NUM_ENVS
    EVAL_NUM_ENVS
    STEPS
    EVAL_STEPS
    EVAL_MAX_STEPS
    EVAL_VIDEO_LENGTH
    COLLECT_STEPS
    EVAL_METRIC_INTERVAL
    LATENT_EVAL_STEPS
    LATENT_COLLECT_STEPS
    STATE_HISTORY_STEPS
    COMMAND_PAST_STEPS
    COMMAND_FUTURE_STEPS
    MODEL_SIZE
    MODEL_SIZES
    SAMPLE_BUDGETS
    SELECTED_SAMPLE_COUNT
    PRETRAIN_UPDATES
    BASELINE_PRETRAIN_UPDATES
    FINETUNE_UPDATES
    BASELINE_FINETUNE_UPDATES
    DANCE102_FINETUNE_UPDATES
    PLANNER_FT_UPDATES
    PLANNER_FT_COLLECT_MAX_STEPS
    PLANNER_FT_BATCH_SIZE
    PLANNER_FT_LR
    PLANNER_FT_WEIGHT_DECAY
    AUDIT_EXPECTED_PRETRAIN_UPDATES
    FINETUNE_BATCH_SIZE
    BATCH_SIZE
    MICRO_BATCH_SIZE
    TRAIN_ENDPOINT_STEPS
    LR
    FINETUNE_LR
    WEIGHT_DECAY
    FINETUNE_WEIGHT_DECAY
    FLOW_STEPS
    FLOW_NOISE_STD
    SKILL_UPDATES
    SKILL_BATCH_SIZE
    PLANNER_UPDATES
    PLANNER_BATCH_SIZE
    LOW_LEVEL_MAX_ITERATIONS
    LOW_LEVEL_LOG_DIR
    LOW_LEVEL_VIDEO_LENGTH
    LOW_LEVEL_VIDEO_INTERVAL
    FORCE_COLLECT
    RUN_ORACLE
    RUN_PREFLIGHT
    RUN_CAPACITY_BACKFILL
    RUN_AGGREGATE
    RUN_AUDIT
    RUN_SWEEP_ANALYSIS
    USE_CHECKPOINT_NORMALIZATION
    AUDIT_PLANNER_VARIANTS
    AUDIT_EXPECTED_SEEDS
    MIN_ORACLE_SURVIVAL
    MIN_ORACLE_SUCCESS_RATE
)

launcher_args=(--mode "${MODE}")
if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" ]]; then
    launcher_args+=(--dry_run)
fi
for key in "${ENV_KEYS[@]}"; do
    if [[ "${!key+x}" ]]; then
        launcher_args+=(--env "${key}=${!key}")
    fi
done
launcher_args+=("${EXTRA_LAUNCHER_ARGS_LIST[@]}")

cmd=(./docker/cluster/cluster_interface.sh job)
if [[ -n "${CLUSTER_PROFILE}" ]]; then
    cmd+=("${CLUSTER_PROFILE}")
fi
cmd+=("${launcher_args[@]}")

echo "[INFO] Repo root: ${REPO_ROOT}"
echo "[INFO] mode=${MODE}, cluster_profile=${CLUSTER_PROFILE:-base}, dry_run=${DRY_RUN}"
echo "[INFO] CLUSTER_PYTHON_EXECUTABLE=${CLUSTER_PYTHON_EXECUTABLE}"
echo "[INFO] CLUSTER_GIT_SYNC_FIRST=${CLUSTER_GIT_SYNC_FIRST}"
echo "[INFO] CLUSTER_EXTRA_RSYNC_EXCLUDES=${CLUSTER_EXTRA_RSYNC_EXCLUDES}"
echo "[INFO] CLUSTER_LINK_ISAACLAB_FROM_PREVIOUS=${CLUSTER_LINK_ISAACLAB_FROM_PREVIOUS}"
echo "[INFO] CLUSTER_SKIP_CACHE_COPY=${CLUSTER_SKIP_CACHE_COPY}"
echo "[INFO] CLUSTER_OVERLAY_SIZE_MB=${CLUSTER_OVERLAY_SIZE_MB}"
echo "[INFO] CLUSTER_USE_SHARED_SIF=${CLUSTER_USE_SHARED_SIF}"
echo "[INFO] CLUSTER_RLOPT_LOCAL_PATH=${CLUSTER_RLOPT_LOCAL_PATH:-<unset>}"
echo "[INFO] CLUSTER_IMITATION_TOOLS_LOCAL_PATH=${CLUSTER_IMITATION_TOOLS_LOCAL_PATH:-<unset>}"
if [[ -n "${CLUSTER_EXTRA_RSYNC_SPECS:-}" ]]; then
    echo "[INFO] CLUSTER_EXTRA_RSYNC_SPECS=${CLUSTER_EXTRA_RSYNC_SPECS}"
fi
if [[ "${MODE}" == lafan1-* ]]; then
    echo "[INFO] LAFAN1 mode: CLUSTER_G1_MANIFEST_REFRESH_POLICY=${CLUSTER_G1_MANIFEST_REFRESH_POLICY}"
    if [[ -z "${CLUSTER_G1_MANIFEST_PATH}" ]]; then
        echo "[INFO] LAFAN1 mode: CLUSTER_G1_MANIFEST_PATH is empty, so cluster preflight uses the full default manifest."
    else
        echo "[INFO] LAFAN1 mode: CLUSTER_G1_MANIFEST_PATH=${CLUSTER_G1_MANIFEST_PATH}"
    fi
fi
printf "[CMD]"
printf " %q" "${cmd[@]}"
printf "\n"

if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" ]]; then
    exit 0
fi

"${cmd[@]}"
