#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

TASK="${TASK:-Isaac-Imitation-G1-v0}"
ALGO="${ALGO:-IPMD}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-10173}"
SEEDS_STR="${SEEDS:-2024 2025 2026}"
COMMAND_SPACES_STR="${COMMAND_SPACES:-single_frame_full_body full_body_trajectory ee_trajectory}"
COMMAND_FUTURE_STEPS="${COMMAND_FUTURE_STEPS:-25}"
COMMAND_OBSERVATION_SOURCE="${COMMAND_OBSERVATION_SOURCE:-reference}"
MANIFEST="${MANIFEST:-}"
REFRESH_ZARR_DATASET="${REFRESH_ZARR_DATASET:-false}"

PROJECT_NAME="${PROJECT_NAME:-G1-Imitation-Command-Space}"
GROUP_NAME="${GROUP_NAME:-g1_command_space_oracle_dance102_4096_1b_h25}"
RUN_PREFIX="${RUN_PREFIX:-g1_cmd_space_oracle_dance102_4096_1b_h25}"
SAVE_INTERVAL="${SAVE_INTERVAL:-50000000}"
EXTRA_OVERRIDES_STR="${EXTRA_OVERRIDES:-}"

DRY_RUN="${DRY_RUN:-0}"
VIDEO="${VIDEO:-1}"
VIDEO_LENGTH="${VIDEO_LENGTH:-200}"
VIDEO_INTERVAL="${VIDEO_INTERVAL:-2000}"
CLUSTER_PROFILE="${CLUSTER_PROFILE:-}"

if [[ "${ALGO^^}" != "IPMD" ]]; then
    echo "[ERROR] This command-space launcher currently supports ALGO=IPMD only." >&2
    exit 1
fi

read -r -a SEED_LIST <<< "$SEEDS_STR"
read -r -a COMMAND_SPACE_LIST <<< "$COMMAND_SPACES_STR"
read -r -a EXTRA_OVERRIDES_LIST <<< "$EXTRA_OVERRIDES_STR"

COMMON_OVERRIDES=(
    "env.latent_patch_past_steps=0"
    "env.latent_patch_future_steps=${COMMAND_FUTURE_STEPS}"
    "env.command_observation_source=${COMMAND_OBSERVATION_SOURCE}"
    "env.refresh_zarr_dataset=${REFRESH_ZARR_DATASET}"
    "agent.ipmd.use_latent_command=false"
    "agent.logger.project_name=${PROJECT_NAME}"
    "agent.logger.group_name=${GROUP_NAME}"
    "agent.save_interval=${SAVE_INTERVAL}"
)

if [[ -n "$MANIFEST" ]]; then
    COMMON_OVERRIDES+=("env.lafan1_manifest_path=${MANIFEST}")
fi
if [[ -n "$EXTRA_OVERRIDES_STR" ]]; then
    COMMON_OVERRIDES+=("${EXTRA_OVERRIDES_LIST[@]}")
fi

submit_one() {
    local command_space="$1"
    local seed="$2"
    local run_name="${RUN_PREFIX}_${command_space}_seed${seed}"
    local cmd=(./docker/cluster/cluster_interface.sh job)

    if [[ -n "$CLUSTER_PROFILE" ]]; then
        cmd+=("$CLUSTER_PROFILE")
    fi

    cmd+=(
        --task "$TASK"
        --num_envs "$NUM_ENVS"
        --headless
        --algo "$ALGO"
        --max_iterations "$MAX_ITERATIONS"
        --kit_args=--/app/extensions/fsWatcherEnabled=false
        "agent.seed=${seed}"
        "agent.command_space=${command_space}"
        "agent.logger.exp_name=${run_name}"
        "${COMMON_OVERRIDES[@]}"
    )

    if [[ "$VIDEO" == "1" || "$VIDEO" == "true" ]]; then
        cmd+=(
            --video
            --video_length "$VIDEO_LENGTH"
            --video_interval "$VIDEO_INTERVAL"
        )
    fi

    printf "\n[%s] Submitting %s\n" "$(date '+%F %T')" "$run_name"
    printf "[CMD] "
    printf "%q " "${cmd[@]}"
    printf "\n"

    if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
        return 0
    fi
    "${cmd[@]}"
}

echo "[INFO] Repo root: $REPO_ROOT"
echo "[INFO] task=${TASK}, algo=${ALGO}, num_envs=${NUM_ENVS}, max_iterations=${MAX_ITERATIONS}"
echo "[INFO] command_spaces='${COMMAND_SPACES_STR}', command_future_steps=${COMMAND_FUTURE_STEPS}, command_observation_source='${COMMAND_OBSERVATION_SOURCE}', seeds='${SEEDS_STR}'"
echo "[INFO] project='${PROJECT_NAME}', group='${GROUP_NAME}', dry_run='${DRY_RUN}'"

for command_space in "${COMMAND_SPACE_LIST[@]}"; do
    for seed in "${SEED_LIST[@]}"; do
        submit_one "$command_space" "$seed"
    done
done

echo
echo "[INFO] Submitted all requested command-space oracle ablation jobs."
