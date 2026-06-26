#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

TASK="${TASK:-Isaac-Imitation-G1-v0}"
ALGO="${ALGO:-IPMD}"
NUM_ENVS="${NUM_ENVS:-16}"
MAX_ITERATIONS="${MAX_ITERATIONS:-1}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-600}"
SEEDS_STR="${SEEDS:-2024}"
COMMAND_SPACES_STR="${COMMAND_SPACES:-single_frame_full_body full_body_trajectory ee_trajectory}"
COMMAND_FUTURE_STEPS="${COMMAND_FUTURE_STEPS:-25}"
COMMAND_OBSERVATION_SOURCE="${COMMAND_OBSERVATION_SOURCE:-reference}"
MANIFEST="${MANIFEST:-./data/unitree/manifests/g1_unitree_dance102_manifest.json}"
REFRESH_ZARR_DATASET="${REFRESH_ZARR_DATASET:-false}"
LOGGER_BACKEND="${LOGGER_BACKEND:-}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1}"
EXTRA_OVERRIDES_STR="${EXTRA_OVERRIDES:-}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "${ALGO^^}" != "IPMD" ]]; then
    echo "[ERROR] This command-space launcher currently supports ALGO=IPMD only." >&2
    exit 1
fi

if ! command -v pixi >/dev/null 2>&1; then
    echo "[ERROR] 'pixi' command is required but not found." >&2
    exit 1
fi

if [[ "$DRY_RUN" != "1" && "$DRY_RUN" != "true" ]]; then
    if ! command -v timeout >/dev/null 2>&1; then
        echo "[ERROR] 'timeout' command is required but not found." >&2
        exit 1
    fi
fi

read -r -a SEED_LIST <<< "$SEEDS_STR"
read -r -a COMMAND_SPACE_LIST <<< "$COMMAND_SPACES_STR"
read -r -a EXTRA_OVERRIDES_LIST <<< "$EXTRA_OVERRIDES_STR"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$SCRIPT_DIR/logs/oracle_smoke_${TIMESTAMP}"
if [[ "$DRY_RUN" != "1" && "$DRY_RUN" != "true" ]]; then
    mkdir -p "$LOG_DIR"
fi

COMMON_OVERRIDES=(
    "env.lafan1_manifest_path=${MANIFEST}"
    "env.refresh_zarr_dataset=${REFRESH_ZARR_DATASET}"
    "env.latent_patch_past_steps=0"
    "env.latent_patch_future_steps=${COMMAND_FUTURE_STEPS}"
    "env.command_observation_source=${COMMAND_OBSERVATION_SOURCE}"
    "agent.ipmd.use_latent_command=false"
    "agent.logger.backend=${LOGGER_BACKEND}"
    "agent.save_interval=${SAVE_INTERVAL}"
)

if [[ -n "$EXTRA_OVERRIDES_STR" ]]; then
    COMMON_OVERRIDES+=("${EXTRA_OVERRIDES_LIST[@]}")
fi

run_one() {
    local command_space="$1"
    local seed="$2"
    local run_name="cmd_space_oracle_${command_space}_seed${seed}_local"
    local log_file="$LOG_DIR/${run_name}.log"
    local cmd=(
        pixi run -e isaaclab python scripts/rlopt/train.py
        --task "$TASK"
        --num_envs "$NUM_ENVS"
        --headless
        --algo "$ALGO"
        --max_iterations "$MAX_ITERATIONS"
        --log_interval 1000
        --kit_args=--/app/extensions/fsWatcherEnabled=false
        "agent.seed=${seed}"
        "agent.command_space=${command_space}"
        "agent.logger.exp_name=${run_name}"
        "${COMMON_OVERRIDES[@]}"
    )

    printf "\n[%s] Running %s\n" "$(date '+%F %T')" "$run_name"
    printf "[CMD] "
    printf "%q " "${cmd[@]}"
    printf "\n"

    if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
        return 0
    fi

    set +e
    timeout --signal=TERM --kill-after=20s --preserve-status "$TIMEOUT_SECONDS" \
        "${cmd[@]}" >"$log_file" 2>&1
    local rc=$?
    set -e

    case "$rc" in
        0)
            echo "[DONE] ${run_name} completed (log: $log_file)"
            ;;
        124|137|143)
            echo "[TIMEOUT] ${run_name} hit timeout (log: $log_file)"
            ;;
        *)
            echo "[FAIL] ${run_name} exited with code ${rc} (log: $log_file)"
            ;;
    esac
}

echo "[INFO] Repo root: $REPO_ROOT"
echo "[INFO] Logs dir:  $LOG_DIR"
echo "[INFO] task=${TASK}, algo=${ALGO}, num_envs=${NUM_ENVS}, max_iterations=${MAX_ITERATIONS}"
echo "[INFO] command_spaces='${COMMAND_SPACES_STR}', command_future_steps=${COMMAND_FUTURE_STEPS}, command_observation_source='${COMMAND_OBSERVATION_SOURCE}', save_interval=${SAVE_INTERVAL}, manifest='${MANIFEST}'"

for command_space in "${COMMAND_SPACE_LIST[@]}"; do
    for seed in "${SEED_LIST[@]}"; do
        run_one "$command_space" "$seed"
    done
done

echo
if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
    echo "[INFO] Finished command-space oracle smoke dry run."
else
    echo "[INFO] Finished command-space oracle smoke sweep. Check logs under: $LOG_DIR"
fi
