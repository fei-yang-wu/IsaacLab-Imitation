#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

TASK="${TASK:-Isaac-Imitation-G1-v0}"
ALGO="${ALGO:-IPMD}"
NUM_ENVS="${NUM_ENVS:-128}"
STEPS="${STEPS:-1000}"
SEEDS_STR="${SEEDS:-2024 2025 2026}"
COMMAND_SPACES_STR="${COMMAND_SPACES:-single_frame_full_body full_body_trajectory ee_trajectory}"
CHECKPOINTS_STR="${CHECKPOINTS:-}"
COMMAND_FUTURE_STEPS="${COMMAND_FUTURE_STEPS:-25}"
COMMAND_PAST_STEPS="${COMMAND_PAST_STEPS:-0}"
PLANNER_MODE="${PLANNER_MODE:-none}"
PLANNER_UPDATE_INTERVAL="${PLANNER_UPDATE_INTERVAL:-1}"
PLANNER_NOISE_STD="${PLANNER_NOISE_STD:-0.0}"
COMMAND_OBSERVATION_SOURCE="${COMMAND_OBSERVATION_SOURCE:-}"
if [[ -z "$COMMAND_OBSERVATION_SOURCE" ]]; then
    if [[ "$PLANNER_MODE" == "none" ]]; then
        COMMAND_OBSERVATION_SOURCE="reference"
    else
        COMMAND_OBSERVATION_SOURCE="planner"
    fi
fi
MANIFEST="${MANIFEST:-/data/unitree/manifests/g1_unitree_dance102_manifest.json}"
REFRESH_ZARR_DATASET="${REFRESH_ZARR_DATASET:-false}"
OUTPUT_TAG="${OUTPUT_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-logs/command_space_ablation/eval_results/${OUTPUT_TAG}}"
EXTRA_OVERRIDES_STR="${EXTRA_OVERRIDES:-}"
DRY_RUN="${DRY_RUN:-0}"
CLUSTER_PROFILE="${CLUSTER_PROFILE:-}"

# Evaluation writes logs/checkpoints to bind mounts and does not need the large
# training overlay by default. Keep these overrideable for unusual cluster setups.
export CLUSTER_OVERLAY_SIZE_MB="${CLUSTER_OVERLAY_SIZE_MB:-8192}"
export CLUSTER_SLURM_TIME_LIMIT="${CLUSTER_SLURM_TIME_LIMIT:-04:00:00}"
export CLUSTER_SKIP_CACHE_COPY="${CLUSTER_SKIP_CACHE_COPY:-1}"

if [[ "${ALGO^^}" != "IPMD" ]]; then
    echo "[ERROR] This cluster evaluator launcher currently supports ALGO=IPMD only." >&2
    exit 1
fi

CHECKPOINTS_AUTO_DISCOVERED=0
if [[ -z "$CHECKPOINTS_STR" ]]; then
    echo "[INFO] CHECKPOINTS unset; discovering latest cluster checkpoints."
    discovery_output="$(
        experiments/command_space_ablation/list_cluster_checkpoints.sh
    )"
    printf "%s\n" "$discovery_output"
    CHECKPOINTS_STR="$(
        printf "%s\n" "$discovery_output" \
            | sed -n 's/^CHECKPOINTS_CONTAINER=//p'
    )"
    CHECKPOINTS_AUTO_DISCOVERED=1
fi

if [[ -z "$CHECKPOINTS_STR" ]]; then
    echo "[ERROR] No checkpoints found. Wait for training checkpoints or set CHECKPOINTS explicitly." >&2
    exit 1
fi

read -r -a SEED_LIST <<< "$SEEDS_STR"
read -r -a COMMAND_SPACE_LIST <<< "$COMMAND_SPACES_STR"
if [[ "$CHECKPOINTS_AUTO_DISCOVERED" == "1" ]]; then
    eval "CHECKPOINT_LIST=($CHECKPOINTS_STR)"
else
    read -r -a CHECKPOINT_LIST <<< "$CHECKPOINTS_STR"
fi
read -r -a EXTRA_OVERRIDES_LIST <<< "$EXTRA_OVERRIDES_STR"

EXPECTED_CHECKPOINTS=$(( ${#SEED_LIST[@]} * ${#COMMAND_SPACE_LIST[@]} ))
if [[ "${#CHECKPOINT_LIST[@]}" -ne "$EXPECTED_CHECKPOINTS" ]]; then
    echo "[ERROR] Expected $EXPECTED_CHECKPOINTS checkpoints for ${#COMMAND_SPACE_LIST[@]} command spaces x ${#SEED_LIST[@]} seeds, got ${#CHECKPOINT_LIST[@]}." >&2
    echo "[INFO] Order must be command-space major, then seed, matching submit_cluster_oracle_ablation.sh." >&2
    exit 1
fi

submit_one() {
    local command_space="$1"
    local seed="$2"
    local checkpoint="$3"
    local label="${command_space}_seed${seed}_${PLANNER_MODE}"
    local cmd=(./docker/cluster/cluster_interface.sh job)

    if [[ -n "$CLUSTER_PROFILE" ]]; then
        cmd+=("$CLUSTER_PROFILE")
    fi

    cmd+=(
        --task "$TASK"
        --algo "$ALGO"
        --checkpoint "$checkpoint"
        --command_space "$command_space"
        --command_past_steps "$COMMAND_PAST_STEPS"
        --command_future_steps "$COMMAND_FUTURE_STEPS"
        --command_observation_source "$COMMAND_OBSERVATION_SOURCE"
        --planner_mode "$PLANNER_MODE"
        --planner_update_interval "$PLANNER_UPDATE_INTERVAL"
        --planner_noise_std "$PLANNER_NOISE_STD"
        --motion_manifest "$MANIFEST"
        --num_envs "$NUM_ENVS"
        --steps "$STEPS"
        --seed "$seed"
        --label "$label"
        --output_json "$OUTPUT_DIR/${label}.json"
        --output_csv "$OUTPUT_DIR/${label}.csv"
        --headless
        --kit_args=--/app/extensions/fsWatcherEnabled=false
    )
    if [[ "$REFRESH_ZARR_DATASET" == "1" || "$REFRESH_ZARR_DATASET" == "true" ]]; then
        cmd+=(--refresh_zarr_dataset)
    fi
    if [[ -n "$EXTRA_OVERRIDES_STR" ]]; then
        cmd+=("${EXTRA_OVERRIDES_LIST[@]}")
    fi

    printf "\n[%s] Submitting eval %s\n" "$(date '+%F %T')" "$label"
    printf "[CMD] "
    printf "%q " \
        CLUSTER_PYTHON_EXECUTABLE=experiments/command_space_ablation/evaluate_checkpoint.py \
        CLUSTER_APPEND_DEFAULT_G1_MANIFEST=0 \
        CLUSTER_SKIP_CACHE_COPY="$CLUSTER_SKIP_CACHE_COPY" \
        "${cmd[@]}"
    printf "\n"

    if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
        return 0
    fi
    CLUSTER_PYTHON_EXECUTABLE=experiments/command_space_ablation/evaluate_checkpoint.py \
    CLUSTER_APPEND_DEFAULT_G1_MANIFEST=0 \
    CLUSTER_SKIP_CACHE_COPY="$CLUSTER_SKIP_CACHE_COPY" \
        "${cmd[@]}"
}

echo "[INFO] Repo root: $REPO_ROOT"
echo "[INFO] Output dir inside container/project: $OUTPUT_DIR"
echo "[INFO] task=${TASK}, algo=${ALGO}, num_envs=${NUM_ENVS}, steps=${STEPS}"
echo "[INFO] command_spaces='${COMMAND_SPACES_STR}', seeds='${SEEDS_STR}', planner_mode='${PLANNER_MODE}'"
echo "[INFO] cluster eval defaults: CLUSTER_OVERLAY_SIZE_MB=${CLUSTER_OVERLAY_SIZE_MB}, CLUSTER_SLURM_TIME_LIMIT=${CLUSTER_SLURM_TIME_LIMIT}"
echo "[INFO] cluster eval cache copy: CLUSTER_SKIP_CACHE_COPY=${CLUSTER_SKIP_CACHE_COPY}"

checkpoint_index=0
for command_space in "${COMMAND_SPACE_LIST[@]}"; do
    for seed in "${SEED_LIST[@]}"; do
        checkpoint="${CHECKPOINT_LIST[$checkpoint_index]}"
        checkpoint_index=$(( checkpoint_index + 1 ))
        submit_one "$command_space" "$seed" "$checkpoint"
    done
done

echo
if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
    echo "[INFO] Finished cluster evaluation dry run."
else
    echo "[INFO] Submitted all requested command-space evaluation jobs."
    echo "[INFO] Merge per-checkpoint CSVs under: $OUTPUT_DIR"
fi
