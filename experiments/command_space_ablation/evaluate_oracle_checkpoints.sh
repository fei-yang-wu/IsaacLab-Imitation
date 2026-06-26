#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

TASK="${TASK:-Isaac-Imitation-G1-v0}"
ALGO="${ALGO:-IPMD}"
NUM_ENVS="${NUM_ENVS:-128}"
STEPS="${STEPS:-1000}"
SEEDS_STR="${SEEDS:-2024}"
COMMAND_SPACES_STR="${COMMAND_SPACES:-single_frame_full_body full_body_trajectory ee_trajectory}"
CHECKPOINTS_STR="${CHECKPOINTS:-}"
COMMAND_FUTURE_STEPS="${COMMAND_FUTURE_STEPS:-25}"
COMMAND_PAST_STEPS="${COMMAND_PAST_STEPS:-0}"
PLANNER_MODE="${PLANNER_MODE:-none}"
COMMAND_OBSERVATION_SOURCE="${COMMAND_OBSERVATION_SOURCE:-}"
if [[ -z "$COMMAND_OBSERVATION_SOURCE" ]]; then
    if [[ "$PLANNER_MODE" == "none" ]]; then
        COMMAND_OBSERVATION_SOURCE="reference"
    else
        COMMAND_OBSERVATION_SOURCE="planner"
    fi
fi
PLANNER_UPDATE_INTERVAL="${PLANNER_UPDATE_INTERVAL:-1}"
PLANNER_NOISE_STD="${PLANNER_NOISE_STD:-0.0}"
MANIFEST="${MANIFEST:-./data/unitree/manifests/g1_unitree_dance102_manifest.json}"
REFRESH_ZARR_DATASET="${REFRESH_ZARR_DATASET:-false}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/eval_results/$(date +%Y%m%d_%H%M%S)}"
OUTPUT_CSV="${OUTPUT_CSV:-$OUTPUT_DIR/summary.csv}"
APPEND_CSV="${APPEND_CSV:-0}"
EXTRA_OVERRIDES_STR="${EXTRA_OVERRIDES:-}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "${ALGO^^}" != "IPMD" ]]; then
    echo "[ERROR] This evaluator wrapper currently supports ALGO=IPMD only." >&2
    exit 1
fi

if [[ -z "$CHECKPOINTS_STR" ]]; then
    echo "[ERROR] CHECKPOINTS must list one checkpoint per command-space/seed pair." >&2
    exit 1
fi

if ! command -v pixi >/dev/null 2>&1; then
    echo "[ERROR] 'pixi' command is required but not found." >&2
    exit 1
fi

read -r -a SEED_LIST <<< "$SEEDS_STR"
read -r -a COMMAND_SPACE_LIST <<< "$COMMAND_SPACES_STR"
read -r -a CHECKPOINT_LIST <<< "$CHECKPOINTS_STR"
read -r -a EXTRA_OVERRIDES_LIST <<< "$EXTRA_OVERRIDES_STR"

EXPECTED_CHECKPOINTS=$(( ${#SEED_LIST[@]} * ${#COMMAND_SPACE_LIST[@]} ))
if [[ "${#CHECKPOINT_LIST[@]}" -ne "$EXPECTED_CHECKPOINTS" ]]; then
    echo "[ERROR] Expected $EXPECTED_CHECKPOINTS checkpoints for ${#COMMAND_SPACE_LIST[@]} command spaces x ${#SEED_LIST[@]} seeds, got ${#CHECKPOINT_LIST[@]}." >&2
    echo "[INFO] Order must be command-space major, then seed, matching submit_cluster_oracle_ablation.sh." >&2
    exit 1
fi

if [[ "$DRY_RUN" != "1" && "$DRY_RUN" != "true" ]]; then
    mkdir -p "$OUTPUT_DIR"
    if [[ "$APPEND_CSV" != "1" && "$APPEND_CSV" != "true" ]]; then
        rm -f "$OUTPUT_CSV"
    fi
fi

echo "[INFO] Repo root: $REPO_ROOT"
echo "[INFO] Output dir: $OUTPUT_DIR"
echo "[INFO] task=${TASK}, algo=${ALGO}, num_envs=${NUM_ENVS}, steps=${STEPS}"
echo "[INFO] command_spaces='${COMMAND_SPACES_STR}', seeds='${SEEDS_STR}', command_future_steps=${COMMAND_FUTURE_STEPS}"
echo "[INFO] command_observation_source=${COMMAND_OBSERVATION_SOURCE}, planner_mode=${PLANNER_MODE}, planner_update_interval=${PLANNER_UPDATE_INTERVAL}, planner_noise_std=${PLANNER_NOISE_STD}"

checkpoint_index=0
for command_space in "${COMMAND_SPACE_LIST[@]}"; do
    for seed in "${SEED_LIST[@]}"; do
        checkpoint="${CHECKPOINT_LIST[$checkpoint_index]}"
        checkpoint_index=$(( checkpoint_index + 1 ))
        label="${command_space}_seed${seed}"
        output_json="$OUTPUT_DIR/${label}.json"
        cmd=(
            pixi run -e isaaclab python experiments/command_space_ablation/evaluate_checkpoint.py
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
            --output_json "$output_json"
            --output_csv "$OUTPUT_CSV"
            --append_csv
            --headless
            --kit_args=--/app/extensions/fsWatcherEnabled=false
        )
        if [[ "$REFRESH_ZARR_DATASET" == "1" || "$REFRESH_ZARR_DATASET" == "true" ]]; then
            cmd+=(--refresh_zarr_dataset)
        fi
        if [[ -n "$EXTRA_OVERRIDES_STR" ]]; then
            cmd+=("${EXTRA_OVERRIDES_LIST[@]}")
        fi

        printf "\n[%s] Evaluating %s\n" "$(date '+%F %T')" "$label"
        printf "[CMD] "
        printf "%q " "${cmd[@]}"
        printf "\n"

        if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
            continue
        fi
        "${cmd[@]}"
    done
done

if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
    echo "[INFO] Finished command-space evaluation dry run."
else
    pixi run python experiments/command_space_ablation/summarize_eval_csv.py \
        --csv "$OUTPUT_CSV" \
        --output_md "$OUTPUT_DIR/summary.md"
    pixi run python experiments/command_space_ablation/summarize_eval_csv.py \
        --csv "$OUTPUT_CSV" \
        --aggregate \
        --output_md "$OUTPUT_DIR/aggregate.md"
    echo "[INFO] Wrote evaluation CSV: $OUTPUT_CSV"
    echo "[INFO] Wrote evaluation Markdown: $OUTPUT_DIR/summary.md"
    echo "[INFO] Wrote aggregate Markdown: $OUTPUT_DIR/aggregate.md"
fi
