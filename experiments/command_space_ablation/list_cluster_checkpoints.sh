#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

CLUSTER_LOGIN="${CLUSTER_LOGIN:-skynet}"
REMOTE_LOG_ROOT="${REMOTE_LOG_ROOT:-/nethome/fwu91/scratch/Research/IsaacLab/isaaclab/logs/rlopt/ipmd/Isaac-Imitation-G1-v0}"
CONTAINER_LOG_ROOT="${CONTAINER_LOG_ROOT:-logs/rlopt/ipmd/Isaac-Imitation-G1-v0}"
COMMAND_SPACES_STR="${COMMAND_SPACES:-single_frame_full_body full_body_trajectory ee_trajectory}"
SEEDS_STR="${SEEDS:-2024 2025 2026}"
RUN_PREFIX="${RUN_PREFIX:-g1_cmd_space_oracle_dance102_4096_1b_h25}"
ALLOW_MISSING="${ALLOW_MISSING:-0}"
TARGET_STEP="${TARGET_STEP:-}"
TARGET_STEP_TOLERANCE="${TARGET_STEP_TOLERANCE:-1000000}"

echo "[INFO] cluster_login=${CLUSTER_LOGIN}"
echo "[INFO] remote_log_root=${REMOTE_LOG_ROOT}"
echo "[INFO] container_log_root=${CONTAINER_LOG_ROOT}"
echo "[INFO] command_spaces='${COMMAND_SPACES_STR}', seeds='${SEEDS_STR}', run_prefix='${RUN_PREFIX}'"
if [[ -n "$TARGET_STEP" ]]; then
    echo "[INFO] target_step=${TARGET_STEP}, target_step_tolerance=${TARGET_STEP_TOLERANCE}"
fi

printf -v remote_cmd \
    "REMOTE_LOG_ROOT=%q CONTAINER_LOG_ROOT=%q COMMAND_SPACES_STR=%q SEEDS_STR=%q RUN_PREFIX=%q ALLOW_MISSING=%q TARGET_STEP=%q TARGET_STEP_TOLERANCE=%q bash -s" \
    "$REMOTE_LOG_ROOT" \
    "$CONTAINER_LOG_ROOT" \
    "$COMMAND_SPACES_STR" \
    "$SEEDS_STR" \
    "$RUN_PREFIX" \
    "$ALLOW_MISSING" \
    "$TARGET_STEP" \
    "$TARGET_STEP_TOLERANCE"

ssh "$CLUSTER_LOGIN" "$remote_cmd" <<'REMOTE_SCRIPT'
set -euo pipefail

remote_log_root="$REMOTE_LOG_ROOT"
container_log_root="$CONTAINER_LOG_ROOT"
command_spaces_str="$COMMAND_SPACES_STR"
seeds_str="$SEEDS_STR"
run_prefix="$RUN_PREFIX"
allow_missing="$ALLOW_MISSING"
target_step="$TARGET_STEP"
target_step_tolerance="$TARGET_STEP_TOLERANCE"

read -r -a command_spaces <<< "$command_spaces_str"
read -r -a seeds <<< "$seeds_str"

checkpoint_hosts=()
checkpoint_containers=()
missing=0

select_checkpoint() {
    local models_dir="$1"
    local target="$2"
    local tolerance="$3"

    if [ ! -d "$models_dir" ]; then
        return 0
    fi

    if [ -z "$target" ]; then
        find "$models_dir" -maxdepth 1 -type f -name '*.pt' -printf '%T@ %p\n' \
            | sort -n \
            | tail -n 1 \
            | cut -d' ' -f2-
        return 0
    fi

    find "$models_dir" -maxdepth 1 -type f -name 'model_step_*.pt' -printf '%f %p\n' \
        | awk -v target="$target" -v tolerance="$tolerance" '
            {
                step = $1
                sub(/^model_step_/, "", step)
                sub(/\.pt$/, "", step)
                if (step !~ /^[0-9]+$/) {
                    next
                }
                delta = step - target
                if (delta < 0) {
                    delta = -delta
                }
                if (delta <= tolerance && (best == "" || delta < best_delta)) {
                    best = $2
                    best_delta = delta
                }
            }
            END {
                if (best != "") {
                    print best
                }
            }'
}

printf "%-28s %-8s %-12s %-19s %s\n" "command_space" "seed" "status" "run_dir" "checkpoint"

for command_space in "${command_spaces[@]}"; do
    for seed in "${seeds[@]}"; do
        expected_exp_name="${run_prefix}_${command_space}_seed${seed}"
        selected_run_dir=""
        selected_stamp=""

        for run_dir in "$remote_log_root"/*; do
            [ -d "$run_dir" ] || continue
            agent_yaml="$run_dir/params/agent.yaml"
            [ -f "$agent_yaml" ] || continue
            exp_name="$(awk -F': ' '/^  exp_name:/{print $2; exit}' "$agent_yaml")"
            saved_command_space="$(awk '/^command_space:/{print $2; exit}' "$agent_yaml")"
            saved_seed="$(awk '/^seed:/{print $2; exit}' "$agent_yaml")"
            if [ "$exp_name" != "$expected_exp_name" ]; then
                continue
            fi
            if [ "$saved_command_space" != "$command_space" ] || [ "$saved_seed" != "$seed" ]; then
                continue
            fi
            stamp="$(basename "$run_dir")"
            if [ -z "$selected_stamp" ] || [ "$stamp" \> "$selected_stamp" ]; then
                selected_stamp="$stamp"
                selected_run_dir="$run_dir"
            fi
        done

        if [ -z "$selected_run_dir" ]; then
            printf "%-28s %-8s %-12s %-19s %s\n" "$command_space" "$seed" "run_missing" "-" "-"
            missing=1
            continue
        fi

        checkpoint_host="$(select_checkpoint "$selected_run_dir/models" "$target_step" "$target_step_tolerance")"

        if [ -z "$checkpoint_host" ]; then
            printf "%-28s %-8s %-12s %-19s %s\n" \
                "$command_space" "$seed" "ckpt_missing" "$(basename "$selected_run_dir")" "-"
            missing=1
            continue
        fi

        relative_checkpoint="${checkpoint_host#"$remote_log_root"/}"
        checkpoint_container="${container_log_root}/${relative_checkpoint}"
        checkpoint_hosts+=("$checkpoint_host")
        checkpoint_containers+=("$checkpoint_container")
        printf "%-28s %-8s %-12s %-19s %s\n" \
            "$command_space" "$seed" "ok" "$(basename "$selected_run_dir")" "$checkpoint_container"
    done
done

echo
if [ "${#checkpoint_hosts[@]}" -gt 0 ]; then
    printf "CHECKPOINTS_HOST="
    printf "%q " "${checkpoint_hosts[@]}"
    echo
else
    echo "CHECKPOINTS_HOST="
fi
if [ "${#checkpoint_containers[@]}" -gt 0 ]; then
    printf "CHECKPOINTS_CONTAINER="
    printf "%q " "${checkpoint_containers[@]}"
    echo
else
    echo "CHECKPOINTS_CONTAINER="
fi

if [ "$missing" -ne 0 ]; then
    echo "[INFO] One or more checkpoints are not available yet." >&2
    if [ "$allow_missing" != "1" ] && [ "$allow_missing" != "true" ]; then
        exit 1
    fi
fi
REMOTE_SCRIPT
