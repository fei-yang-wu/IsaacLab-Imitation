#!/usr/bin/env bash
# Submit one Slurm job per LAFAN1 horizon cell.
# Defaults to 4 interfaces x W in {10,5,1}; DRY_RUN=1.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ! REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null)"; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${REPO_ROOT}"

DRY_RUN="${DRY_RUN:-1}"
CLUSTER_PROFILE="${CLUSTER_PROFILE:-skynet}"
WINDOWS="${WINDOWS:-10 5 1}"
INTERFACES="${INTERFACES:-latent_cont latent_fsq ee_chunk wb_chunk}"
RANKS="${RANKS:-all}"
SEED="${SEED:-0}"
BUDGET="${BUDGET:-full}"

_REQ_SLURM_QOS="${CLUSTER_SLURM_QOS:-}"
_REQ_SLURM_TIME_LIMIT="${CLUSTER_SLURM_TIME_LIMIT:-}"
_REQ_SLURM_SUBMIT_SCRIPT="${CLUSTER_SLURM_SUBMIT_SCRIPT:-}"
_REQ_SLURM_EXCLUDE="${CLUSTER_SLURM_EXCLUDE:-}"

_CLUSTER_ENV_FILE="${REPO_ROOT}/docker/cluster/.env.${CLUSTER_PROFILE}"
if [[ -z "${OUTPUT_ROOT:-}" && -f "${_CLUSTER_ENV_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${_CLUSTER_ENV_FILE}"
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-${CLUSTER_DATA_DIR:+$(dirname "${CLUSTER_DATA_DIR}")/logs/lafan1_ablation/seed${SEED}}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/coc/flash12/${USER}/Research/IsaacLab/logs/lafan1_ablation/seed${SEED}}"
unset _CLUSTER_ENV_FILE

[[ -n "${_REQ_SLURM_QOS}" ]] && CLUSTER_SLURM_QOS="${_REQ_SLURM_QOS}"
[[ -n "${_REQ_SLURM_TIME_LIMIT}" ]] && CLUSTER_SLURM_TIME_LIMIT="${_REQ_SLURM_TIME_LIMIT}"
[[ -n "${_REQ_SLURM_SUBMIT_SCRIPT}" ]] && CLUSTER_SLURM_SUBMIT_SCRIPT="${_REQ_SLURM_SUBMIT_SCRIPT}"
[[ -n "${_REQ_SLURM_EXCLUDE}" ]] && CLUSTER_SLURM_EXCLUDE="${_REQ_SLURM_EXCLUDE}"
unset _REQ_SLURM_QOS _REQ_SLURM_TIME_LIMIT _REQ_SLURM_SUBMIT_SCRIPT _REQ_SLURM_EXCLUDE

export CLUSTER_SLURM_SUBMIT_SCRIPT="${CLUSTER_SLURM_SUBMIT_SCRIPT:-skynet_pixi}"
export CLUSTER_AUTO_SETUP_G1_DATA="${CLUSTER_AUTO_SETUP_G1_DATA:-0}"
export CLUSTER_APPEND_DEFAULT_G1_MANIFEST="${CLUSTER_APPEND_DEFAULT_G1_MANIFEST:-0}"
export CLUSTER_G1_MANIFEST_REFRESH_POLICY="${CLUSTER_G1_MANIFEST_REFRESH_POLICY:-never}"
export CLUSTER_PYTHON_EXECUTABLE="${CLUSTER_PYTHON_EXECUTABLE:-experiments/lafan1_ablation/run_horizon_cell_cluster.py}"
export CLUSTER_SLURM_JOB_NAME_PREFIX="${CLUSTER_SLURM_JOB_NAME_PREFIX:-lafan1-hz}"

declare -a CELLS=()
for window in ${WINDOWS}; do
    for iface in ${INTERFACES}; do
        CELLS+=("${window}:${iface}")
    done
done

echo "[lafan1_ablation] cluster horizon matrix"
echo "  protocol=one_planner_per_trajectory (horizon_ablation)"
echo "  profile=${CLUSTER_PROFILE} budget=${BUDGET} dry_run=${DRY_RUN}"
echo "  windows=${WINDOWS}"
echo "  interfaces=${INTERFACES}"
echo "  ranks=${RANKS} cells=${#CELLS[@]}"
echo "  output_root=${OUTPUT_ROOT}"
echo "  logger=${LOGGER_BACKEND:-wandb} project=${WANDB_PROJECT:-G1-Imitation-LAFAN1-Ablation}"
echo "  qos=${CLUSTER_SLURM_QOS:-short} time=${CLUSTER_SLURM_TIME_LIMIT:-2-00:00:00}"
echo "  exclude=${CLUSTER_SLURM_EXCLUDE:-cyborg}"
echo "  physics=${PHYSICS_BACKEND:-<task-default>}"
if [[ "${BUDGET}" == "smoke" ]]; then
    _print_total_frames="${SMOKE_TOTAL_FRAMES:-8192}"
    _print_skill_updates="${SMOKE_SKILL_UPDATES:-2}"
    _print_eval_steps="${EVAL_STEPS:-16}"
    _print_rows_per_trajectory="${PLANNER_ROWS_PER_TRAJECTORY:-2}"
else
    _print_total_frames="${TOTAL_FRAMES:-2000000000}"
    _print_skill_updates="${SKILL_UPDATES:-50000}"
    _print_eval_steps="${EVAL_STEPS:-500}"
    _print_rows_per_trajectory="${PLANNER_ROWS_PER_TRAJECTORY:-1000}"
fi
echo "  fresh_stack=${FRESH_STACK:-1} total_frames=${_print_total_frames}"
echo "  skill_updates=${_print_skill_updates} skill_split=${SKILL_TRAIN_SPLIT:-train}/${SKILL_EVAL_SPLIT:-eval} eval_fraction=${SKILL_EVAL_TRAJECTORY_FRACTION:-0.1}"
echo "  state_history_steps=${STATE_HISTORY_STEPS:-9} eval_steps=${_print_eval_steps} rows_per_trajectory=${_print_rows_per_trajectory}"
unset _print_total_frames _print_skill_updates _print_eval_steps _print_rows_per_trajectory
echo

echo "[lafan1_ablation] planned jobs (${#CELLS[@]}):"
for cell in "${CELLS[@]}"; do
    window="${cell%%:*}"
    iface="${cell#*:}"
    echo "  [lafan1-hz-W${window}-${iface}] window=${window} interface=${iface}"
done

case "${DRY_RUN}" in
    1|true|TRUE|yes|YES|on|ON)
        echo
        echo "[lafan1_ablation] DRY_RUN=${DRY_RUN}: not contacting the cluster."
        echo "Re-run with DRY_RUN=0 after confirming the plan above."
        exit 0
        ;;
esac

first_cell="${CELLS[0]}"
first_window="${first_cell%%:*}"
first_iface="${first_cell#*:}"

echo "[lafan1_ablation] syncing workspace once via cluster_interface (no sbatch)..."
CLUSTER_SLURM_DRY_RUN=1 \
    ./docker/cluster/cluster_interface.sh \
    -c "${CLUSTER_PROFILE}" \
    job \
    --window="${first_window}" \
    --interfaces="${first_iface}" \
    --ranks="${RANKS}" \
    --seed="${SEED}" \
    --budget="${BUDGET}" \
    --output-root="${OUTPUT_ROOT}" \
    --dry-run=0

# Resolve the snapshot that was just synced.
_REQ_SLURM_QOS="${CLUSTER_SLURM_QOS:-}"
_REQ_SLURM_TIME_LIMIT="${CLUSTER_SLURM_TIME_LIMIT:-}"
_REQ_SLURM_EXCLUDE="${CLUSTER_SLURM_EXCLUDE:-}"
# shellcheck disable=SC1090
source "${REPO_ROOT}/docker/cluster/.env.${CLUSTER_PROFILE}"
[[ -n "${_REQ_SLURM_QOS}" ]] && CLUSTER_SLURM_QOS="${_REQ_SLURM_QOS}"
[[ -n "${_REQ_SLURM_TIME_LIMIT}" ]] && CLUSTER_SLURM_TIME_LIMIT="${_REQ_SLURM_TIME_LIMIT}"
[[ -n "${_REQ_SLURM_EXCLUDE}" ]] && CLUSTER_SLURM_EXCLUDE="${_REQ_SLURM_EXCLUDE}"
unset _REQ_SLURM_QOS _REQ_SLURM_TIME_LIMIT _REQ_SLURM_EXCLUDE
CLUSTER_LOGIN="${CLUSTER_LOGIN:-skynet}"
WORKSPACE="$(ssh -o BatchMode=yes -o ControlPath=none "${CLUSTER_LOGIN}" \
    "readlink -f '${CLUSTER_ISAACLAB_DIR}_latest'")"
if [[ -z "${WORKSPACE}" || "${WORKSPACE}" == *"No such"* ]]; then
    echo "[ERROR] could not resolve synced workspace via ${CLUSTER_ISAACLAB_DIR}_latest"
    exit 1
fi
echo "[lafan1_ablation] using workspace: ${WORKSPACE}"

submit_one_remote() {
    local window="$1"
    local iface="$2"
    local job_prefix="lafan1-hz-W${window}-${iface}"
    local budget_overrides=""
    if [[ -n "${STATE_HISTORY_STEPS:-}" ]]; then
        budget_overrides+=" export STATE_HISTORY_STEPS=$(printf '%q' "${STATE_HISTORY_STEPS}") &&"
    fi
    if [[ -n "${EVAL_STEPS:-}" ]]; then
        budget_overrides+=" export EVAL_STEPS=$(printf '%q' "${EVAL_STEPS}") &&"
    fi
    if [[ -n "${PLANNER_ROWS_PER_TRAJECTORY:-}" ]]; then
        budget_overrides+=" export PLANNER_ROWS_PER_TRAJECTORY=$(printf '%q' "${PLANNER_ROWS_PER_TRAJECTORY}") &&"
    fi
    if [[ -n "${SKILL_UPDATES:-}" ]]; then
        budget_overrides+=" export SKILL_UPDATES=$(printf '%q' "${SKILL_UPDATES}") &&"
    fi
    if [[ -n "${SKILL_TRAIN_SPLIT:-}" ]]; then
        budget_overrides+=" export SKILL_TRAIN_SPLIT=$(printf '%q' "${SKILL_TRAIN_SPLIT}") &&"
    fi
    if [[ -n "${SKILL_EVAL_SPLIT:-}" ]]; then
        budget_overrides+=" export SKILL_EVAL_SPLIT=$(printf '%q' "${SKILL_EVAL_SPLIT}") &&"
    fi
    if [[ -n "${SKILL_EVAL_TRAJECTORY_FRACTION:-}" ]]; then
        budget_overrides+=" export SKILL_EVAL_TRAJECTORY_FRACTION=$(printf '%q' "${SKILL_EVAL_TRAJECTORY_FRACTION}") &&"
    fi
    if [[ -n "${SKIP_PLANNERS:-}" ]]; then
        budget_overrides+=" export SKIP_PLANNERS=$(printf '%q' "${SKIP_PLANNERS}") &&"
    fi
    if [[ -n "${SKIP_ORACLE_LL:-}" ]]; then
        budget_overrides+=" export SKIP_ORACLE_LL=$(printf '%q' "${SKIP_ORACLE_LL}") &&"
    fi
    if [[ -n "${LATENT_LOW_LEVEL_CHECKPOINT:-}" ]]; then
        budget_overrides+=" export LATENT_LOW_LEVEL_CHECKPOINT=$(printf '%q' "${LATENT_LOW_LEVEL_CHECKPOINT}") &&"
    fi
    if [[ -n "${LOW_LEVEL_CHECKPOINT:-}" ]]; then
        budget_overrides+=" export LOW_LEVEL_CHECKPOINT=$(printf '%q' "${LOW_LEVEL_CHECKPOINT}") &&"
    fi
    if [[ -n "${PHYSICS_BACKEND:-}" ]]; then
        budget_overrides+=" export PHYSICS_BACKEND=$(printf '%q' "${PHYSICS_BACKEND}") &&"
    fi
    echo "[lafan1_ablation] submitting ${job_prefix}..."
    ssh -o BatchMode=yes -o ControlPath=none "${CLUSTER_LOGIN}" \
        "cd $(printf '%q' "${WORKSPACE}") && \
         export CLUSTER_PYTHON_EXECUTABLE=experiments/lafan1_ablation/run_horizon_cell_cluster.py && \
         export CLUSTER_PIXI_ENV=${CLUSTER_PIXI_ENV:-isaaclab} && \
         export CLUSTER_PIXI_CACHE_DIR=$(printf '%q' "${CLUSTER_PIXI_CACHE_DIR}") && \
         export CLUSTER_PIXI_SKIP_INSTALL=${CLUSTER_PIXI_SKIP_INSTALL:-0} && \
         export CLUSTER_PIXI_INSTALL_LOCK_WAIT=${CLUSTER_PIXI_INSTALL_LOCK_WAIT:-7200} && \
         export CLUSTER_DATA_DIR=$(printf '%q' "${CLUSTER_DATA_DIR}") && \
         export CLUSTER_HF_TOKEN_FILE=$(printf '%q' "${CLUSTER_HF_TOKEN_FILE}") && \
         export CLUSTER_WANDB_API_KEY_FILE=$(printf '%q' "${CLUSTER_WANDB_API_KEY_FILE}") && \
         export CLUSTER_SLURM_PARTITION=${CLUSTER_SLURM_PARTITION:-wu-lab} && \
         export CLUSTER_SLURM_QOS=${CLUSTER_SLURM_QOS:-short} && \
         export CLUSTER_SLURM_GPU_GRES=${CLUSTER_SLURM_GPU_GRES:-gpu:a40:1} && \
         export CLUSTER_SLURM_CPUS_PER_TASK=${CLUSTER_SLURM_CPUS_PER_TASK:-6} && \
         export CLUSTER_SLURM_MEM=${CLUSTER_SLURM_MEM:-96G} && \
         export CLUSTER_SLURM_TIME_LIMIT=${CLUSTER_SLURM_TIME_LIMIT:-2-00:00:00} && \
         export CLUSTER_SLURM_JOB_NAME_PREFIX=$(printf '%q' "${job_prefix}") && \
         export CLUSTER_SLURM_OUTPUT_DIR=logs/slurm && \
         export CLUSTER_SLURM_PRINT_JOB_SCRIPT=1 && \
         export CLUSTER_SLURM_EXCLUDE=${CLUSTER_SLURM_EXCLUDE:-cyborg} && \
         export LOGGER_BACKEND=${LOGGER_BACKEND:-wandb} && \
         export WANDB_PROJECT=${WANDB_PROJECT:-G1-Imitation-LAFAN1-Ablation} && \
         export WANDB_GROUP=${WANDB_GROUP:-lafan1_ablation_seed${SEED}} && \
         export FRESH_STACK=${FRESH_STACK:-1} && \
         export TOTAL_FRAMES=${TOTAL_FRAMES:-2000000000} && \
         ${budget_overrides} \
         export PATH=/opt/slurm/Ubuntu-20.04/24.11.0/bin:\$PATH && \
         bash docker/cluster/submit_job_slurm_skynet_pixi.sh \
            $(printf '%q' "${WORKSPACE}") \
            isaac-lab-base \
            --window=${window} \
            --interfaces=${iface} \
            --ranks=${RANKS} \
            --seed=${SEED} \
            --budget=${BUDGET} \
            --output-root=$(printf '%q' "${OUTPUT_ROOT}") \
            --dry-run=0"
}

for cell in "${CELLS[@]}"; do
    window="${cell%%:*}"
    iface="${cell#*:}"
    submit_one_remote "${window}" "${iface}"
done

echo
echo "[lafan1_ablation] submitted ${#CELLS[@]} horizon jobs against ${WORKSPACE}"
ssh -o BatchMode=yes -o ControlPath=none "${CLUSTER_LOGIN}" \
    'export PATH=/opt/slurm/Ubuntu-20.04/24.11.0/bin:$PATH; squeue -u "$USER" -o "%i %T %M %j %R"'
