#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

SEED="${SEED:-0}"
FRAME_CAP="${FRAME_CAP:-2000000000}"
TRAIN_NUM_ENVS="${TRAIN_NUM_ENVS:-4096}"
WANDB_PROJECT="${WANDB_PROJECT:-G1-Imitation-BONES-SEED}"
WANDB_GROUP="${WANDB_GROUP:-bones_seed_paper24_2b}"
EXP_NAME="${EXP_NAME:-bones_seed_paper24_2b_seed${SEED}}"
DATA_ROOT="${DATA_ROOT:-/data/bones_seed_paper24}"
RUN_ROOT="${RUN_ROOT:-logs/bones_seed_language/paper24_2b_seed${SEED}}"
DRY_RUN="${DRY_RUN:-1}"
CLUSTER_PROFILE="${CLUSTER_PROFILE:-skynet}"

export CLUSTER_PYTHON_EXECUTABLE="${CLUSTER_PYTHON_EXECUTABLE:-scripts/rlopt/run_bones_seed_language_pipeline.py}"
export CLUSTER_AUTO_SETUP_G1_DATA="${CLUSTER_AUTO_SETUP_G1_DATA:-0}"
export CLUSTER_APPEND_DEFAULT_G1_MANIFEST="${CLUSTER_APPEND_DEFAULT_G1_MANIFEST:-0}"
export CLUSTER_G1_MANIFEST_REFRESH_POLICY="${CLUSTER_G1_MANIFEST_REFRESH_POLICY:-never}"
export CLUSTER_SLURM_SUBMIT_SCRIPT="${CLUSTER_SLURM_SUBMIT_SCRIPT:-skynet}"
export CLUSTER_SLURM_JOB_NAME_PREFIX="${CLUSTER_SLURM_JOB_NAME_PREFIX:-bones24-2b}"

cmd=(
    ./docker/cluster/cluster_interface.sh
    -c "$CLUSTER_PROFILE"
    job
    --preset paper24
    --data-root "$DATA_ROOT"
    --run-root "$RUN_ROOT"
    --seed "$SEED"
    --frame-cap "$FRAME_CAP"
    --train-num-envs "$TRAIN_NUM_ENVS"
    --logger-backend wandb
    --wandb-project "$WANDB_PROJECT"
    --wandb-group "$WANDB_GROUP"
    --exp-name "$EXP_NAME"
)

if [ -n "${EXTRA_PIPELINE_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    extra_args=(${EXTRA_PIPELINE_ARGS})
    cmd+=("${extra_args[@]}")
fi

echo "[INFO] Repo root: $REPO_ROOT"
echo "[INFO] Skynet profile: $CLUSTER_PROFILE"
echo "[INFO] Run: seed=$SEED frame_cap=$FRAME_CAP train_num_envs=$TRAIN_NUM_ENVS"
echo "[INFO] Data root inside container: $DATA_ROOT"
echo "[INFO] Run root: $RUN_ROOT"
printf "[CMD] "
printf "%q " "${cmd[@]}"
printf "\n"

case "$DRY_RUN" in
    1|true|TRUE|yes|YES|on|ON)
        echo "[INFO] DRY_RUN=$DRY_RUN; not contacting the cluster."
        exit 0
        ;;
esac

"${cmd[@]}"
