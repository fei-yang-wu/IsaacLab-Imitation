#!/usr/bin/env bash
# Submit the BONES-SEED-100 pretrain + low-level IPMD 2B job to Skynet.
#
# Prereqs: the dataset must already be staged at
#   ${CLUSTER_DATA_DIR}/bones_seed_100   (npz + manifests; npz are rsync-excluded)
# e.g.  rsync -az data/bones_seed_100/ skynet:/coc/flash12/$USER/Research/IsaacLab/data/bones_seed_100/
#
# DRY_RUN=1 (default) prints the SLURM job script without submitting.
# DRY_RUN=0 submits.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

SEED="${SEED:-0}"
FRAME_CAP="${FRAME_CAP:-2000000000}"
TRAIN_NUM_ENVS="${TRAIN_NUM_ENVS:-4096}"
WANDB_PROJECT="${WANDB_PROJECT:-g1-bones-seed-100-hl-skill-2b}"
EXP_NAME="${EXP_NAME:-bones_seed_100_pretrain_lowlevel_seed${SEED}}"
DATA_ROOT="${DATA_ROOT:-/data/bones_seed_100}"
DRY_RUN="${DRY_RUN:-1}"
CLUSTER_PROFILE="${CLUSTER_PROFILE:-skynet_bones100}"

cmd=(
    ./docker/cluster/cluster_interface.sh
    -c "$CLUSTER_PROFILE"
    job
    --data-root "$DATA_ROOT"
    --seed "$SEED"
    --frame-cap "$FRAME_CAP"
    --train-num-envs "$TRAIN_NUM_ENVS"
    --logger-backend wandb
    --wandb-project "$WANDB_PROJECT"
    --exp-name "$EXP_NAME"
)

if [[ "$DRY_RUN" == "1" ]]; then
    export CLUSTER_DRY_RUN=1
    echo "[DRY_RUN] ${cmd[*]}"
fi

exec "${cmd[@]}"
