#!/usr/bin/env bash
# Prepare the training table, then train one GR00T head arm.
#
#   ./train.sh <fsq64_10b|ln_hold1_10b> [hydra overrides...]
#   STAGE=prepare ./train.sh <arm>     # table only
#   STAGE=train   ./train.sh <arm>     # head only, table must exist
#
# The W&B group is required rather than defaulted: it is the label the runs are
# found by, and the repo convention is that a human confirms it.
set -euo pipefail
CAMPAIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${CAMPAIGN_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

ARM="${1:?usage: train.sh <fsq64_10b|ln_hold1_10b> [hydra overrides...]}"
shift || true
CONF="${CAMPAIGN_DIR}/conf"
[ -f "${CONF}/train_${ARM}.yaml" ] || { echo "unknown arm ${ARM}" >&2; exit 1; }
STAGE="${STAGE:-all}"
WANDB_GROUP="${WANDB_GROUP:?set WANDB_GROUP to the confirmed W&B group name}"

if [ "${STAGE}" = "all" ] || [ "${STAGE}" = "prepare" ]; then
    echo "[PREPARE] ${ARM}"
    pixi run python -m imitation_experiments.planner.prepare_gr00t_dataset \
        --config-dir "${CONF}" --config-name "prepare_${ARM}"
fi

if [ "${STAGE}" = "all" ] || [ "${STAGE}" = "train" ]; then
    echo "[TRAIN] ${ARM} (group ${WANDB_GROUP})"
    pixi run -e gr00t python -m imitation_experiments.planner.train_gr00t_head \
        --config-dir "${CONF}" --config-name "train_${ARM}" \
        "wandb.group=${WANDB_GROUP}" "$@"
fi
