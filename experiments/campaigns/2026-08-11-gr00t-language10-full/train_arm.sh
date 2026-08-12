#!/usr/bin/env bash
# Train one GR00T head arm (gr00t Pixi environment).
# Usage: train_arm.sh <chunk_mocap|chunk_rollout|z256_mocap|z256_rollout|fsq64_mocap|fsq64_rollout> [extra hydra overrides...]
set -euo pipefail
if [ $# -lt 1 ]; then
    echo "Usage: $0 <arm> [hydra overrides...]" >&2
    exit 1
fi
ARM="$1"
shift
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"
CONF="${REPO_ROOT}/experiments/campaigns/2026-08-11-gr00t-language10-full/conf"
if [ ! -f "${CONF}/train_${ARM}.yaml" ]; then
    echo "Unknown arm ${ARM}: no ${CONF}/train_${ARM}.yaml" >&2
    exit 1
fi

pixi run -e gr00t python -m imitation_experiments.planner.train_gr00t_head \
    --config-dir "${CONF}" --config-name "train_${ARM}" "$@"
