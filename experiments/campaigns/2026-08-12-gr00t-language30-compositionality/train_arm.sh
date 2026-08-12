#!/usr/bin/env bash
# Train one 30-motion GR00T head arm (gr00t Pixi environment).
# Usage: train_arm.sh <z256|explicit|fsq64> [hydra overrides...]
set -euo pipefail
if [ $# -lt 1 ]; then
    echo "Usage: $0 <z256|explicit|fsq64> [hydra overrides...]" >&2
    exit 1
fi
ARM="$1"
shift
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"
CONF="${REPO_ROOT}/experiments/campaigns/2026-08-12-gr00t-language30-compositionality/conf"
if [ ! -f "${CONF}/train_${ARM}.yaml" ]; then
    echo "Unknown arm ${ARM}: no ${CONF}/train_${ARM}.yaml" >&2
    exit 1
fi

pixi run -e gr00t python -m imitation_experiments.planner.train_gr00t_head \
    --config-dir "${CONF}" --config-name "train_${ARM}" "$@"
