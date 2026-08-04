#!/usr/bin/env bash
# Front door for the deterministic-latent end-to-end chain.
# Renders the full plan by default (dry run); pass dry_run=false to execute.
# Any extra arguments are Hydra overrides, e.g.:
#   ./run.sh stages='[pretrain,low_level]' seed=1
#   ./run.sh dry_run=false
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${REPO_ROOT}"
pixi run python -m imitation_experiments.pipeline.run_latent_e2e \
    --config-dir "${SCRIPT_DIR}/conf" \
    --config-name det_latent_e2e \
    "$@"
