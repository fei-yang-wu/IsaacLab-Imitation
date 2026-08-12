#!/usr/bin/env bash
# Run the full EC closed-loop evaluation grid (default Pixi environment).
# Usage: run_ec_eval.sh [hydra overrides...]   e.g. 'arms=...' 'rtc_variants=[false]'
set -euo pipefail
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"
CONF="${REPO_ROOT}/experiments/campaigns/2026-08-11-gr00t-language10-full/conf"

pixi run python -m imitation_experiments.evaluation.eval_gr00t_ec \
    --config-dir "${CONF}" --config-name eval_ec "$@"
