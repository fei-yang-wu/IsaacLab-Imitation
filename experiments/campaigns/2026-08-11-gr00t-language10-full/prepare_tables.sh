#!/usr/bin/env bash
# Build the four GR00T training tables (default Pixi environment).
set -euo pipefail
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"
CONF="${REPO_ROOT}/experiments/campaigns/2026-08-11-gr00t-language10-full/conf"

for name in prepare_z256_rollout prepare_fsq64_rollout prepare_z256_mocap prepare_fsq64_mocap; do
    echo "== ${name}"
    pixi run python -m imitation_experiments.planner.prepare_gr00t_dataset \
        --config-dir "${CONF}" --config-name "${name}"
done
