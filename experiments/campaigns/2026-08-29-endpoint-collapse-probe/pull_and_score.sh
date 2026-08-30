#!/usr/bin/env bash
# Mirror the ICE suffix-arm metrics and print the dose-response table.
set -euo pipefail
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

LOGIN="${LOGIN:-ice}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/data/endpoint_collapse_probe}"
MIRROR="${MIRROR:-logs/endpoint_collapse_probe/ice_mirror}"
OUTPUT_DIR="${OUTPUT_DIR:-logs/endpoint_collapse_probe/aggregate_ice}"

mkdir -p "${MIRROR}"
for run in $(ssh "${LOGIN}" "ls -1 ${REMOTE_ROOT}"); do
    mkdir -p "${MIRROR}/${run}"
    scp -q "${LOGIN}:${REMOTE_ROOT}/${run}/encoder/metrics.jsonl" \
        "${MIRROR}/${run}/metrics.jsonl"
done

exec pixi run python -m imitation_experiments.capacity.aggregate_window_suffix_arms \
    --runs_dir "${MIRROR}" \
    --output_dir "${OUTPUT_DIR}" \
    "$@"
