#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${REPO_ROOT}"

pixi run python -m imitation_experiments.evaluation.plot_tracking_comparison \
  --input "${SCRIPT_DIR}/artifacts/sonic_latent_shared4096_curve.csv" \
  --output "${SCRIPT_DIR}/artifacts/sonic_latent_shared4096_curve.png"
