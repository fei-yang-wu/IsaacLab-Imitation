#!/usr/bin/env bash
# Plan one arm of the star-v2 curve-eval campaign through the cluster control plane.
# This wrapper NEVER submits: it prints a frozen plan and its PLAN_SHA, and the
# control plane prints the separate `submit --plan ... --confirm <PLAN_SHA>`
# command. The W&B project is g1-bs-ablation and the group is latent-star-v2.
set -euo pipefail
CAMPAIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${CAMPAIGN_DIR}" rev-parse --show-toplevel)"
ARM="${1:?usage: submit.sh <arm> <seed> [plan args...]}"
SEED="${2:?usage: submit.sh <arm> <seed> [plan args...]}"
shift 2
cd "${REPO_ROOT}"
exec pixi run python -m imitation_experiments.pipeline.cluster plan \
    --campaign "${CAMPAIGN_DIR}/campaign.yaml" --arm "${ARM}" --seed "${SEED}" "$@"
