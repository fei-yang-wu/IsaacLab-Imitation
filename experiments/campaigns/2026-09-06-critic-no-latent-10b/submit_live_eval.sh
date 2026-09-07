#!/usr/bin/env bash
# Score critic-no-latent's newest checkpoint mid-flight, on the standard board,
# the same way 2026-09-03-combo-50b does.
#
#   ./submit_live_eval.sh            # submit if the newest checkpoint is unscored
#   DRY_RUN=1 ./submit_live_eval.sh  # print what it would do
#
# The eval arm rebuilds the SAME critic (`critic_channels=[reference]`), or the
# strict state-dict restore fails on the value network's first layer.
set -euo pipefail
CAMPAIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${CAMPAIGN_DIR}" rev-parse --show-toplevel)"
DATA="${DATA:-/home/hice1/fwu91/scratch/Research/IsaacLab/data}"
export DATA
export TRAIN_TREE="${TRAIN_TREE:-${DATA}/critic_no_latent_10b/nolatent_seed0/tracker}"
export LIVE_ROOT="${LIVE_ROOT:-${DATA}/critic_no_latent_live/nolatent_seed0/tracker}"
export LIVE_TREE_ROOT="${LIVE_TREE_ROOT:-/data/critic_no_latent_live}"
export ARM="${ARM:-nolatent_live}"
export EVAL_ARM="${EVAL_ARM:-nolatent}"
exec "${REPO_ROOT}/experiments/campaigns/2026-09-03-combo-50b/submit_live_eval.sh"
