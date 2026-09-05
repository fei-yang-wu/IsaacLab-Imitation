#!/usr/bin/env bash
# Score direct-affine-phi's newest checkpoint mid-flight, on the standard
# board, the same way 2026-09-03-combo-50b does.
#
#   ./submit_live_eval.sh            # submit if the newest checkpoint is unscored
#   DRY_RUN=1 ./submit_live_eval.sh  # print what it would do
#
# The arm's own encoder is the command source, so the eval arm sets
# `command_mode: phi`; the encoder path stays on the real tree, not the
# milestone-layout one. The tracker tree grows a run directory per resume,
# which `score_tree.milestone_checkpoints` refuses, so the newest checkpoint
# is relinked into /data/direct_affine_phi64_live and read with --final_only.
set -euo pipefail
CAMPAIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${CAMPAIGN_DIR}" rev-parse --show-toplevel)"
DATA="${DATA:-/home/hice1/fwu91/scratch/Research/IsaacLab/data}"
export DATA
export TRAIN_TREE="${TRAIN_TREE:-${DATA}/direct_affine_phi64/phi_lstm_seed0/tracker}"
export LIVE_ROOT="${LIVE_ROOT:-${DATA}/direct_affine_phi64_live/phi_lstm_seed0/tracker}"
export LIVE_TREE_ROOT="${LIVE_TREE_ROOT:-/data/direct_affine_phi64_live}"
export ARM="${ARM:-phi_lstm_live}"
export EVAL_ARM="${EVAL_ARM:-phi_lstm}"
exec "${REPO_ROOT}/experiments/campaigns/2026-09-03-combo-50b/submit_live_eval.sh"
