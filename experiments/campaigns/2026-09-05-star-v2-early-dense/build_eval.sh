#!/usr/bin/env bash
# Generate the curve-evaluation campaign for the early-dense trees.
#
# The interface fields come from the ORIGINAL training campaign (identical to
# this rerun by construction), the checkpoint trees from this rerun's output
# root, and the encoder files from the archive the rerun itself bound. Rows
# land in /data/eval/star_v2_early_curves on ICE, apart from the 200M-grid
# corpus, so the two runs are never mixed in one directory.
#
#   ./build_eval.sh            # writes eval_campaign.yaml
#   ./submit_eval.sh <arm> 0   # plan one arm's scoring job
set -euo pipefail
CAMPAIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${CAMPAIGN_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"
pixi run python -m imitation_experiments.reporting.build_curve_eval_campaign \
    --campaign experiments/campaigns/2026-08-30-latent-star-v2/campaign.yaml \
    --tree-root /storage/ice-shared/vip-vwt/scratch-fwu91/latent_star_v2_early \
    --encoder-root /storage/ice-shared/vip-vwt/scratch-fwu91/archived_data/latent_star_v2_checkpoints \
    --eval-root /data/eval/star_v2_early_curves \
    --cache-root /storage/ice-shared/vip-vwt/scratch-fwu91/isaac_cache_early_curves \
    --name star-v2-early-curves --wandb-group latent-star-v2-early \
    --out "${CAMPAIGN_DIR}/eval_campaign.yaml"
