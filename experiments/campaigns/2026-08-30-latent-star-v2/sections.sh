#!/usr/bin/env bash
# Print the campaign as its three paper sections, with per-arm status and --
# once rows are scored -- success rate and both MPJPE columns.
#
# Live state comes from ICE: `squeue`/`sacct` for job states and the tracker
# trees for how deep each arm has trained. Pass LOCAL=1 to skip the cluster and
# read a mirrored tree instead.
#
#   ./sections.sh
#   LOCAL=1 ./sections.sh          # no cluster access
#   AT_FRAMES= ./sections.sh      # deepest row per arm instead of the 2B screen
set -uo pipefail
CAMPAIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${CAMPAIGN_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

REMOTE_HOST="${REMOTE_HOST:-ice}"
REMOTE_DATA="${REMOTE_DATA:-/home/hice1/fwu91/scratch/Research/IsaacLab/data/latent_star_v2}"
EVAL_DIR="${EVAL_DIR:-${REPO_ROOT}/logs/latent_star_v2_eval}"
MIRROR="${MIRROR:-${REPO_ROOT}/logs/latent_star_v2_mirror}"
SEED="${SEED:-0}"
# The screen budget every scored column is pinned to. Arms train at different
# speeds, so an unpinned table would compare different checkpoints.
AT_FRAMES="${AT_FRAMES:-2000486400}"

args=(--campaign "${CAMPAIGN_DIR}/campaign.yaml" --eval-dir "${EVAL_DIR}" --seed "${SEED}")
[[ -n "${AT_FRAMES}" ]] && args+=(--at-frames "${AT_FRAMES}")

if [[ "${LOCAL:-0}" == "1" ]]; then
    args+=(--tree-root "${MIRROR}")
else
    states="$(mktemp)"; frames="$(mktemp)"
    trap 'rm -f "${states}" "${frames}"' EXIT
    # squeue carries live states; sacct adds the terminal ones (COMPLETED,
    # FAILED) that have already left the queue.
    ssh "${REMOTE_HOST}" '
        squeue -u $USER -h -o "%j %T" | grep "^latent-star-v2"
        sacct -u $USER -S now-3days -X -o JobName%60,State --noheader \
            | grep "^ *latent-star-v2" | awk "{print \$1, \$2}"
    ' > "${states}" 2>/dev/null
    ssh "${REMOTE_HOST}" "
        for d in ${REMOTE_DATA}/*_seed${SEED}; do
            a=\$(basename \$d _seed${SEED})
            last=\$(find \$d/tracker -name 'model_step_*.pt' 2>/dev/null \
                | grep -oE '[0-9]+\.pt' | sed 's/\.pt//' | sort -n | tail -1)
            echo \"\$a \${last:-}\"
        done
    " > "${frames}" 2>/dev/null
    args+=(--slurm-states "${states}" --frames-file "${frames}")
fi

exec pixi run python -m imitation_experiments.reporting.ablation_sections "${args[@]}"
