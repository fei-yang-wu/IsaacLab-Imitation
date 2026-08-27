#!/usr/bin/env bash
# Pull interface-design-study checkpoints off ICE into the layout `eval.sh`
# reads.
#
# The trainer writes `tracker/<timestamp>_wandb-<run id>/models/model_step_N.pt`.
# The evaluator wants `tracker/fN/models/model_step_N.pt`, because a checkpoint
# tree is addressed by its TRUE cumulative frame count. This campaign runs one
# unchained segment per arm, so `model_step_N` already IS the cumulative count
# and no renaming arithmetic is needed -- do NOT copy this assumption to a
# chained campaign, where the per-segment step counter restarts.
#
# Only `latest.pt` is pulled for the encoder: `best.pt` is another 416 MB and
# the study always evaluates the frozen final encoder.
#
#   ./mirror.sh                     # every arm that has anything to pull
#   ARMS="ctrl" ./mirror.sh
#   FINAL_ONLY=1 ./mirror.sh        # skip milestones, take the 2B checkpoint
set -uo pipefail
CAMPAIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${CAMPAIGN_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

REMOTE_HOST="${REMOTE_HOST:-ice}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/data/interface_design_study}"
MIRROR="${MIRROR:-${REPO_ROOT}/logs/interface_design_study_mirror}"
SEED="${SEED:-0}"
FINAL_ONLY="${FINAL_ONLY:-0}"
FINAL_FRAMES="${FINAL_FRAMES:-2000289792}"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

if [[ "${ARMS:-}" == "" ]]; then
    mapfile -t selected < <(pixi run python -c "
import yaml
c = yaml.safe_load(open('experiments/campaigns/2026-08-19-interface-design-study/campaign.yaml'))
for n, a in c['arms'].items():
    if int(a['vars'].get('tier', 1)) != 4:
        print(n)
")
    ARMS="${selected[*]}"
fi

pulled=0
for arm in ${ARMS}; do
    remote_arm="${REMOTE_ROOT}/${arm}_seed${SEED}"
    local_arm="${MIRROR}/${arm}_seed${SEED}"

    # Encoder, once.
    if [[ ! -s "${local_arm}/encoder/checkpoints/latest.pt" ]]; then
        if ssh "${REMOTE_HOST}" "[ -s ${remote_arm}/encoder/checkpoints/latest.pt ]" 2>/dev/null; then
            mkdir -p "${local_arm}/encoder/checkpoints"
            if rsync -a "${REMOTE_HOST}:${remote_arm}/encoder/checkpoints/latest.pt" \
                     "${local_arm}/encoder/checkpoints/latest.pt"; then
                log "[enc] ${arm}"
                pulled=$((pulled+1))
            else
                log "[FAIL] ${arm} encoder"
            fi
        fi
    fi

    mapfile -t remote_ckpts < <(ssh "${REMOTE_HOST}" \
        "ls ${remote_arm}/tracker/*/models/model_step_*.pt 2>/dev/null" 2>/dev/null)
    [[ "${#remote_ckpts[@]}" -gt 0 ]] || continue

    for path in "${remote_ckpts[@]}"; do
        frames="$(basename "${path}" .pt)"; frames="${frames#model_step_}"
        [[ "${frames}" =~ ^[0-9]+$ ]] || continue
        if [[ "${FINAL_ONLY}" == "1" && "${frames}" != "${FINAL_FRAMES}" ]]; then
            continue
        fi
        dest="${local_arm}/tracker/f${frames}/models/model_step_${frames}.pt"
        [[ -s "${dest}" ]] && continue
        mkdir -p "$(dirname "${dest}")"
        if rsync -a --partial "${REMOTE_HOST}:${path}" "${dest}"; then
            log "[ckpt] ${arm} f${frames}"
            pulled=$((pulled+1))
        else
            log "[FAIL] ${arm} f${frames}"
            rm -f "${dest}"
        fi
    done
done

log "[INFO] pulled ${pulled} file(s); mirror at ${MIRROR}"
du -sh "${MIRROR}" 2>/dev/null | tail -1
df -h /mnt/hsstorage | tail -1
