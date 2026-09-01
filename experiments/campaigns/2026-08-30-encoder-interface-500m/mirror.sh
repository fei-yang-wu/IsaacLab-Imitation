#!/usr/bin/env bash
# Pull encoder-interface-500m tracker checkpoints off ICE into the layout
# `eval.sh` reads.
#
# The trainer names a checkpoint by CUMULATIVE frame count, so `model_step_N`
# is the global budget position and `fN/` mirrors that number directly.
#
# This script pulls trackers only. The per-arm ENCODERS stay on ICE: they are
# 0.74 to 1.26 GB each and the workstation pool holds about 16 GB, so `eval.sh`
# streams one encoder at a time and deletes it after scoring.
#
#   ./mirror.sh                 # every arm, final checkpoint only
#   ARMS="prod h1" ./mirror.sh
#   FINAL_ONLY=0 ./mirror.sh    # every milestone checkpoint
set -uo pipefail
CAMPAIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${CAMPAIGN_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

REMOTE_HOST="${REMOTE_HOST:-ice}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/data/encoder_interface_500m}"
MIRROR="${MIRROR:-${REPO_ROOT}/logs/encoder_interface_500m_mirror}"
SEED="${SEED:-0}"
FINAL_ONLY="${FINAL_ONLY:-1}"
FINAL_FRAMES="${FINAL_FRAMES:-500170752}"
ARMS="${ARMS:-prod suffix1 suffix2 suffix5 suffix9 h1 h2 h5 h10}"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

pulled=0
for arm in ${ARMS}; do
    remote_arm="${REMOTE_ROOT}/${arm}_seed${SEED}"
    local_arm="${MIRROR}/${arm}_seed${SEED}"

    mapfile -t remote_ckpts < <(ssh "${REMOTE_HOST}" \
        "ls ${remote_arm}/tracker/*/models/model_step_*.pt 2>/dev/null" 2>/dev/null)
    [[ "${#remote_ckpts[@]}" -gt 0 ]] || { log "[SKIP] ${arm}: nothing on ICE"; continue; }

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
df -h /mnt/hsstorage | tail -1
