#!/usr/bin/env bash
# Pull smooth-ablation-5b checkpoints off ICE into the layout `eval.sh` reads.
#
# The trainer names checkpoints by CUMULATIVE frame count
# (`cumulative_env_frames`), so `model_step_N` is the global budget position
# even across chained segments; `fN/` mirrors that number directly.
#
#   ./mirror.sh                 # every arm
#   ARMS="base sigma" ./mirror.sh
#   FINAL_ONLY=1 ./mirror.sh    # only the 5B checkpoint
set -uo pipefail
CAMPAIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${CAMPAIGN_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

REMOTE_HOST="${REMOTE_HOST:-ice}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/data/smooth_ablation_5b}"
MIRROR="${MIRROR:-${REPO_ROOT}/logs/smooth_ablation_5b_mirror}"
SEED="${SEED:-0}"
FINAL_ONLY="${FINAL_ONLY:-0}"
FINAL_FRAMES="${FINAL_FRAMES:-5000232960}"
ARMS="${ARMS:-base energy sigma feetacc_weak ar0}"

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
