#!/usr/bin/env bash
# Join the per-seed collections into the single directory `prepare` reads.
#
#   ./merge_seeds.sh <fsq64_10b|ln_hold1_10b> [seed ...]     (default: 0 1)
#
# `prepare_gr00t_dataset` globs one `rollout_training_samples/*.pt`, so the
# seeds are symlinked (not copied — these files are tens of GB) under names
# carrying their seed, which also keeps the two seeds' identical file numbering
# from colliding. `summaries/` keeps every seed's summary.json so the merged
# collection's provenance is not just seed 0's.
set -euo pipefail
CAMPAIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${CAMPAIGN_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

ARM="${1:?usage: merge_seeds.sh <arm> [seed ...]}"
shift || true
SEEDS=("${@:-0}")
[ "$#" -eq 0 ] && SEEDS=(0 1)

ROOT="${REPO_ROOT}/logs/planner_10b/${ARM}"
MERGED="${ROOT}/collection_merged"
mkdir -p "${MERGED}/rollout_training_samples" "${MERGED}/summaries"

total=0
for seed in "${SEEDS[@]}"; do
    src="${ROOT}/collection_seed${seed}/rollout_training_samples"
    [ -d "${src}" ] || { echo "missing collection for seed ${seed}: ${src}" >&2; exit 1; }
    count=0
    for file in "${src}"/*.pt; do
        ln -sf "${file}" "${MERGED}/rollout_training_samples/seed${seed}_$(basename "${file}")"
        count=$(( count + 1 ))
    done
    cp -f "${ROOT}/collection_seed${seed}/summary.json" "${MERGED}/summaries/seed${seed}.json"
    echo "[MERGE] seed ${seed}: ${count} sample files"
    total=$(( total + count ))
done
# The newest seed's summary stands in as the merged summary, so the prepare
# step still records a collection_summary_sha256; `summaries/` holds them all.
cp -f "${ROOT}/collection_seed${SEEDS[-1]}/summary.json" "${MERGED}/summary.json"
echo "[MERGE] ${ARM}: ${total} sample files -> ${MERGED}"
