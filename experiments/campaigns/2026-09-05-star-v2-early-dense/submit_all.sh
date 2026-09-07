#!/usr/bin/env bash
# Submit every arm's early-dense rerun, then retry the ones whose job FAILED.
#
# Training runs kitless, so the Kit startup crash of the evaluation jobs does
# not apply here. A retry is still safe: RLOpt saves under a new run directory
# and the scorer refuses a tree with two run directories, so delete a failed
# arm's output tree before resubmitting it.
#
#   ./submit_all.sh              # submit every arm
#   ARMS="hub g2_mlp" ./submit_all.sh
set -uo pipefail
CAMPAIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${CAMPAIGN_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"
REMOTE_HOST="${REMOTE_HOST:-ice}"

# The loop reads `all`. An ARMS override must be copied into it: an earlier
# version had no branch here at all, so `ARMS=... ./submit_all.sh` looked like
# it worked and silently submitted every arm instead of the eleven asked for
# (2026-09-01).
if [[ -n "${ARMS:-}" ]]; then
    read -r -a all <<< "${ARMS}"
else
    mapfile -t all < <(pixi run python -c "
import yaml
print('\n'.join(yaml.safe_load(open('${CAMPAIGN_DIR}/campaign.yaml'))['arms']))
")
fi

# Skip ONLY arms with a job in the queue right now. A COMPLETED job is NOT a
# reason to skip: it may have scored a different board, and re-running is cheap
# because `eval_checkpoint_tree.py` skips cells that already have a row. Using
# "has a COMPLETED job" here silently starved two arms of a 4,096-clip run
# after the board changed (2026-08-31).
mapfile -t busy < <(ssh "${REMOTE_HOST}" \
    'squeue -u $USER -h -o "%j" | grep "^latent-star-v2-early" | sed -E "s/latent-star-v2-early-(.*)-s0-lowlevel1/\1/"' 2>/dev/null)

skip=" ${busy[*]} "
for arm in "${all[@]}"; do
    [[ "${skip}" == *" ${arm} "* ]] && { printf '[skip] %s\n' "${arm}"; continue; }
    out="$("${CAMPAIGN_DIR}/submit.sh" "${arm}" 0 2>&1)"
    dir="$(echo "${out}" | grep -oE '^\[PLAN\] local dir:.*' | awk '{print $4}')"
    sha="$(echo "${out}" | grep -oE '^PLAN_SHA=.*' | cut -d= -f2)"
    if [[ -z "${dir}" || -z "${sha}" ]]; then printf '[PLANFAIL] %s\n' "${arm}"; continue; fi
    res="$(pixi run python -m imitation_experiments.pipeline.cluster submit \
        --plan "${dir}" --confirm "${sha}" --allow-resubmit 2>&1)"
    job="$(echo "${res}" | grep -oE 'job [0-9]+' | awk '{print $2}' | tail -1)"
    if [[ -z "${job}" ]]; then
        printf '[SUBFAIL] %s: %s\n' "${arm}" "$(echo "${res}" | tail -2 | tr '\n' ' ' | cut -c1-140)"
    else
        printf '[OK] %s job %s\n' "${arm}" "${job}"
    fi
done
