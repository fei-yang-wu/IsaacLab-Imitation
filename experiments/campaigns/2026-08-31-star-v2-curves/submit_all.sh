#!/usr/bin/env bash
# Submit every arm's curve evaluation, then retry the ones that die in Kit
# startup.
#
# Kit crashes on this profile roughly half the time within the first ~25 s,
# on any node, with or without a neighbour -- the same signature recorded on
# 2026-08-27. It is not concurrency: two jobs on ONE node produced one crash
# and one success. Retrying is the fix, and a crash is cheap because it happens
# before any scoring work.
#
# `eval_checkpoint_tree.py` skips already-scored cells, so a retry resumes
# rather than restarting.
#
#   ./submit_all.sh              # submit every arm
#   RETRY_ONLY=1 ./submit_all.sh # only resubmit arms whose last job FAILED
set -uo pipefail
CAMPAIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${CAMPAIGN_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"
REMOTE_HOST="${REMOTE_HOST:-ice}"

mapfile -t all < <(pixi run python -c "
import yaml
print('\n'.join(yaml.safe_load(open('${CAMPAIGN_DIR}/campaign.yaml'))['arms']))
")

# Skip ONLY arms with a job in the queue right now. A COMPLETED job is NOT a
# reason to skip: it may have scored a different board, and re-running is cheap
# because `eval_checkpoint_tree.py` skips cells that already have a row. Using
# "has a COMPLETED job" here silently starved two arms of a 4,096-clip run
# after the board changed (2026-08-31).
mapfile -t busy < <(ssh "${REMOTE_HOST}" \
    'squeue -u $USER -h -o "%j" | grep "^star-v2-curves" | sed -E "s/star-v2-curves-(.*)-s0-score/\1/"' 2>/dev/null)

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
