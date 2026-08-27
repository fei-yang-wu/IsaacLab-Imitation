#!/usr/bin/env bash
# Resolve every arm offline, so a typo in an arm's overrides is found before
# any GPU is asked for. Plans only; nothing is submitted.
#
#   ./plan_all.sh
#   ARMS="recon_endpoint" ./plan_all.sh
set -uo pipefail
CAMPAIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${CAMPAIGN_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

# One seed per arm at the 2B screen, matching the design study.
SEEDS="${SEEDS:-0}"

if [[ "${ARMS:-}" == "" ]]; then
    mapfile -t selected < <(pixi run python -c "
import yaml
campaign = yaml.safe_load(open('${CAMPAIGN_DIR}/campaign.yaml'))
print('\n'.join(campaign['arms']))
")
    ARMS="${selected[*]}"
fi

failed=()
for arm in ${ARMS}; do
    for seed in ${SEEDS}; do
        if ! "${CAMPAIGN_DIR}/submit.sh" "${arm}" "${seed}" >/dev/null 2>&1; then
            failed+=("${arm} seed${seed}")
            printf '[FAIL] %s seed%s\n' "${arm}" "${seed}"
        else
            printf '[ok]   %s seed%s\n' "${arm}" "${seed}"
        fi
    done
done
if [[ "${#failed[@]}" -gt 0 ]]; then
    printf '\n[FATAL] %d plan(s) did not resolve:\n' "${#failed[@]}"
    printf '  %s\n' "${failed[@]}"
    exit 1
fi
echo "[INFO] every plan resolved"
