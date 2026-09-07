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

# Single seed per arm, read from each arm's `submit_seed` in campaign.yaml
# (every v2 arm declares seed 0). Override with SEEDS to force one.
if [[ "${ARMS:-}" == "" ]]; then
    mapfile -t selected < <(pixi run python -c "
import yaml
campaign = yaml.safe_load(open('${CAMPAIGN_DIR}/campaign.yaml'))
print('\n'.join(campaign['arms']))
")
    ARMS="${selected[*]}"
fi

arm_seed() {
    pixi run python -c "
import yaml
c = yaml.safe_load(open('${CAMPAIGN_DIR}/campaign.yaml'))
merged = {**c['vars'], **c['arms']['$1'].get('vars', {})}
print(merged.get('submit_seed', 0))
"
}

failed=()
for arm in ${ARMS}; do
    for seed in ${SEEDS:-$(arm_seed "${arm}")}; do
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
