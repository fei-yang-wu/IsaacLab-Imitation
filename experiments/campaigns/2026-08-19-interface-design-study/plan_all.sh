#!/usr/bin/env bash
# Resolve every arm of one tier offline, so a typo in an arm's overrides is
# found before any GPU is asked for. Plans only; nothing is submitted.
#
#   ./plan_all.sh            # tier 1, seed 0
#   TIERS="2 3" ./plan_all.sh
set -uo pipefail
CAMPAIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${CAMPAIGN_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

TIERS="${TIERS:-1}"
# One seed per arm (user decision, 2026-08-19). Every arm is seed 0.
SEEDS="${SEEDS:-0}"

# Tier 4 is DEFERRED by user decision on 2026-08-19: the co-trained arms wait on
# an encoder-from-checkpoint eval path, and phi/z_phi are dropped for now.
# Planning one would freeze a command for an arm that cannot yet be scored.
if [[ " ${TIERS} " == *" 4 "* && "${FORCE_DEFERRED:-0}" != "1" ]]; then
    echo "[FATAL] tier 4 is deferred; re-enable by moving the arm's tier in campaign.yaml," >&2
    echo "        or set FORCE_DEFERRED=1 to plan it anyway." >&2
    exit 2
fi

mapfile -t arms < <(pixi run python -c "
import sys, yaml
campaign = yaml.safe_load(open('${CAMPAIGN_DIR}/campaign.yaml'))
tiers = {int(t) for t in '${TIERS}'.split()}
for name, arm in campaign['arms'].items():
    if int(arm['vars'].get('tier', 1)) in tiers:
        print(name)
")
[[ "${#arms[@]}" -gt 0 ]] || { echo "[FATAL] no arms for tiers '${TIERS}'"; exit 2; }
echo "[INFO] ${#arms[@]} arm(s) in tier(s) ${TIERS}, seeds ${SEEDS}"

failed=()
for arm in "${arms[@]}"; do
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
