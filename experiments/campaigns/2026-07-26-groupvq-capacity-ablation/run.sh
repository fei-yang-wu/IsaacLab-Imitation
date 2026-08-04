#!/usr/bin/env bash
set -euo pipefail

# Thin entrypoint for the 2026-07-26 grouped-VQ capacity ablation.
#
#   run.sh check   -> CPU pre-flight over every (G, C) grid point
#   run.sh local   -> local 10M-frame wiring gate (MODE=print by default)
#   run.sh ice     -> ICE H200 launcher (MODE=print by default)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GROUP_DIR="${SCRIPT_DIR}/groupvq_ablation"
STAGE="${1:-check}"
shift || true

case "${STAGE}" in
    check) exec pixi run python "${GROUP_DIR}/check_groupvq_encoder_grid.py" "$@" ;;
    local) exec "${GROUP_DIR}/run_local_10m_qualification.sh" "$@" ;;
    ice) exec "${GROUP_DIR}/submit_groupvq_capacity_ablation_ice.sh" "$@" ;;
    *)
        echo "[ERROR] STAGE must be check, local, or ice; got '${STAGE}'." >&2
        exit 2
        ;;
esac
