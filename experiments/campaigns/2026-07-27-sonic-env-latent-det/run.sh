#!/usr/bin/env bash
set -euo pipefail

# Thin entrypoint for the 2026-07-27 SONIC-environment latent-deterministic
# screen.
#
#   run.sh local -> local wiring gate      (MODE=print by default)
#   run.sh ice   -> ICE H100 launcher      (MODE=print by default)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GROUP_DIR="${SCRIPT_DIR}/sonic_env_det"
STAGE="${1:-local}"
shift || true

case "${STAGE}" in
    local) exec "${GROUP_DIR}/run_local_wiring_gate.sh" "$@" ;;
    ice) exec "${GROUP_DIR}/submit_sonic_env_latent_det_ice.sh" "$@" ;;
    *)
        echo "[ERROR] STAGE must be local or ice; got '${STAGE}'." >&2
        exit 2
        ;;
esac
