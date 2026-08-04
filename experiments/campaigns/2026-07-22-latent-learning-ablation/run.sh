#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

command="${1:-help}"
shift || true

case "${command}" in
    local)
        exec env MODE="${MODE:-print}" \
            "${REPO_ROOT}/experiments/campaigns/2026-07-22-latent-learning-ablation/latent_ablation/run_lafan1_local_10m_qualification.sh" \
            "$@"
        ;;
    h200)
        exec env MODE="${MODE:-print}" \
            "${REPO_ROOT}/experiments/campaigns/2026-07-22-latent-learning-ablation/latent_ablation/submit_all_h200_after_local_qualification.sh" \
            "$@"
        ;;
    help|-h|--help)
        echo "Usage: $0 <local|h200>"
        echo "  local  Print or run the twelve-arm local 10M qualification gate."
        echo "  h200   Print, validate, or submit the gated H200 campaign."
        echo "Set MODE and the paths documented in this campaign's README."
        ;;
    *)
        echo "[ERROR] Unknown command: ${command}" >&2
        exit 2
        ;;
esac

