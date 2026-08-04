#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PAPER_WORKFLOW_STATE="staging"

usage() {
    echo "Usage: $0 <status|phase4-lafan1|phase5-bones-seed|bundle> [args...]"
}

require_ready() {
    if [[ "${PAPER_WORKFLOW_STATE}" != "ready" ]]; then
        echo "[ERROR] The paper-facing workflow is ${PAPER_WORKFLOW_STATE}, not ready." >&2
        echo "[ERROR] Read experiments/paper/README.md and wiki/current-status.md." >&2
        return 2
    fi
}

command="${1:-status}"
shift || true

case "${command}" in
    status)
        echo "paper_workflow_state=${PAPER_WORKFLOW_STATE}"
        echo "protocol=wiki/causal-interface-paper-plan.md"
        ;;
    phase4-lafan1)
        require_ready
        exec env DRY_RUN="${DRY_RUN:-1}" \
            "${REPO_ROOT}/experiments/paper/submit_phase4_no_language_skynet.sh" \
            "$@"
        ;;
    phase5-bones-seed)
        require_ready
        exec env DRY_RUN="${DRY_RUN:-1}" \
            "${REPO_ROOT}/experiments/paper/submit_bones_seed_multiseed_pipeline_skynet.sh" \
            "$@"
        ;;
    bundle)
        require_ready
        exec pixi run python \
            "${REPO_ROOT}/experiments/paper/build_paper_release_bundle.py" \
            "$@"
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        usage >&2
        echo "[ERROR] Unknown command: ${command}" >&2
        exit 2
        ;;
esac

