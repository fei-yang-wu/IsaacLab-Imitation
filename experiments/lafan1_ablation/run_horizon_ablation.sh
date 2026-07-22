#!/usr/bin/env bash
# Horizon ablation W in {1,5,10}; one planner per trajectory.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"
# shellcheck source=lib/run_trajectory_planners.sh
source "${SCRIPT_DIR}/lib/run_trajectory_planners.sh"
# shellcheck source=lib/run_latent_interface.sh
source "${SCRIPT_DIR}/lib/run_latent_interface.sh"
# shellcheck source=lib/run_chunk_interface.sh
source "${SCRIPT_DIR}/lib/run_chunk_interface.sh"

WINDOWS="${WINDOWS:-1 5 10}"
INTERFACES="${INTERFACES:-latent_cont latent_fsq ee_chunk wb_chunk}"
TABLE="horizon_ablation"

resolve_budget_knobs
prepare_trajectory_manifests
mkdir -p "${OUTPUT_ROOT}/${TABLE}/tables"

for window in ${WINDOWS}; do
    print_matrix_plan "${TABLE}" "${window}"
    for iface in ${INTERFACES}; do
        case "${iface}" in
            latent_cont|latent_fsq) run_latent_interface "${TABLE}" "${window}" "${iface}" ;;
            ee_chunk|wb_chunk) run_chunk_interface "${TABLE}" "${window}" "${iface}" ;;
            *)
                log "ERROR: unknown interface ${iface}"
                exit 2
                ;;
        esac
    done
done

if [[ "${SKIP_PLANNERS:-0}" == "1" ]]; then
    log "SKIP_PLANNERS=1; skipping table aggregation"
else
    run_cmd "${PYTHON_CMD[@]}" "${SCRIPT_DIR}/aggregate_results.py" \
        --root "${OUTPUT_ROOT}/${TABLE}" \
        --output_dir "${OUTPUT_ROOT}/${TABLE}/tables" \
        --table-name horizon_ablation
fi

log "Horizon ablation complete under ${OUTPUT_ROOT}/${TABLE}"
