#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${MODE:-print}"
LOCAL_QUALIFICATION_ROOT="${LOCAL_QUALIFICATION_ROOT:-}"
PROFILE_FILE="${TRAINING_PROFILE:-${SCRIPT_DIR}/training_profile.h200.pending.env}"
REQUIRED_ARMS=(
    continuous_ae vqvae fsq_recon sonic_fsq_pg cvae
    deterministic gaussian categorical gumbel_multicat gumbel fsq vq
)

case "${MODE}" in
    print) echo "[INFO] Printing H200 commands without opening the qualification gate." ;;
    validate|submit)
        if [[ -z "${LOCAL_QUALIFICATION_ROOT}" ]]; then
            echo "[ERROR] Set LOCAL_QUALIFICATION_ROOT to the completed local_10m run." >&2
            exit 2
        fi
        ;;
    *) echo "[ERROR] MODE must be print, validate, or submit; got ${MODE}." >&2; exit 2 ;;
esac

if [[ "${MODE}" != "print" ]]; then
    for arm in "${REQUIRED_ARMS[@]}"; do
        record="${LOCAL_QUALIFICATION_ROOT}/${arm}/qualification.json"
        if [[ ! -f "${record}" ]]; then
            echo "[ERROR] Missing local qualification for ${arm}: ${record}" >&2
            exit 2
        fi
        pixi run python -c \
            'import json,sys; p=sys.argv[1]; d=json.load(open(p)); assert d.get("passed") is True, (p,d.get("failures"))' \
            "${record}"
    done
    # shellcheck disable=SC1090
    source "${PROFILE_FILE}"
    if [[ "${PROFILE_APPROVED:-0}" != "1" ]]; then
        echo "[ERROR] H200 profile is not approved: ${PROFILE_FILE}" >&2
        exit 2
    fi
    echo "[INFO] All 12 local 10M qualifications passed and the H200 profile is approved."
fi

CHILD_MODE="${MODE}"
if [[ "${MODE}" == "validate" ]]; then
    CHILD_MODE=print
    echo "[INFO] Validation mode: printing the gated commands without submitting."
fi

MODE="${CHILD_MODE}" \
CONFIRM_SUBMIT=lafan1-latent-ablation \
TRAINING_PROFILE="${PROFILE_FILE}" \
"${SCRIPT_DIR}/submit_lafan1_reconstruction_ablation_ice.sh"
MODE="${CHILD_MODE}" \
CONFIRM_SUBMIT=lafan1-latent-ablation \
TRAINING_PROFILE="${PROFILE_FILE}" \
"${SCRIPT_DIR}/submit_lafan1_diffsr_bottleneck_ablation_ice.sh"
