#!/usr/bin/env bash
set -uo pipefail

# Regenerate the released-SONIC reference rows that calibrate every paper-facing
# number in this repo. Three runs, all with the released NVIDIA checkpoint:
#
#   canonical_clean  ranks 12288-16383, randomization none   -> headline calibration
#   canonical_robust ranks 12288-16383, randomization no_push -> robustness partner
#   heldout_clean    ranks 20480-24575, randomization none   -> falsification block
#
# The canonical testbed row is scored by `run_testbed_rows.sh` beside this
# script; these three calibrate the legacy block it replaces.
#
# About 12 minutes per run on one RTX PRO 6000. Existing outputs are kept.
#
#   ./run_sonic_reference_rows.sh
#   ROWS="canonical_clean" ./run_sonic_reference_rows.sh
#   ./run_sonic_reference_rows.sh --report

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

CHECKPOINT="${CHECKPOINT:-/mnt/hsstorage/fwu91/sonic_release/last.pt}"
EXPECTED_CHECKPOINT_SHA256="e6bdab3f64a39336b3d41877d4f497d05f58af275f288ec0e6746c283ded8909"
REFERENCE_ARRAYS="${REFERENCE_ARRAYS:-/mnt/hsstorage/fwu91/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
PERSIST_ID="${PERSIST_ID:-bones_seed_sonic_full_129785@e714bbff}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/logs/sonic_release_4096}"
MAX_STEPS="${MAX_STEPS:-10000}"
NUM_ENVS=4096

# row | rank start | rank end | randomization profile
ROWS_TABLE=(
"canonical_clean|12288|16383|none"
"canonical_robust|12288|16383|no_push"
"heldout_clean|20480|24575|none"
)
ROWS="${ROWS:-canonical_clean canonical_robust heldout_clean}"

RUNTIME_BODY_NAMES="[pelvis,left_hip_roll_link,left_knee_link,left_ankle_roll_link,right_hip_roll_link,right_knee_link,right_ankle_roll_link,torso_link,left_shoulder_roll_link,left_elbow_link,left_wrist_yaw_link,right_shoulder_roll_link,right_elbow_link,right_wrist_yaw_link]"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

row_output() {
    local start="$1" end="$2" profile="$3"
    printf '%s/sonic_release_ranks%s_%s_rand_%s.json' \
        "${OUTPUT_ROOT}" "${start}" "${end}" "${profile}"
}

report() {
    local row start end profile out
    for row in ${ROWS}; do
        for candidate in "${ROWS_TABLE[@]}"; do
            [[ "${candidate%%|*}" == "${row}" ]] || continue
            IFS='|' read -r _ start end profile <<<"${candidate}"
        done
        out="$(row_output "${start}" "${end}" "${profile}")"
        [[ -s "${out}" ]] || { log "[MISSING] ${row}: ${out}"; continue; }
        pixi run python -m imitation_experiments.evaluation.summarize_paper_boards "${out}"
    done
}

if [[ "${1:-}" == "--report" ]]; then
    report
    exit $?
fi

[[ -s "${CHECKPOINT}" ]] || { log "[FATAL] missing checkpoint: ${CHECKPOINT}"; exit 1; }
actual_sha="$(sha256sum "${CHECKPOINT}" | awk '{print $1}')"
[[ "${actual_sha}" == "${EXPECTED_CHECKPOINT_SHA256}" ]] || {
    log "[FATAL] checkpoint SHA-256 mismatch: ${actual_sha}"
    exit 1
}
[[ -s "${REFERENCE_ARRAYS}/reference_arrays_manifest.json" ]] || {
    log "[FATAL] reference arrays missing: ${REFERENCE_ARRAYS}"
    exit 2
}
mkdir -p "${OUTPUT_ROOT}"

for row in ${ROWS}; do
    entry=""
    for candidate in "${ROWS_TABLE[@]}"; do
        [[ "${candidate%%|*}" == "${row}" ]] && entry="${candidate}"
    done
    [[ -n "${entry}" ]] || { log "[SKIP] unknown row ${row}"; continue; }
    IFS='|' read -r _ start end profile <<<"${entry}"
    out="$(row_output "${start}" "${end}" "${profile}")"
    [[ -s "${out}" ]] && { log "[SKIP] already scored ${out}"; continue; }

    ranks=()
    for ((rank = start; rank <= end; rank++)); do ranks+=("${rank}"); done

    log "${row}: ranks ${start}-${end}, randomization ${profile}"
    env TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 \
        HYDRA_FULL_ERROR=1 TORCHDYNAMO_DISABLE=1 \
        pixi run -e isaaclab python -u \
        -m imitation_experiments.lowlevel.evaluate_sonic_release \
        --sonic_checkpoint "${CHECKPOINT}" \
        --sonic_version release \
        --num_envs "${NUM_ENVS}" --steps "${MAX_STEPS}" --seed 0 \
        --randomization "${profile}" --reference_start_frame 0 \
        --reset_schedule sequential --trajectory_ranks "${ranks[@]}" \
        --termination_contract sonic \
        --proprioception_order gravity_last --history_order oldest_first \
        --label "sonic_release_${row}" --output_json "${out}" \
        --kit_args=--/app/extensions/fsWatcherEnabled=false \
        physics=newton_mjwarp \
        env.sim.physics.solver_cfg.njmax=320 \
        env.sim.physics.solver_cfg.nconmax=200 \
        env.events.push_robot=null \
        env.data.manifest=null \
        "env.data.reference_arrays_dir=${REFERENCE_ARRAYS}" \
        "env.data.persist_id=${PERSIST_ID}" \
        env.data.reference_arrays_resident=false \
        env.data.reference_arrays_warm_workers=8 \
        env.data.runtime_cache_device=cuda:0 \
        env.data.reference_prefetch_mode=off \
        env.data.macro_cache_device=cuda:0 \
        "env.data.runtime_cache_body_names=${RUNTIME_BODY_NAMES}" > "${out}.log" 2>&1 || {
        log "[FAIL] ${row}; see ${out}.log"
        exit 1
    }
    if grep -Eq 'overflow.*increase njmax|nefc overflow' "${out}.log"; then
        log "[FAIL] solver constraint buffer overflow in ${out}.log"
        exit 1
    fi
done

report
