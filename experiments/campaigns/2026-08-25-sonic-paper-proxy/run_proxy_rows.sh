#!/usr/bin/env bash
set -uo pipefail

# Score the released NVIDIA SONIC checkpoint on the SONIC-paper-facing proxy
# population, so a released-checkpoint row can face a NAMED column of SONIC
# Table 2 instead of an unnamed block of our corpus.
#
# Board `sonic_proxy_testrep4096_v1`: 4,096 clips whose Table 2 main-category
# mixture matches the paper's test-repetition split. Definition and rationale:
# `imitation_experiments.evaluation.sonic_paper_proxy` and the campaign README.
#
#   clean   randomization none    -> the row that faces the paper
#   robust  randomization no_push -> its randomization partner
#
# About 12 minutes per row on one RTX PRO 6000. Existing outputs are kept.
#
#   ./run_proxy_rows.sh
#   ROWS="clean" ./run_proxy_rows.sh
#   ./run_proxy_rows.sh --report

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/mnt/hsstorage/fwu91}"
REFERENCE_ARRAYS="${REFERENCE_ARRAYS:-/mnt/hsstorage/fwu91/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
PERSIST_ID="${PERSIST_ID:-bones_seed_sonic_full_129785@e714bbff}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/logs/sonic_paper_proxy}"
MAX_STEPS="${MAX_STEPS:-10000}"
NUM_ENVS="${NUM_ENVS:-4096}"
BOARD="${BOARD:-sonic_proxy_testrep4096_v1}"

# checkpoint key | directory under CHECKPOINT_ROOT | --sonic_version | SHA-256
#
# `sonic_release` carries the six-layer action decoder
# `[2048, 2048, 1024, 1024, 512, 512]`, i.e. the paper's 16M rung.
# `sonic_v1_1` carries Table S1's eight-layer
# `[4096, 4096, 2048, 2048, 1024, 1024, 512, 512]`, i.e. the paper's 42M
# architecture, and needs the v1.1 heading-only anchor contract. Both configs
# were read from the public HF repo on 2026-08-25.
CHECKPOINTS_TABLE=(
"release|sonic_release|release|e6bdab3f64a39336b3d41877d4f497d05f58af275f288ec0e6746c283ded8909"
"v1_1|sonic_v1_1|v1_1|af24831ae59424a0cf92cb56e9bb6dc1a59ab859fd055ba13187e9e6f0a59f43"
)
CHECKPOINTS="${CHECKPOINTS:-release v1_1}"

# row | randomization profile
ROWS_TABLE=(
"clean|none"
"robust|no_push"
)
ROWS="${ROWS:-clean robust}"

RUNTIME_BODY_NAMES="[pelvis,left_hip_roll_link,left_knee_link,left_ankle_roll_link,right_hip_roll_link,right_knee_link,right_ankle_roll_link,torso_link,left_shoulder_roll_link,left_elbow_link,left_wrist_yaw_link,right_shoulder_roll_link,right_elbow_link,right_wrist_yaw_link]"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

# $1 checkpoint key, $2 randomization profile. The `release` names are kept
# verbatim so the 2026-08-25 rows scored before v1.1 was added still resolve.
row_output() {
    if [[ "$1" == "release" ]]; then
        printf '%s/sonic_release_%s_rand_%s.json' "${OUTPUT_ROOT}" "${BOARD}" "$2"
    else
        printf '%s/sonic_%s_%s_rand_%s.json' "${OUTPUT_ROOT}" "$1" "${BOARD}" "$2"
    fi
}

profile_for() {
    local candidate
    for candidate in "${ROWS_TABLE[@]}"; do
        [[ "${candidate%%|*}" == "$1" ]] && { printf '%s' "${candidate##*|}"; return 0; }
    done
    return 1
}

report() {
    local key row profile out
    for key in ${CHECKPOINTS}; do
        for row in ${ROWS}; do
            profile="$(profile_for "${row}")" || continue
            out="$(row_output "${key}" "${profile}")"
            [[ -s "${out}" ]] || { log "[MISSING] ${key} ${row}: ${out}"; continue; }
            printf '%-12s %-7s ' "${key}" "${row}"
            pixi run python -m imitation_experiments.evaluation.summarize_paper_boards "${out}"
        done
    done
}

if [[ "${1:-}" == "--report" ]]; then
    report
    exit $?
fi

[[ -s "${REFERENCE_ARRAYS}/reference_arrays_manifest.json" ]] || {
    log "[FATAL] reference arrays missing: ${REFERENCE_ARRAYS}"
    exit 2
}
mkdir -p "${OUTPUT_ROOT}"

# The board is the authority on which clips are scored; read it from the
# registry so the launcher cannot drift from the frozen tuple.
mapfile -t RANKS < <(pixi run python -c "
from imitation_experiments.evaluation.protocol import BOARDS
for case in BOARDS['${BOARD}'].cases:
    print(case.trajectory_rank)
") || { log "[FATAL] could not read board ${BOARD}"; exit 3; }
[[ "${#RANKS[@]}" -eq "${NUM_ENVS}" ]] || {
    log "[FATAL] board ${BOARD} holds ${#RANKS[@]} cases, expected ${NUM_ENVS}"
    exit 3
}

for key in ${CHECKPOINTS}; do
    entry=""
    for candidate in "${CHECKPOINTS_TABLE[@]}"; do
        [[ "${candidate%%|*}" == "${key}" ]] && entry="${candidate}"
    done
    [[ -n "${entry}" ]] || { log "[SKIP] unknown checkpoint ${key}"; continue; }
    IFS='|' read -r _ ckpt_dir version expected_sha <<<"${entry}"
    checkpoint="${CHECKPOINT_ROOT}/${ckpt_dir}/last.pt"
    [[ -s "${checkpoint}" ]] || { log "[SKIP] missing checkpoint ${checkpoint}"; continue; }
    if [[ -n "${expected_sha}" ]]; then
        actual_sha="$(sha256sum "${checkpoint}" | awk '{print $1}')"
        [[ "${actual_sha}" == "${expected_sha}" ]] || {
            log "[FATAL] ${key} SHA-256 mismatch: ${actual_sha}"
            exit 1
        }
    else
        log "${key}: SHA-256 $(sha256sum "${checkpoint}" | awk '{print $1}') (unpinned)"
    fi

    for row in ${ROWS}; do
        profile="$(profile_for "${row}")" || { log "[SKIP] unknown row ${row}"; continue; }
        out="$(row_output "${key}" "${profile}")"
        [[ -s "${out}" ]] && { log "[SKIP] already scored ${out}"; continue; }

        log "${key} ${row}: board ${BOARD}, ${#RANKS[@]} clips, randomization ${profile}"
        env TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 \
            HYDRA_FULL_ERROR=1 TORCHDYNAMO_DISABLE=1 \
            pixi run -e isaaclab python -u \
            -m imitation_experiments.lowlevel.evaluate_sonic_release \
            --sonic_checkpoint "${checkpoint}" \
            --sonic_version "${version}" \
            --num_envs "${NUM_ENVS}" --steps "${MAX_STEPS}" --seed 0 \
            --randomization "${profile}" --reference_start_frame 0 \
            --reset_schedule sequential --trajectory_ranks "${RANKS[@]}" \
            --termination_contract sonic \
            --proprioception_order gravity_last --history_order oldest_first \
            --label "sonic_${key}_${BOARD}_${row}" --output_json "${out}" \
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
            "env.data.runtime_cache_body_names=${RUNTIME_BODY_NAMES}" > "${out}.log" 2>&1
        rc=$?
        # Kit shutdown can mask a Python traceback behind exit 0, so the written
        # result file is the real success test, not `rc`.
        if (( rc != 0 )) || [[ ! -s "${out}" ]]; then
            log "[FAIL] ${key} ${row} exit ${rc}: $(tail -3 "${out}.log" | tr '\n' ' ')"
            continue
        fi
        if grep -Eq 'overflow.*increase njmax|nefc overflow' "${out}.log"; then
            log "[FAIL] solver constraint buffer overflow in ${out}.log"
            continue
        fi
    done
done

report
