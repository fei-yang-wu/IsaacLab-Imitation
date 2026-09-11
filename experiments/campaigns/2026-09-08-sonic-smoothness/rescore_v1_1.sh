#!/usr/bin/env bash
# Re-score SONIC v1.1 on the canonical board WITH the reference-free smoothness
# metrics, so the target for our smoothness work is a measured number rather
# than a blank.
#
# Why this exists: `evaluate_sonic_release` computed only
# `tracking_acceleration_distance_mps2`, which is an ERROR against the
# reference — a controller that copies a jerky clip scores 0 on it. The
# reference-free pair (`body_acc_mps2`, `body_jerk_mps3`) and the action-space
# measure (`action_delta_l2`) lived only in `evaluate_checkpoint`, so every
# SONIC row on disk has acc and nothing else. Those three were ported across on
# 2026-09-08; this rescore fills them in.
#
# Protocol is the 2026-08-17 paper-metric canon verbatim (same board, ranks,
# termination contract, episode length, Newton backend), with two changes:
# `--sonic_version v1_1` and its own checkpoint.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

CHECKPOINT="${CHECKPOINT:-/mnt/hsstorage/fwu91/sonic_v1_1/last.pt}"
REFERENCE_ARRAYS="${REFERENCE_ARRAYS:-/mnt/hsstorage/fwu91/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
PERSIST_ID="${PERSIST_ID:-bones_seed_sonic_full_129785@e714bbff}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/logs/sonic_release_4096}"
MAX_STEPS="${MAX_STEPS:-10000}"
NUM_ENVS="${NUM_ENVS:-4096}"
ROWS="${ROWS:-clean}"
RUNTIME_BODY_NAMES="[pelvis,left_hip_roll_link,left_knee_link,left_ankle_roll_link,right_hip_roll_link,right_knee_link,right_ankle_roll_link,torso_link,left_shoulder_roll_link,left_elbow_link,left_wrist_yaw_link,right_shoulder_roll_link,right_elbow_link,right_wrist_yaw_link]"

log() { printf '[%s] %s\n' "$(date -Iseconds)" "$*"; }
profile_for() { [[ "$1" == "clean" ]] && echo none || echo no_push; }
out_for() { printf '%s/sonic_v1_1_testbed4096_smooth_rand_%s.json' "${OUTPUT_ROOT}" "$(profile_for "$1")"; }

[[ -s "${CHECKPOINT}" ]] || { log "[FATAL] missing checkpoint: ${CHECKPOINT}"; exit 1; }
mkdir -p "${OUTPUT_ROOT}"

mapfile -t ranks < <(pixi run python -c \
    'from imitation_experiments.evaluation.protocol import TESTBED4096_RANKS
print("\n".join(str(rank) for rank in TESTBED4096_RANKS))')
[[ "${#ranks[@]}" -eq 4096 ]] || { log "[FATAL] registry returned ${#ranks[@]} ranks"; exit 2; }

for row in ${ROWS}; do
    profile="$(profile_for "${row}")"
    out="$(out_for "${row}")"
    [[ -s "${out}" ]] && { log "[SKIP] already scored ${out}"; continue; }
    log "testbed ${row}: randomization ${profile}"
    env TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 \
        HYDRA_FULL_ERROR=1 TORCHDYNAMO_DISABLE=1 \
        pixi run -e isaaclab python -u \
        -m imitation_experiments.lowlevel.evaluate_sonic_release \
        --sonic_checkpoint "${CHECKPOINT}" \
        --sonic_version v1_1 \
        --num_envs "${NUM_ENVS}" --steps "${MAX_STEPS}" --seed 0 \
        --randomization "${profile}" --reference_start_frame 0 \
        --reset_schedule sequential --trajectory_ranks "${ranks[@]}" \
        --termination_contract sonic \
        --proprioception_order gravity_last --history_order oldest_first \
        --label "sonic_v1_1_testbed4096_smooth_rand_${profile}" --output_json "${out}" \
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
        log "[FAIL] ${row}; see ${out}.log"; exit 1; }
    if grep -Eq 'overflow.*increase njmax|nefc overflow' "${out}.log"; then
        log "[FAIL] solver constraint buffer overflow in ${out}.log"; exit 1
    fi
    log "[OK] ${out}"
done
