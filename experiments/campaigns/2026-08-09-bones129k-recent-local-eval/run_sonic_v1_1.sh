#!/usr/bin/env bash
set -uo pipefail

# Run NVIDIA's public SONIC v1.1 tracker on the exact rank block used by this
# campaign. The actor is native SONIC encoder -> FSQ -> decoder, not an IPMD
# policy, so it uses the dedicated release evaluator.
#
#   MODE=print ./run_sonic_v1_1.sh
#   MODE=smoke ./run_sonic_v1_1.sh
#   ./run_sonic_v1_1.sh
#   ./run_sonic_v1_1.sh --report

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
while [[ ! -f "${REPO_ROOT}/pixi.toml" ]]; do
    [[ "${REPO_ROOT}" != "/" ]] || { echo "[FATAL] repository root not found" >&2; exit 2; }
    REPO_ROOT="$(dirname "${REPO_ROOT}")"
done
cd "${REPO_ROOT}"

MODE="${MODE:-run}"
[[ "${1:-}" == "--report" ]] && MODE=report

CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/logs/downloaded_checkpoints/nvidia_GEAR_SONIC_9c0ff22/sonic_v1_1/last.pt}"
EXPECTED_CHECKPOINT_SHA256="af24831ae59424a0cf92cb56e9bb6dc1a59ab859fd055ba13187e9e6f0a59f43"
REFERENCE_ARRAYS="${REFERENCE_ARRAYS:-${REPO_ROOT}/data/bones_seed/reference_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
PERSIST_ID="${PERSIST_ID:-bones_seed_sonic_full_129785@e714bbff}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/logs/bones129k_recent_ice_local_eval}"
OUTPUT_JSON="${OUTPUT_ROOT}/scoreboard4096/sonic_v1_1/sonic.json"
NUM_ENVS=4096
RANK_START=12288
RANK_END=16383
MAX_STEPS="${MAX_STEPS:-10000}"

if [[ "${MODE}" == "smoke" ]]; then
    NUM_ENVS=10
    RANK_END=12297
    OUTPUT_JSON="${OUTPUT_ROOT}/sonic_v1_1_smoke10/sonic.json"
fi

RUNTIME_BODY_NAMES="[pelvis,left_hip_roll_link,left_knee_link,left_ankle_roll_link,right_hip_roll_link,right_knee_link,right_ankle_roll_link,torso_link,left_shoulder_roll_link,left_elbow_link,left_wrist_yaw_link,right_shoulder_roll_link,right_elbow_link,right_wrist_yaw_link]"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

validate_result() {
    local path="$1" expected_envs="$2" rank_start="$3" rank_end="$4"
    python3 - "${path}" "${expected_envs}" "${rank_start}" "${rank_end}" \
        "${EXPECTED_CHECKPOINT_SHA256}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
num_envs, rank_start, rank_end = map(int, sys.argv[2:5])
expected_checkpoint_sha = sys.argv[5]
with path.open(encoding="utf-8") as file:
    result = json.load(file)

aggregate = result["aggregate"]
metadata = result["metadata"]
assert result["sonic_version"] == "v1_1", result["sonic_version"]
assert result["actor_spec"]["version"] == "v1_1", result["actor_spec"]
assert result["actor_spec"]["orientation_contract"] == "motion_anchor_ori_heading_mf_nonflat"
assert result["sonic_contract"]["encoder_frame_stride"] == 5
assert result["sonic_contract"]["encoder_orientation_source"] == (
    "expert_anchor_ori_b plus current robot root heading"
)
assert aggregate["num_evaluated_envs"] == num_envs
assert aggregate["done_rate"] == 1.0, aggregate["done_rate"]
assert aggregate["time_out_rate"] == 0.0, aggregate["time_out_rate"]
assert result["stop_reason"] == "all_envs_done", result["stop_reason"]
assert metadata["action_sampling"] == "mode"
assert metadata["randomization_profile"] == "no_push"
assert metadata["randomization_kept"] == {"startup": True, "reset": True, "push": False}
assert metadata["push_perturbation"]["enabled"] is False
assert metadata["reference_start_frame"] == 0
assert metadata["seed"] == 0
assert metadata["termination_contract"]["mode"] == "sonic"
assert set(metadata["termination_contract"]["disabled"]) == {"foot_pos_xyz", "base_too_low"}
ranks = result["trajectory_ranks"]
assert ranks == list(range(rank_start, rank_end + 1)), "trajectory rank order changed"
digest = hashlib.sha256(
    (json.dumps(ranks, separators=(",", ":")) + "\n").encode()
).hexdigest()
assert digest == result["trajectory_ranks_sha256"]
if num_envs == 4096:
    assert digest == "786ef6775930c34179b774cb215e233c3f7b2bb32ef46bb6fc660206324e8285"
checkpoint = Path(result["checkpoint"])
checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
assert checkpoint_sha == expected_checkpoint_sha, checkpoint_sha
allowed = {"anchor_pos", "anchor_ori", "ee_body_pos", "reference_finished", "time_out"}
assert not (set(aggregate["termination_cause_env_counts"]) - allowed)
print(f"OK envs={num_envs} rank_sha256={digest} checkpoint_sha256={checkpoint_sha}")
PY
}

report() {
    if [[ ! -s "${OUTPUT_JSON}" ]]; then
        log "[FATAL] result not found: ${OUTPUT_JSON}"
        return 1
    fi
    validate_result "${OUTPUT_JSON}" "${NUM_ENVS}" "${RANK_START}" "${RANK_END}"
    python3 - "${OUTPUT_JSON}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as file:
    result = json.load(file)
aggregate = result["aggregate"]
metric = result["successful_metrics"]["tracking_mpjpe_mm"]
print(
    "SONIC v1.1: "
    f"{aggregate['completed_env_count']}/{aggregate['num_evaluated_envs']} "
    f"SR={aggregate['completed_tracking_success_rate']:.4f}, "
    f"success-only MPJPE-L={metric['mean']:.2f} mm"
)
PY
}

if [[ "${MODE}" == "report" ]]; then
    report
    exit $?
fi

[[ -s "${CHECKPOINT}" ]] || { log "[FATAL] missing checkpoint: ${CHECKPOINT}"; exit 1; }
[[ -d "${REFERENCE_ARRAYS}" ]] || { log "[FATAL] missing reference arrays: ${REFERENCE_ARRAYS}"; exit 1; }
actual_sha="$(sha256sum "${CHECKPOINT}" | awk '{print $1}')"
[[ "${actual_sha}" == "${EXPECTED_CHECKPOINT_SHA256}" ]] || {
    log "[FATAL] checkpoint SHA-256 mismatch: ${actual_sha}"
    exit 1
}

ranks=()
for ((rank = RANK_START; rank <= RANK_END; rank++)); do ranks+=("${rank}"); done
command=(
    env TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
    HYDRA_FULL_ERROR=1 TORCHDYNAMO_DISABLE=1
    pixi run -e isaaclab python -u
    -m imitation_experiments.lowlevel.evaluate_sonic_release
    --sonic_checkpoint "${CHECKPOINT}"
    --sonic_version v1_1
    --num_envs "${NUM_ENVS}" --steps "${MAX_STEPS}" --seed 0
    --randomization no_push --reference_start_frame 0
    --reset_schedule sequential --trajectory_ranks "${ranks[@]}"
    --termination_contract sonic
    --proprioception_order gravity_last --history_order oldest_first
    --label sonic_v1_1_scoreboard4096 --output_json "${OUTPUT_JSON}"
    --kit_args=--/app/extensions/fsWatcherEnabled=false
    physics=newton_mjwarp
    env.sim.physics.solver_cfg.njmax=320
    env.sim.physics.solver_cfg.nconmax=200
    env.events.push_robot=null
    env.data.manifest=null
    "env.data.reference_arrays_dir=${REFERENCE_ARRAYS}"
    "env.data.persist_id=${PERSIST_ID}"
    env.data.reference_arrays_resident=false
    env.data.reference_arrays_warm_workers=8
    env.data.runtime_cache_device=cpu
    env.data.reference_prefetch_mode=off
    "env.data.runtime_cache_body_names=${RUNTIME_BODY_NAMES}"
)

if [[ "${MODE}" == "print" ]]; then
    printf '[CMD] '
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi
[[ "${MODE}" == "run" || "${MODE}" == "smoke" ]] || {
    log "[FATAL] MODE must be print, smoke, run, or report"
    exit 2
}

mkdir -p "$(dirname "${OUTPUT_JSON}")"
if [[ ! -s "${OUTPUT_JSON}" ]]; then
    log "evaluate SONIC v1.1 on ${NUM_ENVS} environments"
    "${command[@]}" > "${OUTPUT_JSON}.log" 2>&1 || {
        log "[FAIL] evaluator failed; see ${OUTPUT_JSON}.log"
        exit 1
    }
else
    log "skip existing result: ${OUTPUT_JSON}"
fi
if grep -Eq 'overflow.*increase njmax|nefc overflow' "${OUTPUT_JSON}.log"; then
    log "[FAIL] solver constraint buffer overflow in ${OUTPUT_JSON}.log"
    exit 1
fi
report
