#!/usr/bin/env bash
set -euo pipefail

# Recover only the explicit tasks rejected by the original v1 packet-pin audit.
# The planners/checkpoints are reused in place. The v2 pin measures equality
# inside each 5 Hz publication, then the missing evaluations and aggregate run.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

DRY_RUN="${DRY_RUN:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/interface_baselines/lafan1_enc380_route_capacity_5b_oracle100_progressive_b1024_20260730}"
LOW_LEVEL_CHECKPOINT="${LOW_LEVEL_CHECKPOINT:-/data/resume_store/lafan1_enc380_rootqpos_h10_z256_seed0/model_5b.pt}"
SKILL_CHECKPOINT="${SKILL_CHECKPOINT:-/data/enc380_store/lafan1_enc380_rootqpos_h10_z256_seed0/skill_encoder/checkpoints/latest.pt}"
TRACKER_COMPLETION_RECORD="${TRACKER_COMPLETION_RECORD:-/data/resume_store/lafan1_enc380_rootqpos_h10_z256_seed0/completion.json}"
LOW_LEVEL_SHA256="${LOW_LEVEL_SHA256:-d33fa146f54222848da8b9a92eb5579f5acb8b3a46c484399c906b076c219260}"
SKILL_SHA256="${SKILL_SHA256:-1d530fcb5920112b84bc53dbaddf2b3eb3da13a32a379513d8ee8719bc57d546}"
ORIGINAL_ARRAY_JOB_ID="${ORIGINAL_ARRAY_JOB_ID:-5550601}"
ROOT_ARRAY="${ROOT_ARRAY:-0-11%8}"
REMOTE_PROJECT_ROOT="${REMOTE_PROJECT_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/isaaclab}"
REMOTE_OUTPUT_ROOT="${REMOTE_PROJECT_ROOT}/${OUTPUT_ROOT}"

case "${DRY_RUN}" in
    1|true|TRUE|yes|YES) DRY_RUN=1 ;;
    0|false|FALSE|no|NO) DRY_RUN=0 ;;
    *) echo "[ERROR] DRY_RUN must be boolean, got ${DRY_RUN}." >&2; exit 2 ;;
esac
[[ "${ROOT_ARRAY}" == "0-11%8" ]] || {
    echo "[ERROR] ROOT_ARRAY must cover the exact 12 explicit recovery indices." >&2
    exit 2
}
if [[ "${DRY_RUN}" == "0" && ! "${ORIGINAL_ARRAY_JOB_ID}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] ORIGINAL_ARRAY_JOB_ID must be numeric." >&2
    exit 2
fi

export CLUSTER_LOGIN="${CLUSTER_LOGIN:-login-ice.pace.gatech.edu}"
export CLUSTER_SLURM_SUBMIT_SCRIPT="${CLUSTER_SLURM_SUBMIT_SCRIPT:-pace}"
export CLUSTER_PYTHON_EXECUTABLE=-m imitation_experiments.capacity.run_capacity_entry
export CLUSTER_APPEND_DEFAULT_G1_MANIFEST=0
export CLUSTER_SLURM_PARTITION="${CLUSTER_SLURM_PARTITION:-ice-gpu}"
export CLUSTER_SLURM_QOS="${CLUSTER_SLURM_QOS:-coe-ice}"
export CLUSTER_SLURM_GPU_GRES="${CLUSTER_SLURM_GPU_GRES:-gpu:h100:1}"
export CLUSTER_SLURM_CPUS_PER_TASK="${CLUSTER_SLURM_CPUS_PER_TASK:-16}"
export CLUSTER_SLURM_MEM="${CLUSTER_SLURM_MEM:-96G}"
export CLUSTER_GIT_SYNC_FIRST="${CLUSTER_GIT_SYNC_FIRST:-0}"
export CLUSTER_G1_USD_PATH="${CLUSTER_G1_USD_PATH:-repo}"

base_cmd=(./docker/cluster/cluster_interface.sh -c ice_runtime job
    --stage enc380
    --enc380-low-level-checkpoint "${LOW_LEVEL_CHECKPOINT}"
    --enc380-skill-checkpoint "${SKILL_CHECKPOINT}"
    --enc380-completion-record "${TRACKER_COMPLETION_RECORD}"
    --enc380-output-root "${OUTPUT_ROOT}"
    --enc380-low-level-sha256 "${LOW_LEVEL_SHA256}"
    --enc380-skill-sha256 "${SKILL_SHA256}"
)

if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[STAGE] explicit-recovery dependency=afterany:%s array=%s time=15:59:00\n' \
        "${ORIGINAL_ARRAY_JOB_ID}" "${ROOT_ARRAY}"
    printf '[CMD]'; printf ' %q' "${base_cmd[@]}" --enc380-mode root_recovery; printf '\n'
    echo "[STAGE] aggregate dependency=afterok:<explicit-recovery> time=01:00:00"
    printf '[CMD]'; printf ' %q' "${base_cmd[@]}" --enc380-mode aggregate; printf '\n'
    echo "[INFO] Reuses all existing best.pt planner checkpoints in place."
    echo "[INFO] New packet_encoder_pin_v2 proves publication-time equality."
    echo "[INFO] DRY_RUN=1; not contacting ICE."
    exit 0
fi

if ! ssh "${CLUSTER_LOGIN}" test -d "${REMOTE_OUTPUT_ROOT}"; then
    echo "[ERROR] Recovery source study does not exist: ${REMOTE_OUTPUT_ROOT}" >&2
    exit 2
fi
if ssh "${CLUSTER_LOGIN}" test -e "${REMOTE_OUTPUT_ROOT}/aggregate/results.json"; then
    echo "[ERROR] Refusing to overwrite an existing complete aggregate." >&2
    exit 2
fi

submit_stage() {
    local mode="$1" dependency="$2" array="$3" time_limit="$4" job_name="$5"
    local output job_id
    output="$(
        env \
            CLUSTER_SLURM_DEPENDENCY="${dependency}" \
            CLUSTER_SLURM_ARRAY="${array}" \
            CLUSTER_SLURM_TIME_LIMIT="${time_limit}" \
            CLUSTER_SLURM_JOB_NAME_PREFIX="${job_name}" \
            "${base_cmd[@]}" --enc380-mode "${mode}" 2>&1
    )"
    printf '%s\n' "${output}" >&2
    job_id="$(printf '%s\n' "${output}" | awk '/Submitted batch job [0-9]+/{value=$NF} END{print value}')"
    [[ "${job_id}" =~ ^[0-9]+$ ]] || {
        echo "[ERROR] Could not parse Slurm job ID for ${mode}." >&2
        exit 2
    }
    printf '%s' "${job_id}"
}

root_id="$(submit_stage root_recovery "afterany:${ORIGINAL_ARRAY_JOB_ID}" "${ROOT_ARRAY}" 15:59:00 enc380-root-v2)"
aggregate_id="$(submit_stage aggregate "afterok:${root_id}" "" 01:00:00 enc380-aggregate-v2)"

record="$(mktemp)"
{
    printf '{\n'
    printf '  "schema_version": 1,\n'
    printf '  "submitted_at_utc": "%s",\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '  "reason": "v1 pin compared a held 5 Hz command to a 50 Hz recomputed target",\n'
    printf '  "source_array_job": %s,\n' "${ORIGINAL_ARRAY_JOB_ID}"
    printf '  "explicit_route_tasks": "%s",\n' "${ROOT_ARRAY}"
    printf '  "reuses_existing_planner_checkpoints": true,\n'
    printf '  "pin_v2_contract": "same-call expert packet encoder output equals oracle encoder output",\n'
    printf '  "jobs": {"explicit_recovery_array": %s, "aggregate": %s}\n' \
        "${root_id}" "${aggregate_id}"
    printf '}\n'
} > "${record}"
scp "${record}" "${CLUSTER_LOGIN}:${REMOTE_OUTPUT_ROOT}/root_route_recovery_submission.json"
rm -f "${record}"

echo "[PASS] Explicit recovery submitted: root=${root_id} aggregate=${aggregate_id}"
echo "[PASS] Recovery record: ${REMOTE_OUTPUT_ROOT}/root_route_recovery_submission.json"
