#!/usr/bin/env bash
set -euo pipefail

# Fill the missing seed-0 capacity cells for the two auxiliary routes:
#   tasks 0-2: H30 root_qpos planner, tiny/small/large
#   tasks 3-5: direct root_qpos-tracker planner, tiny/small/large
# Both routes reuse existing provenance-bound data; no simulator collection.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

DRY_RUN="${DRY_RUN:-1}"
SOURCE_STUDY_ROOT="${SOURCE_STUDY_ROOT:-logs/interface_baselines/lafan1_enc380_route_capacity_5b_oracle100_progressive_b1024_20260730}"
H30_OUTPUT_ROOT="${H30_OUTPUT_ROOT:-logs/interface_baselines/lafan1_enc380_h30_temporal_medium_seed0_20260730}"
PURE_ROOT_OUTPUT_ROOT="${PURE_ROOT_OUTPUT_ROOT:-${SOURCE_STUDY_ROOT}/pure_root_qpos_tracker}"
PURE_ROOT_SAMPLES="${PURE_ROOT_SAMPLES:-logs/interface_baselines/lafan1_interface_capacity_controlled/oracle_baselines/root_qpos/oracle_demonstrations/rollout_training_samples}"
PURE_ROOT_SAMPLE_SHA256="${PURE_ROOT_SAMPLE_SHA256:-1374eb44647098e1e0a1da21e89ad1edab02fd58195a434a2dc1a6b3b809a3b0}"
LOW_LEVEL_CHECKPOINT="${LOW_LEVEL_CHECKPOINT:-/data/resume_store/lafan1_enc380_rootqpos_h10_z256_seed0/model_5b.pt}"
SKILL_CHECKPOINT="${SKILL_CHECKPOINT:-/data/enc380_store/lafan1_enc380_rootqpos_h10_z256_seed0/skill_encoder/checkpoints/latest.pt}"
TRACKER_COMPLETION_RECORD="${TRACKER_COMPLETION_RECORD:-/data/resume_store/lafan1_enc380_rootqpos_h10_z256_seed0/completion.json}"
LOW_LEVEL_SHA256="${LOW_LEVEL_SHA256:-d33fa146f54222848da8b9a92eb5579f5acb8b3a46c484399c906b076c219260}"
SKILL_SHA256="${SKILL_SHA256:-1d530fcb5920112b84bc53dbaddf2b3eb3da13a32a379513d8ee8719bc57d546}"
REMOTE_PROJECT_ROOT="${REMOTE_PROJECT_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/isaaclab}"
CELL_ARRAY="${CELL_ARRAY:-0-5%6}"

case "${DRY_RUN}" in
    1|true|TRUE|yes|YES) DRY_RUN=1 ;;
    0|false|FALSE|no|NO) DRY_RUN=0 ;;
    *) echo "[ERROR] DRY_RUN must be boolean, got ${DRY_RUN}." >&2; exit 2 ;;
esac
[[ "${CELL_ARRAY}" == "0-5%6" ]] || {
    echo "[ERROR] CELL_ARRAY must be the exact missing-cell grid 0-5%6." >&2
    exit 2
}

export CLUSTER_LOGIN="${CLUSTER_LOGIN:-login-ice.pace.gatech.edu}"
export CLUSTER_SLURM_SUBMIT_SCRIPT="${CLUSTER_SLURM_SUBMIT_SCRIPT:-pace}"
export CLUSTER_PYTHON_EXECUTABLE=source/imitation_experiments/imitation_experiments/capacity/run_capacity_entry.py
export CLUSTER_APPEND_DEFAULT_G1_MANIFEST=0
export CLUSTER_SLURM_PARTITION="${CLUSTER_SLURM_PARTITION:-ice-gpu}"
export CLUSTER_SLURM_QOS="${CLUSTER_SLURM_QOS:-coe-ice}"
export CLUSTER_SLURM_GPU_GRES="${CLUSTER_SLURM_GPU_GRES:-gpu:h100:1}"
export CLUSTER_SLURM_CPUS_PER_TASK="${CLUSTER_SLURM_CPUS_PER_TASK:-16}"
export CLUSTER_SLURM_MEM="${CLUSTER_SLURM_MEM:-96G}"
export CLUSTER_SLURM_TIME_LIMIT="${CLUSTER_SLURM_TIME_LIMIT:-15:59:00}"
export CLUSTER_SLURM_ARRAY="${CELL_ARRAY}"
export CLUSTER_SLURM_JOB_NAME_PREFIX="${CLUSTER_SLURM_JOB_NAME_PREFIX:-aux-capacity-s0}"
export CLUSTER_GIT_SYNC_FIRST="${CLUSTER_GIT_SYNC_FIRST:-0}"
export CLUSTER_G1_USD_PATH="${CLUSTER_G1_USD_PATH:-repo}"

cmd=(
    ./docker/cluster/cluster_interface.sh -c ice_runtime job
    --stage auxiliary_capacity
    --enc380-low-level-checkpoint "${LOW_LEVEL_CHECKPOINT}"
    --enc380-skill-checkpoint "${SKILL_CHECKPOINT}"
    --enc380-completion-record "${TRACKER_COMPLETION_RECORD}"
    --enc380-low-level-sha256 "${LOW_LEVEL_SHA256}"
    --enc380-skill-sha256 "${SKILL_SHA256}"
    --enc380-source-study-root "${SOURCE_STUDY_ROOT}"
    --enc380-h30-output-root "${H30_OUTPUT_ROOT}"
    --pure-root-samples-dir "${PURE_ROOT_SAMPLES}"
    --pure-root-output-root "${PURE_ROOT_OUTPUT_ROOT}"
    --pure-root-sample-sha256 "${PURE_ROOT_SAMPLE_SHA256}"
)

if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[CMD]'; printf ' %q' "${cmd[@]}"; printf '\n'
    echo "[INFO] Array 0-2: H30 tiny/small/large."
    echo "[INFO] Array 3-5: pure-root tiny/small/large."
    echo "[INFO] Seed 0 only; updates 10k/20k/50k; batch 1024."
    echo "[INFO] DRY_RUN=1; not contacting ICE."
    exit 0
fi

remote_h30_manifest="${REMOTE_PROJECT_ROOT}/${H30_OUTPUT_ROOT}/demonstrations/root_qpos_h30/materialization_manifest.json"
remote_pure_sample="${REMOTE_PROJECT_ROOT}/${PURE_ROOT_SAMPLES}/sample_step_000000.pt"
ssh "${CLUSTER_LOGIN}" test -f "${remote_h30_manifest}" || {
    echo "[ERROR] Missing ICE H30 materialization: ${remote_h30_manifest}" >&2
    exit 2
}
remote_rows="$(
    ssh "${CLUSTER_LOGIN}" "jq -r '.row_count' '${remote_h30_manifest}'"
)"
[[ "${remote_rows}" == "4864" ]] || {
    echo "[ERROR] ICE H30 materialization has ${remote_rows} rows, expected 4864." >&2
    exit 2
}
remote_sample_sha="$(
    ssh "${CLUSTER_LOGIN}" "sha256sum '${remote_pure_sample}'" | awk '{print $1}'
)"
[[ "${remote_sample_sha}" == "${PURE_ROOT_SAMPLE_SHA256}" ]] || {
    echo "[ERROR] ICE pure-root sample SHA mismatch: ${remote_sample_sha}" >&2
    exit 2
}

for spec in tiny:10000 small:20000 large:50000; do
    size="${spec%%:*}"
    updates="${spec##*:}"
    h30_dir="${REMOTE_PROJECT_ROOT}/${H30_OUTPUT_ROOT}/planner/${size}/seed0/planner_oracle_u${updates}_b1024"
    pure_dir="${REMOTE_PROJECT_ROOT}/${PURE_ROOT_OUTPUT_ROOT}/${size}/seed0/planner_oracle_u${updates}_b1024"
    for target in "${h30_dir}" "${pure_dir}"; do
        if ssh "${CLUSTER_LOGIN}" test -e "${target}"; then
            echo "[ERROR] Refusing existing ICE output: ${target}" >&2
            exit 2
        fi
    done
done

output="$("${cmd[@]}" 2>&1)"
printf '%s\n' "${output}"
job_id="$(
    printf '%s\n' "${output}" |
        awk '/Submitted batch job [0-9]+/{value=$NF} END{print value}'
)"
[[ "${job_id}" =~ ^[0-9]+$ ]] || {
    echo "[ERROR] Could not parse ICE job ID." >&2
    exit 2
}

record="$(mktemp)"
{
    printf '{\n'
    printf '  "schema_version": 1,\n'
    printf '  "submitted_at_utc": "%s",\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '  "seed": 0,\n'
    printf '  "sizes": ["tiny", "small", "large"],\n'
    printf '  "routes": ["h30_root_qpos", "pure_root_qpos_tracker"],\n'
    printf '  "updates_by_size": {"tiny": 10000, "small": 20000, "large": 50000},\n'
    printf '  "batch_size": 1024,\n'
    printf '  "h30_reused_rows": 4864,\n'
    printf '  "pure_root_sample_sha256": "%s",\n' "${PURE_ROOT_SAMPLE_SHA256}"
    printf '  "array": "%s",\n' "${CELL_ARRAY}"
    printf '  "job_id": %s\n' "${job_id}"
    printf '}\n'
} > "${record}"
ssh "${CLUSTER_LOGIN}" mkdir -p \
    "${REMOTE_PROJECT_ROOT}/${SOURCE_STUDY_ROOT}"
scp "${record}" \
    "${CLUSTER_LOGIN}:${REMOTE_PROJECT_ROOT}/${SOURCE_STUDY_ROOT}/auxiliary_capacity_seed0_submission.json"
rm -f "${record}"

echo "[PASS] Submitted missing H30/pure-root seed-0 capacity cells: ${job_id}"
