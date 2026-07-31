#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

DRY_RUN="${DRY_RUN:-1}"
SAMPLES_DIR="${SAMPLES_DIR:-logs/interface_baselines/lafan1_interface_capacity_controlled/oracle_baselines/root_qpos/oracle_demonstrations/rollout_training_samples}"
OUTPUT_DIR="${OUTPUT_DIR:-logs/interface_baselines/lafan1_enc380_route_capacity_5b_oracle100_progressive_b1024_20260730/pure_root_qpos_tracker/medium/seed0/planner_oracle_u30000_b1024}"
SAMPLE_SHA256="${SAMPLE_SHA256:-1374eb44647098e1e0a1da21e89ad1edab02fd58195a434a2dc1a6b3b809a3b0}"
REMOTE_PROJECT_ROOT="${REMOTE_PROJECT_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/isaaclab}"

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
export CLUSTER_SLURM_JOB_NAME_PREFIX="${CLUSTER_SLURM_JOB_NAME_PREFIX:-rootqpos-planner}"
export CLUSTER_GIT_SYNC_FIRST="${CLUSTER_GIT_SYNC_FIRST:-0}"
export CLUSTER_G1_USD_PATH="${CLUSTER_G1_USD_PATH:-repo}"

cmd=(
    ./docker/cluster/cluster_interface.sh -c ice_runtime job
    --stage pure_root_planner
    --pure-root-samples-dir "${SAMPLES_DIR}"
    --pure-root-output-dir "${OUTPUT_DIR}"
    --pure-root-sample-sha256 "${SAMPLE_SHA256}"
)

if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[CMD]'; printf ' %q' "${cmd[@]}"; printf '\n'
    echo "[INFO] DRY_RUN=1; not contacting ICE."
    exit 0
fi

remote_sample="${REMOTE_PROJECT_ROOT}/${SAMPLES_DIR}/sample_step_000000.pt"
remote_output="${REMOTE_PROJECT_ROOT}/${OUTPUT_DIR}"
remote_sha="$(ssh "${CLUSTER_LOGIN}" "sha256sum '${remote_sample}'" | awk '{print $1}')"
[[ "${remote_sha}" == "${SAMPLE_SHA256}" ]] || {
    echo "[ERROR] ICE sample is missing or has the wrong SHA: ${remote_sample}" >&2
    exit 2
}
if ssh "${CLUSTER_LOGIN}" test -e "${remote_output}"; then
    echo "[ERROR] Refusing existing ICE output: ${remote_output}" >&2
    exit 2
fi

output="$("${cmd[@]}" 2>&1)"
printf '%s\n' "${output}"
job_id="$(printf '%s\n' "${output}" | awk '/Submitted batch job [0-9]+/{value=$NF} END{print value}')"
[[ "${job_id}" =~ ^[0-9]+$ ]] || {
    echo "[ERROR] Could not parse ICE job ID." >&2
    exit 2
}
echo "[PASS] Submitted direct root_qpos planner training: ${job_id}"
