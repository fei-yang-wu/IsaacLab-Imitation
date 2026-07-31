#!/usr/bin/env bash
set -euo pipefail

# Guarded ICE dependency chain for the FB-670 training-budget curve:
#   collect -> train[medium,large] -> progressive eval[medium,large] -> aggregate
#
# collect: one 100-env oracle demonstration session, DEMO_ROWS balanced rows.
# train:   one 30k-update flow planner per size (batch 1024) with optimizer-free
#          milestone snapshots every 1k updates.
# eval:    every 2k-update milestone plus best.pt, closed-loop under the
#          rigorous protocol -- frame-0 start, training-time DR and pushes
#          active, base_too_low-only termination, 4096 environments.
# aggregate: MPJPE / survival / fall-count vs updates tables.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

DRY_RUN="${DRY_RUN:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/interface_baselines/lafan1_fb670_budget_curve_20260730}"
FB_CHECKPOINT="${FB_CHECKPOINT:-logs/downloaded_checkpoints/lafan1_fbchunk_5b_seed0/model_step_5000085504.pt}"
FB_SHA256="${FB_SHA256:-681a712ea8635aaaf89f788d3d73d3142dab0b26fbb2bb6ab805d27c805a0bf6}"
REMOTE_PROJECT_ROOT="${REMOTE_PROJECT_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/isaaclab}"
REMOTE_OUTPUT_ROOT="${REMOTE_PROJECT_ROOT}/${OUTPUT_ROOT}"
SIZE_ARRAY="${SIZE_ARRAY:-0-1}"

case "${DRY_RUN}" in
    1|true|TRUE|yes|YES) DRY_RUN=1 ;;
    0|false|FALSE|no|NO) DRY_RUN=0 ;;
    *) echo "[ERROR] DRY_RUN must be boolean, got ${DRY_RUN}." >&2; exit 2 ;;
esac
[[ "${SIZE_ARRAY}" == "0-1" ]] || {
    echo "[ERROR] SIZE_ARRAY must be the exact two-size grid 0-1 (medium,large)." >&2
    exit 2
}
[[ "${FB_SHA256}" =~ ^[0-9a-f]{64}$ ]] || {
    echo "[ERROR] FB_SHA256 must be a full SHA-256." >&2
    exit 2
}

export CLUSTER_LOGIN="${CLUSTER_LOGIN:-login-ice.pace.gatech.edu}"
export CLUSTER_SLURM_SUBMIT_SCRIPT="${CLUSTER_SLURM_SUBMIT_SCRIPT:-pace}"
export CLUSTER_PYTHON_EXECUTABLE="experiments/campaigns/2026-07-23-lafan1-planner-capacity/run_capacity_entry.py"
export CLUSTER_APPEND_DEFAULT_G1_MANIFEST=0
export CLUSTER_SLURM_PARTITION="${CLUSTER_SLURM_PARTITION:-ice-gpu}"
export CLUSTER_SLURM_QOS="${CLUSTER_SLURM_QOS:-coe-ice}"
export CLUSTER_SLURM_GPU_GRES="${CLUSTER_SLURM_GPU_GRES:-gpu:h100:1}"
export CLUSTER_SLURM_CPUS_PER_TASK="${CLUSTER_SLURM_CPUS_PER_TASK:-16}"
export CLUSTER_SLURM_MEM="${CLUSTER_SLURM_MEM:-96G}"
export CLUSTER_GIT_SYNC_FIRST="${CLUSTER_GIT_SYNC_FIRST:-0}"
export CLUSTER_G1_USD_PATH="${CLUSTER_G1_USD_PATH:-repo}"

base_cmd=(./docker/cluster/cluster_interface.sh -c ice_runtime job
    --stage fb670_curve
    --fb670-output-root "${OUTPUT_ROOT}"
    --fb670-checkpoint "${FB_CHECKPOINT}"
    --fb670-checkpoint-sha256 "${FB_SHA256}"
)

print_stage() {
    local mode="$1" dependency="$2" array="$3" time_limit="$4"
    printf '[STAGE] mode=%s dependency=%s array=%s time=%s\n' \
        "${mode}" "${dependency:-none}" "${array:-none}" "${time_limit}"
    printf '[CMD]'; printf ' %q' "${base_cmd[@]}" --fb670-mode "${mode}"; printf '\n'
}

if [[ "${DRY_RUN}" == "1" ]]; then
    print_stage collect "" "" 04:00:00
    print_stage train 'afterok:<collect>' "${SIZE_ARRAY}" 12:00:00
    print_stage eval 'afterok:<train>' "${SIZE_ARRAY}" 15:59:00
    print_stage aggregate 'afterok:<eval>' "" 01:00:00
    echo "[INFO] Motion: walk1_subject1. Demonstrations: 100 envs, 5000 balanced rows."
    echo "[INFO] Training: medium+large, 30k updates each, batch 1024, milestones every 1k."
    echo "[INFO] Eval: every 2k updates + best.pt; frame-0, DR+pushes ON,"
    echo "[INFO]       base_too_low-only termination (0.40 m), 4096 envs, 500 steps."
    echo "[INFO] DRY_RUN=1; not contacting ICE."
    exit 0
fi

echo "[INFO] Verifying FB tracker checkpoint on ICE..."
remote_sha="$(ssh "${CLUSTER_LOGIN}" "sha256sum '${REMOTE_PROJECT_ROOT}/${FB_CHECKPOINT}'" | awk '{print $1}')"
[[ "${remote_sha}" == "${FB_SHA256}" ]] || {
    echo "[ERROR] Remote FB tracker SHA mismatch: ${remote_sha} != ${FB_SHA256}" >&2
    exit 2
}
echo "[PASS] Remote FB tracker verified."

if ssh "${CLUSTER_LOGIN}" test -e "${REMOTE_OUTPUT_ROOT}"; then
    echo "[ERROR] Refusing existing remote output root: ${REMOTE_OUTPUT_ROOT}" >&2
    exit 2
fi

repo_head="$(git rev-parse HEAD)"
diff_sha="$(git diff --binary -- . ':!logs' | sha256sum | awk '{print $1}')"

submit_stage() {
    local mode="$1" dependency="$2" array="$3" time_limit="$4" job_name="$5"
    local output job_id
    output="$(
        env \
            CLUSTER_SLURM_DEPENDENCY="${dependency}" \
            CLUSTER_SLURM_ARRAY="${array}" \
            CLUSTER_SLURM_TIME_LIMIT="${time_limit}" \
            CLUSTER_SLURM_JOB_NAME_PREFIX="${job_name}" \
            "${base_cmd[@]}" --fb670-mode "${mode}" 2>&1
    )"
    printf '%s\n' "${output}" >&2
    job_id="$(printf '%s\n' "${output}" | awk '/Submitted batch job [0-9]+/{value=$NF} END{print value}')"
    if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] Could not parse Slurm job ID for ${mode}." >&2
        exit 2
    fi
    printf '%s' "${job_id}"
}

collect_id="$(submit_stage collect "" "" 04:00:00 fb670-collect)"
train_id="$(submit_stage train "afterok:${collect_id}" "${SIZE_ARRAY}" 12:00:00 fb670-train)"
eval_id="$(submit_stage eval "afterok:${train_id}" "${SIZE_ARRAY}" 15:59:00 fb670-eval)"
aggregate_id="$(submit_stage aggregate "afterok:${eval_id}" "" 01:00:00 fb670-agg)"

record="$(mktemp)"
{
    printf '{\n'
    printf '  "schema_version": 1,\n'
    printf '  "submitted_at_utc": "%s",\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '  "study": "fb670_training_budget_curve",\n'
    printf '  "output_root": "%s",\n' "${OUTPUT_ROOT}"
    printf '  "repo_head": "%s",\n' "${repo_head}"
    printf '  "working_tree_diff_sha256": "%s",\n' "${diff_sha}"
    printf '  "fb_low_level_checkpoint": "%s",\n' "${FB_CHECKPOINT}"
    printf '  "fb_low_level_sha256": "%s",\n' "${FB_SHA256}"
    printf '  "motion": "walk1_subject1",\n'
    printf '  "demonstrations": {"parallel_environments": 100, "balanced_rows": 5000, "random_starts": "0-200", "tracking_terminations": "disabled"},\n'
    printf '  "planner_training": {"sizes": ["medium", "large"], "updates": 30000, "effective_batch_size": 1024, "milestone_interval": 1000, "holdout_trajectory_fraction": 0.2},\n'
    printf '  "evaluation": {"protocol": "frame0_dr_baseonly", "num_envs": 4096, "steps": 500, "fall_height_m": 0.4, "eval_stride_updates": 2000, "includes_best_checkpoint": true},\n'
    printf '  "jobs": {"collect": %s, "train_array": %s, "eval_array": %s, "aggregate": %s}\n' \
        "${collect_id}" "${train_id}" "${eval_id}" "${aggregate_id}"
    printf '}\n'
} > "${record}"
ssh "${CLUSTER_LOGIN}" mkdir -p "${REMOTE_OUTPUT_ROOT}"
scp "${record}" "${CLUSTER_LOGIN}:${REMOTE_OUTPUT_ROOT}/cluster_submission.json"
rm -f "${record}"

echo "[PASS] fb670 budget-curve chain submitted: collect=${collect_id} train=${train_id} eval=${eval_id} aggregate=${aggregate_id}"
echo "[PASS] Submission record: ${REMOTE_OUTPUT_ROOT}/cluster_submission.json"
