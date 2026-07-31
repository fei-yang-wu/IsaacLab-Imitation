#!/usr/bin/env bash
set -euo pipefail

# Guarded ICE dependency chain for the latent training-budget curve -- the
# matched partner of submit_fb670_budget_curve_ice.sh:
#   collect -> train[medium,large] -> progressive eval[medium,large] -> aggregate
#
# Every budget, size, seed, demonstration count, milestone cadence and
# evaluation setting is identical to the FB-670 chain. Only the command
# interface (258-value DiffSR latent vs 670-value explicit packet) and its
# frozen low-level tracker differ, so the two curves compare directly.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

DRY_RUN="${DRY_RUN:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/interface_baselines/lafan1_latent_budget_curve_20260730}"
LATENT_CHECKPOINT="${LATENT_CHECKPOINT:-logs/downloaded_checkpoints/lafan1_latent_deterministic_5b_seed0/model_step_4525129728.pt}"
LATENT_SKILL_CHECKPOINT="${LATENT_SKILL_CHECKPOINT:-logs/downloaded_checkpoints/lafan1_latent_deterministic_5b_seed0/skill_encoder/latest.pt}"
LATENT_SHA256="${LATENT_SHA256:-785f5327f2356f4a301ac39fc435b78379e9c5a73293c450deb483dd7c188f7c}"
SKILL_SHA256="${SKILL_SHA256:-5c84ff7261c5a3aca732e370ca39f889d68a5d39fb498fa9fde72c653eb264ea}"
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
for sha in "${LATENT_SHA256}" "${SKILL_SHA256}"; do
    [[ "${sha}" =~ ^[0-9a-f]{64}$ ]] || {
        echo "[ERROR] Checkpoint SHAs must be full SHA-256 values." >&2
        exit 2
    }
done

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
    --stage latent_curve
    --latent-curve-output-root "${OUTPUT_ROOT}"
    --latent-curve-checkpoint "${LATENT_CHECKPOINT}"
    --latent-curve-skill-checkpoint "${LATENT_SKILL_CHECKPOINT}"
    --latent-curve-checkpoint-sha256 "${LATENT_SHA256}"
    --latent-curve-skill-sha256 "${SKILL_SHA256}"
)

print_stage() {
    local mode="$1" dependency="$2" array="$3" time_limit="$4"
    printf '[STAGE] mode=%s dependency=%s array=%s time=%s\n' \
        "${mode}" "${dependency:-none}" "${array:-none}" "${time_limit}"
    printf '[CMD]'; printf ' %q' "${base_cmd[@]}" --latent-curve-mode "${mode}"
    printf '\n'
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
    echo "[INFO] Matched partner of the FB-670 curve (same budgets, same protocol)."
    echo "[INFO] DRY_RUN=1; not contacting ICE."
    exit 0
fi

echo "[INFO] Verifying latent tracker + encoder on ICE..."
remote_shas="$(ssh "${CLUSTER_LOGIN}" "sha256sum '${REMOTE_PROJECT_ROOT}/${LATENT_CHECKPOINT}' '${REMOTE_PROJECT_ROOT}/${LATENT_SKILL_CHECKPOINT}'")"
printf '%s\n' "${remote_shas}"
grep -q "^${LATENT_SHA256} " <<<"${remote_shas}" || {
    echo "[ERROR] Remote latent tracker SHA mismatch." >&2
    exit 2
}
grep -q "^${SKILL_SHA256} " <<<"${remote_shas}" || {
    echo "[ERROR] Remote skill encoder SHA mismatch." >&2
    exit 2
}
echo "[PASS] Remote latent checkpoints verified."

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
            "${base_cmd[@]}" --latent-curve-mode "${mode}" 2>&1
    )"
    printf '%s\n' "${output}" >&2
    job_id="$(printf '%s\n' "${output}" | awk '/Submitted batch job [0-9]+/{value=$NF} END{print value}')"
    if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] Could not parse Slurm job ID for ${mode}." >&2
        exit 2
    fi
    printf '%s' "${job_id}"
}

collect_id="$(submit_stage collect "" "" 04:00:00 latcurve-collect)"
train_id="$(submit_stage train "afterok:${collect_id}" "${SIZE_ARRAY}" 12:00:00 latcurve-train)"
eval_id="$(submit_stage eval "afterok:${train_id}" "${SIZE_ARRAY}" 15:59:00 latcurve-eval)"
aggregate_id="$(submit_stage aggregate "afterok:${eval_id}" "" 01:00:00 latcurve-agg)"

record="$(mktemp)"
{
    printf '{\n'
    printf '  "schema_version": 1,\n'
    printf '  "submitted_at_utc": "%s",\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '  "study": "latent_training_budget_curve",\n'
    printf '  "matched_partner_study": "fb670_training_budget_curve",\n'
    printf '  "output_root": "%s",\n' "${OUTPUT_ROOT}"
    printf '  "repo_head": "%s",\n' "${repo_head}"
    printf '  "working_tree_diff_sha256": "%s",\n' "${diff_sha}"
    printf '  "latent_low_level_checkpoint": "%s",\n' "${LATENT_CHECKPOINT}"
    printf '  "latent_low_level_sha256": "%s",\n' "${LATENT_SHA256}"
    printf '  "skill_encoder_checkpoint": "%s",\n' "${LATENT_SKILL_CHECKPOINT}"
    printf '  "skill_encoder_sha256": "%s",\n' "${SKILL_SHA256}"
    printf '  "motion": "walk1_subject1",\n'
    printf '  "command_contract": {"values": 258, "composition": "z256 + sin_cos phase", "hold_steps": 10},\n'
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

echo "[PASS] latent budget-curve chain submitted: collect=${collect_id} train=${train_id} eval=${eval_id} aggregate=${aggregate_id}"
echo "[PASS] Submission record: ${REMOTE_OUTPUT_ROOT}/cluster_submission.json"
