#!/usr/bin/env bash
set -euo pipefail

# Guarded ICE dependency chain for the FB-670-via-latent-tracker diagnostic:
#   pin[medium,large] -> eval[medium,large] (afterok:pin) -> aggregate
#
# pin: packet_source=expert reproduces the true oracle command through the
#      frozen encoder + latent tracker path. A real defect here (permutation,
#      wrong feature normalization, wrong frame split) fails loudly and the
#      eval array below never launches -- no planner compute is spent on a
#      broken pipeline.
# eval: the already-trained FB-670 planner (medium, large; every milestone +
#       best.pt) driving the FROZEN LATENT tracker instead of its own FB
#       tracker. No retraining -- reuses run_fb670_budget_curve.sh's
#       checkpoints and run_latent_budget_curve.sh's frozen tracker/encoder.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

DRY_RUN="${DRY_RUN:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/interface_baselines/lafan1_fb670_via_latent_tracker_20260730}"
FB670_STUDY_ROOT="${FB670_STUDY_ROOT:-logs/interface_baselines/lafan1_fb670_budget_curve_20260730}"
LATENT_CHECKPOINT="${LATENT_CHECKPOINT:-logs/downloaded_checkpoints/lafan1_latent_deterministic_5b_seed0/model_step_4525129728.pt}"
LATENT_SKILL_CHECKPOINT="${LATENT_SKILL_CHECKPOINT:-logs/downloaded_checkpoints/lafan1_latent_deterministic_5b_seed0/skill_encoder/latest.pt}"
REMOTE_PROJECT_ROOT="${REMOTE_PROJECT_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/isaaclab}"
REMOTE_OUTPUT_ROOT="${REMOTE_PROJECT_ROOT}/${OUTPUT_ROOT}"
REMOTE_FB670_STUDY_ROOT="${REMOTE_PROJECT_ROOT}/${FB670_STUDY_ROOT}"
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

export CLUSTER_LOGIN="${CLUSTER_LOGIN:-login-ice.pace.gatech.edu}"
export CLUSTER_SLURM_SUBMIT_SCRIPT="${CLUSTER_SLURM_SUBMIT_SCRIPT:-pace}"
export CLUSTER_PYTHON_EXECUTABLE=source/imitation_experiments/imitation_experiments/capacity/run_capacity_entry.py
export CLUSTER_APPEND_DEFAULT_G1_MANIFEST=0
export CLUSTER_SLURM_PARTITION="${CLUSTER_SLURM_PARTITION:-ice-gpu}"
export CLUSTER_SLURM_QOS="${CLUSTER_SLURM_QOS:-coe-ice}"
export CLUSTER_SLURM_GPU_GRES="${CLUSTER_SLURM_GPU_GRES:-gpu:h100:1}"
export CLUSTER_SLURM_CPUS_PER_TASK="${CLUSTER_SLURM_CPUS_PER_TASK:-16}"
export CLUSTER_SLURM_MEM="${CLUSTER_SLURM_MEM:-96G}"
export CLUSTER_GIT_SYNC_FIRST="${CLUSTER_GIT_SYNC_FIRST:-0}"
export CLUSTER_G1_USD_PATH="${CLUSTER_G1_USD_PATH:-repo}"

base_cmd=(./docker/cluster/cluster_interface.sh -c ice_runtime job
    --stage fb670_via_latent_tracker
    --fb670-route-output-root "${OUTPUT_ROOT}"
    --fb670-route-train-root "${FB670_STUDY_ROOT}"
    --fb670-route-latent-checkpoint "${LATENT_CHECKPOINT}"
    --fb670-route-skill-checkpoint "${LATENT_SKILL_CHECKPOINT}"
)

print_stage() {
    local mode="$1" dependency="$2" array="$3" time_limit="$4"
    printf '[STAGE] mode=%s dependency=%s array=%s time=%s\n' \
        "${mode}" "${dependency:-none}" "${array:-none}" "${time_limit}"
    printf '[CMD]'; printf ' %q' "${base_cmd[@]}" --fb670-route-mode "${mode}"
    printf '\n'
}

if [[ "${DRY_RUN}" == "1" ]]; then
    print_stage pin "" "${SIZE_ARRAY}" 01:00:00
    print_stage eval 'afterok:<pin>' "${SIZE_ARRAY}" 15:59:00
    print_stage aggregate 'afterok:<eval>' "" 01:00:00
    echo "[INFO] Route: FB-670 planner (medium,large; already trained) -> frozen"
    echo "[INFO]        DEFAULT skill encoder -> frozen latent tracker."
    echo "[INFO] pin uses packet_source=expert (true oracle packet); must show"
    echo "[INFO]      near-zero falls and expert_pin_latent_mse < 1e-8, or the"
    echo "[INFO]      eval array below is blocked by its afterok dependency."
    echo "[INFO] eval: every 2k-update milestone + best.pt, same 4096-env"
    echo "[INFO]       frame0_dr_baseonly protocol as the two other curves."
    echo "[INFO] Reuses FB670_STUDY_ROOT=${FB670_STUDY_ROOT} checkpoints -- no retrain."
    echo "[INFO] DRY_RUN=1; not contacting ICE."
    exit 0
fi

echo "[INFO] Verifying FB-670 study root and latent checkpoints exist on ICE..."
ssh "${CLUSTER_LOGIN}" test -d "${REMOTE_FB670_STUDY_ROOT}/medium/seed0/planner_u30000_b1024/checkpoints" || {
    echo "[ERROR] Missing remote FB-670 medium checkpoints under ${REMOTE_FB670_STUDY_ROOT}." >&2
    exit 2
}
ssh "${CLUSTER_LOGIN}" test -d "${REMOTE_FB670_STUDY_ROOT}/large/seed0/planner_u30000_b1024/checkpoints" || {
    echo "[ERROR] Missing remote FB-670 large checkpoints under ${REMOTE_FB670_STUDY_ROOT}." >&2
    exit 2
}
ssh "${CLUSTER_LOGIN}" test -f "${REMOTE_PROJECT_ROOT}/${LATENT_CHECKPOINT}" || {
    echo "[ERROR] Missing remote latent tracker checkpoint." >&2
    exit 2
}
echo "[PASS] Remote FB-670 checkpoints and latent tracker present."

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
            "${base_cmd[@]}" --fb670-route-mode "${mode}" 2>&1
    )"
    printf '%s\n' "${output}" >&2
    job_id="$(printf '%s\n' "${output}" | awk '/Submitted batch job [0-9]+/{value=$NF} END{print value}')"
    if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] Could not parse Slurm job ID for ${mode}." >&2
        exit 2
    fi
    printf '%s' "${job_id}"
}

pin_id="$(submit_stage pin "" "${SIZE_ARRAY}" 01:00:00 fb670route-pin)"
eval_id="$(submit_stage eval "afterok:${pin_id}" "${SIZE_ARRAY}" 15:59:00 fb670route-eval)"
aggregate_id="$(submit_stage aggregate "afterok:${eval_id}" "" 01:00:00 fb670route-agg)"

record="$(mktemp)"
{
    printf '{\n'
    printf '  "schema_version": 1,\n'
    printf '  "submitted_at_utc": "%s",\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '  "study": "fb670_via_latent_tracker_diagnostic",\n'
    printf '  "purpose": "isolate tracker from interface: FB-670 planner prediction routed through the frozen DEFAULT skill encoder into the frozen latent tracker",\n'
    printf '  "reused_planner_study_root": "%s",\n' "${FB670_STUDY_ROOT}"
    printf '  "output_root": "%s",\n' "${OUTPUT_ROOT}"
    printf '  "repo_head": "%s",\n' "${repo_head}"
    printf '  "working_tree_diff_sha256": "%s",\n' "${diff_sha}"
    printf '  "latent_low_level_checkpoint": "%s",\n' "${LATENT_CHECKPOINT}"
    printf '  "skill_encoder_checkpoint": "%s",\n' "${LATENT_SKILL_CHECKPOINT}"
    printf '  "motion": "walk1_subject1",\n'
    printf '  "evaluation": {"protocol": "frame0_dr_baseonly", "num_envs": 4096, "steps": 500, "fall_height_m": 0.4, "eval_stride_updates": 2000, "includes_best_checkpoint": true, "pin_gate": "packet_source=expert must reproduce oracle before planner rows run"},\n'
    printf '  "jobs": {"pin_array": %s, "eval_array": %s, "aggregate": %s}\n' \
        "${pin_id}" "${eval_id}" "${aggregate_id}"
    printf '}\n'
} > "${record}"
ssh "${CLUSTER_LOGIN}" mkdir -p "${REMOTE_OUTPUT_ROOT}"
scp "${record}" "${CLUSTER_LOGIN}:${REMOTE_OUTPUT_ROOT}/cluster_submission.json"
rm -f "${record}"

echo "[PASS] fb670-via-latent-tracker chain submitted: pin=${pin_id} eval=${eval_id} aggregate=${aggregate_id}"
echo "[PASS] Submission record: ${REMOTE_OUTPUT_ROOT}/cluster_submission.json"
