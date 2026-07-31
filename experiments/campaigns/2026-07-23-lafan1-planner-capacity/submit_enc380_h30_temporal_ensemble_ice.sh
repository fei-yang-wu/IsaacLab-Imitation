#!/usr/bin/env bash
set -euo pipefail

# Reuse the matched H10 oracle rows after their one-session collection, train
# one medium H30 planner, evaluate two execution rules, then aggregate against
# the matching H10 medium/seed-0 result after the main route array completes.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

DRY_RUN="${DRY_RUN:-1}"
SOURCE_STUDY_ROOT="${SOURCE_STUDY_ROOT:-logs/interface_baselines/lafan1_enc380_route_capacity_5b_oracle100_progressive_b1024_20260730}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/interface_baselines/lafan1_enc380_h30_temporal_medium_seed0_20260730}"
LOW_LEVEL_CHECKPOINT="${LOW_LEVEL_CHECKPOINT:-/data/resume_store/lafan1_enc380_rootqpos_h10_z256_seed0/model_5b.pt}"
SKILL_CHECKPOINT="${SKILL_CHECKPOINT:-/data/enc380_store/lafan1_enc380_rootqpos_h10_z256_seed0/skill_encoder/checkpoints/latest.pt}"
TRACKER_COMPLETION_RECORD="${TRACKER_COMPLETION_RECORD:-/data/resume_store/lafan1_enc380_rootqpos_h10_z256_seed0/completion.json}"
LOW_LEVEL_SHA256="${LOW_LEVEL_SHA256:-d33fa146f54222848da8b9a92eb5579f5acb8b3a46c484399c906b076c219260}"
SKILL_SHA256="${SKILL_SHA256:-1d530fcb5920112b84bc53dbaddf2b3eb3da13a32a379513d8ee8719bc57d546}"
SOURCE_DEMO_JOB_ID="${SOURCE_DEMO_JOB_ID:-5550599}"
SOURCE_CELL_JOB_ID="${SOURCE_CELL_JOB_ID:-5550601}"
SOURCE_DEMO_DEPENDENCY="${SOURCE_DEMO_DEPENDENCY:-afterok:${SOURCE_DEMO_JOB_ID}}"
REMOTE_PROJECT_ROOT="${REMOTE_PROJECT_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/isaaclab}"
REMOTE_OUTPUT_ROOT="${REMOTE_PROJECT_ROOT}/${OUTPUT_ROOT}"

case "${DRY_RUN}" in
    1|true|TRUE|yes|YES) DRY_RUN=1 ;;
    0|false|FALSE|no|NO) DRY_RUN=0 ;;
    *) echo "[ERROR] DRY_RUN must be boolean, got ${DRY_RUN}." >&2; exit 2 ;;
esac
if [[ "${DRY_RUN}" == "0" ]]; then
    for value in "${SOURCE_DEMO_JOB_ID}" "${SOURCE_CELL_JOB_ID}"; do
        [[ "${value}" =~ ^[0-9]+$ ]] || {
            echo "[ERROR] Actual submission requires numeric source job IDs." >&2
            exit 2
        }
    done
fi
if [[ "${SOURCE_DEMO_DEPENDENCY}" == "none" ]]; then
    SOURCE_DEMO_DEPENDENCY=""
elif [[ ! "${SOURCE_DEMO_DEPENDENCY}" =~ ^afterok:[0-9]+$ ]]; then
    echo "[ERROR] SOURCE_DEMO_DEPENDENCY must be afterok:<job> or none." >&2
    exit 2
fi

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
    --stage enc380_h30
    --enc380-low-level-checkpoint "${LOW_LEVEL_CHECKPOINT}"
    --enc380-skill-checkpoint "${SKILL_CHECKPOINT}"
    --enc380-completion-record "${TRACKER_COMPLETION_RECORD}"
    --enc380-low-level-sha256 "${LOW_LEVEL_SHA256}"
    --enc380-skill-sha256 "${SKILL_SHA256}"
    --enc380-source-study-root "${SOURCE_STUDY_ROOT}"
    --enc380-h30-output-root "${OUTPUT_ROOT}"
)

if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[STAGE] screen dependency=%s time=15:59:00\n' \
        "${SOURCE_DEMO_DEPENDENCY:-none}"
    printf '[CMD]'; printf ' %q' "${base_cmd[@]}" --enc380-h30-mode screen; printf '\n'
    printf '[STAGE] aggregate dependency=afterok:<screen>:%s time=01:00:00\n' \
        "${SOURCE_CELL_JOB_ID}"
    printf '[CMD]'; printf ' %q' "${base_cmd[@]}" --enc380-h30-mode aggregate; printf '\n'
    echo "[INFO] Reuses the exact 4,864 H10 causal rows; no simulator recollection."
    echo "[INFO] One medium seed-0 H30 planner: 30k updates, batch 1024, microbatch 256."
    echo "[INFO] Evaluations: first H10/discard H20 and exponential H30 overlap."
    echo "[INFO] Aggregate includes the matching H10 medium seed-0 baseline."
    echo "[INFO] DRY_RUN=1; not contacting ICE."
    exit 0
fi

workflow_sources=(
    experiments/campaigns/2026-07-23-lafan1-planner-capacity/run_capacity_entry.py
    experiments/campaigns/2026-07-23-lafan1-planner-capacity/run_enc380_h30_temporal_ensemble.sh
    experiments/campaigns/2026-07-23-lafan1-planner-capacity/interface_baselines/materialize_long_horizon_root_qpos.py
    experiments/campaigns/2026-07-23-lafan1-planner-capacity/interface_baselines/packet_to_latent_command.py
    experiments/campaigns/2026-07-23-lafan1-planner-capacity/interface_baselines/aggregate_h30_temporal_ensemble.py
    experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/interface_planner_common.py
    experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/low_level_tracker.py
    experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/planner_latency.py
    experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/planner_sample_schema.py
    experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/train_chunked_transformer_planner.py
    scripts/rlopt/eval_skill_commander_closed_loop.py
)
workflow_source_sha="$(sha256sum "${workflow_sources[@]}" | sha256sum | awk '{print $1}')"
repo_head="$(git rev-parse HEAD)"
diff_sha="$(git diff --binary -- . ':!logs' | sha256sum | awk '{print $1}')"

if ssh "${CLUSTER_LOGIN}" test -e "${REMOTE_OUTPUT_ROOT}"; then
    echo "[ERROR] Refusing existing remote output root: ${REMOTE_OUTPUT_ROOT}" >&2
    exit 2
fi

submit_stage() {
    local mode="$1" dependency="$2" time_limit="$3" job_name="$4"
    local output job_id
    output="$(
        env \
            CLUSTER_SLURM_DEPENDENCY="${dependency}" \
            CLUSTER_SLURM_ARRAY="" \
            CLUSTER_SLURM_TIME_LIMIT="${time_limit}" \
            CLUSTER_SLURM_JOB_NAME_PREFIX="${job_name}" \
            "${base_cmd[@]}" --enc380-h30-mode "${mode}" 2>&1
    )"
    printf '%s\n' "${output}" >&2
    job_id="$(printf '%s\n' "${output}" | awk '/Submitted batch job [0-9]+/{value=$NF} END{print value}')"
    [[ "${job_id}" =~ ^[0-9]+$ ]] || {
        echo "[ERROR] Could not parse Slurm job ID for ${mode}." >&2
        exit 2
    }
    printf '%s' "${job_id}"
}

screen_id="$(submit_stage screen "${SOURCE_DEMO_DEPENDENCY}" 15:59:00 enc380-h30)"
aggregate_id="$(submit_stage aggregate "afterok:${screen_id}:${SOURCE_CELL_JOB_ID}" 01:00:00 enc380-h30-agg)"

workflow_source_sha_after="$(sha256sum "${workflow_sources[@]}" | sha256sum | awk '{print $1}')"
[[ "${workflow_source_sha_after}" == "${workflow_source_sha}" ]] || {
    echo "[ERROR] Workflow sources changed during submission." >&2
    exit 2
}

record="$(mktemp)"
{
    printf '{\n'
    printf '  "schema_version": 1,\n'
    printf '  "submitted_at_utc": "%s",\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '  "output_root": "%s",\n' "${OUTPUT_ROOT}"
    printf '  "source_study_root": "%s",\n' "${SOURCE_STUDY_ROOT}"
    printf '  "repo_head": "%s",\n' "${repo_head}"
    printf '  "working_tree_diff_sha256": "%s",\n' "${diff_sha}"
    printf '  "workflow_source_manifest_sha256": "%s",\n' "${workflow_source_sha}"
    printf '  "low_level_checkpoint": "%s",\n' "${LOW_LEVEL_CHECKPOINT}"
    printf '  "low_level_sha256": "%s",\n' "${LOW_LEVEL_SHA256}"
    printf '  "skill_checkpoint": "%s",\n' "${SKILL_CHECKPOINT}"
    printf '  "skill_sha256": "%s",\n' "${SKILL_SHA256}"
    printf '  "data_reuse": {"same_causal_rows": true, "expected_rows": 4864, "source_horizon_steps": 10, "target_horizon_steps": 30, "simulator_recollection": false},\n'
    printf '  "planner_training": {"model_size": "medium", "seed": 0, "updates": 30000, "effective_batch_size": 1024, "micro_batch_size": 256, "checkpoint_selection": "minimum held-out normalized target RMSE"},\n'
    printf '  "evaluation_modes": ["execute_first10", "temporal_exponential"],\n'
    printf '  "dependencies": {"source_demo_job": %s, "source_cell_array_job": %s},\n' \
        "${SOURCE_DEMO_JOB_ID}" "${SOURCE_CELL_JOB_ID}"
    printf '  "jobs": {"screen": %s, "aggregate": %s}\n' \
        "${screen_id}" "${aggregate_id}"
    printf '}\n'
} > "${record}"
ssh "${CLUSTER_LOGIN}" mkdir -p "${REMOTE_OUTPUT_ROOT}"
scp "${record}" "${CLUSTER_LOGIN}:${REMOTE_OUTPUT_ROOT}/cluster_submission.json"
rm -f "${record}"

echo "[PASS] H30 ICE chain submitted: screen=${screen_id} aggregate=${aggregate_id}"
echo "[PASS] Submission record: ${REMOTE_OUTPUT_ROOT}/cluster_submission.json"
