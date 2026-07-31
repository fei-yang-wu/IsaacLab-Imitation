#!/usr/bin/env bash
set -euo pipefail

# Guarded ICE dependency chain for the completed enc380 tracker:
#   qualify -> one-session walk1 oracle collection -> 24 route tasks -> aggregate
# The 24 ICE tasks are 12 logical model-size/seed cells x two routes. Splitting
# the routes gives each long planner fit the full ICE 16-hour walltime.

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
REMOTE_PROJECT_ROOT="${REMOTE_PROJECT_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/isaaclab}"
REMOTE_OUTPUT_ROOT="${REMOTE_PROJECT_ROOT}/${OUTPUT_ROOT}"
CELL_ARRAY="${CELL_ARRAY:-0-23%8}"

case "${DRY_RUN}" in
    1|true|TRUE|yes|YES) DRY_RUN=1 ;;
    0|false|FALSE|no|NO) DRY_RUN=0 ;;
    *) echo "[ERROR] DRY_RUN must be boolean, got ${DRY_RUN}." >&2; exit 2 ;;
esac
if [[ ! "${CELL_ARRAY}" =~ ^0-23(%[1-9][0-9]*)?$ ]]; then
    echo "[ERROR] CELL_ARRAY must cover the exact 0-23 route-task grid, optionally with %N." >&2
    exit 2
fi
if [[ "${DRY_RUN}" == "0" && ! "${LOW_LEVEL_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "[ERROR] Actual submission requires the completed 5B tracker SHA-256." >&2
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
    --stage enc380
    --enc380-low-level-checkpoint "${LOW_LEVEL_CHECKPOINT}"
    --enc380-skill-checkpoint "${SKILL_CHECKPOINT}"
    --enc380-completion-record "${TRACKER_COMPLETION_RECORD}"
    --enc380-output-root "${OUTPUT_ROOT}"
    --enc380-low-level-sha256 "${LOW_LEVEL_SHA256}"
    --enc380-skill-sha256 "${SKILL_SHA256}"
)

print_stage() {
    local mode="$1" dependency="$2" array="$3" time_limit="$4"
    printf '[STAGE] mode=%s dependency=%s array=%s time=%s\n' \
        "${mode}" "${dependency:-none}" "${array:-none}" "${time_limit}"
    printf '[CMD]'; printf ' %q' "${base_cmd[@]}" --enc380-mode "${mode}"; printf '\n'
}

if [[ "${DRY_RUN}" == "1" ]]; then
    print_stage qualify "" "" 06:00:00
    print_stage demo 'afterok:<qualify>' "" 06:00:00
    print_stage cell 'afterok:<demo>' "${CELL_ARRAY}" 15:59:00
    print_stage aggregate 'afterok:<cell>' "" 01:00:00
    echo "[INFO] Motion: walk1_subject1"
    echo "[INFO] Oracle data: one 100-env session, 100 complete segments total."
    echo "[INFO] Optimizer updates tiny/small/medium/large: 10k/20k/30k/50k."
    echo "[INFO] Effective batch: 1024; microbatches: 1024/512/256/128."
    echo "[INFO] Capacity cells: 12 logical pairs; 24 independent ICE route tasks."
    echo "[INFO] DRY_RUN=1; not contacting ICE."
    exit 0
fi

repo_head="$(git rev-parse HEAD)"
diff_sha="$(git diff --binary -- . ':!logs' | sha256sum | awk '{print $1}')"
workflow_sources=(
    experiments/campaigns/2026-07-23-lafan1-planner-capacity/enc380_capacity_grid.py
    experiments/campaigns/2026-07-23-lafan1-planner-capacity/run_capacity_entry.py
    experiments/campaigns/2026-07-23-lafan1-planner-capacity/run_enc380_planner_route_comparison.sh
    experiments/campaigns/2026-07-23-lafan1-planner-capacity/interface_baselines/aggregate_enc380_route_comparison.py
    experiments/campaigns/2026-07-23-lafan1-planner-capacity/interface_baselines/audit_enc380_motion_selection.py
    experiments/campaigns/2026-07-23-lafan1-planner-capacity/interface_baselines/audit_enc380_tracker_completion.py
    experiments/campaigns/2026-07-23-lafan1-planner-capacity/interface_baselines/audit_enc380_paired_demonstrations.py
    experiments/campaigns/2026-07-23-lafan1-planner-capacity/interface_baselines/audit_packet_encoder_pin.py
    experiments/campaigns/2026-07-23-lafan1-planner-capacity/interface_baselines/materialize_paired_interface_samples.py
    experiments/campaigns/2026-07-23-lafan1-planner-capacity/interface_baselines/packet_to_latent_command.py
    experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/interface_planner_common.py
    experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/low_level_tracker.py
    experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/merge_planner_samples.py
    experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/planner_latency.py
    experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/planner_sample_schema.py
    experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/train_chunked_transformer_planner.py
    experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/validate_latent_skill_checkpoint_binding.py
    experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/audit_diffsr_latent_qualification.py
    scripts/rlopt/eval_skill_commander_closed_loop.py
)
workflow_source_sha="$(sha256sum "${workflow_sources[@]}" | sha256sum | awk '{print $1}')"

if ssh "${CLUSTER_LOGIN}" test -e "${REMOTE_OUTPUT_ROOT}"; then
    echo "[ERROR] Refusing existing remote output root: ${REMOTE_OUTPUT_ROOT}" >&2
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
    if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] Could not parse Slurm job ID for ${mode}." >&2
        exit 2
    fi
    printf '%s' "${job_id}"
}

qualify_id="$(submit_stage qualify "" "" 06:00:00 enc380-qualify)"
demo_id="$(submit_stage demo "afterok:${qualify_id}" "" 06:00:00 enc380-demo)"
cell_id="$(submit_stage cell "afterok:${demo_id}" "${CELL_ARRAY}" 15:59:00 enc380-capacity)"
aggregate_id="$(submit_stage aggregate "afterok:${cell_id}" "" 01:00:00 enc380-aggregate)"

workflow_source_sha_after="$(sha256sum "${workflow_sources[@]}" | sha256sum | awk '{print $1}')"
if [[ "${workflow_source_sha_after}" != "${workflow_source_sha}" ]]; then
    echo "[ERROR] Workflow sources changed while the dependency chain was being submitted." >&2
    exit 2
fi
record="$(mktemp)"
{
    printf '{\n'
    printf '  "schema_version": 1,\n'
    printf '  "submitted_at_utc": "%s",\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '  "output_root": "%s",\n' "${OUTPUT_ROOT}"
    printf '  "repo_head": "%s",\n' "${repo_head}"
    printf '  "working_tree_diff_sha256": "%s",\n' "${diff_sha}"
    printf '  "workflow_source_manifest_sha256": "%s",\n' "${workflow_source_sha}"
    printf '  "low_level_checkpoint": "%s",\n' "${LOW_LEVEL_CHECKPOINT}"
    printf '  "low_level_sha256": "%s",\n' "${LOW_LEVEL_SHA256}"
    printf '  "skill_checkpoint": "%s",\n' "${SKILL_CHECKPOINT}"
    printf '  "tracker_completion_record": "%s",\n' "${TRACKER_COMPLETION_RECORD}"
    printf '  "skill_sha256": "%s",\n' "${SKILL_SHA256}"
    printf '  "motions": ["walk1_subject1"],\n'
    printf '  "sizes": ["tiny", "small", "medium", "large"],\n'
    printf '  "seeds": [0, 1, 2],\n'
    printf '  "route_task_array": "%s",\n' "${CELL_ARRAY}"
    printf '  "oracle_collection": {"sessions": 1, "parallel_environments": 100, "completed_trajectories_total": 100, "completed_trajectories_per_motion": 100},\n'
    printf '  "planner_training": {"stages": ["oracle_supervised"], "effective_batch_size": 1024, "updates_by_size": {"tiny": 10000, "small": 20000, "medium": 30000, "large": 50000}, "micro_batch_by_size": {"tiny": 1024, "small": 512, "medium": 256, "large": 128}, "checkpoint_selection": "minimum held-out normalized target RMSE", "learned_planner_rollout_collection": false, "finetune": false},\n'
    printf '  "jobs": {"qualify": %s, "oracle_collection": %s, "capacity_array": %s, "aggregate": %s}\n' \
        "${qualify_id}" "${demo_id}" "${cell_id}" "${aggregate_id}"
    printf '}\n'
} > "${record}"
ssh "${CLUSTER_LOGIN}" mkdir -p "${REMOTE_OUTPUT_ROOT}"
scp "${record}" "${CLUSTER_LOGIN}:${REMOTE_OUTPUT_ROOT}/cluster_submission.json"
rm -f "${record}"

echo "[PASS] enc380 ICE chain submitted: qualify=${qualify_id} demo=${demo_id} cell=${cell_id} aggregate=${aggregate_id}"
echo "[PASS] Submission record: ${REMOTE_OUTPUT_ROOT}/cluster_submission.json"
