#!/usr/bin/env bash
set -euo pipefail

# Submit the BONES-SEED planner comparison as one dependency chain from a
# single synchronized workspace:
#   prepare -> rollout array -> finetune -> final-eval array -> summarize

workspace_root="$1"
container_profile="$2"
shift 2
common_args=("$@")
generic_submitter="${workspace_root}/docker/cluster/submit_job_slurm.sh"

if [[ ! -x "${generic_submitter}" ]]; then
    echo "[ERROR] Generic Slurm submitter is missing: ${generic_submitter}" >&2
    exit 2
fi

pipeline_array="${CLUSTER_SLURM_PIPELINE_ARRAY:-}"
if [[ ! "${pipeline_array}" =~ ^0-[0-9]+(%[1-9][0-9]*)?$ ]]; then
    echo "[ERROR] CLUSTER_SLURM_PIPELINE_ARRAY must use 0-END or 0-END%MAX_PARALLEL." >&2
    exit 2
fi
array_tail="${pipeline_array#*-}"
array_end="${array_tail%%\%*}"
if ((array_end < 0)); then
    echo "[ERROR] BONES-SEED pipeline array must contain at least goal index 0." >&2
    exit 2
fi

env_arg_value() {
    local requested_key="$1"
    local index=0
    local assignment=""
    for ((index = 0; index < ${#common_args[@]}; index++)); do
        if [[ "${common_args[index]}" != "--env" ]]; then
            continue
        fi
        if ((index + 1 >= ${#common_args[@]})); then
            break
        fi
        assignment="${common_args[index + 1]}"
        if [[ "${assignment%%=*}" == "${requested_key}" ]]; then
            printf '%s' "${assignment#*=}"
        fi
    done
}

mode=""
for ((index = 0; index < ${#common_args[@]}; index++)); do
    if [[ "${common_args[index]}" == "--mode" ]] && ((index + 1 < ${#common_args[@]})); then
        mode="${common_args[index + 1]}"
    fi
done
if [[ "${mode}" != "bones-seed-multigoal-language" ]]; then
    echo "[ERROR] BONES pipeline submitter requires --mode bones-seed-multigoal-language." >&2
    exit 2
fi

output_root="$(env_arg_value OUTPUT_ROOT)"
if [[ -z "${output_root}" ]]; then
    echo "[ERROR] BONES pipeline submission requires an explicit OUTPUT_ROOT." >&2
    exit 2
fi
goal_limit="$(env_arg_value GOAL_LIMIT)"
if [[ ! "${goal_limit}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] BONES pipeline submission requires a positive GOAL_LIMIT." >&2
    exit 2
fi
if ((array_end + 1 != goal_limit)); then
    echo "[ERROR] CLUSTER_SLURM_PIPELINE_ARRAY must cover exactly GOAL_LIMIT=${goal_limit} goals." >&2
    exit 2
fi
seed="$(env_arg_value SEED)"
if [[ ! "${seed}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] BONES pipeline submission requires an explicit non-negative SEED." >&2
    exit 2
fi
submission_record_root="${CLUSTER_SLURM_SUBMISSION_RECORD_ROOT:-}"
if [[ -z "${submission_record_root}" ]]; then
    echo "[ERROR] BONES pipeline submission requires CLUSTER_SLURM_SUBMISSION_RECORD_ROOT." >&2
    exit 2
fi

root_dependency="${CLUSTER_SLURM_DEPENDENCY:-}"
prepare_time="${CLUSTER_SLURM_PREPARE_TIME_LIMIT:-1-00:00:00}"
rollout_time="${CLUSTER_SLURM_ROLLOUT_TIME_LIMIT:-12:00:00}"
finetune_time="${CLUSTER_SLURM_FINETUNE_TIME_LIMIT:-12:00:00}"
final_eval_time="${CLUSTER_SLURM_FINAL_EVAL_TIME_LIMIT:-4:00:00}"
summarize_time="${CLUSTER_SLURM_SUMMARIZE_TIME_LIMIT:-1:00:00}"
archive_sha="$(awk 'NR == 1 {print $1}' "${workspace_root}/workspace.tar.gz.sha256")"
repo_manifest_sha="$(sha256sum "${workspace_root}/repo_sync_manifest.tsv" | awk '{print $1}')"
if [[ ! "${archive_sha}" =~ ^[0-9a-f]{64}$ || ! "${repo_manifest_sha}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "[ERROR] Could not record workspace or repository-manifest SHA-256." >&2
    exit 2
fi
mkdir -p "${submission_record_root}"

submit_stage() {
    local stage="$1"
    local dependency="$2"
    local array_spec="$3"
    local time_limit="$4"
    local resume="$5"
    local job_name="$6"
    local output=""
    local job_id=""
    local -a stage_args=(
        "${common_args[@]}"
        --env "PIPELINE_STAGE=${stage}"
        --env "RESUME=${resume}"
    )

    output="$(
        env \
            CLUSTER_SLURM_DEPENDENCY="${dependency}" \
            CLUSTER_SLURM_ARRAY="${array_spec}" \
            CLUSTER_SLURM_TIME_LIMIT="${time_limit}" \
            CLUSTER_SLURM_JOB_NAME="${job_name}" \
            bash "${generic_submitter}" \
                "${workspace_root}" \
                "${container_profile}" \
                "${stage_args[@]}" 2>&1
    )"
    printf '%s\n' "${output}" >&2
    job_id="$(printf '%s\n' "${output}" | awk '/Submitted batch job [0-9]+/{value=$NF} END{print value}')"
    if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] Could not parse Slurm job ID for stage ${stage}." >&2
        exit 2
    fi
    printf '%s' "${job_id}"
}

prepare_id="$(
    submit_stage prepare "${root_dependency}" "" "${prepare_time}" 1 bones-prepare
)"
rollout_id="$(
    submit_stage rollout "afterok:${prepare_id}" "${pipeline_array}" \
        "${rollout_time}" 1 bones-rollout
)"
finetune_id="$(
    submit_stage finetune "afterok:${rollout_id}" "" \
        "${finetune_time}" 1 bones-finetune
)"
final_eval_id="$(
    submit_stage final-eval "afterok:${finetune_id}" "${pipeline_array}" \
        "${final_eval_time}" 1 bones-final-eval
)"
summarize_id="$(
    submit_stage summarize "afterok:${final_eval_id}" "" \
        "${summarize_time}" 1 bones-summarize
)"

record_path="${workspace_root}/bones_pipeline_submission_$(date -u +%Y%m%dT%H%M%SZ).txt"
{
    printf 'output_root=%s\n' "${output_root}"
    printf 'pipeline_array=%s\n' "${pipeline_array}"
    printf 'goal_limit=%s\n' "${goal_limit}"
    printf 'root_dependency=%s\n' "${root_dependency}"
    printf 'prepare_job_id=%s\n' "${prepare_id}"
    printf 'rollout_array_job_id=%s\n' "${rollout_id}"
    printf 'finetune_job_id=%s\n' "${finetune_id}"
    printf 'final_eval_array_job_id=%s\n' "${final_eval_id}"
    printf 'summarize_job_id=%s\n' "${summarize_id}"
} > "${record_path}"

persistent_record="${submission_record_root}/cluster_submission.json"
{
    printf '{\n'
    printf '  "schema_version": 1,\n'
    printf '  "submitted_at_utc": "%s",\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '  "seed": %s,\n' "${seed}"
    printf '  "output_root": "%s",\n' "${output_root}"
    printf '  "cluster_workspace": "%s",\n' "${workspace_root}"
    printf '  "workspace_archive_sha256": "%s",\n' "${archive_sha}"
    printf '  "repo_sync_manifest_sha256": "%s",\n' "${repo_manifest_sha}"
    printf '  "pipeline_array": "%s",\n' "${pipeline_array}"
    printf '  "goal_limit": %s,\n' "${goal_limit}"
    printf '  "root_dependency": "%s",\n' "${root_dependency}"
    printf '  "jobs": {\n'
    printf '    "prepare": %s,\n' "${prepare_id}"
    printf '    "rollout_array": %s,\n' "${rollout_id}"
    printf '    "finetune": %s,\n' "${finetune_id}"
    printf '    "final_eval_array": %s,\n' "${final_eval_id}"
    printf '    "summarize": %s\n' "${summarize_id}"
    printf '  }\n'
    printf '}\n'
} > "${persistent_record}"

echo "[INFO] BONES-SEED planner dependency chain submitted."
echo "[INFO] prepare=${prepare_id} rollout=${rollout_id} finetune=${finetune_id} final_eval=${final_eval_id} summarize=${summarize_id}"
echo "[INFO] Submission record: ${record_path}"
echo "[INFO] Persistent submission record: ${persistent_record}"
