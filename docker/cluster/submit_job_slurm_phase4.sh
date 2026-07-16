#!/usr/bin/env bash
set -euo pipefail

# Submit the single Phase-4 array through the generic Slurm writer, then bind
# its job ID and synchronized workspace hashes to the persistent result root.

workspace_root="$1"
container_profile="$2"
shift 2
job_args=("$@")
generic_submitter="${workspace_root}/docker/cluster/submit_job_slurm.sh"
record_root="${CLUSTER_SLURM_SUBMISSION_RECORD_ROOT:-}"

if [[ ! -x "${generic_submitter}" ]]; then
    echo "[ERROR] Generic Slurm submitter is missing: ${generic_submitter}" >&2
    exit 2
fi
if [[ -z "${record_root}" ]]; then
    echo "[ERROR] Phase-4 submission requires CLUSTER_SLURM_SUBMISSION_RECORD_ROOT." >&2
    exit 2
fi

env_arg_value() {
    local requested_key="$1"
    local index=0
    local assignment=""
    for ((index = 0; index < ${#job_args[@]}; index++)); do
        if [[ "${job_args[index]}" != "--env" ]] || ((index + 1 >= ${#job_args[@]})); then
            continue
        fi
        assignment="${job_args[index + 1]}"
        if [[ "${assignment%%=*}" == "${requested_key}" ]]; then
            printf '%s' "${assignment#*=}"
        fi
    done
}

output_root="$(env_arg_value OUTPUT_ROOT)"
seeds="$(env_arg_value SEEDS)"
sample_budgets="$(env_arg_value SAMPLE_BUDGETS)"
motion_count="$(env_arg_value EXPECTED_MOTION_COUNT)"
if [[ -z "${output_root}" || -z "${seeds}" || -z "${sample_budgets}" ]]; then
    echo "[ERROR] Phase-4 submission arguments are incomplete." >&2
    exit 2
fi
if [[ ! "${motion_count}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] Phase-4 submission requires a positive motion count." >&2
    exit 2
fi

archive_sha="$(awk 'NR == 1 {print $1}' "${workspace_root}/workspace.tar.gz.sha256")"
repo_manifest_sha="$(sha256sum "${workspace_root}/repo_sync_manifest.tsv" | awk '{print $1}')"
if [[ ! "${archive_sha}" =~ ^[0-9a-f]{64}$ || ! "${repo_manifest_sha}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "[ERROR] Could not record workspace or repository-manifest SHA-256." >&2
    exit 2
fi
mkdir -p "${record_root}"

submit_output="$(
    bash "${generic_submitter}" "${workspace_root}" "${container_profile}" "${job_args[@]}" 2>&1
)"
printf '%s\n' "${submit_output}"
job_id="$(printf '%s\n' "${submit_output}" | awk '/Submitted batch job [0-9]+/{value=$NF} END{print value}')"
if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] Could not parse the Phase-4 Slurm array job ID." >&2
    exit 2
fi

record_path="${record_root}/cluster_submission.json"
{
    printf '{\n'
    printf '  "schema_version": 1,\n'
    printf '  "study": "phase4_no_language_sample_efficiency_v1",\n'
    printf '  "submitted_at_utc": "%s",\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '  "output_root": "%s",\n' "${output_root}"
    printf '  "cluster_workspace": "%s",\n' "${workspace_root}"
    printf '  "workspace_archive_sha256": "%s",\n' "${archive_sha}"
    printf '  "repo_sync_manifest_sha256": "%s",\n' "${repo_manifest_sha}"
    printf '  "array": "%s",\n' "${CLUSTER_SLURM_ARRAY:-}"
    printf '  "seeds": "%s",\n' "${seeds}"
    printf '  "sample_budgets": "%s",\n' "${sample_budgets}"
    printf '  "motion_count": %s,\n' "${motion_count}"
    printf '  "job_id": %s\n' "${job_id}"
    printf '}\n'
} > "${record_path}"
echo "[INFO] Persistent Phase-4 submission record: ${record_path}"
