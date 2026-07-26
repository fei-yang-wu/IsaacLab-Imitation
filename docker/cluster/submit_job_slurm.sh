#!/usr/bin/env bash

workspace_root="$1"
run_singularity_path="$workspace_root/docker/cluster/run_singularity.sh"
workspace_archive="$workspace_root/workspace.tar.gz"
container_profile="$2"
shift 2
time_limit="${CLUSTER_SLURM_TIME_LIMIT:-2-00:00:00}"
gpu_spec="${CLUSTER_SLURM_GPU_SPEC:-a40:1}"
node_list="${CLUSTER_SLURM_NODELIST-dendrite,synapse}"
qos="${CLUSTER_SLURM_QOS:-short}"
dependency="${CLUSTER_SLURM_DEPENDENCY:-}"
array_spec="${CLUSTER_SLURM_ARRAY:-}"
job_name="${CLUSTER_SLURM_JOB_NAME:-training-$(date +"%Y%m%d-%H%M")}"
node_list_directive=""
dependency_directive=""
array_directive=""
job_output_pattern="output_%j.log"
if [ -n "$node_list" ]; then
    node_list_directive="#SBATCH --nodelist=$node_list"
fi
if [ -n "$dependency" ]; then
    if [[ ! "$dependency" =~ ^afterok:[0-9]+(:[0-9]+)*$ ]]; then
        echo "[ERROR] CLUSTER_SLURM_DEPENDENCY must use afterok:<job>[:<job>...]." >&2
        exit 2
    fi
    dependency_directive="#SBATCH --dependency=$dependency"
fi
if [ -n "$array_spec" ]; then
    if [[ ! "$array_spec" =~ ^[0-9]+-[0-9]+(%[1-9][0-9]*)?$ ]]; then
        echo "[ERROR] CLUSTER_SLURM_ARRAY must use START-END or START-END%MAX_PARALLEL." >&2
        exit 2
    fi
    array_start="${array_spec%%-*}"
    array_tail="${array_spec#*-}"
    array_end="${array_tail%%\%*}"
    if ((array_start > array_end)); then
        echo "[ERROR] CLUSTER_SLURM_ARRAY start must be <= end." >&2
        exit 2
    fi
    array_directive="#SBATCH --array=$array_spec"
    job_output_pattern="output_%A_%a.log"
fi
if [[ ! "$job_name" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "[ERROR] CLUSTER_SLURM_JOB_NAME may contain only letters, digits, '.', '_', and '-'." >&2
    exit 2
fi

printf -v quoted_run_singularity_path '%q' "$run_singularity_path"
printf -v quoted_workspace_root '%q' "$workspace_root"
printf -v quoted_container_profile '%q' "$container_profile"
printf -v quoted_job_args '%q ' "$@"

if [ -f "$workspace_archive" ]; then
    printf -v quoted_workspace_archive '%q' "$workspace_archive"
    read -r -d '' job_run_block <<EOT || true
bootstrap_root="\${CLUSTER_JOB_TMPDIR_ROOT:-\${TMPDIR:-/tmp}}/isaaclab-bootstrap-\${SLURM_JOB_ID:-\$\$}"
rm -rf "\$bootstrap_root"
mkdir -p "\$bootstrap_root"
echo "[INFO] Extracting submitted workspace archive into compute-local storage."
if ! tar -xzf $quoted_workspace_archive -C "\$bootstrap_root"; then
    echo "[ERROR] Failed to extract submitted workspace archive: $quoted_workspace_archive" >&2
    exit 1
fi
extracted_workspace="\$bootstrap_root/isaaclab-submission-\${SLURM_JOB_ID:-\$\$}"
mv "\$bootstrap_root/workspace" "\$extracted_workspace"
bash "\$extracted_workspace/docker/cluster/run_singularity.sh" "\$extracted_workspace" $quoted_container_profile $quoted_job_args
job_status=\$?
rm -rf "\$bootstrap_root"
EOT
else
    read -r -d '' job_run_block <<EOT || true
bash $quoted_run_singularity_path $quoted_workspace_root $quoted_container_profile $quoted_job_args
job_status=\$?
EOT
fi

cat <<EOT > job.sh
#!/bin/bash

#SBATCH --gpus-per-node=$gpu_spec
#SBATCH -N1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-gpu=48G
#SBATCH --time=$time_limit
#SBATCH --job-name="$job_name"
#SBATCH --output="$job_output_pattern"
#SBATCH --error="$job_output_pattern"
#SBATCH --partition=wu-lab
$node_list_directive
$dependency_directive
$array_directive
#SBATCH --qos=$qos

echo "[INFO] GPU status before job"
nvidia-smi

# Pass the container profile first to run_singularity.sh, then all arguments intended for the executed script.
$job_run_block

echo "[INFO] GPU status after job"
nvidia-smi
exit \$job_status
EOT
sbatch < job.sh
rm job.sh
