#!/usr/bin/env bash
set -euo pipefail

run_singularity_path="$1/docker/cluster/run_singularity.sh"
workspace_root="$1"
container_profile="$2"
shift 2

export PATH="/opt/slurm/Ubuntu-20.04/24.11.0/bin:$PATH"

time_limit="${CLUSTER_SLURM_TIME_LIMIT:-2-00:00:00}"
partition="${CLUSTER_SLURM_PARTITION:-wu-lab}"
qos="${CLUSTER_SLURM_QOS:-short}"
nodes="${CLUSTER_SLURM_NODES:-1}"
ntasks="${CLUSTER_SLURM_NTASKS:-1}"
cpus_per_task="${CLUSTER_SLURM_CPUS_PER_TASK:-8}"
mem="${CLUSTER_SLURM_MEM:-96G}"
gpu_gres="${CLUSTER_SLURM_GPU_GRES:-gpu:a40:1}"
node_list="${CLUSTER_SLURM_NODELIST:-}"
exclude_nodes="${CLUSTER_SLURM_EXCLUDE:-}"
job_name_prefix="${CLUSTER_SLURM_JOB_NAME_PREFIX:-isaaclab}"
output_dir="${CLUSTER_SLURM_OUTPUT_DIR:-logs/slurm}"
keep_job_script="${CLUSTER_SLURM_KEEP_JOB_SCRIPT:-0}"
print_job_script="${CLUSTER_SLURM_PRINT_JOB_SCRIPT:-1}"
use_srun="${CLUSTER_SLURM_USE_SRUN:-0}"

node_list_directive=""
exclude_directive=""
run_prefix=""
if [ -n "$node_list" ]; then
    node_list_directive="#SBATCH --nodelist=${node_list}"
fi
if [ -n "$exclude_nodes" ]; then
    exclude_directive="#SBATCH --exclude=${exclude_nodes}"
fi
case "$use_srun" in
    1|true|TRUE|yes|YES|on|ON)
        run_prefix="srun"
        ;;
esac

mkdir -p "$output_dir"

printf -v quoted_run_singularity_path '%q' "$run_singularity_path"
printf -v quoted_workspace_root '%q' "$workspace_root"
printf -v quoted_container_profile '%q' "$container_profile"
printf -v quoted_job_args '%q ' "$@"

cat <<EOT > job.sh
#!/bin/bash

#SBATCH --job-name="${job_name_prefix}-$(date +"%Y%m%d-%H%M")"
#SBATCH --output="${output_dir}/%x_%j.log"
#SBATCH --error="${output_dir}/%x_%j.log"
#SBATCH --partition=${partition}
#SBATCH --qos=${qos}
#SBATCH --nodes=${nodes}
#SBATCH --ntasks=${ntasks}
#SBATCH --cpus-per-task=${cpus_per_task}
#SBATCH --mem=${mem}
#SBATCH --gres=${gpu_gres}
#SBATCH --time=${time_limit}
${node_list_directive}
${exclude_directive}

set -euo pipefail
export PATH="/opt/slurm/Ubuntu-20.04/24.11.0/bin:\$PATH"

echo "[INFO] Host: \$(hostname)"
echo "[INFO] Job: \${SLURM_JOB_ID:-unknown}"
echo "[INFO] Partition/QOS/GRES: ${partition}/${qos}/${gpu_gres}"
echo "[INFO] GPU status before job"
nvidia-smi || true

set +e
${run_prefix} bash ${quoted_run_singularity_path} ${quoted_workspace_root} ${quoted_container_profile} ${quoted_job_args}
job_status=\$?
set -e

echo "[INFO] GPU status after job"
nvidia-smi || true
exit \$job_status
EOT

if [ "$print_job_script" = "1" ]; then
    echo "[INFO] Generated Slurm job script:"
    sed 's/^/[SBATCH] /' job.sh
fi

sbatch < job.sh

if [ "$keep_job_script" != "1" ]; then
    rm job.sh
fi
