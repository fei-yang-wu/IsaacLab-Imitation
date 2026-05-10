#!/usr/bin/env bash

run_singularity_path="$1/docker/cluster/run_singularity.sh"
workspace_root="$1"
container_profile="$2"
shift 2
time_limit="${CLUSTER_SLURM_TIME_LIMIT:-2-00:00:00}"

printf -v quoted_run_singularity_path '%q' "$run_singularity_path"
printf -v quoted_workspace_root '%q' "$workspace_root"
printf -v quoted_container_profile '%q' "$container_profile"
printf -v quoted_job_args '%q ' "$@"

cat <<EOT > job.sh
#!/bin/bash

#SBATCH --gpus-per-node=a40:1
#SBATCH -N1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-gpu=48G
#SBATCH --time=$time_limit
#SBATCH --job-name="training-$(date +"%Y-%m-%dT%H:%M")"
#SBATCH --output="output_%j.log"
#SBATCH --error="output_%j.log"
#SBATCH --partition=wu-lab
#SBATCH --nodelist=dendrite,synapse
#SBATCH --qos=short

echo "[INFO] GPU status before job"
nvidia-smi

# Pass the container profile first to run_singularity.sh, then all arguments intended for the executed script
bash $quoted_run_singularity_path $quoted_workspace_root $quoted_container_profile $quoted_job_args
job_status=\$?

echo "[INFO] GPU status after job"
nvidia-smi
exit \$job_status
EOT
sbatch < job.sh
rm job.sh
