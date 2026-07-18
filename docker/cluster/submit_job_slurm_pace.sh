#!/usr/bin/env bash
set -euo pipefail

run_singularity_path="$1/docker/cluster/run_singularity.sh"
workspace_root="$1"
container_profile="$2"
shift 2

time_limit="${CLUSTER_SLURM_TIME_LIMIT:-2-00:00:00}"
account="${CLUSTER_SLURM_ACCOUNT:-}"
qos="${CLUSTER_SLURM_QOS:-}"
partition="${CLUSTER_SLURM_PARTITION:-}"
nodes="${CLUSTER_SLURM_NODES:-1}"
ntasks="${CLUSTER_SLURM_NTASKS:-1}"
cpus_per_task="${CLUSTER_SLURM_CPUS_PER_TASK:-24}"
mem="${CLUSTER_SLURM_MEM:-32G}"
mem_per_gpu="${CLUSTER_SLURM_MEM_PER_GPU:-}"
legacy_gpu_spec="${CLUSTER_SLURM_GPU_SPEC:-l40s:1}"
if [ -n "${CLUSTER_SLURM_GPU_GRES:-}" ]; then
    gpu_gres="$CLUSTER_SLURM_GPU_GRES"
elif [ -n "${CLUSTER_SLURM_GRES:-}" ]; then
    gpu_gres="$CLUSTER_SLURM_GRES"
elif [[ "$legacy_gpu_spec" == gpu:* ]]; then
    gpu_gres="$legacy_gpu_spec"
else
    gpu_gres="gpu:${legacy_gpu_spec}"
fi
constraint="${CLUSTER_SLURM_CONSTRAINT:-}"
node_list="${CLUSTER_SLURM_NODELIST:-}"
exclude_nodes="${CLUSTER_SLURM_EXCLUDE:-}"
modules="${CLUSTER_SLURM_MODULES:-}"
job_name_prefix="${CLUSTER_SLURM_JOB_NAME_PREFIX:-g1-hl-pipeline}"
output_dir="${CLUSTER_SLURM_OUTPUT_DIR:-logs/slurm}"
keep_job_script="${CLUSTER_SLURM_KEEP_JOB_SCRIPT:-0}"
print_job_script="${CLUSTER_SLURM_PRINT_JOB_SCRIPT:-1}"
use_srun="${CLUSTER_SLURM_USE_SRUN:-0}"

account_directive=""
partition_directive=""
qos_directive=""
mem_directive="#SBATCH --mem=${mem}"
constraint_directive=""
node_list_directive=""
exclude_directive=""
module_block=""
run_prefix=""

if [ -n "$account" ]; then
    account_directive="#SBATCH --account=${account}"
fi
if [ -n "$partition" ]; then
    partition_directive="#SBATCH --partition=${partition}"
fi
if [ -n "$qos" ]; then
    qos_directive="#SBATCH --qos=${qos}"
fi
if [ -n "$mem_per_gpu" ]; then
    mem_directive="#SBATCH --mem-per-gpu=${mem_per_gpu}"
fi
if [ -n "$constraint" ]; then
    constraint_directive="#SBATCH --constraint=${constraint}"
fi
if [ -n "$node_list" ]; then
    node_list_directive="#SBATCH --nodelist=${node_list}"
fi
if [ -n "$exclude_nodes" ]; then
    exclude_directive="#SBATCH --exclude=${exclude_nodes}"
fi
if [ -n "$modules" ]; then
    module_block=$(
        cat <<'EOF'
if command -v module >/dev/null 2>&1; then
    for cluster_module in $CLUSTER_SLURM_MODULES; do
        module load "$cluster_module"
    done
else
    echo "[WARNING] CLUSTER_SLURM_MODULES is set, but the module command is not available."
fi
EOF
    )
fi
case "$use_srun" in
    1|true|TRUE|yes|YES|on|ON)
        run_prefix="srun"
        ;;
esac

mkdir -p "$output_dir"

if [ -z "$account" ]; then
    echo "[WARNING] CLUSTER_SLURM_ACCOUNT is unset; submitting without an explicit PACE account."
fi

printf -v quoted_run_singularity_path '%q' "$run_singularity_path"
printf -v quoted_workspace_root '%q' "$workspace_root"
printf -v quoted_container_profile '%q' "$container_profile"
printf -v quoted_job_args '%q ' "$@"

cat <<EOT > job.sh
#!/bin/bash

#SBATCH --job-name="${job_name_prefix}-$(date +"%Y%m%d-%H%M")"
#SBATCH --output="${output_dir}/%x_%j.log"
#SBATCH --error="${output_dir}/%x_%j.log"
${account_directive}
${partition_directive}
${qos_directive}
#SBATCH --nodes=${nodes}
#SBATCH --ntasks=${ntasks}
#SBATCH --cpus-per-task=${cpus_per_task}
${mem_directive}
#SBATCH --gres=${gpu_gres}
#SBATCH --time=${time_limit}
${constraint_directive}
${node_list_directive}
${exclude_directive}

set -euo pipefail

echo "[INFO] Host: \$(hostname)"
echo "[INFO] Job: \${SLURM_JOB_ID:-unknown}"
echo "[INFO] Account/QOS: ${account:-<default>}/${qos:-<default>}"
echo "[INFO] GPU GRES: ${gpu_gres}"
echo "[INFO] GPU status before job"
nvidia-smi || true

${module_block}

# Pass the container profile first to run_singularity.sh, then all arguments intended for the executed script.
# stdbuf forces line-buffered stdout/stderr so run_singularity progress + errors are
# flushed to the Slurm log even if the job dies mid-step (block buffering otherwise
# swallows the last output on failure).
set +e
${run_prefix} stdbuf -oL -eL bash ${quoted_run_singularity_path} ${quoted_workspace_root} ${quoted_container_profile} ${quoted_job_args}
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
