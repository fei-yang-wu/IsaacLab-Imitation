#!/usr/bin/env bash
set -euo pipefail

workspace_root="$1"
_container_profile="${2:-}"
shift 2

export PATH="/opt/slurm/Ubuntu-20.04/24.11.0/bin:${PATH}"

time_limit="${CLUSTER_SLURM_TIME_LIMIT:-2-00:00:00}"
partition="${CLUSTER_SLURM_PARTITION:-wu-lab}"
qos="${CLUSTER_SLURM_QOS:-short}"
nodes="${CLUSTER_SLURM_NODES:-1}"
ntasks="${CLUSTER_SLURM_NTASKS:-1}"
cpus_per_task="${CLUSTER_SLURM_CPUS_PER_TASK:-8}"
mem="${CLUSTER_SLURM_MEM:-96G}"
gpu_gres="${CLUSTER_SLURM_GPU_GRES:-gpu:a40:1}"
job_name_prefix="${CLUSTER_SLURM_JOB_NAME_PREFIX:-isaaclab-pixi}"
output_dir="${CLUSTER_SLURM_OUTPUT_DIR:-logs/slurm}"
pixi_env="${CLUSTER_PIXI_ENV:-isaaclab}"
pixi_cache_dir="${CLUSTER_PIXI_CACHE_DIR:-/coc/flash12/${USER}/Research/IsaacLab/pixi-cache}"
data_dir="${CLUSTER_DATA_DIR:-/coc/flash12/${USER}/Research/IsaacLab/data}"
python_executable="${CLUSTER_PYTHON_EXECUTABLE:-scripts/rlopt/run_bones_seed_language_pipeline.py}"
keep_job_script="${CLUSTER_SLURM_KEEP_JOB_SCRIPT:-0}"
print_job_script="${CLUSTER_SLURM_PRINT_JOB_SCRIPT:-1}"
job_args=()

for arg in "$@"; do
    case "$arg" in
        /data)
            job_args+=("$data_dir")
            ;;
        /data/*)
            job_args+=("${data_dir}${arg#/data}")
            ;;
        *)
            job_args+=("$arg")
            ;;
    esac
done

mkdir -p "$output_dir"

printf -v quoted_workspace_root '%q' "$workspace_root"
printf -v quoted_pixi_env '%q' "$pixi_env"
printf -v quoted_pixi_cache_dir '%q' "$pixi_cache_dir"
printf -v quoted_data_dir '%q' "$data_dir"
printf -v quoted_python_executable '%q' "$python_executable"
printf -v quoted_hf_token_file '%q' "${CLUSTER_HF_TOKEN_FILE:-}"
printf -v quoted_wandb_key_file '%q' "${CLUSTER_WANDB_API_KEY_FILE:-}"
printf -v quoted_job_args '%q ' "${job_args[@]}"

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

set -euo pipefail
export PATH="/opt/slurm/Ubuntu-20.04/24.11.0/bin:/nethome/${USER}/.pixi/bin:\$PATH"

echo "[INFO] Host: \$(hostname)"
echo "[INFO] Job: \${SLURM_JOB_ID:-unknown}"
echo "[INFO] Partition/QOS/GRES: ${partition}/${qos}/${gpu_gres}"
echo "[INFO] Workspace: ${workspace_root}"
echo "[INFO] Pixi env: ${pixi_env}"
echo "[INFO] GPU status before job"
nvidia-smi || true

cd ${quoted_workspace_root}

export PIXI_CACHE_DIR=${quoted_pixi_cache_dir}
export HF_HOME="${data_dir}/hf-home"
export HUGGINGFACE_HUB_CACHE="${data_dir}/hf-cache"
export SENTENCE_TRANSFORMERS_HOME="${data_dir}/sentence-transformers"
export WANDB_DIR="${workspace_root}/logs/wandb"
export WANDB_CACHE_DIR="${data_dir}/wandb-cache"
export TMPDIR="${data_dir}/tmp/isaaclab-\${SLURM_JOB_ID:-pixi}"
export ISAACLAB_IMITATION_UNITREE_USD_CACHE_ROOT="${data_dir}/tmp/isaaclab_unitree_usd"
export TERM=xterm
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export TORCHDYNAMO_DISABLE=1
export RL_WARNINGS=False
export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y
export OMNI_KIT_ACCEPT_EULA=YES
mkdir -p "\$PIXI_CACHE_DIR" "\$HF_HOME" "\$HUGGINGFACE_HUB_CACHE" "\$SENTENCE_TRANSFORMERS_HOME" "\$WANDB_DIR" "\$WANDB_CACHE_DIR" "\$TMPDIR" "\$ISAACLAB_IMITATION_UNITREE_USD_CACHE_ROOT"

hf_token_file=${quoted_hf_token_file}
wandb_key_file=${quoted_wandb_key_file}
if [ -n "\$hf_token_file" ] && [ -f "\$hf_token_file" ]; then
    export HF_TOKEN="\$(tr -d '\r\n' < "\$hf_token_file")"
    export HUGGINGFACE_HUB_TOKEN="\$HF_TOKEN"
    echo "[INFO] Loaded Hugging Face token."
else
    echo "[INFO] No Hugging Face token file configured."
fi
if [ -n "\$wandb_key_file" ] && [ -f "\$wandb_key_file" ]; then
    export WANDB_API_KEY="\$(tr -d '\r\n' < "\$wandb_key_file")"
    echo "[INFO] Loaded W&B API key."
else
    echo "[INFO] No W&B API key file configured."
fi

echo "[INFO] Installing Pixi environment if needed."
pixi install --locked -e ${quoted_pixi_env}

echo "[INFO] Launching workload."
set +e
pixi run -e ${quoted_pixi_env} python ${quoted_python_executable} ${quoted_job_args}
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
