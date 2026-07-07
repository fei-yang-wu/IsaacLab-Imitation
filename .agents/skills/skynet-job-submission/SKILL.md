---
name: skynet-job-submission
description: Skynet Slurm job workflow for this IsaacLab-Imitation repo, including ssh access, scheduler commands, Apptainer/Pixi runner selection, environment files, log monitoring, cancellation, and safety checks. Use when the user mentions Skynet, sky1, wu-lab, Slurm submissions on Skynet, Apptainer on Skynet compute nodes, or asks to submit/monitor/cancel IsaacLab-Imitation jobs on Skynet.
---

# Skynet Job Submission

Use this skill for Skynet cluster work from the `IsaacLab-Imitation` repo.

## Non-Negotiables

- Do not submit a new Slurm job until the user explicitly confirms the exact command or action.
- Treat Apptainer/Singularity as compute-node-only on Skynet. Do not conclude it is unavailable from the login node; test it only inside a Slurm allocation or submitted diagnostic job, after user confirmation.
- Use the `skynet` SSH alias. Add Slurm to PATH on remote commands:

```bash
ssh skynet 'export PATH=/opt/slurm/Ubuntu-20.04/24.11.0/bin:$PATH; squeue -u $USER'
```

- Keep large data, caches, SIFs, and logs under `/coc/flash12/fwu91/Research/IsaacLab`, not `/nethome/fwu91`.
- Never print token contents. Skynet token files are expected at `/nethome/fwu91/.hf_token` and `/nethome/fwu91/.wandb_api_key`.

## Current Repo Files

- Skynet config: `docker/cluster/.env.skynet`
- Apptainer-style submitter: `docker/cluster/submit_job_slurm_skynet.sh`
- Pixi fallback submitter: `docker/cluster/submit_job_slurm_skynet_pixi.sh`
- Container runtime launcher: `docker/cluster/run_singularity.sh`
- Cluster interface: `docker/cluster/cluster_interface.sh`
- BONES-SEED paper24 wrapper: `experiments/submit_bones_seed_paper24_skynet_2b.sh`

`docker/cluster/.env.skynet` should point at Skynet paths:

```bash
CLUSTER_LOGIN=skynet
CLUSTER_ISAACLAB_DIR=/coc/flash12/fwu91/Research/IsaacLab/isaaclab
CLUSTER_SIF_PATH=/coc/flash12/fwu91/Research/IsaacLab/isaaclabsif
CLUSTER_DATA_DIR=/coc/flash12/fwu91/Research/IsaacLab/data
CLUSTER_ISAAC_SIM_CACHE_DIR=/coc/flash12/fwu91/Research/IsaacLab/cache
CLUSTER_HF_TOKEN_FILE=/nethome/fwu91/.hf_token
CLUSTER_WANDB_API_KEY_FILE=/nethome/fwu91/.wandb_api_key
CLUSTER_SLURM_PARTITION=wu-lab
CLUSTER_SLURM_QOS=short
CLUSTER_SLURM_GPU_GRES=gpu:a40:1
CLUSTER_SLURM_CPUS_PER_TASK=8
CLUSTER_SLURM_MEM=96G
CLUSTER_SLURM_TIME_LIMIT=2-00:00:00
```

## Before Submitting

1. Inspect current jobs:

```bash
ssh skynet 'export PATH=/opt/slurm/Ubuntu-20.04/24.11.0/bin:$PATH; squeue -u $USER -o "%i %T %M %j %R"'
```

2. Check whether a previous BONES or IsaacLab snapshot already has useful logs:

```bash
ssh skynet 'ls -1dt /coc/flash12/fwu91/Research/IsaacLab/isaaclab_*/logs/slurm/* 2>/dev/null | head'
```

3. Validate local shell files before proposing submission:

```bash
bash -n docker/cluster/cluster_interface.sh docker/cluster/run_singularity.sh docker/cluster/submit_job_slurm_skynet.sh docker/cluster/submit_job_slurm_skynet_pixi.sh
```

4. Print a dry run when a wrapper supports it:

```bash
DRY_RUN=1 experiments/submit_bones_seed_paper24_skynet_2b.sh
```

5. Present the exact command and ask for confirmation before running with `DRY_RUN=0` or before invoking `cluster_interface.sh ... job`.

## Runner Selection

Prefer the Apptainer-style runner when the user says Apptainer is available on compute nodes or when testing inside Slurm confirms it:

```bash
CLUSTER_SLURM_SUBMIT_SCRIPT=skynet
CLUSTER_SKIP_SINGULARITY_IMAGE_CHECK=0
```

Use the Pixi fallback only when Apptainer cannot be used for the target job:

```bash
CLUSTER_SLURM_SUBMIT_SCRIPT=skynet_pixi
CLUSTER_SKIP_SINGULARITY_IMAGE_CHECK=1
CLUSTER_PIXI_ENV=isaaclab
CLUSTER_PIXI_CACHE_DIR=/coc/flash12/fwu91/Research/IsaacLab/pixi-cache
```

The Apptainer image directory already exists on Skynet:

```bash
/coc/flash12/fwu91/Research/IsaacLab/isaaclabsif
```

Login-node `command -v apptainer` may fail. That is not decisive; Apptainer is compute-node-only for this workflow.

## Compute-Node Diagnostic

Only run this after user confirmation because it submits a Slurm job:

```bash
ssh skynet 'export PATH=/opt/slurm/Ubuntu-20.04/24.11.0/bin:$PATH; tmp=$(mktemp -d /coc/flash12/fwu91/Research/IsaacLab/diag-apptainer.XXXXXX); cat > "$tmp/diag.sbatch" <<'"'"'EOF'"'"'
#!/bin/bash
#SBATCH --job-name=skynet-apptainer-diag
#SBATCH --output=diag_%j.log
#SBATCH --error=diag_%j.log
#SBATCH --partition=wu-lab
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:05:00
set -x
hostname
echo "PATH=$PATH"
for x in apptainer singularity nvidia-smi; do
  echo "CMD:$x"
  command -v "$x" || true
  "$x" --version 2>&1 | head -3 || true
done
EOF
cd "$tmp"; sbatch < diag.sbatch; pwd'
```

After it completes, read the returned `diag_<jobid>.log`.

## Submit Pattern

Use the cluster interface from the repo root. Example for the BONES-SEED wrapper:

```bash
DRY_RUN=0 experiments/submit_bones_seed_paper24_skynet_2b.sh
```

Example for pretrain-only BONES-SEED pipeline:

```bash
DRY_RUN=0 EXTRA_PIPELINE_ARGS='--skip-low-level --skip-rollout-ft' \
  experiments/submit_bones_seed_paper24_skynet_2b.sh
```

Before submitting a full low-level policy job, confirm budget, walltime, job name, runner, data root, and whether existing pretrain checkpoints should be reused.

## Monitoring

Use `squeue` and Slurm logs first:

```bash
ssh skynet 'export PATH=/opt/slurm/Ubuntu-20.04/24.11.0/bin:$PATH; squeue -j <jobid> -o "%i %T %M %j %R"'
ssh skynet 'tail -200 /coc/flash12/fwu91/Research/IsaacLab/<snapshot>/logs/slurm/<log>.log'
```

Use `sacct` for terminal state:

```bash
ssh skynet 'export PATH=/opt/slurm/Ubuntu-20.04/24.11.0/bin:$PATH; sacct -j <jobid> --format=JobID,State,Elapsed,ExitCode -P'
```

When process state matters, use `srun --overlap` inside the allocation:

```bash
ssh skynet 'export PATH=/opt/slurm/Ubuntu-20.04/24.11.0/bin:$PATH; srun --jobid=<jobid> --overlap --ntasks=1 bash -lc "ps -u fwu91 -o pid,ppid,stat,etime,pcpu,pmem,args | head -80"'
```

## Cancellation

Only cancel when the user asks or explicitly approves:

```bash
ssh skynet 'export PATH=/opt/slurm/Ubuntu-20.04/24.11.0/bin:$PATH; scancel <jobid>; sacct -j <jobid> --format=JobID,State,Elapsed,ExitCode -P'
```

## Known Failure Modes

- `apptainer: command not found` on login node: expected; test on compute node.
- Docker exists on compute nodes but may deny `/var/run/docker.sock`; do not rely on Docker unless access is confirmed.
- Pixi direct Isaac Sim jobs need EULA env vars:

```bash
ACCEPT_EULA=Y
PRIVACY_CONSENT=Y
OMNI_KIT_ACCEPT_EULA=YES
```

- Container jobs need Skynet paths propagated through both `cluster_interface.sh` and `run_singularity.sh`; avoid sourcing `.env.cluster` in a way that overwrites `.env.skynet`.
