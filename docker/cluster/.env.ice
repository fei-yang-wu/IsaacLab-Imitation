# PACE-ICE profile (Singularity/apptainer path) for G1 jobs.
# Used via: ./docker/cluster/cluster_interface.sh -c ice job <entry args>
#
# ICE specifics vs skynet:
#  - Runs inside the isaac-lab-base Singularity container (not pixi).
#  - GPU walltime is capped at 16h on every GPU partition.
#  - Requires --account/--qos; coe-gpu (h100/h200) via account=cse, qos=coe-ice.
#  - CLUSTER_DATA_DIR is mounted at /data inside the container; pass --data-root /data/...
#  - Dataset npz are rsync-excluded; stage them separately to CLUSTER_DATA_DIR.
CLUSTER_JOB_SCHEDULER=SLURM
CLUSTER_LOGIN=ice
CLUSTER_ISAACLAB_DIR=/home/hice1/fwu91/scratch/Research/IsaacLab/isaaclab
CLUSTER_SIF_PATH=/home/hice1/fwu91/scratch/Research/IsaacLab/isaaclabsif
# Reuse a once-extracted sandbox instead of re-extracting the 19GB / 198k-file
# tar into a per-job TMPDIR every run (slow on Lustre). Pre-extract with:
#   cd $CLUSTER_SIF_PATH && tar -xf isaac-lab-base.tar   # creates isaac-lab-base.sif/
CLUSTER_USE_SHARED_SIF=1
CLUSTER_SHARED_SIF_PATH=/home/hice1/fwu91/scratch/Research/IsaacLab/isaaclabsif/isaac-lab-base.sif
CLUSTER_DATA_DIR=/home/hice1/fwu91/scratch/Research/IsaacLab/data
CLUSTER_ISAAC_SIM_CACHE_DIR=/home/hice1/fwu91/scratch/Research/IsaacLab/docker-isaac-sim
CLUSTER_JOB_TMPDIR_ROOT=/home/hice1/fwu91/scratch/Research/IsaacLab/job_tmp
CLUSTER_HF_TOKEN_FILE=/home/hice1/fwu91/.hf_token
CLUSTER_WANDB_API_KEY_FILE=/home/hice1/fwu91/.wandb_api_key
CLUSTER_AUTO_SETUP_G1_DATA=0
CLUSTER_APPEND_DEFAULT_G1_MANIFEST=0
CLUSTER_G1_MANIFEST_REFRESH_POLICY=never
CLUSTER_PYTHON_EXECUTABLE=scripts/rlopt/run_bones_seed_100_pretrain_lowlevel.py
CLUSTER_RLOPT_LOCAL_PATH=/mnt/hsstorage/fwu91/Projects/SL/IsaacLab-Imitation/RLOpt
CLUSTER_IMITATION_TOOLS_LOCAL_PATH=/mnt/hsstorage/fwu91/Projects/SL/IsaacLab-Imitation/ImitationLearningTools
CLUSTER_SLURM_SUBMIT_SCRIPT=pace
# ICE auto-assigns partition/QOS/account from the requested GPU (gres). Do NOT
# pin the partition -- forcing partition=coe-gpu restricted jobs to h100/h200,
# which lack RT cores and cannot run Isaac Sim ("No device could be created").
# Leave these unset; request an RT-core GPU by name and SLURM routes to ice-gpu.
# L40S (48GB, RT cores) fits <=4096x48; the RTX PRO 6000 Blackwell (96GB) is
# excluded by this container's Isaac Sim 5.1 (needs a newer image for the scaled
# 8192x32 config -- run that locally for now).
CLUSTER_SLURM_GPU_GRES=gpu:L40S:1
CLUSTER_SLURM_CPUS_PER_TASK=16
CLUSTER_SLURM_MEM=96G
CLUSTER_SLURM_TIME_LIMIT=16:00:00
CLUSTER_SLURM_JOB_NAME_PREFIX=bones100-ice
CLUSTER_SLURM_OUTPUT_DIR=logs/slurm
CLUSTER_SLURM_PRINT_JOB_SCRIPT=1
REMOVE_CODE_COPY_AFTER_JOB=false
REMOVE_OVERLAY_AFTER_JOB=true
