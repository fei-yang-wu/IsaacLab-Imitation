#!/usr/bin/env bash
set -euo pipefail

# A one-variable ablation against W&B run r09s1pc7. This deliberately reuses
# that run's immutable submitted workspace archive and changes only the critic
# command channel from (actor, reference) to (reference). The actor still sees
# the frozen 258-D DiffSR command.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
while [[ ! -x "${REPO_ROOT}/docker/cluster/cluster_interface.sh" ]]; do
    [[ "${REPO_ROOT}" != "/" ]] || { echo "[FATAL] repository root not found" >&2; exit 2; }
    REPO_ROOT="$(dirname "${REPO_ROOT}")"
done
cd "${REPO_ROOT}"

MODE="${MODE:-print}"
CONFIRM_TOKEN="critic-no-latent"
[[ "${MODE}" == "print" || "${MODE}" == "validate" || "${MODE}" == "submit" ]] \
    || { echo "[FATAL] MODE must be print, validate, or submit" >&2; exit 2; }
if [[ "${MODE}" == "submit" && "${CONFIRM_SUBMIT:-}" != "${CONFIRM_TOKEN}" ]]; then
    echo "[FATAL] submission requires CONFIRM_SUBMIT=${CONFIRM_TOKEN}" >&2
    exit 2
fi

ICE_HOST="${ICE_HOST:-ice}"
BASELINE_RUN_ID="r09s1pc7"
BASELINE_JOB_ID="5567801"
BASELINE_WORKSPACE="/storage/ice1/3/2/fwu91/Research/IsaacLab/isaaclab_20260805_210549"
BASELINE_ARCHIVE_SHA256="e20e93be390a9985df0472893f20ce2b68050dd12a89366743a9dfc66f951d05"
REMOTE_DATA_ROOT="/home/hice1/fwu91/scratch/Research/IsaacLab/data"
RUN_TAG="bones129k_z256_critic_no_latent_e16384_r24_5b_seed0"
OUTPUT_DIR="/data/bones129k_critic_ablation/${RUN_TAG}/rlopt_train"
REMOTE_OUTPUT_DIR="${REMOTE_DATA_ROOT}/bones129k_critic_ablation/${RUN_TAG}/rlopt_train"
SUBMISSION_DIR="/storage/ice1/3/2/fwu91/Research/IsaacLab/critic_no_latent_20260806"
SLURM_OUTPUT_DIR="${SUBMISSION_DIR}/logs/slurm"

REMOTE_SUBMIT_SCRIPT="${BASELINE_WORKSPACE}/docker/cluster/submit_job_slurm_pace.sh"
REMOTE_ARGS=(
    "${BASELINE_WORKSPACE}" isaac-lab-base
    --task Isaac-Imitation-G1-v2 --num_envs 16384 --headless
    --algo IPMD --agent rlopt_ipmd_tuned_cfg_entry_point --seed 0
    --max_iterations 12716
    --kit_args=--/app/extensions/fsWatcherEnabled=false
    physics=newton_mjwarp
    env.data.manifest=null
    env.data.reference_arrays_dir=/data/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1
    env.data.persist_id=bones_seed_sonic_full_129785@e714bbff
    env.data.reference_arrays_resident=true
    env.data.reference_arrays_warm_workers=16
    env.data.runtime_cache_device=cpu
    env.data.reference_prefetch_mode=next
    env.data.macro_cache_device=cuda:0
    env.data.runtime_cache_body_names=[pelvis,left_hip_roll_link,left_knee_link,left_ankle_roll_link,right_hip_roll_link,right_knee_link,right_ankle_roll_link,torso_link,left_shoulder_roll_link,left_elbow_link,left_wrist_yaw_link,right_shoulder_roll_link,right_elbow_link,right_wrist_yaw_link]
    agent.collector.frames_per_batch=24
    agent.loss.mini_batch_size=294912
    agent.ipmd.expert_batch_size=24576
    agent.loss.gamma=0.97
    agent.save_interval=50000000
    agent.logger.backend=wandb
    agent.logger.video=false
    agent.logger.project_name=g1-bones-seed
    agent.logger.group_name=bones129k-ablation
    "agent.logger.exp_name=${RUN_TAG}"
    "agent.logger.log_dir=${OUTPUT_DIR}"
    agent.ipmd.hl_skill_checkpoint_path=/data/pretrain_store/bones129k_v2_root_qpos_det_sr_h10_z256_seed0/checkpoints/latest.pt
    env.expert_macro_state_terms=[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b]
    env.rewards.action_rate_l2.weight=0.0
    env.rewards.tracking_reward_points.weight=4.0
    env.enable_termination_curriculum=true
    env.termination_curriculum_start_frames=5000000
    env.termination_curriculum_end_frames=30000000
    env.command_interface.reference.selection=random80_adaptive20
    env.sim.physics.solver_cfg.njmax=289
    env.sim.physics.solver_cfg.nconmax=200
    env.command_interface.actor=latent
    env.command_interface.actor.dim=258
    env.command_interface.encoder=single
    env.command_interface.critic_channels=[reference]
    agent.ipmd.latent_dim=258
    agent.ipmd.command_source=hl_skill
    agent.ipmd.hl_skill_horizon_steps=10
    agent.ipmd.hl_skill_command_mode=z
    agent.ipmd.latent_steps_min=10
    agent.ipmd.latent_steps_max=10
    agent.ipmd.latent_learning.code_period=10
    agent.ipmd.latent_learning.command_phase_mode=sin_cos
    agent.ipmd.latent_learning.code_latent_dim=256
    agent.ipmd.hl_skill_finetune_enabled=false
)

ENV_ARGS=(
    CLUSTER_ISAACLAB_DIR=/home/hice1/fwu91/scratch/Research/IsaacLab/isaaclab
    CLUSTER_PROJECT_LOGS_DIR=/home/hice1/fwu91/scratch/Research/IsaacLab/isaaclab/logs
    CLUSTER_SIF_PATH=/home/hice1/fwu91/scratch/Research/IsaacLab/isaaclabsif
    CLUSTER_SHARED_SIF_PATH=/home/hice1/fwu91/scratch/Research/IsaacLab/isaaclabsif/isaaclab-runtime-3.0.0b2-cu130.sif
    CLUSTER_DATA_DIR=/home/hice1/fwu91/scratch/Research/IsaacLab/data
    CLUSTER_ISAAC_SIM_CACHE_DIR=/home/hice1/fwu91/scratch/Research/IsaacLab/cache
    CLUSTER_JOB_TMPDIR_ROOT=/tmp
    CLUSTER_HF_TOKEN_FILE=/home/hice1/fwu91/.hf_token
    CLUSTER_WANDB_API_KEY_FILE=/home/hice1/fwu91/.wandb_api_key
    CLUSTER_AUTO_SETUP_G1_DATA=0
    CLUSTER_G1_MANIFEST_REFRESH_POLICY=never
    CLUSTER_USE_OVERLAY=0
    CLUSTER_USE_SHARED_SIF=1
    CLUSTER_CU130_RUNTIME_ROOT=/opt/isaaclab-imitation-runtime-spec/.pixi/envs/container-runtime
    CLUSTER_EXTRA_PYTHONPATH_REL=IsaacLab/source/isaaclab:IsaacLab/source/isaaclab_tasks:IsaacLab/source/isaaclab_assets:IsaacLab/source/isaaclab_rl:IsaacLab/source/isaaclab_mimic:source/isaaclab_imitation:RLOpt:ImitationLearningTools
    REMOVE_CODE_COPY_AFTER_JOB=false
    REMOVE_OVERLAY_AFTER_JOB=true
    CLUSTER_SLURM_TIME_LIMIT=15:59:00
    CLUSTER_SLURM_PARTITION=coe-gpu
    CLUSTER_SLURM_QOS=coe-ice
    CLUSTER_SLURM_GPU_GRES=gpu:h200:1
    CLUSTER_SLURM_CPUS_PER_TASK=16
    CLUSTER_SLURM_MEM=160G
    CLUSTER_SLURM_JOB_NAME_PREFIX=b129k-z256-critic-nolatent
    "CLUSTER_SLURM_OUTPUT_DIR=${SLURM_OUTPUT_DIR}"
    CLUSTER_SLURM_KEEP_JOB_SCRIPT=1
    CLUSTER_G1_USD_PATH=repo
    CLUSTER_SIM_BACKEND=newton
    CLUSTER_PYTHON_EXECUTABLE=scripts/rlopt/train.py
    CLUSTER_APPEND_DEFAULT_G1_MANIFEST=0
    CLUSTER_SKIP_CACHE_COPY=1
    CLUSTER_WANDB_TAGS=bones-seed,129785,v2,reset80-adaptive20,rollout24,gamma097,diffsr,root-qpos,z256,h10,hold10,frozen-encoder,critic-no-latent,critic-reference-only,newton
)

printf '[INFO] baseline W&B run: %s; ICE job: %s\n' "${BASELINE_RUN_ID}" "${BASELINE_JOB_ID}"
printf '[INFO] one change: critic_channels=(actor,reference) -> (reference)\n'
printf '[INFO] exact baseline archive: %s\n' "${BASELINE_ARCHIVE_SHA256}"
printf '[INFO] 5B frames; H200 on coe-gpu; W&B g1-bones-seed/bones129k-ablation\n'
printf '[CMD] ssh %q mkdir -p %q and run submit script with:\n' "${ICE_HOST}" "${SUBMISSION_DIR}"
printf '      env '; printf '%q ' "${ENV_ARGS[@]}"; printf '\n      bash %q ' "${REMOTE_SUBMIT_SCRIPT}"; printf '%q ' "${REMOTE_ARGS[@]}"; printf '\n'

[[ "${MODE}" == "print" ]] && exit 0

remote_sha="$(ssh -o BatchMode=yes -o ConnectTimeout=20 "${ICE_HOST}" \
    "sha256sum '${BASELINE_WORKSPACE}/workspace.tar.gz' | awk '{print \$1}'")"
[[ "${remote_sha}" == "${BASELINE_ARCHIVE_SHA256}" ]] \
    || { echo "[FATAL] baseline workspace hash mismatch: ${remote_sha}" >&2; exit 1; }
fresh="$(ssh -o BatchMode=yes -o ConnectTimeout=20 "${ICE_HOST}" \
    "if [[ -e '${REMOTE_OUTPUT_DIR}' ]]; then echo no; else echo yes; fi")"
[[ "${fresh}" == "yes" ]] || { echo "[FATAL] refusing existing output: ${REMOTE_OUTPUT_DIR}" >&2; exit 1; }
echo "[PASS] immutable workspace and fresh output"

[[ "${MODE}" == "validate" ]] && exit 0

printf -v env_cmd '%q ' "${ENV_ARGS[@]}"
printf -v args_cmd '%q ' "${REMOTE_ARGS[@]}"
printf -v remote_cmd 'mkdir -p %q %q && cd %q && env %sbash %q %s' \
    "${SUBMISSION_DIR}" "${SLURM_OUTPUT_DIR}" "${SUBMISSION_DIR}" \
    "${env_cmd}" "${REMOTE_SUBMIT_SCRIPT}" "${args_cmd}"
ssh -o BatchMode=yes -o ConnectTimeout=20 "${ICE_HOST}" "${remote_cmd}"
