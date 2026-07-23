#!/usr/bin/env bash
set -euo pipefail

# Resumable corrected-LAFAN1 explicit-chunk tracker jobs at a 5B-frame cap,
# chained across Slurm segments (every ICE QoS caps 1-GPU jobs at 16h via
# MaxTRESMins gres/gpu=960). Sibling of
# experiments/submit_lafan1_5b_resumable_ice.sh (the latent arm); see that
# script and experiments/submit_bones_seed_sonic_5b_resumable_ice.sh for the
# cross-segment plumbing rationale (per-submission isaaclab_<timestamp>/
# workspace dirs, exp_name-filtered resume scan, /data-bind staging of the
# resume checkpoint).
#
# Interface-ablation arms (one Slurm chain per command space):
#   full_body_trajectory  - held 10-step full-body command chunks
#   ee_trajectory         - held 10-step end-effector command chunks
#
# TASK is Isaac-Imitation-G1-Strict-v0: the vanilla observation/agent
# contract on the same protocol surface as the strict latent default (pelvis
# anchor, strict SONIC terminations, [0, 200] reset starts), so these
# trackers differ from the running latent 5B arm only in the command space.
# Both arms train under the held-window consumption contract
# (env.command_hold_steps=10, matching the from-scratch redo protocol) so
# there is no train/deploy mismatch when a planner later publishes chunks at
# 5 Hz. The learned-reward IPMD terms are zeroed (pure task-reward PPO),
# matching the latent arm's objective structure.
#
# Scale mirrors the validated latent 5B config: 12288 envs x 12 rollout
# steps, minibatch 18432, njmax=320/nconmax=40, ice-gpu H100 80 GB.
# Qualification for ablation rows is training plateau (user decision
# 2026-07-21), not a survival gate.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

SEED="${SEED:-0}"
DRY_RUN="${DRY_RUN:-1}"
FRAME_CAP=5000000000
TRAIN_NUM_ENVS=12288
ROLLOUT_STEPS=12
MINIBATCH_SIZE=18432
NJMAX=320
NCONMAX=40
FRAMES_PER_BATCH=$((TRAIN_NUM_ENVS * ROLLOUT_STEPS))
HORIZON_STEPS="${HORIZON_STEPS:-10}"
COMMAND_SPACES="${COMMAND_SPACES:-full_body_trajectory ee_trajectory}"
TASK_NAME="${TASK_NAME:-Isaac-Imitation-G1-Strict-v0}"
MANIFEST_PATH="${MANIFEST_PATH:-/data/lafan1_corrected_8e95d557/manifests/g1_lafan1_manifest.json}"
EXPECTED_MANIFEST_SHA256="${EXPECTED_MANIFEST_SHA256:-d972c37c41dadbb68c30fc456a9dc9c1bd6d30ed0b7aa9d34b1797472c945db8}"
EXPECTED_NPZ_COUNT="${EXPECTED_NPZ_COUNT:-40}"
REMOTE_PROJECT_ROOT="${REMOTE_PROJECT_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/isaaclab}"
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/data}"
REMOTE_ICE_BASE="$(dirname "${REMOTE_PROJECT_ROOT}")"
WANDB_PROJECT="${WANDB_PROJECT:-g1-lafan1-strict}"
WANDB_GROUP="${WANDB_GROUP:-chunk-trackers-e12288-5b-resumable-jointfix}"

case "${DRY_RUN}" in
    1|true|TRUE|yes|YES|on|ON) is_dry_run=1 ;;
    0|false|FALSE|no|NO|off|OFF) is_dry_run=0 ;;
    *)
        echo "[ERROR] DRY_RUN must be a boolean; got '${DRY_RUN}'." >&2
        exit 2
        ;;
esac

short_space_name() {
    case "$1" in
        full_body_trajectory) printf 'fb' ;;
        ee_trajectory) printf 'ee' ;;
        *)
            echo "[ERROR] Unsupported command space '$1'. Allowed: full_body_trajectory ee_trajectory." >&2
            exit 2
            ;;
    esac
}

if [[ "${is_dry_run}" == "0" ]]; then
    actual_remote_sha="$(ssh -o BatchMode=yes -o ConnectTimeout=10 ice "sha256sum '${REMOTE_DATA_ROOT}/lafan1_corrected_8e95d557/manifests/g1_lafan1_manifest.json'" | awk '{print $1}')"
    remote_npz_count="$(ssh -o BatchMode=yes -o ConnectTimeout=10 ice "find '${REMOTE_DATA_ROOT}/lafan1_corrected_8e95d557' -type f -name '*.npz' | wc -l")"
    if [[ "${actual_remote_sha}" != "${EXPECTED_MANIFEST_SHA256}" || "${remote_npz_count}" != "${EXPECTED_NPZ_COUNT}" ]]; then
        echo "[ERROR] ICE corrected-LAFAN1 data gate failed: sha=${actual_remote_sha}, npz=${remote_npz_count}." >&2
        exit 2
    fi
fi

submit_arm() {
    local command_space="$1"
    local space_short
    space_short="$(short_space_name "${command_space}")"
    local run_tag="${RUN_TAG:-lafan1_strict_${space_short}chunk_h${HORIZON_STEPS}_5b_seed${SEED}_20260721_jointfix_e12288_r12_nj320_nc40}"
    local exp_name="${run_tag}_oracle_low_level"
    local resume_ckpt_container="/data/resume_store/${run_tag}/model_resume.pt"

    # --- Resume detection + checkpoint staging (no pretrain stage here) ---
    local cumulative_frames=0
    local latest_checkpoint=""
    if [[ "${is_dry_run}" == "0" ]]; then
        local resume_state
        resume_state="$(ssh -o BatchMode=yes -o ConnectTimeout=10 ice bash -s -- \
            "${REMOTE_ICE_BASE}" "${TASK_NAME}" "${exp_name}" "${run_tag}" \
            "${REMOTE_DATA_ROOT}" <<'REMOTE_EOF'
set -uo pipefail
ice_base="$1"
task_name="$2"
exp_name="$3"
run_tag="$4"
data_root="$5"
resume_store="${data_root}/resume_store/${run_tag}"
state_file="${resume_store}/resume_state.tsv"

cumulative=0
last_counted=""
if [[ -f "${state_file}" ]]; then
    IFS=$'\t' read -r cumulative last_counted < "${state_file}"
fi

# Scan every per-submission workspace; keep only run dirs whose recorded
# command carries THIS run's exp_name.
run_dirs="$(grep -ls -- "agent.logger.exp_name=${exp_name}" \
    "${ice_base}"/isaaclab*/logs/rlopt/ipmd/"${task_name}"/*/command.txt 2>/dev/null \
    | xargs -r -n1 dirname)"
latest=""
if [[ -n "${run_dirs}" ]]; then
    latest="$(find ${run_dirs} -name 'model_step_*.pt' -printf '%T@\t%p\n' 2>/dev/null \
        | sort -n -k1,1 | tail -1 | cut -f2-)"
fi

if [[ -n "${latest}" && "${latest}" != "${last_counted}" ]]; then
    segment_dir="$(dirname "${latest}")"
    segment_frames="$(find "${segment_dir}" -name 'model_step_*.pt' \
        | sed -E 's#.*model_step_([0-9]+)\.pt#\1#' | sort -n | tail -1)"
    cumulative=$((cumulative + segment_frames))
    mkdir -p "${resume_store}"
    printf '%s\t%s\n' "${cumulative}" "${latest}" > "${state_file}"
fi

# Stage the checkpoint where the next job's container can see it (/data bind).
if [[ -n "${latest}" ]]; then
    mkdir -p "${resume_store}"
    cp -f "${latest}" "${resume_store}/model_resume.pt"
fi

printf '%s\t%s\n' "${cumulative}" "${latest}"
REMOTE_EOF
        )"
        cumulative_frames="$(printf '%s' "${resume_state}" | cut -f1)"
        latest_checkpoint="$(printf '%s' "${resume_state}" | cut -f2-)"
        cumulative_frames="${cumulative_frames:-0}"
    fi

    local max_iterations
    local checkpoint_args=()
    if [[ -n "${latest_checkpoint}" ]]; then
        local remaining_frames=$((FRAME_CAP - cumulative_frames))
        if (( remaining_frames <= 0 )); then
            echo "[INFO] ${run_tag} already reached FRAME_CAP=${FRAME_CAP} (cumulative ${cumulative_frames} frames). Not submitting."
            return 0
        fi
        max_iterations=$(( (remaining_frames + FRAMES_PER_BATCH - 1) / FRAMES_PER_BATCH ))
        echo "[INFO] Resuming ${run_tag} from ${latest_checkpoint} (${cumulative_frames}/${FRAME_CAP} cumulative frames done; ${max_iterations} iterations remaining this segment)."
        checkpoint_args=(--checkpoint "${resume_ckpt_container}")
    else
        max_iterations=$(( (FRAME_CAP + FRAMES_PER_BATCH - 1) / FRAMES_PER_BATCH ))
        echo "[INFO] No checkpoint found for ${run_tag}; submitting a fresh first segment (${max_iterations} iterations for ${FRAME_CAP} frames)."
    fi

    export CLUSTER_LOGIN="${CLUSTER_LOGIN:-login-ice.pace.gatech.edu}"
    export CLUSTER_SLURM_SUBMIT_SCRIPT="${CLUSTER_SLURM_SUBMIT_SCRIPT:-pace}"
    export CLUSTER_PYTHON_EXECUTABLE="scripts/rlopt/train.py"
    export CLUSTER_APPEND_DEFAULT_G1_MANIFEST=0
    export CLUSTER_G1_MANIFEST_REFRESH_POLICY="${CLUSTER_G1_MANIFEST_REFRESH_POLICY:-auto}"
    export CLUSTER_SLURM_TIME_LIMIT="${CLUSTER_SLURM_TIME_LIMIT:-15:59:00}"
    export CLUSTER_SLURM_PARTITION="${CLUSTER_SLURM_PARTITION:-ice-gpu}"
    export CLUSTER_SLURM_QOS="${CLUSTER_SLURM_QOS:-coe-ice}"
    export CLUSTER_SLURM_GPU_GRES="${CLUSTER_SLURM_GPU_GRES:-gpu:h100:1}"
    export CLUSTER_SLURM_CPUS_PER_TASK=16
    export CLUSTER_SLURM_MEM=96G
    export CLUSTER_SLURM_JOB_NAME_PREFIX="lafan1-${space_short}-5b-resume"
    export CLUSTER_GIT_SYNC_FIRST="${CLUSTER_GIT_SYNC_FIRST:-0}"
    export CLUSTER_G1_USD_PATH=repo

    local cmd=(./docker/cluster/cluster_interface.sh -c ice_runtime job
        --task "${TASK_NAME}"
        --num_envs "${TRAIN_NUM_ENVS}"
        --headless
        --algo IPMD
        --max_iterations "${max_iterations}"
        --kit_args=--/app/extensions/fsWatcherEnabled=false
    )
    if (( ${#checkpoint_args[@]} > 0 )); then
        cmd+=("${checkpoint_args[@]}")
    fi
    cmd+=(
        physics=newton_mjwarp
        "env.sim.physics.solver_cfg.njmax=${NJMAX}"
        "env.sim.physics.solver_cfg.nconmax=${NCONMAX}"
        "env.lafan1_manifest_path=${MANIFEST_PATH}"
        env.refresh_zarr_dataset=false
        env.latent_patch_past_steps=0
        "env.latent_patch_future_steps=${HORIZON_STEPS}"
        "env.command_hold_steps=${HORIZON_STEPS}"
        env.command_observation_source=reference
        "agent.seed=${SEED}"
        "agent.command_space=${command_space}"
        agent.ipmd.use_latent_command=false
        "agent.collector.frames_per_batch=${ROLLOUT_STEPS}"
        "agent.loss.mini_batch_size=${MINIBATCH_SIZE}"
        "agent.value_function.num_cells=[768,512,256]"
        agent.ipmd.reward_loss_coeff=0.0
        agent.ipmd.reward_l2_coeff=0.0
        agent.ipmd.reward_grad_penalty_coeff=0.0
        agent.ipmd.reward_logit_reg_coeff=0.0
        agent.ipmd.reward_param_weight_decay_coeff=0.0
        agent.ipmd.use_estimated_rewards_for_ppo=false
        agent.ipmd.env_reward_weight=1.0
        agent.ipmd.bc_coef=0.0
        agent.ipmd.rollout_bc_coef=0.0
        agent.save_interval=100000000
        agent.logger.backend=wandb
        "agent.logger.project_name=${WANDB_PROJECT}"
        "agent.logger.group_name=${WANDB_GROUP}"
        "agent.logger.exp_name=${exp_name}"
    )

    echo "[INFO] Arm ${command_space}: task='${TASK_NAME}' run_tag='${run_tag}' max_iterations='${max_iterations}' frames_per_batch='${FRAMES_PER_BATCH}'"
    printf "[CMD] "
    printf "%q " "${cmd[@]}"
    printf "\n"

    if [[ "${is_dry_run}" == "1" ]]; then
        echo "[INFO] DRY_RUN=${DRY_RUN}; not contacting the cluster."
        return 0
    fi
    "${cmd[@]}"
}

read -r -a COMMAND_SPACE_LIST <<< "${COMMAND_SPACES}"
if [[ -n "${RUN_TAG:-}" && "${#COMMAND_SPACE_LIST[@]}" -gt 1 ]]; then
    echo "[ERROR] RUN_TAG overrides the per-arm tag; use it only with a single COMMAND_SPACES entry." >&2
    exit 2
fi
for command_space in "${COMMAND_SPACE_LIST[@]}"; do
    submit_arm "${command_space}"
done
