#!/usr/bin/env bash
set -euo pipefail

# Resumable corrected-LAFAN1 latent-tracker VARIANT jobs at a 5B-frame cap,
# chained across Slurm segments (every ICE QoS caps 1-GPU jobs at 16h).
# Clone of experiments/submit_lafan1_5b_resumable_ice.sh (the main
# categorical/Gumbel latent arm) parameterized by VARIANT; see that script
# and the BONES-SEED sibling for the cross-segment plumbing rationale.
#
# Interface-ablation variants (one Slurm chain per variant):
#   VARIANT=fsq         - FSQ-quantized skill encoder (SONIC-style tokens):
#                         pretrain with --latent-mode fsq (--fsq-levels
#                         ${FSQ_LEVELS}), then the standard frozen-encoder
#                         low-level training. The default levels match the
#                         released SONIC/GR00T-WholeBodyControl token space:
#                         tokens of shape (2, 32) at 32 levels per dim
#                         (gear_sonic all_mlp_v1.yaml: num_fsq_levels=32,
#                         max_num_tokens=2) -> 64 values x 32 levels = ~320
#                         bits per 5 Hz command. Record bits/command either
#                         way; FSQ levels set the interface bandwidth.
#   VARIANT=sonic_joint - SONIC-style joint training: same categorical
#                         pretrain contract as the main latent arm, but the
#                         skill encoder keeps training during RL
#                         (agent.ipmd.hl_skill_finetune_enabled=true: PG loss
#                         hl_skill_pg_coeff on top of the offline diffsr
#                         reconstruction loss and anchor loss). Segment
#                         resume is safe: load_model restores the finetuned
#                         encoder via hl_skill_command_sampler_state_dict
#                         after the sampler is built from the pretrain path.
#
# TASK is Isaac-Imitation-G1-Latent-v0 (strict/legacy-optimizer surface) and
# the scale mirrors the validated 5B config: 12288 envs x 12 rollout steps,
# minibatch 18432, njmax=320/nconmax=40, ice-gpu H100 80 GB. Qualification
# for ablation rows is training plateau (user decision 2026-07-21).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

VARIANT="${VARIANT:?Set VARIANT=fsq or VARIANT=sonic_joint}"
case "${VARIANT}" in
    fsq|sonic_joint) ;;
    *)
        echo "[ERROR] VARIANT must be 'fsq' or 'sonic_joint'; got '${VARIANT}'." >&2
        exit 2
        ;;
esac

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
# SONIC-matched token space: 64 dims x 32 levels (2 tokens x 32 dims in the
# release; our encoder emits one flat 64-dim code per held horizon).
FSQ_LEVELS="${FSQ_LEVELS:-$(printf '32 %.0s' $(seq 1 64))}"
# SONIC-matched encoder trunk: the release's G1 motion-frame encoder MLP is
# hidden_dims [2048, 1024, 512, 512] (gear_sonic encoders/g1_mf_mlp.yaml).
FSQ_ENCODER_HIDDEN_DIMS="${FSQ_ENCODER_HIDDEN_DIMS:-2048 1024 512 512}"
# SONIC re-encodes the latent every control step over the sliding future
# window (no hold, no phase clock); our main latent arm holds z for the
# horizon (the planner-friendly contract). Both SONIC-flavored variants
# default to per-step renewal; phase features are dropped because the phase
# would be a constant at hold=1 (SONIC has no phase channel either).
LATENT_HOLD_STEPS="${LATENT_HOLD_STEPS:-1}"
if [[ "${LATENT_HOLD_STEPS}" == "1" ]]; then
    PHASE_MODE="${PHASE_MODE:-none}"
else
    PHASE_MODE="${PHASE_MODE:-sin_cos}"
fi
RUN_TAG="${RUN_TAG:-lafan1_strict_${VARIANT}_h${HORIZON_STEPS}_z256_5b_seed${SEED}_20260721_jointfix_e12288_r12_nj320_nc40}"
EXP_NAME="${EXP_NAME:-${RUN_TAG}_oracle_low_level}"
MANIFEST_PATH="${MANIFEST_PATH:-/data/lafan1_corrected_8e95d557/manifests/g1_lafan1_manifest.json}"
DATASET_PATH="${DATASET_PATH:-/data/lafan1_corrected_8e95d557/g1_hl_diffsr}"
TASK_NAME="${TASK_NAME:-Isaac-Imitation-G1-Latent-v0}"
PRETRAIN_OUTPUT_DIR="logs/lafan1_strict/${RUN_TAG}/skill_encoder_${VARIANT}_h${HORIZON_STEPS}_z256"
EXPECTED_MANIFEST_SHA256="${EXPECTED_MANIFEST_SHA256:-d972c37c41dadbb68c30fc456a9dc9c1bd6d30ed0b7aa9d34b1797472c945db8}"
EXPECTED_NPZ_COUNT="${EXPECTED_NPZ_COUNT:-40}"
REMOTE_PROJECT_ROOT="${REMOTE_PROJECT_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/isaaclab}"
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/data}"
REMOTE_ICE_BASE="$(dirname "${REMOTE_PROJECT_ROOT}")"
PRETRAIN_CKPT_CONTAINER="/data/pretrain_store/${RUN_TAG}/checkpoints/latest.pt"
RESUME_CKPT_CONTAINER="/data/resume_store/${RUN_TAG}/model_resume.pt"

case "${DRY_RUN}" in
    1|true|TRUE|yes|YES|on|ON) is_dry_run=1 ;;
    0|false|FALSE|no|NO|off|OFF) is_dry_run=0 ;;
    *)
        echo "[ERROR] DRY_RUN must be a boolean; got '${DRY_RUN}'." >&2
        exit 2
        ;;
esac

if [[ "${is_dry_run}" == "0" ]]; then
    actual_remote_sha="$(ssh -o BatchMode=yes -o ConnectTimeout=10 ice "sha256sum '${REMOTE_DATA_ROOT}/lafan1_corrected_8e95d557/manifests/g1_lafan1_manifest.json'" | awk '{print $1}')"
    remote_npz_count="$(ssh -o BatchMode=yes -o ConnectTimeout=10 ice "find '${REMOTE_DATA_ROOT}/lafan1_corrected_8e95d557' -type f -name '*.npz' | wc -l")"
    if [[ "${actual_remote_sha}" != "${EXPECTED_MANIFEST_SHA256}" || "${remote_npz_count}" != "${EXPECTED_NPZ_COUNT}" ]]; then
        echo "[ERROR] ICE corrected-LAFAN1 data gate failed: sha=${actual_remote_sha}, npz=${remote_npz_count}." >&2
        exit 2
    fi
fi

# --- Resume detection + checkpoint staging (same as the main latent arm) ---
cumulative_frames=0
latest_checkpoint=""
pretrain_ready=0
if [[ "${is_dry_run}" == "0" ]]; then
    resume_state="$(ssh -o BatchMode=yes -o ConnectTimeout=10 ice bash -s -- \
        "${REMOTE_ICE_BASE}" "${TASK_NAME}" "${EXP_NAME}" "${RUN_TAG}" \
        "${PRETRAIN_OUTPUT_DIR}" "${REMOTE_DATA_ROOT}" <<'REMOTE_EOF'
set -uo pipefail
ice_base="$1"
task_name="$2"
exp_name="$3"
run_tag="$4"
pretrain_subpath="$5"
data_root="$6"
resume_store="${data_root}/resume_store/${run_tag}"
pretrain_store="${data_root}/pretrain_store/${run_tag}"
state_file="${resume_store}/resume_state.tsv"

# Stage the newest synced-back skill-encoder pretrain into the stable store
# (idempotent: once staged, later segments reuse the exact same weights).
if [[ ! -f "${pretrain_store}/checkpoints/latest.pt" ]]; then
    src="$(ls -dt "${ice_base}"/isaaclab*/"${pretrain_subpath}" 2>/dev/null | head -1)"
    if [[ -n "${src}" && -f "${src}/checkpoints/latest.pt" ]]; then
        mkdir -p "${pretrain_store}"
        rsync -a --exclude wandb "${src}/" "${pretrain_store}/"
    fi
fi
pretrain_ready=0
[[ -f "${pretrain_store}/checkpoints/latest.pt" ]] && pretrain_ready=1

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

printf '%s\t%s\t%s\n' "${cumulative}" "${pretrain_ready}" "${latest}"
REMOTE_EOF
    )"
    cumulative_frames="$(printf '%s' "${resume_state}" | cut -f1)"
    pretrain_ready="$(printf '%s' "${resume_state}" | cut -f2)"
    latest_checkpoint="$(printf '%s' "${resume_state}" | cut -f3-)"
    cumulative_frames="${cumulative_frames:-0}"
    pretrain_ready="${pretrain_ready:-0}"
fi

# Variant-specific pretrain flags (fresh first segment only).
pretrain_variant_args=()
case "${VARIANT}" in
    fsq)
        read -r -a fsq_level_list <<< "${FSQ_LEVELS}"
        read -r -a fsq_encoder_dims_list <<< "${FSQ_ENCODER_HIDDEN_DIMS}"
        pretrain_variant_args=(
            --latent-mode fsq
            --fsq-levels "${fsq_level_list[@]}"
            --encoder-hidden-dims "${fsq_encoder_dims_list[@]}"
        )
        ;;
    sonic_joint)
        # Same categorical pretrain contract as the main latent arm.
        pretrain_variant_args=(
            --categorical-groups 64
            --categorical-categories 128
            --gumbel-hard
        )
        ;;
esac

# Variant-specific low-level training overrides (every segment).
train_variant_args=()
if [[ "${VARIANT}" == "sonic_joint" ]]; then
    train_variant_args=(
        --train-override "agent.ipmd.hl_skill_finetune_enabled=true"
    )
fi

if [[ -n "${latest_checkpoint}" ]]; then
    if [[ "${pretrain_ready}" != "1" ]]; then
        echo "[ERROR] Found a training checkpoint but no staged pretrain for ${RUN_TAG}; refusing an inconsistent resume." >&2
        exit 2
    fi
    remaining_frames=$((FRAME_CAP - cumulative_frames))
    if (( remaining_frames <= 0 )); then
        echo "[INFO] ${RUN_TAG} already reached FRAME_CAP=${FRAME_CAP} (cumulative ${cumulative_frames} frames). Not submitting."
        exit 0
    fi
    max_iterations=$(( (remaining_frames + FRAMES_PER_BATCH - 1) / FRAMES_PER_BATCH ))
    echo "[INFO] Resuming ${RUN_TAG} from ${latest_checkpoint} (${cumulative_frames}/${FRAME_CAP} cumulative frames done; ${max_iterations} iterations remaining this segment)."
    extra_args=(
        --assert-kitless
        --skip-pretrain
        --pretrained-checkpoint "${PRETRAIN_CKPT_CONTAINER}"
        --train-checkpoint "${RESUME_CKPT_CONTAINER}"
        --train-override "physics=newton_mjwarp"
        --train-override "env.sim.physics.solver_cfg.njmax=${NJMAX}"
        --train-override "env.sim.physics.solver_cfg.nconmax=${NCONMAX}"
        --train-override "env.refresh_zarr_dataset=false"
        "${train_variant_args[@]}"
    )
elif [[ "${pretrain_ready}" == "1" ]]; then
    max_iterations=$(( (FRAME_CAP + FRAMES_PER_BATCH - 1) / FRAMES_PER_BATCH ))
    echo "[INFO] No training checkpoint for ${RUN_TAG}, but a staged pretrain exists; starting low-level training from scratch with the staged skill encoder (${max_iterations} iterations for ${FRAME_CAP} frames)."
    extra_args=(
        --assert-kitless
        --skip-pretrain
        --pretrained-checkpoint "${PRETRAIN_CKPT_CONTAINER}"
        --train-override physics=newton_mjwarp
        --train-override "env.sim.physics.solver_cfg.njmax=${NJMAX}"
        --train-override "env.sim.physics.solver_cfg.nconmax=${NCONMAX}"
        --train-override env.refresh_zarr_dataset=false
        "${train_variant_args[@]}"
    )
else
    max_iterations=$(( (FRAME_CAP + FRAMES_PER_BATCH - 1) / FRAMES_PER_BATCH ))
    echo "[INFO] No checkpoint or staged pretrain found for ${RUN_TAG}; submitting a fresh first segment with pretrain (${max_iterations} iterations for ${FRAME_CAP} frames)."
    extra_args=(
        --assert-kitless
        --pretrain-output-dir "${PRETRAIN_OUTPUT_DIR}"
        "${pretrain_variant_args[@]}"
        --pretrain-override physics=newton_mjwarp
        --pretrain-override env.refresh_zarr_dataset=true
        --train-override physics=newton_mjwarp
        --train-override "env.sim.physics.solver_cfg.njmax=${NJMAX}"
        --train-override "env.sim.physics.solver_cfg.nconmax=${NCONMAX}"
        --train-override env.refresh_zarr_dataset=false
        "${train_variant_args[@]}"
    )
fi
extra_args+=(
    --latent-hold-steps "${LATENT_HOLD_STEPS}"
    --phase-mode "${PHASE_MODE}"
)
printf -v extra_args_string '%q ' "${extra_args[@]}"

export TASK="${TASK_NAME}"
export FRAME_CAP
export TRAIN_NUM_ENVS
export ROLLOUT_STEPS
export MINIBATCH_SIZE
export MAX_ITERATIONS="${max_iterations}"
export PRETRAIN_NUM_ENVS=16
# 50k updates: the pipeline's own default and the previous full pretrain
# setting.
export PRETRAIN_UPDATES=50000
export PRETRAIN_BATCH_SIZE=8192
export HORIZON_STEPS
export TRAIN_VIDEO=0
export SAVE_INTERVAL=100000000
export MANIFEST_PATH
export DATASET_PATH
export WANDB_PROJECT="${WANDB_PROJECT:-g1-lafan1-strict}"
export WANDB_GROUP="${WANDB_GROUP:-${VARIANT}-e12288-5b-resumable-jointfix}"
export EXP_NAME
export CLUSTER_CONFIG=ice_runtime
# H100 80 GB is the default: the scaled 12288-env config OOMs on the 48 GB
# ice-bw-gpu Blackwell cards. QoS caps 1-GPU jobs at 16h.
export CLUSTER_SLURM_TIME_LIMIT="${CLUSTER_SLURM_TIME_LIMIT:-15:59:00}"
export CLUSTER_SLURM_PARTITION="${CLUSTER_SLURM_PARTITION:-ice-gpu}"
export CLUSTER_SLURM_QOS=coe-ice
export CLUSTER_SLURM_GPU_GRES="${CLUSTER_SLURM_GPU_GRES:-gpu:h100:1}"
export CLUSTER_SLURM_CPUS_PER_TASK=16
export CLUSTER_SLURM_MEM=96G
export CLUSTER_SLURM_JOB_NAME_PREFIX="lafan1-${VARIANT}-5b"
export CLUSTER_G1_USD_PATH=repo
export EXTRA_PIPELINE_ARGS="${extra_args_string}"
export DRY_RUN

exec "${REPO_ROOT}/experiments/submit_hl_skill_pipeline_pace_2b.sh"
