#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

TASK="${TASK:-Isaac-Imitation-G1-Latent-v0}"
SEED="${SEED:-0}"
TRAIN_NUM_ENVS="${TRAIN_NUM_ENVS:-4096}"
FRAME_CAP="${FRAME_CAP:-2000000000}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-${FRAMES_PER_ENV_BATCH:-24}}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-}"
LOSS_EPOCHS="${LOSS_EPOCHS:-}"
FRAMES_PER_BATCH=$((TRAIN_NUM_ENVS * ROLLOUT_STEPS))
MAX_ITERATIONS="${MAX_ITERATIONS:-$((FRAME_CAP / FRAMES_PER_BATCH))}"
EFFECTIVE_FRAME_CAP=$((MAX_ITERATIONS * FRAMES_PER_BATCH))

PRETRAIN_NUM_ENVS="${PRETRAIN_NUM_ENVS:-16}"
PRETRAIN_UPDATES="${PRETRAIN_UPDATES:-5000}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-8192}"
HORIZON_STEPS="${HORIZON_STEPS:-25}"
VIDEO_LENGTH="${VIDEO_LENGTH:-300}"
VIDEO_INTERVAL="${VIDEO_INTERVAL:-20000}"
TRAIN_VIDEO="${TRAIN_VIDEO:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100000000}"
MANIFEST_PATH="${MANIFEST_PATH:-/data/lafan1/manifests/g1_lafan1_manifest.json}"
DATASET_PATH="${DATASET_PATH:-/data/lafan1/g1_hl_diffsr}"
WANDB_PROJECT="${WANDB_PROJECT:-G1-Imitation-RLOpt-IPMD}"
WANDB_GROUP="${WANDB_GROUP:-hl_skill_pipeline_pace_l40s_2b}"
EXP_NAME="${EXP_NAME:-hl_skill_pipeline_pace_l40s_2b_seed${SEED}}"
CLUSTER_PROFILE="${CLUSTER_PROFILE:-}"
CLUSTER_CONFIG="${CLUSTER_CONFIG:-ice_runtime}"
DRY_RUN="${DRY_RUN:-1}"

if [ "$MAX_ITERATIONS" -lt 1 ]; then
    echo "[ERROR] MAX_ITERATIONS resolved to $MAX_ITERATIONS. Check TRAIN_NUM_ENVS=$TRAIN_NUM_ENVS and FRAME_CAP=$FRAME_CAP." >&2
    exit 1
fi

export CLUSTER_LOGIN="${CLUSTER_LOGIN:-login-ice.pace.gatech.edu}"
export CLUSTER_SLURM_SUBMIT_SCRIPT="${CLUSTER_SLURM_SUBMIT_SCRIPT:-pace}"
export CLUSTER_PYTHON_EXECUTABLE="${CLUSTER_PYTHON_EXECUTABLE:-scripts/rlopt/train_hl_skill_pipeline.py}"
export CLUSTER_APPEND_DEFAULT_G1_MANIFEST="${CLUSTER_APPEND_DEFAULT_G1_MANIFEST:-0}"
export CLUSTER_G1_MANIFEST_REFRESH_POLICY="${CLUSTER_G1_MANIFEST_REFRESH_POLICY:-auto}"
export CLUSTER_SLURM_TIME_LIMIT="${CLUSTER_SLURM_TIME_LIMIT:-16:00:00}"
export CLUSTER_SLURM_PARTITION="${CLUSTER_SLURM_PARTITION:-ice-gpu}"
export CLUSTER_SLURM_QOS="${CLUSTER_SLURM_QOS:-coe-ice}"
export CLUSTER_SLURM_GPU_GRES="${CLUSTER_SLURM_GPU_GRES:-gpu:l40s:1}"
export CLUSTER_SLURM_CPUS_PER_TASK="${CLUSTER_SLURM_CPUS_PER_TASK:-24}"
export CLUSTER_SLURM_MEM="${CLUSTER_SLURM_MEM:-32G}"
export CLUSTER_SLURM_JOB_NAME_PREFIX="${CLUSTER_SLURM_JOB_NAME_PREFIX:-g1-hl-2b}"
export CLUSTER_GIT_SYNC_FIRST="${CLUSTER_GIT_SYNC_FIRST:-0}"

EXTRA_PIPELINE_ARGS_STR="${EXTRA_PIPELINE_ARGS:-}"
read -r -a EXTRA_PIPELINE_ARGS_LIST <<< "$EXTRA_PIPELINE_ARGS_STR"

cmd=(./docker/cluster/cluster_interface.sh)
if [ -n "$CLUSTER_CONFIG" ]; then
    cmd+=(-c "$CLUSTER_CONFIG")
fi
cmd+=(job)
if [ -n "$CLUSTER_PROFILE" ]; then
    cmd+=("$CLUSTER_PROFILE")
fi
cmd+=(
    --task "$TASK"
    --seed "$SEED"
    --headless
    --app-arg=--kit_args=--/app/extensions/fsWatcherEnabled=false
    --manifest-path "$MANIFEST_PATH"
    --dataset-path "$DATASET_PATH"
    --pretrain-num-envs "$PRETRAIN_NUM_ENVS"
    --pretrain-updates "$PRETRAIN_UPDATES"
    --pretrain-batch-size "$PRETRAIN_BATCH_SIZE"
    --horizon-steps "$HORIZON_STEPS"
    --train-num-envs "$TRAIN_NUM_ENVS"
    --train-max-iterations "$MAX_ITERATIONS"
    --train-override "agent.collector.frames_per_batch=${ROLLOUT_STEPS}"
    --video-length "$VIDEO_LENGTH"
    --video-interval "$VIDEO_INTERVAL"
    --save-interval "$SAVE_INTERVAL"
    --logger-backend wandb
    --wandb-project "$WANDB_PROJECT"
    --wandb-group "$WANDB_GROUP"
    --exp-name "$EXP_NAME"
)
if [ -n "$MINIBATCH_SIZE" ]; then
    cmd+=(--train-override "agent.loss.mini_batch_size=${MINIBATCH_SIZE}")
fi
if [ -n "$LOSS_EPOCHS" ]; then
    cmd+=(--train-override "agent.loss.epochs=${LOSS_EPOCHS}")
fi
case "$TRAIN_VIDEO" in
    1|true|TRUE|yes|YES|on|ON) cmd+=(--train-video) ;;
    0|false|FALSE|no|NO|off|OFF) cmd+=(--no-train-video) ;;
    *)
        echo "[ERROR] TRAIN_VIDEO must be a boolean; got '$TRAIN_VIDEO'." >&2
        exit 1
        ;;
esac
if [ -n "$EXTRA_PIPELINE_ARGS_STR" ]; then
    cmd+=("${EXTRA_PIPELINE_ARGS_LIST[@]}")
fi

echo "[INFO] Repo root: $REPO_ROOT"
echo "[INFO] PACE login: $CLUSTER_LOGIN"
echo "[INFO] Slurm: account='${CLUSTER_SLURM_ACCOUNT:-<unset>}' partition='$CLUSTER_SLURM_PARTITION' qos='$CLUSTER_SLURM_QOS' gres='$CLUSTER_SLURM_GPU_GRES' cpus='$CLUSTER_SLURM_CPUS_PER_TASK' mem='$CLUSTER_SLURM_MEM' time='$CLUSTER_SLURM_TIME_LIMIT'"
echo "[INFO] Pipeline: task='$TASK' seed='$SEED' horizon_steps='$HORIZON_STEPS' train_num_envs='$TRAIN_NUM_ENVS' rollout_steps='$ROLLOUT_STEPS' minibatch_size='${MINIBATCH_SIZE:-<config>}' loss_epochs='${LOSS_EPOCHS:-<config>}' frames_per_batch='$FRAMES_PER_BATCH' max_iterations='$MAX_ITERATIONS' effective_frame_cap='$EFFECTIVE_FRAME_CAP' train_video='$TRAIN_VIDEO'"
echo "[INFO] Manifest: path='$MANIFEST_PATH' refresh_policy='$CLUSTER_G1_MANIFEST_REFRESH_POLICY'"
if [ -z "${CLUSTER_SLURM_ACCOUNT:-}" ]; then
    echo "[WARNING] Set CLUSTER_SLURM_ACCOUNT before a real PACE submission unless your account is selected by default."
fi
printf "[CMD] "
printf "%q " "${cmd[@]}"
printf "\n"

case "$DRY_RUN" in
    1|true|TRUE|yes|YES|on|ON)
        echo "[INFO] DRY_RUN=$DRY_RUN; not contacting the cluster."
        exit 0
        ;;
esac

"${cmd[@]}"
