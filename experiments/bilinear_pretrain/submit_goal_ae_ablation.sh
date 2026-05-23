#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

TASK="${TASK:-Isaac-Imitation-G1-Latent-Goal-v0}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-10173}"
SEEDS_STR="${SEEDS:-42}"
GOAL_STEPS_STR="${GOAL_STEPS:-25}"
COMMAND_PERIOD_STR="${COMMAND_PERIODS:-same}"
MANIFEST="${MANIFEST:-}"
REFRESH_ZARR_DATASET="${REFRESH_ZARR_DATASET:-false}"

PROJECT_NAME="${PROJECT_NAME:-G1-Imitation-RLOpt-Pretrain}"
GROUP_NAME="${GROUP_NAME:-g1_dance102_goal_ae_projector_4096_1b}"
RUN_PREFIX="${RUN_PREFIX:-g1_dance102_goal_ae_projector_4096_1b}"

LATENT_DIM="${LATENT_DIM:-128}"
FEATURE_DIM="${FEATURE_DIM:-64}"
TRAIN_POSTERIOR_THROUGH_POLICY="${TRAIN_POSTERIOR_THROUGH_POLICY:-false}"
FREEZE_ENCODER="${FREEZE_ENCODER:-false}"
ONLINE_SR_UPDATE_STEPS="${ONLINE_SR_UPDATE_STEPS:-8}"
SR_BATCH_SIZE="${SR_BATCH_SIZE:-4096}"
SAMPLE_EVAL_INTERVAL="${SAMPLE_EVAL_INTERVAL:-50}"
SAVE_INTERVAL="${SAVE_INTERVAL:-50000000}"

DRY_RUN="${DRY_RUN:-0}"
VIDEO="${VIDEO:-1}"
VIDEO_LENGTH="${VIDEO_LENGTH:-200}"
VIDEO_INTERVAL="${VIDEO_INTERVAL:-2000}"
CLUSTER_PROFILE="${CLUSTER_PROFILE:-}"

read -r -a SEED_LIST <<< "$SEEDS_STR"
read -r -a GOAL_STEPS_LIST <<< "$GOAL_STEPS_STR"

period_for_goal() {
    local goal_steps="$1"
    if [[ "$COMMAND_PERIOD_STR" == "same" ]]; then
        echo "$goal_steps"
        return
    fi
    local first_period
    read -r first_period _ <<< "$COMMAND_PERIOD_STR"
    echo "$first_period"
}

submit_one() {
    local goal_steps="$1"
    local seed="$2"
    local command_period
    command_period="$(period_for_goal "$goal_steps")"
    local run_name="${RUN_PREFIX}_goal${goal_steps}_period${command_period}_seed${seed}"
    local cmd=(./docker/cluster/cluster_interface.sh job)

    if [[ -n "$CLUSTER_PROFILE" ]]; then
        cmd+=("$CLUSTER_PROFILE")
    fi

    cmd+=(
        --task "$TASK"
        --num_envs "$NUM_ENVS"
        --headless
        --algo IPMD_BILINEAR
        --max_iterations "$MAX_ITERATIONS"
        --kit_args=--/app/extensions/fsWatcherEnabled=false
        "agent.seed=${seed}"
        "agent.logger.exp_name=${run_name}"
        "agent.logger.project_name=${PROJECT_NAME}"
        "agent.logger.group_name=${GROUP_NAME}"
        "agent.bilinear.detach_features_for_policy=true"
        "agent.bilinear.policy_include_raw_state=false"
        "agent.bilinear.use_ema_for_policy=true"
        "agent.bilinear.feature_dim=${FEATURE_DIM}"
        "agent.bilinear.sr_batch_size=${SR_BATCH_SIZE}"
        "agent.bilinear.sample_eval_interval=${SAMPLE_EVAL_INTERVAL}"
        "agent.bilinear.update_steps=${ONLINE_SR_UPDATE_STEPS}"
        "agent.bilinear.offline_pretrain.enabled=false"
        "agent.bilinear.offline_pretrain.policy_bc_updates=0"
        "agent.ipmd.latent_dim=${LATENT_DIM}"
        "agent.ipmd.latent_steps_min=${command_period}"
        "agent.ipmd.latent_steps_max=${command_period}"
        "agent.ipmd.latent_learning.posterior_command_period=${command_period}"
        "agent.ipmd.latent_learning.train_posterior_through_policy=${TRAIN_POSTERIOR_THROUGH_POLICY}"
        "agent.ipmd.latent_learning.freeze_encoder=${FREEZE_ENCODER}"
        "agent.save_interval=${SAVE_INTERVAL}"
        "env.latent_command_dim=${LATENT_DIM}"
        "env.latent_goal_steps=${goal_steps}"
    )

    if [[ -n "$MANIFEST" ]]; then
        cmd+=("env.lafan1_manifest_path=${MANIFEST}")
    fi
    if [[ -n "$REFRESH_ZARR_DATASET" ]]; then
        cmd+=("env.refresh_zarr_dataset=${REFRESH_ZARR_DATASET}")
    fi

    if [[ "$VIDEO" == "1" || "$VIDEO" == "true" ]]; then
        cmd+=(
            --video
            --video_length "$VIDEO_LENGTH"
            --video_interval "$VIDEO_INTERVAL"
        )
    fi

    printf "\n[%s] Submitting %s\n" "$(date '+%F %T')" "$run_name"
    printf "[CMD] "
    printf "%q " "${cmd[@]}"
    printf "\n"
    if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
        return 0
    fi
    "${cmd[@]}"
}

echo "[INFO] Repo root: $REPO_ROOT"
echo "[INFO] task=${TASK}, goals='${GOAL_STEPS_STR}', periods='${COMMAND_PERIOD_STR}', latent_dim=${LATENT_DIM}, feature_dim=${FEATURE_DIM}, group='${GROUP_NAME}', dry_run='${DRY_RUN}'"

for goal_steps in "${GOAL_STEPS_LIST[@]}"; do
    for seed in "${SEED_LIST[@]}"; do
        submit_one "$goal_steps" "$seed"
    done
done

echo
echo "[INFO] Submitted all requested goal-AE ablation jobs."
