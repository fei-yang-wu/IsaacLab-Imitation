#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

TASK="${TASK:-Isaac-Imitation-G1-Latent-v0}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-10173}"
SEEDS_STR="${SEEDS:-42}"
VARIANTS_STR="${VARIANTS:-scratch pretrained_finetune pretrained_frozen random_frozen pretrained_bc_finetune}"

PROJECT_NAME="${PROJECT_NAME:-G1-Imitation-RLOpt-Pretrain}"
GROUP_NAME="${GROUP_NAME:-g1_bilinear_sr_pretrain_feature_only_latent_4096_1b}"
RUN_PREFIX="${RUN_PREFIX:-g1_bilinear_sr_pretrain_feature_only_latent_4096_1b}"

OFFLINE_NUM_UPDATES="${OFFLINE_NUM_UPDATES:-2000}"
OFFLINE_BATCH_SIZE="${OFFLINE_BATCH_SIZE:-8192}"
OFFLINE_LOG_INTERVAL="${OFFLINE_LOG_INTERVAL:-100}"
OFFLINE_POLICY_BC_UPDATES="${OFFLINE_POLICY_BC_UPDATES:-2000}"
OFFLINE_POLICY_BC_BATCH_SIZE="${OFFLINE_POLICY_BC_BATCH_SIZE:-8192}"
ONLINE_SR_UPDATE_STEPS="${ONLINE_SR_UPDATE_STEPS:-8}"
SR_BATCH_SIZE="${SR_BATCH_SIZE:-4096}"
SAMPLE_EVAL_INTERVAL="${SAMPLE_EVAL_INTERVAL:-50}"

DRY_RUN="${DRY_RUN:-0}"
VIDEO="${VIDEO:-1}"
VIDEO_LENGTH="${VIDEO_LENGTH:-200}"
VIDEO_INTERVAL="${VIDEO_INTERVAL:-2000}"
CLUSTER_PROFILE="${CLUSTER_PROFILE:-}"

read -r -a SEED_LIST <<< "$SEEDS_STR"
read -r -a VARIANT_LIST <<< "$VARIANTS_STR"

COMMON_OVERRIDES=(
    "agent.logger.project_name=${PROJECT_NAME}"
    "agent.logger.group_name=${GROUP_NAME}"
    "agent.bilinear.detach_features_for_policy=false"
    "agent.bilinear.policy_include_raw_state=false"
    "agent.bilinear.use_ema_for_policy=true"
    "agent.bilinear.sr_batch_size=${SR_BATCH_SIZE}"
    "agent.bilinear.sample_eval_interval=${SAMPLE_EVAL_INTERVAL}"
    "agent.bilinear.offline_pretrain.num_updates=${OFFLINE_NUM_UPDATES}"
    "agent.bilinear.offline_pretrain.batch_size=${OFFLINE_BATCH_SIZE}"
    "agent.bilinear.offline_pretrain.log_interval=${OFFLINE_LOG_INTERVAL}"
    "agent.bilinear.offline_pretrain.policy_bc_batch_size=${OFFLINE_POLICY_BC_BATCH_SIZE}"
)

get_variant_overrides() {
    local variant="$1"
    VARIANT_OVERRIDES=()
    case "$variant" in
        scratch)
            VARIANT_OVERRIDES=(
                "agent.bilinear.offline_pretrain.enabled=false"
                "agent.bilinear.offline_pretrain.policy_bc_updates=0"
                "agent.bilinear.update_steps=${ONLINE_SR_UPDATE_STEPS}"
            )
            ;;
        pretrained_finetune)
            VARIANT_OVERRIDES=(
                "agent.bilinear.offline_pretrain.enabled=true"
                "agent.bilinear.offline_pretrain.policy_bc_updates=0"
                "agent.bilinear.update_steps=${ONLINE_SR_UPDATE_STEPS}"
            )
            ;;
        pretrained_bc_finetune)
            VARIANT_OVERRIDES=(
                "agent.bilinear.offline_pretrain.enabled=true"
                "agent.bilinear.offline_pretrain.policy_bc_updates=${OFFLINE_POLICY_BC_UPDATES}"
                "agent.bilinear.update_steps=${ONLINE_SR_UPDATE_STEPS}"
            )
            ;;
        pretrained_frozen)
            VARIANT_OVERRIDES=(
                "agent.bilinear.offline_pretrain.enabled=true"
                "agent.bilinear.offline_pretrain.policy_bc_updates=0"
                "agent.bilinear.update_steps=0"
            )
            ;;
        random_frozen)
            VARIANT_OVERRIDES=(
                "agent.bilinear.offline_pretrain.enabled=false"
                "agent.bilinear.offline_pretrain.policy_bc_updates=0"
                "agent.bilinear.update_steps=0"
            )
            ;;
        *)
            echo "[ERROR] Unknown variant '$variant'. Supported: scratch pretrained_finetune pretrained_frozen random_frozen pretrained_bc_finetune" >&2
            exit 1
            ;;
    esac
}

submit_one() {
    local variant="$1"
    local seed="$2"
    local run_name="${RUN_PREFIX}_${variant}_seed${seed}"
    local cmd=(./docker/cluster/cluster_interface.sh job)

    if [[ -n "$CLUSTER_PROFILE" ]]; then
        cmd+=("$CLUSTER_PROFILE")
    fi

    get_variant_overrides "$variant"

    cmd+=(
        --task "$TASK"
        --num_envs "$NUM_ENVS"
        --headless
        --algo IPMD_BILINEAR
        --max_iterations "$MAX_ITERATIONS"
        --kit_args=--/app/extensions/fsWatcherEnabled=false
        "agent.seed=${seed}"
        "agent.logger.exp_name=${run_name}"
        "${COMMON_OVERRIDES[@]}"
        "${VARIANT_OVERRIDES[@]}"
    )

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
echo "[INFO] task=${TASK}, num_envs=${NUM_ENVS}, max_iterations=${MAX_ITERATIONS}, seeds='${SEEDS_STR}', variants='${VARIANTS_STR}', project='${PROJECT_NAME}', group='${GROUP_NAME}', dry_run='${DRY_RUN}'"

for variant in "${VARIANT_LIST[@]}"; do
    for seed in "${SEED_LIST[@]}"; do
        submit_one "$variant" "$seed"
    done
done

echo
echo "[INFO] Submitted all requested bilinear pretrain ablation jobs."
