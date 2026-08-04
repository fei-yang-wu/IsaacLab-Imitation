#!/usr/bin/env bash
set -euo pipefail

# DiffSR (spectral) grouped-VQ capacity ablation on corrected LAFAN1.
#
# Every job performs a fresh, matched 50k-update h10 encoder pretrain with the
# `gumbel_multicat` grouped codebook at its own (G, C) point, then trains a
# frozen-encoder low-level controller under the same protocol as the
# 2026-07-22 latent-learning study. Only G and C move across arms.
#
# Default MODE=print. Submission requires an approved profile, an explicit
# confirmation token, and a complete passing local 10M qualification root.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=./groupvq_grid.sh
source "${SCRIPT_DIR}/groupvq_grid.sh"

# Reuse the approved H200 geometry from the 2026-07-22 campaign rather than
# copying it, so both studies share one wall-clock-screened profile.
DEFAULT_PROFILE="${REPO_ROOT}/experiments/campaigns/2026-07-22-latent-learning-ablation/latent_ablation/training_profile.h200.approved.env"

MODE="${MODE:-print}"
PROFILE_FILE="${TRAINING_PROFILE:-${DEFAULT_PROFILE}}"
if [[ ! -f "${PROFILE_FILE}" ]]; then
    echo "[ERROR] Missing training profile: ${PROFILE_FILE}" >&2
    exit 2
fi
# shellcheck disable=SC1090
source "${PROFILE_FILE}"

MANIFEST_PATH="${MANIFEST_PATH:-/data/lafan1_corrected_8e95d557/manifests/g1_lafan1_manifest.json}"
DATASET_PATH="${DATASET_PATH:-/data/lafan1_corrected_8e95d557/g1_hl_diffsr}"
WANDB_PROJECT="${WANDB_PROJECT:-g1-lafan1-groupvq-capacity-ablation-ice}"
WANDB_GROUP="${WANDB_GROUP:-groupvq-capacity-h10-seed0}"
SEED="${SEED:-0}"
# 100M rather than the 25M of the 2026-07-22 study: ICE scratch has ~20-40 GB
# of headroom, 25M would cost ~7.2 GB per arm, and the Study A/B runs were
# thinned to this same granularity on 2026-07-26. Plateau-checkpoint selection
# therefore resolves to 100M for every arm of both studies.
SAVE_INTERVAL="${SAVE_INTERVAL:-100000000}"
ARMS="${ARMS:-$(groupvq_arm_names | tr '\n' ' ')}"
LOCAL_QUALIFICATION_ROOT="${LOCAL_QUALIFICATION_ROOT:-}"
TRAIN_CHECKPOINT="${TRAIN_CHECKPOINT:-}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-}"
COMPLETED_FRAMES="${COMPLETED_FRAMES:-0}"

case "${MODE}" in
    print) ;;
    validate|submit)
        if [[ "${PROFILE_APPROVED:-0}" != "1" ]]; then
            echo "[ERROR] ${MODE} requires PROFILE_APPROVED=1 in ${PROFILE_FILE}." >&2
            exit 2
        fi
        if [[ -z "${LOCAL_QUALIFICATION_ROOT}" ]]; then
            echo "[ERROR] ${MODE} requires LOCAL_QUALIFICATION_ROOT with one passing record per arm." >&2
            exit 2
        fi
        if [[ "${MODE}" == "submit" && "${CONFIRM_SUBMIT:-}" != "lafan1-groupvq-capacity" ]]; then
            echo "[ERROR] Submission requires CONFIRM_SUBMIT=lafan1-groupvq-capacity." >&2
            exit 2
        fi
        ;;
    *) echo "[ERROR] MODE must be print, validate, or submit; got ${MODE}." >&2; exit 2 ;;
esac

# Every arm must have a passing local record before any cluster time is used.
if [[ "${MODE}" == "validate" || "${MODE}" == "submit" ]]; then
    grid_record="${LOCAL_QUALIFICATION_ROOT}/encoder_grid_check.json"
    if [[ ! -f "${grid_record}" ]]; then
        echo "[ERROR] Missing encoder grid pre-flight record: ${grid_record}" >&2
        exit 2
    fi
    if ! pixi run python -c "
import json, sys
record = json.load(open('${grid_record}'))
failed = [row['arm'] for row in record['grid'] if not row.get('passed')]
if failed:
    print('[ERROR] Failed encoder grid points: ' + ', '.join(failed), file=sys.stderr)
    sys.exit(1)
"; then
        exit 2
    fi
    for arm in ${ARMS}; do
        qualification="${LOCAL_QUALIFICATION_ROOT}/${arm}/qualification.json"
        if [[ ! -f "${qualification}" ]]; then
            echo "[ERROR] Missing local qualification record: ${qualification}" >&2
            exit 2
        fi
        if ! pixi run python -c "
import json, sys
record = json.load(open('${qualification}'))
if not record.get('passed'):
    print('[ERROR] Local qualification failed for ${arm}.', file=sys.stderr)
    sys.exit(1)
"; then
            exit 2
        fi
    done
    echo "[INFO] Local qualification satisfied for: ${ARMS}"
fi

if [[ -n "${TRAIN_CHECKPOINT}" ]]; then
    if [[ "${ARMS}" == *" "* || -z "${PRETRAINED_CHECKPOINT}" ]]; then
        echo "[ERROR] Resume requires one ARMS value plus TRAIN_CHECKPOINT and PRETRAINED_CHECKPOINT." >&2
        exit 2
    fi
fi

remaining_frames=$((FRAME_CAP - COMPLETED_FRAMES))
if (( remaining_frames <= 0 )); then
    echo "[INFO] FRAME_CAP=${FRAME_CAP} already credited by COMPLETED_FRAMES=${COMPLETED_FRAMES}."
    exit 0
fi

FRAMES_PER_BATCH=$((TRAIN_NUM_ENVS * ROLLOUT_STEPS))
cap_iterations=$((remaining_frames / FRAMES_PER_BATCH))
wall_iterations=$((SEGMENT_TRAIN_SECONDS * ASSUMED_FPS / FRAMES_PER_BATCH))
MAX_ITERATIONS=$((cap_iterations < wall_iterations ? cap_iterations : wall_iterations))

for arm in ${ARMS}; do
    groupvq_lookup_arm "${arm}"
    run_tag="lafan1_groupvq_${arm}_b${GROUPVQ_BITS}_h10_z${GROUPVQ_Z_DIM}_seed${SEED}"
    pretrain_dir="logs/groupvq_ablation/${run_tag}/skill_encoder"

    if [[ -n "${TRAIN_CHECKPOINT}" ]]; then
        extra=(
            --assert-kitless
            --skip-pretrain
            --pretrained-checkpoint "${PRETRAINED_CHECKPOINT}"
            --train-checkpoint "${TRAIN_CHECKPOINT}"
            --phase-mode sin_cos
            --latent-hold-steps 10
            --train-override physics=newton_mjwarp
            --train-override agent.ipmd.actor_learning_rate="${ACTOR_LR}"
            --train-override agent.ipmd.critic_learning_rate="${CRITIC_LR}"
            --train-override agent.optim.max_lr="${ACTOR_LR_CAP}"
            --train-override env.sim.physics.solver_cfg.njmax=320
            --train-override env.sim.physics.solver_cfg.nconmax=40
            --train-override env.refresh_zarr_dataset=false
        )
    else
        extra=(
            --assert-kitless
            --pretrain-output-dir "${pretrain_dir}"
            --z-dim "${GROUPVQ_Z_DIM}"
            --encoder-hidden-dims 1024 512 512
            --latent-mode gumbel_multicat
            --categorical-groups "${GROUPVQ_GROUPS}"
            --categorical-categories "${GROUPVQ_CATEGORIES}"
            --gumbel-hard
            --phase-mode sin_cos
            --latent-hold-steps 10
            --pretrain-override physics=newton_mjwarp
            # MUST stay false. Every arm shares one dataset cache under /data;
            # concurrent arms with refresh=true rebuild it underneath each
            # other and die on FileNotFoundError (observed 2026-07-26: four of
            # seven arms killed within 50s and the cache truncated to 56 KB).
            # Build the cache once with build_lafan1_cache_ice.sh, then depend
            # every arm on that job.
            --pretrain-override env.refresh_zarr_dataset=false
            --train-override physics=newton_mjwarp
            --train-override agent.ipmd.actor_learning_rate="${ACTOR_LR}"
            --train-override agent.ipmd.critic_learning_rate="${CRITIC_LR}"
            --train-override agent.optim.max_lr="${ACTOR_LR_CAP}"
            --train-override env.sim.physics.solver_cfg.njmax=320
            --train-override env.sim.physics.solver_cfg.nconmax=40
            --train-override env.refresh_zarr_dataset=false
        )
    fi

    printf -v extra_string '%q ' "${extra[@]}"
    cmd=(env
        TASK=Isaac-Imitation-G1-Latent-v0
        SEED="${SEED}"
        FRAME_CAP="${FRAME_CAP}"
        MAX_ITERATIONS="${MAX_ITERATIONS}"
        TRAIN_NUM_ENVS="${TRAIN_NUM_ENVS}"
        ROLLOUT_STEPS="${ROLLOUT_STEPS}"
        MINIBATCH_SIZE="${MINIBATCH_SIZE}"
        PRETRAIN_NUM_ENVS=16
        PRETRAIN_UPDATES=50000
        PRETRAIN_BATCH_SIZE=8192
        HORIZON_STEPS=10
        TRAIN_VIDEO=0
        SAVE_INTERVAL="${SAVE_INTERVAL}"
        MANIFEST_PATH="${MANIFEST_PATH}"
        DATASET_PATH="${DATASET_PATH}"
        WANDB_PROJECT="${WANDB_PROJECT}"
        WANDB_GROUP="${WANDB_GROUP}"
        EXP_NAME="${run_tag}"
        CLUSTER_CONFIG=ice_runtime
        CLUSTER_SLURM_TIME_LIMIT=15:59:00
        CLUSTER_SLURM_PARTITION="${PARTITION:-ice-gpu}"
        CLUSTER_SLURM_QOS=coe-ice
        CLUSTER_SLURM_GPU_GRES="${GPU_GRES}"
        CLUSTER_SLURM_CPUS_PER_TASK=16
        CLUSTER_SLURM_MEM=128G
        CLUSTER_SLURM_JOB_NAME_PREFIX="lafan-groupvq-${arm}"
        CLUSTER_SLURM_DEPENDENCY="${CACHE_DEPENDENCY:-}"
        # atl1-1-03-010-15-0 reported "No devices were found" / "no
        # CUDA-capable device is detected" on 2026-07-26 while Slurm still
        # showed it as healthy, so it must be excluded explicitly.
        CLUSTER_SLURM_EXCLUDE="${EXCLUDE_NODES:-atl1-1-03-010-15-0}"
        CLUSTER_G1_USD_PATH=repo
        EXTRA_PIPELINE_ARGS="${extra_string}"
        DRY_RUN=0
        "${REPO_ROOT}/experiments/campaigns/2026-07-23-bones-phase5-language-h200/submit_hl_skill_pipeline_pace_2b.sh"
    )

    printf '[PLAN] %s (G=%s C=%s code_dim=%s bits=%s): ' \
        "${arm}" "${GROUPVQ_GROUPS}" "${GROUPVQ_CATEGORIES}" \
        "${GROUPVQ_CODE_DIM}" "${GROUPVQ_BITS}"
    printf '%q ' "${cmd[@]}"
    printf '\n'
    if [[ "${MODE}" == "submit" ]]; then
        "${cmd[@]}"
    fi
done

if [[ "${MODE}" != "submit" ]]; then
    echo "[INFO] MODE=${MODE}: nothing was submitted."
fi
