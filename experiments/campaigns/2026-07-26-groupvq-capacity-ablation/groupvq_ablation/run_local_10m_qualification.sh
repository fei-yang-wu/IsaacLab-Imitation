#!/usr/bin/env bash
set -euo pipefail

# Sequential local 10M-frame wiring gate for the DiffSR grouped-VQ capacity
# grid. Every arm must train to the target frames, produce a loadable
# checkpoint, and retain finite metrics with an early learning signal. This
# validates wiring only; it is not a converged comparison and no result here
# may be reported as a capacity conclusion.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=./groupvq_grid.sh
source "${SCRIPT_DIR}/groupvq_grid.sh"

# The analyzer lives with the 2026-07-22 latent-learning campaign; this study
# reuses it rather than copying the qualification contract.
ANALYZER="${REPO_ROOT}/experiments/campaigns/2026-07-22-latent-learning-ablation/latent_ablation/analyze_local_qualification.py"

MODE="${MODE:-print}"
MANIFEST_PATH="${MANIFEST_PATH:-${REPO_ROOT}/data/lafan1/manifests/g1_lafan1_manifest.json}"
DATASET_PATH="${DATASET_PATH:-/tmp/iltools_g1_lafan1_tracking_corrected_8029acbce33a}"
TARGET_FRAMES="${TARGET_FRAMES:-10000000}"
NUM_ENVS="${NUM_ENVS:-4096}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-12}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-$((NUM_ENVS * ROLLOUT_STEPS / 8))}"
# Wiring/early-learning budget only. H200 production keeps the full 50k-update
# encoder pretraining contract.
PRETRAIN_UPDATES="${PRETRAIN_UPDATES:-5000}"
SEED="${SEED:-0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/logs/groupvq_ablation/local_10m_${RUN_ID}}"
ARMS="${ARMS:-$(groupvq_arm_names | tr '\n' ' ')}"

if [[ "${MODE}" != "print" && "${MODE}" != "run" ]]; then
    echo "[ERROR] MODE must be print or run; got ${MODE}." >&2
    exit 2
fi
if [[ ! -f "${MANIFEST_PATH}" || ! -d "${DATASET_PATH}" ]]; then
    echo "[ERROR] Missing corrected LAFAN1 manifest or cache." >&2
    echo "[ERROR] manifest=${MANIFEST_PATH} dataset=${DATASET_PATH}" >&2
    exit 2
fi
if [[ ! -f "${ANALYZER}" ]]; then
    echo "[ERROR] Missing qualification analyzer: ${ANALYZER}" >&2
    exit 2
fi
if (( TARGET_FRAMES > 10000000 )); then
    echo "[ERROR] This qualification launcher is capped at 10M frames per arm." >&2
    exit 2
fi

FRAMES_PER_BATCH=$((NUM_ENVS * ROLLOUT_STEPS))
MAX_ITERATIONS=$(((TARGET_FRAMES + FRAMES_PER_BATCH - 1) / FRAMES_PER_BATCH))
EFFECTIVE_FRAMES=$((MAX_ITERATIONS * FRAMES_PER_BATCH))

mkdir -p "${OUTPUT_ROOT}"

# Fail the whole gate before any Isaac Lab time if a grid point cannot build.
grid_record="${OUTPUT_ROOT}/encoder_grid_check.json"
if [[ "${MODE}" == "run" ]]; then
    pixi run python "${SCRIPT_DIR}/check_groupvq_encoder_grid.py" --output "${grid_record}"
else
    echo "[PLAN] pixi run python ${SCRIPT_DIR}/check_groupvq_encoder_grid.py --output ${grid_record}"
fi

overall_status=0
for arm in ${ARMS}; do
    groupvq_lookup_arm "${arm}"
    arm_root="${OUTPUT_ROOT}/${arm}"
    train_log="${arm_root}/train.log"
    qualification="${arm_root}/qualification.json"
    mkdir -p "${arm_root}"

    cmd=(env TERM=xterm PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 TORCHDYNAMO_DISABLE=1
        pixi run -e isaaclab python scripts/rlopt/train_hl_skill_pipeline.py
        --headless
        --assert-kitless
        --app-arg=--kit_args=--/app/extensions/fsWatcherEnabled=false
        --task Isaac-Imitation-G1-Latent-v0
        --seed "${SEED}"
        --manifest-path "${MANIFEST_PATH}"
        --dataset-path "${DATASET_PATH}"
        --pretrain-output-dir "${arm_root}/skill_encoder"
        --pretrain-num-envs 16
        --pretrain-updates "${PRETRAIN_UPDATES}"
        --pretrain-batch-size 8192
        --horizon-steps 10
        --z-dim "${GROUPVQ_Z_DIM}"
        --encoder-hidden-dims 1024 512 512
        --latent-mode gumbel_multicat
        --categorical-groups "${GROUPVQ_GROUPS}"
        --categorical-categories "${GROUPVQ_CATEGORIES}"
        --gumbel-hard
        --train-num-envs "${NUM_ENVS}"
        --train-max-iterations "${MAX_ITERATIONS}"
        --train-log-interval 1
        --no-train-video
        --phase-mode sin_cos
        --latent-hold-steps 10
        --save-interval "${TARGET_FRAMES}"
        --logger-backend csv
        --exp-name "local10m_groupvq_${arm}_seed${SEED}"
        --pretrain-override physics=newton_mjwarp
        --pretrain-override env.refresh_zarr_dataset=false
        --train-override physics=newton_mjwarp
        --train-override env.sim.physics.solver_cfg.njmax=320
        --train-override env.sim.physics.solver_cfg.nconmax=40
        --train-override env.refresh_zarr_dataset=false
        --train-override "agent.collector.frames_per_batch=${ROLLOUT_STEPS}"
        --train-override "agent.loss.mini_batch_size=${MINIBATCH_SIZE}"
        --train-override agent.ipmd.actor_learning_rate=1.0e-3
        --train-override agent.ipmd.critic_learning_rate=1.0e-3
        --train-override agent.optim.max_lr=1.0e-3
        --train-override "agent.logger.log_dir=${arm_root}/rlopt_logs")

    printf '[PLAN] %s (G=%s C=%s code_dim=%s bits=%s, %s frames): ' \
        "${arm}" "${GROUPVQ_GROUPS}" "${GROUPVQ_CATEGORIES}" \
        "${GROUPVQ_CODE_DIM}" "${GROUPVQ_BITS}" "${EFFECTIVE_FRAMES}"
    printf '%q ' "${cmd[@]}"
    printf '\n'

    if [[ "${MODE}" == "run" ]]; then
        if ! "${cmd[@]}" 2>&1 | tee "${train_log}"; then
            echo "[ERROR] Training command failed for ${arm}; recording the failed audit." >&2
            overall_status=1
        fi
        if ! pixi run python "${ANALYZER}" \
                --arm "${arm}" \
                --train-log "${train_log}" \
                --run-root "${arm_root}" \
                --target-frames "${TARGET_FRAMES}" \
                --output "${qualification}"; then
            overall_status=1
        fi
    fi
done

echo "[INFO] Local qualification root: ${OUTPUT_ROOT}"
exit "${overall_status}"
