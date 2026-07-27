#!/usr/bin/env bash
set -euo pipefail

# Local wiring gate for the SONIC-environment / continuous-deterministic-latent
# arm. This proves the new `Isaac-Imitation-G1-Latent-Sonic-NoHist-v0` surface
# builds, resets, steps, rewards, terminates, and applies the SONIC threshold
# curriculum at the production geometry. It is a wiring check only: the frame
# budget here is far below anything that could be reported as a result, and
# the 50k-update encoder contract is replaced by a short pretrain.
#
# Per AGENTS.md, stop as soon as the code is visibly doing what the protocol
# intends; do not extend this to demonstrate convergence.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
while [ ! -x "${REPO_ROOT}/docker/cluster/cluster_interface.sh" ]; do
    if [ "${REPO_ROOT}" = "/" ]; then
        echo "[ERROR] Could not locate the repository root above ${SCRIPT_DIR}." >&2
        exit 2
    fi
    REPO_ROOT="$(dirname "${REPO_ROOT}")"
done
cd "${REPO_ROOT}"

MODE="${MODE:-print}"
TASK="${TASK:-Isaac-Imitation-G1-Latent-Sonic-NoHist-v0}"
MANIFEST_PATH="${MANIFEST_PATH:-${REPO_ROOT}/data/lafan1/manifests/g1_lafan1_manifest.json}"
EXPECTED_MANIFEST_SHA256="d972c37c41dadbb68c30fc456a9dc9c1bd6d30ed0b7aa9d34b1797472c945db8"
DATASET_PATH="${DATASET_PATH:-/tmp/iltools_g1_lafan1_tracking_corrected_8029acbce33a}"
# Production geometry so the gate exercises the same VRAM and batch shapes.
NUM_ENVS="${NUM_ENVS:-4096}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-24}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-$((NUM_ENVS * ROLLOUT_STEPS / 8))}"
# Wiring budget only. Cluster production keeps the full 50k-update contract.
PRETRAIN_UPDATES="${PRETRAIN_UPDATES:-2000}"
TARGET_FRAMES="${TARGET_FRAMES:-3000000}"
SEED="${SEED:-0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/logs/sonic_env_det/local_gate_${RUN_ID}}"

if [[ "${MODE}" != "print" && "${MODE}" != "run" ]]; then
    echo "[ERROR] MODE must be print or run; got ${MODE}." >&2
    exit 2
fi
if (( TARGET_FRAMES > 10000000 )); then
    echo "[ERROR] This is a wiring gate; it is capped at 10M frames." >&2
    exit 2
fi
if [[ ! -f "${MANIFEST_PATH}" || ! -d "${DATASET_PATH}" ]]; then
    echo "[ERROR] Missing corrected LAFAN1 manifest or cache." >&2
    echo "[ERROR] manifest=${MANIFEST_PATH} dataset=${DATASET_PATH}" >&2
    exit 2
fi
actual_sha="$(sha256sum "${MANIFEST_PATH}" | awk '{print $1}')"
if [[ "${actual_sha}" != "${EXPECTED_MANIFEST_SHA256}" ]]; then
    echo "[ERROR] Corrected LAFAN1 manifest hash mismatch: ${actual_sha}" >&2
    exit 2
fi

FRAMES_PER_BATCH=$((NUM_ENVS * ROLLOUT_STEPS))
MAX_ITERATIONS=$(((TARGET_FRAMES + FRAMES_PER_BATCH - 1) / FRAMES_PER_BATCH))

mkdir -p "${OUTPUT_ROOT}"

cmd=(env TERM=xterm PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 TORCHDYNAMO_DISABLE=1
    pixi run -e isaaclab python scripts/rlopt/train_hl_skill_pipeline.py
    --headless
    --assert-kitless
    --app-arg=--kit_args=--/app/extensions/fsWatcherEnabled=false
    --task "${TASK}"
    --seed "${SEED}"
    --manifest-path "${MANIFEST_PATH}"
    --dataset-path "${DATASET_PATH}"
    --pretrain-output-dir "${OUTPUT_ROOT}/skill_encoder"
    --pretrain-num-envs 16
    --pretrain-updates "${PRETRAIN_UPDATES}"
    --pretrain-batch-size 8192
    --horizon-steps 10
    --z-dim 256
    --encoder-hidden-dims 1024 512 512
    --latent-mode deterministic
    --phase-mode sin_cos
    --latent-hold-steps 10
    --train-num-envs "${NUM_ENVS}"
    --train-max-iterations "${MAX_ITERATIONS}"
    --train-log-interval 1
    --no-train-video
    --save-interval "${TARGET_FRAMES}"
    --logger-backend csv
    --exp-name "local_gate_sonic_env_det_seed${SEED}"
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
    --train-override "agent.logger.log_dir=${OUTPUT_ROOT}/rlopt_logs")

printf '[PLAN] task=%s envs=%s steps=%s frames=%s: ' \
    "${TASK}" "${NUM_ENVS}" "${ROLLOUT_STEPS}" "$((MAX_ITERATIONS * FRAMES_PER_BATCH))"
printf '%q ' "${cmd[@]}"
printf '\n'

if [[ "${MODE}" == "run" ]]; then
    "${cmd[@]}" 2>&1 | tee "${OUTPUT_ROOT}/train.log"
fi
echo "[INFO] Local gate root: ${OUTPUT_ROOT}"
