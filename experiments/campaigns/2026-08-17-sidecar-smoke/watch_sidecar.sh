#!/usr/bin/env bash
# Workstation half of the smoke: mirror the cluster run's checkpoints and score
# every new one on the EC/MuJoCo rehearsal board, publishing each point to W&B.
#
# The sidecar runs HERE, not on the compute node: it needs the `onnx-export`
# and Embodied-Control `lowlevel-sim` Pixi environments, which exist on the
# workstation and not inside the cluster container image. It is CPU-only, so it
# costs the training job nothing.
#
#   ./watch_sidecar.sh [seed]
#
# Stop it with Ctrl-C; the mirror and the eval outputs are resumable, and the
# per-checkpoint claim files make a restart skip work that is already scored.
#
# Re-running the same arm and seed reuses the output path, so clear the old
# tracker tree (remote and mirror) before restarting a cancelled run —
# otherwise this scores the previous run's checkpoints and attributes them to
# the new one. The W&B run id is pinned here rather than inferred from the
# checkpoint path for the same reason.
set -euo pipefail
CAMPAIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${CAMPAIGN_DIR}" rev-parse --show-toplevel)"
SEED="${1:-0}"
ARM=fsq64_hold10

REMOTE_HOST="${REMOTE_HOST:-ice}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/data/sidecar_smoke/${ARM}_seed${SEED}}"
MIRROR="${MIRROR:-${REPO_ROOT}/logs/sidecar_smoke/${ARM}_seed${SEED}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/logs/eval/sidecar_smoke}"
REFERENCE_ROOT="${REFERENCE_ROOT:-${REPO_ROOT}/data/bones_seed_language10_v1/reference_arrays/root_qpos_v1}"
MODEL_XML="${MODEL_XML:-${REPO_ROOT}/external/Embodied-Control/assets/latent_playkit/model/g1_29dof_rev_1_0.xml}"
PIXI_BIN="${PIXI_BIN:-$(command -v pixi || echo "${HOME}/Storage/.pixi/bin/pixi")}"
MIRROR_INTERVAL_S="${MIRROR_INTERVAL_S:-60}"

mkdir -p "${MIRROR}/tracker" "${MIRROR}/encoder/checkpoints" "${OUTPUT_ROOT}"

# The encoder is written once, before low-level training starts; the sidecar
# finds it by walking up from a tracker checkpoint, so the mirror has to keep
# the arm tree's shape.
echo "[MIRROR] waiting for the encoder checkpoint on ${REMOTE_HOST}..."
until rsync -a "${REMOTE_HOST}:${REMOTE_ROOT}/encoder/checkpoints/latest.pt" \
        "${MIRROR}/encoder/checkpoints/" 2>/dev/null; do
    sleep "${MIRROR_INTERVAL_S}"
done
echo "[MIRROR] encoder present."

mirror_loop() {
    while true; do
        # --append-verify would corrupt a checkpoint still being written;
        # whole-file copies plus the sidecar's stability wait keep partial
        # files out of the eval.
        rsync -a --prune-empty-dirs \
            --include='*/' --include='model_step_*.pt' --exclude='*' \
            "${REMOTE_HOST}:${REMOTE_ROOT}/tracker/" "${MIRROR}/tracker/" || true
        # Re-sync the encoder every pass, not once at startup: a re-run of the
        # same arm and seed writes a NEW encoder to the same path, and pairing
        # fresh tracker checkpoints with a stale encoder fails the export's
        # tensor-identity check.
        rsync -a "${REMOTE_HOST}:${REMOTE_ROOT}/encoder/checkpoints/latest.pt" \
            "${MIRROR}/encoder/checkpoints/" 2>/dev/null || true
        sleep "${MIRROR_INTERVAL_S}"
    done
}
mirror_loop &
MIRROR_PID=$!
trap 'kill ${MIRROR_PID} 2>/dev/null || true' EXIT

cd "${REPO_ROOT}"
exec "${PIXI_BIN}" run python -m imitation_experiments.evaluation.ec_tracker_sidecar \
    scan \
    --checkpoint-tree "${MIRROR}/tracker" \
    --watch --scan-interval-s 60 \
    --reference-root "${REFERENCE_ROOT}" \
    --model "${MODEL_XML}" \
    --output-root "${OUTPUT_ROOT}" \
    --pixi-bin "${PIXI_BIN}" \
    --task-id Isaac-Imitation-G1-v2 \
    --preset fsq64_v2 \
    --wandb-project g1-bones-seed \
    --wandb-group sidecar-smoke \
    --wandb-attach \
    --wandb-run-id "fsq64-hold10-s${SEED}" \
    --wandb-tags sidecar-smoke,plumbing,ec-rehearsal
