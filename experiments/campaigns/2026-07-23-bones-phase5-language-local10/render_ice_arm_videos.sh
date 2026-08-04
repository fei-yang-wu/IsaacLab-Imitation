#!/usr/bin/env bash
set -euo pipefail

# Pull a finished ICE ablation arm's finetuned planner back to the local
# workstation and render per-goal reference-vs-planner side-by-side videos
# locally (per the execution policy: render locally, not on the cluster).
#
# Usage:
#   ARM=A experiments/campaigns/2026-07-23-bones-phase5-language-local10/render_ice_arm_videos.sh
#   ARM=B ...   ARM=C ...
#   SUBSET="0 2 8" ARM=A ...   # only those goal indices
#   PLANNER=pretrained ARM=A ...   # visualize the demo-only planner instead

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

ARM="${ARM:?Set ARM to A, B, or C}"
case "${ARM}" in
    A) ARM_SLUG="A_dagger_demo" ;;
    B) ARM_SLUG="B_oracle_demo" ;;
    C) ARM_SLUG="C_dagger_oracle_demo" ;;
    *) echo "[ERROR] ARM must be A, B, or C (got ${ARM})." >&2; exit 2 ;;
esac
PLANNER="${PLANNER:-finetuned}"
case "${PLANNER}" in
    finetuned)  CKPT_SUBPATH="planner_finetune_planner_rollout" ;;
    pretrained) CKPT_SUBPATH="planner_pretrain_demonstration" ;;
    *) echo "[ERROR] PLANNER must be finetuned or pretrained." >&2; exit 2 ;;
esac

ICE_LOGIN="${ICE_LOGIN:-ice}"
SEED="${SEED:-0}"
ICE_LOGS="${ICE_LOGS:-/home/hice1/fwu91/scratch/Research/IsaacLab/isaaclab/logs}"
ICE_ARM_ROOT="${ICE_LOGS}/interface_baselines/bones_seed_phase5_ice_ablation_${ARM_SLUG}_seed${SEED}"
ICE_CKPT="${ICE_ARM_ROOT}/latent_skill/${CKPT_SUBPATH}/checkpoints/latest.pt"

LOCAL_ROOT="${LOCAL_ROOT:-logs/interface_baselines/ice_ablation_videos/${ARM_SLUG}_seed${SEED}}"
LOCAL_CKPT="${LOCAL_ROOT}/${CKPT_SUBPATH}_latest.pt"

# 1. Pull the planner checkpoint from ICE (small, ~278 MB).
mkdir -p "${LOCAL_ROOT}"
if [[ ! -f "${LOCAL_CKPT}" ]]; then
    echo "[INFO] Pulling arm ${ARM} ${PLANNER} planner from ICE: ${ICE_CKPT}"
    if ! ssh -o ConnectTimeout=8 "${ICE_LOGIN}" "test -f '${ICE_CKPT}'"; then
        echo "[ERROR] Arm ${ARM} ${PLANNER} planner not found on ICE yet: ${ICE_CKPT}" >&2
        echo "[ERROR] (Has the arm's finetune stage completed?)" >&2
        exit 3
    fi
    rsync -az --info=progress2 "${ICE_LOGIN}:${ICE_CKPT}" "${LOCAL_CKPT}"
    # The ICE-trained planner embeds container paths in its metadata; the
    # FrozenSkillCommanderSampler loads the skill encoder from the embedded
    # metadata.sample_metadata.provenance.skill_checkpoint (a /workspace/... path
    # with no override hook). Remap the container project root to this local repo
    # root so those loads resolve to the identical local files.
    REPO_ROOT="${REPO_ROOT}" LOCAL_CKPT="${LOCAL_CKPT}" pixi run python - <<'PY'
import os, torch
repo = os.environ["REPO_ROOT"]
path = os.environ["LOCAL_CKPT"]
ck = torch.load(path, map_location="cpu", weights_only=False)
PREFIXES = ("/workspace/IsaacLab/project", "/workspace/isaaclab/project")
def remap(o):
    if isinstance(o, dict):
        return {k: remap(v) for k, v in o.items()}
    if isinstance(o, list):
        return [remap(v) for v in o]
    if isinstance(o, str):
        for pre in PREFIXES:
            if o.startswith(pre):
                return repo + o[len(pre):]
        return o
    return o
torch.save(remap(ck), path)
print("[INFO] Remapped container paths -> local repo root in", path)
PY
else
    echo "[SKIP] Local planner checkpoint already present: ${LOCAL_CKPT}"
fi

# 2. Render reference-vs-planner videos locally against the ten-goal subset.
echo "[INFO] Rendering arm ${ARM} (${PLANNER}) videos locally..."
PLANNER_CHECKPOINT="${LOCAL_CKPT}" \
PLANNER="${PLANNER}" \
OUT_ROOT="${LOCAL_ROOT}/compare_reference_vs_planner_${PLANNER}" \
SUBSET="${SUBSET:-}" \
REF_VIS="${REF_VIS:-robot}" \
    experiments/campaigns/2026-07-23-bones-phase5-language-local10/render_reference_comparison.sh

echo "[INFO] Arm ${ARM} videos under: ${LOCAL_ROOT}/compare_reference_vs_planner_${PLANNER}"
