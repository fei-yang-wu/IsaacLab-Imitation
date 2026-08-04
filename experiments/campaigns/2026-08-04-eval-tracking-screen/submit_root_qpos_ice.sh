#!/usr/bin/env bash
set -euo pipefail

# s15's reward stack on the ROOT_QPOS command interface, 500M, LAFAN1.
#
# WHY THIS IS A SEPARATE LAUNCHER, not an arm in `arms.sh`:
# every arm in that file shares one `ENCODER_TAG`, because every arm shares the
# command interface and only varies rewards. This run varies the interface, so
# it needs its own encoder and cannot be selected by arm name without silently
# pairing it with the full-body encoder.
#
# The interface: the macro state the skill encoder consumes drops joint
# velocity, so each frame is 29 qpos + 3 root_pos + 6 root_ori = 38 instead of
# 67. Over the 10-step horizon that is a 380-D encoder input against 670. The
# ACTOR command is unchanged at 258 (z 256 + sin_cos phase) -- z_dim is the same,
# only what was compressed into it differs.
#
# Rewards are s15's, sourced from `arms.sh` so there is one definition. s15 is
# the best arm of the screen: MPJPE-G 0.0439 / EE-G 0.0475, -42.1% / -39.5%
# against control, and the first arm to move root_ori (0.0273 against 0.055-0.066
# everywhere else). Holding rewards at s15 makes this a single-variable test of
# the interface, against `s15_ee_stack` seed 0 as its control.
#
#   DRY_RUN=1 ./submit_root_qpos_ice.sh      # default
#   DRY_RUN=0 ./submit_root_qpos_ice.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
while [ ! -x "${REPO_ROOT}/docker/cluster/cluster_interface.sh" ]; do
    [ "${REPO_ROOT}" = "/" ] && { echo "[ERROR] repo root not found" >&2; exit 2; }
    REPO_ROOT="$(dirname "${REPO_ROOT}")"
done
cd "${REPO_ROOT}"

# shellcheck source=arms.sh
source "${SCRIPT_DIR}/arms.sh"

DRY_RUN="${DRY_RUN:-1}"
SEED="${SEED:-0}"
FRAME_CAP="${FRAME_CAP:-500000000}"
SUBMISSION_RECORD="${SUBMISSION_RECORD:-${SCRIPT_DIR}/cluster_submission_root_qpos.json}"

ENCODER_TAG="lafan1_v2_root_qpos_det_sr_h10_z256_seed0"
# Pinned when the encoder's first layer was read as (1024, 380) = 10 x 38.
# NOTHING DOWNSTREAM VALIDATES AN ENCODER'S INPUT SPACE against the interface it
# is paired with, so a swapped or re-run encoder must be re-verified rather than
# assumed. A width mismatch does fail loudly at the first forward
# ("hl/state shape mismatch: expected (N, 38), got (N, 67)"), but that costs a
# container start and a queue slot; this costs a hash.
ENCODER_SHA256="a95d6eb308676c5cebe1b9661eb5685afebb6a67ce0d224824b8ed2c7f146b84"
MACRO_TERMS="[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b]"

fail() { echo "[FATAL] $*" >&2; exit 1; }

DELEGATE="${REPO_ROOT}/experiments/campaigns/2026-08-02-rlopt-hp-search/submit_tuned_5b_ice.sh"
[ -x "${DELEGATE}" ] || fail "delegate launcher not found: ${DELEGATE}"

# --- gate: the encoder on the cluster is the one whose width was verified -----
# Resolved exactly as the delegate resolves it (its own ICE defaults, ssh alias
# `ice`), NOT from docker/cluster/.env.cluster -- that file's CLUSTER_LOGIN
# points at skynet, so reading it here would hash a file on the wrong cluster.
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/data}"
REMOTE_CKPT="${REMOTE_DATA_ROOT}/pretrain_store/${ENCODER_TAG}/checkpoints/latest.pt"
got="$(ssh -o BatchMode=yes -o ConnectTimeout=30 ice \
        "sha256sum '${REMOTE_CKPT}' 2>/dev/null | cut -d' ' -f1" || true)"
[ -n "${got}" ] || fail "encoder not found on the cluster: ${REMOTE_CKPT}"
[ "${got}" = "${ENCODER_SHA256}" ] || fail \
    "encoder hash mismatch at ${REMOTE_CKPT}
       expected ${ENCODER_SHA256} (verified 380-D root_qpos)
       got      ${got}
     Re-verify the input width before pinning a new hash: load the checkpoint and
     read skill_encoder_state_dict's first layer. 380 = root_qpos, 670 = full_body."
echo "[PASS] encoder ${ENCODER_TAG} matches the verified 380-D root_qpos build"

# The screen's definition gate, so this run is on the same tree as its control.
grep -q "motion_foot_pos = RewTerm" \
    "${REPO_ROOT}/source/isaaclab_imitation/isaaclab_imitation/tasks/manager_based/imitation/config/g1/common/rewards.py" \
    || fail "motion_foot_pos missing; not on the current default."
grep -q "def heading_quat" \
    "${REPO_ROOT}/source/isaaclab_imitation/isaaclab_imitation/tasks/manager_based/imitation/mdp/_compiled.py" \
    || fail "heading_quat missing; the SONIC alignment is not in the tree."
echo "[PASS] current definition present (SONIC alignment + foot reward)"

# --- rewards: s15, by lookup, never by copy -----------------------------------
S15_OVERRIDES=""
for spec in "${EVAL_SCREEN_ARM_SPECS[@]}"; do
    [[ "${spec%%|*}" == "s15_ee_stack" ]] && { S15_OVERRIDES="${spec##*|}"; break; }
done
[ -n "${S15_OVERRIDES}" ] || fail "s15_ee_stack not found in arms.sh"
echo "[INFO] s15 rewards: ${S15_OVERRIDES}"

export TRAIN_NUM_ENVS=12288
export ROLLOUT_STEPS=24
export MINIBATCH_SIZE=18432
export NJMAX=288
export NCONMAX=200
export ENCODER_TAG
export LOG_ROOT="/data/eval_tracking_screen"
export WANDB_PROJECT="g1-lafan1"
export WANDB_GROUP="eval-tracking-500m"
export CLUSTER_SLURM_GPU_GRES="${CLUSTER_SLURM_GPU_GRES:-gpu:h100:1}"
export CLUSTER_SLURM_TIME_LIMIT="${CLUSTER_SLURM_TIME_LIMIT:-05:00:00}"
export SEGMENT_WALL_S="${SEGMENT_WALL_S:-18000}"
export SEGMENT_FPS=105000
export SAVE_INTERVAL=100000000

out="$(
    env DRY_RUN="${DRY_RUN}" SEED="${SEED}" FRAME_CAP="${FRAME_CAP}" \
    EXTRA_TUNED_OVERRIDES="${S15_OVERRIDES} env.expert_macro_state_terms=${MACRO_TERMS}" \
    RUN_TAG="lafan1_v2_evaltrack_root_qpos_s15_500m_seed${SEED}" \
    WANDB_TAGS="sr,det,v2,lafan1,tuned,500m,eval-tracking,root-qpos,s15-rewards" \
    "${DELEGATE}" 2>&1
)" || { echo "${out}"; fail "submission failed"; }
echo "${out}"

if [[ "${DRY_RUN}" != "0" ]]; then
    echo "[INFO] DRY_RUN=1; nothing submitted. Re-run with DRY_RUN=0."
    exit 0
fi

job="$(grep -oE 'Submitted batch job [0-9]+' <<<"${out}" | tail -1 | awk '{print $NF}')"
[[ -n "${job}" ]] || job="UNKNOWN"
echo "[OK] root_qpos -> job ${job}"

python3 - "$SUBMISSION_RECORD" "$job" "$ENCODER_TAG" "$ENCODER_SHA256" \
         "$MACRO_TERMS" "$S15_OVERRIDES" "$SEED" "$FRAME_CAP" \
         "$(git -C "${REPO_ROOT}" rev-parse HEAD)" <<'PY'
import json, sys
path, job, tag, sha, terms, rewards, seed, frames, workspace = sys.argv[1:10]
json.dump({
    "campaign": "2026-08-04-eval-tracking-screen",
    "run": "root_qpos_s15",
    "job_id": job,
    "control": "s15_ee_stack seed 0 (same rewards, full_body interface)",
    "command_interface": {
        "expert_macro_state_terms": terms,
        "encoder_tag": tag,
        "encoder_sha256": sha,
        "encoder_input_width": 380,
        "encoder_input_width_full_body": 670,
        "actor_command_dim": 258,
    },
    "reward_overrides": rewards,
    "seed": int(seed),
    "frame_cap": int(frames),
    "workspace_sha": workspace,
}, open(path, "w"), indent=2)
print(f"[OK] wrote {path}")
PY
