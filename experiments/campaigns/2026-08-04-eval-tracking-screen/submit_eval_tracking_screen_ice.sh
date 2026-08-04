#!/usr/bin/env bash
set -euo pipefail

# 500M LAFAN1 arms aimed at EVAL-TIME MPJPE and EE tracking.
#
# Arms and their rationale live in `arms.sh`, sourced here so the table has one
# definition. Read that file before changing anything.
#
# SCORING IS AT EVALUATION, NOT ON THE TRAINING CURVE. Score with
# `score_eval_tracking_screen.sh`, which pulls each 500M checkpoint and runs
# `evaluate_checkpoint --randomization none` (tracking-fidelity protocol, MODE
# actions). Two reasons the training curve will not do:
#   - it is measured with domain randomization and exploration noise live, worth
#     ~1.25x and a further increment respectively;
#   - runs launched before 2026-08-04 logged `mpjpe_mm` as the error at the
#     instant the episode ended, not an episode mean.
#
# CONTROL IS FREE: job 5561149 (`lafan1_v2_foot_reward_5b_seed0_e12288_r24`) is
# the current default and passes 500M. Do not submit a control.
#
# 500M at 12288x24 is 1695 iterations, ~1.2 h on an H100.
#
# DRY_RUN=1 by default.
#
#   DRY_RUN=1 ./submit_eval_tracking_screen_ice.sh
#   DRY_RUN=0 ./submit_eval_tracking_screen_ice.sh
#   DRY_RUN=0 ARMS="s1_bodypos_std010" ./submit_eval_tracking_screen_ice.sh

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
ARMS="${ARMS:-${EVAL_SCREEN_ALL_ARM_NAMES[*]}}"
SEED="${SEED:-0}"
FRAME_CAP="${FRAME_CAP:-500000000}"
SUBMISSION_RECORD="${SUBMISSION_RECORD:-${SCRIPT_DIR}/cluster_submission.json}"

fail() { echo "[FATAL] $*" >&2; exit 1; }

DELEGATE="${REPO_ROOT}/experiments/campaigns/2026-08-02-rlopt-hp-search/submit_tuned_5b_ice.sh"
[ -x "${DELEGATE}" ] || fail "delegate launcher not found: ${DELEGATE}"

# The screen is only meaningful on the current definition; gate that it is here.
REWARDS_CFG="${REPO_ROOT}/source/isaaclab_imitation/isaaclab_imitation/tasks/manager_based/imitation/config/g1/common/rewards.py"
grep -q "motion_foot_pos = RewTerm" "${REWARDS_CFG}" \
    || fail "motion_foot_pos missing; the arms would not be on the current default."
grep -q "def heading_quat" \
    "${REPO_ROOT}/source/isaaclab_imitation/isaaclab_imitation/tasks/manager_based/imitation/mdp/_compiled.py" \
    || fail "heading_quat missing; the SONIC alignment is not in the tree."
echo "[PASS] current definition present (SONIC alignment + foot reward)"

export TRAIN_NUM_ENVS=12288
export ROLLOUT_STEPS=24
export MINIBATCH_SIZE=18432
export NJMAX=288
export NCONMAX=200
export ENCODER_TAG="lafan1_v2_det_sr_h10_z256_seed0"
export LOG_ROOT="/data/eval_tracking_screen"
export WANDB_PROJECT="g1-lafan1"
export WANDB_GROUP="eval-tracking-500m"
# H100: ICE's free H200s sit on drained nodes, and a job pinned to h200 waits
# behind that rather than running. H100 has real capacity.
export CLUSTER_SLURM_GPU_GRES="${CLUSTER_SLURM_GPU_GRES:-gpu:h100:1}"
export CLUSTER_SLURM_TIME_LIMIT="${CLUSTER_SLURM_TIME_LIMIT:-05:00:00}"
export SEGMENT_WALL_S="${SEGMENT_WALL_S:-18000}"
export SEGMENT_FPS=105000
# 500M total with saves every 100M leaves five checkpoints to score from.
export SAVE_INTERVAL=100000000

arm_spec() {
    local want="$1" spec
    for spec in "${EVAL_SCREEN_ARM_SPECS[@]}"; do
        [[ "${spec%%|*}" == "${want}" ]] && { printf '%s' "${spec}"; return 0; }
    done
    return 1
}

declare -A ARM_JOBS=()
for arm in ${ARMS}; do
    spec="$(arm_spec "${arm}")" || fail "unknown arm '${arm}'; known: ${EVAL_SCREEN_ALL_ARM_NAMES[*]}"
    desc="${spec#*|}"; desc="${desc%%|*}"
    overrides="${spec##*|}"

    echo "=============================================================="
    echo "[ARM] ${arm}"
    echo "      ${desc}"
    echo "=============================================================="
    out="$(
        env DRY_RUN="${DRY_RUN}" SEED="${SEED}" FRAME_CAP="${FRAME_CAP}" \
        EXTRA_TUNED_OVERRIDES="${overrides}" \
        RUN_TAG="lafan1_v2_evaltrack_${arm}_500m_seed${SEED}" \
        WANDB_TAGS="sr,det,v2,lafan1,tuned,500m,eval-tracking,${arm}" \
        "${DELEGATE}" 2>&1
    )" || { echo "${out}"; fail "submission failed for ${arm}"; }
    echo "${out}"
    if [[ "${DRY_RUN}" == "0" ]]; then
        job="$(grep -oE 'Submitted batch job [0-9]+' <<<"${out}" | tail -1 | awk '{print $NF}')"
        [[ -n "${job}" ]] || job="UNKNOWN"
        ARM_JOBS["${arm}"]="${job}"
        echo "[OK] ${arm} -> job ${job}"
    fi
    echo
done

if [[ "${DRY_RUN}" != "0" ]]; then
    echo "[INFO] DRY_RUN=1; nothing submitted. Re-run with DRY_RUN=0."
    exit 0
fi

arms_tsv="$(for a in "${!ARM_JOBS[@]}"; do printf '%s\t%s\n' "${a}" "${ARM_JOBS[$a]}"; done)"
ARMS_TSV="${arms_tsv}" \
WORKSPACE_SHA="$(git -C "${REPO_ROOT}" rev-parse HEAD)" \
WORKSPACE_DIRTY="$(git -C "${REPO_ROOT}" status --porcelain | head -1)" \
SEED="${SEED}" FRAME_CAP="${FRAME_CAP}" RECORD="${SUBMISSION_RECORD}" \
python3 - <<'PY'
import json, os, datetime
arms = {}
for line in os.environ["ARMS_TSV"].splitlines():
    if not line.strip():
        continue
    name, job = line.split("\t")
    arms[name] = {
        "job": job,
        "num_envs": 12288,
        "rollout_steps": 24,
        "seed": int(os.environ["SEED"]),
        "total_frames": int(os.environ["FRAME_CAP"]),
    }
record = {
    "campaign": "2026-08-04-eval-tracking-screen",
    "launcher": "experiments/campaigns/2026-08-04-eval-tracking-screen/submit_eval_tracking_screen_ice.sh",
    "task": "Isaac-Imitation-G1-v2",
    "agent_entry_point": "rlopt_ipmd_tuned_cfg_entry_point",
    "goal": "lower eval-time MPJPE and EE tracking error",
    "control_run": "lafan1_v2_foot_reward_5b_seed0_e12288_r24 (job 5561149) @ 500M",
    "scoring": (
        "evaluate_checkpoint --randomization none on the 500M checkpoint "
        "(tracking-fidelity protocol, MODE actions). NOT the training curve."
    ),
    "motivation": (
        "motion_body_pos is 99.5% saturated at 20 mm and supplies 72x less "
        "gradient than tracking_reward_points, so the term whose error is MPJPE "
        "pays almost nothing for further precision"
    ),
    "wandb": {"project": "g1-lafan1", "group": "eval-tracking-500m"},
    "workspace_git_sha": os.environ["WORKSPACE_SHA"],
    "workspace_dirty": bool(os.environ["WORKSPACE_DIRTY"]),
    "submitted_at": datetime.datetime.now().astimezone().isoformat(),
    "arms": arms,
}
path = os.environ["RECORD"]
if os.path.exists(path):
    existing = json.load(open(path))
    existing.setdefault("arms", {}).update(record["arms"])
    existing["submitted_at"] = record["submitted_at"]
    record = existing
with open(path, "w") as handle:
    json.dump(record, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(f"[INFO] wrote {path}")
PY
