#!/usr/bin/env bash
set -euo pipefail

# 1B LAFAN1 arms attacking the dominant failure mode: `foot_pos_xyz`.
#
# WHY. Measured on the 2026-08-03 5B runs at steady state, `foot_pos_xyz` is
# 66% of LAFAN1 and 61% of BONES-SEED terminations that are not a timeout. It is
# the only termination in either our config or SONIC's that constrains
# HORIZONTAL position -- `anchor_pos` and `ee_body_pos` test the Z component
# alone and `anchor_ori` is orientation -- so it is the binding constraint by
# design, in SONIC too.
#
# Yet the feet were barely rewarded. Before the 5-point correction they appeared
# only in `motion_body_pos`, weight 1.0 averaged over 14 bodies, so the two
# ankles carried ~0.14 of effective weight while causing two thirds of deaths.
#
# TWO ARMS, deliberately separated. Running both changes at once would leave the
# result uninterpretable, which is the confound this campaign's predecessors
# kept hitting.
#
#   reward        `motion_foot_pos` at weight 2.0: the reward counterpart of the
#                 termination -- same reroot, anchor and body set -- so the
#                 policy is rewarded for exactly the quantity that kills it.
#   reward_window the same, plus a persistence window of 4 on `foot_pos_xyz`
#                 ALONE. The other three terms stay instantaneous, and the
#                 0.2 m bar is untouched: only a spike shorter than 4 steps
#                 stops ending the episode.
#
# CONTROL IS FREE. The in-flight `lafan1_v2_sonic_aligned_5b_seed0_e12288_r24`
# (job 5561001) is this exact configuration without the foot reward, and passes
# 1B. Do not submit a control.
#
# WINDOW LENGTH. The probe on a 1B checkpoint measured foot_pos_xyz onsets
# resolving within 3 steps 13.4% of the time, within 5 steps 27.5%, within 10
# steps 49.7%. A window of 4 therefore converts roughly a fifth to a quarter of
# the dominant failure into recoveries -- deliberately conservative, since a
# longer window inflates episode length faster.
#
# READ ON MPJPE. Both a foot reward and a window lengthen episodes, so
# `episode/length`, `episode/return` and every per-minute rate move for reasons
# that are not a better policy. Also check the `Episode_Termination/foot_pos_xyz`
# share against the control's, which is the quantity these arms target.
#
# THE WEIGHT IS UNSCREENED. 2.0 is a considered starting point, not a tuned
# value; no hyperparameter screen has been run on it.
#
# DRY_RUN=1 by default.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
while [ ! -x "${REPO_ROOT}/docker/cluster/cluster_interface.sh" ]; do
    [ "${REPO_ROOT}" = "/" ] && { echo "[ERROR] repo root not found" >&2; exit 2; }
    REPO_ROOT="$(dirname "${REPO_ROOT}")"
done
cd "${REPO_ROOT}"

DRY_RUN="${DRY_RUN:-1}"
ARMS="${ARMS:-reward reward_window}"
SEED="${SEED:-0}"
FRAME_CAP="${FRAME_CAP:-1000000000}"
FOOT_REWARD_WEIGHT="${FOOT_REWARD_WEIGHT:-2.0}"
FOOT_WINDOW="${FOOT_WINDOW:-4}"
# Label the run by its actual budget. A fixed "1b" in the tag would collide with
# a concurrent run at a different budget -- same RUN_TAG means same
# LOG_ROOT/<tag>/rlopt_train, i.e. two jobs writing each other's checkpoints.
BUDGET_LABEL="$(awk -v f="${FRAME_CAP}" 'BEGIN{ if (f>=1e9) printf "%gb", f/1e9; else printf "%gm", f/1e6 }')"
SUBMISSION_RECORD="${SUBMISSION_RECORD:-${SCRIPT_DIR}/cluster_submission.json}"

fail() { echo "[FATAL] $*" >&2; exit 1; }

DELEGATE="${REPO_ROOT}/experiments/campaigns/2026-08-02-rlopt-hp-search/submit_tuned_5b_ice.sh"
[ -x "${DELEGATE}" ] || fail "delegate launcher not found: ${DELEGATE}"

REWARDS_CFG="${REPO_ROOT}/source/isaaclab_imitation/isaaclab_imitation/tasks/manager_based/imitation/config/g1/common/rewards.py"
grep -q "motion_foot_pos = RewTerm" "${REWARDS_CFG}" \
    || fail "motion_foot_pos missing from the reward config; the arms would be a no-op."
grep -q "termination_window_terms" "${REPO_ROOT}/scripts/rlopt/train.py" \
    || fail "train.py lacks --termination_window_terms; the window could not be scoped to the feet."
echo "[PASS] foot reward + scoped window present in the working tree"

# Held to the in-flight SONIC-aligned 5B run, which is the control.
export TRAIN_NUM_ENVS=12288
export ROLLOUT_STEPS=24
export MINIBATCH_SIZE=18432
export NJMAX=288
export NCONMAX=200
export ENCODER_TAG="lafan1_v2_det_sr_h10_z256_seed0"
export LOG_ROOT="/data/foot_tracking"
export WANDB_PROJECT="g1-lafan1"
export WANDB_GROUP="${WANDB_GROUP:-foot-tracking-${BUDGET_LABEL}}"
export CLUSTER_SLURM_GPU_GRES="${CLUSTER_SLURM_GPU_GRES:-gpu:h200:1}"
export CLUSTER_SLURM_TIME_LIMIT="${CLUSTER_SLURM_TIME_LIMIT:-05:00:00}"
export SEGMENT_WALL_S="${SEGMENT_WALL_S:-18000}"
export SEGMENT_FPS=105000
export SAVE_INTERVAL=100000000

declare -A ARM_JOBS=()
for arm in ${ARMS}; do
    window_env=()
    case "${arm}" in
        reward)        ;;
        reward_window) window_env=(TERMINATION_WINDOW="${FOOT_WINDOW}" TERMINATION_WINDOW_TERMS=foot_pos_xyz) ;;
        *) fail "unknown arm '${arm}'; expected reward or reward_window." ;;
    esac

    echo "=============================================================="
    echo "[ARM] ${arm}"
    echo "=============================================================="
    out="$(
        env DRY_RUN="${DRY_RUN}" SEED="${SEED}" FRAME_CAP="${FRAME_CAP}" \
        EXTRA_TUNED_OVERRIDES="env.rewards.motion_foot_pos.weight=${FOOT_REWARD_WEIGHT}" \
        RUN_TAG="lafan1_v2_foot_${arm}_${BUDGET_LABEL}_seed${SEED}_e${TRAIN_NUM_ENVS}_r${ROLLOUT_STEPS}" \
        WANDB_TAGS="sr,det,v2,lafan1,tuned,${BUDGET_LABEL},foot-tracking,${arm}" \
        "${window_env[@]}" \
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
ARMS_TSV="${arms_tsv}" WEIGHT="${FOOT_REWARD_WEIGHT}" WINDOW="${FOOT_WINDOW}" \
BUDGET_LABEL="${BUDGET_LABEL}" WANDB_GROUP_RECORD="${WANDB_GROUP}" \
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
        "foot_reward_weight": float(os.environ["WEIGHT"]),
        "run_tag_budget": os.environ.get("BUDGET_LABEL"),
        "foot_termination_window": int(os.environ["WINDOW"]) if name == "reward_window" else None,
        "num_envs": 12288,
        "rollout_steps": 24,
        "seed": int(os.environ["SEED"]),
        "total_frames": int(os.environ["FRAME_CAP"]),
    }
record = {
    "campaign": "2026-08-03-foot-tracking",
    "launcher": "experiments/campaigns/2026-08-03-foot-tracking/submit_foot_tracking_1b_ice.sh",
    "task": "Isaac-Imitation-G1-v2",
    "agent_entry_point": "rlopt_ipmd_tuned_cfg_entry_point",
    "control_run": "lafan1_v2_sonic_aligned_5b_seed0_e12288_r24 (job 5561001) @ 1B",
    "motivation": "foot_pos_xyz is 66% of LAFAN1 non-timeout terminations and the only horizontal-position constraint",
    "scoring_note": "MPJPE and the Episode_Termination/foot_pos_xyz share. Both arms lengthen episodes, so return and per-minute rates move for reasons other than a better policy.",
    "unscreened": "motion_foot_pos.weight has not been through a hyperparameter screen",
    "wandb": {"project": "g1-lafan1", "group": os.environ.get("WANDB_GROUP_RECORD", "foot-tracking")},
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
