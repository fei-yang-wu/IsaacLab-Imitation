#!/usr/bin/env bash
set -euo pipefail

# 1B-frame LAFAN1 arms testing a PERSISTENCE WINDOW on the strict tracking
# terminations: a tracking term must be violated for N consecutive control steps
# before it ends the episode, so a single contact spike, retargeting glitch, or
# push transient no longer destroys the episode and the policy gets the chance to
# recover from it.
#
# WHAT IS HELD FIXED. Everything except the window is the configuration of the
# in-flight run `lafan1_v2_mjwarp_aligned_5b_seed0_e12288_r24`: same task, tuned
# agent entry point, geometry, encoder, corrected-LAFAN1 data, mjwarp-aligned
# solver settings, and seed. That run is therefore the matched instantaneous
# CONTROL at 1B and costs nothing extra -- do not submit a separate control.
#
# HOW TO READ THE RESULT, because two of these metrics are gameable. A window
# lengthens episodes MECHANICALLY, exactly the way loosening a threshold does:
# a local smoke went from ep_len 5.7 at window 1 to 30.9 at window 25 while
# per-step reward fell 0.072 -> 0.021. So `episode/length`, `episode/return` and
# every per-minute rate are NOT evidence here. MPJPE is per-frame and
# length-independent, which is why it is the check -- and the trained policy
# must additionally be evaluated under the STRICT INSTANTANEOUS protocol, since
# that is the protocol every recorded oracle-qualification number is stated
# against.
#
# WINDOW LENGTHS. 3 (60 ms at 50 Hz) is the coded default; 10 (200 ms) is long
# enough to absorb a push-recovery transient. There is no upstream value to
# defer to: GEAR-SONIC v0.1.0 ships the counter shape (`_CummErrorMixin`) but
# every released composition -- `tracking/base`, `tracking/eval`, and the
# `base_adaptive_strict_ori_foot_xyz` that all three `sonic_*` release
# experiments select -- binds the INSTANTANEOUS predicates, and `min_steps`
# appears in no config in that tree at all.
#
# DRY_RUN=1 by default.
#
#   DRY_RUN=1 ./submit_termination_window_1b_ice.sh          # plan only
#   DRY_RUN=0 ./submit_termination_window_1b_ice.sh          # submit windows 3 and 10
#   DRY_RUN=0 WINDOWS="5" ./submit_termination_window_1b_ice.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
while [ ! -x "${REPO_ROOT}/docker/cluster/cluster_interface.sh" ]; do
    [ "${REPO_ROOT}" = "/" ] && { echo "[ERROR] repo root not found" >&2; exit 2; }
    REPO_ROOT="$(dirname "${REPO_ROOT}")"
done
cd "${REPO_ROOT}"

DRY_RUN="${DRY_RUN:-1}"
WINDOWS="${WINDOWS:-3 10}"
SEED="${SEED:-0}"
FRAME_CAP="${FRAME_CAP:-1000000000}"
SUBMISSION_RECORD="${SUBMISSION_RECORD:-${SCRIPT_DIR}/cluster_submission.json}"

fail() { echo "[FATAL] $*" >&2; exit 1; }

# The delegate owns the 5B command: one definition of the task, agent entry
# point, data gates, encoder, and Slurm wiring. Reconstructing that command here
# is exactly the copy-drift that shipped two 5B runs at the wrong rollout on
# 2026-08-03, so this script sets only the deltas.
DELEGATE="${REPO_ROOT}/experiments/campaigns/2026-08-02-rlopt-hp-search/submit_tuned_5b_ice.sh"
[ -x "${DELEGATE}" ] || fail "delegate launcher not found or not executable: ${DELEGATE}"

# --- Held fixed to the in-flight aligned 5B run (the control) ------------------
export TRAIN_NUM_ENVS=12288
export ROLLOUT_STEPS=24
export MINIBATCH_SIZE=18432
export NJMAX=288          # mjwarp-aligned solver settings, not the 320/40 the
export NCONMAX=200        # earlier screen used
export ENCODER_TAG="lafan1_v2_det_sr_h10_z256_seed0"
export LOG_ROOT="/data/term_window"
export WANDB_PROJECT="g1-lafan1"
export WANDB_GROUP="termination-window-1b"

# 1B at the ~120k fps the aligned run measures is ~2.3 h. Requesting a 5 h wall
# rather than the 15:59 the 5B launcher defaults to: it schedules sooner and the
# segment arithmetic below still covers the whole budget in one job, so no
# TIMEOUT can land on it.
export CLUSTER_SLURM_TIME_LIMIT="05:00:00"
export SEGMENT_WALL_S=18000
export SEGMENT_FPS=110000
export SAVE_INTERVAL=100000000

echo "[INFO] windows     : ${WINDOWS}"
echo "[INFO] budget      : ${FRAME_CAP} frames, seed ${SEED}, one segment per arm"
echo "[INFO] control     : lafan1_v2_mjwarp_aligned_5b_seed0_e12288_r24 @ 1B (already running)"
echo "[INFO] record      : ${SUBMISSION_RECORD}"
echo

declare -A ARM_JOBS=()
for window in ${WINDOWS}; do
    [[ "${window}" =~ ^[0-9]+$ ]] && (( window >= 1 )) \
        || fail "WINDOWS must be positive integers; got '${window}'."
    arm="window${window}"
    echo "=============================================================="
    echo "[ARM] ${arm}"
    echo "=============================================================="
    out="$(
        DRY_RUN="${DRY_RUN}" \
        SEED="${SEED}" \
        FRAME_CAP="${FRAME_CAP}" \
        TERMINATION_WINDOW="${window}" \
        RUN_TAG="lafan1_v2_termwin${window}_1b_seed${SEED}_e${TRAIN_NUM_ENVS}_r${ROLLOUT_STEPS}" \
        WANDB_TAGS="sr,det,v2,lafan1,tuned,1b,termwin,termwin${window}" \
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

# Provenance. The arm -> job map plus the hashes of the two trees that decide
# what actually ran; the window lives in the working tree, so `workspace_dirty`
# is the field that says whether the archive shipped uncommitted code.
arms_json="$(for arm in "${!ARM_JOBS[@]}"; do printf '%s\t%s\n' "${arm}" "${ARM_JOBS[$arm]}"; done)"
WORKSPACE_SHA="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
WORKSPACE_DIRTY="$(git -C "${REPO_ROOT}" status --porcelain | head -1)"
RLOPT_SHA="$(git -C "${REPO_ROOT}/RLOpt" rev-parse HEAD 2>/dev/null || echo unknown)"
RLOPT_DIRTY="$(git -C "${REPO_ROOT}/RLOpt" status --porcelain 2>/dev/null | head -1)"

ARMS_TSV="${arms_json}" \
WORKSPACE_SHA="${WORKSPACE_SHA}" WORKSPACE_DIRTY="${WORKSPACE_DIRTY}" \
RLOPT_SHA="${RLOPT_SHA}" RLOPT_DIRTY="${RLOPT_DIRTY}" \
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
        "termination_window": int(name.removeprefix("window")),
        "num_envs": 12288,
        "rollout_steps": 24,
        "seed": int(os.environ["SEED"]),
        "total_frames": int(os.environ["FRAME_CAP"]),
    }

record = {
    "campaign": "2026-08-03-termination-window",
    "launcher": "experiments/campaigns/2026-08-03-termination-window/submit_termination_window_1b_ice.sh",
    "cluster": "ICE (login-ice.pace.gatech.edu)",
    "task": "Isaac-Imitation-G1-v2",
    "agent_entry_point": "rlopt_ipmd_tuned_cfg_entry_point",
    "physics": "newton_mjwarp (njmax 288, nconmax 200)",
    "control_run": "lafan1_v2_mjwarp_aligned_5b_seed0_e12288_r24 @ 1B (instantaneous, W&B group mjwarp-aligned-5b)",
    "log_root": "/data/term_window",
    "manifest_sha256": "d972c37c41dadbb68c30fc456a9dc9c1bd6d30ed0b7aa9d34b1797472c945db8",
    "encoder_checkpoint": "/data/pretrain_store/lafan1_v2_det_sr_h10_z256_seed0/checkpoints/latest.pt",
    "wandb": {"project": "g1-lafan1", "group": "termination-window-1b"},
    "workspace_git_sha": os.environ["WORKSPACE_SHA"],
    "workspace_dirty": bool(os.environ["WORKSPACE_DIRTY"]),
    "rlopt_git_sha": os.environ["RLOPT_SHA"],
    "rlopt_dirty": bool(os.environ["RLOPT_DIRTY"]),
    "submitted_at": datetime.datetime.now().astimezone().isoformat(),
    "arms": arms,
    "scoring_note": (
        "MPJPE only. A window inflates episode length, return and every "
        "per-minute rate mechanically, the same way a threshold relaxation "
        "does. Evaluate the resulting checkpoints under the strict "
        "instantaneous protocol."
    ),
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
