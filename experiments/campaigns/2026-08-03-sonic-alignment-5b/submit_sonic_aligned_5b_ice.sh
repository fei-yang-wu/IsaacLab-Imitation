#!/usr/bin/env bash
set -euo pipefail

# 5B low-level runs on the SONIC-ALIGNED v2 definition, LAFAN1 and BONES-SEED-91.
#
# WHY THESE EXIST. Three v2 divergences from the GEAR-SONIC release were found
# and corrected in place on 2026-08-03 (NVlabs/GR00T-WholeBodyControl @ main
# aa263a8, verified against the whole repo):
#
#   1. Heading extraction. The reroot used Isaac Lab's `yaw_quat` (ZYX Euler
#      yaw), which is DEGENERATE at pitch = 90 degrees -- a two-degree attitude
#      change swung the extracted heading by 180 degrees, spinning the whole
#      rerooted reference and injecting a fictitious catastrophic error into
#      `motion_body_pos`, `motion_body_ori` and `foot_pos_xyz` exactly when the
#      robot pitched over. Now SONIC's `get_heading_q`, the twist about world Z,
#      which is continuous there.
#   2. `feet_acc` weight was -2.5e-6 against SONIC's -2.5e-7: a 10x penalty.
#   3. `tracking_reward_points` tracked 3 points (torso raised 0.5 m, bare
#      wrists) against SONIC's 5 (pelvis, both wrists offset 0.18 m forward,
#      BOTH ANKLES). The feet were absent from the term entirely.
#
# These change what the policy optimizes, so nothing trained before them is
# comparable. The predecessors on the OLD definition are
# `{lafan1,bones91}_v2_mjwarp_aligned_5b_seed0_e12288_r24` (W&B group
# `mjwarp-aligned-5b`), both plateaued by ~1B: LAFAN1 ep_len 360 -> 368 and
# MPJPE 55.9 -> 55.3 between 950M and 1.9B; BONES-SEED ep_len 190 -> 196 and
# MPJPE 43.7 -> 43.8 between 850M and 1.7B. Those are the baseline these runs
# are read against, at matched frames.
#
# EVERYTHING ELSE IS HELD TO THEM: task, tuned agent entry point, 12288 x 24,
# minibatch 18432, njmax 288 / nconmax 200, frozen encoders, seed 0, and the
# tuned environment half including the 5M->30M termination curriculum.
#
# ONE KNOB IS KNOWN-STALE. `tracking_reward_points.weight=4.0` came out of the
# 2026-08-02 screen when that term was 3 points without feet. It now measures a
# different quantity, so the weight is carried forward UNVALIDATED -- keeping it
# is what makes these runs a single-variable test of the definition change, but
# it should be re-screened before it is treated as tuned.
#
# DRY_RUN=1 by default.
#
#   DRY_RUN=1 ./submit_sonic_aligned_5b_ice.sh          # plan both
#   DRY_RUN=0 ./submit_sonic_aligned_5b_ice.sh          # submit both
#   DRY_RUN=0 ARMS="lafan1" ./submit_sonic_aligned_5b_ice.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
while [ ! -x "${REPO_ROOT}/docker/cluster/cluster_interface.sh" ]; do
    [ "${REPO_ROOT}" = "/" ] && { echo "[ERROR] repo root not found" >&2; exit 2; }
    REPO_ROOT="$(dirname "${REPO_ROOT}")"
done
cd "${REPO_ROOT}"

DRY_RUN="${DRY_RUN:-1}"
ARMS="${ARMS:-lafan1 bones91}"
SEED="${SEED:-0}"
FRAME_CAP="${FRAME_CAP:-5000000000}"
SUBMISSION_RECORD="${SUBMISSION_RECORD:-${SCRIPT_DIR}/cluster_submission.json}"

fail() { echo "[FATAL] $*" >&2; exit 1; }

DELEGATE="${REPO_ROOT}/experiments/campaigns/2026-08-02-rlopt-hp-search/submit_tuned_5b_ice.sh"
[ -x "${DELEGATE}" ] || fail "delegate launcher not found: ${DELEGATE}"

# Gate the SONIC alignment is actually in the tree that gets archived. Without
# these the runs would silently reproduce the old definition under a new name,
# which is the single worst outcome available here.
grep -q "def heading_quat" \
    "${REPO_ROOT}/source/isaaclab_imitation/isaaclab_imitation/tasks/manager_based/imitation/mdp/_compiled.py" \
    || fail "heading_quat missing: the workspace still uses yaw_quat for the reroot."
grep -q "yaw_quat(quat_mul" \
    "${REPO_ROOT}/source/isaaclab_imitation/isaaclab_imitation/tasks/manager_based/imitation/mdp/_compiled.py" \
    && fail "a reroot site still calls yaw_quat; the alignment is incomplete."
grep -q -- "-2.5e-7" \
    "${REPO_ROOT}/source/isaaclab_imitation/isaaclab_imitation/tasks/manager_based/imitation/config/g1/common/rewards.py" \
    || fail "feet_acc is not at SONIC's -2.5e-7."
REWARDS_CFG="${REPO_ROOT}/source/isaaclab_imitation/isaaclab_imitation/tasks/manager_based/imitation/config/g1/common/rewards.py"
sed -n '/tracking_reward_points = RewTerm/,/^    )$/p' "${REWARDS_CFG}" \
    | grep -q "left_ankle_roll_link" \
    || fail "tracking_reward_points does not include the ankles; still the 3-point term."
echo "[PASS] SONIC alignment present in the working tree (heading, feet_acc, 5-point)"

# --- Held fixed to the old-definition predecessors -----------------------------
export TRAIN_NUM_ENVS=12288
export ROLLOUT_STEPS=24
export MINIBATCH_SIZE=18432
export NJMAX=288
export NCONMAX=200
export WANDB_PROJECT="g1-lafan1"
export WANDB_GROUP="sonic-aligned-5b"
export LOG_ROOT="/data/sonic_aligned_5b"

# The aligned runs measure 110-120k fps at this geometry. Sized so the FULL 5B
# fits in ONE job: 5e9 / 294912 = 16,955 iterations against a 19,951-iteration
# segment cap. Their predecessors used SEGMENT_FPS=58000 and therefore stopped
# at 3.25B after ~7.7 h of a 15:59 wall, needing a second submission for the
# rest -- 8 h of allocation left idle per job.
export SEGMENT_FPS=105000
export CLUSTER_SLURM_TIME_LIMIT="15:59:00"
export SAVE_INTERVAL=100000000

# H200 rather than the launcher's H100 default: same Hopper architecture, so no
# runtime risk, with ~1.4x the memory bandwidth, which is the binding resource
# for 12288-environment GPU physics plus PPO. ICE exposes h200 nodes on
# `ice-gpu`. The segment arithmetic above is sized at 105k fps and the whole 5B
# needs 16,955 of 19,952 permitted iterations, so a FASTER card only widens that
# margin -- it cannot push the run into a second segment.
export CLUSTER_SLURM_GPU_GRES="${CLUSTER_SLURM_GPU_GRES:-gpu:h200:1}"

declare -A ARM_JOBS=()
for arm in ${ARMS}; do
    case "${arm}" in
        lafan1)
            manifest="/data/lafan1_corrected_8e95d557/manifests/g1_lafan1_manifest.json"
            dataset="/data/lafan1_corrected_8e95d557/g1_hl_diffsr"
            sha="d972c37c41dadbb68c30fc456a9dc9c1bd6d30ed0b7aa9d34b1797472c945db8"
            npz=40
            encoder="lafan1_v2_det_sr_h10_z256_seed0"
            tags="sr,det,v2,lafan1,tuned,5b,sonic-aligned"
            ;;
        bones91)
            manifest="/data/bones_seed_100/manifests/g1_bones_seed_100_sonic_filtered_manifest.json"
            dataset="/data/bones_seed_100/g1_hl_diffsr"
            sha="8d48750177efb3e9118c5d0ca14b69d62abedff16eb8c00585920a34bd87ee8d"
            npz=100
            encoder="bones_seed_91_v2_det_sr_h10_z256_seed0"
            tags="sr,det,v2,bones-seed,tuned,5b,sonic-aligned"
            ;;
        *) fail "unknown arm '${arm}'; expected lafan1 or bones91." ;;
    esac

    echo "=============================================================="
    echo "[ARM] ${arm}"
    echo "=============================================================="
    out="$(
        DRY_RUN="${DRY_RUN}" SEED="${SEED}" FRAME_CAP="${FRAME_CAP}" \
        MANIFEST_PATH="${manifest}" DATASET_PATH="${dataset}" \
        EXPECTED_MANIFEST_SHA256="${sha}" EXPECTED_NPZ_COUNT="${npz}" \
        ENCODER_TAG="${encoder}" \
        RUN_TAG="${arm}_v2_sonic_aligned_5b_seed${SEED}_e${TRAIN_NUM_ENVS}_r${ROLLOUT_STEPS}" \
        WANDB_TAGS="${tags}" \
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
RLOPT_SHA="$(git -C "${REPO_ROOT}/RLOpt" rev-parse HEAD 2>/dev/null || echo unknown)" \
SEED="${SEED}" FRAME_CAP="${FRAME_CAP}" RECORD="${SUBMISSION_RECORD}" \
GPU_GRES_RECORD="${CLUSTER_SLURM_GPU_GRES}" \
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
    "campaign": "2026-08-03-sonic-alignment-5b",
    "launcher": "experiments/campaigns/2026-08-03-sonic-alignment-5b/submit_sonic_aligned_5b_ice.sh",
    "task": "Isaac-Imitation-G1-v2",
    "agent_entry_point": "rlopt_ipmd_tuned_cfg_entry_point",
    "physics": "newton_mjwarp (njmax 288, nconmax 200)",
    "gpu_gres": os.environ.get("GPU_GRES_RECORD", "gpu:h200:1"),
    "definition": "SONIC-aligned v2: heading_quat twist-about-Z, feet_acc -2.5e-7, 5-point tracking_reward_points",
    "sonic_reference": "NVlabs/GR00T-WholeBodyControl main aa263a8 (2026-07-31)",
    "old_definition_predecessors": {
        "lafan1": "lafan1_v2_mjwarp_aligned_5b_seed0_e12288_r24 (W&B p4nlrpl6)",
        "bones91": "bones91_v2_mjwarp_aligned_5b_seed0_e12288_r24 (W&B woykoqp7)",
    },
    "known_stale_knob": "tracking_reward_points.weight=4.0 was tuned against the 3-point term; carried forward unvalidated",
    "wandb": {"project": "g1-lafan1", "group": "sonic-aligned-5b"},
    "workspace_git_sha": os.environ["WORKSPACE_SHA"],
    "workspace_dirty": bool(os.environ["WORKSPACE_DIRTY"]),
    "rlopt_git_sha": os.environ["RLOPT_SHA"],
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
