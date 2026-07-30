#!/usr/bin/env bash
set -euo pipefail

# Per-interface qualification gate: does this command interface work AT ALL,
# before any planner compute is spent on it?
#
# Method (one motion, local, ~10 min):
#   1. REPLAY FLOOR   -- drive the tracker from the reference directly
#      (policy_command_mode=reference). This is the best the frozen controller
#      can do on this motion with a perfect, non-chunked command. It is the
#      floor that any chunked interface is measured against.
#   2. ORACLE STREAM  -- publish EXPERT commands through the interface under
#      test at the real 5 Hz rate, consumed slot-by-slot at 50 Hz. Channels the
#      interface does not carry come from the reference, so the only thing being
#      tested is the interface plumbing plus its information content.
#   3. COMPARE        -- oracle-stream vs replay floor. A large gap means the
#      interface (or its adapter) loses something; a small gap means the
#      interface is sound and any later planner error is the planner's.
#
# This is the gate that catches an interface whose tracker cannot follow it.
# Running a capacity grid on such an interface measures nothing: the planner
# would be blamed for a command space the controller was never able to track.
#
# Usage:
#   INTERFACE=ee_trajectory CHECKPOINT=... qualify_interface.sh
#   INTERFACE=full_body_trajectory CHECKPOINT=... MOTION_NAME=dance1_subject1 qualify_interface.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || (cd "${SCRIPT_DIR}/../../.." && pwd))"
cd "${REPO_ROOT}"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/paths.env"

INTERFACE="${INTERFACE:?set INTERFACE=full_body_trajectory|ee_trajectory|root_qpos|root_points5}"
case "${INTERFACE}" in
    full_body_trajectory) DEFAULT_CKPT="${FBCHUNK_LOW_LEVEL_CHECKPOINT}" ;;
    ee_trajectory)        DEFAULT_CKPT="${EECHUNK_LOW_LEVEL_CHECKPOINT}" ;;
    root_qpos)            DEFAULT_CKPT="${ROOT_QPOS_LOW_LEVEL_CHECKPOINT}" ;;
    root_points5)         DEFAULT_CKPT="${ROOT_POINTS5_LOW_LEVEL_CHECKPOINT}" ;;
    *) echo "[ERROR] unsupported INTERFACE=${INTERFACE}" >&2; exit 2 ;;
esac
CHECKPOINT="${CHECKPOINT:-${DEFAULT_CKPT}}"
STEPS="${STEPS:-700}"
NUM_ENVS="${NUM_ENVS:-10}"
DEVICE="${DEVICE:-cuda:0}"
DRY_RUN="${DRY_RUN:-0}"
# Fail the gate if the streamed interface is worse than the replay floor by more
# than this. Generous by design: it is a smoke gate, not a precision test.
TOL_MM="${TOL_MM:-25}"
# The floor itself must be sane. Faithfully reproducing a floor that is already
# catastrophic proves only that the adapter is consistent, not that the
# interface is usable -- so an absolute ceiling is checked as well. Reference
# points: full-body oracle 23.8mm, latent oracle 30.5mm on walk1.
FLOOR_MAX_MM="${FLOOR_MAX_MM:-120}"
OUT_ROOT="${OUT_ROOT:-logs/interface_baselines/qualification/${INTERFACE}_${MOTION_NAME}}"

NEWTON=(physics=newton_mjwarp
    "env.sim.physics.solver_cfg.njmax=${NJMAX:-320}"
    "env.sim.physics.solver_cfg.nconmax=${NCONMAX:-40}")
# Same full-horizon protocol the campaign uses, so numbers are comparable.
FH=(env.random_reset_step_min=0 env.random_reset_step_max=0
    env.random_reset_full_trajectory=false env.reset_schedule=sequential
    env.reference_start_frame=0 env.wrap_steps=false env.episode_length_s=20.0
    env.terminations.anchor_pos=null env.terminations.anchor_ori=null
    env.terminations.ee_body_pos=null env.terminations.foot_pos_xyz=null
    env.events.physics_material=null env.events.add_joint_default_pos=null
    env.events.base_com=null env.events.push_robot=null
    env.events.randomize_rigid_body_mass=null)

run_case() {  # $1=tag  $2=future_steps  $3=interval  $4..=extra args
    local tag="$1" future="$2" interval="$3"; shift 3
    local out="${OUT_ROOT}/${tag}"
    if [[ -e "${out}/summary.json" ]]; then echo "[SKIP] ${out}"; return 0; fi
    mkdir -p "${out}"
    printf '[CMD] %s\n' "${tag}"
    [[ "${DRY_RUN}" == "1" ]] && return 0
    TERM=xterm PYTHONUNBUFFERED=1 pixi run -e isaaclab python \
        experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/collect_interface_rollout_samples.py \
        --task "${CHUNK_TASK}" --algorithm IPMD --checkpoint "${CHECKPOINT}" \
        --interface "${INTERFACE}" --motion_name "${MOTION_NAME}" \
        --motion_manifest "${MANIFEST}" \
        --planner_interval_steps "${interval}" --command_future_steps "${future}" \
        --command_past_steps 0 \
        --state_history_steps 9 --seed 0 --num_envs "${NUM_ENVS}" \
        --control_steps "${STEPS}" --reference_start_frame 0 --evaluation_only \
        --output_dir "${out}" --kit_args=--/app/extensions/fsWatcherEnabled=false \
        "$@" "${NEWTON[@]}" "${FH[@]}" > "${out}/run.log" 2>&1 \
        || { echo "[FAIL] ${tag} -- see ${out}/run.log"; return 1; }
}

echo "[qualify] interface=${INTERFACE} motion=${MOTION_NAME} steps=${STEPS} tol=${TOL_MM}mm"
echo "[qualify] checkpoint=${CHECKPOINT}"

# (1) Replay floor: publish EVERY control step with a one-frame window, so the
#     tracker receives the current expert command each step -- the unchunked
#     condition it was trained under. Deliberately routed through the SAME
#     streamed path and the same frozen policy-only loader as case (2), so the
#     lone difference between the two cases is the publication interval and any
#     gap is attributable to chunking alone.
#
#     ``native`` is not usable here: it goes through agent.load_model(), which
#     strictly loads the value function as well, and the released checkpoints
#     carry a value config (768-wide) that differs from the current one (512).
#     The frozen loader exists precisely to avoid that coupling.
# future_steps=1 (not 0) because horizon_steps must be > 0; with interval=1 the
# hold phase is always zero, so the tracker still consumes exactly the current
# expert frame every step -- the unchunked condition.
run_case replay_floor 1 1 --low_level_command_mode streamed_vanilla || true
# (2) Oracle streamed through the interface at 5 Hz, consumed slot-by-slot. The
#     10-frame window is what the planner would publish; the tracker still sees
#     one frame per control step.
run_case oracle_stream 9 10 --low_level_command_mode streamed_vanilla || true

[[ "${DRY_RUN}" == "1" ]] && exit 0

python - "$OUT_ROOT" "$TOL_MM" "$INTERFACE" "$CHECKPOINT" "$MOTION_NAME" <<'PY'
import json, sys, pathlib, hashlib
root, tol, interface = pathlib.Path(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
checkpoint, motion_name = sys.argv[4], sys.argv[5]


def _emit(result: str, detail: dict) -> None:
    """Write the machine-readable record downstream gates read.

    The printed table is for humans; the paper entrypoint refuses to spend
    planner compute on an interface without a PASS record here.
    """
    ckpt = pathlib.Path(checkpoint)
    sha = ""
    if ckpt.is_file():
        digest = hashlib.sha256()
        with ckpt.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        sha = digest.hexdigest()
    payload = {
        "interface": interface,
        "motion_name": motion_name,
        "checkpoint": str(ckpt),
        "checkpoint_sha256": sha,
        "tolerance_mm": tol,
        "result": result,
        **detail,
    }
    (root / "qualification.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
def load(tag):
    p = root / tag / "summary.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    m = d["metrics"]
    return {
        "mpjpe": m["tracking_mpjpe_mm"]["mean"],
        "root": m["root_pos_xyz_error_m"]["mean"],
        "ee": m.get("ee_pos_error_m", {}).get("mean"),
        "joint": m.get("joint_pos_rmse_rad", {}).get("mean"),
        "steps": d.get("steps_run"),
        "surv": d["aggregate"].get("survival_steps_mean"),
    }
floor, stream = load("replay_floor"), load("oracle_stream")
print(f"\n=== interface qualification: {interface} ===")
hdr = f"{'case':16}{'MPJPE mm':>10}{'root m':>9}{'ee m':>8}{'joint rad':>11}{'steps':>7}{'surv':>8}"
print(hdr)
for tag, r in (("replay floor", floor), ("oracle stream", stream)):
    if r is None:
        print(f"{tag:16}{'DID NOT RUN':>10}")
        continue
    ee = f"{r['ee']:.3f}" if r["ee"] is not None else "-"
    jt = f"{r['joint']:.4f}" if r["joint"] is not None else "-"
    print(f"{tag:16}{r['mpjpe']:10.1f}{r['root']:9.3f}{ee:>8}{jt:>11}{r['steps']:>7}{str(r['surv']):>8}")
if floor is None or stream is None:
    print("\nRESULT: INCONCLUSIVE -- a case failed to run; see run.log")
    _emit("INCONCLUSIVE", {"replay_floor": floor, "oracle_stream": stream})
    sys.exit(2)
cases = {"replay_floor": floor, "oracle_stream": stream}
floor_max = float(__import__("os").environ.get("FLOOR_MAX_MM", "120"))
if floor["mpjpe"] > floor_max:
    _emit("UNUSABLE_INTERFACE", {**cases, "floor_max_mm": floor_max})
    print(f"\nRESULT: UNUSABLE INTERFACE -- the replay floor is {floor['mpjpe']:.1f} mm")
    print(f"        (limit {floor_max:.0f} mm). The controller cannot track this")
    print("        command space even with a perfect command every control step,")
    print("        so the interface itself is under-determined or mis-specified.")
    print("        A chunking check cannot rescue this; do not spend planner")
    print("        compute here. Reference: full-body 23.8 mm, latent 30.5 mm.")
    sys.exit(1)
gap = stream["mpjpe"] - floor["mpjpe"]
print(f"\nstreamed minus replay floor: {gap:+.1f} mm (tolerance {tol:.0f} mm)")
if gap > tol:
    _emit("FAIL", {**cases, "streamed_minus_floor_mm": gap})
    print("RESULT: FAIL -- chunked streaming loses accuracy the tracker has when")
    print("        fed the reference directly. Fix the interface/adapter before")
    print("        spending planner compute on it.")
    sys.exit(1)
_emit("PASS", {**cases, "streamed_minus_floor_mm": gap})
print("RESULT: PASS -- the interface reproduces the tracker's replay accuracy,")
print("        so later planner error is attributable to the planner.")
PY
