#!/usr/bin/env bash
set -uo pipefail

# Score the eval-tracking screen the way the goal is stated: by EVALUATING each
# arm's 500M checkpoint, not by reading its training curve.
#
# WHY NOT THE TRAINING CURVE. Three independent reasons, all measured:
#   - training runs with domain randomization live, worth ~1.25x on MPJPE
#     (20.21 mm DR off against 25.22 mm DR on, same checkpoint);
#   - training collects with exploration noise while evaluation uses MODE
#     actions;
#   - runs launched before 2026-08-04 logged `mpjpe_mm` as the error at the
#     instant the episode ended rather than an episode mean, worth ~2.1x
#     (64.8 against 30.9 on the same checkpoint).
#
# Each arm gets two passes, both required by AGENTS.md:
#   strict        every termination active. The protocol number, but MPJPE is
#                 scored only over frames a surviving episode reached, so a
#                 policy that dies early can post a flattering value.
#   full_horizon  every early termination off, including base_too_low, fixed
#                 length. Every arm is scored over identical frames. On the
#                 2026-08-03 checkpoint these read 25.22 and 68.08 mm, so the
#                 choice is not cosmetic.
#
# Both use `--randomization none`: MPJPE is a tracking-fidelity metric here, and
# reset randomization alone contributes ~39 mm at frame 0.
#
# Idempotent -- an arm already scored is skipped, so re-running is free.
#
#   ./score_eval_tracking_screen.sh            # score whatever has reached 500M
#   ARMS="s1_bodypos_std010" ./score_eval_tracking_screen.sh
#   FRAMES=300000000 ./score_eval_tracking_screen.sh   # score an earlier point

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
while [ ! -x "${REPO_ROOT}/docker/cluster/cluster_interface.sh" ]; do
    [ "${REPO_ROOT}" = "/" ] && { echo "[ERROR] repo root not found" >&2; exit 2; }
    REPO_ROOT="$(dirname "${REPO_ROOT}")"
done
cd "${REPO_ROOT}"

# shellcheck source=arms.sh
source "${SCRIPT_DIR}/arms.sh"

FRAMES="${FRAMES:-500000000}"
# One environment per LAFAN1 clip. Round-robin assigns trajectory rank from the
# environment index, so 10 envs would sample only 10 of the 40 clips -- and on
# the pre-screen checkpoint that quarter happened to be 5 dance (all survived)
# and 5 fallAndGetUp (all failed), which is not a representative "overall"
# number. 40 covers the manifest exactly once.
NUM_ENVS="${NUM_ENVS:-40}"
STEPS="${STEPS:-500}"
SEED="${SEED:-0}"
OUT_ROOT="${OUT_ROOT:-logs/eval_tracking_screen}"
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/data}"
ENCODER_LOCAL="${ENCODER_LOCAL:-./logs/monitor_5b/encoders/lafan1_v2_det_sr_h10_z256_seed0.pt}"
MANIFEST="${MANIFEST:-./data/lafan1/manifests/g1_lafan1_manifest.json}"
DATASET="${DATASET:-./data/lafan1/zarr/g1_hl_diffsr}"

# TRAIN_SEED selects WHICH RUN to score; SEED is the EVALUATION seed and stays
# 0. They were one variable, which meant scoring a seed-1 repeat also moved the
# evaluation seed -- so the difference against seed 0 would have mixed training
# variance with evaluation variance, and the seed repeat could not measure the
# thing it exists to measure.
TRAIN_SEED="${TRAIN_SEED:-0}"

# arm -> remote run directory. At seed 0 the control is free: it is the 5B
# default run, which passes 500M on the way. Seed repeats of the control are
# submitted as their own 500M runs, so the control must follow TRAIN_SEED like
# every other arm -- otherwise a control repeat re-scores the seed-0 5B run and
# reports zero seed variance for the baseline, which is the one number the
# repeat exists to measure.
declare -A ARM_RUNDIR=()
if [[ "${TRAIN_SEED}" == "0" ]]; then
    ARM_RUNDIR["control"]="${REMOTE_DATA_ROOT}/foot_tracking/lafan1_v2_foot_reward_5b_seed0_e12288_r24"
else
    ARM_RUNDIR["control"]="${REMOTE_DATA_ROOT}/eval_tracking_screen/lafan1_v2_evaltrack_control_500m_seed${TRAIN_SEED}"
fi
for name in "${EVAL_SCREEN_ALL_ARM_NAMES[@]}"; do
    ARM_RUNDIR["${name}"]="${REMOTE_DATA_ROOT}/eval_tracking_screen/lafan1_v2_evaltrack_${name}_500m_seed${TRAIN_SEED}"
done

ARMS="${ARMS:-control ${EVAL_SCREEN_ALL_ARM_NAMES[*]}}"

ssh_ice() { ssh -o BatchMode=yes -o ConnectTimeout=20 ice "$@"; }
log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

# Earliest checkpoint at or beyond $2 frames under run dir $1, or empty.
checkpoint_at_or_after() {
    ssh_ice "find '$1' -name 'model_step_*.pt' 2>/dev/null \
        | awk -F'model_step_' '{n=\$2; sub(/\.pt/,\"\",n); if (n+0 >= $2) print n+0, \$0}' \
        | sort -n | head -1 | cut -d' ' -f2-"
}

run_eval() {  # $1=ckpt $2=out.json $3=label $4=extra
    timeout 3000 env TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 \
        HYDRA_FULL_ERROR=1 TORCHDYNAMO_DISABLE=1 \
    pixi run -e isaaclab python -u -m imitation_experiments.lowlevel.evaluate_checkpoint \
        --task Isaac-Imitation-G1-v2 --algo IPMD --checkpoint "$1" \
        --agent_entry_point rlopt_ipmd_tuned_cfg_entry_point \
        --randomization none \
        --num_envs "${NUM_ENVS}" --steps "${STEPS}" --seed "${SEED}" \
        --reference_start_frame 0 \
        --motion_manifest "${MANIFEST}" --dataset_path "${DATASET}" \
        --output_json "$2" --label "$3" --headless $4 \
        --kit_args=--/app/extensions/fsWatcherEnabled=false \
        physics=newton_mjwarp \
        env.sim.physics.solver_cfg.njmax=288 env.sim.physics.solver_cfg.nconmax=200 \
        env.command_interface.actor.dim=258 \
        agent.ipmd.latent_dim=258 agent.ipmd.command_source=hl_skill \
        agent.ipmd.hl_skill_checkpoint_path="${ENCODER_LOCAL}" \
        agent.ipmd.hl_skill_horizon_steps=10 agent.ipmd.hl_skill_command_mode=z \
        agent.ipmd.latent_steps_min=10 agent.ipmd.latent_steps_max=10 \
        agent.ipmd.latent_learning.code_period=10 \
        agent.ipmd.latent_learning.command_phase_mode=sin_cos \
        agent.ipmd.latent_learning.code_latent_dim=256 \
        agent.ipmd.hl_skill_finetune_enabled=false >"$2.log" 2>&1
}

for arm in ${ARMS}; do
    rundir="${ARM_RUNDIR[${arm}]:-}"
    [[ -n "${rundir}" ]] || { log "unknown arm '${arm}'"; continue; }
    # Seed in the path, or a seed-1 repeat lands on seed 0's directory and the
    # idempotency skip reports it as already scored -- returning seed 0's
    # numbers under the repeat's name.
    out="${OUT_ROOT}/${arm}$([[ "${TRAIN_SEED}" != "0" ]] && printf '_trainseed%s' "${TRAIN_SEED}")"
    mkdir -p "${out}"

    if [[ -s "${out}/strict.json" && -s "${out}/full_horizon.json" ]]; then
        log "${arm}: already scored, skipping"
        continue
    fi

    remote="$(checkpoint_at_or_after "${rundir}" "${FRAMES}")"
    if [[ -z "${remote}" ]]; then
        log "${arm}: no checkpoint at or beyond ${FRAMES} frames yet"
        continue
    fi
    local_ckpt="${out}/$(basename "${remote}")"
    [[ -f "${local_ckpt}" ]] || {
        log "${arm}: pulling $(basename "${remote}")"
        rsync -q -e "ssh -o BatchMode=yes" "ice:${remote}" "${out}/" || {
            log "${arm}: pull failed"; continue; }
    }

    # An arm that changed the observation space or the command interface must be
    # evaluated with the same override, or the actor is rebuilt at the training
    # width's wrong value and the checkpoint fails to restore.
    eval_extra="${EVAL_SCREEN_ARM_EVAL_EXTRA[${arm}]:-}"
    [[ -n "${eval_extra}" ]] && log "${arm}: eval-side overrides: ${eval_extra}"

    log "${arm}: strict"
    run_eval "${local_ckpt}" "${out}/strict.json" "${arm}_strict" "${eval_extra}"
    log "${arm}: full_horizon"
    run_eval "${local_ckpt}" "${out}/full_horizon.json" "${arm}_fullhorizon" \
        "--disable_early_terminations --keep_after_done ${eval_extra}"
done

echo
python3 - "${OUT_ROOT}" <<'PY'
import json, os, sys

root = sys.argv[1]
def get(d, key):
    if isinstance(d, dict):
        if key in d:
            return d[key]
        for v in d.values():
            r = get(v, key)
            if r is not None:
                return r
    return None

rows = []
for arm in sorted(os.listdir(root)) if os.path.isdir(root) else []:
    row = {"arm": arm}
    for pas in ("strict", "full_horizon"):
        p = os.path.join(root, arm, f"{pas}.json")
        if not os.path.exists(p):
            continue
        try:
            d = json.load(open(p))
        except Exception:
            continue
        m = d.get("metrics", {})
        row[pas] = {
            "mpjpe": (m.get("tracking_mpjpe_mm") or {}).get("mean"),
            "mpjpe_g": (m.get("tracked_body_pos_error_m") or {}).get("mean"),
            "ee": (m.get("ee_pos_error_m") or {}).get("mean"),
            "ee_l": (m.get("ee_pos_error_local_m") or {}).get("mean"),
            "root": (m.get("root_pos_xyz_error_m") or {}).get("mean"),
            "surv": get(d, "survival_steps_mean"),
            "rand": get(d, "randomization_profile"),
        }
    if len(row) > 1:
        rows.append(row)

if not rows:
    print("no scored arms yet")
    sys.exit()

# GLOBAL metrics are primary. MPJPE-L is root-relative and subtracts the drift
# that dominates world-frame EE, so ranking on it while reading EE world-frame
# compares two different frames and inverts the ordering -- that mistake put
# s13 top and s12 near the bottom when the reverse is true globally.
hdr = f'{"arm":<26}{"MPJPE-G":>9}{"EE-G":>9}{"root":>9}{"surv":>8}{"MPJPE-G_fh":>12}{"MPJPE-L":>9}'
print(hdr); print("-" * len(hdr))
base = next((r for r in rows if r["arm"] == "control"), None)
for r in rows:
    s = r.get("strict", {}); f = r.get("full_horizon", {})
    fmt = lambda v, n=2: f"{v:.{n}f}" if isinstance(v, (int, float)) else "-"
    line = (f'{r["arm"]:<26}{fmt(s.get("mpjpe_g"),4):>9}{fmt(s.get("ee"),4):>9}'
            f'{fmt(s.get("root"),4):>9}{fmt(s.get("surv"),1):>8}'
            f'{fmt(f.get("mpjpe_g"),4):>12}{fmt(s.get("mpjpe")):>9}')
    if base and r is not base and isinstance(s.get("mpjpe_g"), float) and isinstance(
            base.get("strict", {}).get("mpjpe_g"), float):
        b = base["strict"]
        line += f'   {100*(s["mpjpe_g"]-b["mpjpe_g"])/b["mpjpe_g"]:+.1f}%'
        if isinstance(s.get("ee"), float) and isinstance(b.get("ee"), float):
            line += f' / {100*(s["ee"]-b["ee"])/b["ee"]:+.1f}%'
    print(line)
print()
print("Scored with --randomization none (tracking fidelity), MODE actions.")
print("GLOBAL frame is primary: MPJPE-G = tracked_body_pos_error_m, EE-G =")
print("ee_pos_error_m. MPJPE-L is shown last for reference only -- it is")
print("root-relative and subtracts the drift that dominates the global numbers,")
print("so ranking on it inverts the ordering.")
print("Deltas are MPJPE-G / EE-G against the control.")
print("full_horizon is the honest tracking number; strict can flatter a policy")
print("that dies early, since it only scores frames a live episode reached.")

# Per-motion-class breakdown. On the pre-screen checkpoint every fallAndGetUp
# clip failed and every dance clip survived the full horizon, so a single
# average hides the only split that matters: an arm that only helps the easy
# class is not progress.
def motion_class(name):
    n = (name or "").lower()
    for key in ("fallandgetup", "dance", "walk", "run", "jump", "fight", "sprint"):
        if key in n:
            return key
    return "other"

print()
print("per motion class (strict pass):")
for arm in sorted(os.listdir(root)) if os.path.isdir(root) else []:
    p = os.path.join(root, arm, "strict.json")
    if not os.path.exists(p):
        continue
    try:
        d = json.load(open(p))
    except Exception:
        continue
    per_env = d.get("per_environment") or []
    if not per_env:
        continue
    buckets = {}
    for e in per_env:
        buckets.setdefault(motion_class(e.get("motion_name")), []).append(e)
    print(f"  {arm}")
    for cls in sorted(buckets):
        envs = buckets[cls]
        surv = [e.get("survival_steps", 0) for e in envs]
        full = sum(1 for v in surv if v >= 499)
        print(f"      {cls:<14} n={len(envs):<3} survived_full={full}/{len(envs):<3} "
              f"mean_survival={sum(surv)/len(surv):6.1f}")
PY
