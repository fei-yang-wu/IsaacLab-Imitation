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
NUM_ENVS="${NUM_ENVS:-10}"
STEPS="${STEPS:-500}"
SEED="${SEED:-0}"
OUT_ROOT="${OUT_ROOT:-logs/eval_tracking_screen}"
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/data}"
ENCODER_LOCAL="${ENCODER_LOCAL:-./logs/monitor_5b/encoders/lafan1_v2_det_sr_h10_z256_seed0.pt}"
MANIFEST="${MANIFEST:-./data/lafan1/manifests/g1_lafan1_manifest.json}"
DATASET="${DATASET:-./data/lafan1/zarr/g1_hl_diffsr}"

# arm -> remote run directory. The control is the current default and is free:
# it is a 5B run that passes 500M on the way.
declare -A ARM_RUNDIR=(
  ["control"]="${REMOTE_DATA_ROOT}/foot_tracking/lafan1_v2_foot_reward_5b_seed0_e12288_r24"
)
for name in "${EVAL_SCREEN_ALL_ARM_NAMES[@]}"; do
    ARM_RUNDIR["${name}"]="${REMOTE_DATA_ROOT}/eval_tracking_screen/lafan1_v2_evaltrack_${name}_500m_seed${SEED}"
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
    out="${OUT_ROOT}/${arm}"
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

    log "${arm}: strict"
    run_eval "${local_ckpt}" "${out}/strict.json" "${arm}_strict" ""
    log "${arm}: full_horizon"
    run_eval "${local_ckpt}" "${out}/full_horizon.json" "${arm}_fullhorizon" \
        "--disable_early_terminations --keep_after_done"
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
            "ee": (m.get("ee_pos_error_m") or {}).get("mean"),
            "surv": get(d, "survival_steps_mean"),
            "rand": get(d, "randomization_profile"),
        }
    if len(row) > 1:
        rows.append(row)

if not rows:
    print("no scored arms yet")
    sys.exit()

hdr = f'{"arm":<26}{"MPJPE_s":>9}{"EE_s":>8}{"surv_s":>8}{"MPJPE_fh":>10}{"EE_fh":>8}{"rand":>7}'
print(hdr); print("-" * len(hdr))
base = next((r for r in rows if r["arm"] == "control"), None)
for r in rows:
    s = r.get("strict", {}); f = r.get("full_horizon", {})
    fmt = lambda v, n=2: f"{v:.{n}f}" if isinstance(v, (int, float)) else "-"
    line = (f'{r["arm"]:<26}{fmt(s.get("mpjpe")):>9}{fmt(s.get("ee"),4):>8}'
            f'{fmt(s.get("surv"),1):>8}{fmt(f.get("mpjpe")):>10}{fmt(f.get("ee"),4):>8}'
            f'{str(s.get("rand") or f.get("rand")):>7}')
    if base and r is not base and isinstance(s.get("mpjpe"), float) and isinstance(
            base.get("strict", {}).get("mpjpe"), float):
        d = s["mpjpe"] - base["strict"]["mpjpe"]
        line += f'   {d:+.2f} mm vs control'
    print(line)
print()
print("Scored with --randomization none (tracking fidelity), MODE actions.")
print("full_horizon is the honest tracking number; strict can flatter a policy")
print("that dies early, since it only scores frames a live episode reached.")
PY
