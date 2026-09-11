#!/usr/bin/env bash
# Regular Embodied-Control grading of the energy/torque arms (user request
# 2026-09-10): for each arm, download the newest checkpoint from ICE, export
# a `combo_v3` bundle, verify it, run the async lifecycle rehearsal on the
# MuJoCo plant with the robot's measured sensor noise (one video per motion),
# and measure the foot dither (second-difference residual of the commanded
# target, `docs/evidence/noise_20260910/dither.py`). One row per graded
# checkpoint lands in `logs/energy_torque_ec/results.tsv`.
#
# Runs arms SEQUENTIALLY and takes a lock: a rehearsal shares the plant lanes
# with nothing else (a concurrent rebuild or a second sweep corrupted three
# runs on 2026-09-10). A tick that finds the lock held just reports the
# previous rows.
#
#   ./ec_grade_live.sh                 # all four arms, newest checkpoint each
#   ARMS="e1" ./ec_grade_live.sh
#   FORCE_FRAMES=60000043008 ARMS="e5" ./ec_grade_live.sh   # one exact checkpoint
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
EC="${REPO_ROOT}/external/Embodied-Control"
REMOTE="${REMOTE:-ice}"
DATA="${DATA:-/home/hice1/fwu91/scratch/Research/IsaacLab/data}"
ARMS="${ARMS:-e1 e5 t1 t5}"
LANES="${LANES:-4}"
ENCODER="${ENCODER:-${REPO_ROOT}/logs/latent64_probe_mirror/p5_affine_encoder/latest.pt}"
ROOT="${REPO_ROOT}/logs/energy_torque_ec"
mkdir -p "${ROOT}/ckpt" "${ROOT}/bundles"
RESULTS="${ROOT}/results.tsv"
[ -s "${RESULTS}" ] || printf 'arm\tframes\tclean\tmotions\tankle_target_jitter\tankle_truth_jitter\tleg_target_jitter\tankle_step_p99\tmean_abs_target_minus_joint\tartifacts\tgraded_at\n' > "${RESULTS}"
log() { printf '[%s] %s\n' "$(date -Iseconds)" "$*"; }

exec 9>"${ROOT}/.lock"
if ! flock -n 9; then
    log "another grade is running (lock ${ROOT}/.lock); previous rows:"
    cat "${RESULTS}"
    exit 0
fi

for arm in ${ARMS}; do
    # The two hubs go through the same pipeline as EC baselines for the arms.
    case "${arm}" in
        hub_action01) tree="${DATA}/combo_action01_continue10b/action01_seed0/tracker" ;;
        hub_rate05)   tree="${DATA}/combo_action01_rate05_60000043008_pinned_20260909/action01rate05_seed0/tracker" ;;
        *)            tree="${DATA}/energy_torque_10b/${arm}_seed0/tracker" ;;
    esac
    if [ -n "${FORCE_FRAMES:-}" ]; then
        newest="${FORCE_FRAMES}"
    else
        newest="$(timeout 90 ssh "${REMOTE}" "ls ${tree}/*/models/model_step_*.pt 2>/dev/null | sed 's|.*/||' | sed -E 's/model_step_([0-9]+)\.pt/\1/' | sort -n | tail -1")"
    fi
    [ -n "${newest}" ] || { log "[energy_${arm}] no checkpoint yet"; continue; }
    name="energy_${arm}_f${newest}"
    out="${EC}/artifacts/${name}_measured"
    if [ -s "${out}/summary.json" ]; then log "[${name}] already graded"; continue; fi

    ckpt="${ROOT}/ckpt/${name}.pt"
    if [ ! -s "${ckpt}" ]; then
        remote="$(timeout 90 ssh "${REMOTE}" "ls ${tree}/*/models/model_step_${newest}.pt | head -1")"
        [ -n "${remote}" ] || { log "[${name}] checkpoint not found on ${REMOTE}"; continue; }
        log "[${name}] download ${remote}"
        scp -q "${REMOTE}:${remote}" "${ckpt}.part" && mv "${ckpt}.part" "${ckpt}" || { log "[${name}] download failed"; continue; }
    fi

    bundle="${ROOT}/bundles/${name}"
    if [ ! -f "${bundle}/manifest.json" ]; then
        log "[${name}] export combo_v3 hold 1"
        rm -rf "${bundle}"
        (cd "${REPO_ROOT}" && pixi run -e onnx-export python -m imitation_experiments.lowlevel.export_policy_bundle \
            --checkpoint "${ckpt}" --skill-checkpoint "${ENCODER}" --preset combo_v3 --hold-steps 1 \
            --output "${bundle}" --verify 2>&1 | grep -E "parity|bundle written|verify:|Error|Traceback") || { log "[${name}] export failed"; continue; }
    fi
    rm -rf "${EC}/assets/models/controller/${name}"; cp -r "${bundle}" "${EC}/assets/models/controller/${name}"

    cd "${EC}"
    log "[${name}] verify-bundle"
    pixi run -e native ec lowlevel verify-bundle "assets/models/controller/${name}" 2>&1 | grep -E '"valid"|max_abs_err' | head -3
    log "[${name}] rehearse all --lanes ${LANES} --plant-noise measured (videos per motion)"
    rm -rf "${out}"
    pixi run -e native ec lifecycle rehearse "assets/models/controller/${name}" all --output "${out}" --lanes "${LANES}" --dds-domain-base 110 --plant-noise measured > "${out}.log" 2>&1
    log "[${name}] rehearse exit $? (1 = some motion failed, expected)"
    summary="$(python3 - "${out}" <<'PY'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1]) / "summary.json"
if not p.exists():
    print("nan\tnan"); sys.exit()
s = json.load(open(p))
print(f"{s.get('episodes_passed')}\t{s.get('episodes')}")
PY
)"
    dither="$(timeout 300 pixi run -e native python docs/evidence/noise_20260910/dither.py "${out}" 2>&1 | grep -vE "Warning|warn" | tail -1)"
    log "[${name}] clean $(echo "${summary}" | cut -f1)/$(echo "${summary}" | cut -f2) | ${dither}"
    # dither.py prints: "<dir>: n=N ankle jitter: target A truth B measured C | leg: target D truth E | ankle target step p99 F rad, mean |target-joint| G rad"
    a=$(echo "${dither}" | sed -nE 's/.*ankle jitter: target +([0-9.]+) +truth +([0-9.]+).*/\1/p'); b=$(echo "${dither}" | sed -nE 's/.*ankle jitter: target +([0-9.]+) +truth +([0-9.]+).*/\2/p')
    d=$(echo "${dither}" | sed -nE 's/.*leg: target +([0-9.]+).*/\1/p'); f=$(echo "${dither}" | sed -nE 's/.*step p99 +([0-9.]+) rad.*/\1/p'); g=$(echo "${dither}" | sed -nE 's/.*mean \|target-joint\| +([0-9.]+) rad.*/\1/p')
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${arm}" "${newest}" "$(echo "${summary}" | cut -f1)" "$(echo "${summary}" | cut -f2)" "${a:-nan}" "${b:-nan}" "${d:-nan}" "${f:-nan}" "${g:-nan}" "${out}" "$(date -Iseconds)" >> "${RESULTS}"
    cd "${REPO_ROOT}"
done
log "results:"; cat "${RESULTS}"
