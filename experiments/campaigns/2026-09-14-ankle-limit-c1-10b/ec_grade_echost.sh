#!/usr/bin/env bash
# EC grading of the ankle-limit arms (c1l1, c1l2) on the deployment host, the
# same pass as 2026-09-13-e5-hardware-gap-10b/ec_grade_echost.sh: newest
# checkpoint from ICE, `combo_v3` hold-1 bundle with the shared p5_affine
# encoder, async lifecycle rehearsal on echost (DDS plant, measured sensor
# noise, live anchor, cuda inference, one video per motion), grade from the
# plant's true state. Rows land in `logs/ankle_limit_ec/results_echost.tsv`
# (same columns as the e5 pass). The per-motion fault reasons come from
# `logs/e5_hardware_gap_ec/classify_faults.py` run against
# artifacts/ankl_echost on echost.
#
# Takes the e5 pass's lock (`logs/e5_hardware_gap_ec/.lock`): echost runs the
# four plant lanes and nothing else may share them.
#
#   ./ec_grade_echost.sh                 # both arms, newest checkpoint each
#   ARMS="c1l2" ./ec_grade_echost.sh
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
REMOTE="${REMOTE:-ice}"
ECHOST="${ECHOST:-fwu91@192.168.1.185}"
ECHOST_EC="${ECHOST_EC:-Documents/SL/Embodied-Control}"
DATA="${DATA:-/home/hice1/fwu91/scratch/Research/IsaacLab/data}"
ARMS="${ARMS:-c1l1 c1l2}"
LANES="${LANES:-4}"
ENCODER="${ENCODER:-${REPO_ROOT}/logs/latent64_probe_mirror/p5_affine_encoder/latest.pt}"
ROOT="${REPO_ROOT}/logs/ankle_limit_ec"
mkdir -p "${ROOT}/ckpt" "${ROOT}/bundles" "${ROOT}/grades"
RESULTS="${ROOT}/results_echost.tsv"
[ -s "${RESULTS}" ] || printf 'arm\tframes\tclean\tmotions\tmpjpe_l_mm_all\tmpjpe_l_mm_passed\tmpjpe_g_mm_all\tdrift_m_all\tankle_target_dither\tall_target_dither\tanchor_error_m\tbody_acc_mps2\tbody_jerk_mps3\tacc_distance_mps2\taction_delta_l2\tjoint_torques_l2\tenergy_consumption\tprovider\tremote_dir\tgraded_at\n' > "${RESULTS}"
log() { printf '[%s] %s\n' "$(date -Iseconds)" "$*"; }

LOCK="${REPO_ROOT}/logs/e5_hardware_gap_ec/.lock"
mkdir -p "$(dirname "${LOCK}")"
exec 9>"${LOCK}"
if ! flock -n 9; then
    log "another echost grade is running (lock ${LOCK}); previous rows:"
    cat "${RESULTS}"
    exit 0
fi

for arm in ${ARMS}; do
    tree="${DATA}/ankle_limit_c1_10b/${arm}_seed0/tracker"
    if [ -n "${FORCE_FRAMES:-}" ]; then
        newest="${FORCE_FRAMES}"
    else
        newest="$(timeout 90 ssh "${REMOTE}" "ls ${tree}/*/models/model_step_*.pt 2>/dev/null | sed 's|.*/||' | sed -E 's/model_step_([0-9]+)\.pt/\1/' | sort -n | tail -1")"
    fi
    [ -n "${newest}" ] || { log "[ankl_${arm}] no checkpoint yet"; continue; }
    name="ankl_${arm}_f${newest}"
    if grep -q "^${arm}	${newest}	" "${RESULTS}"; then log "[${name}] already graded"; continue; fi

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

    log "[${name}] push bundle to ${ECHOST}"
    rsync -aq --delete "${bundle}/" "${ECHOST}:${ECHOST_EC}/assets/models/controller/${name}/" || { log "[${name}] push failed"; continue; }

    anchor_flag=""
    out="artifacts/ankl_echost/${name}"
    log "[${name}] echost: verify + rehearse all --lanes ${LANES} --plant-noise measured ${anchor_flag}"
    timeout 3600 ssh "${ECHOST}" "export PATH=\$HOME/.pixi/bin:\$PATH; cd ${ECHOST_EC} && mkdir -p artifacts/ankl_echost && rm -rf ${out} && \
        pixi run -e native ec lowlevel verify-native-bundle assets/models/controller/${name} --provider cuda 2>&1 | grep -E 'max_abs|FAIL' | tr -d '\n '; echo; \
        pixi run -e native ec lifecycle rehearse assets/models/controller/${name} all --output ${out} --lanes ${LANES} --dds-domain-base 120 --plant-noise measured ${anchor_flag} > ${out}.log 2>&1; echo \"rehearse exit \$?\"; \
        pixi run -e native ec lifecycle rehearsal-grade ${out} 2>&1 | grep -vi warn | tail -4" 2>&1 | sed "s/^/[${name}] /"
    grade="${ROOT}/grades/${name}.grade.json"
    scp -q "${ECHOST}:${ECHOST_EC}/${out}/grade.json" "${grade}" || { log "[${name}] no grade.json"; continue; }
    scp -q "${ECHOST}:${ECHOST_EC}/${out}/grade.tsv" "${ROOT}/grades/${name}.grade.tsv" 2>/dev/null
    python3 - "${arm}" "${newest}" "${grade}" "${out}" "${RESULTS}" <<'PY'
import json, sys, time
arm, frames, grade, out, results = sys.argv[1:6]
g = json.load(open(grade)); a = g["all"]; p = g["passed_only"]
prov = "cuda"
row = [arm, frames, str(g["passed"]), str(g["motions"]), f"{a['mpjpe_l_mm']:.2f}", f"{p['mpjpe_l_mm']:.2f}", f"{a['mpjpe_g_mm']:.1f}", f"{a['drift_max_m']:.3f}", f"{a['ankle_target_dither']:.4f}", f"{a['all_target_dither']:.4f}", f"{a['anchor_error_max_m']:.3f}",
       f"{a.get('body_acc_mps2', float('nan')):.2f}", f"{a.get('body_jerk_mps3', float('nan')):.1f}", f"{a.get('tracking_acceleration_distance_mps2', float('nan')):.2f}", f"{a.get('action_delta_l2', float('nan')):.3f}", f"{a.get('joint_torques_l2', float('nan')):.0f}", f"{a.get('energy_consumption', float('nan')):.1f}",
       prov, out, time.strftime("%Y-%m-%dT%H:%M:%S")]
open(results, "a").write("\t".join(row) + "\n")
print("[%s_f%s] clean %s/%s | L %s (passed %s) | G %s | ankle dither %s | jerk %s | acc_dist %s | adelta %s | torque_l2 %s | energy %s" % (arm, frames, row[2], row[3], row[4], row[5], row[6], row[8], row[12], row[13], row[14], row[15], row[16]))
PY
done
log "results:"; column -t -s$'\t' "${RESULTS}" | cut -c1-160
