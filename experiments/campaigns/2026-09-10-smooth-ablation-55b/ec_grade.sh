#!/usr/bin/env bash
# Grade one controller bundle in Embodied-Control the way the deployment
# workstation graded action01_55b / rate05_60b on 2026-09-10: the async MuJoCo
# plant rehearsal over all 43 `bones` motions, once under the robot's MEASURED
# sensor-noise envelope and once under the plant's training noise, then the
# foot-dither statistic. Every episode renders its own inspection video
# (`<out>/<motion>/sim/seed_0/video.mp4`, offscreen EGL) -- there is no
# separate video step.
#
#   ./ec_grade.sh <bundle_name>                       # bundle already in EC assets
#   ./ec_grade.sh <bundle_name> <checkpoint.pt>       # export first (preset combo_v3)
#
# Rehearse exits 1 whenever any motion fails; that is a result, not an error.
# Pinning/pushing is NOT done here (naming pending); see the plan section 6.
set -uo pipefail
NAME="${1:?bundle name}"; CKPT="${2:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
EC="${REPO_ROOT}/external/Embodied-Control"
ENCODER="${ENCODER:-${REPO_ROOT}/logs/latent64_probe_mirror/p5_affine_encoder/latest.pt}"
BUNDLE="assets/models/controller/${NAME}"
LANES="${LANES:-4}"
log() { printf '[%s] %s\n' "$(date -Iseconds)" "$*"; }

if [ -n "${CKPT}" ]; then
    log "export ${NAME} from ${CKPT}"
    out="${REPO_ROOT}/logs/action_rate_bundles/bundles/${NAME}"
    rm -rf "${out}"
    (cd "${REPO_ROOT}" && pixi run -e onnx-export python -m imitation_experiments.lowlevel.export_policy_bundle \
        --checkpoint "${CKPT}" --skill-checkpoint "${ENCODER}" --preset combo_v3 --hold-steps 1 \
        --output "${out}" --verify 2>&1 | grep -E "parity|bundle written|verify:|Error") || { log "[FAIL] export"; exit 2; }
    rm -rf "${EC}/${BUNDLE}"; cp -r "${out}" "${EC}/${BUNDLE}"
fi
cd "${EC}"
[ -f "${BUNDLE}/manifest.json" ] || { log "[FAIL] no bundle at ${BUNDLE}"; exit 2; }

log "verify-bundle ${NAME}"
pixi run -e native ec lowlevel verify-bundle "${BUNDLE}" 2>&1 | grep -E '"valid"|max_abs_err' | head -4

for pass in measured training; do
    OUT="artifacts/smooth_${NAME}_${pass}"
    if [ -s "${OUT}/summary.json" ]; then log "[SKIP] ${OUT} exists"; continue; fi
    mkdir -p "${OUT}"
    if [ "${pass}" = measured ]; then extra="--dds-domain-base 100 --plant-noise measured"; else extra="--dds-domain-base 190 --fetch"; fi
    log "rehearse ${pass}: ${BUNDLE} all --lanes ${LANES} ${extra}"
    pixi run -e native ec lifecycle rehearse "${BUNDLE}" all --output "${OUT}" --lanes "${LANES}" ${extra} > "${OUT}.log" 2>&1
    log "rehearse ${pass} exit $? (1 = some motion failed, expected)"
    python3 - "${OUT}" <<'PY'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1]) / "summary.json"
if not p.exists():
    print("   no summary.json"); sys.exit()
s = json.load(open(p))
rows = s.get("rows", [])
passed = sum(1 for r in rows if r.get("passed"))
print(f"   clean motions: {s.get('motions_all_passed', passed)}/{s.get('motions', len(rows))}")
vids = list((pathlib.Path(sys.argv[1])).glob("*/sim/seed_0/video.mp4"))
print(f"   videos: {len(vids)} under {sys.argv[1]}/<motion>/sim/seed_0/video.mp4")
PY
done

log "dither (measured pass)"
timeout 300 pixi run -e native python docs/evidence/noise_20260910/dither.py "artifacts/smooth_${NAME}_measured" 2>&1 | grep -vE "Warning|warn" | tail -3
