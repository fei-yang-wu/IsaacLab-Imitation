#!/usr/bin/env bash
# Score an UNTRAINED tracker on the curve board, locally: the frame-0 point
# that every convergence curve starts from.
#
# The evaluator arguments are the hub's `score` stage from campaign.yaml,
# verbatim, with the two ICE paths (encoder file, reference arrays) mapped to
# the workstation mirrors. The agent is built exactly as for a trained cell
# (pretrained hub encoder, 66-wide command, hold 1) and its actor is left at
# initialization (`--untrained_policy`, seeded by `--seed 0`).
#
# The row lands in the curve corpus as `untrained_seed0_clean_f0.json`. The
# arm name is `untrained`, not `hub`: the actor init is not bit-identical to
# the hub's own training start (construction order differs), and the point is
# meant to be shared by every arm.
#
#   ./eval_untrained_local.sh
set -euo pipefail
CAMPAIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${CAMPAIGN_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/logs/latent_star_v2_curves}"
ENCODER_LOCAL="${ENCODER_LOCAL:-${REPO_ROOT}/logs/latent_star_v2_mirror/hub_seed0/encoder/checkpoints/latest.pt}"
REFERENCE_ARRAYS="${REFERENCE_ARRAYS:-/mnt/hsstorage/fwu91/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
BOARD=bones_testbed4096_v1
OUT="${OUTPUT_ROOT}/untrained_seed0_clean_f0.json"

[[ -s "${ENCODER_LOCAL}" ]] || { echo "[FATAL] encoder missing: ${ENCODER_LOCAL}"; exit 2; }
[[ -s "${REFERENCE_ARRAYS}/reference_arrays_manifest.json" ]] || {
    echo "[FATAL] reference arrays missing: ${REFERENCE_ARRAYS}"; exit 2; }
mkdir -p "${OUTPUT_ROOT}"

# Evaluator args after `--` of the hub stage, paths remapped, board expanded.
mapfile -t common_args < <(pixi run python - "${CAMPAIGN_DIR}/campaign.yaml" \
    "${ENCODER_LOCAL}" "${REFERENCE_ARRAYS}" "${BOARD}" <<'PY'
import sys, yaml
from imitation_experiments.evaluation.protocol import BOARDS
campaign_path, encoder, ref_arrays, board = sys.argv[1:5]
campaign = yaml.safe_load(open(campaign_path))
stage = campaign["arms"]["hub"]["stages"][0]
args = stage["args"][stage["args"].index("--") + 1 :]
out = []
for a in args:
    if a.startswith("agent.ipmd.hl_skill_checkpoint_path="):
        a = f"agent.ipmd.hl_skill_checkpoint_path={encoder}"
    elif a.startswith("env.data.reference_arrays_dir="):
        a = f"env.data.reference_arrays_dir={ref_arrays}"
    elif a == "env.data.reference_arrays_resident=true":
        a = "env.data.reference_arrays_resident=false"
    out.append(a)
ranks = [str(c.trajectory_rank) for c in BOARDS[board].cases]
out += ["--num_envs", str(len(ranks)), "--trajectory_ranks", *ranks]
print("\n".join(out))
PY
)

echo "[INFO] untrained policy on ${BOARD}, $(( ${#common_args[@]} )) evaluator tokens"
env TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 \
    TORCHDYNAMO_DISABLE=1 \
    pixi run -e isaaclab python -u -m imitation_experiments.lowlevel.evaluate_checkpoint \
    --untrained_policy --output_json "${OUT}" --label untrained_seed0_clean_f0 \
    --kit_args=--/app/extensions/fsWatcherEnabled=false \
    "${common_args[@]}" > "${OUT}.log" 2>&1
rc=$?
if (( rc != 0 )) || [[ ! -s "${OUT}" ]]; then
    echo "[FAIL] exit ${rc}; $(grep -iE 'error|out of memory' "${OUT}.log" | tail -2 | tr '\n' ' ' | cut -c1-200)"
    exit 1
fi
echo "[OK] ${OUT}"
pixi run python -m imitation_experiments.evaluation.summarize_paper_boards "${OUT}"
