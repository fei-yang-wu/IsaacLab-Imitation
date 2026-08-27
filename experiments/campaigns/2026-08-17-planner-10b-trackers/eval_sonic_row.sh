#!/usr/bin/env bash
# The released NVIDIA SONIC v1.1 row, on THIS campaign's motion set and
# protocol, so it can sit next to the 10B planner rows.
#
#   ./eval_sonic_row.sh planner   # SONIC decoder driven by the GR00T head
#   ./eval_sonic_row.sh oracle    # SONIC decoder driven by its own encoder
#
# What is matched with `./eval.sh <arm>` rows:
#   * the 28-motion set and their manifest ranks (arms.sh, EXCLUDE_RANKS),
#   * 20 episodes per goal (560 environments), 2000-step cap,
#   * fall-only success (`--termination_contract fall_only`),
#   * unperturbed rollouts: `--randomization none` plus start on the reference
#     (`--reference_start_frame 0`), the same thing `--deterministic_tracking`
#     does on our side,
#   * clean observations. The SONIC evaluator disables observation corruption
#     by contract, so our comparable rows must run `OBS_NOISE=off ./eval.sh`.
#
# What CANNOT be matched, and must be said with the number: the decoder is
# SONIC's own (a different tracker from our 10B trackers, trained on SONIC's
# corpus), and this is a different evaluator binary. The row answers "how does
# the released system score under our protocol", not "which interface is
# better" — that comparison needs one tracker.
set -euo pipefail
CAMPAIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${CAMPAIGN_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"
source "${CAMPAIGN_DIR}/arms.sh"
load_motions

ROW="${1:-planner}"
shift || true

SONIC_CHECKPOINT="${SONIC_CHECKPOINT:-${HOME}/.cache/huggingface/hub/models--nvidia--GEAR-SONIC/blobs/af24831ae59424a0cf92cb56e9bb6dc1a59ab859fd055ba13187e9e6f0a59f43}"
# The SONIC-token head from the 2026-08-12 campaign: it predicts the 64-D FSQ
# token SONIC's own encoder produces, at hold 1, over a 30-step horizon.
HEAD="${HEAD:-${REPO_ROOT}/outputs/gr00t_language30/arms/sonic_hold1/checkpoints/update_0012000.pt}"
[ -f "${SONIC_CHECKPOINT}" ] || { echo "missing SONIC release blob: ${SONIC_CHECKPOINT}" >&2; exit 1; }

EPISODES_PER_GOAL="${EPISODES_PER_GOAL:-20}"
NUM_ENVS=$(( NUM_GOALS * EPISODES_PER_GOAL ))
STEPS="${STEPS:-2000}"
PUBLISH_INTERVAL="${PUBLISH_INTERVAL:-10}"

GOALS=()
for motion in "${MOTIONS[@]}"; do
    for _ in $(seq 1 "${EPISODES_PER_GOAL}"); do GOALS+=("${motion}"); done
done

PLANNER_ARGS=()
if [ "${ROW}" = "planner" ]; then
    [ -f "${HEAD}" ] || { echo "missing head: ${HEAD}" >&2; exit 1; }
    [ -f "${GOAL_FEATURES}" ] || { echo "missing goal features: ${GOAL_FEATURES}" >&2; exit 1; }
    PLANNER_ARGS=(
        --gr00t_checkpoint "${HEAD}"
        --gr00t_goal_features "${GOAL_FEATURES}"
        --gr00t_goals_per_env "${GOALS[@]}"
        --gr00t_publish_interval "${PUBLISH_INTERVAL}"
    )
elif [ "${ROW}" != "oracle" ]; then
    echo "row must be oracle|planner, got ${ROW}" >&2
    exit 1
fi

LABEL="sonic_release__${ROW}__n${EPISODES_PER_GOAL}__m${NUM_GOALS}__dr_off${LABEL_SUFFIX:+__${LABEL_SUFFIX}}"
OUTPUT_DIR="${REPO_ROOT}/logs/planner_10b/isaac_eval/${LABEL}"
if [ -e "${OUTPUT_DIR}" ]; then
    echo "Refusing to overwrite existing ${OUTPUT_DIR}" >&2
    exit 1
fi
mkdir -p "${OUTPUT_DIR}"

pixi run -e isaaclab python -m imitation_experiments.lowlevel.evaluate_sonic_release \
    --headless --task Isaac-Imitation-G1-v2 \
    --sonic_checkpoint "${SONIC_CHECKPOINT}" --sonic_version v1_1 \
    --label "${LABEL}" \
    --num_envs "${NUM_ENVS}" --steps "${STEPS}" --seed "${SEED:-0}" \
    --trajectory_ranks "${RANKS[@]}" --reference_start_frame 0 \
    --randomization none --state_history_steps 9 \
    --termination_contract "${TERMINATION:-fall_only}" \
    --output_json "${OUTPUT_DIR}/summary.json" \
    "${PLANNER_ARGS[@]}" \
    physics=newton_mjwarp \
    env.sim.physics.solver_cfg.njmax="${NJMAX:-320}" \
    env.sim.physics.solver_cfg.nconmax=200 \
    env.data.manifest=null env.data.cache_dir=null \
    env.data.reference_arrays_dir="${REFERENCE_ARRAYS}" \
    env.data.persist_id="${PERSIST_ID}" \
    env.data.persist_dir=null env.data.macro_cache_device=cuda:0 \
    env.data.wrap_steps=false \
    "${RUNTIME_BODY_NAMES}" \
    "$@"

echo "retained: ${OUTPUT_DIR}"
