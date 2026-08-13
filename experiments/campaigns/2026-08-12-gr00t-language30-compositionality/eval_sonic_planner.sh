#!/usr/bin/env bash
# Isaac closed-loop evaluation of the SONIC-style hold-1 planner.
#
# The released SONIC v1.1 decoder is unchanged; only the source of its 64-D
# FSQ token changes, from its own reference-driven encoder to the GR00T head's
# causal prediction. Running the same binary with no `--gr00t_checkpoint`
# gives the oracle row of the same table, so the two differ in exactly one
# variable.
#
# The head predicts a 30-latent horizon and republishes every
# `PUBLISH_INTERVAL` control steps, consuming one latent per 50 Hz step. Slots
# past the interval are discarded (receding horizon).
#
# Usage: eval_sonic_planner.sh [oracle|planner] [extra args...]
set -euo pipefail
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

ROW="${1:-planner}"
shift || true

SONIC_CHECKPOINT="${SONIC_CHECKPOINT:-${HOME}/.cache/huggingface/hub/models--nvidia--GEAR-SONIC/blobs/af24831ae59424a0cf92cb56e9bb6dc1a59ab859fd055ba13187e9e6f0a59f43}"
DATA_ROOT="${REPO_ROOT}/data/bones_seed_language30_compositionality_v1"
MANIFEST_LANG="${DATA_ROOT}/manifests/g1_bones_seed_language30_compositionality_v1_manifest_language.json"
HEAD="${HEAD:-${REPO_ROOT}/outputs/gr00t_language30/arms/sonic_hold1/checkpoints/update_0012000.pt}"
GOAL_FEATURES="${REPO_ROOT}/outputs/gr00t_language30/goal_features/goal_features.pt"

readarray -t MOTIONS < <(
    "${REPO_ROOT}/.pixi/envs/default/bin/python" - "${MANIFEST_LANG}" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1]))
motions = manifest["motions"] if "motions" in manifest else manifest
print("\n".join(m["name"] for m in motions))
PY
)
NUM_GOALS="${#MOTIONS[@]}"
RANKS=($(seq 0 $((NUM_GOALS - 1))))
# Single-motion scoring, so a SONIC row can be compared against the
# single-motion ablation on the same motion rather than against a 30-goal
# average. SINGLE_MOTION is the manifest RANK.
if [ -n "${SINGLE_MOTION:-}" ]; then
    MOTIONS=("${MOTIONS[${SINGLE_MOTION}]}")
    RANKS=("${SINGLE_MOTION}")
    NUM_GOALS=1
fi
# Episodes per goal. The head's flow sampler is stochastic, so one episode per
# goal is not a stable number; extra episodes are extra environments in the
# same process.
EPISODES_PER_GOAL="${EPISODES_PER_GOAL:-5}"
NUM_ENVS=$(( NUM_GOALS * EPISODES_PER_GOAL ))
# `build_env_rank_assignment` splits environments into contiguous equal blocks,
# so the goal list must be blocked the same way. A cycling list silently pairs
# each environment with someone else's language; the guard catches it, but the
# blocking is what makes it correct.
GOALS=()
for motion in "${MOTIONS[@]}"; do
    for _ in $(seq 1 "${EPISODES_PER_GOAL}"); do
        GOALS+=("${motion}")
    done
done

# Row name selects the command source; OUTPUT_NAME separates protocol variants
# (strict vs full-horizon diagnostic) of the same row into their own retained
# directories, because the launcher refuses to overwrite one.
OUTPUT_DIR="${REPO_ROOT}/logs/gr00t_language30_sonic_eval/${OUTPUT_NAME:-${ROW}}"
if [ -e "${OUTPUT_DIR}" ]; then
    echo "Refusing to overwrite existing ${OUTPUT_DIR}" >&2
    exit 1
fi
mkdir -p "${OUTPUT_DIR}"

PLANNER_ARGS=()
if [ "${ROW}" = "planner" ]; then
    for required in "${HEAD}" "${GOAL_FEATURES}"; do
        [ -f "${required}" ] || { echo "missing input: ${required}" >&2; exit 1; }
    done
    PLANNER_ARGS=(
        --gr00t_checkpoint "${HEAD}"
        --gr00t_goal_features "${GOAL_FEATURES}"
        --gr00t_goals_per_env "${GOALS[@]}"
        --gr00t_publish_interval "${PUBLISH_INTERVAL:-10}"
    )
elif [ "${ROW}" != "oracle" ]; then
    echo "row must be oracle|planner, got ${ROW}" >&2
    exit 1
fi

VIDEO_ARGS=()
if [ "${VIDEO:-0}" = "1" ]; then
    VIDEO_ARGS=(--video --video_dir "${OUTPUT_DIR}/videos" --video_length 600)
fi

pixi run -e isaaclab python -m imitation_experiments.lowlevel.evaluate_sonic_release \
    --headless --task Isaac-Imitation-G1-v2 \
    --sonic_checkpoint "${SONIC_CHECKPOINT}" --sonic_version v1_1 \
    --label "sonic_hold1_${ROW}" \
    --num_envs "${NUM_ENVS}" --steps "${STEPS:-2000}" --seed "${SEED:-0}" \
    --trajectory_ranks "${RANKS[@]}" --reference_start_frame 0 \
    --randomization no_push --state_history_steps 9 \
    --termination_contract "${TERMINATION:-sonic}" \
    --output_json "${OUTPUT_DIR}/summary.json" \
    "${PLANNER_ARGS[@]}" ${VIDEO_ARGS[@]+"${VIDEO_ARGS[@]}"} \
    ${PHYSICS:+physics=${PHYSICS}} \
    env.sim.physics.solver_cfg.njmax="${NJMAX:-320}" \
    env.sim.physics.solver_cfg.nconmax="${NCONMAX:-200}" \
    env.data.manifest=null env.data.cache_dir=null \
    env.data.reference_arrays_dir="${DATA_ROOT}/reference_arrays/root_qpos_v1" \
    env.data.persist_id=bones_seed_language30_compositionality_v1@f31fd755 \
    env.data.persist_dir=null env.data.macro_cache_device=cuda:0 \
    env.data.wrap_steps=false \
    'env.data.runtime_cache_body_names=[pelvis,left_hip_roll_link,left_knee_link,left_ankle_roll_link,right_hip_roll_link,right_knee_link,right_ankle_roll_link,torso_link,left_shoulder_roll_link,left_elbow_link,left_wrist_yaw_link,right_shoulder_roll_link,right_elbow_link,right_wrist_yaw_link]' \
    "$@"

echo "retained: ${OUTPUT_DIR}"
