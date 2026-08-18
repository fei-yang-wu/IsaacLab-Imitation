#!/usr/bin/env bash
# Isaac closed-loop evaluation of a trained head — the number of record.
#
#   ./eval.sh <fsq64_10b|ln_hold1_10b> [extra args...]
#
# Protocol, matched to the 2026-08-13 table so the rows are comparable: 29
# motions x 20 episodes, fall-only success (tracking terminations off,
# `base_too_low` active), 2000-step cap, Newton/MJWarp, push off, sensor noise
# ON (the environment default), exponential temporal ensembling with decay 0.5.
#
# ENSEMBLE=none reproduces the un-ensembled row of that table.
set -euo pipefail
CAMPAIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${CAMPAIGN_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"
source "${CAMPAIGN_DIR}/arms.sh"

ARM="${1:?usage: eval.sh <fsq64_10b|ln_hold1_10b>}"
shift || true
arm_config "${ARM}"
load_motions

UPDATE="${UPDATE:-0012000}"
HEAD="${HEAD:-${REPO_ROOT}/outputs/planner_10b/arms/${ARM}/checkpoints/update_${UPDATE}.pt}"
for required in "${HEAD}" "${GOAL_FEATURES}"; do
    [ -f "${required}" ] || { echo "missing input: ${required}" >&2; exit 1; }
done

EPISODES_PER_GOAL="${EPISODES_PER_GOAL:-20}"
NUM_ENVS=$(( NUM_GOALS * EPISODES_PER_GOAL ))
MAX_STEPS="${MAX_STEPS:-2000}"
ENSEMBLE="${ENSEMBLE:-exponential}"

# One goal per environment, in the same order the environments are laid out, so
# the goal is explicit and never inferred from the reference cursor.
GOALS=()
for motion in "${MOTIONS[@]}"; do
    for _ in $(seq 1 "${EPISODES_PER_GOAL}"); do GOALS+=("${motion}"); done
done

LABEL="${ARM}__u${UPDATE}__${ENSEMBLE}${LABEL_SUFFIX:+__${LABEL_SUFFIX}}"
OUTPUT_DIR="${REPO_ROOT}/logs/planner_10b/isaac_eval/${LABEL}"
if [ -e "${OUTPUT_DIR}" ]; then
    echo "Refusing to overwrite existing ${OUTPUT_DIR}" >&2
    exit 1
fi

pixi run -e isaaclab python scripts/rlopt/eval_skill_commander_closed_loop.py \
    --headless --task Isaac-Imitation-G1-v2 --algorithm IPMD \
    --agent_entry_point rlopt_ipmd_tuned_cfg_entry_point \
    --checkpoint "${TRACKER}" --skill_checkpoint "${ENCODER}" \
    --gr00t_checkpoint "${HEAD}" --gr00t_goal_features "${GOAL_FEATURES}" \
    --gr00t_goals_per_env "${GOALS[@]}" --gr00t_route latent \
    --gr00t_consumption "${GR00T_CONSUMPTION:-open_loop}" \
    --gr00t_inference_steps "${ODE_STEPS:-4}" \
    --gr00t_samples_per_publication "${SAMPLES:-1}" \
    --gr00t_temporal_ensemble "${ENSEMBLE}" \
    --gr00t_temporal_ensemble_decay "${ENSEMBLE_DECAY:-0.5}" \
    --gr00t_consume_slots "${CONSUME_SLOTS}" \
    --language_embeddings "${LANGUAGE_EMBEDDINGS}" \
    --state_history_steps 9 --output_dir "${OUTPUT_DIR}" --label "${LABEL}" \
    --num_envs "${NUM_ENVS}" --max_steps "${MAX_STEPS}" --seed "${SEED:-0}" \
    --metric_interval "${METRIC_INTERVAL:-10}" \
    --motion_names "${MOTIONS[@]}" --trajectory_ranks "${RANKS[@]}" \
    --disable_push_event --disable_reward_clipping --assert-kitless \
    --disable_tracking_terminations --fall_only_success \
    physics=newton_mjwarp env.data.manifest=null env.data.cache_dir=null \
    env.data.reference_arrays_dir="${REFERENCE_ARRAYS}" \
    env.data.persist_id="${PERSIST_ID}" \
    env.data.persist_dir=null env.data.macro_cache_device=cuda:0 \
    env.data.wrap_steps=false \
    "${RUNTIME_BODY_NAMES}" \
    env.command_interface.actor.dim="${ACTOR_DIM}" \
    'env.expert_macro_state_terms=[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b]' \
    agent.logger.backend= agent.ipmd.command_source=hl_skill \
    agent.ipmd.hl_skill_checkpoint_path="${ENCODER}" \
    agent.ipmd.hl_skill_finetune_enabled=false \
    agent.ipmd.latent_dim="${ACTOR_DIM}" \
    agent.ipmd.latent_steps_min="${HOLD}" agent.ipmd.latent_steps_max="${HOLD}" \
    agent.ipmd.hl_skill_horizon_steps=10 agent.ipmd.hl_skill_command_mode=z \
    agent.ipmd.latent_learning.command_phase_mode=sin_cos \
    agent.ipmd.latent_learning.code_latent_dim="${CODE_DIM}" \
    agent.ipmd.latent_learning.code_period="${CODE_PERIOD}" \
    env.sim.physics.solver_cfg.njmax="${NJMAX}" \
    env.sim.physics.solver_cfg.nconmax=200 \
    "${ARM_CFG[@]}" \
    "$@"

echo "retained: ${OUTPUT_DIR}"
