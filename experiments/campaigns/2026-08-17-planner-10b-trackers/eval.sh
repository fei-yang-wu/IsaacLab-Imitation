#!/usr/bin/env bash
# Isaac closed-loop evaluation of a trained head — the number of record.
#
#   ./eval.sh <fsq64_10b|ln_hold1_10b> [extra args...]
#
# Protocol: 28 motions x 20 episodes, fall-only success (tracking terminations
# off, `base_too_low` active), 2000-step cap, Newton/MJWarp, exponential
# temporal ensembling with decay 0.5.
#
# DR=off (the default, user directive 2026-08-17) passes
# `--deterministic_tracking`: pushes AND domain randomization off, and the
# episode starts exactly on the reference. The evaluator prefixes every metric
# with `deterministic_tracking/` so an unperturbed number can never be pooled
# with a perturbed one. DR=on restores the perturbed protocol of the
# 2026-08-13 table (`--disable_push_event`, other randomization kept), which is
# what the 46.95 mm row was measured under — so a DR=off number is NOT directly
# comparable with it.
#
# VIDEO=1 records the rollout. Render one episode per motion (
# `EPISODES_PER_GOAL=1`) for a clip pass; recording 560 environments is a very
# large render for no extra information.
#
# ENSEMBLE=none drops temporal ensembling.
set -euo pipefail
CAMPAIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${CAMPAIGN_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"
source "${CAMPAIGN_DIR}/arms.sh"

ARM="${1:?usage: eval.sh <fsq64_10b|ln_hold1_10b>}"
shift || true
arm_config "${ARM}"
load_motions

# The arm table fixes the tracker's trained cadence; FORCE_CONSUME_SLOTS
# changes only how much of each prediction is consumed before re-planning,
# which is a deployment knob, not a tracker change.
CONSUME_SLOTS="${FORCE_CONSUME_SLOTS:-${CONSUME_SLOTS}}"

UPDATE="${UPDATE:-0012000}"
HEAD="${HEAD:-${REPO_ROOT}/outputs/planner_10b/arms/${ARM}/checkpoints/update_${UPDATE}.pt}"
# ORACLE=1 drops the head and lets the frozen encoder publish the latents: the
# same protocol, the same tracker, the reference in the loop. It is the
# planner's ceiling, so the planner's error is (row - oracle), not the row.
ORACLE="${ORACLE:-0}"
if [ "${ORACLE}" != "1" ]; then
    for required in "${HEAD}" "${GOAL_FEATURES}"; do
        [ -f "${required}" ] || { echo "missing input: ${required}" >&2; exit 1; }
    done
fi
# OBS_NOISE=off disables the observation corruption the tracker was TRAINED
# with. `--deterministic_tracking` does not touch it (it removes events and
# reset ranges only), while the SONIC release evaluator disables it by
# contract — so a row compared against a released SONIC number must set this,
# or the comparison hands SONIC clean observations and us noisy ones.
OBS_NOISE="${OBS_NOISE:-on}"
case "${OBS_NOISE}" in
    on) OBS_ARGS=() ;;
    off) OBS_ARGS=(
            env.observations.policy.enable_corruption=false
            env.observations.critic.enable_corruption=false
         ) ;;
    *) echo "OBS_NOISE must be on|off, got ${OBS_NOISE}" >&2; exit 1 ;;
esac

EPISODES_PER_GOAL="${EPISODES_PER_GOAL:-20}"
NUM_ENVS=$(( NUM_GOALS * EPISODES_PER_GOAL ))
MAX_STEPS="${MAX_STEPS:-2000}"
ENSEMBLE="${ENSEMBLE:-exponential}"
DR="${DR:-off}"
VIDEO="${VIDEO:-0}"

case "${DR}" in
    off) PROTOCOL_ARGS=(--deterministic_tracking) ;;
    on)  PROTOCOL_ARGS=(--disable_push_event) ;;
    *) echo "DR must be off|on, got ${DR}" >&2; exit 1 ;;
esac
VIDEO_SUFFIX=""
if [ "${VIDEO}" = "1" ]; then
    PROTOCOL_ARGS+=(--video)
    VIDEO_SUFFIX="__video"
fi

# One goal per environment, in the same order the environments are laid out, so
# the goal is explicit and never inferred from the reference cursor.
GOALS=()
for motion in "${MOTIONS[@]}"; do
    for _ in $(seq 1 "${EPISODES_PER_GOAL}"); do GOALS+=("${motion}"); done
done

# Built after GOALS: the per-environment goal assignment is one of the flags.
if [ "${ORACLE}" = "1" ]; then
    PLANNER_ARGS=()
else
    PLANNER_ARGS=(
        --gr00t_checkpoint "${HEAD}"
        --gr00t_goal_features "${GOAL_FEATURES}"
        --gr00t_goals_per_env "${GOALS[@]}"
        --gr00t_route latent
        --gr00t_consumption "${GR00T_CONSUMPTION:-open_loop}"
        --gr00t_inference_steps "${ODE_STEPS:-4}"
        --gr00t_samples_per_publication "${SAMPLES:-1}"
        --gr00t_temporal_ensemble "${ENSEMBLE}"
        --gr00t_temporal_ensemble_decay "${ENSEMBLE_DECAY:-0.5}"
        --gr00t_consume_slots "${CONSUME_SLOTS}"
    )
fi

ORACLE_SUFFIX=""
[ "${ORACLE}" = "1" ] && ORACLE_SUFFIX="__oracle"
NOISE_SUFFIX=""
[ "${OBS_NOISE}" = "off" ] && NOISE_SUFFIX="__cleanobs"
LABEL="${ARM}__u${UPDATE}__${ENSEMBLE}__dr${DR}${ORACLE_SUFFIX}${NOISE_SUFFIX}${VIDEO_SUFFIX}${LABEL_SUFFIX:+__${LABEL_SUFFIX}}"
OUTPUT_DIR="${REPO_ROOT}/logs/planner_10b/isaac_eval/${LABEL}"
if [ -e "${OUTPUT_DIR}" ]; then
    echo "Refusing to overwrite existing ${OUTPUT_DIR}" >&2
    exit 1
fi

pixi run -e isaaclab python scripts/rlopt/eval_skill_commander_closed_loop.py \
    --headless --task Isaac-Imitation-G1-v2 --algorithm IPMD \
    --agent_entry_point rlopt_ipmd_tuned_cfg_entry_point \
    --checkpoint "${TRACKER}" --skill_checkpoint "${ENCODER}" \
    ${PLANNER_ARGS[@]+"${PLANNER_ARGS[@]}"} \
    --language_embeddings "${LANGUAGE_EMBEDDINGS}" \
    --state_history_steps 9 --output_dir "${OUTPUT_DIR}" --label "${LABEL}" \
    --num_envs "${NUM_ENVS}" --max_steps "${MAX_STEPS}" --seed "${SEED:-0}" \
    --metric_interval "${METRIC_INTERVAL:-10}" \
    --motion_names "${MOTIONS[@]}" --trajectory_ranks "${RANKS[@]}" \
    --disable_reward_clipping --assert-kitless \
    --disable_tracking_terminations --fall_only_success \
    "${PROTOCOL_ARGS[@]}" \
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
    ${OBS_ARGS[@]+"${OBS_ARGS[@]}"} \
    "$@"

echo "retained: ${OUTPUT_DIR}"
