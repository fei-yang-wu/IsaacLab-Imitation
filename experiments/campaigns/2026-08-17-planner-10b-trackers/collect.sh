#!/usr/bin/env bash
# Oracle-latent rollout collection for one arm and one seed.
#
# The frozen tracker drives the robot from its own encoder's latent, and a row
# is saved on EVERY control step, so the causal state history carries real
# closed-loop dynamics and this tracker's own `last_action`. The publication
# cadence is the arm's trained one (hold 10 or hold 1); only the SAMPLING is
# per-step, which gives the prepare join ten times the training pairs on a
# hold-10 arm and the exact per-step target on a hold-1 arm.
#
#   ./collect.sh <fsq64_10b|ln_hold1_10b> [seed]
#
# ONE run per arm, not several merged: `prepare_gr00t_dataset` keys rows by
# (env_id, episode_id, control_step) and every run numbers environments from
# zero, so two collections of the same arm collide on that key and the prepare
# step refuses them ("collection is not join-safe"). 30 goals x 93 environments
# x 500 steps is about 1.11M rows before early reference ends, against the
# 889,044 rows of the 2026-08-13 arm that reached 46.95 mm.
set -euo pipefail
CAMPAIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${CAMPAIGN_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"
source "${CAMPAIGN_DIR}/arms.sh"

ARM="${1:?usage: collect.sh <fsq64_10b|ln_hold1_10b> [seed]}"
SEED="${2:-0}"
arm_config "${ARM}"
load_motions

PER_GOAL="${PER_GOAL:-93}"
NUM_ENVS=$(( NUM_GOALS * PER_GOAL ))
MAX_STEPS="${MAX_STEPS:-500}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/logs/planner_10b/${ARM}/collection_seed${SEED}}"
if [ -e "${OUTPUT_DIR}" ]; then
    echo "Refusing to overwrite existing ${OUTPUT_DIR}" >&2
    exit 1
fi

echo "[COLLECT] ${ARM} seed ${SEED}: ${NUM_GOALS} goals x ${PER_GOAL} envs"
echo "[COLLECT] hold ${HOLD}, code_period ${CODE_PERIOD}, actor dim ${ACTOR_DIM}"

# The 30-frame expert lookahead is stored only for the arm whose targets are
# re-encoded from it (`latent.source: fsq_prequant`). On the hold-1 arm the
# target is the stored per-step z, and a lookahead on every row would multiply
# the collection's size for data the prepare step never reads.
#
# FORCE_ROOT_QPOS=1 stores it anyway. That is how the hold-1 arm gets a
# collection an EXPLICIT head can train on: same rollout states as its latent
# counterpart, plus the lookahead the chunk target needs.
SAMPLE_ARGS=()
if [ "${FORCE_ROOT_QPOS:-0}" = "1" ]; then
    WITH_ROOT_QPOS=1
fi
if [ "${WITH_ROOT_QPOS}" = "1" ]; then
    SAMPLE_ARGS=(--require_root_qpos_samples --sample_future_window_frames 30)
fi

pixi run -e isaaclab python scripts/rlopt/eval_skill_commander_closed_loop.py \
    --headless --task Isaac-Imitation-G1-v2 --algorithm IPMD \
    --agent_entry_point rlopt_ipmd_tuned_cfg_entry_point \
    --checkpoint "${TRACKER}" --skill_checkpoint "${ENCODER}" \
    --language_embeddings "${LANGUAGE_EMBEDDINGS}" \
    --state_history_steps 9 \
    --output_dir "${OUTPUT_DIR}" --label "planner_10b_${ARM}_seed${SEED}" \
    --num_envs "${NUM_ENVS}" --max_steps "${MAX_STEPS}" --seed "${SEED}" \
    --metric_interval 10 \
    --motion_names "${MOTIONS[@]}" --trajectory_ranks "${RANKS[@]}" \
    --save_rollout_training_samples --sample_every_control_step \
    --sample_rows_per_file 8192 \
    ${SAMPLE_ARGS[@]+"${SAMPLE_ARGS[@]}"} \
    --disable_tracking_terminations --fall_only_success \
    --disable_push_event --disable_reward_clipping --assert-kitless \
    physics=newton_mjwarp \
    env.data.manifest=null env.data.cache_dir=null \
    env.data.reference_arrays_dir="${REFERENCE_ARRAYS}" \
    env.data.persist_id="${PERSIST_ID}" \
    env.data.persist_dir=null env.data.reference_arrays_warm_workers=2 \
    env.data.macro_cache_device=cuda:0 \
    "${RUNTIME_BODY_NAMES}" \
    env.data.wrap_steps=false \
    env.command_interface.actor.dim="${ACTOR_DIM}" \
    'env.expert_macro_state_terms=[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b]' \
    agent.logger.backend= \
    agent.ipmd.command_source=hl_skill \
    agent.ipmd.hl_skill_checkpoint_path="${ENCODER}" \
    agent.ipmd.hl_skill_finetune_enabled=false \
    agent.ipmd.latent_dim="${ACTOR_DIM}" \
    agent.ipmd.latent_steps_min="${HOLD}" agent.ipmd.latent_steps_max="${HOLD}" \
    agent.ipmd.hl_skill_horizon_steps=10 \
    agent.ipmd.hl_skill_command_mode=z \
    agent.ipmd.latent_learning.command_phase_mode=sin_cos \
    agent.ipmd.latent_learning.code_latent_dim="${CODE_DIM}" \
    agent.ipmd.latent_learning.code_period="${CODE_PERIOD}" \
    env.sim.physics.solver_cfg.njmax="${NJMAX}" \
    env.sim.physics.solver_cfg.nconmax=200 \
    "${ARM_CFG[@]}" \
    "${@:3}"

echo "retained: ${OUTPUT_DIR}"
