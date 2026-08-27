#!/usr/bin/env bash
# Isaac closed-loop evaluation of the EXPLICIT chunk head (`explicit_10b`),
# on either of its two routes:
#
#   ./eval_explicit.sh native   [extra args...]   # explicit tracker row (B)
#   ./eval_explicit.sh encoded  [extra args...]   # re-encoded latent row (C)
#
#   ASYNC=1 ./eval_explicit.sh <route>            # D1 service-backed run
#
# The head trains on the fsq64_10b collection's `chunk_target` (the only
# collection storing the 30-frame root_qpos lookahead), so its packets live
# in the ROBOT_HEADING frame of the request instant.
#
# * `native` drives the 7.6B `root_qpos_explicit` tracker (frames unmatched
#   with the 10B latent trackers — state that caveat with the row). That
#   tracker predates SONIC v1.1, so the env runs in the default `robot`
#   anchor mode and `--gr00t_packet_frame heading` pins the head's heading
#   frame on the chunk term at every publish.
# * `encoded` routes the same head's packet through the frozen ln_hold1_10b
#   encoder onto the 10B `cont_det_ln_hold1` tracker — the tracker-matched
#   latent-vs-explicit row against `eval.sh ln_hold1_10b`.
#   `--gr00t_packet_consume_frames 10` gives it the latent arm's re-plan
#   cadence (30-frame plan, re-planned every 10 steps).
#
# Protocol matches eval.sh: 28 motions x 20 episodes, fall-only success,
# 2000-step cap, Newton/MJWarp, DR=off (`--deterministic_tracking`),
# ENSEMBLE none on both routes (async refuses it; the sync row is its
# matched companion).
set -euo pipefail
CAMPAIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${CAMPAIGN_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"
source "${CAMPAIGN_DIR}/arms.sh"

ROUTE="${1:?usage: eval_explicit.sh <native|encoded>}"
shift || true
load_motions

UPDATE="${UPDATE:-0012000}"
# ARM selects which explicit head runs: `explicit_10b` trains on the fsq64
# collection, `explicit_ln_10b` on the hold-1 collection re-run with the
# lookahead (the confound-free row against the latent hold-1 arm).
ARM="${ARM:-explicit_10b}"
HEAD="${HEAD:-${REPO_ROOT}/outputs/planner_10b/arms/${ARM}/checkpoints/update_${UPDATE}.pt}"
ASYNC="${ASYNC:-0}"
LEAD_STEPS="${LEAD_STEPS:-5}"

case "${ROUTE}" in
    native)
        GR00T_ROUTE=chunk_native
        TRACKER="${REPO_ROOT}/logs/bones129k_4096_scoreboard/root_qpos_explicit/model_step_7600078848.pt"
        ENCODER="${REPO_ROOT}/logs/rollout24_gamma097_foot_disabled_eval/encoder/latest.pt"
        ACTOR_DIM=258; CODE_DIM=256; HOLD=10; CODE_PERIOD=10; NJMAX=289
        # The 7.6B explicit tracker keeps its own training-time geometry: the
        # default policy cells and the default `robot` anchor mode.
        ARM_CFG=()
        ACTOR_CFG=(
            env.command_interface.actor=chunk
            env.command_interface.actor.source=external
            env.command_interface.actor.horizon=30
            env.command_interface.actor.hold_steps=10
            'env.command_interface.actor.components=[joint_qpos,root_pos,root_ori]'
        )
        ROUTE_ARGS=(--gr00t_packet_frame heading)
        ;;
    encoded)
        GR00T_ROUTE=chunk_encoded
        # TRACKER_ARM picks which latent tracker the packet is encoded onto.
        # `fsq64_10b` is the confound-free pairing for the `explicit_10b` head:
        # that head trained on the fsq64 collection, so encoding onto the fsq64
        # tracker matches the rollout states, the tracker and the budget with
        # the fsq64 latent row, leaving the planner's output space as the only
        # difference.
        arm_config "${TRACKER_ARM:-ln_hold1_10b}"
        ACTOR_CFG=(env.command_interface.actor.dim="${ACTOR_DIM}")
        # CONSUME_FRAMES=none drops the cursor and re-plans at every
        # publication (the pre-2026-08-18 path), which is the control that
        # separates the interface result from the cursor implementation.
        CONSUME_FRAMES="${CONSUME_FRAMES:-10}"
        if [ "${CONSUME_FRAMES}" = "none" ]; then
            ROUTE_ARGS=()
        else
            ROUTE_ARGS=(--gr00t_packet_consume_frames "${CONSUME_FRAMES}")
        fi
        ;;
    *) echo "route must be native|encoded, got ${ROUTE}" >&2; exit 1 ;;
esac

for required in "${HEAD}" "${GOAL_FEATURES}" "${TRACKER}" "${ENCODER}"; do
    [ -f "${required}" ] || { echo "missing input: ${required}" >&2; exit 1; }
done

EPISODES_PER_GOAL="${EPISODES_PER_GOAL:-20}"
NUM_ENVS=$(( NUM_GOALS * EPISODES_PER_GOAL ))
MAX_STEPS="${MAX_STEPS:-2000}"
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

if [ "${ASYNC}" = "1" ]; then
    SERVICE_SEED="${SERVICE_SEED:-0}"
    ENDPOINT="${ENDPOINT:-ipc:///tmp/gr00t_batch_${ARM}_${ROUTE}_$$.ipc}"
    SERVICE_LOG="${SERVICE_LOG:-${REPO_ROOT}/logs/planner_10b/isaac_eval/service_${ARM}_${ROUTE}_$$.log}"
    mkdir -p "$(dirname "${SERVICE_LOG}")"
    pixi run -e gr00t python -m imitation_experiments.planner.gr00t_batch_service \
        --checkpoint "${HEAD}" \
        --goal-features "${GOAL_FEATURES}" \
        --endpoint "${ENDPOINT}" \
        --seed "${SERVICE_SEED}" --dtype float32 \
        > "${SERVICE_LOG}" 2>&1 &
    SERVICE_PID=$!
    trap 'kill "${SERVICE_PID}" 2>/dev/null || true' EXIT
    for _ in $(seq 1 240); do
        if ! kill -0 "${SERVICE_PID}" 2>/dev/null; then
            echo "service exited before ready; log tail:" >&2
            tail -5 "${SERVICE_LOG}" >&2
            exit 1
        fi
        grep -q '"ready":true' "${SERVICE_LOG}" && break
        sleep 1
    done
    grep -q '"ready":true' "${SERVICE_LOG}" || { echo "service never became ready" >&2; exit 1; }
    echo "[D1] service ready on ${ENDPOINT} (pid ${SERVICE_PID})"
    ROUTE_ARGS+=(--gr00t_service "${ENDPOINT}" --gr00t_lead_steps "${LEAD_STEPS}")
    ASYNC_SUFFIX="__async_lead${LEAD_STEPS}"
else
    ASYNC_SUFFIX=""
fi

GOALS=()
for motion in "${MOTIONS[@]}"; do
    for _ in $(seq 1 "${EPISODES_PER_GOAL}"); do GOALS+=("${motion}"); done
done

TRACKER_TAG=""
[ "${ROUTE}" = "encoded" ] && TRACKER_TAG="__on_${TRACKER_ARM:-ln_hold1_10b}"
LABEL="${ARM}__u${UPDATE}__${ROUTE}${TRACKER_TAG}__dr${DR}${ASYNC_SUFFIX}${VIDEO_SUFFIX}${LABEL_SUFFIX:+__${LABEL_SUFFIX}}"
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
    --gr00t_goals_per_env "${GOALS[@]}" --gr00t_route "${GR00T_ROUTE}" \
    --gr00t_consumption open_loop \
    --gr00t_inference_steps "${ODE_STEPS:-4}" \
    --gr00t_samples_per_publication 1 \
    --gr00t_temporal_ensemble none \
    "${ROUTE_ARGS[@]}" \
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
    "${ACTOR_CFG[@]}" \
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
    ${ARM_CFG[@]+"${ARM_CFG[@]}"} \
    "$@"

echo "retained: ${OUTPUT_DIR}"
