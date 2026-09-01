#!/usr/bin/env bash
# Studio-lit clips of the language-conditioned planner stack, for the teaser
# and the results row.
#
#   ./render_studio_clips.sh                    # the four figure motions
#   RANKS_TO_RENDER="13 29" ./render_studio_clips.sh
#   PREVIEW=1 ./render_studio_clips.sh          # style x shot contact sheet
#
# This is `scripts/viz/render_paper_policy_video.py` (three-point studio rig,
# cyclorama, chase camera on a 35 lens) driven by the GR00T head instead of the
# frozen encoder's oracle latents, so the clip shows the deployed system:
# language goal -> head -> latent -> tracker. One MP4 per rank, one Isaac
# process for all of them, so the 4.9 GB head loads once.
#
# The renderer needs the Kit RTX camera, which exists only under PhysX, so
# these clips run `physics=physx` while every reported number stays on the
# Newton evaluator.
set -euo pipefail
CAMPAIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${CAMPAIGN_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"
source "${CAMPAIGN_DIR}/arms.sh"

ARM="${ARM:-ln_hold1_10b}"
arm_config "${ARM}"
load_motions

UPDATE="${UPDATE:-0012000}"
HEAD="${HEAD:-${REPO_ROOT}/outputs/planner_10b/arms/${ARM}/checkpoints/update_${UPDATE}.pt}"
# TAKES_DIR=<dir> re-renders saved motion takes instead of driving the stack.
# Nothing is loaded but the scene: no head, no tracker, no physics.
TAKES=()
RANKS_TO_RENDER="${RANKS_TO_RENDER:-29 17 13 2}"
if [ -n "${TAKES_DIR:-}" ]; then
    for rank in ${RANKS_TO_RENDER:-}; do
        for candidate in "${TAKES_DIR}"/rank-$(printf '%06d' "${rank}")-*.npz; do
            [ -f "${candidate}" ] && TAKES+=("${candidate}")
        done
    done
    [ "${#TAKES[@]}" -gt 0 ] || { echo "no takes in ${TAKES_DIR}" >&2; exit 1; }
elif [ "${ORACLE:-0}" != "1" ]; then
    for required in "${HEAD}" "${GOAL_FEATURES}"; do
        [ -f "${required}" ] || { echo "missing input: ${required}" >&2; exit 1; }
    done
fi

# Manifest ranks in data/bones_seed_language30_compositionality_v1:
#   29 slow walk forward
#   17 lifting a crate, walk start, walk forward
#   13 opening the right side door from inside, walking out, closing it
#    2 picking a big heavy object up low up front and putting it away high

# The goal is explicit per clip and is read from the manifest by rank, so the
# language can never drift from the reference the clip plays.
GOALS=()
for rank in ${RANKS_TO_RENDER}; do
    GOALS+=("${ALL_MOTIONS[$rank]}")
done

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/logs/planner_10b/studio_clips/${ARM}_u${UPDATE}}"
STYLE="${STYLE:-studio_light}"
SHOT="${SHOT:-hero_low}"
VIDEO_WIDTH="${VIDEO_WIDTH:-1920}"
VIDEO_HEIGHT="${VIDEO_HEIGHT:-1080}"
STILLS_EVERY="${STILLS_EVERY:-50}"

EXTRA_ARGS=()
[ "${PREVIEW:-0}" = "1" ] && EXTRA_ARGS+=(--preview)
# SHOT=sequence renders the stroboscopic composite instead of a clip: one image
# per motion with the robot drawn at several poses. A walking motion spreads
# itself along the path it travels; a standing manipulation does not move, so
# the composite lays its cut-outs out across the frame (--sequence_spread).
if [ "${SHOT}" = "sequence" ]; then
    EXTRA_ARGS+=(
        --sequence_poses "${SEQUENCE_POSES:-6}"
        --sequence_spread "${SEQUENCE_SPREAD:-auto}"
        --sequence_spread_pitch "${SEQUENCE_SPREAD_PITCH:-0.85}"
    )
    # POSE_STEPS="12 40 88 ..." draws exactly those frames instead of sampling
    # evenly along the path.
    if [ -n "${POSE_STEPS:-}" ]; then
        EXTRA_ARGS+=(--sequence_pose_steps "${POSE_STEPS// /,}")
    fi
    EXTRA_ARGS+=(
        # Solid poses, not a fade: a faded pose reproduces badly on paper and
        # the reading direction already carries the order.
        --sequence_alpha_min "${SEQUENCE_ALPHA_MIN:-1.0}"
        --sequence_time_direction "${SEQUENCE_TIME_DIRECTION:-right_to_left}"
    )
fi
# RECORD_TAKES=<dir> saves the achieved motion of every clip. A take replays
# with `--takes`, which needs neither the head nor the tracker, so reframing or
# relighting a figure later costs a render and nothing else.
[ -n "${RECORD_TAKES:-}" ] && EXTRA_ARGS+=(--record_takes "${RECORD_TAKES}")
# STRAIGHTEN_TAKES=off keeps a replayed take's real root path. The composite
# still lays the poses out, so the robot walks its own arc and only the
# cut-outs move.
[ -n "${STRAIGHTEN_TAKES:-}" ] && EXTRA_ARGS+=(--straighten_takes "${STRAIGHTEN_TAKES}")

if [ "${#TAKES[@]}" -gt 0 ]; then
    DRIVE_ARGS=(--takes "${TAKES[@]}")
elif [ "${ORACLE:-0}" = "1" ]; then
    # No head: the frozen encoder publishes the latents, which is the tracker's
    # own ceiling rather than the deployed system. Say so in any caption.
    DRIVE_ARGS=(--checkpoint "${TRACKER}" --ranks ${RANKS_TO_RENDER})
else
    DRIVE_ARGS=(
        --checkpoint "${TRACKER}"
        --ranks ${RANKS_TO_RENDER}
        --gr00t_checkpoint "${HEAD}"
        --gr00t_goal_features "${GOAL_FEATURES}"
        --gr00t_goals "${GOALS[@]}"
        --gr00t_consumption open_loop
        --gr00t_inference_steps "${ODE_STEPS:-4}"
        --gr00t_samples_per_publication 1
        --gr00t_temporal_ensemble "${ENSEMBLE:-none}"
        --gr00t_consume_slots "${FORCE_CONSUME_SLOTS:-30}"
        --state_history_steps 9
    )
fi

pixi run -e isaaclab python scripts/viz/render_paper_policy_video.py \
    --task Isaac-Imitation-G1-v2 --algo IPMD \
    --agent_entry_point rlopt_ipmd_tuned_cfg_entry_point \
    "${DRIVE_ARGS[@]}" \
    --style "${STYLE}" --shot "${SHOT}" \
    --video_width "${VIDEO_WIDTH}" --video_height "${VIDEO_HEIGHT}" \
    --stills_every "${STILLS_EVERY}" \
    --output_dir "${OUTPUT_DIR}" --headless \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
    physics=physx env.data.manifest=null env.data.cache_dir=null \
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
    "${ARM_CFG[@]}" \
    "$@"

echo "retained: ${OUTPUT_DIR}"
