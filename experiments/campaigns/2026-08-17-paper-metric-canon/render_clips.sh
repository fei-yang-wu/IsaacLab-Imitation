#!/usr/bin/env bash
set -uo pipefail

# Render one clip per run at one environment, so the camera frames the robot.
# Both sides run the SAME pinned rank under the same clean protocol, so the two
# videos are directly comparable.
#
# Two clip sets, selected from the testbed results:
#
#   SET=failures (default) the fastest failures in the four largest families
#                 where the released checkpoint completes and our arm does not
#                 (kneeling, dance, high jump, reach jump). Each ends on
#                 `ee_body_pos` or `anchor_ori` within a second, so the
#                 protocol clip IS the failure moment.
#   SET=successes the median-difficulty clip of each of the ten largest motion
#                 families that BOTH sides complete. Typical behavior, not
#                 cherry-picked: the representative is the family's median by
#                 minimum reference pelvis height.
#
# `VIDEO_LENGTH` must cover the clip or the render stops early; the success set
# is filtered to 400 frames or fewer and uses 450.
#
#   ./render_clips.sh
#   SET=successes ./render_clips.sh
#   RANKS="15135" SIDES="ours" ./render_clips.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

MIRROR="${MIRROR:-${REPO_ROOT}/logs/bottleneck_10b_mirror}"
ARM="${ARM:-cont_det_ln_hold1}"
FRAMES="${FRAMES:-10000269312}"
SONIC_CHECKPOINT="${SONIC_CHECKPOINT:-/mnt/hsstorage/fwu91/sonic_release/last.pt}"
REFERENCE_ARRAYS="${REFERENCE_ARRAYS:-/mnt/hsstorage/fwu91/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
PERSIST_ID="${PERSIST_ID:-bones_seed_sonic_full_129785@e714bbff}"

SET="${SET:-failures}"
# Captured before the per-set defaults so an explicit override still wins.
OUTPUT_ROOT_OVERRIDE="${OUTPUT_ROOT:-}"
VIDEO_LENGTH_OVERRIDE="${VIDEO_LENGTH:-}"

# rank | short name
FAILURE_CLIPS=(
"15135|kneeling_stop"
"65247|dance_take_the_l"
"31600|high_jump"
"49681|reach_jump"
)
SUCCESS_CLIPS=(
"26867|jog_arc_cw_stop"
"10543|walk_ff_stop_315"
"35154|jump_ff_180"
"16372|injured_leg_jog_loop"
"103805|dance_hiphop_tls_step"
"59632|turn_run_270"
"22323|medium_big_light_two_hands_walk"
"56107|crouch_ff_loop_270"
"75358|small_heavy_one_hand_put_down"
"53256|big_light_two_hands_walk"
)

if [[ "${SET}" == "successes" ]]; then
    CLIPS=("${SUCCESS_CLIPS[@]}")
    DEFAULT_RANKS="26867 10543 35154 16372 103805 59632 22323 56107 75358 53256"
    DEFAULT_SIDES="ours sonic"
    VIDEO_LENGTH="${VIDEO_LENGTH_OVERRIDE:-450}"
    OUTPUT_ROOT="${OUTPUT_ROOT_OVERRIDE:-${REPO_ROOT}/logs/testbed4096/success_videos}"
else
    CLIPS=("${FAILURE_CLIPS[@]}")
    DEFAULT_RANKS="15135 65247 31600 49681"
    # ours      protocol run: ends at the failure, so the clip IS that moment
    # ours_full early terminations off: what the policy does past that point
    # sonic     the released checkpoint on the identical rank, protocol run
    DEFAULT_SIDES="ours ours_full sonic"
    VIDEO_LENGTH="${VIDEO_LENGTH_OVERRIDE:-260}"
    OUTPUT_ROOT="${OUTPUT_ROOT_OVERRIDE:-${REPO_ROOT}/logs/testbed4096/failure_videos}"
fi
RANKS="${RANKS:-${DEFAULT_RANKS}}"
SIDES="${SIDES:-${DEFAULT_SIDES}}"

SCALED_CELLS="[2048,2048,1024,1024,512,512]"
RUNTIME_BODY_NAMES="[pelvis,left_hip_roll_link,left_knee_link,left_ankle_roll_link,right_hip_roll_link,right_knee_link,right_ankle_roll_link,torso_link,left_shoulder_roll_link,left_elbow_link,left_wrist_yaw_link,right_shoulder_roll_link,right_elbow_link,right_wrist_yaw_link]"
SONIC_TERMS=(
    env.terminations.anchor_pos.params.threshold=0.25
    env.terminations.anchor_pos.params.down_threshold=0.25
    env.terminations.anchor_ori.params.threshold=1.0
    env.terminations.ee_body_pos.params.threshold=0.25
    env.terminations.ee_body_pos.params.down_threshold=0.25
    env.terminations.foot_pos_xyz=null
    env.terminations.base_too_low=null
)
DATA=(
    env.data.manifest=null
    "env.data.reference_arrays_dir=${REFERENCE_ARRAYS}"
    "env.data.persist_id=${PERSIST_ID}"
    env.data.reference_arrays_resident=false
    env.data.reference_arrays_warm_workers=4
    env.data.runtime_cache_device=cuda:0
    env.data.reference_prefetch_mode=off
    env.data.macro_cache_device=cuda:0
    "env.data.runtime_cache_body_names=${RUNTIME_BODY_NAMES}"
)

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

name_for() {
    local rank="$1" entry
    for entry in "${CLIPS[@]}"; do
        [[ "${entry%%|*}" == "${rank}" ]] && { echo "${entry#*|}"; return; }
    done
    echo "rank${rank}"
}

mkdir -p "${OUTPUT_ROOT}"
checkpoint="${MIRROR}/${ARM}_seed0/tracker/f${FRAMES}/models/model_step_${FRAMES}.pt"
encoder="${MIRROR}/${ARM}_seed0/encoder/checkpoints/latest.pt"

for rank in ${RANKS}; do
    clip="$(name_for "${rank}")"
    for side in ${SIDES}; do
        label="${clip}_r${rank}_${side}"
        out="${OUTPUT_ROOT}/${label}.json"
        [[ -s "${out}" ]] && { log "[SKIP] ${label}"; continue; }
        log "render ${label}"

        if [[ "${side}" == "ours" || "${side}" == "ours_full" ]]; then
            [[ -s "${checkpoint}" ]] || { log "[SKIP] no checkpoint ${checkpoint}"; continue; }
            horizon=()
            [[ "${side}" == "ours_full" ]] && horizon=(--disable_early_terminations)
            env TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 \
                HYDRA_FULL_ERROR=1 TORCHDYNAMO_DISABLE=1 \
                pixi run -e isaaclab python -u \
                -m imitation_experiments.lowlevel.evaluate_checkpoint \
                --task Isaac-Imitation-G1-v2 --algo IPMD \
                --checkpoint "${checkpoint}" \
                --agent_entry_point rlopt_ipmd_tuned_cfg_entry_point \
                --randomization none --action_sampling mode \
                --num_envs 1 --steps "${VIDEO_LENGTH}" --seed 0 \
                --reference_start_frame 0 --reset_schedule sequential \
                --trajectory_ranks "${rank}" \
                "${horizon[@]}" \
                --video --video_dir "${OUTPUT_ROOT}" --video_length "${VIDEO_LENGTH}" \
                --output_json "${out}" --label "${label}" \
                --kit_args=--/app/extensions/fsWatcherEnabled=false \
                physics=newton_mjwarp \
                env.sim.physics.solver_cfg.njmax=320 \
                env.sim.physics.solver_cfg.nconmax=200 \
                env.events.push_robot=null \
                "${DATA[@]}" \
                env.command_interface.actor=latent \
                env.command_interface.actor.dim=258 \
                env.command_interface.encoder=single \
                agent.ipmd.latent_dim=258 \
                agent.ipmd.command_source=hl_skill \
                "agent.ipmd.hl_skill_checkpoint_path=${encoder}" \
                agent.ipmd.hl_skill_horizon_steps=10 \
                agent.ipmd.hl_skill_command_mode=z \
                agent.ipmd.latent_steps_min=1 \
                agent.ipmd.latent_steps_max=1 \
                agent.ipmd.latent_learning.code_period=1 \
                agent.ipmd.latent_learning.command_phase_mode=sin_cos \
                agent.ipmd.latent_learning.code_latent_dim=256 \
                agent.ipmd.hl_skill_finetune_enabled=false \
                env.expert_macro_state_terms=[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b] \
                env.expert_macro_frame_stride=1 \
                env.expert_macro_anchor_mode=robot_heading \
                "${SONIC_TERMS[@]}" \
                "agent.policy.num_cells=${SCALED_CELLS}" \
                agent.policy.activation_fn=silu \
                "agent.value_function.num_cells=${SCALED_CELLS}" \
                agent.value_function.activation_fn=silu > "${out}.log" 2>&1
        else
            env TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 \
                HYDRA_FULL_ERROR=1 TORCHDYNAMO_DISABLE=1 \
                pixi run -e isaaclab python -u \
                -m imitation_experiments.lowlevel.evaluate_sonic_release \
                --sonic_checkpoint "${SONIC_CHECKPOINT}" --sonic_version release \
                --num_envs 1 --steps "${VIDEO_LENGTH}" --seed 0 \
                --randomization none --reference_start_frame 0 \
                --reset_schedule sequential --trajectory_ranks "${rank}" \
                --termination_contract sonic \
                --proprioception_order gravity_last --history_order oldest_first \
                --allow_incomplete_release \
                --video --video_dir "${OUTPUT_ROOT}" --video_length "${VIDEO_LENGTH}" \
                --output_json "${out}" --label "${label}" \
                --kit_args=--/app/extensions/fsWatcherEnabled=false \
                physics=newton_mjwarp \
                env.sim.physics.solver_cfg.njmax=320 \
                env.sim.physics.solver_cfg.nconmax=200 \
                env.events.push_robot=null \
                "${DATA[@]}" > "${out}.log" 2>&1
        fi
        rc=$?
        (( rc == 0 )) || log "[FAIL] ${label} exit ${rc}: $(tail -3 "${out}.log" | tr '\n' ' ')"
    done
done

log "videos in ${OUTPUT_ROOT}"
ls -1 "${OUTPUT_ROOT}"/*.mp4 2>/dev/null || log "[WARN] no mp4 written"
