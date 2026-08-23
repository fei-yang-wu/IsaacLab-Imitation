#!/usr/bin/env bash
# Isaac/Newton 4,096-motion scoreboard row for the IRL explicit arm.
#
# Protocol copied from `2026-08-08-bones129k-4096-scoreboard/run.sh` (sonic
# pass) so the row is comparable with the recorded explicit baseline: 4,096
# environments, ranks 12288-16383 pinned, frame-0 starts, seed 0, mode
# actions, no_push, Newton/MJWarp, released-SONIC thresholds, foot_pos_xyz
# and base_too_low disabled.
#
# Differences vs the baseline row that the report must state:
#   - this arm trains the IPMD reward estimator, so the checkpoint carries a
#     reward_estimator_state_dict sized for the normalized reward_input
#     group; eval must construct the same estimator
#     (env.enable_reward_input_observations=true, agent.reward_estimation=true)
#     for the strict restore;
#   - network is the scaled [2048,2048,1024,1024,512,512] recipe, while the
#     recorded 08-05 explicit baseline used [1024,1024,512].
#
#   CHECKPOINT=logs/reward_estimation_4096/irl_explicit_root_qpos/model_step_4000186368.pt \
#     ./eval_scoreboard4096.sh
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

CHECKPOINT="${CHECKPOINT:?set CHECKPOINT=<local .pt path>}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/logs/reward_estimation_4096}"
REFERENCE_ARRAYS="${REFERENCE_ARRAYS:-/mnt/hsstorage/fwu91/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
PERSIST_ID="${PERSIST_ID:-bones_seed_sonic_full_129785@e714bbff}"
NUM_ENVS=4096
RANK_START=12288
RANK_END=16383
MAX_STEPS="${MAX_STEPS:-10000}"
SEED=0
SCALED_CELLS="[2048,2048,1024,1024,512,512]"
RUNTIME_BODY_NAMES="[pelvis,left_hip_roll_link,left_knee_link,left_ankle_roll_link,right_hip_roll_link,right_knee_link,right_ankle_roll_link,torso_link,left_shoulder_roll_link,left_elbow_link,left_wrist_yaw_link,right_shoulder_roll_link,right_elbow_link,right_wrist_yaw_link]"

ARM="${ARM:-irl_explicit_root_qpos}"
FRAMES="$(basename "${CHECKPOINT}" | sed -E 's/^model_step_([0-9]+)\.pt$/\1/')"
OUT="${OUTPUT_ROOT}/${ARM}_f${FRAMES}.json"
mkdir -p "${OUTPUT_ROOT}"

ranks=()
for ((r = RANK_START; r <= RANK_END; r++)); do ranks+=("${r}"); done

env TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 \
    HYDRA_FULL_ERROR=1 TORCHDYNAMO_DISABLE=1 \
    pixi run -e isaaclab python -u \
    -m imitation_experiments.lowlevel.evaluate_checkpoint \
    --task Isaac-Imitation-G1-v2 --algo IPMD \
    --checkpoint "${CHECKPOINT}" \
    --agent_entry_point rlopt_ipmd_tuned_cfg_entry_point \
    --randomization no_push --action_sampling mode \
    --num_envs "${NUM_ENVS}" --steps "${MAX_STEPS}" --seed "${SEED}" \
    --reference_start_frame 0 --reset_schedule sequential \
    --trajectory_ranks "${ranks[@]}" \
    --output_json "${OUT}" --label "${ARM}_sonic" --headless \
    --kit_args=--/app/extensions/fsWatcherEnabled=false \
    physics=newton_mjwarp \
    env.sim.physics.solver_cfg.njmax=320 \
    env.sim.physics.solver_cfg.nconmax=200 \
    env.events.push_robot=null \
    env.data.manifest=null \
    "env.data.reference_arrays_dir=${REFERENCE_ARRAYS}" \
    "env.data.persist_id=${PERSIST_ID}" \
    env.data.reference_arrays_resident=false \
    env.data.reference_arrays_warm_workers=8 \
    env.data.runtime_cache_device=cuda:0 \
    env.data.reference_prefetch_mode=off \
    env.data.macro_cache_device=cuda:0 \
    "env.data.runtime_cache_body_names=${RUNTIME_BODY_NAMES}" \
    env.command_interface.actor=explicit \
    "env.command_interface.actor.components=[joint_qpos,root_pos,root_ori]" \
    agent.ipmd.use_latent_command=false \
    agent.command_space=root_qpos \
    "agent.command_components=[joint_qpos,root_pos,root_ori]" \
    agent.ipmd.command_source=random \
    agent.ipmd.hl_skill_checkpoint_path=null \
    env.enable_reward_input_observations=true \
    agent.reward_estimation=true \
    env.expert_macro_state_terms=[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b] \
    env.expert_macro_frame_stride=1 \
    env.terminations.anchor_pos.params.threshold=0.25 \
    env.terminations.anchor_pos.params.down_threshold=0.25 \
    env.terminations.anchor_ori.params.threshold=1.0 \
    env.terminations.ee_body_pos.params.threshold=0.25 \
    env.terminations.ee_body_pos.params.down_threshold=0.25 \
    env.terminations.foot_pos_xyz=null \
    env.terminations.base_too_low=null \
    "agent.policy.num_cells=${SCALED_CELLS}" \
    agent.policy.activation_fn=silu \
    "agent.value_function.num_cells=${SCALED_CELLS}" \
    agent.value_function.activation_fn=silu > "${OUT}.log" 2>&1
rc=$?
echo "exit=${rc} out=${OUT}"
exit "${rc}"
