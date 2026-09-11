#!/usr/bin/env bash
# Inference-time action low-pass sweep on the finished combo-50b checkpoint.
#
# The question: how much of combo-50b's roughness is in the OPEN-LOOP action
# sequence, and can a filter buy it back without retraining? SONIC v1.1 on this
# board scores jerk 115.9 / action_delta 0.555 / acc 3.445 against combo-50b's
# 192.9 / 0.855 / 4.418, so the target is a 40% jerk cut. We have 6.2 mm of
# MPJPE-L headroom over SONIC to spend.
#
# `a_t <- (1 - alpha) * a_{t-1} + alpha * pi(s_t)`; alpha 1.0 is the unfiltered
# control. The filter runs before the action metrics and before env.step, and
# re-seeds per environment on reset.
#
#   ./sweep.sh                    # alphas below
#   ALPHAS="0.5" ./sweep.sh       # one point
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

MIRROR="${MIRROR:-${REPO_ROOT}/logs/latent64_probe_mirror}"
CHECKPOINT="${CHECKPOINT:-${MIRROR}/ckpt/combo50b_latest_f50000166912.pt}"
ENCODER="${ENCODER:-${MIRROR}/p5_affine_encoder/latest.pt}"
REFERENCE_ARRAYS="${REFERENCE_ARRAYS:-/mnt/hsstorage/fwu91/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
PERSIST_ID="${PERSIST_ID:-bones_seed_sonic_full_129785@e714bbff}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/logs/action_lowpass_sweep}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_STEPS="${MAX_STEPS:-10000}"
ALPHAS="${ALPHAS:-1.0 0.7 0.5 0.3}"
RUNTIME_BODY_NAMES="[pelvis,left_hip_roll_link,left_knee_link,left_ankle_roll_link,right_hip_roll_link,right_knee_link,right_ankle_roll_link,torso_link,left_shoulder_roll_link,left_elbow_link,left_wrist_yaw_link,right_shoulder_roll_link,right_elbow_link,right_wrist_yaw_link]"

log() { printf '[%s] %s\n' "$(date -Iseconds)" "$*"; }
for required in "${CHECKPOINT}" "${ENCODER}" "${REFERENCE_ARRAYS}"; do
    [[ -e "${required}" ]] || { log "[FATAL] missing ${required}"; exit 1; }
done
mkdir -p "${OUTPUT_ROOT}"

mapfile -t ranks < <(pixi run python -c \
    'from imitation_experiments.evaluation.protocol import TESTBED4096_RANKS
print("\n".join(str(rank) for rank in TESTBED4096_RANKS))')
[[ "${#ranks[@]}" -eq 4096 ]] || { log "[FATAL] registry returned ${#ranks[@]} ranks"; exit 2; }
# `build_env_rank_assignment` requires num_envs to be a positive multiple of the
# rank count, so a smaller smoke run takes a prefix of the board rather than the
# whole thing. Full runs leave NUM_ENVS at 4096 and use every rank.
if [[ "${NUM_ENVS}" -lt "${#ranks[@]}" ]]; then
    ranks=("${ranks[@]:0:${NUM_ENVS}}")
    log "[SMOKE] using the first ${NUM_ENVS} ranks; this is NOT the board"
fi

for alpha in ${ALPHAS}; do
    out="${OUTPUT_ROOT}/combo50b_f50000166912_alpha${alpha}.json"
    [[ -s "${out}" ]] && { log "[SKIP] already scored ${out}"; continue; }
    log "alpha ${alpha}"
    env TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 \
        HYDRA_FULL_ERROR=1 TORCHDYNAMO_DISABLE=1 \
        pixi run -e isaaclab python -u \
        -m imitation_experiments.lowlevel.evaluate_checkpoint \
        --task Isaac-Imitation-G1-v2 --algo IPMD \
        --agent_entry_point rlopt_ipmd_tuned_cfg_entry_point \
        --checkpoint "${CHECKPOINT}" \
        --num_envs "${NUM_ENVS}" --steps "${MAX_STEPS}" --seed 0 \
        --randomization none --action_sampling mode \
        --reference_start_frame 0 --reset_schedule sequential \
        --trajectory_ranks "${ranks[@]}" \
        --skill_encoder_source pretrained \
        --action_lowpass_alpha "${alpha}" \
        --label "combo50b_alpha${alpha}" --output_json "${out}" \
        --headless \
        --kit_args=--/app/extensions/fsWatcherEnabled=false \
        physics=newton_mjwarp \
        env.sim.physics.solver_cfg.njmax=320 \
        env.sim.physics.solver_cfg.nconmax=200 \
        env.events.push_robot=null \
        env.data.manifest=null \
        "env.data.reference_arrays_dir=${REFERENCE_ARRAYS}" \
        "env.data.persist_id=${PERSIST_ID}" \
        env.data.reference_arrays_resident=false \
        env.data.runtime_cache_device=cpu \
        env.data.reference_prefetch_mode=off \
        env.data.macro_cache_device=cuda:0 \
        "env.data.runtime_cache_body_names=${RUNTIME_BODY_NAMES}" \
        env.command_interface.actor=latent \
        env.command_interface.actor.dim=66 \
        env.command_interface.encoder=single \
        'env.expert_macro_state_terms=[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b]' \
        env.expert_macro_frame_stride=1 \
        env.expert_macro_anchor_mode=robot_heading \
        agent.logger.backend= \
        agent.ipmd.command_source=hl_skill \
        "agent.ipmd.hl_skill_checkpoint_path=${ENCODER}" \
        agent.ipmd.hl_skill_finetune_enabled=false \
        agent.ipmd.latent_dim=66 \
        agent.ipmd.latent_steps_min=1 agent.ipmd.latent_steps_max=1 \
        agent.ipmd.hl_skill_horizon_steps=10 \
        agent.ipmd.hl_skill_command_mode=z \
        agent.ipmd.latent_learning.command_phase_mode=sin_cos \
        agent.ipmd.latent_learning.code_latent_dim=64 \
        agent.ipmd.latent_learning.code_period=1 \
        env.observations.policy.projected_gravity.history_length=10 \
        env.observations.policy.base_ang_vel.history_length=10 \
        env.observations.policy.joint_pos_rel.history_length=10 \
        env.observations.policy.joint_vel_rel.history_length=10 \
        env.observations.policy.last_action.history_length=10 \
        env.terminations.anchor_pos.params.threshold=0.25 \
        env.terminations.anchor_pos.params.down_threshold=0.25 \
        env.terminations.anchor_ori.params.threshold=1.0 \
        env.terminations.ee_body_pos.params.threshold=0.25 \
        env.terminations.ee_body_pos.params.down_threshold=0.25 \
        env.terminations.foot_pos_xyz=null \
        env.terminations.base_too_low=null \
        'agent.policy.num_cells=[2048,2048,1024,1024,512,512]' \
        'agent.value_function.num_cells=[2048,2048,1024,1024,512,512]' \
        agent.policy.activation_fn=silu \
        agent.value_function.activation_fn=silu > "${out}.log" 2>&1 || {
        log "[FAIL] alpha ${alpha}; see ${out}.log"; continue; }
    # The Isaac entrypoint returns 0 even when the Python body raises, so the
    # written row is the only trustworthy success signal.
    [[ -s "${out}" ]] || {
        log "[FAIL] alpha ${alpha}: no row written; tail of ${out}.log:"
        tail -5 "${out}.log" || true
        continue
    }
    log "[OK] ${out}"
done
