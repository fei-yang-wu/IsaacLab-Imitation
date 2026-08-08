#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

MODE="${MODE:-print}"
CANDIDATES="${CANDIDATES:-${SCRIPT_DIR}/candidates.json}"
CHECKPOINT="${CHECKPOINT:-logs/rollout24_gamma097_foot_disabled_eval/checkpoints/model_step_3500015616.pt}"
ENCODER="${ENCODER:-logs/rollout24_gamma097_foot_disabled_eval/encoder/latest.pt}"
REFERENCE_ARRAYS="${REFERENCE_ARRAYS:-/mnt/hsstorage/fwu91/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
PERSIST_ID="${PERSIST_ID:-bones_seed_sonic_full_129785@e714bbff}"
ENVS_PER_MOTION="${ENVS_PER_MOTION:-128}"
STEPS="${STEPS:-1100}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/bones_language10_candidate_screen/rollout24_gamma097_3p5b_seed0}"
RAW_JSON="${OUTPUT_ROOT}/candidate_screen_sonic_eval.json"
RANKED_JSON="${OUTPUT_ROOT}/candidate_ranking.json"
RANKED_CSV="${OUTPUT_ROOT}/candidate_ranking.csv"

EXPECTED_CHECKPOINT_SHA256="23fdd62a784fd3c57f30a466e8b5a1fb94d31176a211254c0443e126c8ea283e"
EXPECTED_ENCODER_SHA256="d191d8656620059a569edbad82ca182cb2d2f85839300153cb618d1e29f8c5e7"

for artifact in "${CANDIDATES}" "${CHECKPOINT}" "${ENCODER}" \
    "${REFERENCE_ARRAYS}/reference_arrays_manifest.json"; do
    [[ -f "${artifact}" ]] || { echo "[ERROR] Missing ${artifact}" >&2; exit 2; }
done
[[ "$(sha256sum "${CHECKPOINT}" | awk '{print $1}')" == "${EXPECTED_CHECKPOINT_SHA256}" ]] \
    || { echo "[ERROR] Checkpoint hash mismatch." >&2; exit 2; }
[[ "$(sha256sum "${ENCODER}" | awk '{print $1}')" == "${EXPECTED_ENCODER_SHA256}" ]] \
    || { echo "[ERROR] Encoder hash mismatch." >&2; exit 2; }

mapfile -t TRAJECTORY_RANKS < <(
    pixi run python -c \
        'import json,sys; print(*[x["trajectory_rank"] for x in json.load(open(sys.argv[1]))["candidates"]], sep="\n")' \
        "${CANDIDATES}"
)
NUM_ENVS=$(( ${#TRAJECTORY_RANKS[@]} * ENVS_PER_MOTION ))

eval_cmd=(
    pixi run -e isaaclab python -u
    -m imitation_experiments.lowlevel.evaluate_checkpoint
    --task Isaac-Imitation-G1-v2 --algo IPMD
    --checkpoint "${CHECKPOINT}"
    --agent_entry_point rlopt_ipmd_tuned_cfg_entry_point
    --randomization no_push --action_sampling mode
    --num_envs "${NUM_ENVS}" --steps "${STEPS}" --seed 0
    --reference_start_frame 0 --reset_schedule custom
    --trajectory_ranks "${TRAJECTORY_RANKS[@]}"
    --output_json "${RAW_JSON}"
    physics=newton_mjwarp
    env.events.push_robot=null
    env.data.manifest=null
    "env.data.reference_arrays_dir=${REFERENCE_ARRAYS}"
    "env.data.persist_id=${PERSIST_ID}"
    env.data.reference_arrays_warm_workers=8
    env.data.reference_prefetch_mode=next
    env.data.macro_cache_device=cuda:0
    env.data.runtime_cache_body_names=[pelvis,left_hip_roll_link,left_knee_link,left_ankle_roll_link,right_hip_roll_link,right_knee_link,right_ankle_roll_link,torso_link,left_shoulder_roll_link,left_elbow_link,left_wrist_yaw_link,right_shoulder_roll_link,right_elbow_link,right_wrist_yaw_link]
    env.command_interface.actor.dim=258
    env.expert_macro_state_terms=[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b]
    env.terminations.anchor_pos.params.threshold=0.25
    env.terminations.anchor_pos.params.down_threshold=0.25
    env.terminations.anchor_ori.params.threshold=1.0
    env.terminations.ee_body_pos.params.threshold=0.25
    env.terminations.ee_body_pos.params.down_threshold=0.25
    env.terminations.foot_pos_xyz=null
    env.terminations.base_too_low=null
    agent.ipmd.latent_dim=258
    agent.ipmd.command_source=hl_skill
    "agent.ipmd.hl_skill_checkpoint_path=${ENCODER}"
    agent.ipmd.hl_skill_horizon_steps=10
    agent.ipmd.hl_skill_command_mode=z
    agent.ipmd.latent_steps_min=10
    agent.ipmd.latent_steps_max=10
    agent.ipmd.latent_learning.code_period=10
    agent.ipmd.latent_learning.command_phase_mode=sin_cos
    agent.ipmd.latent_learning.code_latent_dim=256
    agent.ipmd.hl_skill_finetune_enabled=false
)

aggregate_cmd=(
    pixi run python -m imitation_experiments.lowlevel.motion_candidate_screen
    --evaluation_json "${RAW_JSON}"
    --candidates_json "${CANDIDATES}"
    --output_json "${RANKED_JSON}"
    --output_csv "${RANKED_CSV}"
)

echo "[INFO] candidates=${#TRAJECTORY_RANKS[@]} envs_per_motion=${ENVS_PER_MOTION} total_envs=${NUM_ENVS}"
echo "[INFO] deterministic policy; SONIC terminations; startup/reset randomization; no push"
case "${MODE}" in
    print)
        printf '[EVAL]'; printf ' %q' "${eval_cmd[@]}"; printf '\n'
        printf '[AGG ]'; printf ' %q' "${aggregate_cmd[@]}"; printf '\n'
        ;;
    screen)
        mkdir -p "${OUTPUT_ROOT}"
        "${eval_cmd[@]}" 2>&1 | tee "${OUTPUT_ROOT}/candidate_screen.log"
        "${aggregate_cmd[@]}"
        ;;
    aggregate)
        "${aggregate_cmd[@]}"
        ;;
    *)
        echo "[ERROR] MODE must be print, screen, or aggregate." >&2
        exit 2
        ;;
esac
