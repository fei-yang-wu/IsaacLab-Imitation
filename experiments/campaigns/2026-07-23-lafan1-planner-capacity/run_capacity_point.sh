#!/usr/bin/env bash
set -euo pipefail

# One matched planner-capacity point for the LAFAN1 one-motion study: for a fixed
# MODEL_SIZE and PLANNER_SEED, train + evaluate a flow-matching planner on all
# three interfaces (latent_skill, full_body_trajectory, ee_trajectory), both
# demonstration-only and rollout-finetuned. Reuses the shared oracle baselines
# from prepare_oracle_baselines.sh. Resumable: each step is skipped if its
# artifact already exists.
#
# Usage:
#   DRY_RUN=1 MODEL_SIZE=small PLANNER_SEED=1 run_capacity_point.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || (cd "${SCRIPT_DIR}/../../.." && pwd))"
cd "${REPO_ROOT}"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/paths.env"

MODEL_SIZE="${MODEL_SIZE:-small}"
PLANNER_SEED="${PLANNER_SEED:-1}"
EVAL_SEED="${EVAL_SEED:-0}"
DEVICE="${DEVICE:-cuda:0}"
DRY_RUN="${DRY_RUN:-0}"
EVAL_STEPS="${EVAL_STEPS:-700}"
# Demonstration budget, in ROWS. This must match what prepare_oracle_baselines.sh
# actually collected (DEMO_TRAJECTORIES=100 -> 5000 rows), because it is passed
# as the trainer's --max_samples cap: leaving it at the old 1000 would silently
# subsample away 80% of the demonstration set and quietly reinstate the thin-data
# regime the 100-trajectory re-collection exists to leave. Quote the budget as
# 100 trajectories; 5000 rows is the derived number (one row per env per publish).
DEMO_ROWS="${DEMO_ROWS:-5000}"
# Unchanged on purpose: the optimizer budget must stay identical across
# interfaces and sizes, so more data means fewer epochs, not more updates.
NUM_UPDATES="${NUM_UPDATES:-2000}"
# Rollout-collection horizon for the finetune stage. The balanced collector stops
# as soon as it has the exact per-motion row budget, so this is slack, not extra
# data: 10 envs x COLLECT_STEPS / 10-step publication interval must EXCEED
# DEMO_ROWS with margin, because any env that reaches the end of the motion
# (wrap_steps=false) leaves the budget unreachable. AGENTS.md allows the outer
# collector to continue across resets until the exact row count is met.
COLLECT_STEPS="${COLLECT_STEPS:-15000}"
# DEMO_ONLY=1 stops after the demonstration-only planner and its evaluation,
# skipping rollout collection, merge, finetune and the finetuned evaluation.
# The demo-only row is the paper-leading comparison (identical footing for both
# interfaces); the finetune stage is protocol-sensitive -- one DAgger round is
# capacity-dependent and oracle-driven aggregation is a null control -- so it is
# not worth its compute on a second motion.
DEMO_ONLY="${DEMO_ONLY:-0}"
STUDY_ROOT="${STUDY_ROOT:-logs/interface_baselines/lafan1_planner_capacity_20260723}"
ORACLE_ROOT="${ORACLE_ROOT:-${STUDY_ROOT}/oracle_baselines}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${STUDY_ROOT}/scaling/seed${PLANNER_SEED}}"
POINT_ROOT="${OUTPUT_ROOT}/${MODEL_SIZE}"

case "${MODEL_SIZE}" in tiny|small|medium|large) ;; *)
    echo "[ERROR] MODEL_SIZE must be tiny|small|medium|large." >&2; exit 2 ;; esac

# Runtime abstraction: local uses pixi; ICE container sets ISAAC_PY/PLAIN_PY to
# /isaac-sim/python.sh. NEWTON_ARGS match the Newton-trained frozen oracles.
: "${ISAAC_PY:=pixi run -e isaaclab python}"
: "${PLAIN_PY:=pixi run python}"
read -r -a ISAAC_PY_ARR <<<"${ISAAC_PY}"
read -r -a PLAIN_PY_ARR <<<"${PLAIN_PY}"
NEWTON_ARGS=(physics=newton_mjwarp
    "env.sim.physics.solver_cfg.njmax=${NJMAX:-320}"
    "env.sim.physics.solver_cfg.nconmax=${NCONMAX:-40}")
LATENT_KITLESS=()
[[ "${ASSERT_KITLESS:-0}" == "1" ]] && LATENT_KITLESS=(--assert-kitless)

# Render a capped video on every eval pass (user request). Full-horizon rendering
# is slow, so cap the clip length; the metrics still run the full horizon. Cameras
# need Kit, so video is disabled under kitless (ICE compute-only GPUs).
VIDEO_ARGS=()
if [[ "${RENDER_VIDEO:-1}" == "1" && "${ASSERT_KITLESS:-0}" != "1" ]]; then
    VIDEO_ARGS=(--video --video_length "${VIDEO_STEPS:-150}")
fi

# Full-horizon eval protocol (user, 2026-07-24): start at reference frame 0, run
# one uninterrupted EVAL_STEPS rollout, ALL terminations disabled, domain
# randomization disabled. With DR off + deterministic policy the envs are
# identical, so few envs suffice.
#
# BOTH surfaces are SONIC-family and share the same term names: the latent task
# Isaac-Imitation-G1-Latent-v0 resolves to ImitationG1LatentStrictEnvCfg, which
# uses G1SonicTerminationsCfg + G1SonicEventCfg -- exactly what the chunk task
# uses. An earlier revision wrongly assumed the latent env used
# G1TerminationsCfg and applied a *different* override set per interface, which
# left foot_pos_xyz ACTIVE on the latent rows only. That terminated every latent
# episode at ~212 steps while the chunk rows ran the full 700, so MPJPE was
# averaged over a 3x shorter, less-drifted window on the latent side and the
# comparison was silently biased toward latent. Keep ONE symmetric override set
# for both interfaces; base_too_low is already None in G1SonicTerminationsCfg.
FH_ENVS="${FH_ENVS:-4}"
_FH_COMMON=(
    env.random_reset_step_min=0 env.random_reset_step_max=0
    env.random_reset_full_trajectory=false env.reset_schedule=sequential
    env.reference_start_frame=0 env.wrap_steps=false env.episode_length_s=20.0
    env.terminations.anchor_pos=null env.terminations.anchor_ori=null
    env.terminations.ee_body_pos=null env.terminations.foot_pos_xyz=null
    env.events.physics_material=null env.events.add_joint_default_pos=null
    env.events.base_com=null env.events.push_robot=null
    env.events.randomize_rigid_body_mass=null
)
LATENT_FH=("${_FH_COMMON[@]}")
CHUNK_FH=("${_FH_COMMON[@]}")

KIT_QUIET=(--kit_args=--/app/extensions/fsWatcherEnabled=false)
LATENT_REWARD_ZEROS=(
    agent.ipmd.reward_loss_coeff=0.0 agent.ipmd.reward_l2_coeff=0.0
    agent.ipmd.reward_grad_penalty_coeff=0.0 agent.ipmd.reward_logit_reg_coeff=0.0
    agent.ipmd.reward_param_weight_decay_coeff=0.0
)
LATENT_CMD=(
    env.latent_command_dim=258 agent.ipmd.latent_dim=258
    agent.ipmd.hl_skill_horizon_steps=10 agent.ipmd.hl_skill_command_mode=z
    agent.ipmd.latent_steps_min=10 agent.ipmd.latent_steps_max=10
    agent.ipmd.latent_learning.command_phase_mode=sin_cos
    agent.ipmd.latent_learning.code_latent_dim=256
    agent.ipmd.latent_learning.code_period=10
)

run_if_missing() {
    local marker="$1"; shift
    if [[ -e "${marker}" ]]; then echo "[SKIP] ${marker}"; return 0; fi
    printf '[CMD]'; printf ' %q' "$@"; printf '\n'
    [[ "${DRY_RUN}" == "1" ]] && return 0
    TERM=xterm PYTHONUNBUFFERED=1 "$@"
    [[ -e "${marker}" ]] || { echo "[ERROR] missing artifact: ${marker}" >&2; exit 2; }
}

# ------------------------------------------------------------ planner train --
train_planner() {  # interface samples_dir output [checkpoint]
    local interface="$1" samples="$2" output="$3" checkpoint="${4:-}"
    local cmd=(
        "${PLAIN_PY_ARR[@]}" -m imitation_experiments.planner.train_chunked_transformer_planner
        --samples_dir "${samples}" --output_dir "${output}"
        --interface "${interface}" --planner_family flow --state_key planner_state
        --device "${DEVICE}" --seed "${PLANNER_SEED}"
        --batch_size 256 --micro_batch_size 32 --num_updates "${NUM_UPDATES}"
        --log_interval 100 --eval_batch_size 512 --eval_max_samples 4096
        --lr 0.0001 --weight_decay 0.0001 --model_size "${MODEL_SIZE}"
        --flow_num_inference_steps 16 --endpoint_num_inference_steps 4
        --flow_inference_noise_std 0.0
    )
    if [[ -n "${checkpoint}" ]]; then cmd+=(--checkpoint "${checkpoint}")
    else cmd+=(--max_samples "${DEMO_ROWS}"); fi
    run_if_missing "${output}/checkpoints/latest.pt" "${cmd[@]}"
}

# --------------------------------------------------------------- latent path -
latent_eval() {  # planner output label
    local planner="$1" output="$2" label="$3"
    run_if_missing "${output}/summary.json" \
        "${ISAAC_PY_ARR[@]}" scripts/rlopt/eval_skill_commander_closed_loop.py "${LATENT_KITLESS[@]}" \
        --headless --device "${DEVICE}" --task "${LATENT_TASK}" --algorithm IPMD \
        --checkpoint "${LATENT_LOW_LEVEL_CHECKPOINT}" \
        --skill_checkpoint "${LATENT_SKILL_CHECKPOINT}" --planner_checkpoint "${planner}" \
        --state_history_steps 9 --output_dir "${output}" --label "${label}" \
        --num_envs "${FH_ENVS}" --max_steps "${EVAL_STEPS}" \
        "${VIDEO_ARGS[@]}" \
        --seed "${EVAL_SEED}" --metric_interval 10 \
        --keep_time_out --keep_early_terminations \
        --motion_name "${MOTION_NAME}" --flow_num_inference_steps 16 \
        --flow_inference_noise_std 0.0 "${KIT_QUIET[@]}" \
        agent.logger.backend= agent.ipmd.command_source=skill_commander \
        "agent.ipmd.skill_commander_checkpoint_path=${planner}" \
        agent.ipmd.skill_commander_use_achieved_state=true \
        agent.ipmd.skill_commander_flow_num_inference_steps=16 \
        agent.ipmd.skill_commander_flow_inference_noise_std=0.0 \
        "agent.ipmd.hl_skill_checkpoint_path=${LATENT_SKILL_CHECKPOINT}" \
        agent.ipmd.hl_skill_finetune_enabled=false \
        "env.lafan1_manifest_path=${MANIFEST}" "env.dataset_path=${LATENT_DATASET_PATH}" \
        env.refresh_zarr_dataset=false \
        env.observations.policy.enable_corruption=false \
        "${LATENT_CMD[@]}" "${LATENT_REWARD_ZEROS[@]}" "${NEWTON_ARGS[@]}" "${LATENT_FH[@]}"
}

latent_collect() {  # pretrained_planner output
    local planner="$1" output="$2"
    run_if_missing "${output}/rollout_training_samples/sample_step_000000.pt" \
        "${ISAAC_PY_ARR[@]}" scripts/rlopt/eval_skill_commander_closed_loop.py "${LATENT_KITLESS[@]}" \
        --headless --device "${DEVICE}" --task "${LATENT_TASK}" --algorithm IPMD \
        --checkpoint "${LATENT_LOW_LEVEL_CHECKPOINT}" \
        --skill_checkpoint "${LATENT_SKILL_CHECKPOINT}" --planner_checkpoint "${planner}" \
        --state_history_steps 9 --output_dir "${output}" \
        --label "capacity_${MODEL_SIZE}_seed${PLANNER_SEED}_latent_rollout" \
        --num_envs 10 --max_steps 1000 --seed "${EVAL_SEED}" --metric_interval 10 \
        --keep_time_out --keep_early_terminations --disable_tracking_terminations \
        --motion_name "${MOTION_NAME}" --balanced_motion_names "${MOTION_NAME}" \
        --balanced_rows_per_motion "${DEMO_ROWS}" --save_rollout_training_samples \
        --continue_after_reset --sample_rows_per_file "${DEMO_ROWS}" \
        --flow_num_inference_steps 16 --flow_inference_noise_std 0.0 "${KIT_QUIET[@]}" \
        agent.logger.backend= agent.ipmd.command_source=skill_commander \
        "agent.ipmd.skill_commander_checkpoint_path=${planner}" \
        agent.ipmd.skill_commander_use_achieved_state=true \
        agent.ipmd.skill_commander_flow_num_inference_steps=16 \
        agent.ipmd.skill_commander_flow_inference_noise_std=0.0 \
        "agent.ipmd.hl_skill_checkpoint_path=${LATENT_SKILL_CHECKPOINT}" \
        agent.ipmd.hl_skill_finetune_enabled=false \
        "env.lafan1_manifest_path=${MANIFEST}" "env.dataset_path=${LATENT_DATASET_PATH}" \
        env.refresh_zarr_dataset=false env.random_reset_step_min=0 \
        env.random_reset_step_max=200 env.random_reset_full_trajectory=false \
        env.reset_schedule=sequential env.wrap_steps=false \
        env.observations.policy.enable_corruption=false \
        "${LATENT_CMD[@]}" "${LATENT_REWARD_ZEROS[@]}" "${NEWTON_ARGS[@]}"
}

# ------------------------------------------------------------- chunk path ----
chunk_eval() {  # interface checkpoint planner output label
    local interface="$1" checkpoint="$2" planner="$3" output="$4" label="$5"
    run_if_missing "${output}/summary.json" \
        "${ISAAC_PY_ARR[@]}" -m imitation_experiments.evaluation.eval_interface_planner_closed_loop \
        --headless --device "${DEVICE}" --task "${CHUNK_TASK}" --algorithm IPMD \
        --checkpoint "${checkpoint}" --low_level_command_mode streamed_vanilla \
        --planner_checkpoint "${planner}" --output_json "${output}/summary.json" \
        --pin_command_joint_order "${PIN_COMMAND_JOINT_ORDER:-auto}" \
        --label "${label}" --motion_manifest "${MANIFEST}" --motion_name "${MOTION_NAME}" \
        --num_envs "${FH_ENVS}" --steps "${EVAL_STEPS}" --seed "${EVAL_SEED}" \
        --state_history_steps 9 --command_past_steps 0 --command_future_steps 9 \
        --planner_update_interval 10 --flow_num_inference_steps 16 \
        --flow_inference_noise_std 0.0 --reset_schedule sequential \
        --reference_start_frame 0 --keep_after_done \
        "${KIT_QUIET[@]}" "${VIDEO_ARGS[@]}" \
        agent.logger.backend= env.observations.policy.enable_corruption=false \
        "${NEWTON_ARGS[@]}" "${CHUNK_FH[@]}"
}

chunk_collect() {  # interface checkpoint pretrained_planner output
    local interface="$1" checkpoint="$2" planner="$3" output="$4"
    run_if_missing "${output}/rollout_training_samples/sample_step_000000.pt" \
        "${ISAAC_PY_ARR[@]}" -m imitation_experiments.evaluation.eval_interface_planner_closed_loop \
        --headless --device "${DEVICE}" --task "${CHUNK_TASK}" --algorithm IPMD \
        --checkpoint "${checkpoint}" --low_level_command_mode streamed_vanilla \
        --planner_checkpoint "${planner}" --output_json "${output}/summary.json" \
        --pin_command_joint_order "${PIN_COMMAND_JOINT_ORDER:-auto}" \
        --label "capacity_${MODEL_SIZE}_seed${PLANNER_SEED}_${interface}_rollout" \
        --motion_manifest "${MANIFEST}" --motion_name "${MOTION_NAME}" \
        --num_envs 10 --steps "${COLLECT_STEPS}" --seed "${EVAL_SEED}" --state_history_steps 9 \
        --command_past_steps 0 --command_future_steps 9 --planner_update_interval 10 \
        --flow_num_inference_steps 16 --flow_inference_noise_std 0.0 \
        --reset_schedule sequential --reference_start_frame 0 \
        --keep_configured_episode_length --disable_tracking_terminations --keep_after_done \
        --save_rollout_training_samples \
        --samples_output_dir "${output}/rollout_training_samples" \
        --sample_rows_per_file "${DEMO_ROWS}" --balanced_rows_per_motion "${DEMO_ROWS}" \
        "${KIT_QUIET[@]}" agent.logger.backend= \
        env.random_reset_step_min=0 env.random_reset_step_max=200 \
        env.random_reset_full_trajectory=false env.reset_schedule=sequential \
        env.wrap_steps=false env.observations.policy.enable_corruption=false \
        "${NEWTON_ARGS[@]}"
}

merge_samples() {  # demos rollout output
    run_if_missing "$3/merge_manifest.json" \
        "${PLAIN_PY_ARR[@]}" -m imitation_experiments.data.merge_planner_samples \
        --source "$1" --source_limit "${DEMO_ROWS}" \
        --source "$2" --source_limit "${DEMO_ROWS}" \
        --seed "${EVAL_SEED}" --output_dir "$3"
}

# ------------------------------------------------------------ orchestration --
# interface | oracle-demo dir | low-level checkpoint | eval-kind
run_interface() {
    local interface="$1" kind="$2" checkpoint="${3:-}"
    local root="${POINT_ROOT}/${interface}"
    local demos="${ORACLE_ROOT}/${interface}/oracle_demonstrations/rollout_training_samples"
    local pretrain="${root}/planner_pretrain" finetune="${root}/planner_finetune"
    local merged="${root}/demonstration_and_rollout_samples"
    local rollout="${root}/planner_rollout_collection"
    mkdir -p "${root}"

    train_planner "${interface}" "${demos}" "${pretrain}"
    if [[ "${kind}" == latent ]]; then
        latent_eval "${pretrain}/checkpoints/latest.pt" "${root}/eval_pretrained_10starts" \
            "capacity_${MODEL_SIZE}_seed${PLANNER_SEED}_${interface}_pretrained"
        [[ "${DEMO_ONLY}" == "1" ]] || latent_collect "${pretrain}/checkpoints/latest.pt" "${rollout}"
    else
        chunk_eval "${interface}" "${checkpoint}" "${pretrain}/checkpoints/latest.pt" \
            "${root}/eval_pretrained_10starts" \
            "capacity_${MODEL_SIZE}_seed${PLANNER_SEED}_${interface}_pretrained"
        [[ "${DEMO_ONLY}" == "1" ]] || chunk_collect "${interface}" "${checkpoint}" "${pretrain}/checkpoints/latest.pt" "${rollout}"
    fi
    if [[ "${DEMO_ONLY}" == "1" ]]; then
        echo "[SKIP] DEMO_ONLY=1: rollout collection, merge, finetune and finetuned eval"
        return 0
    fi
    merge_samples "${demos}" "${rollout}/rollout_training_samples" "${merged}"
    train_planner "${interface}" "${merged}" "${finetune}" "${pretrain}/checkpoints/latest.pt"
    if [[ "${kind}" == latent ]]; then
        latent_eval "${finetune}/checkpoints/latest.pt" "${root}/eval_finetuned_10starts" \
            "capacity_${MODEL_SIZE}_seed${PLANNER_SEED}_${interface}_finetuned"
    else
        chunk_eval "${interface}" "${checkpoint}" "${finetune}/checkpoints/latest.pt" \
            "${root}/eval_finetuned_10starts" \
            "capacity_${MODEL_SIZE}_seed${PLANNER_SEED}_${interface}_finetuned"
    fi
}

# INTERFACES selects which rows run (EE requires the ee-chunk env adapter).
for interface in ${INTERFACES:-latent_skill full_body_trajectory ee_trajectory}; do
    case "${interface}" in
        latent_skill) run_interface latent_skill latent ;;
        full_body_trajectory) run_interface full_body_trajectory chunk "${FBCHUNK_LOW_LEVEL_CHECKPOINT}" ;;
        ee_trajectory) run_interface ee_trajectory chunk "${EECHUNK_LOW_LEVEL_CHECKPOINT}" ;;
        # Reduced explicit interfaces (qualified 2026-07-28). They use the same
        # chunk path as full-body: a single-frame tracker consuming one slot per
        # control step, with the packet published at 5 Hz.
        root_qpos) run_interface root_qpos chunk "${ROOT_QPOS_LOW_LEVEL_CHECKPOINT}" ;;
        root_points5) run_interface root_points5 chunk "${ROOT_POINTS5_LOW_LEVEL_CHECKPOINT}" ;;
        *) echo "[ERROR] unknown interface ${interface}" >&2; exit 2 ;;
    esac
done

echo "[PASS] Completed ${MODEL_SIZE} seed ${PLANNER_SEED} for: ${INTERFACES:-latent_skill full_body_trajectory ee_trajectory}"
