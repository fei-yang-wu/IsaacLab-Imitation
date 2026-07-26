#!/usr/bin/env bash
set -euo pipefail

# Generate the shared oracle baselines for the LAFAN1 one-motion planner-capacity
# study: for each of the 3 interfaces, (a) a frame-0 / ~700-step oracle metrics
# summary.json used to oracle-normalize planner MPJPE, and (b) balanced oracle
# demonstration rows used to pretrain each planner. Run ONCE; the capacity sweep
# reuses these artifacts across every size/seed cell.
#
# Usage:
#   DRY_RUN=1 experiments/campaigns/2026-07-23-lafan1-planner-capacity/prepare_oracle_baselines.sh
#   experiments/campaigns/2026-07-23-lafan1-planner-capacity/prepare_oracle_baselines.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || (cd "${SCRIPT_DIR}/../../.." && pwd))"
cd "${REPO_ROOT}"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/paths.env"

DEVICE="${DEVICE:-cuda:0}"
DRY_RUN="${DRY_RUN:-0}"
ORACLE_SEED="${ORACLE_SEED:-0}"
DEMO_ROWS="${DEMO_ROWS:-1000}"
ORACLE_STEPS="${ORACLE_STEPS:-700}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/interface_baselines/lafan1_planner_capacity_20260723/oracle_baselines}"

# Runtime abstraction: local uses pixi; inside the ICE container ISAAC_PY is set
# to /isaac-sim/python.sh (pixi is unavailable there). Command tokens are arrays.
: "${ISAAC_PY:=pixi run -e isaaclab python}"
read -r -a ISAAC_PY_ARR <<<"${ISAAC_PY}"
# The frozen oracles were trained with the Newton backend; eval must match it.
NEWTON_ARGS=(physics=newton_mjwarp
    "env.sim.physics.solver_cfg.njmax=${NJMAX:-320}"
    "env.sim.physics.solver_cfg.nconmax=${NCONMAX:-40}")
# On a compute-only GPU (ICE h100, no display) the latent evaluator needs this.
LATENT_KITLESS=()
[[ "${ASSERT_KITLESS:-0}" == "1" ]] && LATENT_KITLESS=(--assert-kitless)

# Full-episode video on the oracle diagnostic passes (user request). Cameras need
# Kit, so this is disabled under kitless (ICE compute-only GPUs).
VIDEO_ARGS=()
if [[ "${RENDER_VIDEO:-1}" == "1" && "${ASSERT_KITLESS:-0}" != "1" ]]; then
    VIDEO_ARGS=(--video --video_length "${VIDEO_STEPS:-150}")
fi

KIT_QUIET=(--kit_args=--/app/extensions/fsWatcherEnabled=false)

# Reward-term zeros shared by every latent invocation (learned-reward off).
LATENT_REWARD_ZEROS=(
    agent.ipmd.reward_loss_coeff=0.0 agent.ipmd.reward_l2_coeff=0.0
    agent.ipmd.reward_grad_penalty_coeff=0.0 agent.ipmd.reward_logit_reg_coeff=0.0
    agent.ipmd.reward_param_weight_decay_coeff=0.0
)
# Latent command contract (matches the deterministic DiffSR encoder: z256+phase).
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

# Full-horizon oracle protocol (matches the planner eval, user 2026-07-24):
# frame 0, single ORACLE_STEPS rollout, all terminations + domain randomization
# off. BOTH surfaces are SONIC-family and share these term names, so the
# override set must be identical for both interfaces -- see the long note in
# run_capacity_point.sh for the asymmetry this replaces.
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

# ---------------------------------------------------------------- LATENT -----
latent_oracle="${OUTPUT_ROOT}/latent_skill/oracle_frame0_${ORACLE_STEPS}"
latent_demos="${OUTPUT_ROOT}/latent_skill/oracle_demonstrations"

latent_common=(
    "${ISAAC_PY_ARR[@]}" scripts/rlopt/eval_skill_commander_closed_loop.py
    "${LATENT_KITLESS[@]}"
    --headless --device "${DEVICE}"
    --task "${LATENT_TASK}" --algorithm IPMD
    --checkpoint "${LATENT_LOW_LEVEL_CHECKPOINT}"
    --skill_checkpoint "${LATENT_SKILL_CHECKPOINT}"
    --motion_name "${MOTION_NAME}" --seed "${ORACLE_SEED}" --metric_interval 10
    --flow_inference_noise_std 0.0
    --keep_time_out --keep_early_terminations
    "${KIT_QUIET[@]}"
    agent.logger.backend=
    agent.ipmd.command_source=hl_skill
    "agent.ipmd.hl_skill_checkpoint_path=${LATENT_SKILL_CHECKPOINT}"
    agent.ipmd.hl_skill_finetune_enabled=false
    "env.lafan1_manifest_path=${MANIFEST}" "env.dataset_path=${LATENT_DATASET_PATH}"
    env.refresh_zarr_dataset=false env.random_reset_full_trajectory=false
    env.reset_schedule=sequential env.wrap_steps=false
    env.observations.policy.enable_corruption=false
    "${LATENT_CMD[@]}" "${LATENT_REWARD_ZEROS[@]}" "${NEWTON_ARGS[@]}"
)

# (a) frame-0 / ORACLE_STEPS full-horizon oracle metrics summary.
run_if_missing "${latent_oracle}/summary.json" \
    "${latent_common[@]}" \
    --num_envs 10 --max_steps "${ORACLE_STEPS}" \
    "${VIDEO_ARGS[@]}" \
    --output_dir "${latent_oracle}" --label latent_oracle_frame0_${ORACLE_STEPS} \
    "${LATENT_FH[@]}"

# (b) balanced oracle demonstration rows (broad coverage over the motion).
run_if_missing "${latent_demos}/rollout_training_samples/sample_step_000000.pt" \
    "${latent_common[@]}" \
    --num_envs 10 --max_steps 1000 --disable_tracking_terminations \
    --save_rollout_training_samples --continue_after_reset \
    --balanced_rows_per_motion "${DEMO_ROWS}" --balanced_motion_names "${MOTION_NAME}" \
    --sample_rows_per_file "${DEMO_ROWS}" \
    --output_dir "${latent_demos}" --label latent_oracle_demonstrations \
    env.random_reset_step_min=0 env.random_reset_step_max=200

# ------------------------------------------------------------- FB / EE CHUNK -
chunk_oracle() {
    local interface="$1" checkpoint="$2"
    local oracle="${OUTPUT_ROOT}/${interface}/oracle_frame0_${ORACLE_STEPS}"
    local demos="${OUTPUT_ROOT}/${interface}/oracle_demonstrations"
    local collect=(
        "${ISAAC_PY_ARR[@]}"
        experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/collect_interface_rollout_samples.py
        --task "${CHUNK_TASK}" --algorithm IPMD --checkpoint "${checkpoint}"
        --interface "${interface}" --motion_name "${MOTION_NAME}"
        --motion_manifest "${MANIFEST}"
        --planner_interval_steps 10 --command_future_steps 9 --command_past_steps 0
        --low_level_command_mode streamed_vanilla --state_history_steps 9
        --seed "${ORACLE_SEED}" --num_envs 10 "${KIT_QUIET[@]}" "${NEWTON_ARGS[@]}"
    )
    # (a) frame-0 / ORACLE_STEPS full-horizon oracle metrics summary (eval only).
    run_if_missing "${oracle}/summary.json" \
        "${collect[@]}" \
        --control_steps "${ORACLE_STEPS}" \
        --reference_start_frame 0 --evaluation_only "${VIDEO_ARGS[@]}" \
        --output_dir "${oracle}" "${CHUNK_FH[@]}"
    # (b) balanced oracle demonstration rows.
    run_if_missing "${demos}/rollout_training_samples/sample_step_000000.pt" \
        "${collect[@]}" \
        --control_steps 1000 --reset_schedule sequential --reference_start_frame 0 \
        --balanced_rows_per_motion "${DEMO_ROWS}" --balanced_motion_names "${MOTION_NAME}" \
        --sample_rows_per_file "${DEMO_ROWS}" \
        --output_dir "${demos}"
}

# streamed_vanilla is full-body only; EE oracle waits on the ee-chunk adapter.
INTERFACES="${INTERFACES:-latent_skill full_body_trajectory ee_trajectory}"
[[ " ${INTERFACES} " == *" full_body_trajectory "* ]] && \
    chunk_oracle full_body_trajectory "${FBCHUNK_LOW_LEVEL_CHECKPOINT}"
[[ " ${INTERFACES} " == *" ee_trajectory "* ]] && \
    chunk_oracle ee_trajectory "${EECHUNK_LOW_LEVEL_CHECKPOINT}"

echo "[PASS] Oracle baselines prepared under ${OUTPUT_ROOT} for: ${INTERFACES}"
