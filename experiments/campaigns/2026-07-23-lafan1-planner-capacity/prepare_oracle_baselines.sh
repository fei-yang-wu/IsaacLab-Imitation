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
ORACLE_STEPS="${ORACLE_STEPS:-700}"
ORACLE_ENVS="${ORACLE_ENVS:-10}"

# ------------------------------------------------- DEMONSTRATION BUDGET ------
# The budget is expressed in TRAJECTORIES, not rows. A trajectory is one episode
# in one environment -- the unit VLA work reports and the unit a held-out split
# has to respect. Rows are a derived quantity: one row per environment per
# planner publish, so a 500-step episode published every 10 steps contributes 50
# rows. Reporting "1000 rows" conflated the two and made a 10-trajectory set
# look like a large one.
DEMO_TRAJECTORIES="${DEMO_TRAJECTORIES:-100}"
DEMO_ENVS="${DEMO_ENVS:-10}"
# Episode length, pinned identically for every interface. Both collectors
# otherwise pick their own: the chunk collector stretches the timeout to cover
# all control steps (recorded 20.04 s -> one uninterrupted rollout), while the
# latent one keeps the task default (10.0 s -> a reset every 500 steps).
# Different reset cadence means different training state distributions, which is
# not something a capacity comparison may vary.
DEMO_EPISODE_LENGTH_S="${DEMO_EPISODE_LENGTH_S:-10.0}"
DEMO_CONTROL_HZ="${DEMO_CONTROL_HZ:-50}"
DEMO_PUBLISH_INTERVAL="${DEMO_PUBLISH_INTERVAL:-10}"

# Derived. Kept as arithmetic rather than magic numbers so changing the budget
# in trajectories cannot silently desynchronize the step count from the row
# count -- collection stops at whichever limit binds first, so a mismatch would
# quietly truncate the set.
DEMO_EPISODE_STEPS=$(awk "BEGIN{printf \"%d\", ${DEMO_EPISODE_LENGTH_S} * ${DEMO_CONTROL_HZ}}")
if (( DEMO_TRAJECTORIES % DEMO_ENVS != 0 )); then
    echo "[ERROR] DEMO_TRAJECTORIES (${DEMO_TRAJECTORIES}) must be a multiple of DEMO_ENVS (${DEMO_ENVS}) so every environment contributes equally." >&2
    exit 2
fi
DEMO_EPISODES_PER_ENV=$(( DEMO_TRAJECTORIES / DEMO_ENVS ))
DEMO_STEPS=$(( DEMO_EPISODES_PER_ENV * DEMO_EPISODE_STEPS ))
DEMO_ROWS="${DEMO_ROWS:-$(( DEMO_TRAJECTORIES * DEMO_EPISODE_STEPS / DEMO_PUBLISH_INTERVAL ))}"
echo "[INFO] Demonstration budget: ${DEMO_TRAJECTORIES} trajectories" \
     "(${DEMO_ENVS} envs x ${DEMO_EPISODES_PER_ENV} episodes x ${DEMO_EPISODE_STEPS} steps)" \
     "-> ${DEMO_STEPS} control steps, ${DEMO_ROWS} rows."
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

# INTERFACES gates every block below, latent included. It used to gate only the
# chunk rows, so the two latent Isaac runs fired on every invocation -- including
# `INTERFACES=root_points5`, and once per interface when a caller loops.
INTERFACES="${INTERFACES:-latent_skill full_body_trajectory ee_trajectory}"

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

if [[ " ${INTERFACES} " == *" latent_skill "* ]]; then
    # (a) frame-0 / ORACLE_STEPS full-horizon oracle metrics summary.
    run_if_missing "${latent_oracle}/summary.json" \
        "${latent_common[@]}" \
        --num_envs 10 --max_steps "${ORACLE_STEPS}" \
        "${VIDEO_ARGS[@]}" \
        --output_dir "${latent_oracle}" --label latent_oracle_frame0_${ORACLE_STEPS} \
        "${LATENT_FH[@]}"

    # (b) balanced oracle demonstration rows (broad coverage over the motion).
    #
    # --allow_random_reset is required for the two overrides on the last line to
    # survive. Without it `eval_skill_commander_closed_loop.py:1282-1289` forces
    # random_reset_step_min/max back to 0/0 *after* Hydra has applied them, so
    # every recorded latent demonstration set until 2026-07-28 started at frame
    # 0 while the explicit sets started uniformly in 0-200 (confirmed in the
    # recorded summary.json: latent max 0, explicit max 200). The override was
    # dead code, not a setting.
    run_if_missing "${latent_demos}/rollout_training_samples/sample_step_000000.pt" \
        "${latent_common[@]}" \
        --num_envs "${DEMO_ENVS}" --max_steps "${DEMO_STEPS}" \
        --disable_tracking_terminations \
        --save_rollout_training_samples --continue_after_reset \
        --allow_random_reset \
        --balanced_rows_per_motion "${DEMO_ROWS}" --balanced_motion_names "${MOTION_NAME}" \
        --sample_rows_per_file "${DEMO_ROWS}" \
        --output_dir "${latent_demos}" --label latent_oracle_demonstrations \
        env.random_reset_step_min=0 env.random_reset_step_max=200 \
        "env.episode_length_s=${DEMO_EPISODE_LENGTH_S}"
else
    echo "[SKIP] latent_skill not in INTERFACES"
fi

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
        --seed "${ORACLE_SEED}" "${KIT_QUIET[@]}" "${NEWTON_ARGS[@]}"
    )
    # (a) frame-0 / ORACLE_STEPS full-horizon oracle metrics summary (eval only).
    run_if_missing "${oracle}/summary.json" \
        "${collect[@]}" \
        --control_steps "${ORACLE_STEPS}" --num_envs "${ORACLE_ENVS}" \
        --reference_start_frame 0 --evaluation_only "${VIDEO_ARGS[@]}" \
        --output_dir "${oracle}" "${CHUNK_FH[@]}"
    # (b) balanced oracle demonstration rows.
    #
    # The two flags below exist to make this collection identical to the latent
    # arm's. Without them the demonstration sets were collected under different
    # start distributions, so every downstream capacity comparison was
    # uncontrolled (audit 2026-07-28):
    #
    #   --disable_tracking_terminations  disables anchor_pos / anchor_ori /
    #       ee_body_pos, which the latent call has disabled since it was
    #       written. This is the asymmetry that actually mattered: with those
    #       terms ACTIVE, an explicit episode was terminated and reset the
    #       moment its tracker drifted, so the explicit demonstration sets
    #       contain only well-tracked states while the latent set contains
    #       drifted ones too (recorded state std 0.854-0.904 vs latent 1.014).
    #       An explicit planner trained only on well-tracked states has never
    #       seen the drifted states it must recover from at test time, which
    #       biases the closed-loop comparison against it. The flag also pins
    #       random_reset_step_min/max to 0/200 -- already what the explicit arms
    #       recorded, so that part is a no-op here and the equalizing change on
    #       the latent side is --allow_random_reset.
    #
    #   --keep_configured_episode_length  the collector otherwise stretches
    #       episode_length_s to cover all 1000 control steps (recorded 20.04),
    #       giving one uninterrupted rollout per env. The latent call does NOT
    #       pass --extend_episode_length_for_max_steps, so it kept the 10 s /
    #       500-step episode and reset once mid-collection. Both are now pinned
    #       to DEMO_EPISODE_LENGTH_S so the reset cadence is identical.
    run_if_missing "${demos}/rollout_training_samples/sample_step_000000.pt" \
        "${collect[@]}" \
        --control_steps "${DEMO_STEPS}" --num_envs "${DEMO_ENVS}" \
        --reset_schedule sequential --reference_start_frame 0 \
        --disable_tracking_terminations --keep_configured_episode_length \
        --balanced_rows_per_motion "${DEMO_ROWS}" --balanced_motion_names "${MOTION_NAME}" \
        --sample_rows_per_file "${DEMO_ROWS}" \
        --output_dir "${demos}" \
        "env.episode_length_s=${DEMO_EPISODE_LENGTH_S}"
}

[[ " ${INTERFACES} " == *" full_body_trajectory "* ]] && \
    chunk_oracle full_body_trajectory "${FBCHUNK_LOW_LEVEL_CHECKPOINT}"
[[ " ${INTERFACES} " == *" ee_trajectory "* ]] && \
    chunk_oracle ee_trajectory "${EECHUNK_LOW_LEVEL_CHECKPOINT}"
# Reduced explicit interfaces (qualified 2026-07-28): same chunk path, own
# controller. chunk_oracle is interface-generic, so these need no special case
# beyond naming their checkpoint.
[[ " ${INTERFACES} " == *" root_qpos "* ]] && \
    chunk_oracle root_qpos "${ROOT_QPOS_LOW_LEVEL_CHECKPOINT}"
[[ " ${INTERFACES} " == *" root_points5 "* ]] && \
    chunk_oracle root_points5 "${ROOT_POINTS5_LOW_LEVEL_CHECKPOINT}"

echo "[PASS] Oracle baselines prepared under ${OUTPUT_ROOT} for: ${INTERFACES}"
