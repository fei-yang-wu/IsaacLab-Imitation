#!/usr/bin/env bash
set -euo pipefail

# BB3: does command error cost more BEFORE or AFTER the encoder?
#
#   packet side:  expert packet + noise -> [frozen encoder] -> z -> latent tracker
#   z side:       expert packet -> [frozen encoder] -> z + noise -> latent tracker
#
# Both alphas are in per-dimension std units of the clean quantity, calibrated
# from the SAME oracle packets, so a given alpha is the same relative
# perturbation on either side. One shared tracker, so the two curves compare
# directly with no oracle-normalization.
#
# Driven from the EXPERT packet, not a planner: alpha=0 is then the verified
# oracle (30.4 mm) and the curve measures interface error tolerance with planner
# quality removed entirely. That is what B2 in the plan says the planner-driven
# slope cannot do, because there command error co-varies with capacity.
#
# WHY FEW ALPHAS AND REPEATS, not many alphas once:
#   Evaluation is non-deterministic. Five identical runs of one cell measured
#   sd 16.85 mm / cv 12.4% (+/-24% at 95%) on 2026-07-28 -- pure GPU
#   non-determinism in the Newton rollout, with the seed pinned, DR off and
#   flow_inference_noise_std=0. A single run per alpha would produce a curve
#   whose shape is mostly noise. Three repeats per point buys an error bar; the
#   alpha grid is deliberately coarse to pay for it.
#
# Prediction under test: the offline measurement says the encoder AMPLIFIES
# relative error ~1.9x, so the packet-side curve should degrade faster. If the
# curves overlap, that offline number does not transfer to closed loop and
# should be dropped from the paper rather than explained.
#
# Usage:
#   DRY_RUN=1 experiments/campaigns/2026-07-23-lafan1-planner-capacity/run_bb3_noise_curves.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/paths.env"

DEVICE="${DEVICE:-cuda:0}"
DRY_RUN="${DRY_RUN:-0}"
EVAL_STEPS="${EVAL_STEPS:-700}"
FH_ENVS="${FH_ENVS:-4}"
ALPHAS="${ALPHAS:-0.10 0.25 0.50}"
REPS="${REPS:-3}"
SIDES="${SIDES:-packet z}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/interface_baselines/bb3_noise_curves}"
STUDY_ROOT="${STUDY_ROOT:-logs/interface_baselines/lafan1_interface_capacity}"
# Any full-body planner checkpoint satisfies the CLI; with packet_source=expert
# it is never called. Kept explicit so the run records which one was loaded.
PLANNER="${PLANNER:-${STUDY_ROOT}/scaling/seed0/large/full_body_trajectory/planner_pretrain/checkpoints/latest.pt}"
NOISE_REF="${NOISE_REF:-${STUDY_ROOT}/oracle_baselines/full_body_trajectory/oracle_demonstrations/rollout_training_samples/sample_step_000000.pt}"

: "${ISAAC_PY:=pixi run -e isaaclab python}"
read -r -a ISAAC_PY_ARR <<<"${ISAAC_PY}"
NEWTON_ARGS=(physics=newton_mjwarp
    "env.sim.physics.solver_cfg.njmax=${NJMAX:-320}"
    "env.sim.physics.solver_cfg.nconmax=${NCONMAX:-40}")
KIT_QUIET=(--kit_args=--/app/extensions/fsWatcherEnabled=false)
FH_OVERRIDES=(
    env.random_reset_step_min=0 env.random_reset_step_max=0
    env.random_reset_full_trajectory=false env.reset_schedule=sequential
    env.reference_start_frame=0 env.wrap_steps=false env.episode_length_s=20.0
    env.terminations.anchor_pos=null env.terminations.anchor_ori=null
    env.terminations.ee_body_pos=null env.terminations.foot_pos_xyz=null
    env.events.physics_material=null env.events.add_joint_default_pos=null
    env.events.base_com=null env.events.push_robot=null
    env.events.randomize_rigid_body_mass=null
)
LATENT_CMD=(
    env.latent_command_dim=258 agent.ipmd.latent_dim=258
    agent.ipmd.hl_skill_horizon_steps=10 agent.ipmd.hl_skill_command_mode=z
    agent.ipmd.latent_steps_min=10 agent.ipmd.latent_steps_max=10
    agent.ipmd.latent_learning.command_phase_mode=sin_cos
    agent.ipmd.latent_learning.code_latent_dim=256
    agent.ipmd.latent_learning.code_period=10
)
REWARD_ZEROS=(
    agent.ipmd.reward_loss_coeff=0.0 agent.ipmd.reward_l2_coeff=0.0
    agent.ipmd.reward_grad_penalty_coeff=0.0 agent.ipmd.reward_logit_reg_coeff=0.0
    agent.ipmd.reward_param_weight_decay_coeff=0.0
)

[[ -f "${PLANNER}" ]]   || { echo "[ERROR] planner not found: ${PLANNER}" >&2; exit 2; }
[[ -f "${NOISE_REF}" ]] || { echo "[ERROR] noise reference not found: ${NOISE_REF}" >&2; exit 2; }

ran=0; skipped=0
for side in ${SIDES}; do
for alpha in ${ALPHAS}; do
for rep in $(seq 1 "${REPS}"); do
    out="${OUTPUT_ROOT}/${side}/alpha${alpha}/rep${rep}"
    if [[ -f "${out}/summary.json" ]]; then
        echo "[SKIP] ${out}"; skipped=$((skipped+1)); continue
    fi
    if [[ "${side}" == "packet" ]]; then
        noise_args=(--packet_noise_alpha "${alpha}" --z_noise_alpha 0.0)
    else
        noise_args=(--packet_noise_alpha 0.0 --z_noise_alpha "${alpha}")
    fi
    echo "[RUN ] side=${side} alpha=${alpha} rep=${rep} -> ${out}"
    [[ "${DRY_RUN}" == "1" ]] && continue
    mkdir -p "${out}"
    # noise_seed varies per repeat so repeats resample the perturbation rather
    # than re-drawing only the rollout's own non-determinism.
    TERM=xterm PYTHONUNBUFFERED=1 "${ISAAC_PY_ARR[@]}" \
        scripts/rlopt/eval_skill_commander_closed_loop.py \
        --headless --device "${DEVICE}" --task "${LATENT_TASK}" --algorithm IPMD \
        --checkpoint "${LATENT_LOW_LEVEL_CHECKPOINT}" \
        --skill_checkpoint "${LATENT_SKILL_CHECKPOINT}" \
        --packet_planner_checkpoint "${PLANNER}" \
        --packet_interface full_body_trajectory --packet_source expert \
        "${noise_args[@]}" --noise_reference_samples "${NOISE_REF}" \
        --noise_seed "${rep}" \
        --motion_name "${MOTION_NAME}" --seed 0 --metric_interval 10 \
        --num_envs "${FH_ENVS}" --max_steps "${EVAL_STEPS}" \
        --state_history_steps 9 --flow_num_inference_steps 16 \
        --flow_inference_noise_std 0.0 \
        --keep_time_out --keep_early_terminations \
        --output_dir "${out}" --label "bb3_${side}_a${alpha}_rep${rep}" \
        "${KIT_QUIET[@]}" agent.logger.backend= \
        agent.ipmd.command_source=hl_skill \
        "agent.ipmd.hl_skill_checkpoint_path=${LATENT_SKILL_CHECKPOINT}" \
        agent.ipmd.hl_skill_finetune_enabled=false \
        "env.lafan1_manifest_path=${MANIFEST}" \
        "env.dataset_path=${LATENT_DATASET_PATH}" \
        env.refresh_zarr_dataset=false \
        env.observations.policy.enable_corruption=false \
        "${LATENT_CMD[@]}" "${REWARD_ZEROS[@]}" "${NEWTON_ARGS[@]}" \
        "${FH_OVERRIDES[@]}" > "${out}/run.log" 2>&1 || {
            echo "[FAIL] ${out} -- see ${out}/run.log" >&2; exit 2; }
    ran=$((ran+1))
done; done; done

echo "[DONE] ran=${ran} skipped=${skipped}"
