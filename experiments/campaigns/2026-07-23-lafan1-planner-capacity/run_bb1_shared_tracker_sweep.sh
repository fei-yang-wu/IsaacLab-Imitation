#!/usr/bin/env bash
set -euo pipefail

# BB1 shared-tracker sweep: drive the ONE latent tracker from each explicit-packet
# planner through the frozen skill encoder, so the only thing that differs from
# the latent row is the planner's output space.
#
#     explicit:  planner -> packet -> [frozen encoder] -> z -> latent tracker
#     latent:    planner ---------------------------------> z -> latent tracker
#
# Both rows then share tracker, decoder and oracle ceiling, so the comparison
# needs no oracle-normalization -- which is what removes the single biggest
# interpretive weakness of the per-interface-tracker study.
#
# Eval-only: reuses the already-trained planner checkpoints, the frozen encoder
# and the frozen latent tracker. No training.
#
# The `expert` pin (PACKET_SOURCE=expert) MUST reproduce the latent oracle
# (30.42 mm) before any planner row is interpretable; it verifies the term-major
# -> frame-interleaved reorder, the [state ; 9-frame window] split, and that the
# encoder is fed raw rather than normalized features. Run it first:
#
#   PACKET_SOURCE=expert SIZES=large SEEDS=0 ./run_bb1_shared_tracker_sweep.sh
#
# Usage:
#   DRY_RUN=1 experiments/campaigns/2026-07-23-lafan1-planner-capacity/run_bb1_shared_tracker_sweep.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/paths.env"

DEVICE="${DEVICE:-cuda:0}"
DRY_RUN="${DRY_RUN:-0}"
EVAL_SEED="${EVAL_SEED:-0}"
EVAL_STEPS="${EVAL_STEPS:-700}"
FH_ENVS="${FH_ENVS:-4}"
SIZES="${SIZES:-tiny small medium large}"
SEEDS="${SEEDS:-0 1 2}"
PACKET_INTERFACES="${PACKET_INTERFACES:-full_body_trajectory}"
PACKET_SOURCE="${PACKET_SOURCE:-planner}"
# Repeat each cell REPEATS times into <out>/rep<N>. Everything is held fixed --
# same seed, same checkpoint, no domain randomization, flow_inference_noise_std=0
# -- so any spread across repeats is pure evaluation non-determinism (GPU
# reductions in Newton/mjwarp contact solving). Two identical large/seed0 runs
# were measured 11% apart on 2026-07-28, which means every single-run number in
# this study is an n=1 draw, not an exact value. Use REPEATS to size that error
# bar before quoting any gap.
REPEATS="${REPEATS:-1}"
STUDY_ROOT="${STUDY_ROOT:-logs/interface_baselines/lafan1_interface_capacity}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/interface_baselines/bb1_shared_tracker}"

: "${ISAAC_PY:=pixi run -e isaaclab python}"
read -r -a ISAAC_PY_ARR <<<"${ISAAC_PY}"
NEWTON_ARGS=(physics=newton_mjwarp
    "env.sim.physics.solver_cfg.njmax=${NJMAX:-320}"
    "env.sim.physics.solver_cfg.nconmax=${NCONMAX:-40}")
KIT_QUIET=(--kit_args=--/app/extensions/fsWatcherEnabled=false)

# Same full-horizon protocol as the capacity grid: frame 0, one uninterrupted
# rollout, tracking terminations and domain randomization off. Falls are still
# detected, from raw torso height rather than a termination term.
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

ran=0; skipped=0; missing=0
for interface in ${PACKET_INTERFACES}; do
for seed in ${SEEDS}; do
for size in ${SIZES}; do
    planner="${STUDY_ROOT}/scaling/seed${seed}/${size}/${interface}/planner_pretrain/checkpoints/latest.pt"
  for rep in $(seq 1 "${REPEATS}"); do
    if [[ "${REPEATS}" == "1" ]]; then
        out="${OUTPUT_ROOT}/${interface}/seed${seed}/${size}"
    else
        out="${OUTPUT_ROOT}/${interface}/seed${seed}/${size}/rep${rep}"
    fi
    if [[ ! -f "${planner}" ]]; then
        echo "[MISS] no planner checkpoint: ${planner}"; missing=$((missing+1)); break
    fi
    if [[ -f "${out}/summary.json" ]]; then
        echo "[SKIP] ${out}/summary.json"; skipped=$((skipped+1)); continue
    fi
    echo "[RUN ] ${interface} seed${seed} ${size} -> ${out}"
    [[ "${DRY_RUN}" == "1" ]] && continue
    mkdir -p "${out}"
    TERM=xterm PYTHONUNBUFFERED=1 "${ISAAC_PY_ARR[@]}" \
        scripts/rlopt/eval_skill_commander_closed_loop.py \
        --headless --device "${DEVICE}" --task "${LATENT_TASK}" --algorithm IPMD \
        --checkpoint "${LATENT_LOW_LEVEL_CHECKPOINT}" \
        --skill_checkpoint "${LATENT_SKILL_CHECKPOINT}" \
        --packet_planner_checkpoint "${planner}" \
        --packet_interface "${interface}" --packet_source "${PACKET_SOURCE}" \
        --motion_name "${MOTION_NAME}" --seed "${EVAL_SEED}" --metric_interval 10 \
        --num_envs "${FH_ENVS}" --max_steps "${EVAL_STEPS}" \
        --state_history_steps 9 --flow_num_inference_steps 16 \
        --flow_inference_noise_std 0.0 \
        --keep_time_out --keep_early_terminations \
        --output_dir "${out}" --label "bb1_${interface}_seed${seed}_${size}" \
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
  done
done; done; done

echo "[DONE] ran=${ran} skipped=${skipped} missing=${missing}"
