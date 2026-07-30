#!/usr/bin/env bash
set -euo pipefail

# Finetune ablation: METHOD B (oracle-driven aggregation) vs METHOD A (DAgger).
#
# The main pipeline finetunes with DAgger: the *pretrained planner* drives the
# robot, the states it visits are recorded, and the expert command at each of
# those states becomes the label
# (build_planner_sample: causal_state_history=learner state,
#  causal_target=expert window at that state). One DAgger round, 2000 updates.
#
# Method B keeps everything identical except WHO DRIVES: the oracle commands the
# robot, so the aggregated states come from the expert's own closed-loop
# distribution rather than the learner's. Same row budget, same merge, same
# optimizer budget, same starting checkpoint, same evaluation.
#
# This isolates the one factor that differs. It matters because the main
# pipeline's DAgger round changed *two* things at once relative to pretraining:
# the driver AND the start distribution (pretrain demos are oracle-driven from
# frame 0 sequential; the DAgger collection uses random starts 0-200). Method B
# is collected with the SAME random-start range as the DAgger collection, so a
# method A vs B difference is attributable to the driver alone.
#
# Motivation: after the joint-order fix, finetuning helps enormously at tiny
# (359 -> 88 mm) but HURTS at small/medium/large (e.g. 124 -> 254 mm at large).
# A plausible cause is that DAgger aggregates states visited by a still-poor
# pretrained planner, which are far from where a good policy operates.
#
# Usage:
#   DRY_RUN=1 experiments/campaigns/2026-07-23-lafan1-planner-capacity/run_finetune_method_b.sh
#   MODEL_SIZE=medium SEEDS="0 1 2" INTERFACES=full_body_trajectory run_finetune_method_b.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || (cd "${SCRIPT_DIR}/../../.." && pwd))"
cd "${REPO_ROOT}"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/paths.env"

MODEL_SIZE="${MODEL_SIZE:-medium}"
SEEDS="${SEEDS:-0 1 2}"
INTERFACES="${INTERFACES:-full_body_trajectory}"
EVAL_SEED="${EVAL_SEED:-0}"
DEVICE="${DEVICE:-cuda:0}"
DRY_RUN="${DRY_RUN:-0}"
EVAL_STEPS="${EVAL_STEPS:-700}"
FH_ENVS="${FH_ENVS:-4}"
DEMO_ROWS="${DEMO_ROWS:-1000}"
NUM_UPDATES="${NUM_UPDATES:-2000}"
COLLECT_STEPS="${COLLECT_STEPS:-3000}"
STUDY_ROOT="${STUDY_ROOT:-logs/interface_baselines/lafan1_planner_capacity_20260723}"
ORACLE_ROOT="${ORACLE_ROOT:-${STUDY_ROOT}/oracle_baselines}"
OUT_ROOT="${OUT_ROOT:-${STUDY_ROOT}/finetune_method_b}"

: "${ISAAC_PY:=pixi run -e isaaclab python}"
: "${PLAIN_PY:=pixi run python}"
read -r -a ISAAC_PY_ARR <<<"${ISAAC_PY}"
read -r -a PLAIN_PY_ARR <<<"${PLAIN_PY}"
NEWTON_ARGS=(physics=newton_mjwarp
    "env.sim.physics.solver_cfg.njmax=${NJMAX:-320}"
    "env.sim.physics.solver_cfg.nconmax=${NCONMAX:-40}")
KIT_QUIET=(--kit_args=--/app/extensions/fsWatcherEnabled=false)

# Evaluation protocol -- byte-identical to the main sweep so method A and B rows
# are directly comparable.
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

run_if_missing() {
    local marker="$1"; shift
    if [[ -e "${marker}" ]]; then echo "[SKIP] ${marker}"; return 0; fi
    printf '[CMD]'; printf ' %q' "$@"; printf '\n'
    [[ "${DRY_RUN}" == "1" ]] && return 0
    TERM=xterm PYTHONUNBUFFERED=1 "$@"
    [[ -e "${marker}" ]] || { echo "[ERROR] missing artifact: ${marker}" >&2; exit 2; }
}

for seed in ${SEEDS}; do
for interface in ${INTERFACES}; do
    case "${interface}" in
        full_body_trajectory) LOW_LEVEL="${FBCHUNK_LOW_LEVEL_CHECKPOINT}" ;;
        ee_trajectory)        LOW_LEVEL="${EECHUNK_LOW_LEVEL_CHECKPOINT}" ;;
        *) echo "[ERROR] method B currently covers the chunk interfaces only; got ${interface}" >&2; exit 2 ;;
    esac
    root="${OUT_ROOT}/seed${seed}/${MODEL_SIZE}/${interface}"
    pretrain="${STUDY_ROOT}/scaling/seed${seed}/${MODEL_SIZE}/${interface}/planner_pretrain/checkpoints/latest.pt"
    demos="${ORACLE_ROOT}/${interface}/oracle_demonstrations/rollout_training_samples"
    oracle_agg="${root}/oracle_aggregation"
    merged="${root}/demonstration_and_oracle_samples"
    finetune="${root}/planner_finetune_b"
    if [[ ! -e "${pretrain}" && "${DRY_RUN}" != "1" ]]; then
        echo "[ERROR] missing pretrained planner: ${pretrain}" >&2; exit 2
    fi
    mkdir -p "${root}"

    # (1) Oracle-driven aggregation. No --planner_checkpoint, so the env fills
    #     the command buffer from the expert (command_observation_source=
    #     planner_oracle) and the ORACLE drives. Random starts 0-200 match the
    #     DAgger collection, so the driver is the only difference.
    run_if_missing "${oracle_agg}/rollout_training_samples/sample_step_000000.pt" \
        "${ISAAC_PY_ARR[@]}" -m imitation_experiments.data.collect_interface_rollout_samples \
        --task "${CHUNK_TASK}" --algorithm IPMD --checkpoint "${LOW_LEVEL}" \
        --interface "${interface}" --motion_name "${MOTION_NAME}" \
        --motion_manifest "${MANIFEST}" \
        --planner_interval_steps 10 --command_future_steps 9 --command_past_steps 0 \
        --low_level_command_mode streamed_vanilla --state_history_steps 9 \
        --seed "${EVAL_SEED}" --num_envs 10 --control_steps "${COLLECT_STEPS}" \
        --reset_schedule sequential --reference_start_frame 0 \
        --balanced_rows_per_motion "${DEMO_ROWS}" \
        --balanced_motion_names "${MOTION_NAME}" \
        --sample_rows_per_file "${DEMO_ROWS}" \
        --output_dir "${oracle_agg}" \
        "${KIT_QUIET[@]}" "${NEWTON_ARGS[@]}" \
        env.random_reset_step_min=0 env.random_reset_step_max=200 \
        env.random_reset_full_trajectory=false env.wrap_steps=false \
        env.observations.policy.enable_corruption=false

    # (2) Same merge as method A: demos + aggregation, equal per-source limits.
    run_if_missing "${merged}/merge_manifest.json" \
        "${PLAIN_PY_ARR[@]}" -m imitation_experiments.data.merge_planner_samples \
        --source "${demos}" --source_limit "${DEMO_ROWS}" \
        --source "${oracle_agg}/rollout_training_samples" --source_limit "${DEMO_ROWS}" \
        --seed "${EVAL_SEED}" --output_dir "${merged}"

    # (3) Same optimizer budget, same starting checkpoint as method A.
    run_if_missing "${finetune}/checkpoints/latest.pt" \
        "${PLAIN_PY_ARR[@]}" -m imitation_experiments.planner.train_chunked_transformer_planner \
        --samples_dir "${merged}" --output_dir "${finetune}" \
        --interface "${interface}" --planner_family flow --state_key planner_state \
        --device "${DEVICE}" --seed "${seed}" \
        --batch_size 256 --micro_batch_size 32 --num_updates "${NUM_UPDATES}" \
        --log_interval 100 --eval_batch_size 512 --eval_max_samples 4096 \
        --lr 0.0001 --weight_decay 0.0001 --model_size "${MODEL_SIZE}" \
        --flow_num_inference_steps 16 --endpoint_num_inference_steps 4 \
        --flow_inference_noise_std 0.0 --checkpoint "${pretrain}"

    # (4) Identical evaluation to the main sweep.
    run_if_missing "${root}/eval_finetuned_b/summary.json" \
        "${ISAAC_PY_ARR[@]}" -m imitation_experiments.evaluation.eval_interface_planner_closed_loop \
        --headless --device "${DEVICE}" --task "${CHUNK_TASK}" --algorithm IPMD \
        --checkpoint "${LOW_LEVEL}" --low_level_command_mode streamed_vanilla \
        --planner_checkpoint "${finetune}/checkpoints/latest.pt" \
        --output_json "${root}/eval_finetuned_b/summary.json" \
        --pin_command_joint_order on \
        --label "methodb_${MODEL_SIZE}_seed${seed}_${interface}_finetuned" \
        --motion_manifest "${MANIFEST}" --motion_name "${MOTION_NAME}" \
        --num_envs "${FH_ENVS}" --steps "${EVAL_STEPS}" --seed "${EVAL_SEED}" \
        --state_history_steps 9 --command_past_steps 0 --command_future_steps 9 \
        --planner_update_interval 10 --flow_num_inference_steps 16 \
        --flow_inference_noise_std 0.0 --reset_schedule sequential \
        --reference_start_frame 0 --keep_after_done \
        "${KIT_QUIET[@]}" agent.logger.backend= \
        env.observations.policy.enable_corruption=false \
        "${NEWTON_ARGS[@]}" "${_FH_COMMON[@]}"
done
done

echo "[PASS] finetune method B complete under ${OUT_ROOT}"
