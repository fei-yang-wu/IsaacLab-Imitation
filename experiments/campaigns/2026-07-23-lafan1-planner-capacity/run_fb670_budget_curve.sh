#!/usr/bin/env bash
set -euo pipefail

# FB-670 training-budget curve (2026-07-30 user request):
#   Does the explicit 670-value full-body planner converge, and how does its
#   closed-loop tracking / survival move across optimization budget?
#
# Stages (select via STAGES, space separated):
#   collect    one balanced 100-env oracle demonstration collection (DEMO_ROWS
#              rows, random starts 0-200, tracking terminations disabled --
#              the frozen demonstration protocol).
#   train      one flow planner per MODEL_SIZE at TRAIN_UPDATES updates with
#              optimizer-free milestone snapshots every MILESTONE_INTERVAL.
#   eval       closed-loop evaluation of every EVAL_STRIDE-th milestone plus
#              best.pt under the rigorous protocol: reference frame 0,
#              training-time domain randomization and pushes ACTIVE,
#              base_too_low (torso < FALL_HEIGHT_M) as the only early
#              termination, EVAL_ENVS parallel environments, EVAL_STEPS steps.
#              MPJPE accumulates only over valid pre-termination transitions.
#   aggregate  MPJPE / survival / fall-count vs updates curve tables.
#
# The demonstration and training recipes are copied from the frozen
# prepare_oracle_baselines.sh / enc380 route-comparison budgets so this curve
# is comparable with the 2026-07-30 progressive-budget grids. The evaluation
# protocol matches run_frame0_dr_baseonly_evaluation.sh except for the
# environment count (4096 vs 100) -- a deliberate averaging upgrade.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
while [[ ! -x "${REPO_ROOT}/docker/cluster/cluster_interface.sh" ]]; do
    if [[ "${REPO_ROOT}" == / ]]; then
        echo "[ERROR] Could not locate repository root above ${SCRIPT_DIR}." >&2
        exit 2
    fi
    REPO_ROOT="$(dirname "${REPO_ROOT}")"
done
cd "${REPO_ROOT}"

STAGES="${STAGES:-collect train eval aggregate}"
STUDY_ROOT="${STUDY_ROOT:-logs/interface_baselines/lafan1_fb670_budget_curve_20260730}"
MOTION_NAME="${MOTION_NAME:-walk1_subject1}"
MANIFEST="${MANIFEST:-data/lafan1/manifests/g1_lafan1_manifest.json}"
CHUNK_TASK="${CHUNK_TASK:-Isaac-Imitation-G1-Strict-v0}"
DEVICE="${DEVICE:-cuda:0}"

FB_LOW_LEVEL_CHECKPOINT="${FB_LOW_LEVEL_CHECKPOINT:-logs/downloaded_checkpoints/lafan1_fbchunk_5b_seed0/model_step_5000085504.pt}"
EXPECTED_FB_SHA256="${EXPECTED_FB_SHA256:-681a712ea8635aaaf89f788d3d73d3142dab0b26fbb2bb6ab805d27c805a0bf6}"

# Demonstration collection (frozen protocol; see prepare_oracle_baselines.sh
# for why --disable_tracking_terminations and --keep_configured_episode_length
# are mandatory for a comparable demonstration distribution).
DEMO_ROWS="${DEMO_ROWS:-5000}"
DEMO_ENVS="${DEMO_ENVS:-100}"
DEMO_STEPS="${DEMO_STEPS:-15000}"
DEMO_EPISODE_LENGTH_S="${DEMO_EPISODE_LENGTH_S:-10.0}"
DEMO_SEED="${DEMO_SEED:-0}"

# Planner training (matched to the enc380 progressive-budget recipe).
MODEL_SIZE="${MODEL_SIZE:-medium}"
PLANNER_SEED="${PLANNER_SEED:-0}"
TRAIN_UPDATES="${TRAIN_UPDATES:-30000}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
MILESTONE_INTERVAL="${MILESTONE_INTERVAL:-1000}"
case "${MODEL_SIZE}" in
    tiny)   default_micro=1024 ;;
    small)  default_micro=512 ;;
    medium) default_micro=256 ;;
    large)  default_micro=128 ;;
    *) echo "[ERROR] Unknown MODEL_SIZE=${MODEL_SIZE}." >&2; exit 2 ;;
esac
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-${default_micro}}"

# Closed-loop evaluation.
EVAL_ENVS="${EVAL_ENVS:-4096}"
EVAL_STEPS="${EVAL_STEPS:-500}"
EVAL_SEED="${EVAL_SEED:-0}"
EVAL_STRIDE="${EVAL_STRIDE:-2000}"
FALL_HEIGHT_M="${FALL_HEIGHT_M:-0.4}"

MODEL_SIZES_FOR_AGGREGATE="${MODEL_SIZES_FOR_AGGREGATE:-medium large}"

: "${ISAAC_PY:=pixi run -e isaaclab python}"
: "${PLAIN_PY:=pixi run python}"
read -r -a ISAAC_PY_ARR <<<"${ISAAC_PY}"
read -r -a PLAIN_PY_ARR <<<"${PLAIN_PY}"

SHARED_DIR="experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines"
DEMOS_DIR="${STUDY_ROOT}/oracle_demonstrations"
SAMPLES_DIR="${DEMOS_DIR}/rollout_training_samples"
RUN_TAG="planner_u${TRAIN_UPDATES}_b${BATCH_SIZE}"

KIT_QUIET=(--kit_args=--/app/extensions/fsWatcherEnabled=false)
NEWTON_ARGS=(
    physics=newton_mjwarp
    env.sim.physics.solver_cfg.njmax=320
    env.sim.physics.solver_cfg.nconmax=40
)
# Fixed start / no-corruption env contract shared by collect and eval. Domain
# randomization events are deliberately NOT overridden anywhere in this file:
# the eval protocol keeps training-time physical randomization and pushes.
ENV_COMMON=(
    agent.logger.backend=
    env.refresh_zarr_dataset=false
    env.wrap_steps=false
    env.random_reset_full_trajectory=false
    env.observations.policy.enable_corruption=false
)

verify_fb_checkpoint() {
    [[ -f "${FB_LOW_LEVEL_CHECKPOINT}" ]] || {
        echo "[ERROR] Missing FB tracker checkpoint: ${FB_LOW_LEVEL_CHECKPOINT}" >&2
        exit 2
    }
    local actual
    actual="$(sha256sum "${FB_LOW_LEVEL_CHECKPOINT}" | awk '{print $1}')"
    [[ "${actual}" == "${EXPECTED_FB_SHA256}" ]] || {
        echo "[ERROR] FB tracker SHA mismatch: ${actual} != ${EXPECTED_FB_SHA256}" >&2
        exit 2
    }
    echo "[PASS] FB tracker checkpoint verified (${EXPECTED_FB_SHA256:0:16}...)."
}

run_if_missing() {
    local marker="$1"
    shift
    if [[ -f "${marker}" ]]; then
        echo "[SKIP] ${marker}"
        return 0
    fi
    mkdir -p "$(dirname "${marker}")"
    printf '[CMD]'; printf ' %q' "$@"; printf '\n'
    "$@"
    [[ -f "${marker}" ]] || {
        echo "[ERROR] Stage completed without producing ${marker}." >&2
        exit 2
    }
}

stage_collect() {
    verify_fb_checkpoint
    run_if_missing "${SAMPLES_DIR}/sample_step_000000.pt" \
        "${ISAAC_PY_ARR[@]}" -m imitation_experiments.data.collect_interface_rollout_samples \
        --headless --device "${DEVICE}" \
        --task "${CHUNK_TASK}" --algorithm IPMD \
        --checkpoint "${FB_LOW_LEVEL_CHECKPOINT}" \
        --interface full_body_trajectory \
        --motion_name "${MOTION_NAME}" --motion_manifest "${MANIFEST}" \
        --planner_interval_steps 10 --command_future_steps 9 --command_past_steps 0 \
        --low_level_command_mode streamed_vanilla --state_history_steps 9 \
        --seed "${DEMO_SEED}" \
        --control_steps "${DEMO_STEPS}" --num_envs "${DEMO_ENVS}" \
        --reset_schedule sequential --reference_start_frame 0 \
        --disable_tracking_terminations --keep_configured_episode_length \
        --balanced_rows_per_motion "${DEMO_ROWS}" \
        --balanced_motion_names "${MOTION_NAME}" \
        --sample_rows_per_file "${DEMO_ROWS}" \
        --output_dir "${DEMOS_DIR}" \
        "env.episode_length_s=${DEMO_EPISODE_LENGTH_S}" \
        "${KIT_QUIET[@]}" "${ENV_COMMON[@]}" "${NEWTON_ARGS[@]}"
    sha256sum "${SAMPLES_DIR}/sample_step_000000.pt" \
        > "${DEMOS_DIR}/sample_sha256.txt"
    echo "[PASS] Demonstration rows: $(cat "${DEMOS_DIR}/sample_sha256.txt")"
}

stage_train() {
    local out="${STUDY_ROOT}/${MODEL_SIZE}/seed${PLANNER_SEED}/${RUN_TAG}"
    [[ -f "${SAMPLES_DIR}/sample_step_000000.pt" ]] || {
        echo "[ERROR] Missing demonstration samples: ${SAMPLES_DIR}" >&2
        exit 2
    }
    run_if_missing "${out}/checkpoints/best.pt" \
        "${PLAIN_PY_ARR[@]}" -m imitation_experiments.planner.train_chunked_transformer_planner \
        --samples_dir "${SAMPLES_DIR}" \
        --output_dir "${out}" \
        --interface full_body_trajectory \
        --planner_family flow \
        --state_key planner_state \
        --training_stage oracle \
        --device "${DEVICE}" \
        --seed "${PLANNER_SEED}" \
        --model_size "${MODEL_SIZE}" \
        --batch_size "${BATCH_SIZE}" \
        --micro_batch_size "${MICRO_BATCH_SIZE}" \
        --num_updates "${TRAIN_UPDATES}" \
        --milestone_interval "${MILESTONE_INTERVAL}" \
        --max_samples 0 \
        --lr 0.0001 --weight_decay 0.0001 \
        --flow_num_inference_steps 16 \
        --endpoint_num_inference_steps 4 \
        --flow_inference_noise_std 0.0
    local expected=$(( TRAIN_UPDATES / MILESTONE_INTERVAL ))
    local actual
    actual="$(find "${out}/checkpoints" -name 'update_*.pt' | wc -l)"
    [[ "${actual}" -eq "${expected}" ]] || {
        echo "[ERROR] Expected ${expected} milestone snapshots, found ${actual}." >&2
        exit 2
    }
    echo "[PASS] Trained ${MODEL_SIZE} with ${actual} milestone snapshots: ${out}"
}

eval_one() {
    local planner="$1" tag="$2"
    local out="${STUDY_ROOT}/${MODEL_SIZE}/seed${PLANNER_SEED}/eval_frame0_dr_baseonly_${EVAL_ENVS}env_seed${EVAL_SEED}/${tag}"
    [[ -f "${planner}" ]] || {
        echo "[ERROR] Missing planner checkpoint: ${planner}" >&2
        exit 2
    }
    run_if_missing "${out}/summary.json" \
        "${ISAAC_PY_ARR[@]}" -m imitation_experiments.evaluation.eval_interface_planner_closed_loop \
        --headless --device "${DEVICE}" \
        --task "${CHUNK_TASK}" --algorithm IPMD \
        --checkpoint "${FB_LOW_LEVEL_CHECKPOINT}" \
        --low_level_command_mode streamed_vanilla \
        --planner_checkpoint "${planner}" \
        --output_json "${out}/summary.json" \
        --pin_command_joint_order auto \
        --label "fb670_curve_${MODEL_SIZE}_${tag}" \
        --motion_manifest "${MANIFEST}" --motion_name "${MOTION_NAME}" \
        --num_envs "${EVAL_ENVS}" --steps "${EVAL_STEPS}" \
        --seed "${EVAL_SEED}" --state_history_steps 9 \
        --command_past_steps 0 --command_future_steps 9 \
        --planner_update_interval 10 \
        --flow_num_inference_steps 16 --flow_inference_noise_std 0.0 \
        --reset_schedule sequential --reference_start_frame 0 \
        --base_only_termination --fall_height_m "${FALL_HEIGHT_M}" \
        "${KIT_QUIET[@]}" \
        env.reference_start_frame=0 \
        env.random_reset_step_min=0 env.random_reset_step_max=0 \
        env.reset_schedule=sequential \
        "${ENV_COMMON[@]}" "${NEWTON_ARGS[@]}"
}

stage_eval() {
    verify_fb_checkpoint
    local ckpt_dir="${STUDY_ROOT}/${MODEL_SIZE}/seed${PLANNER_SEED}/${RUN_TAG}/checkpoints"
    [[ -d "${ckpt_dir}" ]] || {
        echo "[ERROR] Missing checkpoint dir: ${ckpt_dir}" >&2
        exit 2
    }
    if (( EVAL_STRIDE % MILESTONE_INTERVAL != 0 )); then
        echo "[ERROR] EVAL_STRIDE (${EVAL_STRIDE}) must be a multiple of MILESTONE_INTERVAL (${MILESTONE_INTERVAL})." >&2
        exit 2
    fi
    local update
    for (( update = EVAL_STRIDE; update <= TRAIN_UPDATES; update += EVAL_STRIDE )); do
        eval_one "$(printf '%s/update_%07d.pt' "${ckpt_dir}" "${update}")" \
            "$(printf 'update_%07d' "${update}")"
    done
    eval_one "${ckpt_dir}/best.pt" "best"
    echo "[PASS] Evaluated $(( TRAIN_UPDATES / EVAL_STRIDE )) milestones + best for ${MODEL_SIZE}."
}

stage_aggregate() {
    local sizes=()
    read -r -a sizes <<<"${MODEL_SIZES_FOR_AGGREGATE}"
    "${PLAIN_PY_ARR[@]}" \
        -m imitation_experiments.capacity.aggregate_planner_budget_curve \
        --study_root "${STUDY_ROOT}" \
        --interface full_body_trajectory \
        --sizes "${sizes[@]}" \
        --seed "${PLANNER_SEED}" \
        --run_tag "${RUN_TAG}" \
        --eval_dir_name "eval_frame0_dr_baseonly_${EVAL_ENVS}env_seed${EVAL_SEED}" \
        --expected_num_envs "${EVAL_ENVS}" \
        --expected_fall_height_m "${FALL_HEIGHT_M}" \
        --output_dir "${STUDY_ROOT}/budget_curve_summary"
}

for stage in ${STAGES}; do
    echo "[STAGE] ${stage} (MODEL_SIZE=${MODEL_SIZE})"
    case "${stage}" in
        collect)   stage_collect ;;
        train)     stage_train ;;
        eval)      stage_eval ;;
        aggregate) stage_aggregate ;;
        *) echo "[ERROR] Unknown stage '${stage}'." >&2; exit 2 ;;
    esac
done
echo "[DONE] ${STAGES}"
