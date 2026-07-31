#!/usr/bin/env bash
set -euo pipefail

# BB1 shared-tracker route for the FB-670 budget curve: drive the FROZEN
# LATENT tracker from the already-trained FB-670 planner's predictions,
# routed through the FROZEN DEFAULT skill encoder (the one that produced the
# latent_skill oracle -- it was fit on the same 670-value full-body macro
# state the FB planner predicts, so no new encoder is needed):
#
#   FB-670 planner -> 670 packet -> [frozen DEFAULT encoder] -> z258 -> latent tracker
#
# This isolates the TRACKER from the INTERFACE: the FB planner never changes,
# only which low-level policy consumes its prediction. If this route survives
# far better than "FB planner -> its own FB tracker" (the fb670_budget_curve
# results), the FB tracker itself is brittle to planner error, independent of
# whether full-body is a harder *planning* problem. No retraining: reuses the
# milestone checkpoints already produced by run_fb670_budget_curve.sh and the
# frozen latent tracker + encoder already used by run_latent_budget_curve.sh.
#
# Stages:
#   pin        one packet_source=expert cell per size -- feeds the TRUE oracle
#              packet (not a planner prediction) through the same encoder path.
#              Must show ~0 falls; a real defect here (permutation, wrong
#              normalization, wrong frame split) is caught before any planner
#              row is spent. See packet_to_latent_command.py's module docstring
#              for the two known traps (term-major vs frame-interleaved,
#              raw vs normalized features).
#   eval       every 2k-update FB milestone + best.pt, packet_source=planner,
#              same rigorous protocol as the two existing budget curves:
#              frame-0, training-time DR and pushes ACTIVE, base_too_low-only
#              termination, 4096 envs, 500 steps.
#   aggregate  MPJPE / survival / fall-count vs updates, tagged as a distinct
#              interface so it never conflates with the direct FB-670 curve.

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

STAGES="${STAGES:-pin eval aggregate}"
FB670_STUDY_ROOT="${FB670_STUDY_ROOT:-logs/interface_baselines/lafan1_fb670_budget_curve_20260730}"
STUDY_ROOT="${STUDY_ROOT:-logs/interface_baselines/lafan1_fb670_via_latent_tracker_20260730}"
MOTION_NAME="${MOTION_NAME:-walk1_subject1}"
MANIFEST="${MANIFEST:-data/lafan1/manifests/g1_lafan1_manifest.json}"
LATENT_DATASET_PATH="${LATENT_DATASET_PATH:-data/lafan1/zarr/latent_walk1_subject1_corrected_8e95d557}"
LATENT_TASK="${LATENT_TASK:-Isaac-Imitation-G1-Latent-Strict-v0}"
DEVICE="${DEVICE:-cuda:0}"

LATENT_LOW_LEVEL_CHECKPOINT="${LATENT_LOW_LEVEL_CHECKPOINT:-logs/downloaded_checkpoints/lafan1_latent_deterministic_5b_seed0/model_step_4525129728.pt}"
LATENT_SKILL_CHECKPOINT="${LATENT_SKILL_CHECKPOINT:-logs/downloaded_checkpoints/lafan1_latent_deterministic_5b_seed0/skill_encoder/latest.pt}"
EXPECTED_LATENT_SHA256="${EXPECTED_LATENT_SHA256:-785f5327f2356f4a301ac39fc435b78379e9c5a73293c450deb483dd7c188f7c}"
EXPECTED_SKILL_SHA256="${EXPECTED_SKILL_SHA256:-5c84ff7261c5a3aca732e370ca39f889d68a5d39fb498fa9fde72c653eb264ea}"

MODEL_SIZE="${MODEL_SIZE:-medium}"
PLANNER_SEED="${PLANNER_SEED:-0}"
TRAIN_UPDATES="${TRAIN_UPDATES:-30000}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
MILESTONE_INTERVAL="${MILESTONE_INTERVAL:-1000}"
EVAL_STRIDE="${EVAL_STRIDE:-2000}"

EVAL_ENVS="${EVAL_ENVS:-4096}"
EVAL_STEPS="${EVAL_STEPS:-500}"
EVAL_SEED="${EVAL_SEED:-0}"
FALL_HEIGHT_M="${FALL_HEIGHT_M:-0.4}"
ASSERT_KITLESS="${ASSERT_KITLESS:-0}"
MODEL_SIZES_FOR_AGGREGATE="${MODEL_SIZES_FOR_AGGREGATE:-medium large}"
FB670_RUN_TAG="${FB670_RUN_TAG:-planner_u${TRAIN_UPDATES}_b${BATCH_SIZE}}"

: "${ISAAC_PY:=pixi run -e isaaclab python}"
: "${PLAIN_PY:=pixi run python}"
read -r -a ISAAC_PY_ARR <<<"${ISAAC_PY}"
read -r -a PLAIN_PY_ARR <<<"${PLAIN_PY}"

SHARED_DIR="experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines"

KITLESS=()
[[ "${ASSERT_KITLESS}" == "1" ]] && KITLESS=(--assert-kitless)
KIT_QUIET=(--kit_args=--/app/extensions/fsWatcherEnabled=false)
NEWTON_ARGS=(
    physics=newton_mjwarp
    env.sim.physics.solver_cfg.njmax=320
    env.sim.physics.solver_cfg.nconmax=40
)
LATENT_CMD=(
    env.latent_command_dim=258 agent.ipmd.latent_dim=258
    agent.ipmd.hl_skill_horizon_steps=10 agent.ipmd.hl_skill_command_mode=z
    agent.ipmd.latent_steps_min=10 agent.ipmd.latent_steps_max=10
    agent.ipmd.latent_learning.command_phase_mode=sin_cos
    agent.ipmd.latent_learning.code_latent_dim=256
    agent.ipmd.latent_learning.code_period=10
)
LATENT_REWARD_ZEROS=(
    agent.ipmd.reward_loss_coeff=0.0 agent.ipmd.reward_l2_coeff=0.0
    agent.ipmd.reward_grad_penalty_coeff=0.0 agent.ipmd.reward_logit_reg_coeff=0.0
    agent.ipmd.reward_param_weight_decay_coeff=0.0
)
# DR events are deliberately NOT overridden: the protocol keeps training-time
# physical randomization and interval pushes active, matching both existing
# curves exactly.
ENV_COMMON=(
    agent.logger.backend=
    env.refresh_zarr_dataset=false
    env.wrap_steps=false
    env.random_reset_full_trajectory=false
    env.reset_schedule=sequential
    env.observations.policy.enable_corruption=false
)

verify_checkpoints() {
    local path sha
    for path in "${LATENT_LOW_LEVEL_CHECKPOINT}" "${LATENT_SKILL_CHECKPOINT}"; do
        [[ -f "${path}" ]] || {
            echo "[ERROR] Missing latent checkpoint: ${path}" >&2
            exit 2
        }
    done
    sha="$(sha256sum "${LATENT_LOW_LEVEL_CHECKPOINT}" | awk '{print $1}')"
    [[ "${sha}" == "${EXPECTED_LATENT_SHA256}" ]] || {
        echo "[ERROR] Latent tracker SHA mismatch: ${sha} != ${EXPECTED_LATENT_SHA256}" >&2
        exit 2
    }
    sha="$(sha256sum "${LATENT_SKILL_CHECKPOINT}" | awk '{print $1}')"
    [[ "${sha}" == "${EXPECTED_SKILL_SHA256}" ]] || {
        echo "[ERROR] Skill encoder SHA mismatch: ${sha} != ${EXPECTED_SKILL_SHA256}" >&2
        exit 2
    }
    echo "[PASS] Latent tracker + encoder verified."
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

route_eval() {
    local packet_source="$1" planner="$2" out="$3"
    "${ISAAC_PY_ARR[@]}" scripts/rlopt/eval_skill_commander_closed_loop.py \
        "${KITLESS[@]}" \
        --headless --device "${DEVICE}" \
        --task "${LATENT_TASK}" --algorithm IPMD \
        --checkpoint "${LATENT_LOW_LEVEL_CHECKPOINT}" \
        --skill_checkpoint "${LATENT_SKILL_CHECKPOINT}" \
        --motion_name "${MOTION_NAME}" --state_history_steps 9 --metric_interval 10 \
        --flow_num_inference_steps 16 --flow_inference_noise_std 0.0 \
        --packet_planner_checkpoint "${planner}" \
        --packet_interface full_body_trajectory --packet_source "${packet_source}" \
        --seed "${EVAL_SEED}" --num_envs "${EVAL_ENVS}" --max_steps "${EVAL_STEPS}" \
        --output_dir "${out}" --label "fb670_via_latent_${packet_source}" \
        --base_only_termination --fall_height_m "${FALL_HEIGHT_M}" \
        --extend_episode_length_for_max_steps --disable_reward_clipping \
        "${KIT_QUIET[@]}" \
        agent.ipmd.command_source=hl_skill \
        "agent.ipmd.hl_skill_checkpoint_path=${LATENT_SKILL_CHECKPOINT}" \
        agent.ipmd.hl_skill_finetune_enabled=false \
        "env.lafan1_manifest_path=${MANIFEST}" \
        "env.dataset_path=${LATENT_DATASET_PATH}" \
        env.reference_start_frame=0 \
        env.random_reset_step_min=0 env.random_reset_step_max=0 \
        "${LATENT_CMD[@]}" "${LATENT_REWARD_ZEROS[@]}" \
        "${ENV_COMMON[@]}" "${NEWTON_ARGS[@]}"
}

stage_pin() {
    verify_checkpoints
    # Any FB planner checkpoint works as the metadata carrier -- packet_source
    # =expert substitutes the true oracle packet regardless of what the
    # planner predicts, so this validates only the encoder/tracker path.
    local planner="${FB670_STUDY_ROOT}/${MODEL_SIZE}/seed${PLANNER_SEED}/${FB670_RUN_TAG}/checkpoints/best.pt"
    [[ -f "${planner}" ]] || {
        echo "[ERROR] Missing FB-670 planner (needed only for metadata): ${planner}" >&2
        exit 2
    }
    local out="${STUDY_ROOT}/pin_${MODEL_SIZE}"
    run_if_missing "${out}/summary.json" route_eval expert "${planner}" "${out}"
    "${PLAIN_PY_ARR[@]}" - "${out}/summary.json" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1]))
packet = summary["metadata"].get("packet_encoder_command", {})
checks = {
    "packet_source==expert": packet.get("packet_source") == "expert",
    "packet_interface==full_body_trajectory": packet.get("packet_interface") == "full_body_trajectory",
    "packet_target_dim==670": int(packet.get("packet_target_dim", -1)) == 670,
    "encoder_input_width==670": int(packet.get("encoder_input_width", -1)) == 670,
    "layout_verified": packet.get("layout_verified") is True,
    "expert_pin_latent_mse<1e-8": float(packet.get("expert_pin_latent_mse", float("inf"))) < 1e-8,
}
falls = summary["aggregate"].get("termination_cause_env_counts", {}).get("base_too_low", -1)
checks["near_zero_falls"] = falls < int(0.02 * summary["metadata"]["num_envs"])
failed = [k for k, ok in checks.items() if not ok]
mpjpe = summary["metrics"]["tracking_mpjpe_mm"]["mean"]
print(f"[PIN] fall_events={falls} mpjpe_mm={mpjpe:.2f} checks={checks}")
if failed:
    raise SystemExit(f"[ERROR] Pin test failed: {failed}")
print("[PASS] Pin test: encoder+tracker path reproduces the oracle command exactly.")
PY
}

eval_one() {
    local planner="$1" tag="$2"
    local out="${STUDY_ROOT}/${MODEL_SIZE}/seed${PLANNER_SEED}/eval_frame0_dr_baseonly_${EVAL_ENVS}env_seed${EVAL_SEED}/${tag}"
    [[ -f "${planner}" ]] || {
        echo "[ERROR] Missing FB-670 planner checkpoint: ${planner}" >&2
        exit 2
    }
    run_if_missing "${out}/summary.json" route_eval planner "${planner}" "${out}"
}

stage_eval() {
    verify_checkpoints
    local ckpt_dir="${FB670_STUDY_ROOT}/${MODEL_SIZE}/seed${PLANNER_SEED}/${FB670_RUN_TAG}/checkpoints"
    [[ -d "${ckpt_dir}" ]] || {
        echo "[ERROR] Missing FB-670 checkpoint dir: ${ckpt_dir}" >&2
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
        --train_root "${FB670_STUDY_ROOT}" \
        --interface full_body_trajectory_via_latent_tracker \
        --sizes "${sizes[@]}" \
        --seed "${PLANNER_SEED}" \
        --run_tag "${FB670_RUN_TAG}" \
        --eval_dir_name "eval_frame0_dr_baseonly_${EVAL_ENVS}env_seed${EVAL_SEED}" \
        --expected_num_envs "${EVAL_ENVS}" \
        --expected_fall_height_m "${FALL_HEIGHT_M}" \
        --output_dir "${STUDY_ROOT}/budget_curve_summary"
}

for stage in ${STAGES}; do
    echo "[STAGE] ${stage} (MODEL_SIZE=${MODEL_SIZE})"
    case "${stage}" in
        pin)       stage_pin ;;
        eval)      stage_eval ;;
        aggregate) stage_aggregate ;;
        *) echo "[ERROR] Unknown stage '${stage}'." >&2; exit 2 ;;
    esac
done
echo "[DONE] ${STAGES}"
