#!/usr/bin/env bash
set -euo pipefail

# Channel-wise oracle-substitution ladder for the explicit (full-body) packet.
#
# DIAGNOSTIC ONLY. At each 5 Hz publication, selected channel groups of the
# planner's predicted 670-value packet are overwritten with the expert
# (ground-truth) values for exactly the frames that packet covers. The frozen
# tracker still receives a complete, well-formed 67-per-frame packet, so no
# low-level retraining is required and the streamed-vanilla equivalence
# certificate is untouched -- only each channel's PROVENANCE changes.
#
# Because a substituted row receives ground-truth future information, it is an
# UPPER BOUND on what the explicit interface could achieve, never a planner
# result. Interpretation is therefore one-directional:
#   * a substituted row that STILL loses to latent is a valid, strong result;
#   * a substituted row that WINS only localises which channel group to target
#     in a follow-up that retrains the planner on a reduced target.
#
# The ladder decomposes the measured FB gap (about 380 mm planner vs 23.8 mm
# oracle) into per-channel contributions, answering whether the explicit
# interface fails because of prediction difficulty (and where) or because of
# how predictions are anchored at publication.
#
# Usage:
#   DRY_RUN=1 experiments/campaigns/2026-07-23-lafan1-planner-capacity/run_oracle_substitution_ladder.sh
#   MODEL_SIZE=medium SEEDS="0 1 2" run_oracle_substitution_ladder.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || (cd "${SCRIPT_DIR}/../../.." && pwd))"
cd "${REPO_ROOT}"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/paths.env"

MODEL_SIZE="${MODEL_SIZE:-medium}"
SEEDS="${SEEDS:-0 1 2}"
# Planner stage directory suffix as written by run_capacity_point.sh:
# planner_pretrain (demonstration-only) or planner_finetune (rollout-finetuned).
STAGE="${STAGE:-finetune}"
EVAL_SEED="${EVAL_SEED:-0}"
DEVICE="${DEVICE:-cuda:0}"
DRY_RUN="${DRY_RUN:-0}"
EVAL_STEPS="${EVAL_STEPS:-700}"
FH_ENVS="${FH_ENVS:-4}"
STUDY_ROOT="${STUDY_ROOT:-logs/interface_baselines/lafan1_planner_capacity_20260723}"
LADDER_ROOT="${LADDER_ROOT:-${STUDY_ROOT}/oracle_substitution_ladder}"

# Channel groups to substitute, one eval each. "none" and "all" are already
# measured by the main sweep (planner row) and the oracle baseline, so the
# default ladder covers only the informative middle rungs.
# root_anchor is refused by the evaluator: substituted anchors do not land in
# the basis the tracker consumes (see ORACLE_SUBSTITUTION_VERIFIED_GROUPS).
VARIANTS="${VARIANTS:-qvel qpos}"

: "${ISAAC_PY:=pixi run -e isaaclab python}"
read -r -a ISAAC_PY_ARR <<<"${ISAAC_PY}"
NEWTON_ARGS=(physics=newton_mjwarp
    "env.sim.physics.solver_cfg.njmax=${NJMAX:-320}"
    "env.sim.physics.solver_cfg.nconmax=${NCONMAX:-40}")

VIDEO_ARGS=()
if [[ "${RENDER_VIDEO:-0}" == "1" && "${ASSERT_KITLESS:-0}" != "1" ]]; then
    VIDEO_ARGS=(--video --video_length "${VIDEO_STEPS:-150}")
fi
KIT_QUIET=(--kit_args=--/app/extensions/fsWatcherEnabled=false)

# Must stay byte-identical to the main sweep's protocol so the ladder rows are
# directly comparable to the already-measured none/all endpoints.
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
    planner="${STUDY_ROOT}/scaling/seed${seed}/${MODEL_SIZE}/full_body_trajectory/planner_${STAGE}/checkpoints/latest.pt"
    if [[ ! -e "${planner}" && "${DRY_RUN}" != "1" ]]; then
        echo "[ERROR] missing planner checkpoint: ${planner}" >&2; exit 2
    fi
    for variant in ${VARIANTS}; do
        out="${LADDER_ROOT}/seed${seed}/${MODEL_SIZE}/${STAGE}/subst_${variant//,/_}"
        run_if_missing "${out}/summary.json" \
            "${ISAAC_PY_ARR[@]}" -m imitation_experiments.evaluation.eval_interface_planner_closed_loop \
            --headless --device "${DEVICE}" --task "${CHUNK_TASK}" --algorithm IPMD \
            --checkpoint "${FBCHUNK_LOW_LEVEL_CHECKPOINT}" \
            --low_level_command_mode streamed_vanilla \
            --planner_checkpoint "${planner}" --output_json "${out}/summary.json" \
            --label "ladder_${MODEL_SIZE}_seed${seed}_${STAGE}_subst_${variant//,/_}" \
            --motion_manifest "${MANIFEST}" --motion_name "${MOTION_NAME}" \
            --num_envs "${FH_ENVS}" --steps "${EVAL_STEPS}" --seed "${EVAL_SEED}" \
            --state_history_steps 9 --command_past_steps 0 --command_future_steps 9 \
            --planner_update_interval 10 --flow_num_inference_steps 16 \
            --flow_inference_noise_std 0.0 --reset_schedule sequential \
            --reference_start_frame 0 --keep_after_done \
            --oracle_substitute "${variant}" \
            --pin_command_joint_order on \
            "${KIT_QUIET[@]}" "${VIDEO_ARGS[@]}" \
            agent.logger.backend= env.observations.policy.enable_corruption=false \
            "${NEWTON_ARGS[@]}" "${_FH_COMMON[@]}"
    done
done

echo "[PASS] oracle-substitution ladder complete under ${LADDER_ROOT}"
echo "[NOTE] every row here is an UPPER BOUND on the explicit interface."
