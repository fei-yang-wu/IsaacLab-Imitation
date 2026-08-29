#!/usr/bin/env bash
# Score smooth-ablation-5b checkpoints on `bones_testbed4096_v1`.
#
# Every arm shares ONE interface (the campaign varies rewards and the
# exploration contract, never the command channel), so unlike the pareto
# launcher this one carries a single fixed override set: 258-D hold-1
# `sin_cos`, `root_qpos` macro state, the frozen `diffntp_chunk_h1_ee_wide`
# encoder. Evaluation is deterministic (`--action_sampling mode`), so the
# `sigma` arm's clamped exploration contract needs no eval-side override.
#
# Requires the 2026-08-29 evaluator (commit 85141ff): rows carry the
# reference-free `body_jerk_mps3` / `action_delta_l2` smoothness metrics.
#
#   ./eval.sh                          # every mirrored arm, clean + robust
#   ARMS=base ROWS=clean ./eval.sh
#   ./eval.sh --report
set -uo pipefail
CAMPAIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${CAMPAIGN_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

MIRROR="${MIRROR:-${REPO_ROOT}/logs/smooth_ablation_5b_mirror}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/logs/smooth_ablation_5b_eval}"
ENCODER="${ENCODER:-${REPO_ROOT}/logs/pareto_stack_mirror/diffntp_chunk_h1_ee_wide_seed0/encoder/checkpoints/latest.pt}"
REFERENCE_ARRAYS="${REFERENCE_ARRAYS:-/mnt/hsstorage/fwu91/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
PERSIST_ID="${PERSIST_ID:-bones_seed_sonic_full_129785@e714bbff}"
MAX_STEPS="${MAX_STEPS:-10000}"
SEED="${SEED:-0}"
ROWS="${ROWS:-clean robust}"
ARMS="${ARMS:-base energy sigma feetacc_weak ar0 hist ema lcp lstm}"
SCALED_CELLS="[2048,2048,1024,1024,512,512]"
RUNTIME_BODY_NAMES="[pelvis,left_hip_roll_link,left_knee_link,left_ankle_roll_link,right_hip_roll_link,right_knee_link,right_ankle_roll_link,torso_link,left_shoulder_roll_link,left_elbow_link,left_wrist_yaw_link,right_shoulder_roll_link,right_elbow_link,right_wrist_yaw_link]"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
randomization_for() { [[ "$1" == "robust" ]] && echo no_push || echo none; }
out_for() { printf '%s/%s_seed%s_%s_f%s.json' "${OUTPUT_ROOT}" "$1" "$2" "$3" "$4"; }

report() {
    shopt -s nullglob
    local files=("${OUTPUT_ROOT}"/*.json)
    [[ "${#files[@]}" -gt 0 ]] || { log "[INFO] nothing scored yet"; return 0; }
    for out in "${files[@]}"; do
        printf '%-52s ' "$(basename "${out}" .json)"
        pixi run python -m imitation_experiments.evaluation.summarize_paper_boards "${out}"
    done
}

if [[ "${1:-}" == "--report" ]]; then
    report
    exit $?
fi

[[ -s "${REFERENCE_ARRAYS}/reference_arrays_manifest.json" ]] || {
    log "[FATAL] reference arrays missing: ${REFERENCE_ARRAYS}"; exit 2; }
[[ -s "${ENCODER}" ]] || { log "[FATAL] encoder missing: ${ENCODER}"; exit 2; }
mkdir -p "${OUTPUT_ROOT}"

mapfile -t ranks < <(pixi run python -c "
from imitation_experiments.evaluation.protocol import BOARDS
print('\n'.join(str(case.trajectory_rank) for case in BOARDS['bones_testbed4096_v1'].cases))
")
[[ "${#ranks[@]}" -gt 0 ]] || { log "[FATAL] board returned no ranks"; exit 2; }

for arm in ${ARMS}; do
    tree="${MIRROR}/${arm}_seed${SEED}"
    [[ -d "${tree}/tracker" ]] || { log "[SKIP] no mirror ${tree}"; continue; }
    mapfile -t frames < <(ls -1 "${tree}/tracker" 2>/dev/null | sed -n 's/^f\([0-9]\+\)$/\1/p' | sort -n)
    [[ "${#frames[@]}" -gt 0 ]] || { log "[SKIP] no checkpoints in ${tree}/tracker"; continue; }
    final="${frames[-1]}"

    # The `hist` arm widens the actor input with ten-step histories; the
    # policy restore is strict, so its eval MUST repeat those overrides.
    # The `ema` arm's filter lives in the ENV action term, so its eval must
    # repeat the alpha or it would score an unfiltered policy.
    arm_extra=()
    if [[ "${arm}" == "hist" ]]; then
        arm_extra=(
            env.observations.policy.projected_gravity.history_length=10
            env.observations.policy.base_ang_vel.history_length=10
            env.observations.policy.joint_pos_rel.history_length=10
            env.observations.policy.joint_vel_rel.history_length=10
            env.observations.policy.last_action.history_length=10
        )
    elif [[ "${arm}" == "ema" ]]; then
        arm_extra=(env.actions.joint_pos.ema_alpha=0.65)
    elif [[ "${arm}" == "lstm" ]]; then
        # The LSTM lives in the actor; the strict restore needs the same
        # architecture flags. NOTE: recurrent-state handling in
        # evaluate_checkpoint's step loop must be qualified before this
        # arm's rows are cited.
        arm_extra=(agent.ppo.rnn_hidden_size=256)
    fi

    for row in ${ROWS}; do
        profile="$(randomization_for "${row}")"
        checkpoint="${tree}/tracker/f${final}/models/model_step_${final}.pt"
        out="$(out_for "${arm}" "${SEED}" "${row}" "${final}")"
        [[ -s "${out}" ]] && { log "[SKIP] already scored $(basename "${out}")"; continue; }

        log "${arm} seed${SEED} ${row} f${final} (${#ranks[@]} clips)"
        env TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 \
            HYDRA_FULL_ERROR=1 TORCHDYNAMO_DISABLE=1 \
            pixi run -e isaaclab python -u \
            -m imitation_experiments.lowlevel.evaluate_checkpoint \
            --checkpoint "${checkpoint}" \
            --output_json "${out}" \
            --label "${arm}_seed${SEED}_${row}_f${final}" \
            --task Isaac-Imitation-G1-v2 --algo IPMD \
            --agent_entry_point rlopt_ipmd_tuned_cfg_entry_point \
            --randomization "${profile}" --action_sampling mode \
            --num_envs "${#ranks[@]}" --steps "${MAX_STEPS}" --seed 0 \
            --reference_start_frame 0 --reset_schedule sequential \
            --trajectory_ranks "${ranks[@]}" \
            --headless \
            --kit_args=--/app/extensions/fsWatcherEnabled=false \
            --skill_encoder_source pretrained \
            physics=newton_mjwarp \
            env.sim.physics.solver_cfg.njmax=320 \
            env.sim.physics.solver_cfg.nconmax=200 \
            env.events.push_robot=null \
            env.data.manifest=null \
            "env.data.reference_arrays_dir=${REFERENCE_ARRAYS}" \
            "env.data.persist_id=${PERSIST_ID}" \
            env.data.reference_arrays_resident=false \
            env.data.reference_arrays_warm_workers=8 \
            env.data.runtime_cache_device=cuda:0 \
            env.data.reference_prefetch_mode=off \
            env.data.macro_cache_device=cuda:0 \
            "env.data.runtime_cache_body_names=${RUNTIME_BODY_NAMES}" \
            env.command_interface.actor=latent \
            env.command_interface.actor.dim=258 \
            env.command_interface.encoder=single \
            agent.ipmd.latent_dim=258 \
            agent.ipmd.command_source=hl_skill \
            "agent.ipmd.hl_skill_checkpoint_path=${ENCODER}" \
            agent.ipmd.hl_skill_horizon_steps=10 \
            agent.ipmd.hl_skill_command_mode=z \
            agent.ipmd.latent_steps_min=1 \
            agent.ipmd.latent_steps_max=1 \
            agent.ipmd.latent_learning.code_period=1 \
            agent.ipmd.latent_learning.command_phase_mode=sin_cos \
            agent.ipmd.latent_learning.command_phase_source=hold \
            agent.ipmd.latent_learning.command_phase_period=0 \
            agent.ipmd.latent_learning.code_latent_dim=256 \
            agent.ipmd.hl_skill_finetune_enabled=false \
            "env.expert_macro_state_terms=[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b]" \
            env.expert_macro_frame_stride=1 \
            env.expert_macro_anchor_mode=robot_heading \
            env.terminations.anchor_pos.params.threshold=0.25 \
            env.terminations.anchor_pos.params.down_threshold=0.25 \
            env.terminations.anchor_ori.params.threshold=1.0 \
            env.terminations.ee_body_pos.params.threshold=0.25 \
            env.terminations.ee_body_pos.params.down_threshold=0.25 \
            env.terminations.foot_pos_xyz=null \
            env.terminations.base_too_low=null \
            "agent.policy.num_cells=${SCALED_CELLS}" \
            agent.policy.activation_fn=silu \
            "agent.value_function.num_cells=${SCALED_CELLS}" \
            agent.value_function.activation_fn=silu \
            ${arm_extra[@]+"${arm_extra[@]}"} > "${out}.log" 2>&1
        rc=$?
        if (( rc != 0 )); then
            log "[FAIL] ${arm} ${row} f${final} exit ${rc}: $(tail -3 "${out}.log" | tr '\n' ' ' | cut -c1-150)"
            continue
        fi
        if [[ ! -s "${out}" ]]; then
            log "[FAIL] ${arm} ${row} f${final}: exit 0 but no row written: $(grep -iE 'error|out of memory' "${out}.log" | tail -1)"
            continue
        fi
        log "[OK] $(basename "${out}")"
    done
done

report
