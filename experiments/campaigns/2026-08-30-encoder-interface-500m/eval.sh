#!/usr/bin/env bash
# Score encoder-interface-500m checkpoints on `bones_testbed4096_v1`, the
# population behind the `paper_testbed4096_v1` profile.
#
# Unlike the smooth-ablation launcher, this campaign varies the ENCODER, so
# two overrides move per arm and everything else is held fixed:
#
#   `agent.ipmd.hl_skill_checkpoint_path`   the encoder the tracker was bound to
#   `agent.ipmd.hl_skill_horizon_steps`     that encoder's pretrain horizon
#
# A wrong pair fails loudly: FrozenHighLevelSkillCommandSampler raises on a
# horizon mismatch rather than feeding the encoder a wrong-width window.
#
# The probe encoders stay on ICE and are streamed one at a time, because they
# are 0.74 to 1.26 GB each and the workstation pool holds about 16 GB. Each is
# deleted after its arm is scored; set PRUNE_ENCODER=0 to keep them.
#
#   ./eval.sh                          # every mirrored arm, clean + robust
#   ARMS=prod ROWS=clean ./eval.sh
#   ./eval.sh --report
set -uo pipefail
CAMPAIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${CAMPAIGN_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

MIRROR="${MIRROR:-${REPO_ROOT}/logs/encoder_interface_500m_mirror}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/logs/encoder_interface_500m_eval}"
ENCODER_CACHE="${ENCODER_CACHE:-${MIRROR}/encoders}"
PROD_ENCODER="${PROD_ENCODER:-${REPO_ROOT}/logs/pareto_stack_mirror/diffntp_chunk_h1_ee_wide_seed0/encoder/checkpoints/latest.pt}"
REMOTE_HOST="${REMOTE_HOST:-ice}"
REMOTE_ENCODER_ROOT="${REMOTE_ENCODER_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/data/endpoint_collapse_probe}"
PRUNE_ENCODER="${PRUNE_ENCODER:-1}"
REFERENCE_ARRAYS="${REFERENCE_ARRAYS:-/mnt/hsstorage/fwu91/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
PERSIST_ID="${PERSIST_ID:-bones_seed_sonic_full_129785@e714bbff}"
MAX_STEPS="${MAX_STEPS:-10000}"
# Per-row watchdog. A row normally finishes in about two minutes; on
# 2026-08-30 one row wedged in Isaac Lab startup and produced nothing for over
# two hours, which killed the rest of the sweep. Bound each row instead.
ROW_TIMEOUT="${ROW_TIMEOUT:-1800}"
SEED="${SEED:-0}"
ROWS="${ROWS:-clean robust}"
# The deciding 4,096-clip board. Rows from a different board are a different
# population and must never share this directory or a table column.
BOARD="${BOARD:-bones_testbed4096_v1}"
ARMS="${ARMS:-prod suffix1 suffix2 suffix5 suffix9 h1 h2 h5 h10}"
SCALED_CELLS="[2048,2048,1024,1024,512,512]"
RUNTIME_BODY_NAMES="[pelvis,left_hip_roll_link,left_knee_link,left_ankle_roll_link,right_hip_roll_link,right_knee_link,right_ankle_roll_link,torso_link,left_shoulder_roll_link,left_elbow_link,left_wrist_yaw_link,right_shoulder_roll_link,right_elbow_link,right_wrist_yaw_link]"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
randomization_for() { [[ "$1" == "robust" ]] && echo no_push || echo none; }
out_for() { printf '%s/%s_seed%s_%s_f%s.json' "${OUTPUT_ROOT}" "$1" "$2" "$3" "$4"; }

# Each arm's pretrain horizon. The suffix family keeps horizon 10 and moves
# only how much of the window the encoder sees; the horizon family shrinks the
# horizon itself, so the encoder input and both pretrain targets move together.
horizon_for() {
    case "$1" in
        h1) echo 1 ;;
        h2) echo 2 ;;
        h5) echo 5 ;;
        *) echo 10 ;;
    esac
}

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
mkdir -p "${OUTPUT_ROOT}" "${ENCODER_CACHE}"

mapfile -t ranks < <(pixi run python -c "
from imitation_experiments.evaluation.protocol import BOARDS
print('\n'.join(str(case.trajectory_rank) for case in BOARDS['${BOARD}'].cases))
")
[[ "${#ranks[@]}" -gt 0 ]] || { log "[FATAL] board returned no ranks"; exit 2; }

for arm in ${ARMS}; do
    tree="${MIRROR}/${arm}_seed${SEED}"
    [[ -d "${tree}/tracker" ]] || { log "[SKIP] no mirror ${tree}"; continue; }
    mapfile -t frames < <(ls -1 "${tree}/tracker" 2>/dev/null | sed -n 's/^f\([0-9]\+\)$/\1/p' | sort -n)
    [[ "${#frames[@]}" -gt 0 ]] || { log "[SKIP] no checkpoints in ${tree}/tracker"; continue; }
    final="${frames[-1]}"
    horizon="$(horizon_for "${arm}")"

    # `prod` binds the round-4 pareto encoder already on this workstation.
    # Every other arm binds a probe encoder that lives on ICE.
    streamed=0
    if [[ "${arm}" == "prod" ]]; then
        encoder="${PROD_ENCODER}"
    else
        encoder="${ENCODER_CACHE}/${arm}_seed${SEED}_latest.pt"
        if [[ ! -s "${encoder}" ]]; then
            log "[pull] ${arm} encoder"
            if ! rsync -a --partial \
                "${REMOTE_HOST}:${REMOTE_ENCODER_ROOT}/${arm}_seed${SEED}/encoder/checkpoints/latest.pt" \
                "${encoder}"; then
                log "[FAIL] ${arm}: encoder pull failed"
                rm -f "${encoder}"
                continue
            fi
            streamed=1
        fi
    fi
    [[ -s "${encoder}" ]] || { log "[FAIL] ${arm}: encoder missing ${encoder}"; continue; }

    for row in ${ROWS}; do
        profile="$(randomization_for "${row}")"
        checkpoint="${tree}/tracker/f${final}/models/model_step_${final}.pt"
        out="$(out_for "${arm}" "${SEED}" "${row}" "${final}")"
        [[ -s "${out}" ]] && { log "[SKIP] already scored $(basename "${out}")"; continue; }

        log "${arm} seed${SEED} ${row} f${final} horizon ${horizon} (${#ranks[@]} clips)"
        timeout "${ROW_TIMEOUT}" \
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
            "agent.ipmd.hl_skill_checkpoint_path=${encoder}" \
            "agent.ipmd.hl_skill_horizon_steps=${horizon}" \
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
            agent.value_function.activation_fn=silu > "${out}.log" 2>&1
        rc=$?
        if (( rc == 124 )); then
            log "[TIMEOUT] ${arm} ${row} f${final}: no row after ${ROW_TIMEOUT}s"
            continue
        fi
        if (( rc != 0 )); then
            log "[FAIL] ${arm} ${row} f${final} exit ${rc}: $(tail -3 "${out}.log" | tr '\n' ' ' | cut -c1-150)"
            continue
        fi
        if [[ ! -s "${out}" ]]; then
            # Report the FIRST hard error, not the last line: an OOM is
            # followed by a teardown `AttributeError: '_is_closed'` that hides
            # the real cause (2026-08-30).
            log "[FAIL] ${arm} ${row} f${final}: exit 0 but no row written: $(grep -iE 'outofmemory|out of memory|AcceleratorError|RuntimeError' "${out}.log" | head -1 | cut -c1-160)"
            continue
        fi
        log "[OK] $(basename "${out}")"
    done

    if [[ "${PRUNE_ENCODER}" == "1" && "${streamed}" == "1" ]]; then
        rm -f "${encoder}"
        log "[prune] ${arm} encoder removed"
    fi
done

report
