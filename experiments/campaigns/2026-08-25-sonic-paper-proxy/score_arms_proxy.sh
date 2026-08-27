#!/usr/bin/env bash
set -uo pipefail

# Score our own latent trackers on the SONIC-paper-facing proxy population,
# beside the released-checkpoint row from `run_proxy_rows.sh`.
#
# This board is a CALIBRATION board: it exists so a row can name the SONIC
# Table 2 column it faces. The comparison board for ranking our arms against
# each other is still `bones_testbed4096_v1`
# (`2026-08-17-paper-metric-canon/score_latent_arms_testbed.sh`). Use this one
# only to put an arm next to a SONIC paper number.
#
# Ranks come from the registry, never a copied literal.
#
#   ./score_arms_proxy.sh
#   ARMS="ln_hold1_sonicreset_30b" ROWS="clean" ./score_arms_proxy.sh
#   ./score_arms_proxy.sh --report

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

MIRROR="${MIRROR:-${REPO_ROOT}/logs/bottleneck_10b_mirror}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/logs/sonic_paper_proxy}"
REFERENCE_ARRAYS="${REFERENCE_ARRAYS:-/mnt/hsstorage/fwu91/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
PERSIST_ID="${PERSIST_ID:-bones_seed_sonic_full_129785@e714bbff}"
MAX_STEPS="${MAX_STEPS:-10000}"
NUM_ENVS="${NUM_ENVS:-4096}"
BOARD="${BOARD:-sonic_proxy_testrep4096_v1}"

# Every arm below trains the scaled actor/critic; the agent entry point's
# default is smaller, so omitting this fails the strict state-dict restore.
SCALED_CELLS="[2048,2048,1024,1024,512,512]"
RUNTIME_BODY_NAMES="[pelvis,left_hip_roll_link,left_knee_link,left_ankle_roll_link,right_hip_roll_link,right_knee_link,right_ankle_roll_link,torso_link,left_shoulder_roll_link,left_elbow_link,left_wrist_yaw_link,right_shoulder_roll_link,right_elbow_link,right_wrist_yaw_link]"

# label | mirror arm | frames | z_dim | command_dim | hold
# `ln_hold1_sonicreset` at 30B is the leader of the legacy scoreboard
# (0.9707 SR / 21.75 mm). Its encoder file is byte-identical to its base
# `cont_det_ln_hold1` encoder (SHA-256 `ef9678ed…`), so the frozen binding the
# 30B chain declares is the one scored here.
ARMS_TABLE=(
"ln_hold1_sonicreset_30b|ln_hold1_sonicreset|30000021504|256|258|1"
"cont_det_ln_hold1_10b|cont_det_ln_hold1|10000269312|256|258|1"
)
ARMS="${ARMS:-ln_hold1_sonicreset_30b}"
ROWS="${ROWS:-clean robust}"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
profile_for() { [[ "$1" == "clean" ]] && echo none || echo no_push; }
out_for() { printf '%s/%s_%s_rand_%s.json' "${OUTPUT_ROOT}" "$1" "${BOARD}" "$(profile_for "$2")"; }

report() {
    local arm row out
    for arm in ${ARMS}; do
        for row in ${ROWS}; do
            out="$(out_for "${arm}" "${row}")"
            [[ -s "${out}" ]] || { log "[MISSING] ${arm} ${row}"; continue; }
            printf '%-28s %-7s ' "${arm}" "${row}"
            pixi run python -m imitation_experiments.evaluation.summarize_paper_boards "${out}"
        done
    done
}

if [[ "${1:-}" == "--report" ]]; then
    report
    exit $?
fi

[[ -s "${REFERENCE_ARRAYS}/reference_arrays_manifest.json" ]] || {
    log "[FATAL] reference arrays missing: ${REFERENCE_ARRAYS}"
    exit 2
}
mkdir -p "${OUTPUT_ROOT}"

mapfile -t RANKS < <(pixi run python -c "
from imitation_experiments.evaluation.protocol import BOARDS
for case in BOARDS['${BOARD}'].cases:
    print(case.trajectory_rank)
") || { log "[FATAL] could not read board ${BOARD}"; exit 3; }
[[ "${#RANKS[@]}" -eq "${NUM_ENVS}" ]] || {
    log "[FATAL] board ${BOARD} holds ${#RANKS[@]} cases, expected ${NUM_ENVS}"
    exit 3
}

for arm in ${ARMS}; do
    entry=""
    for candidate in "${ARMS_TABLE[@]}"; do
        [[ "${candidate%%|*}" == "${arm}" ]] && entry="${candidate}"
    done
    [[ -n "${entry}" ]] || { log "[SKIP] unknown arm ${arm}"; continue; }
    IFS='|' read -r _ mirror_arm frames z_dim command_dim hold <<<"${entry}"

    checkpoint="${MIRROR}/${mirror_arm}_seed0/tracker/f${frames}/models/model_step_${frames}.pt"
    encoder="${MIRROR}/${mirror_arm}_seed0/encoder/checkpoints/latest.pt"
    [[ -s "${checkpoint}" ]] || { log "[SKIP] no checkpoint ${checkpoint}"; continue; }
    [[ -s "${encoder}" ]] || { log "[SKIP] no encoder ${encoder}"; continue; }

    for row in ${ROWS}; do
        profile="$(profile_for "${row}")"
        out="$(out_for "${arm}" "${row}")"
        [[ -s "${out}" ]] && { log "[SKIP] already scored ${out}"; continue; }

        log "${arm} ${row} (z ${z_dim}, hold ${hold}, ${frames} frames)"
        env TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 \
            HYDRA_FULL_ERROR=1 TORCHDYNAMO_DISABLE=1 \
            pixi run -e isaaclab python -u \
            -m imitation_experiments.lowlevel.evaluate_checkpoint \
            --task Isaac-Imitation-G1-v2 --algo IPMD \
            --checkpoint "${checkpoint}" \
            --agent_entry_point rlopt_ipmd_tuned_cfg_entry_point \
            --randomization "${profile}" --action_sampling mode \
            --num_envs "${NUM_ENVS}" --steps "${MAX_STEPS}" --seed 0 \
            --reference_start_frame 0 --reset_schedule sequential \
            --trajectory_ranks "${RANKS[@]}" \
            --output_json "${out}" --label "${arm}_${BOARD}_${row}" --headless \
            --skill_encoder_source auto \
            --kit_args=--/app/extensions/fsWatcherEnabled=false \
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
            "env.command_interface.actor.dim=${command_dim}" \
            env.command_interface.encoder=single \
            "agent.ipmd.latent_dim=${command_dim}" \
            agent.ipmd.command_source=hl_skill \
            "agent.ipmd.hl_skill_checkpoint_path=${encoder}" \
            agent.ipmd.hl_skill_horizon_steps=10 \
            agent.ipmd.hl_skill_command_mode=z \
            "agent.ipmd.latent_steps_min=${hold}" \
            "agent.ipmd.latent_steps_max=${hold}" \
            "agent.ipmd.latent_learning.code_period=${hold}" \
            agent.ipmd.latent_learning.command_phase_mode=sin_cos \
            "agent.ipmd.latent_learning.code_latent_dim=${z_dim}" \
            agent.ipmd.hl_skill_finetune_enabled=false \
            env.expert_macro_state_terms=[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b] \
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
        # Isaac Lab's Kit shutdown can mask a Python traceback behind exit 0,
        # so the written result file is the real success test, not `rc`.
        if (( rc != 0 )) || [[ ! -s "${out}" ]]; then
            log "[FAIL] ${arm} ${row} exit ${rc}: $(tail -3 "${out}.log" | tr '\n' ' ')"
            continue
        fi
        if grep -Eq 'overflow.*increase njmax|nefc overflow' "${out}.log"; then
            log "[FAIL] solver constraint buffer overflow in ${out}.log"
            continue
        fi
    done
done

report
