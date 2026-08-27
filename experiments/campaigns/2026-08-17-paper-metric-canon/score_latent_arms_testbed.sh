#!/usr/bin/env bash
set -uo pipefail

# Score latent low-level arms on the canonical comparison testbed
# (`bones_testbed4096_v1`). Ranks come from the registry, never a copied
# literal, so this cannot drift from `TESTBED4096_RANKS`.
#
# Two rows per arm, matching the paper profiles:
#   clean   randomization none    -> paper_testbed4096_v1        (headline)
#   robust  randomization no_push -> paper_testbed4096_robust_v1
#
# Everything else is the frozen protocol: 4,096 environments, frame-0 starts,
# seed 0, mode actions, Newton/MJWarp, released-SONIC thresholds,
# `foot_pos_xyz` and `base_too_low` disabled, `robot_heading` macro anchor.
#
#   ./score_latent_arms_testbed.sh
#   ARMS="cont_det_ln_hold1" ROWS="clean" ./score_latent_arms_testbed.sh
#   ./score_latent_arms_testbed.sh --report

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

MIRROR="${MIRROR:-${REPO_ROOT}/logs/bottleneck_10b_mirror}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/logs/testbed4096}"
REFERENCE_ARRAYS="${REFERENCE_ARRAYS:-/mnt/hsstorage/fwu91/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
PERSIST_ID="${PERSIST_ID:-bones_seed_sonic_full_129785@e714bbff}"
MAX_STEPS="${MAX_STEPS:-10000}"
SCALED_CELLS="[2048,2048,1024,1024,512,512]"
RUNTIME_BODY_NAMES="[pelvis,left_hip_roll_link,left_knee_link,left_ankle_roll_link,right_hip_roll_link,right_knee_link,right_ankle_roll_link,torso_link,left_shoulder_roll_link,left_elbow_link,left_wrist_yaw_link,right_shoulder_roll_link,right_elbow_link,right_wrist_yaw_link]"

# arm | frames | z_dim | command_dim | hold
ARMS_TABLE=(
"cont_det_ln_hold1|10000269312|256|258|1"
"cont_det_hold1|10000269312|256|258|1"
"cont_det_hold1_resetramp|10000269312|256|258|1"
"fsq64_hold10|10000269312|64|66|10"
"jepa_pure_256d_hold1|10000269312|256|258|1"
"jepa_sigreg_ebm_hold10_256d|10000269312|256|258|10"
"jepa_sigreg_ebm_hold10_fsq64|10000269312|64|66|10"
"jepa_ntp_hold10_256d|8500543488|256|258|10"
# The 50B-chain leader at the checkpoint the smoothness finetunes branch from.
# This board never carried it, so without this row the finetunes below have no
# baseline on the canonical comparison population.
"ln_hold1_sonicreset|46500151296|256|258|1"
# 2026-08-26 smoothness finetunes (`2026-08-26-smooth-finetune`), +2B each.
"ar01|48500047872|256|258|1"
"ar003|48500047872|256|258|1"
"ar01shake4|48500047872|256|258|1"
)
# Every arm of `2026-08-15-latent-bottleneck-10b` that has a scorable mirror.
# The two `*_dyn` arms are absent on purpose: their encoder is fine-tuned inside
# the tracker checkpoint, so pairing it with the pretrained encoder file this
# runner passes would score a mismatched pair.
ARMS="${ARMS:-cont_det_ln_hold1 cont_det_hold1 cont_det_hold1_resetramp \
fsq64_hold10 jepa_sigreg_ebm_hold10_256d jepa_sigreg_ebm_hold10_fsq64 \
jepa_ntp_hold10_256d jepa_pure_256d_hold1}"
ROWS="${ROWS:-clean robust}"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
profile_for() { [[ "$1" == "clean" ]] && echo none || echo no_push; }
out_for() { printf '%s/%s_rand_%s.json' "${OUTPUT_ROOT}" "$1" "$(profile_for "$2")"; }

report() {
    local arm row out
    for arm in ${ARMS}; do
        for row in ${ROWS}; do
            out="$(out_for "${arm}" "${row}")"
            [[ -s "${out}" ]] || { log "[MISSING] ${arm} ${row}"; continue; }
            printf '%-32s %-7s ' "${arm}" "${row}"
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

mapfile -t ranks < <(pixi run python -c \
    'from imitation_experiments.evaluation.protocol import TESTBED4096_RANKS
print("\n".join(str(rank) for rank in TESTBED4096_RANKS))')
[[ "${#ranks[@]}" -eq 4096 ]] || { log "[FATAL] registry returned ${#ranks[@]} ranks"; exit 2; }

for arm in ${ARMS}; do
    entry=""
    for candidate in "${ARMS_TABLE[@]}"; do
        [[ "${candidate%%|*}" == "${arm}" ]] && entry="${candidate}"
    done
    [[ -n "${entry}" ]] || { log "[SKIP] unknown arm ${arm}"; continue; }
    IFS='|' read -r _ frames z_dim command_dim hold <<<"${entry}"

    checkpoint="${MIRROR}/${arm}_seed0/tracker/f${frames}/models/model_step_${frames}.pt"
    encoder="${MIRROR}/${arm}_seed0/encoder/checkpoints/latest.pt"
    [[ -s "${checkpoint}" ]] || { log "[SKIP] no checkpoint ${checkpoint}"; continue; }
    [[ -s "${encoder}" ]] || { log "[SKIP] no encoder ${encoder}"; continue; }

    for row in ${ROWS}; do
        [[ "${row}" == "clean" || "${row}" == "robust" ]] || { log "[SKIP] unknown row ${row}"; continue; }
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
            --num_envs 4096 --steps "${MAX_STEPS}" --seed 0 \
            --reference_start_frame 0 --reset_schedule sequential \
            --trajectory_ranks "${ranks[@]}" \
            --output_json "${out}" --label "${arm}_testbed_${row}" --headless \
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
        # Isaac Lab's Kit shutdown can mask a Python traceback behind exit 0 --
        # measured 2026-08-26, when three CUDA-OOM runs were logged [OK] while
        # writing no result at all. The written file is the real success test.
        if (( rc != 0 )) || [[ ! -s "${out}" ]]; then
            log "[FAIL] ${arm} ${row} exit ${rc}: $(tail -3 "${out}.log" | tr '\n' ' ')"
            continue
        fi
        if grep -Eq 'overflow.*increase njmax|nefc overflow' "${out}.log"; then
            log "[FAIL] ${arm} ${row}: solver constraint buffer overflow"
            continue
        fi
        log "[OK] ${arm} ${row} -> ${out}"
    done
done

report
