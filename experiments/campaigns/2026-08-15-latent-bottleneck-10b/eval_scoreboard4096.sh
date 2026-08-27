#!/usr/bin/env bash
# Isaac/Newton 4,096-motion scoreboard for the latent-bottleneck-10b arms.
#
# Protocol copied from `2026-08-08-bones129k-4096-scoreboard/run.sh` so the rows
# stay comparable with the 2026-08-09 table: 4,096 environments, ranks
# 12288-16383 pinned, frame-0 starts, seed 0, mode actions, `no_push`,
# Newton/MJWarp, released-SONIC thresholds, `foot_pos_xyz` and `base_too_low`
# disabled.
#
# Three settings differ per arm here and must NOT be hardcoded as the old
# runner did: the command hold (1 or 10 control steps), the latent width, and
# the macro anchor frame, which is `robot_heading` for every arm in this
# campaign (SONIC v1.1), not the old default.
#
# Checkpoints come from the ICE mirror written by the campaign's collection
# step; each file name carries its TRUE cumulative frame count.
#
#   ./eval_scoreboard4096.sh                 # every arm below
#   ARMS="cont_det_ln_hold1" ./eval_scoreboard4096.sh
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

MIRROR="${MIRROR:-${REPO_ROOT}/logs/bottleneck_10b_mirror}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/logs/bottleneck_10b_4096}"
REFERENCE_ARRAYS="${REFERENCE_ARRAYS:-/mnt/hsstorage/fwu91/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
PERSIST_ID="${PERSIST_ID:-bones_seed_sonic_full_129785@e714bbff}"
NUM_ENVS="${NUM_ENVS:-4096}"
RANK_START=12288
RANK_END=16383
MAX_STEPS="${MAX_STEPS:-10000}"
SEED=0
SCALED_CELLS="[2048,2048,1024,1024,512,512]"
RUNTIME_BODY_NAMES="[pelvis,left_hip_roll_link,left_knee_link,left_ankle_roll_link,right_hip_roll_link,right_knee_link,right_ankle_roll_link,torso_link,left_shoulder_roll_link,left_elbow_link,left_wrist_yaw_link,right_shoulder_roll_link,right_elbow_link,right_wrist_yaw_link]"

# arm | frames | z_dim | command_dim | hold | encoder_source (default auto)
# The dyn arms score with `--skill_encoder_source checkpoint`: their encoder is
# fine-tuned inside the tracker checkpoint and `load_model` restores it from
# `hl_skill_command_sampler_state_dict` (premise that they could not be scored
# was retired 2026-08-20; the pretrained file below only seeds construction).
ARMS_TABLE=(
"cont_det_ln_hold1|10000269312|256|258|1"
"cont_det_hold1_resetramp|10000269312|256|258|1"
"cont_det_hold1|10000269312|256|258|1"
"jepa_pure_256d_hold1|10000269312|256|258|1"
"fsq64_hold10|10000269312|64|66|10"
"jepa_sigreg_ebm_hold10_256d|10000269312|256|258|10"
"jepa_sigreg_ebm_hold10_fsq64|10000269312|64|66|10"
"jepa_ntp_hold10_256d|8500543488|256|258|10"
# 20B continuations from 2026-08-18-sonic-reset-20b; same protocol, same
# mirror tree, encoder copied from the base arm (binding identical by
# construction).
"ln_hold1_sonicreset|20000145408|256|258|1"
"fsq64_hold10_sonicreset|20000145408|64|66|10"
# 30B continuations from 2026-08-21-sonic-reset-30b, same recipe as the 20B
# rows, budget the only change.
"ln_hold1_sonicreset|30000021504|256|258|1"
"fsq64_hold10_sonicreset|30000021504|64|66|10"
# Online-dynamics finetune arms (dyn_block: achieved ring + offline DiffSR,
# pg_coeff=0). Encoder read from the tracker checkpoint, see note above.
"cont_det_hold1_dyn|10000269312|256|258|1|checkpoint"
"cont_det_hold1_resetramp_dyn|10000269312|256|258|1|checkpoint"
"fsq64_hold10_dyn|10000269312|64|66|10|checkpoint"
)
ARMS="${ARMS:-cont_det_ln_hold1 cont_det_hold1_resetramp cont_det_hold1 jepa_pure_256d_hold1 fsq64_hold10 jepa_sigreg_ebm_hold10_256d jepa_sigreg_ebm_hold10_fsq64 jepa_ntp_hold10_256d}"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

mkdir -p "${OUTPUT_ROOT}"
[ -s "${REFERENCE_ARRAYS}/reference_arrays_manifest.json" ] || {
    log "[FATAL] reference arrays missing: ${REFERENCE_ARRAYS}"
    exit 2
}

ranks=()
for ((r = RANK_START; r <= RANK_END; r++)); do ranks+=("${r}"); done

for arm in ${ARMS}; do
    row=""
    for candidate in "${ARMS_TABLE[@]}"; do
        [[ "${candidate%%|*}" == "${arm}" ]] && row="${candidate}"
    done
    [ -n "${row}" ] || { log "[SKIP] unknown arm ${arm}"; continue; }
    IFS='|' read -r _ frames z_dim command_dim hold enc_src <<<"${row}"
    enc_src="${enc_src:-auto}"

    checkpoint="${MIRROR}/${arm}_seed0/tracker/f${frames}/models/model_step_${frames}.pt"
    encoder="${MIRROR}/${arm}_seed0/encoder/checkpoints/latest.pt"
    out="${OUTPUT_ROOT}/${arm}_f${frames}.json"
    [ -s "${checkpoint}" ] || { log "[SKIP] no checkpoint ${checkpoint}"; continue; }
    [ -s "${out}" ] && { log "[SKIP] already scored ${out}"; continue; }

    log "${arm} @ ${frames} frames (z ${z_dim}, hold ${hold})"
    env TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 \
        HYDRA_FULL_ERROR=1 TORCHDYNAMO_DISABLE=1 \
        pixi run -e isaaclab python -u \
        -m imitation_experiments.lowlevel.evaluate_checkpoint \
        --task Isaac-Imitation-G1-v2 --algo IPMD \
        --checkpoint "${checkpoint}" \
        --agent_entry_point rlopt_ipmd_tuned_cfg_entry_point \
        --randomization no_push --action_sampling mode \
        --num_envs "${NUM_ENVS}" --steps "${MAX_STEPS}" --seed "${SEED}" \
        --reference_start_frame 0 --reset_schedule sequential \
        --trajectory_ranks "${ranks[@]}" \
        --output_json "${out}" --label "${arm}_f${frames}" --headless \
        --skill_encoder_source "${enc_src}" \
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
    if (( rc != 0 )); then
        log "[FAIL] ${arm} exit ${rc}: $(tail -3 "${out}.log" | tr '\n' ' ')"
        continue
    fi
    if grep -Eq 'overflow.*increase njmax|nefc overflow' "${out}.log"; then
        log "[FAIL] ${arm}: solver constraint buffer overflow"
        continue
    fi
    log "[OK] ${arm} -> ${out}"
done
log "SCOREBOARD DONE"
