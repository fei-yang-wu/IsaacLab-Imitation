#!/usr/bin/env bash
# Frozen 4,096-motion SONIC scoreboard for the latent-design ablation.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

TRAIN_SEED="${TRAIN_SEED:-0}"
MIRROR="${MIRROR:-${REPO_ROOT}/logs/bones_latent_ablation_mirror}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/logs/bones_latent_ablation_4096}"
REFERENCE_ARRAYS="${REFERENCE_ARRAYS:-/mnt/hsstorage/fwu91/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
PERSIST_ID="${PERSIST_ID:-bones_seed_sonic_full_129785@e714bbff}"
NUM_ENVS="${NUM_ENVS:-4096}"
RANK_START=12288
RANK_END=16383
MAX_STEPS="${MAX_STEPS:-10000}"
EVAL_SEED=0
EXPECTED_FRAMES="${EXPECTED_FRAMES:-10000269312}"
SCALED_CELLS="[2048,2048,1024,1024,512,512]"
RUNTIME_BODY_NAMES="[pelvis,left_hip_roll_link,left_knee_link,left_ankle_roll_link,right_hip_roll_link,right_knee_link,right_ankle_roll_link,torso_link,left_shoulder_roll_link,left_elbow_link,left_wrist_yaw_link,right_shoulder_roll_link,right_elbow_link,right_wrist_yaw_link]"

# arm | z_dim | command_dim
ARMS_TABLE=(
"spectral_cont256|256|258"
"spectral_fsq64|64|66"
"recon_cont256|256|258"
"recon_fsq64|64|66"
)
ARMS="${ARMS:-spectral_cont256 spectral_fsq64 recon_cont256 recon_fsq64}"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

mkdir -p "${OUTPUT_ROOT}"
[ -s "${REFERENCE_ARRAYS}/reference_arrays_manifest.json" ] || {
    log "[FATAL] reference arrays missing: ${REFERENCE_ARRAYS}"
    exit 2
}

ranks=()
for ((rank = RANK_START; rank <= RANK_END; rank++)); do ranks+=("${rank}"); done

for arm in ${ARMS}; do
    row=""
    for candidate in "${ARMS_TABLE[@]}"; do
        [[ "${candidate%%|*}" == "${arm}" ]] && row="${candidate}"
    done
    [ -n "${row}" ] || { log "[SKIP] unknown arm ${arm}"; continue; }
    IFS='|' read -r _ z_dim command_dim <<<"${row}"

    root="${MIRROR}/${arm}_seed${TRAIN_SEED}"
    checkpoint="${root}/tracker/f${EXPECTED_FRAMES}/models/model_step_${EXPECTED_FRAMES}.pt"
    encoder="${root}/encoder/checkpoints/latest.pt"
    out="${OUTPUT_ROOT}/${arm}_seed${TRAIN_SEED}_f${EXPECTED_FRAMES}.json"
    [ -s "${checkpoint}" ] || { log "[SKIP] no checkpoint ${checkpoint}"; continue; }
    [ -s "${encoder}" ] || { log "[SKIP] no encoder ${encoder}"; continue; }
    [ -s "${out}" ] && { log "[SKIP] already scored ${out}"; continue; }

    log "${arm}, train seed ${TRAIN_SEED}, ${EXPECTED_FRAMES} frames"
    env TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 \
        HYDRA_FULL_ERROR=1 TORCHDYNAMO_DISABLE=1 \
        pixi run -e isaaclab python -u \
        -m imitation_experiments.lowlevel.evaluate_checkpoint \
        --task Isaac-Imitation-G1-v2 --algo IPMD \
        --checkpoint "${checkpoint}" \
        --agent_entry_point rlopt_ipmd_tuned_cfg_entry_point \
        --randomization no_push --action_sampling mode \
        --num_envs "${NUM_ENVS}" --steps "${MAX_STEPS}" --seed "${EVAL_SEED}" \
        --reference_start_frame 0 --reset_schedule sequential \
        --trajectory_ranks "${ranks[@]}" \
        --output_json "${out}" --label "${arm}_seed${TRAIN_SEED}" --headless \
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
        agent.ipmd.latent_steps_min=10 \
        agent.ipmd.latent_steps_max=10 \
        agent.ipmd.latent_learning.code_period=10 \
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
    if ((rc != 0)); then
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
