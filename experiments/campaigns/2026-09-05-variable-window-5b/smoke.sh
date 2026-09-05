#!/usr/bin/env bash
# Wiring qualification for the variable-window campaign. Proves an arm RUNS;
# it proves nothing about the arm's quality.
#
# Per arm: (1) the encoder pretrain completes real offline updates with a
# finite loss and writes a loadable checkpoint; (2) the frozen encoder drives
# one 128-frame IPMD iteration at the arm's live policy and command width.
# Flags are composed from `campaign.yaml`, so the smoke exercises the command
# line the cluster freezes. Only the budget knobs are overridden.
#
#   ./smoke.sh                       # every pretraining arm
#   ARMS="vw_padded vw_stride" ./smoke.sh
#
# PIXI_MANIFEST selects the pixi project that owns the isaaclab environment
# (default: this checkout). SMOKE_LAUNCHER, when set, is a Python file run in
# place of the script, receiving the script path as its first argument; use it
# to import a worktree's RLOpt from another checkout's isaaclab environment.
set -uo pipefail
CAMPAIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${CAMPAIGN_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"
CAMPAIGN="${CAMPAIGN_DIR}/campaign.yaml"
PIXI_MANIFEST="${PIXI_MANIFEST:-${REPO_ROOT}/pixi.toml}"

SMOKE_ROOT="${SMOKE_ROOT:-${REPO_ROOT}/logs/variable_window_smoke}"
REFERENCE_ARRAYS="${REFERENCE_ARRAYS:-/mnt/hsstorage/fwu91/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
PERSIST_ID="${PERSIST_ID:-bones_seed_sonic_full_129785@e714bbff}"
PRETRAIN_UPDATES="${PRETRAIN_UPDATES:-4}"
PRETRAIN_BATCH="${PRETRAIN_BATCH:-8192}"
PRETRAIN_ENVS="${PRETRAIN_ENVS:-16}"
LOWLEVEL_ENVS="${LOWLEVEL_ENVS:-64}"
LOWLEVEL_STEPS="${LOWLEVEL_STEPS:-2}"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

run_py() { pixi run --manifest-path "${PIXI_MANIFEST}" python "$@"; }
run_isaac() {
    local launcher=()
    if [[ -n "${SMOKE_LAUNCHER:-}" ]]; then
        launcher=("${SMOKE_LAUNCHER}")
    fi
    env TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 \
        HYDRA_FULL_ERROR=1 TORCHDYNAMO_DISABLE=1 \
        pixi run --manifest-path "${PIXI_MANIFEST}" -e isaaclab python -u \
        ${launcher[@]+"${launcher[@]}"} "$@"
}

# One merged var of an arm, lists newline-separated, ${vars.X} resolved.
arm_field() {
    run_py - "${CAMPAIGN}" "$1" "$2" <<'PY'
import sys, yaml, pathlib
path, name, field = sys.argv[1:4]
campaign = yaml.safe_load(pathlib.Path(path).read_text())
merged = {**campaign["vars"], **campaign["arms"][name]["vars"]}
value = merged[field]
while isinstance(value, str) and value.startswith("${vars.") and value.endswith("}"):
    value = merged[value[len("${vars.") : -1]]
if isinstance(value, list):
    print("\n".join(str(v) for v in value))
else:
    print(value)
PY
}

pretraining_arms() {
    run_py - "${CAMPAIGN}" <<'PY'
import sys, yaml, pathlib
campaign = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text())
for name, arm in campaign["arms"].items():
    if any(stage["name"] == "pretrain" for stage in arm["stages"]):
        print(name)
PY
}

DATA_OVERRIDES=(
    physics=newton_mjwarp
    env.data.manifest=null
    "env.data.reference_arrays_dir=${REFERENCE_ARRAYS}"
    "env.data.persist_id=${PERSIST_ID}"
    env.data.reference_arrays_resident=false
    env.data.reference_arrays_warm_workers=4
    env.data.runtime_cache_device=cuda:0
    env.data.macro_cache_device=cuda:0
    "env.data.runtime_cache_body_names=[pelvis,left_hip_roll_link,left_knee_link,left_ankle_roll_link,right_hip_roll_link,right_knee_link,right_ankle_roll_link,torso_link,left_shoulder_roll_link,left_elbow_link,left_wrist_yaw_link,right_shoulder_roll_link,right_elbow_link,right_wrist_yaw_link]"
)

if [[ -n "${ARMS:-}" ]]; then
    read -r -a selected <<< "${ARMS}"
else
    mapfile -t selected < <(pretraining_arms)
fi

failures=0
for arm in "${selected[@]}"; do
    work="${SMOKE_ROOT}/${arm}"
    # REUSE_PRETRAIN=1 keeps a finished pretrain and re-runs the tracker only.
    if [[ "${REUSE_PRETRAIN:-0}" != "1" || ! -s "${work}/encoder/checkpoints/latest.pt" ]]; then
        rm -rf "${work}"
    fi
    mkdir -p "${work}"
    mapfile -t window_tokens < <(arm_field "${arm}" window_args)
    mapfile -t source_tokens < <(arm_field "${arm}" source_args)
    mapfile -t history_tokens < <(arm_field "${arm}" history_args)
    horizon="$(arm_field "${arm}" pretrain_horizon)"
    live="$(arm_field "${arm}" live_horizon)"
    macro_terms="$(arm_field "${arm}" macro_terms)"
    z_dim="$(arm_field "${arm}" z_dim)"
    command_dim="$(arm_field "${arm}" command_dim)"

    checkpoint="${work}/encoder/checkpoints/latest.pt"
    if [[ "${REUSE_PRETRAIN:-0}" == "1" && -s "${checkpoint}" ]]; then
        log "${arm}: pretrain reused (${checkpoint})"
        pre_rc=0
    else
    log "${arm}: pretrain (${PRETRAIN_UPDATES} updates, batch ${PRETRAIN_BATCH}, horizon ${horizon})"
    run_isaac scripts/rlopt/train_hl_skill_diffsr.py \
        --task Isaac-Imitation-G1-v2 --num_envs "${PRETRAIN_ENVS}" --seed 0 \
        --device cuda:0 --headless --assert-kitless --logger_backend none \
        --output_dir "${work}/encoder" \
        --horizon_steps "${horizon}" --encoder_window_mode intermediate \
        --transition_objective jepa_ntp --jepa_loss sigreg_ebm \
        --jepa_ntp_head diff_chunk --jepa_ntp_chunk_span boundary_next \
        --jepa_endpoint_coeff 0 --diffsr_phi_parameterization affine \
        --latent_mode deterministic --z_dim "${z_dim}" --encoder_layer_norm \
        "${window_tokens[@]}" "${source_tokens[@]}" \
        --encoder_hidden_dims 2048 1024 512 512 --encoder_activation silu \
        --diffsr_feature_dim 256 --diffsr_embed_dim 1024 \
        --diffsr_g_hidden_dims 1024 1024 512 --diffsr_mu_hidden_dims 1024 1024 512 \
        --batch_size "${PRETRAIN_BATCH}" --num_updates "${PRETRAIN_UPDATES}" \
        --log_interval 1 --eval_batches 1 \
        "${DATA_OVERRIDES[@]}" \
        "env.expert_macro_state_terms=${macro_terms}" \
        env.expert_macro_frame_stride=1 env.expert_macro_anchor_mode=robot_heading \
        > "${work}/pretrain.log" 2>&1
    pre_rc=$?
    fi
    if (( pre_rc != 0 )) || [[ ! -s "${checkpoint}" ]]; then
        log "[FAIL] ${arm} pretrain exit ${pre_rc}: $(grep -E 'Error|error' "${work}/pretrain.log" | tail -2 | tr '\n' ' ' | cut -c1-300)"
        failures=$((failures+1)); continue
    fi

    log "${arm}: lowlevel (${LOWLEVEL_ENVS} x ${LOWLEVEL_STEPS} frames, live ${live}, command ${command_dim})"
    run_isaac scripts/rlopt/train.py \
        --task Isaac-Imitation-G1-v2 --algo IPMD \
        --agent rlopt_ipmd_tuned_fullbatch_cfg_entry_point \
        --num_envs "${LOWLEVEL_ENVS}" --seed 0 --headless --assert-kitless \
        --max_iterations 1 \
        "agent.logger.log_dir=${work}/tracker" \
        agent.logger.backend=none agent.logger.video=false \
        env.command_interface.actor=latent \
        "env.command_interface.actor.dim=${command_dim}" \
        env.command_interface.encoder=single \
        "agent.ipmd.latent_dim=${command_dim}" \
        agent.ipmd.command_source=hl_skill \
        "agent.ipmd.hl_skill_checkpoint_path=${checkpoint}" \
        "agent.ipmd.hl_skill_horizon_steps=${horizon}" \
        "agent.ipmd.hl_skill_live_horizon=${live}" \
        agent.ipmd.hl_skill_command_mode=z \
        agent.ipmd.hl_skill_finetune_enabled=false \
        agent.ipmd.latent_steps_min=1 agent.ipmd.latent_steps_max=1 \
        agent.ipmd.latent_learning.code_period=1 \
        agent.ipmd.latent_learning.command_phase_mode=sin_cos \
        "agent.ipmd.latent_learning.code_latent_dim=${z_dim}" \
        "${history_tokens[@]}" \
        agent.optim.weight_decay=1.0e-2 \
        agent.ipmd.critic_lr_schedule=linear agent.ipmd.critic_lr_final=1.0e-5 \
        env.rewards.motion_ee_pos.weight=1.0 \
        env.rewards.motion_global_anchor_pos_wide.weight=1.0 \
        env.rewards.action_rate_l2.weight=-0.03 \
        env.rewards.tracking_reward_points.weight=4.0 \
        env.enable_termination_curriculum=true \
        env.termination_curriculum_start_frames=5000000 \
        env.termination_curriculum_end_frames=30000000 \
        env.command_interface.reference.selection=random80_adaptive20 \
        env.data.reference_prefetch_mode=next \
        "agent.collector.frames_per_batch=${LOWLEVEL_STEPS}" \
        agent.ipmd.expert_batch_size=256 \
        agent.loss.gamma=0.97 \
        agent.save_interval=250000000 \
        env.sim.physics.solver_cfg.njmax=320 env.sim.physics.solver_cfg.nconmax=200 \
        "agent.policy.num_cells=[2048,2048,1024,1024,512,512]" \
        "agent.value_function.num_cells=[2048,2048,1024,1024,512,512]" \
        agent.policy.activation_fn=silu agent.value_function.activation_fn=silu \
        "${DATA_OVERRIDES[@]}" \
        "env.expert_macro_state_terms=${macro_terms}" \
        env.expert_macro_frame_stride=1 env.expert_macro_anchor_mode=robot_heading \
        > "${work}/lowlevel.log" 2>&1
    low_rc=$?
    if (( low_rc != 0 )); then
        log "[FAIL] ${arm} lowlevel exit ${low_rc}: $(grep -E 'Error|error' "${work}/lowlevel.log" | tail -2 | tr '\n' ' ' | cut -c1-300)"
        failures=$((failures+1)); continue
    fi
    loss="$(grep -o 'train/jepa_ntp_loss[^,}]*' "${work}/pretrain.log" | tail -1)"
    log "[PASS] ${arm}: ${loss}"
done
log "smoke done: ${failures} failure(s)"
exit "${failures}"
