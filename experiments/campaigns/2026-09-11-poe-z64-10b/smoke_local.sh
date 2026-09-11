#!/usr/bin/env bash
# Local qualification of the product-of-experts arm on the workstation:
#   1. a short encoder pretrain with phi(s,z)=z and mu(s,s',t)
#   2. a check that the checkpoint config carries both settings
#   3. one tracker iteration at 32 environments binding that encoder
# Stops once the wiring visibly works; convergence is the cluster's job.
#
#   ./smoke_local.sh                       # 50 updates, batch 1024, 32 envs
#   PRETRAIN_UPDATES=500 PRETRAIN_BATCH=4096 ./smoke_local.sh
#   SKIP_LOWLEVEL=1 ./smoke_local.sh       # pretrain + config check only
#   CACHE_DEVICE=cpu ./smoke_local.sh      # shared GPU: keep the ref caches on CPU
set -uo pipefail
CAMPAIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${CAMPAIGN_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"
WORK="${WORK:-${REPO_ROOT}/logs/poe_z64_smoke}"
REFERENCE_ARRAYS="${REFERENCE_ARRAYS:-/mnt/hsstorage/fwu91/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
PERSIST_ID="${PERSIST_ID:-bones_seed_sonic_full_129785@e714bbff}"
PRETRAIN_UPDATES="${PRETRAIN_UPDATES:-50}"
PRETRAIN_BATCH="${PRETRAIN_BATCH:-1024}"
LOWLEVEL_ENVS="${LOWLEVEL_ENVS:-32}"
mkdir -p "${WORK}"
log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
[[ -s "${REFERENCE_ARRAYS}/reference_arrays_manifest.json" ]] || { log "[FATAL] reference arrays missing: ${REFERENCE_ARRAYS}"; exit 2; }

DATA_OVERRIDES=(
    physics=newton_mjwarp
    env.data.manifest=null
    "env.data.reference_arrays_dir=${REFERENCE_ARRAYS}"
    "env.data.persist_id=${PERSIST_ID}"
    env.data.reference_arrays_resident=false
    env.data.reference_arrays_warm_workers=4
    # cuda:0 in the campaign; cpu when the workstation GPU is shared and the
    # 129k-clip macro cache does not fit beside other users' jobs
    # (CUBLAS_STATUS_ALLOC_FAILED on the first matmul, 2026-09-11).
    "env.data.runtime_cache_device=${CACHE_DEVICE:-cuda:0}"
    "env.data.macro_cache_device=${CACHE_DEVICE:-cuda:0}"
    "env.data.runtime_cache_body_names=[pelvis,left_hip_roll_link,left_knee_link,left_ankle_roll_link,right_hip_roll_link,right_knee_link,right_ankle_roll_link,torso_link,left_shoulder_roll_link,left_elbow_link,left_wrist_yaw_link,right_shoulder_roll_link,right_elbow_link,right_wrist_yaw_link]"
    "env.expert_macro_state_terms=[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b]"
    env.expert_macro_frame_stride=1
    env.expert_macro_anchor_mode=robot_heading
)

# The campaign's pretrain tokens, read from campaign.yaml so this smoke cannot
# drift from the job: objective_args + ln_args + pretrain_tail, with the
# update count and batch size replaced by the smoke's.
mapfile -t ARM_TOKENS < <(pixi run python - "${CAMPAIGN_DIR}/campaign.yaml" "${PRETRAIN_UPDATES}" "${PRETRAIN_BATCH}" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))["vars"]
toks = list(c["objective_args"]) + list(c["ln_args"]) + list(c["pretrain_tail"])
out, i = [], 0
while i < len(toks):
    t = str(toks[i])
    if t == "--num_updates": out += [t, sys.argv[2]]; i += 2; continue
    if t == "--batch_size": out += [t, sys.argv[3]]; i += 2; continue
    if t.startswith("${"): i += 1; continue
    out.append(t); i += 1
print("\n".join(out))
PY
)

log "pretrain: ${PRETRAIN_UPDATES} updates, batch ${PRETRAIN_BATCH}"
env TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 TORCHDYNAMO_DISABLE=1 \
    pixi run -e isaaclab python -u scripts/rlopt/train_hl_skill_diffsr.py \
    --task Isaac-Imitation-G1-v2 --num_envs 16 --seed 0 --device cuda:0 --headless --assert-kitless \
    --logger_backend none --output_dir "${WORK}/encoder" \
    --horizon_steps 10 --encoder_window_mode intermediate \
    --transition_objective jepa_ntp --latent_mode deterministic --z_dim 64 \
    "${ARM_TOKENS[@]}" "${DATA_OVERRIDES[@]}" > "${WORK}/pretrain.log" 2>&1
rc=$?
CKPT="${WORK}/encoder/checkpoints/latest.pt"
if (( rc != 0 )) || [[ ! -s "${CKPT}" ]]; then
    log "[FAIL] pretrain exit ${rc}: $(grep -iE 'error|traceback' "${WORK}/pretrain.log" | tail -2 | tr '\n' ' ' | cut -c1-200)"; exit 1
fi
log "[OK] pretrain -> ${CKPT}"

log "checkpoint config:"
pixi run python - "${CKPT}" <<'PY' || exit 1
import sys, torch
blob = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
cfg = blob.get("config", {})
phi = cfg.get("diffsr_phi_parameterization"); mu = cfg.get("diffsr_mu_conditioning")
fd = cfg.get("diffsr_feature_dim"); zd = cfg.get("z_dim")
print(f"  phi_parameterization={phi} mu_conditioning={mu} feature_dim={fd} z_dim={zd}")
assert phi == "identity" and mu == "pair" and fd == zd == 64, "checkpoint does not carry the PoE settings"
sd = blob.get("diffsr_state_dict") or {}
mu_in = [v.shape[1] for k, v in sd.items() if k.endswith("mu_net.input_layer.weight") or ("mu_net" in k and k.endswith("0.weight"))]
print(f"  mu_net first-layer input widths seen: {mu_in[:2]} (pair => includes the 380-wide source)")
PY
log "[OK] checkpoint carries identity phi + pair mu"

if [[ "${SKIP_LOWLEVEL:-0}" == "1" ]]; then log "[SKIP] lowlevel"; exit 0; fi
log "lowlevel: 1 iteration at ${LOWLEVEL_ENVS} envs binding the smoke encoder"
env TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 TORCHDYNAMO_DISABLE=1 \
    pixi run -e isaaclab python -u scripts/rlopt/train.py \
    --task Isaac-Imitation-G1-v2 --algo IPMD --agent rlopt_ipmd_tuned_fullbatch_cfg_entry_point \
    --num_envs "${LOWLEVEL_ENVS}" --seed 0 --headless --assert-kitless --max_iterations 1 \
    "agent.logger.log_dir=${WORK}/tracker" agent.logger.backend=none agent.logger.video=false \
    env.command_interface.actor=latent env.command_interface.actor.dim=66 env.command_interface.encoder=single \
    agent.ipmd.latent_dim=66 agent.ipmd.command_source=hl_skill \
    "agent.ipmd.hl_skill_checkpoint_path=${CKPT}" agent.ipmd.hl_skill_horizon_steps=10 \
    agent.ipmd.hl_skill_command_mode=z agent.ipmd.hl_skill_finetune_enabled=false \
    agent.ipmd.latent_steps_min=1 agent.ipmd.latent_steps_max=1 agent.ipmd.latent_learning.code_period=1 \
    agent.ipmd.latent_learning.command_phase_mode=sin_cos agent.ipmd.latent_learning.code_latent_dim=64 \
    env.rewards.motion_ee_pos.weight=1.0 env.rewards.motion_global_anchor_pos_wide.weight=1.0 \
    env.rewards.action_rate_l2.weight=-0.03 env.rewards.tracking_reward_points.weight=4.0 \
    env.command_interface.reference.selection=random80_adaptive20 env.data.reference_prefetch_mode=next \
    agent.collector.frames_per_batch=24 agent.ipmd.expert_batch_size=256 agent.loss.gamma=0.97 \
    env.sim.physics.solver_cfg.njmax=320 env.sim.physics.solver_cfg.nconmax=200 \
    "agent.policy.num_cells=[2048,2048,1024,1024,512,512]" "agent.value_function.num_cells=[2048,2048,1024,1024,512,512]" \
    agent.policy.activation_fn=silu agent.value_function.activation_fn=silu \
    "${DATA_OVERRIDES[@]}" > "${WORK}/lowlevel.log" 2>&1
rc=$?
if (( rc != 0 )); then
    log "[FAIL] lowlevel exit ${rc}: $(grep -iE 'error|traceback' "${WORK}/lowlevel.log" | tail -2 | tr '\n' ' ' | cut -c1-200)"; exit 1
fi
log "[PASS] pretrain + config + one tracker iteration. Logs under ${WORK}"
