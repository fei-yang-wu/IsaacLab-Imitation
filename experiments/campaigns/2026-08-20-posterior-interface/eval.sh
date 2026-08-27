#!/usr/bin/env bash
# Score the posterior arms on the canonical testbed, the same protocol the
# frozen star's rows use, so the two campaigns sit in one table.
#
# The posterior route has no pretrained encoder file: the code is learned during
# RL, so the encoder lives inside the tracker checkpoint and is restored by
# `load_model`. There is therefore no `hl_skill_checkpoint_path` here, and
# `--skill_encoder_source checkpoint` records that provenance in the summary.
#
# Every per-arm field is read back out of `campaign.yaml`, so evaluation cannot
# drift from training.
#
# Rows:
#   milestone  every mirrored checkpoint on `bones_milestone_testbed256_v1`
#              (256 clips, testbed population) -- the budget axis
#   clean      the final checkpoint on `bones_testbed4096_v1`   -- row of record
#   robust     the final checkpoint on `bones_testbed4096_robust_v1`
#
#   ./eval.sh                                   # every mirrored arm, clean row
#   ROWS=milestone ./eval.sh                    # the budget axis
#   ARMS="post_recon_ae" ROWS=clean ./eval.sh
#   ./eval.sh --report
set -uo pipefail
CAMPAIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${CAMPAIGN_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

MIRROR="${MIRROR:-${REPO_ROOT}/logs/posterior_interface_mirror}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/logs/posterior_interface_eval}"
REFERENCE_ARRAYS="${REFERENCE_ARRAYS:-/mnt/hsstorage/fwu91/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
PERSIST_ID="${PERSIST_ID:-bones_seed_sonic_full_129785@e714bbff}"
FRAMES="${FRAMES:-2000289792}"
ROWS="${ROWS:-clean}"
MAX_STEPS="${MAX_STEPS:-10000}"
SCALED="[2048,2048,1024,1024,512,512]"
BODIES="[pelvis,left_hip_roll_link,left_knee_link,left_ankle_roll_link,right_hip_roll_link,right_knee_link,right_ankle_roll_link,torso_link,left_shoulder_roll_link,left_elbow_link,left_wrist_yaw_link,right_shoulder_roll_link,right_elbow_link,right_wrist_yaw_link]"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
randomization_for() { [[ "$1" == "robust" ]] && echo no_push || echo none; }
board_for() {
    case "$1" in
        milestone) echo bones_milestone_testbed256_v1 ;;
        clean|robust) echo bones_testbed4096_v1 ;;
    esac
}
ranks_for() {
    pixi run python -c "
from imitation_experiments.evaluation.protocol import BOARDS
print('\n'.join(str(case.trajectory_rank) for case in BOARDS['$1'].cases))"
}

arm_field() {
    pixi run python - "$1" "$2" <<'PY'
import sys, yaml, pathlib
name, field = sys.argv[1:3]
c = yaml.safe_load(
    pathlib.Path(
        "experiments/campaigns/2026-08-20-posterior-interface/campaign.yaml"
    ).read_text()
)
merged = {**c["vars"], **c["arms"][name]["vars"]}
value = merged[field]
if isinstance(value, list):
    value = " ".join(str(v) for v in value)
print(value)
PY
}

if [[ "${1:-}" == "--report" ]]; then
    shopt -s nullglob
    for out in "${OUTPUT_ROOT}"/*.json; do
        printf '%-34s ' "$(basename "${out}" .json)"
        pixi run python -m imitation_experiments.evaluation.summarize_paper_boards "${out}"
    done
    exit 0
fi

[[ -s "${REFERENCE_ARRAYS}/reference_arrays_manifest.json" ]] || {
    log "[FATAL] reference arrays missing"; exit 2; }
mkdir -p "${OUTPUT_ROOT}"

if [[ "${ARMS:-}" == "" ]]; then
    mapfile -t all < <(pixi run python -c "
import yaml
c = yaml.safe_load(open('experiments/campaigns/2026-08-20-posterior-interface/campaign.yaml'))
print('\n'.join(c['arms']))")
    ARMS="${all[*]}"
fi

for arm in ${ARMS}; do
  tree="${MIRROR}/${arm}_seed0"
  # Checkpoint tree names carry the TRUE cumulative frame count; the
  # per-segment step counter restarts on every chained resume.
  mapfile -t frames < <(ls -1 "${tree}/tracker" 2>/dev/null | sed -n 's/^f\([0-9]\+\)$/\1/p' | sort -n)
  [[ "${#frames[@]}" -gt 0 ]] || { log "[SKIP] no checkpoints in ${tree}/tracker"; continue; }
  quant="$(arm_field "${arm}" quantizer)"
  code="$(arm_field "${arm}" code_latent_dim)"
  cmd="$(arm_field "${arm}" command_dim)"
  space="$(arm_field "${arm}" space_args)"
  pg="$(arm_field "${arm}" through_policy)"
  recon="$(arm_field "${arm}" recon_coeff)"

  for row in ${ROWS}; do
    board="$(board_for "${row}")"
    [[ -n "${board}" ]] || { log "[SKIP] unknown row ${row}"; continue; }
    prof="$(randomization_for "${row}")"

    mapfile -t ranks < <(ranks_for "${board}")
    [[ "${#ranks[@]}" -gt 0 ]] || { log "[FATAL] board ${board} returned no ranks"; exit 2; }

    # The budget axis wants every mirrored checkpoint; a row of record wants
    # only the last one. `FRAMES` still pins the final frame count.
    if [[ "${row}" == "milestone" ]]; then
        selected=("${frames[@]}")
    else
        selected=("${FRAMES}")
    fi

    for frame in "${selected[@]}"; do
    ck="${tree}/tracker/f${frame}/models/model_step_${frame}.pt"
    [[ -s "${ck}" ]] || { log "[SKIP] missing ${ck}"; continue; }
    out="${OUTPUT_ROOT}/${arm}_seed0_${row}_f${frame}.json"
    [[ -s "${out}" ]] && { log "[SKIP] scored $(basename "${out}")"; continue; }
    log "${arm} ${row} f${frame} (q ${quant}, code ${code}, cmd ${cmd}, ${board}, ${#ranks[@]} clips)"
    env TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 \
        HYDRA_FULL_ERROR=1 TORCHDYNAMO_DISABLE=1 \
        pixi run -e isaaclab python -u \
        -m imitation_experiments.lowlevel.evaluate_checkpoint \
        --task Isaac-Imitation-G1-v2 --algo IPMD --checkpoint "${ck}" \
        --agent_entry_point rlopt_ipmd_posterior_root_qpos_cfg_entry_point \
        --skill_encoder_source checkpoint \
        --randomization "${prof}" --action_sampling mode \
        --num_envs "${#ranks[@]}" --steps "${MAX_STEPS}" --seed 0 \
        --reference_start_frame 0 --reset_schedule sequential \
        --trajectory_ranks "${ranks[@]}" \
        --output_json "${out}" --label "${arm}_${row}_f${frame}" --headless \
        --kit_args=--/app/extensions/fsWatcherEnabled=false \
        physics=newton_mjwarp \
        env.sim.physics.solver_cfg.njmax=320 env.sim.physics.solver_cfg.nconmax=200 \
        env.events.push_robot=null env.data.manifest=null \
        "env.data.reference_arrays_dir=${REFERENCE_ARRAYS}" \
        "env.data.persist_id=${PERSIST_ID}" \
        env.data.reference_arrays_resident=false \
        env.data.reference_arrays_warm_workers=8 \
        env.data.runtime_cache_device=cuda:0 \
        env.data.reference_prefetch_mode=off env.data.macro_cache_device=cuda:0 \
        "env.data.runtime_cache_body_names=${BODIES}" \
        env.command_interface.actor=latent "env.command_interface.actor.dim=${cmd}" \
        env.command_interface.encoder=single \
        "env.command_interface.encoder.components=[joint_qpos,root_pos,root_ori]" \
        "agent.ipmd.latent_dim=${cmd}" \
        agent.ipmd.latent_steps_min=10 agent.ipmd.latent_steps_max=10 \
        agent.ipmd.latent_learning.method=patch_autoencoder \
        "agent.ipmd.latent_learning.quantizer=${quant}" \
        "agent.ipmd.latent_learning.code_latent_dim=${code}" \
        agent.ipmd.latent_learning.command_phase_mode=sin_cos \
        agent.ipmd.latent_learning.code_period=10 \
        agent.ipmd.latent_learning.patch_past_steps=0 \
        agent.ipmd.latent_learning.patch_future_steps=9 \
        "agent.ipmd.latent_learning.train_posterior_through_policy=${pg}" \
        "agent.ipmd.latent_learning.recon_coeff=${recon}" \
        agent.ipmd.latent_learning.kl_coeff=0.0 \
        ${space:+${space}} \
        env.expert_macro_state_terms=[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b] \
        env.expert_macro_frame_stride=1 env.expert_macro_anchor_mode=robot_heading \
        env.terminations.anchor_pos.params.threshold=0.25 \
        env.terminations.anchor_pos.params.down_threshold=0.25 \
        env.terminations.anchor_ori.params.threshold=1.0 \
        env.terminations.ee_body_pos.params.threshold=0.25 \
        env.terminations.ee_body_pos.params.down_threshold=0.25 \
        env.terminations.foot_pos_xyz=null env.terminations.base_too_low=null \
        "agent.policy.num_cells=${SCALED}" agent.policy.activation_fn=silu \
        "agent.value_function.num_cells=${SCALED}" agent.value_function.activation_fn=silu \
        > "${out}.log" 2>&1
    rc=$?
    (( rc != 0 )) && log "[FAIL] ${arm} ${row} f${frame} exit ${rc}: $(tail -3 "${out}.log" | tr '\n' ' ' | cut -c1-150)" || log "[OK] ${arm} ${row} f${frame}"
    done
  done
done
