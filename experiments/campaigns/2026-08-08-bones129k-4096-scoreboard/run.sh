#!/usr/bin/env bash
set -uo pipefail

# One frozen 4,096-motion scoreboard for every BONES-129k low-level arm that
# finished, so SONIC success rate and success-only MPJPE-L come from ONE
# protocol instead of per-campaign one-offs.
#
# The protocol is copied verbatim from
# `experiments/campaigns/2026-08-06-bones129k-sonic-fsq-scale/monitor_and_eval.sh`
# so the rows it already produced (old_z256, fsq64_tuned, fsq64_sonic) stay
# valid and are NOT recomputed here:
#
#   4,096 environments, one per motion, trajectory ranks 12288-16383 PINNED
#   (rank SHA-256 786ef677...), frame-0 starts, seed 0, deterministic mode
#   actions, `no_push` randomization (startup + reset randomization kept),
#   Newton/MJWarp, released SONIC thresholds (anchor_pos / ee_body_pos 0.25 m,
#   anchor_ori 1.0 rad), `foot_pos_xyz` and `base_too_low` disabled.
#
# The released SONIC checkpoint's row on the same ranks is
# `logs/sonic_release_4096/sonic_release_ranks12288_16383.json`
# (SR 0.9937, success-only MPJPE-L 28.65 mm).
#
#   ./run.sh                # pull + evaluate every pending arm
#   ARMS="critic_no_latent" ./run.sh
#   PASSES="sonic full_horizon" ./run.sh
#   ./run.sh --report       # table only

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
while [[ ! -x "${REPO_ROOT}/docker/cluster/cluster_interface.sh" ]]; do
    [[ "${REPO_ROOT}" != "/" ]] || { echo "[FATAL] repository root not found" >&2; exit 2; }
    REPO_ROOT="$(dirname "${REPO_ROOT}")"
done
cd "${REPO_ROOT}"

MODE="run"
[[ "${1:-}" == "--report" ]] && MODE="report"

REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/logs/bones129k_4096_scoreboard}"
REFERENCE_ARRAYS="${REFERENCE_ARRAYS:-/mnt/hsstorage/fwu91/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
PERSIST_ID="${PERSIST_ID:-bones_seed_sonic_full_129785@e714bbff}"
NUM_ENVS=4096
RANK_START=12288
RANK_END=16383
MAX_STEPS="${MAX_STEPS:-10000}"
SEED=0
PASSES="${PASSES:-sonic}"

TUNED_CELLS="[1024,1024,512]"
SCALED_CELLS="[2048,2048,1024,1024,512,512]"

STRIDE5="${REMOTE_DATA_ROOT}/bones129k_latent_mode_stride5"
SKILL="${REMOTE_DATA_ROOT}/bones129k_skill_encoding"
CRITIC="${REMOTE_DATA_ROOT}/bones129k_critic_ablation/bones129k_z256_critic_no_latent_e16384_r24_5b_seed0"

LATENT_SAMPLER="${REMOTE_DATA_ROOT}/bones129k_latent_sampler"

# arm | tracker checkpoint | encoder checkpoint | command dim | code dim | cells | macro stride | extra hydra overrides
#
# An encoder field of `-` marks an EXPLICIT arm: it publishes a raw command
# instead of a latent, so it has no encoder to pull and takes the explicit
# command-interface overrides below rather than the hl_skill ones. Its command
# dim and code dim are ignored.
ARMS_TABLE=(
"critic_no_latent|${CRITIC}/rlopt_train/2026-08-06_21-25-59_wandb-mp85ex1f/models/model_step_5000134656.pt|${REMOTE_DATA_ROOT}/pretrain_store/bones129k_v2_root_qpos_det_sr_h10_z256_seed0/checkpoints/latest.pt|258|256|${TUNED_CELLS}|1|env.command_interface.critic_channels=[reference]"
"skill_state_occupancy|${SKILL}/bones129k_skill_state_occupancy_h10_z256_seed0/rlopt_train/2026-08-06_14-35-25_wandb-8la4o48g/models/model_step_5000134656.pt|${SKILL}/bones129k_skill_state_occupancy_h10_z256_seed0/encoder/checkpoints/latest.pt|258|256|${TUNED_CELLS}|1|"
"skill_semimarkov_chain|${SKILL}/bones129k_skill_semimarkov_chain_h10_z256_seed0/rlopt_train/2026-08-06_14-35-25_wandb-k7s6uha0/models/model_step_5000134656.pt|${SKILL}/bones129k_skill_semimarkov_chain_h10_z256_seed0/encoder/checkpoints/latest.pt|258|256|${TUNED_CELLS}|1|"
"skill_endpoint_delta|${SKILL}/bones129k_skill_endpoint_delta_h10_z256_seed0/rlopt_train/2026-08-06_14-35-18_wandb-3296logf/models/model_step_5000134656.pt|${SKILL}/bones129k_skill_endpoint_delta_h10_z256_seed0/encoder/checkpoints/latest.pt|258|256|${TUNED_CELLS}|1|"
"stride5_det64|${STRIDE5}/det64_tracker/rlopt_train/2026-08-07_18-59-51_wandb-3up12ht3/models/model_step_5000134656.pt|${STRIDE5}/det64_encoder/checkpoints/latest.pt|66|64|${SCALED_CELLS}|5|"
"stride5_fsq64|${STRIDE5}/fsq64_tracker/rlopt_train/2026-08-07_19-00-45_wandb-qz095qxj/models/model_step_5000134656.pt|${STRIDE5}/fsq64_encoder/checkpoints/latest.pt|66|64|${SCALED_CELLS}|5|"
"stride5_gumbel64|${STRIDE5}/gumbel64_tracker/rlopt_train/2026-08-07_19-00-50_wandb-nsk6pjb3/models/model_step_4750049280.pt|${STRIDE5}/gumbel64_encoder/checkpoints/latest.pt|66|64|${SCALED_CELLS}|5|"
# The comparison target for every latent arm: the direct 38-D root_qpos
# command, renewed every control step. Its ONLY surviving checkpoint on ICE is
# at 7,600,078,848 frames, so this row has 52% MORE training than the 5B rows
# and the comparison FAVORS it. The report prints each row's frame count; do
# not read this line as frame-matched.
"root_qpos_explicit|${LATENT_SAMPLER}/bones129k_root_qpos_explicit_reset80_e16384_r24_10b_seed0/rlopt_train/2026-08-05_21-08-40_wandb-mmsa9roe/models/model_step_7600078848.pt|-|0|0|${TUNED_CELLS}|1|"
)

ARMS="${ARMS:-critic_no_latent skill_state_occupancy skill_semimarkov_chain skill_endpoint_delta stride5_det64 stride5_fsq64 stride5_gumbel64 root_qpos_explicit}"

RUNTIME_BODY_NAMES="[pelvis,left_hip_roll_link,left_knee_link,left_ankle_roll_link,right_hip_roll_link,right_knee_link,right_ankle_roll_link,torso_link,left_shoulder_roll_link,left_elbow_link,left_wrist_yaw_link,right_shoulder_roll_link,right_elbow_link,right_wrist_yaw_link]"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

pull_verified() {
    local remote="$1" local_path="$2"
    mkdir -p "$(dirname "${local_path}")"
    if [[ ! -s "${local_path}" ]]; then
        log "pull ${remote}"
        rsync -q --partial --inplace -e "ssh -o BatchMode=yes" "ice:${remote}" "${local_path}" || return 1
    fi
    sha256sum "${local_path}" | awk '{print $1}' > "${local_path}.sha256"
    printf '%s\n' "${remote}" > "${local_path}.remote_path"
}

run_eval() {
    local pass="$1" arm="$2" checkpoint="$3" encoder="$4"
    local command_dim="$5" code_dim="$6" cells="$7" stride="$8" extra_hydra="$9" out="${10}"
    local extra=() hydra_extra=() interface=() rc
    [[ "${pass}" == "full_horizon" ]] && extra+=(--disable_early_terminations)
    [[ -n "${extra_hydra}" ]] && hydra_extra+=("${extra_hydra}")

    if [[ "${encoder}" == "-" ]]; then
        # Explicit arm: a raw 38-D root_qpos command renewed every control
        # step. No encoder, no latent dim, no hold.
        interface=(
            env.command_interface.actor=explicit
            env.command_interface.actor.components=[joint_qpos,root_pos,root_ori]
            agent.ipmd.use_latent_command=false
            agent.command_space=root_qpos
            agent.command_components=[joint_qpos,root_pos,root_ori]
            agent.ipmd.command_source=random
            agent.ipmd.hl_skill_checkpoint_path=null
        )
    else
        interface=(
            "env.command_interface.actor.dim=${command_dim}"
            "agent.ipmd.latent_dim=${command_dim}"
            agent.ipmd.command_source=hl_skill
            "agent.ipmd.hl_skill_checkpoint_path=${encoder}"
            agent.ipmd.hl_skill_horizon_steps=10
            agent.ipmd.hl_skill_command_mode=z
            agent.ipmd.latent_steps_min=10
            agent.ipmd.latent_steps_max=10
            agent.ipmd.latent_learning.code_period=10
            agent.ipmd.latent_learning.command_phase_mode=sin_cos
            "agent.ipmd.latent_learning.code_latent_dim=${code_dim}"
            agent.ipmd.hl_skill_finetune_enabled=false
        )
    fi

    local ranks=()
    local r
    for ((r = RANK_START; r <= RANK_END; r++)); do ranks+=("${r}"); done

    log "${arm} ${pass}"
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
        --output_json "${out}" --label "${arm}_${pass}" --headless \
        "${extra[@]}" \
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
        "${interface[@]}" \
        "${hydra_extra[@]}" \
        env.expert_macro_state_terms=[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b] \
        "env.expert_macro_frame_stride=${stride}" \
        env.terminations.anchor_pos.params.threshold=0.25 \
        env.terminations.anchor_pos.params.down_threshold=0.25 \
        env.terminations.anchor_ori.params.threshold=1.0 \
        env.terminations.ee_body_pos.params.threshold=0.25 \
        env.terminations.ee_body_pos.params.down_threshold=0.25 \
        env.terminations.foot_pos_xyz=null \
        env.terminations.base_too_low=null \
        "agent.policy.num_cells=${cells}" \
        agent.policy.activation_fn=silu \
        "agent.value_function.num_cells=${cells}" \
        agent.value_function.activation_fn=silu > "${out}.log" 2>&1
    rc=$?
    (( rc == 0 )) || return "${rc}"
    if grep -Eq 'overflow.*increase njmax|nefc overflow' "${out}.log"; then
        log "[FAIL] solver constraint buffer overflow in ${out}.log"
        return 1
    fi
}

validate_result() {
    local pass="$1" json="$2"
    python3 - "${pass}" "${json}" <<'PY'
import hashlib
import json
import sys

pas, path = sys.argv[1:]
d = json.load(open(path, encoding="utf-8"))
a, m = d["aggregate"], d["metadata"]
assert a["done_rate"] == 1.0, a["done_rate"]
assert a["time_out_rate"] == 0.0, a["time_out_rate"]
assert d["stop_reason"] == "all_envs_done", d["stop_reason"]
assert m["action_sampling"] == "mode", m["action_sampling"]
assert m["num_envs"] == 4096, m["num_envs"]
assert m["randomization_profile"] == "no_push", m["randomization_profile"]
ranks = [e["trajectory_rank"] for e in d["per_environment"]]
assert sorted(ranks) == list(range(12288, 16384)), "rank block changed"
if pas == "sonic":
    assert m["early_terminations_enabled"] is True
else:
    assert m["early_terminations_enabled"] is False
digest = hashlib.sha256(
    json.dumps(ranks, separators=(",", ":")).encode() + b"\n"
).hexdigest()
print(f"OK {pas} rank_sha256={digest}")
PY
}

evaluate_arm() {
    local row arm tracker encoder command_dim code_dim cells stride extra
    for row in "${ARMS_TABLE[@]}"; do
        IFS='|' read -r arm tracker encoder command_dim code_dim cells stride extra <<<"${row}"
        [[ "${arm}" == "$1" ]] || continue
        local out_dir="${OUTPUT_ROOT}/${arm}"
        mkdir -p "${out_dir}"
        local local_tracker="${out_dir}/$(basename "${tracker}")"
        local local_encoder="${out_dir}/encoder.pt"
        pull_verified "${tracker}" "${local_tracker}" || return 1
        if [[ "${encoder}" == "-" ]]; then
            local_encoder="-"
        else
            pull_verified "${encoder}" "${local_encoder}" || return 1
        fi
        local pass
        for pass in ${PASSES}; do
            if [[ ! -s "${out_dir}/${pass}.json" ]]; then
                run_eval "${pass}" "${arm}" "${local_tracker}" "${local_encoder}" \
                    "${command_dim}" "${code_dim}" "${cells}" "${stride}" "${extra}" \
                    "${out_dir}/${pass}.json" || {
                    log "[FAIL] ${arm} ${pass}; see ${out_dir}/${pass}.json.log"
                    return 1
                }
            fi
            validate_result "${pass}" "${out_dir}/${pass}.json" \
                > "${out_dir}/${pass}.validation" || return 1
        done
        return 0
    done
    log "[FATAL] unknown arm: $1"
    return 2
}

report() {
    python3 - "${OUTPUT_ROOT}" "${REPO_ROOT}" <<'PY'
import json
import pathlib
import sys

out_root, repo_root = (pathlib.Path(p) for p in sys.argv[1:])
rows = []


def add(name, path, frames):
    if not path.is_file():
        return
    d = json.load(open(path, encoding="utf-8"))
    a = d["aggregate"]
    s = d.get("successful_metrics", {}).get("tracking_mpjpe_mm", {})
    rows.append((name, frames, a["tracking_success_rate"], s.get("mean")))


scale = repo_root / "logs/bones129k_sonic_fsq_scale_eval/5000134656"
for arm in ("old_z256", "fsq64_tuned", "fsq64_sonic"):
    add(arm, scale / arm / "sonic.json", 5000134656)
for d in sorted(out_root.glob("*/sonic.json")):
    ckpt = next(d.parent.glob("model_step_*.pt"), None)
    frames = int(ckpt.stem.split("_")[-1]) if ckpt else 0
    add(d.parent.name, d, frames)

print(f"{'arm':24} {'frames':>13} {'SONIC SR':>9} {'succ MPJPE-L mm':>16}")
for name, frames, sr, mpjpe in sorted(rows, key=lambda r: -r[2]):
    mp = "-" if mpjpe is None else f"{mpjpe:.2f}"
    print(f"{name:24} {frames:>13} {sr:>9.4f} {mp:>16}")
print("\nreleased SONIC, same ranks: SR 0.9937, success-only MPJPE-L 28.65 mm")
PY
}

if [[ "${MODE}" == "report" ]]; then
    report
    exit 0
fi

mkdir -p "${OUTPUT_ROOT}"
failed=()
for arm in ${ARMS}; do
    evaluate_arm "${arm}" || failed+=("${arm}")
done
report
if ((${#failed[@]})); then
    log "[FAIL] arms: ${failed[*]}"
    exit 1
fi
