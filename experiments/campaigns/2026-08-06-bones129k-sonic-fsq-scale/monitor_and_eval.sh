#!/usr/bin/env bash
set -uo pipefail

# Pull matched checkpoints for the old continuous-z256 tracker and the two new
# pretrained FSQ64 trackers, then evaluate them locally. A frame target becomes
# eligible only after BOTH new trackers have crossed it. Re-running is safe:
# completed arm/target pairs are skipped.
#
#   ./monitor_and_eval.sh             # one pull/evaluate cycle
#   ./monitor_and_eval.sh --latest    # newest checkpoint shared by FSQ arms
#   ./monitor_and_eval.sh --poll-latest # newest shared checkpoint every 2h
#   ./monitor_and_eval.sh --report    # status only
#   ./monitor_and_eval.sh --watch     # poll until every target is evaluated
#
# Override THRESHOLDS for a focused pass, for example:
#   THRESHOLDS=250000000 ./monitor_and_eval.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
while [[ ! -x "${REPO_ROOT}/docker/cluster/cluster_interface.sh" ]]; do
    [[ "${REPO_ROOT}" != "/" ]] || { echo "[FATAL] repository root not found" >&2; exit 2; }
    REPO_ROOT="$(dirname "${REPO_ROOT}")"
done
cd "${REPO_ROOT}"

MODE="once"
case "${1:-}" in
    --watch) MODE="watch" ;;
    --latest) MODE="latest" ;;
    --poll-latest) MODE="poll_latest" ;;
    --report) MODE="report" ;;
    --once|"") MODE="once" ;;
    *) echo "[FATAL] unknown argument: $1" >&2; exit 2 ;;
esac

INTERVAL="${INTERVAL:-7200}"
THRESHOLDS="${THRESHOLDS:-250000000 500000000 750000000 1000000000 1500000000 2000000000 3000000000 4000000000 5000000000}"
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/logs/bones129k_sonic_fsq_scale_eval}"
REFERENCE_ARRAYS="${REFERENCE_ARRAYS:-/mnt/hsstorage/fwu91/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
PERSIST_ID="${PERSIST_ID:-bones_seed_sonic_full_129785@e714bbff}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_STEPS="${MAX_STEPS:-10000}"
SEED="${SEED:-0}"
ACTIVE_JOB_IDS="${ACTIVE_JOB_IDS:-5570680,5570936,5571183}"

BASELINE_ENCODER_REMOTE="${REMOTE_DATA_ROOT}/pretrain_store/bones129k_v2_root_qpos_det_sr_h10_z256_seed0/checkpoints/latest.pt"
FSQ64_ENCODER_REMOTE="${REMOTE_DATA_ROOT}/bones129k_sonic_fsq_scale/shared_scaled_fsq64_encoder/checkpoints/latest.pt"
BASELINE_ENCODER_SHA256="d191d8656620059a569edbad82ca182cb2d2f85839300153cb618d1e29f8c5e7"
FSQ64_ENCODER_SHA256="6a4a724872273a6a2850e433881e7746dbce2b7ccb92e4ee18153cffad77da14"

BASELINE_RUN="${REMOTE_DATA_ROOT}/bones129k_latent_sampler/bones129k_reset80_diffsr_reset80_e16384_r24_10b_seed0/rlopt_train"
TUNED_RUN="${REMOTE_DATA_ROOT}/bones129k_sonic_fsq_scale/tuned_tracker/rlopt_train"
SONIC_RUN="${REMOTE_DATA_ROOT}/bones129k_sonic_fsq_scale/sonic_tracker_h200_retry1/rlopt_train"
CRITIC_NO_LATENT_RUN="${REMOTE_DATA_ROOT}/bones129k_critic_ablation/bones129k_z256_critic_no_latent_e16384_r24_5b_seed0/rlopt_train"

# arm | run directory | encoder kind | command width | code width | network cells
RUNS=(
    "old_z256|${BASELINE_RUN}|z256|258|256|[1024,1024,512]"
    "old_z256_critic_no_latent|${CRITIC_NO_LATENT_RUN}|z256|258|256|[1024,1024,512]"
    "fsq64_tuned|${TUNED_RUN}|fsq64|66|64|[1024,1024,512]"
    "fsq64_sonic|${SONIC_RUN}|fsq64|66|64|[2048,2048,1024,1024,512,512]"
)

RUNTIME_BODY_NAMES="[pelvis,left_hip_roll_link,left_knee_link,left_ankle_roll_link,right_hip_roll_link,right_knee_link,right_ankle_roll_link,torso_link,left_shoulder_roll_link,left_elbow_link,left_wrist_yaw_link,right_shoulder_roll_link,right_elbow_link,right_wrist_yaw_link]"

ssh_ice() { ssh -o BatchMode=yes -o ConnectTimeout=20 ice "$@"; }
log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

latest_frames() {
    ssh_ice "find '$1' -type f -name 'model_step_*.pt' 2>/dev/null \
        | sed 's/.*model_step_//; s/\.pt//' | sort -n | tail -1"
}

checkpoint_at_or_after() {
    ssh_ice "find '$1' -type f -name 'model_step_*.pt' 2>/dev/null \
        | awk -F'model_step_' '{n=\$2; sub(/\.pt/,\"\",n); if (n+0 >= $2) print n+0, \$0}' \
        | sort -n | head -1 | cut -d' ' -f2-"
}

pull_verified() {
    local remote="$1" local_path="$2" expected_sha="${3:-}" actual_sha
    mkdir -p "$(dirname "${local_path}")"
    if [[ ! -s "${local_path}" ]]; then
        log "pull $(basename "${remote}")"
        rsync -q --partial --inplace -e "ssh -o BatchMode=yes" "ice:${remote}" "${local_path}" || return 1
    fi
    actual_sha="$(sha256sum "${local_path}" | awk '{print $1}')"
    if [[ -n "${expected_sha}" && "${actual_sha}" != "${expected_sha}" ]]; then
        log "[FAIL] hash mismatch for ${local_path}: ${actual_sha} != ${expected_sha}"
        return 1
    fi
    printf '%s\n' "${actual_sha}" > "${local_path}.sha256"
}

encoder_for() {
    local kind="$1"
    case "${kind}" in
        z256) printf '%s/encoders/old_z256.pt' "${OUTPUT_ROOT}" ;;
        fsq64) printf '%s/encoders/fsq64_scaled.pt' "${OUTPUT_ROOT}" ;;
        *) return 2 ;;
    esac
}

ensure_encoder() {
    local kind="$1" local_path
    local_path="$(encoder_for "${kind}")"
    case "${kind}" in
        z256) pull_verified "${BASELINE_ENCODER_REMOTE}" "${local_path}" "${BASELINE_ENCODER_SHA256}" ;;
        fsq64) pull_verified "${FSQ64_ENCODER_REMOTE}" "${local_path}" "${FSQ64_ENCODER_SHA256}" ;;
        *) return 2 ;;
    esac
}

run_eval() {
    local pass="$1" arm="$2" frames="$3" checkpoint="$4" encoder="$5"
    local command_dim="$6" code_dim="$7" cells="$8" out="$9"
    local extra=() command_interface_extra=() rc
    if [[ "${pass}" == "full_horizon" ]]; then
        extra+=(--disable_early_terminations)
    fi
    if [[ "${arm}" == "old_z256_critic_no_latent" ]]; then
        command_interface_extra+=("env.command_interface.critic_channels=[reference]")
    fi

    log "${arm} ${frames} frames: ${pass}"
    timeout 7200 env TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 \
        HYDRA_FULL_ERROR=1 TORCHDYNAMO_DISABLE=1 \
        pixi run -e isaaclab python -u \
        -m imitation_experiments.lowlevel.evaluate_checkpoint \
        --task Isaac-Imitation-G1-v2 --algo IPMD \
        --checkpoint "${checkpoint}" \
        --agent_entry_point rlopt_ipmd_tuned_cfg_entry_point \
        --randomization no_push --action_sampling mode \
        --num_envs "${NUM_ENVS}" --steps "${MAX_STEPS}" --seed "${SEED}" \
        --reference_start_frame 0 --reset_schedule sequential \
        --output_json "${out}" --label "${arm}_${frames}_${pass}" --headless \
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
        "env.command_interface.actor.dim=${command_dim}" \
        "${command_interface_extra[@]}" \
        env.expert_macro_state_terms=[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b] \
        env.terminations.anchor_pos.params.threshold=0.25 \
        env.terminations.anchor_pos.params.down_threshold=0.25 \
        env.terminations.anchor_ori.params.threshold=1.0 \
        env.terminations.ee_body_pos.params.threshold=0.25 \
        env.terminations.ee_body_pos.params.down_threshold=0.25 \
        env.terminations.foot_pos_xyz=null \
        env.terminations.base_too_low=null \
        "agent.ipmd.latent_dim=${command_dim}" \
        agent.ipmd.command_source=hl_skill \
        "agent.ipmd.hl_skill_checkpoint_path=${encoder}" \
        agent.ipmd.hl_skill_horizon_steps=10 \
        agent.ipmd.hl_skill_command_mode=z \
        agent.ipmd.latent_steps_min=10 \
        agent.ipmd.latent_steps_max=10 \
        agent.ipmd.latent_learning.code_period=10 \
        agent.ipmd.latent_learning.command_phase_mode=sin_cos \
        "agent.ipmd.latent_learning.code_latent_dim=${code_dim}" \
        agent.ipmd.hl_skill_finetune_enabled=false \
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
import json
import sys

pas, path = sys.argv[1:]
d = json.load(open(path, encoding="utf-8"))
a = d["aggregate"]
m = d["metadata"]
assert a["done_rate"] == 1.0, a["done_rate"]
assert a["time_out_rate"] == 0.0, a["time_out_rate"]
assert d["stop_reason"] == "all_envs_done", d["stop_reason"]
assert d["steps_run"] < d["max_steps"], (d["steps_run"], d["max_steps"])
assert m["action_sampling"] == "mode", m["action_sampling"]
assert m["randomization_profile"] == "no_push", m["randomization_profile"]
assert m["randomization_kept"] == {"startup": True, "reset": True, "push": False}
assert m["push_perturbation"]["enabled"] is False
if pas == "sonic":
    allowed = {"anchor_pos", "anchor_ori", "ee_body_pos", "reference_finished"}
    fired = {k for k, v in a["termination_cause_env_counts"].items() if v}
    assert fired <= allowed, fired
print("valid")
PY
}

evaluate_arm() {
    local arm="$1" run_dir="$2" encoder_kind="$3" command_dim="$4"
    local code_dim="$5" cells="$6" threshold="$7" remote checkpoint frames out_dir encoder pass

    out_dir="${OUTPUT_ROOT}/${threshold}/${arm}"
    if [[ -e "${out_dir}/.done" ]]; then
        return 0
    fi
    remote="$(checkpoint_at_or_after "${run_dir}" "${threshold}")"
    [[ -n "${remote}" ]] || { log "[WAIT] ${arm}: no checkpoint at ${threshold}"; return 1; }
    frames="$(basename "${remote}" | sed 's/model_step_//; s/\.pt//')"
    checkpoint="${out_dir}/$(basename "${remote}")"
    encoder="$(encoder_for "${encoder_kind}")"

    ensure_encoder "${encoder_kind}" || return 1
    pull_verified "${remote}" "${checkpoint}" || return 1
    printf '%s\n' "${remote}" > "${checkpoint}.remote_path"

    for pass in sonic full_horizon; do
        if [[ ! -s "${out_dir}/${pass}.json" ]]; then
            run_eval "${pass}" "${arm}" "${frames}" "${checkpoint}" "${encoder}" \
                "${command_dim}" "${code_dim}" "${cells}" "${out_dir}/${pass}.json" || {
                log "[FAIL] ${arm} ${pass}; see ${out_dir}/${pass}.json.log"
                return 1
            }
        fi
        validate_result "${pass}" "${out_dir}/${pass}.json" > "${out_dir}/${pass}.validation" || return 1
        jq -c '[.per_environment[].trajectory_rank]' "${out_dir}/${pass}.json" \
            | sha256sum | awk '{print $1}' > "${out_dir}/${pass}.rank_sha256"
    done
    touch "${out_dir}/.done"
}

report() {
    local threshold arm path sr successes mpjpe full_mpjpe rank_hash frames targets
    printf '\n%-12s %-13s %12s %10s %12s %12s %12s %s\n' \
        target arm actual_frames sonic_sr successes mpjpe_l_mm full_mpjpe rank_sha256
    targets="$(find "${OUTPUT_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
        | grep -E '^[0-9]+$' | sort -n)"
    [[ -n "${targets}" ]] || targets="${THRESHOLDS}"
    for threshold in ${targets}; do
        for arm in old_z256 old_z256_critic_no_latent fsq64_tuned fsq64_sonic; do
            path="${OUTPUT_ROOT}/${threshold}/${arm}"
            [[ -s "${path}/sonic.json" ]] || continue
            frames="$(jq -r '.metadata.checkpoint' "${path}/sonic.json" | sed 's/.*model_step_//; s/\.pt//')"
            sr="$(jq -r '.aggregate.completed_tracking_success_rate' "${path}/sonic.json")"
            successes="$(jq -r '.successful_metrics.tracking_mpjpe_mm.num_successful_envs' "${path}/sonic.json")/${NUM_ENVS}"
            mpjpe="$(jq -r '.successful_metrics.tracking_mpjpe_mm.mean' "${path}/sonic.json")"
            full_mpjpe="$(jq -r '.metrics.tracking_mpjpe_mm.mean // .successful_metrics.tracking_mpjpe_mm.mean // "-"' "${path}/full_horizon.json" 2>/dev/null || printf '%s' -)"
            rank_hash="$(cat "${path}/sonic.rank_sha256" 2>/dev/null || printf '%s' -)"
            printf '%-12s %-13s %12s %10s %12s %12s %12s %s\n' \
                "${threshold}" "${arm}" "${frames}" "${sr}" "${successes}" "${mpjpe}" "${full_mpjpe}" "${rank_hash}"
        done
    done
}

cycle() {
    local tuned_latest sonic_latest frontier cycle_thresholds threshold entry arm run_dir kind command_dim code_dim cells pending=0
    tuned_latest="$(latest_frames "${TUNED_RUN}")"
    sonic_latest="$(latest_frames "${SONIC_RUN}")"
    log "new checkpoint frontier: tuned=${tuned_latest:-none}, sonic=${sonic_latest:-none}"
    if [[ -z "${tuned_latest}" || -z "${sonic_latest}" ]]; then
        report
        return 1
    fi
    (( tuned_latest < sonic_latest )) && frontier="${tuned_latest}" || frontier="${sonic_latest}"
    cycle_thresholds="${THRESHOLDS}"
    [[ "${MODE}" == "latest" || "${MODE}" == "poll_latest" ]] && cycle_thresholds="${frontier}"

    for threshold in ${cycle_thresholds}; do
        if (( threshold > frontier )); then
            pending=1
            continue
        fi
        for entry in "${RUNS[@]}"; do
            IFS='|' read -r arm run_dir kind command_dim code_dim cells <<< "${entry}"
            [[ "${MODE}" == "report" ]] && continue
            evaluate_arm "${arm}" "${run_dir}" "${kind}" "${command_dim}" "${code_dim}" "${cells}" "${threshold}" \
                || pending=1
        done
    done
    report
    return "${pending}"
}

latest_completed_epoch() {
    {
        find "${OUTPUT_ROOT}" -mindepth 3 -maxdepth 3 -type f -name .done -printf '%T@\n' 2>/dev/null
        [[ ! -e "${OUTPUT_ROOT}/poll_latest.last_cycle" ]] \
            || stat -c '%Y' "${OUTPUT_ROOT}/poll_latest.last_cycle"
    } | sort -n | tail -1 | cut -d. -f1
}

wait_until_due() {
    local last_completed now due remaining chunk
    last_completed="$(latest_completed_epoch)"
    [[ -n "${last_completed}" ]] || return 0
    due=$((last_completed + INTERVAL))
    while true; do
        now="$(date +%s)"
        remaining=$((due - now))
        (( remaining > 0 )) || return 0
        (( remaining < 60 )) && chunk="${remaining}" || chunk=60
        sleep "${chunk}"
    done
}

tracked_jobs_active() {
    ssh_ice "squeue -h -j '${ACTIVE_JOB_IDS}'" | grep -q .
}

mkdir -p "${OUTPUT_ROOT}"
if [[ "${MODE}" == "poll_latest" ]]; then
    poll_rc=0
    exec 9>"${OUTPUT_ROOT}/poll_latest.lock"
    if ! flock -n 9; then
        log "another poll-latest monitor owns ${OUTPUT_ROOT}/poll_latest.lock"
        exit 0
    fi
    while true; do
        wait_until_due
        cycle || poll_rc=$?
        touch "${OUTPUT_ROOT}/poll_latest.last_cycle"
        if (( poll_rc == 0 )); then
            if ! tracked_jobs_active; then
                log "tracked jobs inactive and final shared checkpoint evaluated"
                break
            fi
        fi
        poll_rc=0
    done
elif [[ "${MODE}" == "watch" ]]; then
    while true; do
        cycle && { log "all requested targets evaluated"; break; }
        log "sleeping ${INTERVAL}s"
        sleep "${INTERVAL}"
    done
else
    cycle || true
fi
