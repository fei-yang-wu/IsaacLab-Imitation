#!/usr/bin/env bash
set -uo pipefail

# Compare the latest finished 2026-08-08/09 ICE low-level trackers with one
# fixed local protocol. The selected10 suite uses the canonical language-ten
# motions. The scoreboard4096 suite uses the frozen ranks 12288-16383 from the
# BONES-129k scoreboard.
#
#   ./run.sh
#   SUITES=selected10 ARMS="expert_heading critic_no_latent" ./run.sh
#   ./run.sh --report

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
while [[ ! -f "${REPO_ROOT}/pixi.toml" ]]; do
    [[ "${REPO_ROOT}" != "/" ]] || { echo "[FATAL] repository root not found" >&2; exit 2; }
    REPO_ROOT="$(dirname "${REPO_ROOT}")"
done
cd "${REPO_ROOT}"

MODE=run
[[ "${1:-}" == "--report" ]] && MODE=report

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${REPO_ROOT}/logs/downloaded_checkpoints/bones129k_recent_ice}"
SHARED_ENCODER="${SHARED_ENCODER:-${REPO_ROOT}/logs/downloaded_checkpoints/bones129k_anchor_frame/expert_heading_encoder_latest.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/logs/bones129k_recent_ice_local_eval}"
SELECTED_MANIFEST="${SELECTED_MANIFEST:-${REPO_ROOT}/data/bones_seed_language10_l2t_eval_v1/manifests/g1_bones_seed_language10_l2t_eval_v1_manifest.json}"
SELECTED_CACHE="${SELECTED_CACHE:-${REPO_ROOT}/data/bones_seed_language10_l2t_eval_v1/zarr/g1_bones_seed_language10_l2t_eval_v1}"
REFERENCE_ARRAYS="${REFERENCE_ARRAYS:-${REPO_ROOT}/data/bones_seed/reference_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
PERSIST_ID="${PERSIST_ID:-bones_seed_sonic_full_129785@e714bbff}"

SUITES="${SUITES:-selected10 scoreboard4096}"
ARMS="${ARMS:-expert_heading critic_no_latent fsq64_heading_critic_no_latent z256_scaled ee_reward encoder_finetune fullbody_encoder}"
SELECTED_MAX_STEPS="${SELECTED_MAX_STEPS:-1200}"
SCOREBOARD_MAX_STEPS="${SCOREBOARD_MAX_STEPS:-10000}"
SEED=0

TUNED_CELLS="[1024,1024,512]"
SCALED_CELLS="[2048,2048,1024,1024,512,512]"
ROOT_QPOS_TERMS="[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b]"
FULL_BODY_TERMS="[expert_motion,expert_anchor_pos_b,expert_anchor_ori_b]"
RUNTIME_BODY_NAMES="[pelvis,left_hip_roll_link,left_knee_link,left_ankle_roll_link,right_hip_roll_link,right_knee_link,right_ankle_roll_link,torso_link,left_shoulder_roll_link,left_elbow_link,left_wrist_yaw_link,right_shoulder_roll_link,right_elbow_link,right_wrist_yaw_link]"

# arm | tracker | encoder | command dim | code dim | cells | macro terms |
# anchor mode | critic channels | encoder fine-tune | reward override
ARMS_TABLE=(
"expert_heading|${CHECKPOINT_ROOT}/expert_heading/model_step_7450263552.pt|${SHARED_ENCODER}|258|256|${TUNED_CELLS}|${ROOT_QPOS_TERMS}|expert_heading|default|false|"
"critic_no_latent|${CHECKPOINT_ROOT}/critic_no_latent/model_step_7550140416.pt|${SHARED_ENCODER}|258|256|${TUNED_CELLS}|${ROOT_QPOS_TERMS}|expert_heading|reference|false|"
"fsq64_heading_critic_no_latent|${CHECKPOINT_ROOT}/fsq64_heading_critic_no_latent/model_step_5750390784.pt|${CHECKPOINT_ROOT}/encoders/fsq64_heading.pt|66|64|${SCALED_CELLS}|${ROOT_QPOS_TERMS}|expert_heading|reference|false|"
"z256_scaled|${CHECKPOINT_ROOT}/z256_scaled/model_step_5750390784.pt|${CHECKPOINT_ROOT}/encoders/z256_scaled.pt|258|256|${SCALED_CELLS}|${ROOT_QPOS_TERMS}|robot|default|false|"
"ee_reward|${CHECKPOINT_ROOT}/ee_reward/model_step_7500201984.pt|${SHARED_ENCODER}|258|256|${TUNED_CELLS}|${ROOT_QPOS_TERMS}|expert_heading|reference|false|env.rewards.motion_ee_pos.weight=2.0"
"encoder_finetune|${CHECKPOINT_ROOT}/encoder_finetune/model_step_6350045184.pt|${SHARED_ENCODER}|258|256|${TUNED_CELLS}|${ROOT_QPOS_TERMS}|expert_heading|reference|true|"
"fullbody_encoder|${CHECKPOINT_ROOT}/fullbody_encoder/model_step_4850319360.pt|${CHECKPOINT_ROOT}/encoders/fullbody_heading.pt|258|256|${TUNED_CELLS}|${FULL_BODY_TERMS}|expert_heading|reference|false|"
)

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

require_inputs() {
    local path
    for path in "${SELECTED_MANIFEST}" "${SELECTED_CACHE}" "${REFERENCE_ARRAYS}"; do
        [[ -e "${path}" ]] || { log "[FATAL] missing input: ${path}"; return 1; }
    done
    local row arm tracker encoder rest
    for row in "${ARMS_TABLE[@]}"; do
        IFS='|' read -r arm tracker encoder rest <<<"${row}"
        [[ -s "${tracker}" ]] || { log "[FATAL] missing tracker: ${tracker}"; return 1; }
        [[ -s "${encoder}" ]] || { log "[FATAL] missing encoder: ${encoder}"; return 1; }
    done
}

run_eval() {
    local suite="$1" arm="$2" tracker="$3" encoder="$4" command_dim="$5"
    local code_dim="$6" cells="$7" macro_terms="$8" anchor_mode="$9"
    local critic_channels="${10}" finetune="${11}" reward_override="${12}" out="${13}"
    local num_envs steps
    local data_args=() rank_args=() critic_args=() finetune_args=() reward_args=()

    if [[ "${suite}" == "selected10" ]]; then
        num_envs=10
        steps="${SELECTED_MAX_STEPS}"
        data_args+=(--motion_manifest "${SELECTED_MANIFEST}" --dataset_path "${SELECTED_CACHE}")
    elif [[ "${suite}" == "scoreboard4096" ]]; then
        num_envs=4096
        steps="${SCOREBOARD_MAX_STEPS}"
        data_args+=(
            env.data.manifest=null
            "env.data.reference_arrays_dir=${REFERENCE_ARRAYS}"
            "env.data.persist_id=${PERSIST_ID}"
            env.data.reference_arrays_resident=false
            env.data.reference_arrays_warm_workers=8
            # The local RTX A4500 has 20 GiB. The 4,096-environment Newton
            # scene uses about 13 GiB, so the 49 GiB runtime arrays and the
            # 8-14 GiB macro cache must stay on host memory.
            env.data.runtime_cache_device=cpu
            env.data.reference_prefetch_mode=off
            env.data.macro_cache_device=cpu
            "env.data.runtime_cache_body_names=${RUNTIME_BODY_NAMES}"
        )
        local ranks=() rank
        for ((rank = 12288; rank <= 16383; rank++)); do ranks+=("${rank}"); done
        rank_args+=(--trajectory_ranks "${ranks[@]}")
    else
        log "[FATAL] unknown suite: ${suite}"
        return 2
    fi

    [[ "${critic_channels}" == "default" ]] || \
        critic_args+=("env.command_interface.critic_channels=[${critic_channels}]")
    [[ -z "${reward_override}" ]] || reward_args+=("${reward_override}")
    if [[ "${finetune}" == "true" ]]; then
        finetune_args+=(
            agent.ipmd.hl_skill_finetune_enabled=true
            agent.ipmd.hl_skill_lr=3.0e-5
            agent.ipmd.hl_skill_pg_coeff=0.05
            agent.ipmd.hl_skill_offline_diffsr_coeff=1.0
            agent.ipmd.hl_skill_anchor_coeff=0.01
            agent.ipmd.hl_skill_grad_clip_norm=1.0
            agent.ipmd.hl_skill_offline_batch_size=8192
            agent.ipmd.hl_skill_update_interval=1
            agent.ipmd.hl_skill_train_diffsr=false
        )
    else
        finetune_args+=(agent.ipmd.hl_skill_finetune_enabled=false)
    fi

    mkdir -p "$(dirname "${out}")"
    log "${suite}: ${arm}"
    env TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 \
        HYDRA_FULL_ERROR=1 TORCHDYNAMO_DISABLE=1 \
        pixi run -e isaaclab python -u \
        -m imitation_experiments.lowlevel.evaluate_checkpoint \
        --task Isaac-Imitation-G1-v2 --algo IPMD \
        --checkpoint "${tracker}" \
        --agent_entry_point rlopt_ipmd_tuned_cfg_entry_point \
        --randomization no_push --action_sampling mode \
        --num_envs "${num_envs}" --steps "${steps}" --seed "${SEED}" \
        --reference_start_frame 0 --reset_schedule sequential \
        --output_json "${out}" --label "${arm}_${suite}" \
        "${data_args[@]}" "${rank_args[@]}" \
        --kit_args=--/app/extensions/fsWatcherEnabled=false \
        physics=newton_mjwarp \
        env.sim.physics.solver_cfg.njmax=320 \
        env.sim.physics.solver_cfg.nconmax=200 \
        env.events.push_robot=null \
        env.command_interface.actor=latent \
        "env.command_interface.actor.dim=${command_dim}" \
        env.command_interface.encoder=single \
        "${critic_args[@]}" \
        "env.expert_macro_state_terms=${macro_terms}" \
        env.expert_macro_frame_stride=1 \
        "env.expert_macro_anchor_mode=${anchor_mode}" \
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
        "${finetune_args[@]}" \
        agent.logger.backend= \
        "agent.policy.num_cells=${cells}" \
        agent.policy.activation_fn=silu \
        "agent.value_function.num_cells=${cells}" \
        agent.value_function.activation_fn=silu \
        env.rewards.action_rate_l2.weight=0.0 \
        env.rewards.tracking_reward_points.weight=4.0 \
        "${reward_args[@]}" \
        env.terminations.anchor_pos.params.threshold=0.25 \
        env.terminations.anchor_pos.params.down_threshold=0.25 \
        env.terminations.anchor_ori.params.threshold=1.0 \
        env.terminations.ee_body_pos.params.threshold=0.25 \
        env.terminations.ee_body_pos.params.down_threshold=0.25 \
        env.terminations.foot_pos_xyz=null \
        env.terminations.base_too_low=null > "${out}.log" 2>&1
}

validate_result() {
    local suite="$1" json="$2"
    python3 - "${suite}" "${json}" <<'PY'
import hashlib
import json
import sys

suite, path = sys.argv[1:]
with open(path, encoding="utf-8") as file:
    result = json.load(file)
aggregate = result["aggregate"]
metadata = result["metadata"]
expected_envs = 10 if suite == "selected10" else 4096
assert aggregate["done_rate"] == 1.0, aggregate["done_rate"]
assert aggregate["time_out_rate"] == 0.0, aggregate["time_out_rate"]
assert result["stop_reason"] == "all_envs_done", result["stop_reason"]
assert metadata["num_envs"] == expected_envs, metadata["num_envs"]
assert metadata["seed"] == 0, metadata["seed"]
assert metadata["reference_start_frame"] == 0, metadata["reference_start_frame"]
assert metadata["action_sampling"] == "mode", metadata["action_sampling"]
assert metadata["randomization_profile"] == "no_push", metadata["randomization_profile"]
assert metadata["randomization_kept"] == {"startup": True, "reset": True, "push": False}
assert metadata["early_terminations_enabled"] is True
ranks = [row["trajectory_rank"] for row in result["per_environment"]]
expected_ranks = list(range(10)) if suite == "selected10" else list(range(12288, 16384))
assert ranks == expected_ranks, "trajectory rank order changed"
digest = hashlib.sha256(json.dumps(ranks, separators=(",", ":")).encode() + b"\n").hexdigest()
allowed = {"anchor_pos", "anchor_ori", "ee_body_pos", "reference_finished"}
observed = set(aggregate["termination_cause_env_counts"])
assert not (observed - allowed - {"time_out"}), sorted(observed - allowed - {"time_out"})
print(f"OK suite={suite} envs={expected_envs} rank_sha256={digest}")
PY
}

evaluate_arm() {
    local target="$1" row arm tracker encoder command_dim code_dim cells
    local macro_terms anchor_mode critic_channels finetune reward_override suite out
    for row in "${ARMS_TABLE[@]}"; do
        IFS='|' read -r arm tracker encoder command_dim code_dim cells macro_terms \
            anchor_mode critic_channels finetune reward_override <<<"${row}"
        [[ "${arm}" == "${target}" ]] || continue
        for suite in ${SUITES}; do
            out="${OUTPUT_ROOT}/${suite}/${arm}/sonic.json"
            if [[ ! -s "${out}" ]]; then
                run_eval "${suite}" "${arm}" "${tracker}" "${encoder}" \
                    "${command_dim}" "${code_dim}" "${cells}" "${macro_terms}" \
                    "${anchor_mode}" "${critic_channels}" "${finetune}" \
                    "${reward_override}" "${out}" || {
                    log "[FAIL] ${suite} ${arm}; see ${out}.log"
                    return 1
                }
            else
                log "skip existing ${suite}: ${arm}"
            fi
            validate_result "${suite}" "${out}" > "${out}.validation" || return 1
        done
        return 0
    done
    log "[FATAL] unknown arm: ${target}"
    return 2
}

report() {
    python3 - "${OUTPUT_ROOT}" <<'PY'
import json
import math
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
print(f"{'suite':15} {'arm':34} {'n':>5} {'SONIC SR':>9} {'succ MPJPE-L mm':>16}")
for suite in ("selected10", "scoreboard4096"):
    for path in sorted((root / suite).glob("*/sonic.json")):
        with path.open(encoding="utf-8") as file:
            result = json.load(file)
        aggregate = result["aggregate"]
        metric = result.get("successful_metrics", {}).get("tracking_mpjpe_mm", {})
        value = metric.get("mean")
        mpjpe = "-" if value is None or not math.isfinite(value) else f"{value:.2f}"
        print(
            f"{suite:15} {path.parent.name:34} "
            f"{aggregate['num_evaluated_envs']:>5} "
            f"{aggregate['completed_tracking_success_rate']:>9.4f} {mpjpe:>16}"
        )
PY
}

if [[ "${MODE}" == "report" ]]; then
    report
    exit 0
fi

require_inputs || exit 1
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
