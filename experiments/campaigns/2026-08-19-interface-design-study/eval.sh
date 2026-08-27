#!/usr/bin/env bash
# Score interface-design-study checkpoints.
#
# Every per-arm interface setting -- command width, hold, code width, phase,
# command mode, macro terms, stride, anchor -- is read back out of
# `campaign.yaml`, so evaluation cannot drift from training. Do not hardcode
# any of them here; the 2026-08-15 runner did and needed a warning comment
# about the three settings that silently differed per arm.
#
# Rows:
#   milestone  every mirrored checkpoint on `paper_milestone_testbed256_v1`
#              (256 clips, testbed population) -- the budget axis
#   clean      the final checkpoint on `paper_testbed4096_v1`      -- row of record
#   robust     the final checkpoint on `paper_testbed4096_robust_v1`
#
#   ./eval.sh                                   # every mirrored arm, all rows
#   ARMS="ctrl obj_recon" ROWS=clean ./eval.sh
#   ./eval.sh --report
set -uo pipefail
CAMPAIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${CAMPAIGN_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

MIRROR="${MIRROR:-${REPO_ROOT}/logs/interface_design_study_mirror}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/logs/interface_design_study_eval}"
REFERENCE_ARRAYS="${REFERENCE_ARRAYS:-/mnt/hsstorage/fwu91/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
PERSIST_ID="${PERSIST_ID:-bones_seed_sonic_full_129785@e714bbff}"
MAX_STEPS="${MAX_STEPS:-10000}"
SEEDS="${SEEDS:-0}"
ROWS="${ROWS:-milestone clean robust}"
SCALED_CELLS="[2048,2048,1024,1024,512,512]"
RUNTIME_BODY_NAMES="[pelvis,left_hip_roll_link,left_knee_link,left_ankle_roll_link,right_hip_roll_link,right_knee_link,right_ankle_roll_link,torso_link,left_shoulder_roll_link,left_elbow_link,left_wrist_yaw_link,right_shoulder_roll_link,right_elbow_link,right_wrist_yaw_link]"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

board_for() {
    case "$1" in
        milestone) echo bones_milestone_testbed256_v1 ;;
        clean|robust) echo bones_testbed4096_v1 ;;
    esac
}
randomization_for() { [[ "$1" == "robust" ]] && echo no_push || echo none; }

# Arm interface, straight from the campaign. One line per field, so a missing
# field is a loud failure rather than a silent default.
arm_vars() {
    pixi run python - "$1" <<'PY'
import sys, yaml, pathlib
name = sys.argv[1]
campaign = yaml.safe_load(
    pathlib.Path(
        "experiments/campaigns/2026-08-19-interface-design-study/campaign.yaml"
    ).read_text()
)
base = dict(campaign["vars"])
arm = campaign["arms"][name]["vars"]
merged = {**base, **arm}
z_dim = int(merged["z_dim"])
code_latent = merged.get("code_latent_dim", z_dim)
if isinstance(code_latent, str) and code_latent.startswith("${"):
    code_latent = z_dim
fields = {
    "z_dim": z_dim,
    "command_dim": int(merged["command_dim"]),
    "code_latent_dim": int(code_latent),
    "hold": int(merged["hold"]),
    "phase_mode": merged["phase_mode"],
    "command_mode": merged["command_mode"],
    "macro_terms": merged["macro_terms"],
    "stride": int(merged["stride"]),
    "anchor_mode": merged["anchor_mode"],
    "tier": int(merged.get("tier", 1)),
}
# Single-quoted: `macro_terms` is a bracket expression, and an unquoted
# assignment would be subject to pathname expansion when the caller evals it.
for key, value in fields.items():
    print(f"{key}='{value}'")
PY
}

if [[ "${ARMS:-}" == "" ]]; then
    mapfile -t all_arms < <(pixi run python -c "
import yaml
campaign = yaml.safe_load(open('experiments/campaigns/2026-08-19-interface-design-study/campaign.yaml'))
print('\n'.join(campaign['arms']))
")
    ARMS="${all_arms[*]}"
fi

out_for() { printf '%s/%s_seed%s_%s_f%s.json' "${OUTPUT_ROOT}" "$1" "$2" "$3" "$4"; }

report() {
    shopt -s nullglob
    local files=("${OUTPUT_ROOT}"/*.json)
    [[ "${#files[@]}" -gt 0 ]] || { log "[INFO] nothing scored yet"; return 0; }
    for out in "${files[@]}"; do
        printf '%-56s ' "$(basename "${out}" .json)"
        pixi run python -m imitation_experiments.evaluation.summarize_paper_boards "${out}"
    done
}

if [[ "${1:-}" == "--report" ]]; then
    report
    exit $?
fi

[[ -s "${REFERENCE_ARRAYS}/reference_arrays_manifest.json" ]] || {
    log "[FATAL] reference arrays missing: ${REFERENCE_ARRAYS}"; exit 2; }
mkdir -p "${OUTPUT_ROOT}"

for arm in ${ARMS}; do
  for seed in ${SEEDS}; do
    tree="${MIRROR}/${arm}_seed${seed}"
    encoder="${tree}/encoder/checkpoints/latest.pt"
    [[ -d "${tree}/tracker" ]] || { log "[SKIP] no mirror ${tree}"; continue; }
    [[ -s "${encoder}" ]] || { log "[SKIP] no encoder ${encoder}"; continue; }

    if ! vars_text="$(arm_vars "${arm}" 2>/dev/null)"; then
        log "[SKIP] ${arm} is not an arm of this campaign"; continue
    fi
    eval "${vars_text}"

    # Checkpoint tree names carry the TRUE cumulative frame count; the
    # per-segment step counter restarts on every chained resume.
    mapfile -t frames < <(ls -1 "${tree}/tracker" 2>/dev/null | sed -n 's/^f\([0-9]\+\)$/\1/p' | sort -n)
    [[ "${#frames[@]}" -gt 0 ]] || { log "[SKIP] no checkpoints in ${tree}/tracker"; continue; }
    final="${frames[-1]}"

    for row in ${ROWS}; do
        board="$(board_for "${row}")"
        [[ -n "${board}" ]] || { log "[SKIP] unknown row ${row}"; continue; }
        profile="$(randomization_for "${row}")"

        if [[ "${row}" == "milestone" ]]; then
            selected=("${frames[@]}")
        else
            selected=("${final}")
        fi

        mapfile -t ranks < <(pixi run python -c "
from imitation_experiments.evaluation.protocol import BOARDS
print('\n'.join(str(case.trajectory_rank) for case in BOARDS['${board}'].cases))
")
        [[ "${#ranks[@]}" -gt 0 ]] || { log "[FATAL] board ${board} returned no ranks"; exit 2; }

        for frame in "${selected[@]}"; do
            checkpoint="${tree}/tracker/f${frame}/models/model_step_${frame}.pt"
            [[ -s "${checkpoint}" ]] || { log "[SKIP] missing ${checkpoint}"; continue; }
            out="$(out_for "${arm}" "${seed}" "${row}" "${frame}")"
            [[ -s "${out}" ]] && { log "[SKIP] already scored $(basename "${out}")"; continue; }

            log "${arm} seed${seed} ${row} f${frame} (z ${z_dim}, cmd ${command_dim}, hold ${hold}, ${board}, ${#ranks[@]} clips)"
            env TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 \
                HYDRA_FULL_ERROR=1 TORCHDYNAMO_DISABLE=1 \
                pixi run -e isaaclab python -u \
                -m imitation_experiments.lowlevel.evaluate_checkpoint \
                --task Isaac-Imitation-G1-v2 --algo IPMD \
                --checkpoint "${checkpoint}" \
                --agent_entry_point rlopt_ipmd_tuned_cfg_entry_point \
                --randomization "${profile}" --action_sampling mode \
                --num_envs "${#ranks[@]}" --steps "${MAX_STEPS}" --seed 0 \
                --reference_start_frame 0 --reset_schedule sequential \
                --trajectory_ranks "${ranks[@]}" \
                --output_json "${out}" \
                --label "${arm}_seed${seed}_${row}_f${frame}" --headless \
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
                "agent.ipmd.hl_skill_command_mode=${command_mode}" \
                "agent.ipmd.latent_steps_min=${hold}" \
                "agent.ipmd.latent_steps_max=${hold}" \
                "agent.ipmd.latent_learning.code_period=${hold}" \
                "agent.ipmd.latent_learning.command_phase_mode=${phase_mode}" \
                "agent.ipmd.latent_learning.code_latent_dim=${code_latent_dim}" \
                agent.ipmd.hl_skill_finetune_enabled=false \
                "env.expert_macro_state_terms=${macro_terms}" \
                "env.expert_macro_frame_stride=${stride}" \
                "env.expert_macro_anchor_mode=${anchor_mode}" \
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
                log "[FAIL] ${arm} seed${seed} ${row} f${frame} exit ${rc}: $(tail -3 "${out}.log" | tr '\n' ' ')"
                continue
            fi
            if grep -Eq 'overflow.*increase njmax|nefc overflow' "${out}.log"; then
                log "[FAIL] ${arm} seed${seed} ${row} f${frame}: solver constraint buffer overflow"
                continue
            fi
            log "[OK] $(basename "${out}")"
        done
    done
  done
done

report
