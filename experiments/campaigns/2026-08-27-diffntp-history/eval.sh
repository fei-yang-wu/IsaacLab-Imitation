#!/usr/bin/env bash
# Score diffntp-history checkpoints.
#
# Every per-arm interface setting -- command width, hold, code width, phase,
# macro terms, stride, anchor -- and the five history overrides are read back
# out of `campaign.yaml`, so evaluation cannot drift from training. Do not
# hardcode them here.
#
# The history overrides are load-bearing: they widen the actor input, and the
# policy restore is strict. An eval that omits them fails to load the
# checkpoint.
#
# Rows:
#   milestone  every mirrored checkpoint on `bones_milestone_testbed256_v1`
#   clean      the final checkpoint on `bones_testbed4096_v1`   -- row of record
#   robust     the final checkpoint on `bones_testbed4096_robust_v1`
#
#   ./eval.sh                                        # every mirrored arm
#   ARMS=diffntp_pair_hist ROWS=robust ./eval.sh
#   FRAMES="2000289792 4000186368" ROWS=robust ./eval.sh
#   ./eval.sh --report
set -uo pipefail
CAMPAIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${CAMPAIGN_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

CAMPAIGN_YAML="experiments/campaigns/2026-08-27-diffntp-history/campaign.yaml"
MIRROR="${MIRROR:-${REPO_ROOT}/logs/diffntp_history_mirror}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/logs/diffntp_history_eval}"
REFERENCE_ARRAYS="${REFERENCE_ARRAYS:-/mnt/hsstorage/fwu91/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
PERSIST_ID="${PERSIST_ID:-bones_seed_sonic_full_129785@e714bbff}"
MAX_STEPS="${MAX_STEPS:-10000}"
SEEDS="${SEEDS:-0}"
ROWS="${ROWS:-clean robust}"
# Empty -> the final mirrored checkpoint for clean/robust, every one for
# milestone.
FRAMES="${FRAMES:-}"
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

arm_vars() {
    pixi run python - "$1" "$2" "${CAMPAIGN_YAML}" <<'PY'
import sys, yaml, pathlib
name, seed, yaml_path = sys.argv[1], sys.argv[2], sys.argv[3]
campaign = yaml.safe_load(pathlib.Path(yaml_path).read_text())
merged = {**campaign["vars"], **campaign["arms"][name].get("vars", {})}
z_dim = int(merged["z_dim"])
enc = merged["encoder_ckpt"]
enc = enc.replace("${vars.encoder_arm}", merged["encoder_arm"])
enc = enc.replace("${vars.seed}", str(seed))
mirrors = {
    "/data/pareto_stack/": "logs/pareto_stack_mirror/",
    "/data/interface_combos/": "logs/interface_combos_mirror/",
}
for container, local in mirrors.items():
    if enc.startswith(container):
        enc = local + enc[len(container):]
        break
else:
    raise SystemExit(f"unmapped encoder path for arm {name}: {enc}")
# The history overrides travel with the arm: they change the actor input
# width, and the policy restore is strict.
history = " ".join(merged["history_args"])
# Macro-state terms, stride and anchor live in data_overrides for this
# campaign, not in named vars.
macro = [t for t in merged["data_overrides"] if t.startswith("env.expert_macro")]
fields = {
    "encoder_local": enc,
    "z_dim": z_dim,
    "command_dim": int(merged["command_dim"]),
    "code_latent_dim": z_dim,
    "hold": int(merged["hold"]),
    "phase_mode": merged["phase_mode"],
    "history_args": history,
    "macro_args": " ".join(macro).replace("${vars.stride}", str(merged["stride"])).replace("${vars.anchor_mode}", str(merged["anchor_mode"])),
}
for key, value in fields.items():
    print(f"{key}='{value}'")
PY
}

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

if [[ "${ARMS:-}" == "" ]]; then
    mapfile -t all_arms < <(pixi run python -c "
import yaml
print('\n'.join(yaml.safe_load(open('${CAMPAIGN_YAML}'))['arms']))
")
    ARMS="${all_arms[*]}"
fi

for arm in ${ARMS}; do
  for seed in ${SEEDS}; do
    tree="${MIRROR}/${arm}_seed${seed}"
    [[ -d "${tree}/tracker" ]] || { log "[SKIP] no mirror ${tree}"; continue; }

    if ! vars_text="$(arm_vars "${arm}" "${seed}" 2>/dev/null)"; then
        log "[SKIP] ${arm} is not an arm of this campaign"; continue
    fi
    eval "${vars_text}"
    # Arrays, not bare words: `macro_args` holds a bracket expression, and an
    # unquoted expansion would expose it to pathname expansion.
    read -r -a macro_arr <<< "${macro_args}"
    read -r -a history_arr <<< "${history_args}"
    encoder="${encoder_local}"
    [[ -s "${encoder}" ]] || { log "[SKIP] no encoder ${encoder}"; continue; }

    mapfile -t frames < <(ls -1 "${tree}/tracker" 2>/dev/null | sed -n 's/^f\([0-9]\+\)$/\1/p' | sort -n)
    [[ "${#frames[@]}" -gt 0 ]] || { log "[SKIP] no checkpoints in ${tree}/tracker"; continue; }
    final="${frames[-1]}"

    for row in ${ROWS}; do
        board="$(board_for "${row}")"
        [[ -n "${board}" ]] || { log "[SKIP] unknown row ${row}"; continue; }
        profile="$(randomization_for "${row}")"

        if [[ -n "${FRAMES}" ]]; then
            read -r -a selected <<< "${FRAMES}"
        elif [[ "${row}" == "milestone" ]]; then
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
                --checkpoint "${checkpoint}" \
                --output_json "${out}" \
                --label "${arm}_seed${seed}_${row}_f${frame}" \
                --task Isaac-Imitation-G1-v2 --algo IPMD \
                --agent_entry_point rlopt_ipmd_tuned_cfg_entry_point \
                --randomization "${profile}" --action_sampling mode \
                --num_envs "${#ranks[@]}" --steps "${MAX_STEPS}" --seed 0 \
                --reference_start_frame 0 --reset_schedule sequential \
                --trajectory_ranks "${ranks[@]}" \
                --headless \
                --kit_args=--/app/extensions/fsWatcherEnabled=false \
                --skill_encoder_source auto \
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
                "agent.ipmd.latent_learning.command_phase_mode=${phase_mode}" \
                "agent.ipmd.latent_learning.code_latent_dim=${code_latent_dim}" \
                agent.ipmd.hl_skill_finetune_enabled=false \
                "${macro_arr[@]}" \
                "${history_arr[@]}" \
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
            # Isaac Sim's shutdown path can swallow a Python exception and still
            # exit 0. The output file is the only trustworthy success signal.
            if [[ ! -s "${out}" ]]; then
                log "[FAIL] ${arm} seed${seed} ${row} f${frame}: exit 0 but no row written: $(grep -iE 'error|out of memory' "${out}.log" | tail -1)"
                continue
            fi
            log "[OK] $(basename "${out}")"
        done
    done
  done
done

report
