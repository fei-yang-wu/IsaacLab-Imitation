#!/usr/bin/env bash
# Score pareto-stack checkpoints.
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

MIRROR="${MIRROR:-${REPO_ROOT}/logs/pareto_stack_mirror}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/logs/pareto_stack_eval}"
REFERENCE_ARRAYS="${REFERENCE_ARRAYS:-/mnt/hsstorage/fwu91/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
PERSIST_ID="${PERSIST_ID:-bones_seed_sonic_full_129785@e714bbff}"
MAX_STEPS="${MAX_STEPS:-10000}"
# SEEDS defaults to each arm's `submit_seed` from campaign.yaml.
SEEDS="${SEEDS:-}"
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
    pixi run python - "$1" "$2" <<'PY'
import sys, yaml, pathlib
name, seed = sys.argv[1], sys.argv[2]
campaign = yaml.safe_load(
    pathlib.Path(
        "experiments/campaigns/2026-08-22-pareto-stack/campaign.yaml"
    ).read_text()
)
base = dict(campaign["vars"])
arm = campaign["arms"][name]["vars"]
merged = {**base, **arm}
z_dim = int(merged["z_dim"])
code_latent = merged.get("code_latent_dim", z_dim)
if isinstance(code_latent, str) and code_latent.startswith("${"):
    code_latent = z_dim
# Resolve the encoder file this arm binds, mapped to the LOCAL mirrors.
# `${vars.X}` indirection is a named parent-encoder var; the default is the
# arm's own output tree.
enc = merged["encoder_ckpt"]
while isinstance(enc, str) and enc.startswith("${vars.") and enc.endswith("}"):
    enc = merged[enc[len("${vars.") : -1]]
enc = enc.replace("${vars.output_root}", merged["output_root"])
enc = enc.replace("${vars.arm}", name).replace("${vars.seed}", str(seed))
mirrors = {
    "/data/pareto_stack/": "logs/pareto_stack_mirror/",
    "/data/interface_design_study/": "logs/interface_design_study_mirror/",
    "/data/interface_combos/": "logs/interface_combos_mirror/",
}
for container, local in mirrors.items():
    if enc.startswith(container):
        enc = local + enc[len(container):]
        break
else:
    raise SystemExit(f"unmapped encoder path for arm {name}: {enc}")
# Per-arm env overrides (asymmetric critic etc.) must also apply at eval:
# the checkpoint's value function was built against them and the restore is
# strict. Only env.* tokens pass through.
extra = merged.get("extra_args") or []
extra_env = " ".join(t for t in extra if isinstance(t, str) and t.startswith("env."))
fields = {
    "extra_env": extra_env,
    "encoder_local": enc,
    "encoder_source": merged.get("encoder_source", "pretrained"),
    "z_dim": z_dim,
    "command_dim": int(merged["command_dim"]),
    "code_latent_dim": int(code_latent),
    "hold": int(merged["hold"]),
    "phase_mode": merged["phase_mode"],
    "phase_source": merged["phase_source"],
    "phase_period": int(merged["phase_period"]),
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
campaign = yaml.safe_load(open('experiments/campaigns/2026-08-22-pareto-stack/campaign.yaml'))
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

arm_seed() {
    pixi run python -c "
import yaml
c = yaml.safe_load(open('experiments/campaigns/2026-08-22-pareto-stack/campaign.yaml'))
merged = {**c['vars'], **c['arms']['$1'].get('vars', {})}
print(merged.get('submit_seed', 0))
"
}

for arm in ${ARMS}; do
  for seed in ${SEEDS:-$(arm_seed "${arm}")}; do
    tree="${MIRROR}/${arm}_seed${seed}"
    [[ -d "${tree}/tracker" ]] || { log "[SKIP] no mirror ${tree}"; continue; }

    if ! vars_text="$(arm_vars "${arm}" "${seed}" 2>/dev/null)"; then
        log "[SKIP] ${arm} is not an arm of this campaign"; continue
    fi
    eval "${vars_text}"
    encoder="${encoder_local}"
    [[ -s "${encoder}" ]] || { log "[SKIP] no encoder ${encoder}"; continue; }

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

        # Every override except the per-cell identity, so the single-cell and
        # tree-scoring paths cannot drift from each other.
        common_args=(
            --task Isaac-Imitation-G1-v2 --algo IPMD
            --agent_entry_point rlopt_ipmd_tuned_cfg_entry_point
            --randomization "${profile}" --action_sampling mode
            --num_envs "${#ranks[@]}" --steps "${MAX_STEPS}" --seed 0
            --reference_start_frame 0 --reset_schedule sequential
            --trajectory_ranks "${ranks[@]}"
            --headless
            --kit_args=--/app/extensions/fsWatcherEnabled=false
            --skill_encoder_source "${encoder_source}"
            physics=newton_mjwarp
            env.sim.physics.solver_cfg.njmax=320
            env.sim.physics.solver_cfg.nconmax=200
            env.events.push_robot=null
            env.data.manifest=null
            "env.data.reference_arrays_dir=${REFERENCE_ARRAYS}"
            "env.data.persist_id=${PERSIST_ID}"
            env.data.reference_arrays_resident=false
            env.data.reference_arrays_warm_workers=8
            env.data.runtime_cache_device=cuda:0
            env.data.reference_prefetch_mode=off
            env.data.macro_cache_device=cuda:0
            "env.data.runtime_cache_body_names=${RUNTIME_BODY_NAMES}"
            env.command_interface.actor=latent
            "env.command_interface.actor.dim=${command_dim}"
            env.command_interface.encoder=single
            "agent.ipmd.latent_dim=${command_dim}"
            agent.ipmd.command_source=hl_skill
            "agent.ipmd.hl_skill_checkpoint_path=${encoder}"
            agent.ipmd.hl_skill_horizon_steps=10
            "agent.ipmd.hl_skill_command_mode=${command_mode}"
            "agent.ipmd.latent_steps_min=${hold}"
            "agent.ipmd.latent_steps_max=${hold}"
            "agent.ipmd.latent_learning.code_period=${hold}"
            "agent.ipmd.latent_learning.command_phase_mode=${phase_mode}"
            "agent.ipmd.latent_learning.command_phase_source=${phase_source}"
            "agent.ipmd.latent_learning.command_phase_period=${phase_period}"
            "agent.ipmd.latent_learning.code_latent_dim=${code_latent_dim}"
            agent.ipmd.hl_skill_finetune_enabled=false
            "env.expert_macro_state_terms=${macro_terms}"
            "env.expert_macro_frame_stride=${stride}"
            "env.expert_macro_anchor_mode=${anchor_mode}"
            ${extra_env:+${extra_env}}
            env.terminations.anchor_pos.params.threshold=0.25
            env.terminations.anchor_pos.params.down_threshold=0.25
            env.terminations.anchor_ori.params.threshold=1.0
            env.terminations.ee_body_pos.params.threshold=0.25
            env.terminations.ee_body_pos.params.down_threshold=0.25
            env.terminations.foot_pos_xyz=null
            env.terminations.base_too_low=null
            "agent.policy.num_cells=${SCALED_CELLS}"
            agent.policy.activation_fn=silu
            "agent.value_function.num_cells=${SCALED_CELLS}"
            agent.value_function.activation_fn=silu
        )

        # TREE_SCORER=1 scores the whole budget axis in ONE simulation start
        # instead of one start per checkpoint. The rows are identical; only the
        # startup cost differs.
        if [[ "${TREE_SCORER:-0}" == "1" && "${row}" == "milestone" ]]; then
            log "${arm} seed${seed} ${row} tree (${#frames[@]} cells, z ${z_dim}, cmd ${command_dim}, hold ${hold}, ${board}, ${#ranks[@]} clips)"
            tree_log="${OUTPUT_ROOT}/${arm}_seed${seed}_${row}_tree.log"
            env TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 \
                HYDRA_FULL_ERROR=1 TORCHDYNAMO_DISABLE=1 \
                pixi run -e isaaclab python -u \
                scripts/rlopt/eval_checkpoint_tree.py \
                --tree "${tree}" --output_root "${OUTPUT_ROOT}" \
                --arm "${arm}" --seed "${seed}" --row "${row}" \
                -- "${common_args[@]}" > "${tree_log}" 2>&1
            rc=$?
            if (( rc != 0 )); then
                log "[FAIL] ${arm} ${row} tree exit ${rc}: $(tail -3 "${tree_log}" | tr '\n' ' ' | cut -c1-150)"
                continue
            fi
            # Same rule as the single-cell path: a Warp CUDA OOM can exit 0
            # without writing a row, so the files are the success signal.
            written=0
            for frame in "${frames[@]}"; do
                [[ -s "$(out_for "${arm}" "${seed}" "${row}" "${frame}")" ]] && written=$((written+1))
            done
            if (( written != ${#frames[@]} )); then
                log "[FAIL] ${arm} ${row} tree: ${written}/${#frames[@]} rows written: $(grep -iE 'error|out of memory' "${tree_log}" | tail -1)"
            else
                log "[OK] ${arm} ${row} tree ${written}/${#frames[@]}"
            fi
            continue
        fi

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
                "${common_args[@]}" > "${out}.log" 2>&1
            rc=$?
            if (( rc != 0 )); then
                log "[FAIL] ${arm} seed${seed} ${row} f${frame} exit ${rc}: $(tail -3 "${out}.log" | tr '\n' ' ')"
                continue
            fi
            if grep -Eq 'overflow.*increase njmax|nefc overflow' "${out}.log"; then
                log "[FAIL] ${arm} seed${seed} ${row} f${frame}: solver constraint buffer overflow"
                continue
            fi
            # Isaac Sim's shutdown path can swallow a Python exception and
            # still exit 0 (seen 2026-08-26: Warp CUDA OOM reported [OK] four
            # times with no row written). The output file is the only
            # trustworthy success signal.
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
