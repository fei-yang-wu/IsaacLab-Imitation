#!/usr/bin/env bash
# Wiring qualification for the interface design study. Proves an arm RUNS; it
# proves nothing about the arm's quality.
#
# Two stages per arm, the pattern the 2026-08-18 ablation used:
#   1. the encoder pretrain completes real offline updates with a finite loss,
#      writes a loadable checkpoint, and -- for a quantized arm -- uses more
#      than one code level;
#   2. the frozen encoder drives one 128-frame IPMD iteration at the arm's own
#      command width.
#
# Every flag is composed from `campaign.yaml`, so a smoke exercises the same
# command line the cluster will freeze. Only the budget knobs are overridden.
#
#   ./smoke.sh                       # every active arm (tiers 1, 2, 3)
#   ARMS="ctrl bn_vq_ema" ./smoke.sh
#   TIERS=1 ./smoke.sh
#   ./smoke.sh --report
set -uo pipefail
CAMPAIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${CAMPAIGN_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

SMOKE_ROOT="${SMOKE_ROOT:-${REPO_ROOT}/logs/interface_design_study_smoke}"
REFERENCE_ARRAYS="${REFERENCE_ARRAYS:-/mnt/hsstorage/fwu91/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
PERSIST_ID="${PERSIST_ID:-bones_seed_sonic_full_129785@e714bbff}"
PRETRAIN_UPDATES="${PRETRAIN_UPDATES:-4}"
# The PRODUCTION batch. A smaller one is not just slower to converge, it changes
# the wiring answer: at batch 256 a 512-entry VQ codebook cannot hit enough
# codes and reports a fully collapsed code, while at 8192 it never does. Batch
# must not be a confound between the smoke and the run it qualifies.
PRETRAIN_BATCH="${PRETRAIN_BATCH:-8192}"
PRETRAIN_ENVS="${PRETRAIN_ENVS:-16}"
LOWLEVEL_ENVS="${LOWLEVEL_ENVS:-64}"
LOWLEVEL_STEPS="${LOWLEVEL_STEPS:-2}"   # 64 x 2 = 128 frames
TIERS="${TIERS:-1 2 3}"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

# Tokens for one arm's pretrain command, newline separated so nothing needs
# quoting on the way back into bash.
pretrain_tokens() {
    pixi run python - "$1" "$2" "$3" "$4" <<'PY'
import sys, yaml, pathlib

name, out_dir, updates, batch = sys.argv[1:5]
campaign = yaml.safe_load(
    pathlib.Path(
        "experiments/campaigns/2026-08-19-interface-design-study/campaign.yaml"
    ).read_text()
)
merged = {**campaign["vars"], **campaign["arms"][name]["vars"]}

tokens: list[str] = []
tokens += ["--output_dir", out_dir]
tokens += ["--horizon_steps", "10"]
tokens += ["--encoder_window_mode", str(merged["window_mode"])]
tokens += ["--transition_objective", str(merged["objective"])]
tokens += [str(t) for t in merged["objective_args"]]
tokens += ["--latent_mode", str(merged["latent_mode"])]
tokens += [str(t) for t in merged["mode_args"]]
tokens += ["--z_dim", str(merged["z_dim"])]
tokens += [str(t) for t in merged["ln_args"]]
tokens += ["--encoder_hidden_dims", "2048", "1024", "512", "512"]
tokens += ["--encoder_activation", "silu"]
tokens += ["--diffsr_feature_dim", "256", "--diffsr_embed_dim", "1024"]
tokens += ["--diffsr_g_hidden_dims", "1024", "1024", "512"]
tokens += ["--diffsr_mu_hidden_dims", "1024", "1024", "512"]
tokens += ["--batch_size", batch, "--num_updates", updates]
tokens += ["--log_interval", "1", "--eval_batches", "1"]
print("\n".join(tokens))
PY
}

arm_field() {
    pixi run python - "$1" "$2" <<'PY'
import sys, yaml, pathlib
name, field = sys.argv[1:3]
campaign = yaml.safe_load(
    pathlib.Path(
        "experiments/campaigns/2026-08-19-interface-design-study/campaign.yaml"
    ).read_text()
)
merged = {**campaign["vars"], **campaign["arms"][name]["vars"]}
value = merged[field]
if isinstance(value, str) and value.startswith("${"):
    value = merged["z_dim"]
print(value)
PY
}

verdict_file() { printf '%s/%s/verdict.json' "${SMOKE_ROOT}" "$1"; }

report() {
    shopt -s nullglob
    local any=0
    for v in "${SMOKE_ROOT}"/*/verdict.json; do
        any=1
        pixi run python -c "
import json, sys
d = json.load(open('${v}'))
flags = ' '.join(f'{k}={v}' for k, v in d.get('checks', {}).items())
print(f\"{d['arm']:<30}{d['status']:<8}{flags}\")
"
    done
    (( any )) || log "[INFO] nothing smoked yet"
}

if [[ "${1:-}" == "--report" ]]; then report; exit $?; fi

[[ -s "${REFERENCE_ARRAYS}/reference_arrays_manifest.json" ]] || {
    log "[FATAL] reference arrays missing: ${REFERENCE_ARRAYS}"; exit 2; }

if [[ "${ARMS:-}" == "" ]]; then
    mapfile -t selected < <(pixi run python -c "
import yaml
campaign = yaml.safe_load(open('experiments/campaigns/2026-08-19-interface-design-study/campaign.yaml'))
tiers = {int(t) for t in '${TIERS}'.split()}
for name, arm in campaign['arms'].items():
    if int(arm['vars'].get('tier', 1)) in tiers:
        print(name)
")
    ARMS="${selected[*]}"
fi

DATA_OVERRIDES=(
    physics=newton_mjwarp
    env.data.manifest=null
    "env.data.reference_arrays_dir=${REFERENCE_ARRAYS}"
    "env.data.persist_id=${PERSIST_ID}"
    env.data.reference_arrays_resident=false
    env.data.reference_arrays_warm_workers=4
    env.data.runtime_cache_device=cuda:0
    env.data.macro_cache_device=cuda:0
)

failures=0
for arm in ${ARMS}; do
    work="${SMOKE_ROOT}/${arm}"
    verdict="$(verdict_file "${arm}")"
    if [[ -s "${verdict}" ]] && grep -q '"status": "pass"' "${verdict}"; then
        log "[SKIP] ${arm} already passed"
        continue
    fi
    mkdir -p "${work}"

    macro_terms="$(arm_field "${arm}" macro_terms)"
    stride="$(arm_field "${arm}" stride)"
    anchor_mode="$(arm_field "${arm}" anchor_mode)"
    command_dim="$(arm_field "${arm}" command_dim)"
    code_latent_dim="$(arm_field "${arm}" code_latent_dim)"
    hold="$(arm_field "${arm}" hold)"
    phase_mode="$(arm_field "${arm}" phase_mode)"
    command_mode="$(arm_field "${arm}" command_mode)"
    objective="$(arm_field "${arm}" objective)"
    latent_mode="$(arm_field "${arm}" latent_mode)"
    z_dim="$(arm_field "${arm}" z_dim)"

    mapfile -t arm_tokens < <(pretrain_tokens "${arm}" "${work}/encoder" "${PRETRAIN_UPDATES}" "${PRETRAIN_BATCH}")
    [[ "${#arm_tokens[@]}" -gt 0 ]] || { log "[FAIL] ${arm}: could not compose pretrain"; failures=$((failures+1)); continue; }

    log "${arm}: pretrain (${PRETRAIN_UPDATES} updates, batch ${PRETRAIN_BATCH})"
    env TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 \
        HYDRA_FULL_ERROR=1 TORCHDYNAMO_DISABLE=1 \
        pixi run -e isaaclab python -u scripts/rlopt/train_hl_skill_diffsr.py \
        --task Isaac-Imitation-G1-v2 --num_envs "${PRETRAIN_ENVS}" --seed 0 \
        --device cuda:0 --headless --assert-kitless \
        --logger_backend none \
        "${arm_tokens[@]}" \
        "${DATA_OVERRIDES[@]}" \
        "env.expert_macro_state_terms=${macro_terms}" \
        "env.expert_macro_frame_stride=${stride}" \
        "env.expert_macro_anchor_mode=${anchor_mode}" \
        > "${work}/pretrain.log" 2>&1
    pre_rc=$?

    checkpoint="${work}/encoder/checkpoints/latest.pt"
    if (( pre_rc != 0 )) || [[ ! -s "${checkpoint}" ]]; then
        log "[FAIL] ${arm} pretrain exit ${pre_rc}: $(tail -3 "${work}/pretrain.log" | tr '\n' ' ' | cut -c1-200)"
        printf '{"arm": "%s", "status": "fail", "stage": "pretrain", "exit": %d}\n' "${arm}" "${pre_rc}" > "${verdict}"
        failures=$((failures+1)); continue
    fi

    log "${arm}: lowlevel (${LOWLEVEL_ENVS} x ${LOWLEVEL_STEPS} = $((LOWLEVEL_ENVS*LOWLEVEL_STEPS)) frames, command ${command_dim})"
    env TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 \
        HYDRA_FULL_ERROR=1 TORCHDYNAMO_DISABLE=1 \
        pixi run -e isaaclab python -u scripts/rlopt/train.py \
        --task Isaac-Imitation-G1-v2 --algo IPMD \
        --agent rlopt_ipmd_tuned_cfg_entry_point \
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
        agent.ipmd.hl_skill_horizon_steps=10 \
        "agent.ipmd.hl_skill_command_mode=${command_mode}" \
        "agent.ipmd.latent_steps_min=${hold}" \
        "agent.ipmd.latent_steps_max=${hold}" \
        "agent.ipmd.latent_learning.code_period=${hold}" \
        "agent.ipmd.latent_learning.command_phase_mode=${phase_mode}" \
        "agent.ipmd.latent_learning.code_latent_dim=${code_latent_dim}" \
        agent.ipmd.hl_skill_finetune_enabled=false \
        env.rewards.action_rate_l2.weight=0.0 \
        env.rewards.tracking_reward_points.weight=4.0 \
        env.enable_termination_curriculum=true \
        env.termination_curriculum_start_frames=5000000 \
        env.termination_curriculum_end_frames=30000000 \
        env.command_interface.reference.selection=random80_adaptive20 \
        env.data.reference_prefetch_mode=next \
        "agent.collector.frames_per_batch=${LOWLEVEL_STEPS}" \
        agent.loss.mini_batch_size=64 \
        agent.ipmd.expert_batch_size=256 \
        agent.loss.gamma=0.97 \
        agent.save_interval=250000000 \
        env.sim.physics.solver_cfg.njmax=320 \
        env.sim.physics.solver_cfg.nconmax=200 \
        "agent.policy.num_cells=[2048,2048,1024,1024,512,512]" \
        "agent.value_function.num_cells=[2048,2048,1024,1024,512,512]" \
        agent.policy.activation_fn=silu \
        agent.value_function.activation_fn=silu \
        "${DATA_OVERRIDES[@]}" \
        "env.expert_macro_state_terms=${macro_terms}" \
        "env.expert_macro_frame_stride=${stride}" \
        "env.expert_macro_anchor_mode=${anchor_mode}" \
        > "${work}/lowlevel.log" 2>&1
    low_rc=$?

    pixi run python -m imitation_experiments.lowlevel.smoke_verdict \
        --arm "${arm}" --checkpoint "${checkpoint}" \
        --pretrain-log "${work}/pretrain.log" \
        --lowlevel-log "${work}/lowlevel.log" \
        --lowlevel-exit "${low_rc}" \
        --expected-command-dim "${command_dim}" \
        --expected-objective "${objective}" \
        --expected-latent-mode "${latent_mode}" \
        --expected-z-dim "${z_dim}" \
        --output "${verdict}" >> "${work}/verdict.log" 2>&1
    ver_rc=$?
    if (( ver_rc != 0 )); then
        log "[FAIL] ${arm}: $(pixi run python -c "
import json
try:
    d = json.load(open('${verdict}'))
    print(d.get('reason', 'verdict failed'))
except Exception as exc:
    print('no verdict:', exc)
")"
        failures=$((failures+1)); continue
    fi
    if [[ "${SMOKE_KEEP_CHECKPOINTS:-0}" != "1" ]]; then
        rm -rf "${work}/encoder/checkpoints" "${work}/tracker"
    fi
    log "[PASS] ${arm}"
done

echo
report
if (( failures > 0 )); then
    log "[FATAL] ${failures} arm(s) failed the wiring smoke"
    exit 1
fi
log "[INFO] every smoked arm passed"
