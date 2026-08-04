#!/usr/bin/env bash
set -uo pipefail

# Watch the two mjwarp-aligned 5B runs on ICE and, as each crosses a frame
# threshold, pull that checkpoint down and score it locally.
#
# Why local: AGENTS.md prefers the workstation for inference and metric
# inspection, because a fresh Isaac Lab container is expensive to initialize per
# cluster job and these are single-checkpoint evaluations, not training.
#
# WHAT THE EVAL IS, stated plainly so the numbers are not over-read. This runs
# `scripts/audit/sim2sim_backend_eval.py`, which drives the tracker with the
# latent the frozen DiffSR encoder produces from the reference -- i.e. the
# oracle command -- and reports the environment's own MPJPE plus survival. That
# is a mid-training progress check. It is NOT the paper-facing low-level oracle
# qualification: that is `imitation_experiments.lowlevel.evaluate_checkpoint`
# under the strict frame-0 protocol with its own audit, and it should be run on
# the final checkpoint, not on these.
#
# Two passes per checkpoint, because one alone misleads:
#   strict      terminations active. The protocol number, but MPJPE is scored
#               only over frames a surviving episode reached, so a policy that
#               falls early can post a flattering value.
#   full-horizon terminations disabled, fixed length. Every checkpoint is scored
#               over identical frames. AGENTS.md requires this pass alongside
#               any oracle evaluation for exactly that reason.
#
# Idempotent: a threshold already evaluated is skipped, so re-running is free
# and the watch loop can be restarted at any time.
#
#   ./monitor_5b_and_eval.sh                 # one poll+act cycle, then exit
#   ./monitor_5b_and_eval.sh --watch         # poll until every threshold is done
#   ./monitor_5b_and_eval.sh --report        # status only, never evaluates

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
while [ ! -x "${REPO_ROOT}/docker/cluster/cluster_interface.sh" ]; do
    [ "${REPO_ROOT}" = "/" ] && { echo "[ERROR] repo root not found" >&2; exit 2; }
    REPO_ROOT="$(dirname "${REPO_ROOT}")"
done
cd "${REPO_ROOT}"

MODE="once"
case "${1:-}" in
    --watch) MODE="watch" ;;
    --report) MODE="report" ;;
    --once|"") MODE="once" ;;
    *) echo "[ERROR] unknown argument '$1'" >&2; exit 2 ;;
esac

INTERVAL="${INTERVAL:-900}"
THRESHOLDS="${THRESHOLDS:-1000000000 2000000000}"
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/data}"
OUT_ROOT="${OUT_ROOT:-logs/monitor_5b}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-128}"
EVAL_STEPS="${EVAL_STEPS:-500}"
EVAL_SEED="${EVAL_SEED:-0}"

# run_tag | local manifest | local zarr cache | encoder tag
RUNS=(
"lafan1_v2_mjwarp_aligned_5b_seed0_e12288_r24|./data/lafan1/manifests/g1_lafan1_manifest.json|./data/lafan1/zarr/g1_hl_diffsr|lafan1_v2_det_sr_h10_z256_seed0"
"bones91_v2_mjwarp_aligned_5b_seed0_e12288_r24|./data/bones_seed_100/manifests/g1_bones_seed_100_sonic_filtered_manifest.json|./data/bones_seed_100/g1_hl_diffsr|bones_seed_91_v2_det_sr_h10_z256_seed0"
)

ssh_ice() { ssh -o BatchMode=yes -o ConnectTimeout=15 ice "$@"; }
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

# Latest checkpoint frame count for a run, or empty.
latest_frames() {
    ssh_ice "find '${REMOTE_DATA_ROOT}/tuned_5b/$1' -name 'model_step_*.pt' 2>/dev/null \
        | sed 's/.*model_step_//; s/\.pt//' | sort -n | tail -1"
}

# Remote path of the earliest checkpoint at or beyond $2 frames.
checkpoint_at_or_after() {
    ssh_ice "find '${REMOTE_DATA_ROOT}/tuned_5b/$1' -name 'model_step_*.pt' 2>/dev/null \
        | awk -F'model_step_' '{n=\$2; sub(/\.pt/,\"\",n); if (n+0 >= $2) print n+0, \$0}' \
        | sort -n | head -1 | cut -d' ' -f2-"
}

evaluate() {
    local run_tag="$1" manifest="$2" cache="$3" encoder_tag="$4" threshold="$5" remote_ckpt="$6"
    local tag out ckpt encoder frames
    frames="$(basename "${remote_ckpt}" | sed 's/.*model_step_//; s/\.pt//')"
    tag="$(printf '%s_%dM' "${run_tag%%_v2_*}" $((threshold / 1000000)))"
    out="${OUT_ROOT}/${run_tag}/${threshold}"
    mkdir -p "${out}"

    ckpt="${out}/$(basename "${remote_ckpt}")"
    encoder="${OUT_ROOT}/encoders/${encoder_tag}.pt"
    mkdir -p "$(dirname "${encoder}")"

    log "  pulling $(basename "${remote_ckpt}") (${frames} frames)"
    rsync -q --partial --inplace "ice:${remote_ckpt}" "${ckpt}" \
        || { log "  [FAIL] checkpoint rsync"; return 1; }
    if [ ! -s "${encoder}" ]; then
        log "  pulling encoder ${encoder_tag}"
        rsync -q --partial --inplace \
            "ice:${REMOTE_DATA_ROOT}/pretrain_store/${encoder_tag}/checkpoints/latest.pt" \
            "${encoder}" || { log "  [FAIL] encoder rsync"; return 1; }
    fi

    # The recipe half that does not live on the agent config, plus the encoder
    # binding. Mirrors the training invocation so the eval scores the policy the
    # run is actually producing.
    local overrides=(
        "env.data.manifest=${manifest}"
        "env.data.cache_dir=${cache}"
        env.data.cache_refresh=false
        env.command_interface.encoder.future_steps=9
        env.command_interface.actor.dim=258
        agent.ipmd.latent_dim=258
        agent.ipmd.command_source=hl_skill
        "agent.ipmd.hl_skill_checkpoint_path=${encoder}"
        agent.ipmd.hl_skill_horizon_steps=10
        agent.ipmd.latent_steps_min=10
        agent.ipmd.latent_steps_max=10
        agent.ipmd.latent_learning.code_period=10
        agent.ipmd.hl_skill_finetune_enabled=false
        agent.policy.normalize_input=true
        agent.value_function.normalize_input=true
        agent.policy.activation_fn=silu
        agent.value_function.activation_fn=silu
        "agent.policy.num_cells=[1024,1024,512]"
        "agent.value_function.num_cells=[1024,1024,512]"
        agent.logger.backend=
    )

    local pass rc=0
    for pass in strict full_horizon; do
        local extra=()
        [ "${pass}" = "strict" ] && extra=(--keep_terminations)
        log "  eval ${pass}"
        env TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 \
            TORCHDYNAMO_DISABLE=1 \
            timeout 3600 pixi run -e isaaclab python scripts/audit/sim2sim_backend_eval.py \
            --task Isaac-Imitation-G1-v2 --algo IPMD \
            --checkpoint "${ckpt}" \
            --num_envs "${EVAL_NUM_ENVS}" --steps "${EVAL_STEPS}" --seed "${EVAL_SEED}" \
            --output "${out}/${pass}.json" \
            "${extra[@]}" \
            physics=newton_mjwarp \
            env.sim.physics.solver_cfg.njmax=288 env.sim.physics.solver_cfg.nconmax=200 \
            "${overrides[@]}" > "${out}/${pass}.log" 2>&1 || rc=1
        if [ -s "${out}/${pass}.json" ]; then
            python3 - "${out}/${pass}.json" "${pass}" "${frames}" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); m = d["metrics"]
print(f"    {sys.argv[2]:13s} frames={int(sys.argv[3]):>12,}  "
      f"MPJPE mean {m['mpjpe_mm_mean']:7.2f} mm  final {m['mpjpe_mm_final']:7.2f}  "
      f"survived {m['survived_frac']:.3f}")
PY
        else
            log "    [FAIL] ${pass} produced no summary; see ${out}/${pass}.log"
            rc=1
        fi
    done
    # Only mark done when both passes produced numbers, so a transient failure
    # is retried on the next cycle instead of being silently skipped forever.
    [ "${rc}" -eq 0 ] && touch "${out}/.done"
    return "${rc}"
}

cycle() {
    local pending=0 entry run_tag manifest cache encoder_tag frames threshold remote_ckpt
    local queue; queue="$(ssh_ice "squeue -u \$USER -h -o '%i %T %M'" 2>/dev/null)"
    [ -n "${queue}" ] && log "queue:" && sed 's/^/  /' <<<"${queue}"

    for entry in "${RUNS[@]}"; do
        IFS='|' read -r run_tag manifest cache encoder_tag <<<"${entry}"
        frames="$(latest_frames "${run_tag}")"
        log "${run_tag}: latest ${frames:-none} frames"
        [ -z "${frames}" ] && { pending=1; continue; }

        for threshold in ${THRESHOLDS}; do
            local out="${OUT_ROOT}/${run_tag}/${threshold}"
            [ -e "${out}/.done" ] && continue
            if [ "${frames}" -lt "${threshold}" ]; then
                pending=1
                continue
            fi
            [ "${MODE}" = "report" ] && { log "  ${threshold} ready (report mode, not evaluating)"; pending=1; continue; }
            remote_ckpt="$(checkpoint_at_or_after "${run_tag}" "${threshold}")"
            [ -z "${remote_ckpt}" ] && { pending=1; continue; }
            log "  threshold ${threshold} reached"
            evaluate "${run_tag}" "${manifest}" "${cache}" "${encoder_tag}" "${threshold}" "${remote_ckpt}" \
                || pending=1
        done
    done
    return "${pending}"
}

if [ "${MODE}" = "watch" ]; then
    while true; do
        cycle && { log "every threshold evaluated; exiting."; break; }
        log "sleeping ${INTERVAL}s"
        sleep "${INTERVAL}"
    done
else
    cycle || true
fi
