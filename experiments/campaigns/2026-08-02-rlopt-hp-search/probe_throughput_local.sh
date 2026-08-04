#!/usr/bin/env bash
# Phase 1: throughput probes for the G1 v2 low-level tracker.
#
# The objective is wall-clock time to a quality level, which factors as
#   time_to_quality = frames_needed / frames_per_second.
# This script measures only the denominator. It makes no learning claim: 30
# iterations is far too few, and every probe is scored purely on fps and VRAM.
#
# It exists because the denominator looks like the bigger lever. On the cluster
# collection is ~82% of iteration time (1.66 s vs 0.37 s), and a local probe at
# 12288 x 24 averaged only ~30% GPU utilization while using 55.6 of 96 GB. A
# GPU that idles most of the time is not compute-bound, so more parallel envs
# should convert directly into fps.
#
# Note on batch geometry: at a fixed mini_batch_size, raising num_envs does NOT
# change optimizer steps per frame (that is epochs / mini_batch_size). It raises
# frames per iteration, so the collector refreshes data fewer times per frame --
# the same freshness trade the `a8_r24_matched` arm isolates.
#
# Usage, from the repository root:
#   ./experiments/campaigns/2026-08-02-rlopt-hp-search/probe_throughput_local.sh
#     -> dry run.
#   DRY_RUN=0 ./experiments/campaigns/2026-08-02-rlopt-hp-search/probe_throughput_local.sh
#     -> ~40 min.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "${REPO_ROOT}"

DRY_RUN="${DRY_RUN:-1}"
PROBE_ROOT="${PROBE_ROOT:-logs/rlopt_throughput_probe_20260802}"
ITERATIONS="${ITERATIONS:-30}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-12}"
MINIBATCH="${MINIBATCH:-18432}"

TASK_NAME="${TASK_NAME:-Isaac-Imitation-G1-v2}"
MANIFEST_PATH="${MANIFEST_PATH:-./data/lafan1/manifests/g1_lafan1_manifest.json}"
DATASET_PATH="${DATASET_PATH:-./data/lafan1/zarr/g1_hl_diffsr}"
ENCODER_CKPT="${ENCODER_CKPT:-logs/downloaded_checkpoints/lafan1_latent_deterministic_5b_seed0/skill_encoder/best.pt}"

fail() { echo "[FATAL] $*" >&2; exit 1; }
[[ -f "${MANIFEST_PATH}" ]] || fail "manifest not found: ${MANIFEST_PATH}"
[[ -d "${DATASET_PATH}" ]] || fail "dataset cache not found: ${DATASET_PATH}"
[[ -f "${ENCODER_CKPT}" ]] || fail "encoder checkpoint not found: ${ENCODER_CKPT}"

# name | num_envs | extra Hydra overrides
# torch.compile and cudagraphs are off in every run to date; they act on the
# 18% of iteration time that is not collection, plus the policy forward inside
# collection, so they are worth one probe each.
PROBE_SPECS=(
"p0_e12288|12288|"
"p1_e16384|16384|"
"p2_e24576|24576|"
"p3_e32768|32768|"
"p4_e12288_compile|12288|agent.compile.compile=true"
"p5_e12288_cudagraphs|12288|agent.compile.compile=true agent.compile.cudagraphs=true"
)

ALL_NAMES=()
for spec in "${PROBE_SPECS[@]}"; do ALL_NAMES+=("${spec%%|*}"); done
PROBES="${PROBES:-${ALL_NAMES[*]}}"

echo "[INFO] probe root : ${PROBE_ROOT}"
echo "[INFO] iterations : ${ITERATIONS} at ${ROLLOUT_STEPS} steps"
echo "[INFO] probes     : ${PROBES}"
echo "[INFO] dry run    : ${DRY_RUN}"
echo

run_probe() {
    local name="$1" num_envs="$2" extra="$3"
    local probe_dir="${PROBE_ROOT}/${name}"
    local frames_per_batch=$((num_envs * ROLLOUT_STEPS))

    local -a cmd=(
        env TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
    )
    # TORCHDYNAMO_DISABLE=1 is the repo's habit and would silently neuter the
    # compile probes, so only set it when compilation is not under test.
    if [[ "${extra}" != *compile=true* ]]; then
        cmd+=(TORCHDYNAMO_DISABLE=1)
    fi
    cmd+=(
        pixi run -e isaaclab python scripts/rlopt/train_physx.py
        --task "${TASK_NAME}" --algo IPMD
        --num_envs "${num_envs}" --max_iterations "${ITERATIONS}" --headless
        --seed 0
        --kit_args=--/app/extensions/fsWatcherEnabled=false
        "env.data.manifest=${MANIFEST_PATH}"
        "env.data.cache_dir=${DATASET_PATH}"
        env.data.cache_refresh=false
        env.command_interface.encoder.future_steps=9
        agent.ipmd.command_source=hl_skill
        "agent.ipmd.hl_skill_checkpoint_path=${ENCODER_CKPT}"
        agent.ipmd.hl_skill_horizon_steps=10
        agent.ipmd.hl_skill_finetune_enabled=false
        agent.ipmd.latent_steps_min=10
        agent.ipmd.latent_steps_max=10
        agent.ipmd.latent_learning.code_period=10
        "agent.collector.frames_per_batch=${ROLLOUT_STEPS}"
        "agent.loss.mini_batch_size=${MINIBATCH}"
        agent.logger.backend=csv
        "agent.logger.log_dir=${probe_dir}"
        --log_interval "${frames_per_batch}"
        agent.save_interval=1000000000
    )
    # shellcheck disable=SC2206
    [[ -n "${extra}" ]] && cmd+=(${extra})

    echo "=== ${name}: ${num_envs} envs x ${ROLLOUT_STEPS} = ${frames_per_batch} frames/iter ${extra}"
    if [[ "${DRY_RUN}" != "0" ]]; then
        printf '    +'; printf ' %q' "${cmd[@]}"; printf '\n\n'
        return 0
    fi

    mkdir -p "${probe_dir}"
    # Sample VRAM and utilization while the probe runs; an OOM or a still-idle
    # GPU is the whole point of the measurement.
    ( while :; do
        nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader >> "${probe_dir}/gpu.csv"
        command sleep 5
      done ) &
    local sampler=$!

    local status=0 start_s end_s
    start_s="$(date +%s)"
    "${cmd[@]}" > "${probe_dir}/train.log" 2>&1 || status=$?
    end_s="$(date +%s)"
    kill "${sampler}" 2>/dev/null || true

    echo "{\"probe\": \"${name}\", \"num_envs\": ${num_envs}, \"rollout_steps\": ${ROLLOUT_STEPS}," \
         "\"frames_per_batch\": ${frames_per_batch}, \"iterations\": ${ITERATIONS}," \
         "\"extra\": \"${extra}\", \"wall_time_s\": $((end_s - start_s)), \"exit_status\": ${status}}" \
         > "${probe_dir}/probe.json"

    if [[ "${status}" != "0" ]]; then
        echo "    [WARN] exited ${status} (OOM?); see ${probe_dir}/train.log"
    else
        echo "    [OK] $(( (end_s - start_s) / 60 )) min"
    fi
    echo
}

for spec in "${PROBE_SPECS[@]}"; do
    name="${spec%%|*}"; rest="${spec#*|}"
    num_envs="${rest%%|*}"; extra="${rest#*|}"
    for requested in ${PROBES}; do
        if [[ "${requested}" == "${name}" ]]; then
            run_probe "${name}" "${num_envs}" "${extra}"
            break
        fi
    done
done

[[ "${DRY_RUN}" != "0" ]] && echo "[INFO] dry run only. Re-run with DRY_RUN=0." || true
