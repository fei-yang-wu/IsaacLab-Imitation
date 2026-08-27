#!/usr/bin/env bash
# Phase D1: asynchronous service-backed evaluation of a trained head.
#
#   ./eval_async.sh <fsq64_10b|ln_hold1_10b> [extra args...]
#
# Starts the batched zmq chunk service in the `gr00t` Pixi environment, waits
# for its ready record, runs the Isaac evaluator with `--gr00t_service`, and
# tears the service down. The service owns the head under the upstream torch
# 2.9 pin; the evaluator keeps Isaac's torch 2.11 — only raw float32 bytes
# cross the socket.
#
# The async protocol (lead-time request, swap at expiry, deadline miss =
# hold + count) lives in the sampler; LEAD_STEPS sets the request lead in
# control steps. Async rows are labelled `planner_execution: async_service`
# and are never pooled with sync rows. Per the relaxed D1 gate (2026-08-18),
# run the sync companion separately with `./eval.sh` on the same seed and
# report both; no numeric equivalence bound applies.
#
# Ensembling is not supported in async mode; ENSEMBLE is forced to none.
set -euo pipefail
CAMPAIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${CAMPAIGN_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"
source "${CAMPAIGN_DIR}/arms.sh"

ARM="${1:?usage: eval_async.sh <fsq64_10b|ln_hold1_10b>}"
shift || true
arm_config "${ARM}"

UPDATE="${UPDATE:-0012000}"
HEAD="${HEAD:-${REPO_ROOT}/outputs/planner_10b/arms/${ARM}/checkpoints/update_${UPDATE}.pt}"
LEAD_STEPS="${LEAD_STEPS:-5}"
SERVICE_SEED="${SERVICE_SEED:-0}"
SERVICE_DTYPE="${SERVICE_DTYPE:-float32}"
ENDPOINT="${ENDPOINT:-ipc:///tmp/gr00t_batch_${ARM}_$$.ipc}"
SERVICE_LOG="${SERVICE_LOG:-${REPO_ROOT}/logs/planner_10b/isaac_eval/service_${ARM}_$$.log}"
mkdir -p "$(dirname "${SERVICE_LOG}")"
[ -f "${HEAD}" ] || { echo "missing head checkpoint: ${HEAD}" >&2; exit 1; }

pixi run -e gr00t python -m imitation_experiments.planner.gr00t_batch_service \
    --checkpoint "${HEAD}" \
    --goal-features "${GOAL_FEATURES}" \
    --endpoint "${ENDPOINT}" \
    --seed "${SERVICE_SEED}" --dtype "${SERVICE_DTYPE}" \
    > "${SERVICE_LOG}" 2>&1 &
SERVICE_PID=$!
trap 'kill "${SERVICE_PID}" 2>/dev/null || true' EXIT

# The ready record is the first stdout line; a dead service must fail here,
# not as an opaque zmq timeout mid-evaluation.
for _ in $(seq 1 240); do
    if ! kill -0 "${SERVICE_PID}" 2>/dev/null; then
        echo "service exited before ready; log tail:" >&2
        tail -5 "${SERVICE_LOG}" >&2
        exit 1
    fi
    grep -q '"ready":true' "${SERVICE_LOG}" && break
    sleep 1
done
grep -q '"ready":true' "${SERVICE_LOG}" || { echo "service never became ready" >&2; exit 1; }
echo "[D1] service ready on ${ENDPOINT} (pid ${SERVICE_PID})"

LABEL_SUFFIX="${LABEL_SUFFIX:-async_lead${LEAD_STEPS}}" \
ENSEMBLE=none \
    "${CAMPAIGN_DIR}/eval.sh" "${ARM}" \
    --gr00t_service "${ENDPOINT}" \
    --gr00t_lead_steps "${LEAD_STEPS}" \
    "$@"
