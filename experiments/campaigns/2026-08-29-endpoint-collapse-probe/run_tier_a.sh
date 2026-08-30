#!/usr/bin/env bash
# Tier A: offline window-usage probes on the frozen diffntp_chunk encoder.
set -euo pipefail
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

CHECKPOINT="${CHECKPOINT:-logs/pareto_stack_mirror/diffntp_chunk_h1_ee_wide_seed0/encoder/checkpoints/latest.pt}"
REF_ARRAYS="${REF_ARRAYS:-/mnt/hsstorage/fwu91/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-logs/endpoint_collapse_probe/tier_a_$(date +%Y-%m-%d_%H-%M-%S)}"

exec pixi run python -m imitation_experiments.capacity.probe_skill_window_usage \
    --skill_checkpoint "${CHECKPOINT}" \
    --reference_arrays_dir "${REF_ARRAYS}" \
    --output_dir "${OUTPUT_DIR}" \
    "$@"
