#!/usr/bin/env bash
set -euo pipefail

# Run an arbitrary repository script with Isaac Sim's Kit interpreter while
# sourcing PyTorch and CUDA libraries from the immutable CU130 runtime.  The
# target remains the real Python __main__, which is required by Hydra.
runtime_root="${ISAACLAB_CU130_RUNTIME_ROOT:-/opt/isaaclab-imitation-runtime-spec/.pixi/envs/container-runtime}"
runtime_site=""
for candidate in "${runtime_root}"/lib/python*/site-packages; do
    if [[ -d "${candidate}/torch" ]]; then
        runtime_site="${candidate}"
        break
    fi
done
if [[ -z "${runtime_site}" ]]; then
    echo "[ERROR] CU130 runtime site-packages not found under ${runtime_root}." >&2
    exit 1
fi

runtime_nccl="${runtime_site}/nvidia/nccl/lib/libnccl.so.2"
if [[ ! -f "${runtime_nccl}" ]]; then
    echo "[ERROR] CU130 runtime NCCL not found: ${runtime_nccl}" >&2
    exit 1
fi
runtime_nvidia_libs="$(find "${runtime_site}/nvidia" -mindepth 2 -maxdepth 3 -type d -name lib -print 2>/dev/null | paste -sd: -)"
if [[ -n "${runtime_nvidia_libs}" ]]; then
    export LD_LIBRARY_PATH="${runtime_nvidia_libs}:${LD_LIBRARY_PATH:-}"
fi
export LD_PRELOAD="${runtime_nccl}${LD_PRELOAD:+:${LD_PRELOAD}}"

bootstrap_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/kit_bootstrap"
export ISAACLAB_CU130_SITE_PACKAGES="${runtime_site}"
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${bootstrap_dir}"

exec /isaac-sim/python.sh "$@"
