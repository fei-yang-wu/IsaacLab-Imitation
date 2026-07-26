#!/usr/bin/env bash
set -euo pipefail

runtime_root="${ISAACLAB_CU130_RUNTIME_ROOT:-/opt/isaaclab-imitation-runtime}"
runtime_python=""
for candidate in \
    "${runtime_root}/bin/python" \
    "/opt/isaaclab-imitation-runtime-spec/.pixi/envs/container-runtime/bin/python"; do
    if [[ -x "${candidate}" ]]; then
        runtime_python="${candidate}"
        break
    fi
done

if [[ -z "${runtime_python}" ]]; then
    echo "[ERROR] CU130 runtime Python not found." >&2
    exit 1
fi

runtime_prefix="$(cd "$(dirname "${runtime_python}")/.." && pwd)"
runtime_site=""
for candidate in "${runtime_prefix}"/lib/python*/site-packages; do
    if [[ -d "${candidate}/torch" ]]; then
        runtime_site="${candidate}"
        break
    fi
done
if [[ -z "${runtime_site}" ]]; then
    echo "[ERROR] CU130 runtime site-packages not found under ${runtime_prefix}." >&2
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
export ISAACLAB_CU130_SITE_PACKAGES="${runtime_site}"

# The outer cluster launcher may be Isaac Sim's Python. Its inherited
# PYTHONPATH includes Kit's copy of the Python standard library, which is not
# compatible with the conda-forge runtime interpreter. Preserve repository and
# Isaac Lab paths while removing only the Kit-interpreter entries.
clean_pythonpath=""
IFS=: read -r -a pythonpath_entries <<< "${PYTHONPATH:-}"
for entry in "${pythonpath_entries[@]}"; do
    if [[ -z "${entry}" || "${entry}" == /isaac-sim/kit/python* ]]; then
        continue
    fi
    clean_pythonpath="${clean_pythonpath:+${clean_pythonpath}:}${entry}"
done
export PYTHONPATH="${clean_pythonpath}"

exec "${runtime_python}" "$@"
