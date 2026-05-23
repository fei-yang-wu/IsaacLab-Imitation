#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-SL}"
PYTHON_BIN="${PYTHON_BIN:-python}"

log() {
    echo "[download_g1_lafan1_data] $*"
}

fail() {
    echo "[download_g1_lafan1_data][error] $*" >&2
    exit 1
}

command -v conda >/dev/null 2>&1 || fail "conda is required"

log "Repo root: ${REPO_ROOT}"
log "Downloading and preparing the Hugging Face G1 LAFAN1 dataset into ${REPO_ROOT}/data/"
log "Conda env: ${CONDA_ENV_NAME}"

exec conda run --no-capture-output -n "${CONDA_ENV_NAME}" "${PYTHON_BIN}" \
    "${SCRIPT_DIR}/setup_lafan1_dataset.py" \
    --prepare-npz \
    --headless \
    "$@"
