#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PIXI_ENVIRONMENT="${PIXI_ENVIRONMENT:-isaaclab}"
PYTHON_BIN="${PYTHON_BIN:-python}"

log() {
    echo "[download_g1_lafan1_data] $*"
}

fail() {
    echo "[download_g1_lafan1_data][error] $*" >&2
    exit 1
}

command -v pixi >/dev/null 2>&1 || fail "pixi is required"

log "Repo root: ${REPO_ROOT}"
log "Downloading and preparing the Hugging Face G1 LAFAN1 dataset into ${REPO_ROOT}/data/"
log "Pixi environment: ${PIXI_ENVIRONMENT}"

cd "${REPO_ROOT}"
exec pixi run --environment "${PIXI_ENVIRONMENT}" "${PYTHON_BIN}" \
    "scripts/setup_lafan1_dataset.py" --prepare-npz --headless "$@"
