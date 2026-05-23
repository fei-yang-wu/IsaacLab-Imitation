#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ISAACLAB_DIR="${REPO_ROOT}/IsaacLab"
RLOPT_DIR="${REPO_ROOT}/RLOpt"
IMITATION_TOOLS_DIR="${REPO_ROOT}/ImitationLearningTools"
ISAACLAB_IMITATION_DIR="${REPO_ROOT}/source/isaaclab_imitation"

log() {
    echo "[install] $*"
}

fail() {
    echo "[install][error] $*" >&2
    exit 1
}

require_command() {
    local command_name="$1"
    command -v "${command_name}" >/dev/null 2>&1 || fail "Missing required command: ${command_name}"
}

check_python_version() {
    require_command python

    local python_version
    python_version="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

    if [[ "${python_version}" != "3.11" ]]; then
        fail "Active Python is ${python_version}. IsaacLab 5.1.0 requires Python 3.11. Activate a Python 3.11 conda environment and rerun."
    fi

    log "Using Python ${python_version}"
}

install_uv_with_conda() {
    require_command conda

    if [[ -z "${CONDA_PREFIX:-}" ]]; then
        log "No active conda environment detected. 'conda install -y uv' will target the default conda environment."
    else
        log "Installing uv into conda environment: ${CONDA_PREFIX}"
    fi

    conda install -y conda-forge::uv
    require_command uv
}

submodule_has_code() {
    local path="$1"
    local marker="$2"
    [[ -f "${path}/${marker}" || -d "${path}/${marker}" ]]
}

ensure_submodules_ready() {
    local needs_update=0

    if ! submodule_has_code "${ISAACLAB_DIR}" "pyproject.toml"; then
        log "IsaacLab checkout is incomplete."
        needs_update=1
    fi

    if ! submodule_has_code "${RLOPT_DIR}" "pyproject.toml"; then
        log "RLOpt checkout is incomplete."
        needs_update=1
    fi

    if ! submodule_has_code "${IMITATION_TOOLS_DIR}" "pyproject.toml"; then
        log "ImitationLearningTools checkout is incomplete."
        needs_update=1
    fi

    if [[ "${needs_update}" -eq 1 ]]; then
        log "Initializing and updating git submodules."
        git -C "${REPO_ROOT}" submodule sync --recursive
        git -C "${REPO_ROOT}" submodule update --init --recursive
    else
        log "All submodule checkouts are present."
    fi

    submodule_has_code "${ISAACLAB_DIR}" "pyproject.toml" || fail "IsaacLab is still missing after submodule update."
    submodule_has_code "${RLOPT_DIR}" "pyproject.toml" || fail "RLOpt is still missing after submodule update."
    submodule_has_code "${IMITATION_TOOLS_DIR}" "pyproject.toml" || fail "ImitationLearningTools is still missing after submodule update."
}

install_editable_package() {
    local package_dir="$1"
    log "Installing editable package: ${package_dir}"
    (
        cd "${package_dir}"
        uv pip install -e .
    )
}

install_lerobot_dependencies() {
    case "${INSTALL_LEROBOT:-0}" in
        1|true|TRUE|yes|YES|on|ON)
            log "Installing optional LeRobot streaming dependencies."
            uv pip install -e "${IMITATION_TOOLS_DIR}[lerobot]" datasets
            ;;
        *)
            log "Skipping optional LeRobot dependencies. Set INSTALL_LEROBOT=1 to enable them."
            ;;
    esac
}

install_isaaclab_dependencies() {
    log "Installing Isaac Sim 5.1.0"
    uv pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

    log "Installing PyTorch 2.7.0 / torchvision 0.22.0 for CUDA 12.8"
    uv pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
}

install_isaaclab_packages() {
    local isaaclab_entrypoint=""

    if [[ -x "${ISAACLAB_DIR}/isaaclab" ]]; then
        isaaclab_entrypoint="./isaaclab"
    elif [[ -x "${ISAACLAB_DIR}/isaaclab.sh" ]]; then
        isaaclab_entrypoint="./isaaclab.sh"
    else
        fail "Could not find an executable IsaacLab installer entrypoint in ${ISAACLAB_DIR}"
    fi

    log "Installing IsaacLab packages via ${isaaclab_entrypoint} -i none"
    (
        cd "${ISAACLAB_DIR}"
        "${isaaclab_entrypoint}" -i none
    )
}

install_isaaclab_imitation() {
    log "Installing IsaacLab-Imitation package"
    uv pip install -e "${ISAACLAB_IMITATION_DIR}"
}

main() {
    check_python_version
    install_uv_with_conda
    ensure_submodules_ready
    install_editable_package "${IMITATION_TOOLS_DIR}"
    install_lerobot_dependencies
    install_editable_package "${RLOPT_DIR}"
    install_isaaclab_dependencies
    install_isaaclab_packages
    install_isaaclab_imitation
    log "Workspace installation completed."
}

main "$@"
