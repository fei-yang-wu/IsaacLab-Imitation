#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ISAACLAB_DIR="${REPO_ROOT}/IsaacLab"
RLOPT_DIR="${REPO_ROOT}/RLOpt"
IMITATION_TOOLS_DIR="${REPO_ROOT}/ImitationLearningTools"

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

install_pixi_environment() {
    local environment="$1"
    log "Installing Pixi environment: ${environment}"
    (
        cd "${REPO_ROOT}"
        pixi install --environment "${environment}"
    )
}

main() {
    require_command pixi
    ensure_submodules_ready

    if [[ "${PIXI_INSTALL_ALL:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
        log "Installing all Pixi environments."
        (
            cd "${REPO_ROOT}"
            pixi install --all
        )
    else
        local environment="${PIXI_ENVIRONMENT:-default}"
        install_pixi_environment "${environment}"

        if [[ "${INSTALL_LEROBOT:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
            install_pixi_environment "lerobot"
        fi
    fi

    log "Workspace Pixi installation completed."
    log "Use 'pixi run ...' for the default environment or 'pixi run -e isaaclab ...' for Isaac Lab workflows."
}

main "$@"
