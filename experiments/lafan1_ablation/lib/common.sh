#!/usr/bin/env bash
# Shared helpers for LAFAN1 horizon ablation.
set -euo pipefail

_lafan1_ablation_lib_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_lafan1_ablation_root="$(cd "${_lafan1_ablation_lib_dir}/.." && pwd)"
if ! REPO_ROOT="$(git -C "${_lafan1_ablation_root}" rev-parse --show-toplevel 2>/dev/null)"; then
    REPO_ROOT="$(cd "${_lafan1_ablation_root}/../.." && pwd)"
fi
cd "${REPO_ROOT}"

PYTHON_CMD_STR="${LAFAN1_ABLATION_PYTHON_CMD:-pixi run python}"
ISAACLAB_PYTHON_CMD_STR="${LAFAN1_ABLATION_ISAACLAB_PYTHON_CMD:-pixi run -e isaaclab python}"
# shellcheck disable=SC2206
PYTHON_CMD=(${PYTHON_CMD_STR})
# shellcheck disable=SC2206
ISAACLAB_PYTHON_CMD=(${ISAACLAB_PYTHON_CMD_STR})

MATRIX_YAML="${MATRIX_YAML:-${_lafan1_ablation_root}/matrix.yaml}"
FULL_MANIFEST="${FULL_MANIFEST:-data/lafan1/manifests/g1_lafan1_manifest.json}"
LL_MANIFEST="${LL_MANIFEST:-${FULL_MANIFEST}}"
SEED="${SEED:-0}"
DRY_RUN="${DRY_RUN:-0}"
BUDGET="${BUDGET:-full}"  # full | smoke
RANKS="${RANKS:-}"        # empty -> budget default (smoke: 0, full: all)

OUTPUT_ROOT="${OUTPUT_ROOT:-logs/lafan1_ablation/seed${SEED}}"
TRAJECTORY_MANIFEST_ROOT="${TRAJECTORY_MANIFEST_ROOT:-${OUTPUT_ROOT}/trajectory_manifests}"

LOGGER_BACKEND="${LOGGER_BACKEND:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-G1-Imitation-LAFAN1-Ablation}"
WANDB_GROUP="${WANDB_GROUP:-lafan1_ablation_seed${SEED}}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

PHYSICS_BACKEND="${PHYSICS_BACKEND:-}"

log() { printf '[lafan1_ablation] %s\n' "$*"; }

append_wandb_train_overrides() {
    local -n _cmd_ref="$1"
    local exp_name="$2"
    _cmd_ref+=("agent.logger.backend=${LOGGER_BACKEND}")
    _cmd_ref+=("agent.logger.exp_name=${exp_name}")
    if [[ "${LOGGER_BACKEND}" == "wandb" ]]; then
        _cmd_ref+=("agent.logger.project_name=${WANDB_PROJECT}")
        _cmd_ref+=("agent.logger.group_name=${WANDB_GROUP}")
        if [[ -n "${WANDB_ENTITY}" ]]; then
            _cmd_ref+=("agent.logger.entity=${WANDB_ENTITY}")
        fi
    fi
}

append_wandb_diffsr_flags() {
    local -n _cmd_ref="$1"
    local run_name="$2"
    _cmd_ref+=(--logger_backend "${LOGGER_BACKEND}")
    if [[ "${LOGGER_BACKEND}" == "wandb" ]]; then
        _cmd_ref+=(--wandb_project "${WANDB_PROJECT}")
        _cmd_ref+=(--wandb_group "${WANDB_GROUP}")
        _cmd_ref+=(--wandb_run_name "${run_name}")
        if [[ -n "${WANDB_ENTITY}" ]]; then
            _cmd_ref+=(--wandb_entity "${WANDB_ENTITY}")
        fi
    fi
}

append_smoke_ll_save_interval() {
    local -n _cmd_ref="$1"
    local _total_frames="$2"
    if [[ "${BUDGET}" == "smoke" ]]; then
        _cmd_ref+=("agent.save_interval=${SMOKE_SAVE_INTERVAL:-1}")
    fi
}

run_cmd() {
    printf '[CMD]'
    printf ' %q' "$@"
    printf '\n'
    if [[ "${DRY_RUN}" == "1" ]]; then
        return 0
    fi
    # Do not use `local` here: in bash, `local st=$?` can clobber the status.
    "$@"
    st=$?
    if [[ "${st}" -ne 0 ]]; then
        log "ERROR: command failed with exit ${st}"
        exit "${st}"
    fi
}

using_newton_backend() {
    [[ "${PHYSICS_BACKEND}" == newton* ]]
}

physics_override_arg() {
    if [[ -z "${PHYSICS_BACKEND}" ]]; then
        return 0
    fi
    printf 'physics=%s' "${PHYSICS_BACKEND}"
}

append_physics_override() {
    local -n _cmd_ref="$1"
    local mode="${2:-}"
    if [[ "${mode}" == "kitless" ]] && using_newton_backend; then
        _cmd_ref+=(--assert-kitless)
    fi
    local physics_arg
    physics_arg="$(physics_override_arg)"
    if [[ -n "${physics_arg}" ]]; then
        _cmd_ref+=("${physics_arg}")
    fi
}

run_isaac_cmd() {
    local cmd=("${ISAACLAB_PYTHON_CMD[@]}" "$@")
    append_physics_override cmd
    run_cmd "${cmd[@]}"
}

no_random_reset_overrides() {
    cat <<EOF
env.reference_start_frame=0
env.random_reset_step_min=0
env.random_reset_step_max=0
env.random_reset_full_trajectory=false
EOF
}

append_no_random_reset_overrides() {
    local -n _cmd_ref="$1"
    _cmd_ref+=(
        "env.reference_start_frame=0"
        "env.random_reset_step_min=0"
        "env.random_reset_step_max=0"
        "env.random_reset_full_trajectory=false"
    )
}

ll_train_script() {
    if [[ -n "${LL_TRAIN_SCRIPT:-}" ]]; then
        printf '%s' "${LL_TRAIN_SCRIPT}"
    else
        printf '%s' "scripts/rlopt/train.py"
    fi
}

fresh_stack_enabled() {
    if [[ -n "${FRESH_STACK:-}" ]]; then
        [[ "${FRESH_STACK}" == "1" ]]
        return
    fi
    [[ "${BUDGET}" != "smoke" ]]
}

archive_or_wipe_dir() {
    local path="$1"
    local label="$2"
    if [[ ! -e "${path}" ]]; then
        return 0
    fi
    if [[ "${DRY_RUN}" == "1" ]]; then
        log "FRESH_STACK: would wipe ${label}: ${path}"
        return 0
    fi
    local stamp
    stamp="$(date +%Y%m%d_%H%M%S)"
    local bak="${path}.bak_${stamp}"
    log "FRESH_STACK: moving ${label} -> ${bak}"
    mv "${path}" "${bak}"
}

prepare_fresh_ll_dir() {
    local ll_dir="$1"
    if ! fresh_stack_enabled; then
        mkdir -p "${ll_dir}"
        return 0
    fi
    if [[ -d "${ll_dir}" ]] && find "${ll_dir}" -type f -name 'model_step_*.pt' -print -quit | grep -q .; then
        archive_or_wipe_dir "${ll_dir}" "oracle_ll"
    fi
    mkdir -p "${ll_dir}"
}

prepare_fresh_trajectory_dir() {
    local traj_root="$1"
    if ! fresh_stack_enabled; then
        mkdir -p "${traj_root}"
        return 0
    fi
    if [[ -d "${traj_root}" ]]; then
        archive_or_wipe_dir "${traj_root}" "trajectories"
    fi
    mkdir -p "${traj_root}"
}

prepare_cell_zarr_cache() {
    local dataset_path="$1"
    mkdir -p "${dataset_path}"
    if [[ -n "${ISAACLAB_IMITATION_LAFAN1_ZARR_CACHE_ROOT:-}" ]]; then
        mkdir -p "${ISAACLAB_IMITATION_LAFAN1_ZARR_CACHE_ROOT}"
        return 0
    fi
    export ISAACLAB_IMITATION_LAFAN1_ZARR_CACHE_ROOT="${dataset_path}"
    log "zarr cache root -> ${ISAACLAB_IMITATION_LAFAN1_ZARR_CACHE_ROOT}"
}

resolve_ll_checkpoint_fresh() {
    local ll_dir="$1"
    local min_mtime_epoch="${2:-0}"
    local override="${3:-}"
    local found
    if [[ -n "${override}" ]]; then
        printf '%s' "${override}"
        return 0
    fi
    found="$(find_latest_model_checkpoint "${ll_dir}" || true)"
    if [[ -z "${found}" ]]; then
        return 1
    fi
    if [[ "${min_mtime_epoch}" -gt 0 ]]; then
        local mtime
        mtime="$(stat -c '%Y' "${found}" 2>/dev/null || stat -f '%m' "${found}")"
        if [[ "${mtime}" -lt "${min_mtime_epoch}" ]]; then
            log "ERROR: refusing stale LL checkpoint (mtime ${mtime} < ${min_mtime_epoch}): ${found}"
            return 1
        fi
    fi
    printf '%s' "${found}"
}

resolve_budget_knobs() {
    if [[ "${BUDGET}" == "smoke" ]]; then
        PRETRAIN_UPDATES="${PRETRAIN_UPDATES:-2}"
        FINETUNE_UPDATES="${FINETUNE_UPDATES:-2}"
        EVAL_STEPS="${EVAL_STEPS:-16}"
        PLANNER_ROWS_PER_TRAJECTORY="${PLANNER_ROWS_PER_TRAJECTORY:-2}"
        COLLECT_STEPS="${COLLECT_STEPS:-${PLANNER_ROWS_PER_TRAJECTORY}}"
        PLANNER_NUM_ENVS="${PLANNER_NUM_ENVS:-1}"
        PLANNER_BATCH_SIZE="${PLANNER_BATCH_SIZE:-256}"
    else
        PRETRAIN_UPDATES="${PRETRAIN_UPDATES:-2000}"
        FINETUNE_UPDATES="${FINETUNE_UPDATES:-2000}"
        EVAL_STEPS="${EVAL_STEPS:-500}"
        PLANNER_ROWS_PER_TRAJECTORY="${PLANNER_ROWS_PER_TRAJECTORY:-1000}"
        COLLECT_STEPS="${COLLECT_STEPS:-${PLANNER_ROWS_PER_TRAJECTORY}}"
        PLANNER_NUM_ENVS="${PLANNER_NUM_ENVS:-1}"
        PLANNER_BATCH_SIZE="${PLANNER_BATCH_SIZE:-256}"
    fi
    FLOW_STEPS="${FLOW_STEPS:-16}"
    FLOW_NOISE_STD="${FLOW_NOISE_STD:-0.0}"
    FINETUNE_LR="${FINETUNE_LR:-1.0e-4}"
    FINETUNE_WEIGHT_DECAY="${FINETUNE_WEIGHT_DECAY:-1.0e-4}"
    STATE_HISTORY_STEPS="${STATE_HISTORY_STEPS:-9}"
    COMMAND_PAST_STEPS="${COMMAND_PAST_STEPS:-0}"
    Z_DIM="${Z_DIM:-256}"
    LATENT_DIM="${LATENT_DIM:-$((Z_DIM + 2))}"
    export PRETRAIN_UPDATES FINETUNE_UPDATES EVAL_STEPS COLLECT_STEPS
    export PLANNER_ROWS_PER_TRAJECTORY
    export PLANNER_NUM_ENVS PLANNER_BATCH_SIZE FLOW_STEPS FLOW_NOISE_STD
    export FINETUNE_LR FINETUNE_WEIGHT_DECAY STATE_HISTORY_STEPS COMMAND_PAST_STEPS
    export Z_DIM LATENT_DIM
}

find_latest_model_checkpoint() {
    local root="$1"
    if [[ ! -d "${root}" ]]; then
        return 1
    fi
    find "${root}" -type f -name 'model_step_*.pt' -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr \
        | head -n 1 \
        | cut -d' ' -f2-
}

resolve_skill_checkpoint() {
    local encoder_dir="$1"
    local override="${2:-}"
    if [[ -n "${override}" ]]; then
        printf '%s' "${override}"
        return 0
    fi
    if [[ -f "${encoder_dir}/checkpoints/best.pt" ]]; then
        printf '%s' "${encoder_dir}/checkpoints/best.pt"
        return 0
    fi
    if [[ -f "${encoder_dir}/checkpoints/latest.pt" ]]; then
        printf '%s' "${encoder_dir}/checkpoints/latest.pt"
        return 0
    fi
    return 1
}

resolve_ll_checkpoint() {
    local ll_dir="$1"
    local override="${2:-}"
    if [[ -n "${override}" ]]; then
        printf '%s' "${override}"
        return 0
    fi
    local found
    found="$(find_latest_model_checkpoint "${ll_dir}" || true)"
    if [[ -n "${found}" ]]; then
        printf '%s' "${found}"
        return 0
    fi
    return 1
}

promote_eval_summary() {
    local src_dir="$1"
    local dst_json="$2"
    local interface_id="$3"
    local table="$4"
    local window="$5"
    local rank="$6"
    local traj_name="$7"
    local setting="$8"
    local src_json="${src_dir}/summary.json"
    if [[ "${DRY_RUN}" == "1" ]]; then
        log "promote ${setting}: ${src_json} -> ${dst_json}"
        return 0
    fi
    if [[ ! -f "${src_json}" ]]; then
        log "ERROR: missing eval summary: ${src_json}"
        return 2
    fi
    run_cmd "${PYTHON_CMD[@]}" "${_lafan1_ablation_lib_dir}/promote_eval_summary.py" \
        --src "${src_json}" \
        --dst "${dst_json}" \
        --interface "${interface_id}" \
        --table "${table}" \
        --window "${window}" \
        --rank "${rank}" \
        --trajectory "${traj_name}" \
        --setting "${setting}"
}

encoder_window_mode_for_w() {
    # W=1 has no intermediate frames, so fall back to full.
    if [[ -n "${ENCODER_WINDOW_MODE:-}" ]]; then
        printf '%s' "${ENCODER_WINDOW_MODE}"
        return 0
    fi
    local w="$1"
    if [[ "${w}" -le 1 ]]; then
        printf 'full'
    else
        printf 'intermediate'
    fi
}

resolve_ranks() {
    if [[ -n "${RANKS}" ]]; then
        printf '%s' "${RANKS}"
        return 0
    fi
    if [[ "${BUDGET}" == "smoke" ]]; then
        printf '0'
    else
        printf 'all'
    fi
}

ensure_full_manifest() {
    if [[ -f "${FULL_MANIFEST}" ]]; then
        return 0
    fi
    if [[ "${DRY_RUN}" == "1" ]]; then
        log "WARN: manifest missing (${FULL_MANIFEST}); DRY_RUN continues with planned paths"
        return 0
    fi
    cat >&2 <<EOF
[ERROR] LAFAN1 manifest not found: ${FULL_MANIFEST}

Prepare data first:
  ./scripts/data/download_g1_lafan1_data.sh
or:
  pixi run python scripts/data/write_lafan1_npz_manifest.py \\
      --npz_dir data/lafan1/npz/g1 \\
      --manifest_path data/lafan1/manifests/g1_lafan1_manifest.json
EOF
    return 2
}

# Writes one-motion manifests and populates TRAJECTORY_ROWS as lines:
#   rank<TAB>name<TAB>clean_name<TAB>traj_root<TAB>single_manifest<TAB>steps
prepare_trajectory_manifests() {
    if [[ "${SKIP_PLANNERS:-0}" == "1" ]]; then
        TRAJECTORY_ROWS=()
        log "SKIP_PLANNERS=1; no per-trajectory manifests needed"
        return 0
    fi

    local ranks_spec
    ranks_spec="$(resolve_ranks)"
    TRAJECTORY_ROWS=()
    export TRAJECTORY_MANIFEST_ROOT

    if [[ "${DRY_RUN}" == "1" && ! -f "${FULL_MANIFEST}" ]]; then
        # Synthetic single-trajectory plan for dry-run without data.
        TRAJECTORY_ROWS+=(
            $'0\tsmoke_motion\tsmoke_motion\t'"${TRAJECTORY_MANIFEST_ROOT}/rank_0_smoke_motion"$'\t'"${TRAJECTORY_MANIFEST_ROOT}/rank_0_smoke_motion/manifest_single.json"$'\t100'
        )
        log "DRY_RUN synthetic trajectory: rank=0 smoke_motion"
        return 0
    fi

    ensure_full_manifest
    mkdir -p "${TRAJECTORY_MANIFEST_ROOT}"
    local line
    while IFS= read -r line; do
        [[ -n "${line}" ]] || continue
        TRAJECTORY_ROWS+=("${line}")
    done < <(
        "${PYTHON_CMD[@]}" experiments/interface_baselines/select_lafan1_trajectories.py \
            --manifest "${FULL_MANIFEST}" \
            --ranks "${ranks_spec}" \
            --output_root "${TRAJECTORY_MANIFEST_ROOT}"
    )
    if [[ "${#TRAJECTORY_ROWS[@]}" -eq 0 ]]; then
        log "ERROR: no trajectories selected (ranks=${ranks_spec})"
        return 2
    fi
    log "selected ${#TRAJECTORY_ROWS[@]} trajector(ies) for per-trajectory planners (ranks=${ranks_spec})"
}

interface_root() {
    local table="$1"
    local window="$2"
    local interface="$3"
    printf '%s/%s/W%s/%s' "${OUTPUT_ROOT}" "${table}" "${window}" "${interface}"
}

trajectory_root() {
    local table="$1"
    local window="$2"
    local interface="$3"
    local rank="$4"
    local clean_name="$5"
    printf '%s/trajectories/rank_%s_%s' \
        "$(interface_root "${table}" "${window}" "${interface}")" \
        "${rank}" \
        "${clean_name}"
}

print_matrix_plan() {
    local table="$1"
    local window="$2"
    local interfaces="${INTERFACES:-latent_cont latent_fsq ee_chunk wb_chunk}"
    log "protocol=one_planner_per_trajectory table=${table} budget=${BUDGET} dry_run=${DRY_RUN}"
    log "output_root=${OUTPUT_ROOT}"
    log "ll_manifest=${LL_MANIFEST}"
    log "full_manifest=${FULL_MANIFEST}"
    log "trajectories=${#TRAJECTORY_ROWS[@]}"
    local iface row rank name clean
    for iface in ${interfaces}; do
        log "interface=${iface} W=${window} window_mode=$(encoder_window_mode_for_w "${window}")"
        log "  shared: encoder+oracle_ll under $(interface_root "${table}" "${window}" "${iface}")"
        for row in "${TRAJECTORY_ROWS[@]}"; do
            IFS=$'\t' read -r rank name clean _rest <<<"${row}"
            log "  planner cell: rank=${rank} name=${name} -> $(trajectory_root "${table}" "${window}" "${iface}" "${rank}" "${clean}")"
        done
    done
}

write_eval_stubs() {
    local traj_root="$1"
    local interface_id="$2"
    local table="$3"
    local window="$4"
    local rank="$5"
    local traj_name="$6"
    local extra_json="${7:-}"

    local eval_dir="${traj_root}/eval"
    mkdir -p "${eval_dir}/oracle" "${eval_dir}/pretrained" "${eval_dir}/finetuned"
    if [[ "${DRY_RUN}" == "1" || "${WRITE_EVAL_STUBS:-1}" != "1" ]]; then
        return 0
    fi
    local setting
    for setting in oracle pretrained finetuned; do
        cat >"${eval_dir}/${setting}/summary.json" <<EOF
{
  "interface": "${interface_id}",
  "table": "${table}",
  "window": ${window},
  "rank": ${rank},
  "trajectory": "${traj_name}",
  "setting": "${setting}",
  "planner_unit": "trajectory",
  "status": "pending_full_eval",
  "success_rate": null,
  "mpjpe_l_mm": null,
  "e_vel": null,
  "e_acc": null
  ${extra_json}
}
EOF
    done
}

export REPO_ROOT PYTHON_CMD ISAACLAB_PYTHON_CMD MATRIX_YAML
export FULL_MANIFEST LL_MANIFEST SEED DRY_RUN BUDGET RANKS
export OUTPUT_ROOT TRAJECTORY_MANIFEST_ROOT
export LOGGER_BACKEND WANDB_PROJECT WANDB_GROUP WANDB_ENTITY PHYSICS_BACKEND
export TRAJECTORY_ROWS
