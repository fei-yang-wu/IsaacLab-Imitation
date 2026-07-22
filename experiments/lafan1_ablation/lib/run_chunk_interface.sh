#!/usr/bin/env bash
# Chunk interface runner.
set -euo pipefail

run_chunk_interface() {
    local table="$1"
    local window="$2"
    local interface_id="$3"

    local command_space
    case "${interface_id}" in
        ee_chunk) command_space="ee_trajectory" ;;
        wb_chunk) command_space="full_body_trajectory" ;;
        *)
            log "ERROR: unknown chunk interface ${interface_id}"
            return 2
            ;;
    esac

    resolve_budget_knobs

    local root
    root="$(interface_root "${table}" "${window}" "${interface_id}")"
    local ll_dir="${root}/oracle_ll"
    local dataset_path="${root}/zarr"

    local total_frames num_envs max_iterations
    if [[ "${BUDGET}" == "smoke" ]]; then
        total_frames="${SMOKE_TOTAL_FRAMES:-8192}"
        num_envs="${SMOKE_NUM_ENVS:-16}"
        max_iterations="${SMOKE_MAX_ITERATIONS:-1}"
    else
        total_frames="${TOTAL_FRAMES:-2000000000}"
        num_envs="${NUM_ENVS:-4096}"
        max_iterations=""
    fi

    local task="${CHUNK_TASK:-Isaac-Imitation-G1-v0}"
    local algo="${CHUNK_ALGO:-IPMD}"
    local device="${DEVICE:-cuda:0}"
    local shared_id="${table}_W${window}_${interface_id}_seed${SEED}"
    local command_future_steps=$((window - 1))

    mkdir -p "${root}"
    prepare_fresh_ll_dir "${ll_dir}"
    prepare_fresh_trajectory_dir "${root}/trajectories"
    prepare_cell_zarr_cache "${dataset_path}"
    log "chunk shared stack ${shared_id} -> ${root}"
    log "  command_space=${command_space} command_slots=${window} command_future_steps=${command_future_steps}"
    log "  ll_manifest=${LL_MANIFEST}"
    log "  physics=${PHYSICS_BACKEND:-<task-default>}"
    log "  ll_train_script=$(ll_train_script) total_frames=${total_frames}"

    local ll_train_started_epoch=0
    if [[ "${SKIP_ORACLE_LL:-0}" != "1" ]]; then
        ll_train_started_epoch="$(date +%s)"
        local train_cmd=(
            "${ISAACLAB_PYTHON_CMD[@]}" "$(ll_train_script)"
            --headless
            --device "${device}"
            --num_envs "${num_envs}"
            --task "${task}"
            --algo "${algo}"
            --seed "${SEED}"
            --kit_args=--/app/extensions/fsWatcherEnabled=false
            "agent.collector.total_frames=${total_frames}"
            "agent.logger.log_dir=${ll_dir}"
            "agent.ipmd.use_latent_command=false"
            "env.lafan1_manifest_path=${LL_MANIFEST}"
            "env.dataset_path=${dataset_path}"
            "env.refresh_zarr_dataset=${REFRESH_ZARR_DATASET:-true}"
            "env.latent_patch_past_steps=0"
            "env.latent_patch_future_steps=${command_future_steps}"
            "env.command_observation_source=reference"
            "agent.command_space=${command_space}"
        )
        append_no_random_reset_overrides train_cmd
        append_physics_override train_cmd kitless
        append_wandb_train_overrides train_cmd "${shared_id}_oracle_ll"
        append_smoke_ll_save_interval train_cmd "${total_frames}"
        if [[ -n "${max_iterations}" ]]; then
            train_cmd+=(--max_iterations "${max_iterations}")
        fi
        run_cmd "${train_cmd[@]}"
    fi

    local ll_ckpt
    local ckpt_override=""
    if [[ "${interface_id}" == "ee_chunk" ]]; then
        ckpt_override="${EE_TRAJECTORY_CHECKPOINT:-${CHUNK_LOW_LEVEL_CHECKPOINT:-${LOW_LEVEL_CHECKPOINT:-}}}"
    else
        ckpt_override="${FULL_BODY_TRAJECTORY_CHECKPOINT:-${CHUNK_LOW_LEVEL_CHECKPOINT:-${LOW_LEVEL_CHECKPOINT:-}}}"
    fi
    if ! ll_ckpt="$(resolve_ll_checkpoint_fresh "${ll_dir}" "${ll_train_started_epoch}" "${ckpt_override}")"; then
        if [[ "${DRY_RUN}" == "1" ]]; then
            ll_ckpt="${ll_dir}/models/model_step_smoke.pt"
        else
            log "ERROR: low-level checkpoint missing under ${ll_dir}"
            return 2
        fi
    fi
    log "ll_ckpt=${ll_ckpt}"
    printf '%s\n' "${ll_ckpt}" >"${root}/low_level_checkpoint.txt"

    if [[ "${SKIP_PLANNERS:-0}" == "1" ]]; then
        log "SKIP_PLANNERS=1; skipping per-trajectory planner stages"
        return 0
    fi
    run_chunk_trajectory_planners \
        "${table}" "${window}" "${interface_id}" "${command_space}" "${root}" "${ll_ckpt}"
}
