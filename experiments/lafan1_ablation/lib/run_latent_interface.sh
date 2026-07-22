#!/usr/bin/env bash
# Latent interface runners.
set -euo pipefail

_run_latent_cont_interface() {
    local table="$1"
    local window="$2"
    local interface_id="$3"

    resolve_budget_knobs

    local root
    root="$(interface_root "${table}" "${window}" "${interface_id}")"
    local encoder_dir="${root}/encoder"
    local ll_dir="${root}/oracle_ll"
    local dataset_path="${root}/zarr"
    local window_mode
    window_mode="$(encoder_window_mode_for_w "${window}")"

    local total_frames skill_updates num_envs
    if [[ "${BUDGET}" == "smoke" ]]; then
        total_frames="${SMOKE_TOTAL_FRAMES:-8192}"
        skill_updates="${SMOKE_SKILL_UPDATES:-2}"
        num_envs="${SMOKE_NUM_ENVS:-16}"
    else
        total_frames="${TOTAL_FRAMES:-2000000000}"
        skill_updates="${SKILL_UPDATES:-50000}"
        num_envs="${NUM_ENVS:-4096}"
    fi

    local task="${LATENT_TASK:-Isaac-Imitation-G1-Latent-v0}"
    local algo="${LATENT_ALGO:-IPMD}"
    local device="${DEVICE:-cuda:0}"
    local shared_id="${table}_W${window}_${interface_id}_seed${SEED}"

    mkdir -p "${root}" "${encoder_dir}"
    prepare_fresh_ll_dir "${ll_dir}"
    prepare_fresh_trajectory_dir "${root}/trajectories"
    prepare_cell_zarr_cache "${dataset_path}"
    log "latent_cont shared stack ${shared_id} -> ${root}"
    log "  latent_mode=deterministic W=${window} window_mode=${window_mode}"
    log "  ll_manifest=${LL_MANIFEST}"
    log "  physics=${PHYSICS_BACKEND:-<task-default>}"
    log "  ll_train_script=$(ll_train_script) total_frames=${total_frames}"

    if [[ "${SKIP_ENCODER:-0}" != "1" ]]; then
        local enc_cmd=(
            "${ISAACLAB_PYTHON_CMD[@]}" scripts/rlopt/train_hl_skill_diffsr.py
            --headless
            --device "${device}"
            --task "${task}"
            --num_envs "${num_envs}"
            --seed "${SEED}"
            --output_dir "${encoder_dir}"
            --horizon_steps "${window}"
            --encoder_window_mode "${window_mode}"
            --z_dim "${Z_DIM}"
            --latent_mode deterministic
            --batch_size "${SKILL_BATCH_SIZE:-8192}"
            --num_updates "${skill_updates}"
            --log_interval 100
            --eval_batches "${SKILL_EVAL_BATCHES:-4}"
            --train_split "${SKILL_TRAIN_SPLIT:-train}"
            --eval_split "${SKILL_EVAL_SPLIT:-eval}"
            --eval_trajectory_fraction "${SKILL_EVAL_TRAJECTORY_FRACTION:-0.1}"
            --trajectory_split_seed "${SEED}"
            "env.lafan1_manifest_path=${LL_MANIFEST}"
            "env.dataset_path=${dataset_path}"
            "env.refresh_zarr_dataset=true"
        )
        append_no_random_reset_overrides enc_cmd
        append_physics_override enc_cmd kitless
        append_wandb_diffsr_flags enc_cmd "${shared_id}_encoder"
        run_cmd "${enc_cmd[@]}"
    fi

    local skill_ckpt
    if ! skill_ckpt="$(resolve_skill_checkpoint "${encoder_dir}" "${SKILL_CHECKPOINT:-}")"; then
        if [[ "${DRY_RUN}" == "1" ]]; then
            skill_ckpt="${encoder_dir}/checkpoints/best.pt"
        else
            log "ERROR: skill checkpoint missing under ${encoder_dir}"
            return 2
        fi
    fi

    local ll_train_started_epoch=0
    if [[ "${SKIP_ORACLE_LL:-0}" != "1" ]]; then
        ll_train_started_epoch="$(date +%s)"
        local ll_cmd=(
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
            "agent.ipmd.hl_skill_checkpoint_path=${skill_ckpt}"
            "agent.ipmd.hl_skill_horizon_steps=${window}"
            "agent.ipmd.latent_steps_min=${window}"
            "agent.ipmd.latent_steps_max=${window}"
            "agent.ipmd.latent_learning.code_period=${window}"
            "agent.ipmd.command_source=hl_skill"
            "env.lafan1_manifest_path=${LL_MANIFEST}"
            "env.dataset_path=${dataset_path}"
            "env.refresh_zarr_dataset=false"
        )
        append_no_random_reset_overrides ll_cmd
        append_physics_override ll_cmd kitless
        append_wandb_train_overrides ll_cmd "${shared_id}_oracle_ll"
        append_smoke_ll_save_interval ll_cmd "${total_frames}"
        run_cmd "${ll_cmd[@]}"
    fi

    local ll_ckpt
    if ! ll_ckpt="$(resolve_ll_checkpoint_fresh "${ll_dir}" "${ll_train_started_epoch}" "${LATENT_LOW_LEVEL_CHECKPOINT:-${LOW_LEVEL_CHECKPOINT:-}}")"; then
        if [[ "${DRY_RUN}" == "1" ]]; then
            ll_ckpt="${ll_dir}/models/model_step_smoke.pt"
        else
            log "ERROR: low-level checkpoint missing under ${ll_dir}"
            return 2
        fi
    fi
    log "skill_ckpt=${skill_ckpt}"
    log "ll_ckpt=${ll_ckpt}"
    printf '%s\n' "${skill_ckpt}" >"${root}/skill_checkpoint.txt"
    printf '%s\n' "${ll_ckpt}" >"${root}/low_level_checkpoint.txt"

    if [[ "${SKIP_PLANNERS:-0}" == "1" ]]; then
        log "SKIP_PLANNERS=1; skipping per-trajectory planner stages"
        return 0
    fi
    run_latent_trajectory_planners \
        "${table}" "${window}" "${interface_id}" "${root}" "${skill_ckpt}" "${ll_ckpt}"
}

_run_latent_fsq_interface() {
    local table="$1"
    local window="$2"
    local interface_id="$3"

    resolve_budget_knobs

    local root
    root="$(interface_root "${table}" "${window}" "${interface_id}")"
    local ll_dir="${root}/oracle_ll"
    local dataset_path="${root}/zarr"

    local total_frames num_envs
    if [[ "${BUDGET}" == "smoke" ]]; then
        total_frames="${SMOKE_TOTAL_FRAMES:-8192}"
        num_envs="${SMOKE_NUM_ENVS:-16}"
    else
        total_frames="${TOTAL_FRAMES:-2000000000}"
        num_envs="${NUM_ENVS:-4096}"
    fi

    local task="${FSQ_TASK:-Isaac-Imitation-G1-Latent-VQVAE-v0}"
    local algo="${FSQ_ALGO:-IPMD}"
    local device="${DEVICE:-cuda:0}"
    local fsq_latent_dim="${FSQ_LATENT_DIM:-64}"
    local shared_id="${table}_W${window}_${interface_id}_seed${SEED}"
    # shellcheck disable=SC2206
    local fsq_levels=(${FSQ_LEVELS:-8 8 8 5 5})
    local fsq_levels_hydra
    fsq_levels_hydra="[$(IFS=,; echo "${fsq_levels[*]}")]"

    mkdir -p "${root}"
    prepare_fresh_ll_dir "${ll_dir}"
    prepare_fresh_trajectory_dir "${root}/trajectories"
    prepare_cell_zarr_cache "${dataset_path}"
    log "latent_fsq online stack ${shared_id} -> ${root}"
    log "  task=${task} quantizer=fsq W=${window} latent_dim=${fsq_latent_dim}"
    log "  fsq_levels=${fsq_levels_hydra}"
    log "  ll_manifest=${LL_MANIFEST}"
    log "  physics=${PHYSICS_BACKEND:-<task-default>}"
    log "  ll_train_script=$(ll_train_script) total_frames=${total_frames}"
    log "  note: no DiffSR / offline encoder pretrain (encoder trains online with LL)"

    local ll_train_started_epoch=0
    if [[ "${SKIP_ORACLE_LL:-0}" != "1" ]]; then
        ll_train_started_epoch="$(date +%s)"
        local ll_cmd=(
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
            "agent.ipmd.command_source=posterior"
            "agent.ipmd.latent_learning.method=patch_vqvae"
            "agent.ipmd.latent_learning.quantizer=fsq"
            "agent.ipmd.latent_learning.fsq_levels=${fsq_levels_hydra}"
            "agent.ipmd.latent_learning.freeze_encoder=false"
            "agent.ipmd.latent_dim=${fsq_latent_dim}"
            "agent.ipmd.latent_learning.code_latent_dim=${fsq_latent_dim}"
            "agent.ipmd.latent_steps_min=${window}"
            "agent.ipmd.latent_steps_max=${window}"
            "agent.ipmd.latent_learning.code_period=${window}"
            "agent.ipmd.latent_learning.command_phase_mode=none"
            "env.latent_command_dim=${fsq_latent_dim}"
            "env.lafan1_manifest_path=${LL_MANIFEST}"
            "env.dataset_path=${dataset_path}"
            "env.refresh_zarr_dataset=true"
        )
        append_no_random_reset_overrides ll_cmd
        append_physics_override ll_cmd kitless
        append_wandb_train_overrides ll_cmd "${shared_id}_online_ll"
        append_smoke_ll_save_interval ll_cmd "${total_frames}"
        run_cmd "${ll_cmd[@]}"
    fi

    local ll_ckpt
    if ! ll_ckpt="$(resolve_ll_checkpoint_fresh "${ll_dir}" "${ll_train_started_epoch}" "${LATENT_LOW_LEVEL_CHECKPOINT:-${LOW_LEVEL_CHECKPOINT:-}}")"; then
        if [[ "${DRY_RUN}" == "1" ]]; then
            ll_ckpt="${ll_dir}/models/model_step_smoke.pt"
        else
            log "ERROR: low-level checkpoint missing under ${ll_dir}"
            return 2
        fi
    fi
    log "ll_ckpt=${ll_ckpt}"
    printf '%s\n' "${ll_ckpt}" >"${root}/low_level_checkpoint.txt"
    : >"${root}/skill_checkpoint.txt"
    printf 'online_fsq_ipmd_vqvae\n' >"${root}/encoder_protocol.txt"

    if [[ "${SKIP_PLANNERS:-0}" == "1" ]]; then
        log "SKIP_PLANNERS=1; skipping per-trajectory FSQ planner stages"
        return 0
    fi
    run_fsq_trajectory_planners \
        "${table}" "${window}" "${interface_id}" "${root}" "${ll_ckpt}"
}

run_latent_interface() {
    local table="$1"
    local window="$2"
    local interface_id="$3"

    case "${interface_id}" in
        latent_cont)
            _run_latent_cont_interface "${table}" "${window}" "${interface_id}"
            ;;
        latent_fsq)
            _run_latent_fsq_interface "${table}" "${window}" "${interface_id}"
            ;;
        *)
            log "ERROR: unknown latent interface ${interface_id}"
            return 2
            ;;
    esac
}
