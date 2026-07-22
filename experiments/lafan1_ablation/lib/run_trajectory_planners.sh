#!/usr/bin/env bash
# Per-trajectory planner stages.
set -euo pipefail

_latent_overrides() {
    local window="$1"
    local skill_ckpt="$2"
    cat <<EOF
env.latent_command_dim=${LATENT_DIM}
agent.ipmd.latent_dim=${LATENT_DIM}
agent.ipmd.hl_skill_horizon_steps=${window}
agent.ipmd.hl_skill_command_mode=z
agent.ipmd.latent_steps_min=${window}
agent.ipmd.latent_steps_max=${window}
agent.ipmd.latent_learning.command_phase_mode=sin_cos
agent.ipmd.latent_learning.code_latent_dim=${Z_DIM}
agent.ipmd.latent_learning.code_period=${window}
agent.ipmd.hl_skill_checkpoint_path=${skill_ckpt}
agent.ipmd.hl_skill_finetune_enabled=false
EOF
}

run_latent_trajectory_planners() {
    local table="$1"
    local window="$2"
    local interface_id="$3"
    local root="$4"
    local skill_ckpt="$5"
    local ll_ckpt="$6"

    local task="${LATENT_TASK:-Isaac-Imitation-G1-Latent-v0}"
    local algo="${LATENT_ALGO:-IPMD}"
    local device="${DEVICE:-cuda:0}"
    local dataset_path="${root}/zarr"
    local row rank name clean _traj_src single_manifest _steps traj_root
    local planner_dir samples_dir pretrain_ckpt ft_ckpt
    local oracle_dir pretrained_dir finetuned_dir
    local ov reset_ov
    mapfile -t ov < <(_latent_overrides "${window}" "${skill_ckpt}")
    mapfile -t reset_ov < <(no_random_reset_overrides)

    for row in "${TRAJECTORY_ROWS[@]}"; do
        IFS=$'\t' read -r rank name clean _traj_src single_manifest _steps <<<"${row}"
        traj_root="$(trajectory_root "${table}" "${window}" "${interface_id}" "${rank}" "${clean}")"
        planner_dir="${traj_root}/planner_pretrain"
        samples_dir="${traj_root}/oracle_drive_samples"
        oracle_dir="${traj_root}/eval_raw/oracle"
        pretrained_dir="${traj_root}/eval_raw/pretrained"
        finetuned_dir="${traj_root}/eval_raw/finetuned"
        mkdir -p "${planner_dir}" "${samples_dir}" "${oracle_dir}" "${pretrained_dir}" "${finetuned_dir}" \
            "${traj_root}/eval/oracle" "${traj_root}/eval/pretrained" "${traj_root}/eval/finetuned"

        log "latent planner trajectory rank=${rank} name=${name}"
        log "  single_manifest=${single_manifest}"
        log "  traj_root=${traj_root}"

        # Pretrain
        run_isaac_cmd scripts/rlopt/train_skill_commander.py \
            --headless \
            --device "${device}" \
            --task "${task}" \
            --num_envs "${PLANNER_NUM_ENVS}" \
            --seed "${SEED}" \
            --output_dir "${planner_dir}" \
            --skill_checkpoint "${skill_ckpt}" \
            --no_language \
            --state_history_steps "${STATE_HISTORY_STEPS}" \
            --planner_type flow_matching \
            --batch_size "${PLANNER_BATCH_SIZE}" \
            --num_updates "${PRETRAIN_UPDATES}" \
            --log_interval 50 \
            --eval_batches 1 \
            --train_split all \
            --eval_split all \
            --flow_num_inference_steps "${FLOW_STEPS}" \
            --flow_inference_noise_std "${FLOW_NOISE_STD}" \
            "env.lafan1_manifest_path=${single_manifest}" \
            "env.dataset_path=${dataset_path}" \
            "env.refresh_zarr_dataset=false" \
            "${reset_ov[@]}"
        pretrain_ckpt="${planner_dir}/checkpoints/latest.pt"

        # Oracle eval
        run_isaac_cmd scripts/rlopt/eval_skill_commander_closed_loop.py \
            --headless \
            --task "${task}" \
            --algorithm "${algo}" \
            --checkpoint "${ll_ckpt}" \
            --planner_checkpoint "${pretrain_ckpt}" \
            --skill_checkpoint "${skill_ckpt}" \
            --output_dir "${oracle_dir}" \
            --label "${interface_id}_oracle_rank${rank}" \
            --motion_name "${name}" \
            --num_envs "${PLANNER_NUM_ENVS}" \
            --max_steps "${EVAL_STEPS}" \
            --seed "${SEED}" \
            --metric_interval 1 \
            --keep_time_out \
            --flow_num_inference_steps "${FLOW_STEPS}" \
            --flow_inference_noise_std "${FLOW_NOISE_STD}" \
            "agent.ipmd.command_source=hl_skill" \
            "env.lafan1_manifest_path=${single_manifest}" \
            "env.dataset_path=${dataset_path}" \
            "env.refresh_zarr_dataset=false" \
            "${reset_ov[@]}" \
            "${ov[@]}" \
            --kit_args=--/app/extensions/fsWatcherEnabled=false
        promote_eval_summary "${oracle_dir}" "${traj_root}/eval/oracle/summary.json" \
            "${interface_id}" "${table}" "${window}" "${rank}" "${name}" "oracle"

        # Collect rollout rows
        run_isaac_cmd scripts/rlopt/eval_skill_commander_closed_loop.py \
            --headless \
            --task "${task}" \
            --algorithm "${algo}" \
            --checkpoint "${ll_ckpt}" \
            --planner_checkpoint "${pretrain_ckpt}" \
            --skill_checkpoint "${skill_ckpt}" \
            --output_dir "${samples_dir}" \
            --label "${interface_id}_collect_rank${rank}" \
            --motion_name "${name}" \
            --num_envs "${PLANNER_NUM_ENVS}" \
            --max_steps "${COLLECT_STEPS}" \
            --seed "${SEED}" \
            --metric_interval 1 \
            --continue_after_reset \
            --save_rollout_training_samples \
            --keep_time_out \
            --flow_num_inference_steps "${FLOW_STEPS}" \
            --flow_inference_noise_std "${FLOW_NOISE_STD}" \
            "agent.ipmd.command_source=hl_skill" \
            "env.lafan1_manifest_path=${single_manifest}" \
            "env.dataset_path=${dataset_path}" \
            "env.refresh_zarr_dataset=false" \
            "${reset_ov[@]}" \
            "${ov[@]}" \
            --kit_args=--/app/extensions/fsWatcherEnabled=false

        # Pretrained eval
        run_isaac_cmd scripts/rlopt/eval_skill_commander_closed_loop.py \
            --headless \
            --task "${task}" \
            --algorithm "${algo}" \
            --checkpoint "${ll_ckpt}" \
            --planner_checkpoint "${pretrain_ckpt}" \
            --skill_checkpoint "${skill_ckpt}" \
            --output_dir "${pretrained_dir}" \
            --label "${interface_id}_pretrained_rank${rank}" \
            --motion_name "${name}" \
            --num_envs "${PLANNER_NUM_ENVS}" \
            --max_steps "${EVAL_STEPS}" \
            --seed "${SEED}" \
            --metric_interval 1 \
            --keep_time_out \
            --flow_num_inference_steps "${FLOW_STEPS}" \
            --flow_inference_noise_std "${FLOW_NOISE_STD}" \
            "agent.ipmd.command_source=skill_commander" \
            "agent.ipmd.skill_commander_checkpoint_path=${pretrain_ckpt}" \
            "agent.ipmd.skill_commander_embeddings_path=" \
            "agent.ipmd.skill_commander_use_achieved_state=true" \
            "agent.ipmd.skill_commander_flow_num_inference_steps=${FLOW_STEPS}" \
            "agent.ipmd.skill_commander_flow_inference_noise_std=${FLOW_NOISE_STD}" \
            "env.lafan1_manifest_path=${single_manifest}" \
            "env.dataset_path=${dataset_path}" \
            "env.refresh_zarr_dataset=false" \
            "${reset_ov[@]}" \
            "${ov[@]}" \
            --kit_args=--/app/extensions/fsWatcherEnabled=false
        promote_eval_summary "${pretrained_dir}" "${traj_root}/eval/pretrained/summary.json" \
            "${interface_id}" "${table}" "${window}" "${rank}" "${name}" "pretrained"

        # Finetune
        run_cmd "${PYTHON_CMD[@]}" scripts/rlopt/finetune_skill_commander_rollout.py \
            --checkpoint "${pretrain_ckpt}" \
            --samples_dir "${samples_dir}/rollout_training_samples" \
            --output_dir "${traj_root}/planner_finetune" \
            --seed "${SEED}" \
            --num_updates "${FINETUNE_UPDATES}" \
            --batch_size "${PLANNER_BATCH_SIZE}" \
            --lr "${FINETUNE_LR}" \
            --weight_decay "${FINETUNE_WEIGHT_DECAY}" \
            --flow_loss_coeff 1.0 \
            --endpoint_loss_coeff 1.0 \
            --flow_num_inference_steps "${FLOW_STEPS}" \
            --flow_inference_noise_std "${FLOW_NOISE_STD}"
        ft_ckpt="${traj_root}/planner_finetune/checkpoints/latest.pt"

        # Finetuned eval
        run_isaac_cmd scripts/rlopt/eval_skill_commander_closed_loop.py \
            --headless \
            --task "${task}" \
            --algorithm "${algo}" \
            --checkpoint "${ll_ckpt}" \
            --planner_checkpoint "${ft_ckpt}" \
            --skill_checkpoint "${skill_ckpt}" \
            --output_dir "${finetuned_dir}" \
            --label "${interface_id}_finetuned_rank${rank}" \
            --motion_name "${name}" \
            --num_envs "${PLANNER_NUM_ENVS}" \
            --max_steps "${EVAL_STEPS}" \
            --seed "${SEED}" \
            --metric_interval 1 \
            --keep_time_out \
            --flow_num_inference_steps "${FLOW_STEPS}" \
            --flow_inference_noise_std "${FLOW_NOISE_STD}" \
            "agent.ipmd.command_source=skill_commander" \
            "agent.ipmd.skill_commander_checkpoint_path=${ft_ckpt}" \
            "agent.ipmd.skill_commander_embeddings_path=" \
            "agent.ipmd.skill_commander_use_achieved_state=true" \
            "agent.ipmd.skill_commander_flow_num_inference_steps=${FLOW_STEPS}" \
            "agent.ipmd.skill_commander_flow_inference_noise_std=${FLOW_NOISE_STD}" \
            "env.lafan1_manifest_path=${single_manifest}" \
            "env.dataset_path=${dataset_path}" \
            "env.refresh_zarr_dataset=false" \
            "${reset_ov[@]}" \
            "${ov[@]}" \
            --kit_args=--/app/extensions/fsWatcherEnabled=false
        promote_eval_summary "${finetuned_dir}" "${traj_root}/eval/finetuned/summary.json" \
            "${interface_id}" "${table}" "${window}" "${rank}" "${name}" "finetuned"
    done
}

_fsq_overrides() {
    local window="$1"
    local fsq_latent_dim="${FSQ_LATENT_DIM:-64}"
    # shellcheck disable=SC2206
    local fsq_levels=(${FSQ_LEVELS:-8 8 8 5 5})
    local fsq_levels_hydra
    fsq_levels_hydra="[$(IFS=,; echo "${fsq_levels[*]}")]"
    cat <<EOF
env.latent_command_dim=${fsq_latent_dim}
agent.ipmd.latent_dim=${fsq_latent_dim}
agent.ipmd.latent_learning.code_latent_dim=${fsq_latent_dim}
agent.ipmd.latent_steps_min=${window}
agent.ipmd.latent_steps_max=${window}
agent.ipmd.latent_learning.code_period=${window}
agent.ipmd.latent_learning.method=patch_vqvae
agent.ipmd.latent_learning.quantizer=fsq
agent.ipmd.latent_learning.fsq_levels=${fsq_levels_hydra}
agent.ipmd.latent_learning.freeze_encoder=true
agent.ipmd.latent_learning.command_phase_mode=none
agent.ipmd.hl_skill_command_mode=z
agent.ipmd.hl_skill_horizon_steps=${window}
EOF
}

run_fsq_trajectory_planners() {
    local table="$1"
    local window="$2"
    local interface_id="$3"
    local root="$4"
    local ll_ckpt="$5"

    local task="${FSQ_TASK:-Isaac-Imitation-G1-Latent-VQVAE-v0}"
    local algo="${FSQ_ALGO:-IPMD}"
    local device="${DEVICE:-cuda:0}"
    local fsq_latent_dim="${FSQ_LATENT_DIM:-64}"
    local dataset_path="${root}/zarr"
    local row rank name clean _traj_src single_manifest _steps traj_root
    local planner_dir samples_dir pretrain_ckpt ft_ckpt stub_skill
    local oracle_dir pretrained_dir finetuned_dir
    local ov reset_ov
    mapfile -t ov < <(_fsq_overrides "${window}")
    mapfile -t reset_ov < <(no_random_reset_overrides)

    for row in "${TRAJECTORY_ROWS[@]}"; do
        IFS=$'\t' read -r rank name clean _traj_src single_manifest _steps <<<"${row}"
        traj_root="$(trajectory_root "${table}" "${window}" "${interface_id}" "${rank}" "${clean}")"
        planner_dir="${traj_root}/planner_pretrain"
        samples_dir="${traj_root}/oracle_drive_samples"
        oracle_dir="${traj_root}/eval_raw/oracle"
        pretrained_dir="${traj_root}/eval_raw/pretrained"
        finetuned_dir="${traj_root}/eval_raw/finetuned"
        mkdir -p "${planner_dir}" "${samples_dir}" "${oracle_dir}" "${pretrained_dir}" "${finetuned_dir}" \
            "${traj_root}/eval/oracle" "${traj_root}/eval/pretrained" "${traj_root}/eval/finetuned"

        log "fsq planner trajectory rank=${rank} name=${name}"
        log "  task=${task} single_manifest=${single_manifest}"
        log "  traj_root=${traj_root}"

        # Pretrain
        run_isaac_cmd scripts/rlopt/train_skill_commander_from_ipmd_vqvae.py \
            --headless \
            --device "${device}" \
            --task "${task}" \
            --num_envs "${PLANNER_NUM_ENVS}" \
            --seed "${SEED}" \
            --output_dir "${planner_dir}" \
            --ll_checkpoint "${ll_ckpt}" \
            --horizon_steps "${window}" \
            --state_history_steps "${STATE_HISTORY_STEPS}" \
            --z_dim "${fsq_latent_dim}" \
            --planner_type flow_matching \
            --batch_size "${PLANNER_BATCH_SIZE}" \
            --num_updates "${PRETRAIN_UPDATES}" \
            --log_interval 50 \
            --eval_batches 1 \
            --flow_num_inference_steps "${FLOW_STEPS}" \
            --flow_inference_noise_std "${FLOW_NOISE_STD}" \
            "env.lafan1_manifest_path=${single_manifest}" \
            "env.dataset_path=${dataset_path}" \
            "env.refresh_zarr_dataset=false" \
            "${reset_ov[@]}" \
            "${ov[@]}"
        pretrain_ckpt="${planner_dir}/checkpoints/latest.pt"
        stub_skill="${planner_dir}/checkpoints/diffsr_compat_stub_skill.pt"

        # Oracle eval
        run_isaac_cmd scripts/rlopt/eval_skill_commander_closed_loop.py \
            --headless \
            --task "${task}" \
            --algorithm "${algo}" \
            --checkpoint "${ll_ckpt}" \
            --planner_checkpoint "${pretrain_ckpt}" \
            --skill_checkpoint "${stub_skill}" \
            --output_dir "${oracle_dir}" \
            --label "${interface_id}_oracle_rank${rank}" \
            --motion_name "${name}" \
            --num_envs "${PLANNER_NUM_ENVS}" \
            --max_steps "${EVAL_STEPS}" \
            --seed "${SEED}" \
            --metric_interval 1 \
            --keep_time_out \
            --target_z_source ipmd_vqvae \
            "agent.ipmd.command_source=posterior" \
            "env.lafan1_manifest_path=${single_manifest}" \
            "env.dataset_path=${dataset_path}" \
            "env.refresh_zarr_dataset=false" \
            "${reset_ov[@]}" \
            "${ov[@]}" \
            --kit_args=--/app/extensions/fsWatcherEnabled=false
        promote_eval_summary "${oracle_dir}" "${traj_root}/eval/oracle/summary.json" \
            "${interface_id}" "${table}" "${window}" "${rank}" "${name}" "oracle"

        # Collect rollout rows
        run_isaac_cmd scripts/rlopt/eval_skill_commander_closed_loop.py \
            --headless \
            --task "${task}" \
            --algorithm "${algo}" \
            --checkpoint "${ll_ckpt}" \
            --planner_checkpoint "${pretrain_ckpt}" \
            --skill_checkpoint "${stub_skill}" \
            --output_dir "${samples_dir}" \
            --label "${interface_id}_collect_rank${rank}" \
            --motion_name "${name}" \
            --num_envs "${PLANNER_NUM_ENVS}" \
            --max_steps "${COLLECT_STEPS}" \
            --seed "${SEED}" \
            --metric_interval 1 \
            --continue_after_reset \
            --save_rollout_training_samples \
            --keep_time_out \
            --target_z_source ipmd_vqvae \
            "agent.ipmd.command_source=posterior" \
            "env.lafan1_manifest_path=${single_manifest}" \
            "env.dataset_path=${dataset_path}" \
            "env.refresh_zarr_dataset=false" \
            "${reset_ov[@]}" \
            "${ov[@]}" \
            --kit_args=--/app/extensions/fsWatcherEnabled=false

        # Pretrained eval
        run_isaac_cmd scripts/rlopt/eval_skill_commander_closed_loop.py \
            --headless \
            --task "${task}" \
            --algorithm "${algo}" \
            --checkpoint "${ll_ckpt}" \
            --planner_checkpoint "${pretrain_ckpt}" \
            --skill_checkpoint "${stub_skill}" \
            --output_dir "${pretrained_dir}" \
            --label "${interface_id}_pretrained_rank${rank}" \
            --motion_name "${name}" \
            --num_envs "${PLANNER_NUM_ENVS}" \
            --max_steps "${EVAL_STEPS}" \
            --seed "${SEED}" \
            --metric_interval 1 \
            --keep_time_out \
            --target_z_source ipmd_vqvae \
            --flow_num_inference_steps "${FLOW_STEPS}" \
            --flow_inference_noise_std "${FLOW_NOISE_STD}" \
            "agent.ipmd.command_source=skill_commander" \
            "agent.ipmd.skill_commander_checkpoint_path=${pretrain_ckpt}" \
            "agent.ipmd.skill_commander_embeddings_path=" \
            "agent.ipmd.skill_commander_use_achieved_state=true" \
            "agent.ipmd.skill_commander_flow_num_inference_steps=${FLOW_STEPS}" \
            "agent.ipmd.skill_commander_flow_inference_noise_std=${FLOW_NOISE_STD}" \
            "env.lafan1_manifest_path=${single_manifest}" \
            "env.dataset_path=${dataset_path}" \
            "env.refresh_zarr_dataset=false" \
            "${reset_ov[@]}" \
            "${ov[@]}" \
            --kit_args=--/app/extensions/fsWatcherEnabled=false
        promote_eval_summary "${pretrained_dir}" "${traj_root}/eval/pretrained/summary.json" \
            "${interface_id}" "${table}" "${window}" "${rank}" "${name}" "pretrained"

        # Finetune
        run_cmd "${PYTHON_CMD[@]}" scripts/rlopt/finetune_skill_commander_rollout.py \
            --checkpoint "${pretrain_ckpt}" \
            --samples_dir "${samples_dir}/rollout_training_samples" \
            --output_dir "${traj_root}/planner_finetune" \
            --seed "${SEED}" \
            --num_updates "${FINETUNE_UPDATES}" \
            --batch_size "${PLANNER_BATCH_SIZE}" \
            --lr "${FINETUNE_LR}" \
            --weight_decay "${FINETUNE_WEIGHT_DECAY}" \
            --flow_loss_coeff 1.0 \
            --endpoint_loss_coeff 1.0 \
            --flow_num_inference_steps "${FLOW_STEPS}" \
            --flow_inference_noise_std "${FLOW_NOISE_STD}"
        ft_ckpt="${traj_root}/planner_finetune/checkpoints/latest.pt"

        # Finetuned eval
        run_isaac_cmd scripts/rlopt/eval_skill_commander_closed_loop.py \
            --headless \
            --task "${task}" \
            --algorithm "${algo}" \
            --checkpoint "${ll_ckpt}" \
            --planner_checkpoint "${ft_ckpt}" \
            --skill_checkpoint "${stub_skill}" \
            --output_dir "${finetuned_dir}" \
            --label "${interface_id}_finetuned_rank${rank}" \
            --motion_name "${name}" \
            --num_envs "${PLANNER_NUM_ENVS}" \
            --max_steps "${EVAL_STEPS}" \
            --seed "${SEED}" \
            --metric_interval 1 \
            --keep_time_out \
            --target_z_source ipmd_vqvae \
            --flow_num_inference_steps "${FLOW_STEPS}" \
            --flow_inference_noise_std "${FLOW_NOISE_STD}" \
            "agent.ipmd.command_source=skill_commander" \
            "agent.ipmd.skill_commander_checkpoint_path=${ft_ckpt}" \
            "agent.ipmd.skill_commander_embeddings_path=" \
            "agent.ipmd.skill_commander_use_achieved_state=true" \
            "agent.ipmd.skill_commander_flow_num_inference_steps=${FLOW_STEPS}" \
            "agent.ipmd.skill_commander_flow_inference_noise_std=${FLOW_NOISE_STD}" \
            "env.lafan1_manifest_path=${single_manifest}" \
            "env.dataset_path=${dataset_path}" \
            "env.refresh_zarr_dataset=false" \
            "${reset_ov[@]}" \
            "${ov[@]}" \
            --kit_args=--/app/extensions/fsWatcherEnabled=false
        promote_eval_summary "${finetuned_dir}" "${traj_root}/eval/finetuned/summary.json" \
            "${interface_id}" "${table}" "${window}" "${rank}" "${name}" "finetuned"
    done
}

run_chunk_trajectory_planners() {
    local table="$1"
    local window="$2"
    local interface_id="$3"
    local command_space="$4"
    local root="$5"
    local ll_ckpt="$6"

    local task="${CHUNK_TASK:-Isaac-Imitation-G1-v0}"
    local algo="${CHUNK_ALGO:-IPMD}"
    local row rank name clean _traj_src single_manifest _steps traj_root
    local samples_dir pretrain_dir finetune_dir
    local oracle_dir pretrained_dir finetuned_dir
    local reset_ov
    local command_future_steps=$((window - 1))
    mapfile -t reset_ov < <(no_random_reset_overrides)

    for row in "${TRAJECTORY_ROWS[@]}"; do
        IFS=$'\t' read -r rank name clean _traj_src single_manifest _steps <<<"${row}"
        traj_root="$(trajectory_root "${table}" "${window}" "${interface_id}" "${rank}" "${clean}")"
        samples_dir="${traj_root}/oracle_drive_samples"
        pretrain_dir="${traj_root}/planner_pretrain"
        finetune_dir="${traj_root}/planner_finetune"
        oracle_dir="${traj_root}/eval_raw/oracle"
        pretrained_dir="${traj_root}/eval_raw/pretrained"
        finetuned_dir="${traj_root}/eval_raw/finetuned"
        mkdir -p "${samples_dir}" "${pretrain_dir}" "${finetune_dir}" \
            "${oracle_dir}" "${pretrained_dir}" "${finetuned_dir}" \
            "${traj_root}/eval/oracle" "${traj_root}/eval/pretrained" "${traj_root}/eval/finetuned"

        log "chunk planner trajectory rank=${rank} name=${name}"
        log "  command_space=${command_space} command_slots=${window} command_future_steps=${command_future_steps} single_manifest=${single_manifest}"

        # Oracle eval
        run_isaac_cmd experiments/command_space_ablation/evaluate_checkpoint.py \
            --headless \
            --task "${task}" \
            --algo "${algo}" \
            --checkpoint "${ll_ckpt}" \
            --command_space "${command_space}" \
            --command_past_steps "${COMMAND_PAST_STEPS}" \
            --command_future_steps "${command_future_steps}" \
            --command_observation_source planner \
            --planner_mode reference \
            --planner_update_interval "${window}" \
            --motion_manifest "${single_manifest}" \
            --num_envs "${PLANNER_NUM_ENVS}" \
            --steps "${EVAL_STEPS}" \
            --seed "${SEED}" \
            --label "${interface_id}_oracle_rank${rank}" \
            --output_json "${oracle_dir}/summary.json" \
            --preserve_episode_length \
            "env.dataset_path=${root}/zarr" \
            "${reset_ov[@]}" \
            --kit_args=--/app/extensions/fsWatcherEnabled=false
        promote_eval_summary "${oracle_dir}" "${traj_root}/eval/oracle/summary.json" \
            "${interface_id}" "${table}" "${window}" "${rank}" "${name}" "oracle"

        # Collect rollout rows
        run_isaac_cmd experiments/interface_baselines/collect_interface_rollout_samples.py \
            --headless \
            --task "${task}" \
            --algo "${algo}" \
            --checkpoint "${ll_ckpt}" \
            --interface "${command_space}" \
            --output_dir "${samples_dir}" \
            --motion_manifest "${single_manifest}" \
            --num_envs "${PLANNER_NUM_ENVS}" \
            --steps "${COLLECT_STEPS}" \
            --seed "${SEED}" \
            --state_history_steps "${STATE_HISTORY_STEPS}" \
            --planner_interval_steps "${window}" \
            --command_past_steps "${COMMAND_PAST_STEPS}" \
            --command_future_steps "${command_future_steps}" \
            "env.dataset_path=${root}/zarr" \
            "${reset_ov[@]}" \
            --kit_args=--/app/extensions/fsWatcherEnabled=false

        # Pretrain
        run_cmd "${PYTHON_CMD[@]}" experiments/interface_baselines/train_interface_planner.py \
            --samples_dir "${samples_dir}/rollout_training_samples" \
            --output_dir "${pretrain_dir}" \
            --interface "${command_space}" \
            --state_key expert_planner_state \
            --seed "${SEED}" \
            --num_updates "${PRETRAIN_UPDATES}" \
            --batch_size "${PLANNER_BATCH_SIZE}" \
            --lr "${FINETUNE_LR}" \
            --weight_decay "${FINETUNE_WEIGHT_DECAY}" \
            --flow_num_inference_steps "${FLOW_STEPS}" \
            --flow_inference_noise_std "${FLOW_NOISE_STD}"

        # Pretrained eval
        run_isaac_cmd experiments/interface_baselines/eval_interface_planner_closed_loop.py \
            --headless \
            --task "${task}" \
            --algo "${algo}" \
            --checkpoint "${ll_ckpt}" \
            --planner_checkpoint "${pretrain_dir}/checkpoints/latest.pt" \
            --output_json "${pretrained_dir}/summary.json" \
            --label "${interface_id}_pretrained_rank${rank}" \
            --motion_manifest "${single_manifest}" \
            --num_envs "${PLANNER_NUM_ENVS}" \
            --steps "${EVAL_STEPS}" \
            --seed "${SEED}" \
            --state_history_steps "${STATE_HISTORY_STEPS}" \
            --command_past_steps "${COMMAND_PAST_STEPS}" \
            --command_future_steps "${command_future_steps}" \
            --planner_update_interval "${window}" \
            --flow_num_inference_steps "${FLOW_STEPS}" \
            --flow_inference_noise_std "${FLOW_NOISE_STD}" \
            "env.dataset_path=${root}/zarr" \
            "${reset_ov[@]}" \
            --kit_args=--/app/extensions/fsWatcherEnabled=false
        promote_eval_summary "${pretrained_dir}" "${traj_root}/eval/pretrained/summary.json" \
            "${interface_id}" "${table}" "${window}" "${rank}" "${name}" "pretrained"

        # Finetune
        run_cmd "${PYTHON_CMD[@]}" experiments/interface_baselines/train_interface_planner.py \
            --samples_dir "${samples_dir}/rollout_training_samples" \
            --output_dir "${finetune_dir}" \
            --interface "${command_space}" \
            --state_key planner_state \
            --checkpoint "${pretrain_dir}/checkpoints/latest.pt" \
            --seed "${SEED}" \
            --num_updates "${FINETUNE_UPDATES}" \
            --batch_size "${PLANNER_BATCH_SIZE}" \
            --lr "${FINETUNE_LR}" \
            --weight_decay "${FINETUNE_WEIGHT_DECAY}" \
            --flow_num_inference_steps "${FLOW_STEPS}" \
            --flow_inference_noise_std "${FLOW_NOISE_STD}"

        # Finetuned eval
        run_isaac_cmd experiments/interface_baselines/eval_interface_planner_closed_loop.py \
            --headless \
            --task "${task}" \
            --algo "${algo}" \
            --checkpoint "${ll_ckpt}" \
            --planner_checkpoint "${finetune_dir}/checkpoints/latest.pt" \
            --output_json "${finetuned_dir}/summary.json" \
            --label "${interface_id}_finetuned_rank${rank}" \
            --motion_manifest "${single_manifest}" \
            --num_envs "${PLANNER_NUM_ENVS}" \
            --steps "${EVAL_STEPS}" \
            --seed "${SEED}" \
            --state_history_steps "${STATE_HISTORY_STEPS}" \
            --command_past_steps "${COMMAND_PAST_STEPS}" \
            --command_future_steps "${command_future_steps}" \
            --planner_update_interval "${window}" \
            --flow_num_inference_steps "${FLOW_STEPS}" \
            --flow_inference_noise_std "${FLOW_NOISE_STD}" \
            "env.dataset_path=${root}/zarr" \
            "${reset_ov[@]}" \
            --kit_args=--/app/extensions/fsWatcherEnabled=false
        promote_eval_summary "${finetuned_dir}" "${traj_root}/eval/finetuned/summary.json" \
            "${interface_id}" "${table}" "${window}" "${rank}" "${name}" "finetuned"
    done
}
