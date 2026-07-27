#!/usr/bin/env bash
set -uo pipefail
cd /mnt/hsstorage/fwu91/Projects/SL/IsaacLab-Imitation
source experiments/campaigns/2026-07-23-lafan1-planner-capacity/paths.env
SP=/tmp/claude-1169732/-mnt-hsstorage-fwu91-Projects-SL-IsaacLab-Imitation/8340342d-5101-4ea9-8e2f-402c72185d28/scratchpad
VID=logs/interface_baselines/videos_final
FH="env.random_reset_step_min=0 env.random_reset_step_max=0 env.random_reset_full_trajectory=false
env.reset_schedule=sequential env.reference_start_frame=0 env.wrap_steps=false env.episode_length_s=20.0
env.terminations.anchor_pos=null env.terminations.anchor_ori=null env.terminations.ee_body_pos=null
env.terminations.foot_pos_xyz=null env.events.physics_material=null env.events.add_joint_default_pos=null
env.events.base_com=null env.events.push_robot=null env.events.randomize_rigid_body_mass=null"
NEW="physics=newton_mjwarp env.sim.physics.solver_cfg.njmax=320 env.sim.physics.solver_cfg.nconmax=40"

render_fb () { # $1=motion $2=manifest $3=out
  P=logs/interface_baselines/lafan1_planner_capacity_20260723_REBUILD/scaling/seed0/large/full_body_trajectory/planner_pretrain/checkpoints/latest.pt
  timeout 3000 pixi run -e isaaclab python \
    experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/eval_interface_planner_closed_loop.py \
    --headless --device cuda:0 --task "${CHUNK_TASK}" --algorithm IPMD \
    --checkpoint "${FBCHUNK_LOW_LEVEL_CHECKPOINT}" --low_level_command_mode streamed_vanilla \
    --planner_checkpoint "$P" --output_json "$3/summary.json" --label "$(basename $3)" \
    --motion_manifest "$2" --motion_name "$1" --num_envs 4 --steps 700 --seed 0 \
    --state_history_steps 9 --command_past_steps 0 --command_future_steps 9 \
    --planner_update_interval 10 --flow_num_inference_steps 16 --flow_inference_noise_std 0.0 \
    --reset_schedule sequential --reference_start_frame 0 --keep_after_done \
    --pin_command_joint_order on --video --video_length 400 \
    --kit_args=--/app/extensions/fsWatcherEnabled=false \
    agent.logger.backend= env.observations.policy.enable_corruption=false $NEW $FH > "$3.log" 2>&1
  echo "  $(basename $3) exit=$?"
}
render_lat () { # $1=motion $2=manifest $3=cache $4=out
  P=logs/interface_baselines/lafan1_planner_capacity_20260723_REBUILD/scaling/seed0/large/latent_skill/planner_pretrain/checkpoints/latest.pt
  timeout 3000 pixi run -e isaaclab python scripts/rlopt/eval_skill_commander_closed_loop.py \
    --headless --device cuda:0 --task "${LATENT_TASK}" --algorithm IPMD \
    --checkpoint "${LATENT_LOW_LEVEL_CHECKPOINT}" --skill_checkpoint "${LATENT_SKILL_CHECKPOINT}" \
    --planner_checkpoint "$P" --state_history_steps 9 --output_dir "$4" --label "$(basename $4)" \
    --num_envs 4 --max_steps 700 --video --video_length 400 --seed 0 --metric_interval 10 \
    --keep_time_out --keep_early_terminations --motion_name "$1" \
    --flow_num_inference_steps 16 --flow_inference_noise_std 0.0 \
    --kit_args=--/app/extensions/fsWatcherEnabled=false \
    agent.logger.backend= agent.ipmd.command_source=skill_commander \
    "agent.ipmd.skill_commander_checkpoint_path=$P" agent.ipmd.skill_commander_use_achieved_state=true \
    agent.ipmd.skill_commander_flow_num_inference_steps=16 agent.ipmd.skill_commander_flow_inference_noise_std=0.0 \
    "agent.ipmd.hl_skill_checkpoint_path=${LATENT_SKILL_CHECKPOINT}" agent.ipmd.hl_skill_finetune_enabled=false \
    "env.lafan1_manifest_path=$2" "env.dataset_path=$3" env.refresh_zarr_dataset=false \
    env.observations.policy.enable_corruption=false \
    env.latent_command_dim=258 agent.ipmd.latent_dim=258 agent.ipmd.hl_skill_horizon_steps=10 \
    agent.ipmd.hl_skill_command_mode=z agent.ipmd.latent_steps_min=10 agent.ipmd.latent_steps_max=10 \
    agent.ipmd.latent_learning.command_phase_mode=sin_cos agent.ipmd.latent_learning.code_latent_dim=256 \
    agent.ipmd.latent_learning.code_period=10 agent.ipmd.reward_loss_coeff=0.0 agent.ipmd.reward_l2_coeff=0.0 \
    agent.ipmd.reward_grad_penalty_coeff=0.0 agent.ipmd.reward_logit_reg_coeff=0.0 \
    agent.ipmd.reward_param_weight_decay_coeff=0.0 $NEW $FH > "$4.log" 2>&1
  echo "  $(basename $4) exit=$?"
}
mkdir -p $VID
W=data/lafan1/manifests/g1_lafan1_walk1_subject1_manifest.json
FULL=data/lafan1/manifests/g1_lafan1_manifest.json
render_lat walk1_subject1 "$W" data/lafan1/zarr/latent_walk1_subject1_corrected_8e95d557 $VID/walk1_latent_large
render_fb  walk1_subject1 "$W" $VID/walk1_fb_large
render_lat dance1_subject1 "$FULL" data/lafan1/zarr/latent_dance1_subject1_corrected_8e95d557 $VID/dance1_latent_large
render_fb  dance1_subject1 "$FULL" $VID/dance1_fb_large
