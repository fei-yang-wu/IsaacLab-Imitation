# Motion-count scaling

Run all commands from the repository root. Install the Isaac Lab environment
first:

```bash
pixi install -e isaaclab
```

## Choose the data size

Use one block at a time.

```bash
# 91 motions
N=91
NAME=bs91-deter
MANIFEST=data/bones_seed_100/manifests/g1_bones_seed_100_sonic_filtered_manifest.json
DATASET=data/bones_seed_100/g1_hl_diffsr_sonic_filtered
```

```bash
# 5,000 motions
N=5000
NAME=bs5000-deter
MANIFEST=data/bones_seed_sonic_129k_50hz/manifests/bones-seed-sonic-5000.json
DATASET=data/bones_seed_sonic_129k_50hz/g1_hl_diffsr_5000
```

Each manifest must use its own matching dataset cache.

## Train the skill encoder

```bash
SKILL_DIR="logs/skill_encoder/${NAME}"

pixi run -e isaaclab python scripts/rlopt/train_hl_skill_diffsr.py \
  --headless --assert-kitless --num_envs 16 --seed 0 \
  --output_dir "${SKILL_DIR}" \
  --latent_mode deterministic --z_dim 256 \
  --horizon_steps 10 --encoder_window_mode intermediate \
  --diffsr_feature_dim 256 --num_updates 50000 \
  --reconstruction_eval --window_probe_eval \
  --window_probe_train_batches 8 --window_probe_eval_batches 4 \
  "env.lafan1_manifest_path=${MANIFEST}" \
  "env.dataset_path=${DATASET}" \
  physics=newton_mjwarp env.refresh_zarr_dataset=false
```

Set `env.refresh_zarr_dataset=true` only when building a new cache.

## Train the low-level policy

The original runs used 12,288 environments, 12 rollout steps, and 33,909
updates: 5,000,085,504 environment frames.

```bash
POLICY_NAME="policy-${NAME}-E12288-R12"
POLICY_DIR="logs/policy/${POLICY_NAME}"
SKILL_CHECKPOINT="${SKILL_DIR}/checkpoints/best.pt"

pixi run -e isaaclab python scripts/rlopt/train.py \
  --headless --assert-kitless \
  --task Isaac-Imitation-G1-Latent-v0 --algo IPMD \
  --num_envs 12288 --seed 0 --max_iterations 33909 \
  agent.logger.project_name=g1-bones-seed-scaling \
  agent.logger.group_name=policy \
  "agent.logger.exp_name=${POLICY_NAME}" \
  "agent.logger.log_dir=${POLICY_DIR}" \
  agent.save_interval=100000000 \
  "agent.ipmd.hl_skill_checkpoint_path=${SKILL_CHECKPOINT}" \
  agent.ipmd.hl_skill_horizon_steps=10 \
  agent.ipmd.latent_steps_min=10 \
  agent.ipmd.latent_steps_max=10 \
  agent.ipmd.latent_learning.code_period=10 \
  agent.collector.frames_per_batch=12 \
  agent.loss.mini_batch_size=18432 \
  "env.lafan1_manifest_path=${MANIFEST}" \
  "env.dataset_path=${DATASET}" \
  physics=newton_mjwarp \
  env.sim.physics.solver_cfg.njmax=320 \
  env.sim.physics.solver_cfg.nconmax=40
```

## Evaluate

Evaluate every motion for 1,000 steps with the matching manifest, cache, skill
checkpoint, and policy checkpoint.

```bash
LOW_LEVEL_CHECKPOINT="$(find "${POLICY_DIR}" -type f -name 'model_step_*.pt' | sort -V | tail -n 1)"
EVAL_DIR="logs/qualification/${NAME}/oracle"

pixi run -e isaaclab python experiments/scaling_motion/eval_skill_commander_closed_loop.py \
  --algorithm IPMD \
  --checkpoint "${LOW_LEVEL_CHECKPOINT}" \
  --skill_checkpoint "${SKILL_CHECKPOINT}" \
  --output_dir "${EVAL_DIR}" \
  --label "${NAME}_diffsr_oracle" \
  --num_envs "${N}" --max_steps 1000 \
  --keep_time_out --extend_episode_length_for_max_steps \
  --keep_early_terminations --disable_reward_clipping \
  agent.logger.backend= \
  env.reset_schedule=sequential env.wrap_steps=false \
  env.observations.policy.enable_corruption=false \
  "agent.ipmd.hl_skill_checkpoint_path=${SKILL_CHECKPOINT}" \
  agent.ipmd.hl_skill_horizon_steps=10 \
  agent.ipmd.latent_steps_min=10 \
  agent.ipmd.latent_steps_max=10 \
  agent.ipmd.latent_learning.code_period=10 \
  "env.lafan1_manifest_path=${MANIFEST}" \
  "env.dataset_path=${DATASET}" \
  "env.keys=[qpos,qvel,next_qpos,next_qvel,body_pos_w,body_quat_w,body_lin_vel_w,body_ang_vel_w]" \
  physics=newton_mjwarp \
  env.sim.physics.solver_cfg.njmax=320 \
  env.sim.physics.solver_cfg.nconmax=40
```

Compare the two `summary.json` files under `logs/qualification/<name>/oracle`.
Keep every setting above fixed; only change the manifest, matching cache, and
motion count.
