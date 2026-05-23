# LeRobot Offline Pretraining

Last refreshed: 2026-05-12.

This page records the current offline dataset approach for G1 latent bilinear
pretraining. The implementation keeps internet-facing dataset ingestion out of
the optimizer loop and treats TorchRL replay as the local training cache. The
top-level repo pins the exact RLOpt and ImitationLearningTools commits required
for this LeRobot path.

## Current Decision

Use the Unitree WBT LeRobot datasets as the storage and streaming format, then
convert episodes into canonical TensorDict transitions before training samples
are drawn. The first target dataset is:

```text
unitreerobotics/G1_WBT_Brainco_Pickup_Pillow
```

Keep the dishwasher WBT dataset as the next target after the first mapper is
validated:

```text
unitreerobotics/G1_WBT_Brainco_Collect_Plates_Into_Dishwasher
```

For serious IPMD/bilinear work, only use `Isaac-Imitation-G1-Latent-v0`.
Vanilla G1 is still useful for mapper/debug checks, but not as the current
comparison surface.

## Ownership

`./ImitationLearningTools` owns the reusable data layer:

- `iltools/datasets/lerobot_stream.py`
- `UnitreeG1WBT29DofMapper`
- `StreamingTensorDictReplayCache`
- fake WBT episode tests under `tests/datasets/test_lerobot_stream.py`

`./RLOpt` owns the algorithm-facing cache/sampler layer:

- `OfflineDatasetConfig` on `RLOptConfig`
- `build_offline_expert_sampler(...)`
- `StreamingOfflineExpertSampler`
- bilinear offline pretrain sampling from the offline cache when configured

`IsaacLab-Imitation` owns env-specific action constants and debugging tools:

- `ImitationRLEnv.get_offline_dataset_mapper_params()`
- default Unitree WBT config on the G1 bilinear agent config
- `scripts/preview_unitree_lerobot_episode.py`
- `scripts/replay_unitree_lerobot_reference.py`

## Tensor Contract

The Unitree WBT mapper consumes low-dimensional fields:

- `episode_index`
- `observation.state.robot_q_current`
- `action.robot_q_desired`

The expected Unitree configuration width is `36`: root position `0:3`, root
quaternion `3:7`, and 29 G1 joints `7:`.

`robot_q_current[7:]` and `robot_q_desired[7:]` are treated as Unitree SDK motor
order, not Isaac action/articulation order:

```text
left_hip_pitch, left_hip_roll, left_hip_yaw, left_knee,
left_ankle_pitch, left_ankle_roll,
right_hip_pitch, right_hip_roll, right_hip_yaw, right_knee,
right_ankle_pitch, right_ankle_roll,
waist_yaw, waist_roll, waist_pitch,
left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw, left_elbow,
left_wrist_roll, left_wrist_pitch, left_wrist_yaw,
right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw, right_elbow,
right_wrist_roll, right_wrist_pitch, right_wrist_yaw
```

The ILT mapper reorders those joints into the env-provided target/action order
before producing `joint_pos`, `joint_pos_rel`, `expert_motion`, and
`expert_action`.

The mapper produces the bilinear G1 training keys before data enters the replay
buffer:

- `("policy", "base_ang_vel")`
- `("policy", "joint_pos_rel")`
- `("policy", "joint_vel_rel")`
- `("policy", "last_action")`
- matching `("next", "policy", ...)` keys
- `action`
- `expert_action`
- `("policy", "expert_motion")`, critic copy, and reward-input copy
- `("policy", "expert_anchor_pos_b")` and `("policy", "expert_anchor_ori_b")`

`expert_action` is computed from the desired robot configuration:

```text
expert_action = (robot_q_desired[7:] in target order - default_joint_pos) / action_scale
```

Here `default_joint_pos` means the `JointPositionAction` offset used to invert
the env action transform, in the same target/action order. It is not required to equal
`robot.data.default_joint_pos`. `last_action` is the previous valid
`expert_action`, with zeros only at episode start. `joint_vel_rel` and
`base_ang_vel` are finite-differenced per episode because the Unitree WBT
dataset does not currently expose qvel.

## Validation Rule

Fail fast at construction or cache-fill time. The optimizer loop should not run
schema guards every iteration.

Current construction-time checks include:

- offline source, mapper, and cache backend are supported
- required LeRobot fields exist
- `robot_q_current` and `robot_q_desired` have width `36`
- G1 action/default/scale widths are `29`
- action scale contains no zero entries
- env action manager uses `JointPositionAction`
- env action offset pool has 29 values and can be exported as
  `default_joint_pos_pool`
- offline pretrain can sample a small preflight batch before training begins

## Training Path

LeRobot handles internet streaming. The background producer groups rows by
episode, maps each episode into TensorDict transitions, and extends a local
TorchRL `TensorDictReplayBuffer` backed by `LazyMemmapStorage` and
`TensorDictRoundRobinWriter`. Bilinear offline pretraining samples from that
local cache.

TorchRL `prefetch` is only used on the local replay-buffer `sample()` side after
data is already cached. Remote decode and TensorDict conversion stay in the
producer thread.

```bash
CONDA_ENV="${CONDA_ENV:-SL}"
conda activate "$CONDA_ENV"

# Hugging Face Hub CLI for LeRobot dataset access.
uv pip install --system -U "huggingface_hub[cli]"
hf auth login
hf auth whoami

# Optional: only for direct git push/pull to https://huggingface.co.
# This uses Git's plaintext store helper, scoped to Hugging Face only.
git config --global credential.https://huggingface.co.helper store

# If you are already logged in:
hf auth list
TOKEN_NAME=home-ubuntu
hf auth switch --token-name "$TOKEN_NAME" --add-to-git-credential
```

Small latent smoke command:

```bash
TERM=xterm PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 TORCHDYNAMO_DISABLE=1 \
conda run -n "${CONDA_ENV:-SL}" python scripts/rlopt/train.py \
    --task Isaac-Imitation-G1-Latent-v0 \
    --algo IPMD_BILINEAR \
    --num_envs 16 \
    --max_iterations 1 \
    --log_interval 1000 \
    --headless \
    --kit_args=--/app/extensions/fsWatcherEnabled=false \
    agent.logger.backend= \
    agent.bilinear.offline_pretrain.enabled=true \
    agent.bilinear.offline_pretrain.num_updates=2 \
    agent.bilinear.offline_pretrain.batch_size=32 \
    agent.offline_dataset.enabled=true \
    agent.offline_dataset.min_ready_transitions=64 \
    agent.offline_dataset.max_cache_transitions=1024 \
    agent.offline_dataset.max_episodes=1 \
    agent.offline_dataset.starvation_timeout_s=120
```

For larger runs, the bilinear config streams the repo list in
`data/unitree/g1_wbt_lerobot_repos.json`. Leave
`agent.offline_dataset.max_episodes_per_repo=0` for full multi-repo training, and
raise `min_ready_transitions`, `max_cache_transitions`, and `max_episodes`
deliberately. Do not add optimizer-loop schema validation to compensate for an
under-validated mapper.

Use the Isaac-free cache probe before expensive training:

```bash
conda run -n "${CONDA_ENV:-SL}" python scripts/validate_lerobot_streaming_cache.py \
    --repo_ids_file data/unitree/g1_wbt_lerobot_repos.json \
    --max_episodes_per_repo 1 \
    --min_ready_transitions 32 \
    --max_cache_transitions 20000 \
    --batch_size 16 \
    --drain
```

## Replay And Preview

Renderer-free dataset preview works without Isaac Sim:

```bash
conda run -n "${CONDA_ENV:-SL}" python scripts/preview_unitree_lerobot_episode.py \
    --repo_id unitreerobotics/G1_WBT_Brainco_Pickup_Pillow \
    --episode_index 0 \
    --max_frames 180 \
    --gif_output logs/unitree_lerobot_preview/g1_wbt_pillow_ep0_180.gif
```

This writes a PNG summary, optional GIF, and NPZ tensors under
`logs/unitree_lerobot_preview/`.

Isaac replay is available for visual reference once RTX rendering is healthy.
Use the desired command field when checking the reference/action label sequence;
switch to `observation.state.robot_q_current` only when debugging measured robot
tracking quality:

```bash
TERM=xterm PYTHONUNBUFFERED=1 \
conda run -n "${CONDA_ENV:-SL}" python scripts/replay_unitree_lerobot_reference.py \
    --headless \
    --device cuda:0 \
    --repo_id unitreerobotics/G1_WBT_Brainco_Pickup_Pillow \
    --episode_index 0 \
    --state_field action.robot_q_desired \
    --root_z_alignment none \
    --max_frames 180 \
    --output_fps 30 \
    --video_output logs/unitree_lerobot_replay/g1_wbt_pillow_ep0_180.mp4
```

The same replay path can bake an Isaac FK reference NPZ for tracking/reward
debugging. Keep `--output_fps 30` for native Unitree WBT control-rate data, or
raise it deliberately when you want interpolation to a faster control rate:

```bash
TERM=xterm PYTHONUNBUFFERED=1 \
conda run -n "${CONDA_ENV:-SL}" python scripts/replay_unitree_lerobot_reference.py \
    --headless \
    --device cuda:0 \
    --repo_id unitreerobotics/G1_WBT_Brainco_Pickup_Pillow \
    --episode_index 0 \
    --state_field action.robot_q_desired \
    --root_z_alignment none \
    --max_frames 180 \
    --output_fps 30 \
    --no_video \
    --npz_output data/unitree/npz/g1_wbt_pillow_ep0_30hz.npz \
    --overwrite_npz
```

For multi-episode conversion, the batch converter also accepts LeRobot jobs:

```json
[
  {
    "source_type": "lerobot",
    "repo_id": "unitreerobotics/G1_WBT_Brainco_Pickup_Pillow",
    "split": "train",
    "episode_index": 0,
    "state_field": "action.robot_q_desired",
    "root_z_alignment": "none",
    "max_frames": 180,
    "input_fps": 30,
    "output_name": "data/unitree/npz/g1_wbt_pillow_ep0_30hz.npz"
  }
]
```

```bash
TERM=xterm PYTHONUNBUFFERED=1 \
conda run -n "${CONDA_ENV:-SL}" python scripts/batch_csv_to_npz.py \
    --headless \
    --device cuda:0 \
    --jobs_json data/unitree/lerobot_jobs.json \
    --output_fps 30
```

The converter writes `qpos`/`qvel`, Isaac body states, and `joint_names` in the
same target/action order used by the G1 env. Build a tracking manifest from
the generated NPZ folder with:

```bash
conda run -n "${CONDA_ENV:-SL}" python scripts/write_lafan1_npz_manifest.py \
    --npz_dir data/unitree/npz \
    --manifest_path data/unitree/manifests/g1_wbt_pillow_30hz.json \
    --dataset_name unitree_lerobot
```

For 30 Hz tracking experiments, the env auto-syncs its control step to an NPZ
manifest when every source has the same FPS. A 30 Hz NPZ manifest selects
240 Hz physics and an 8-step decimation unless timing is overridden explicitly.
The equivalent manual override is:

```bash
env.sim.dt=0.004166666666666667 env.decimation=8
```

## Host Re-Image Note

RTX rendering is currently blocked by the host, not by a polluted conda env.
A clean Python 3.11 conda environment with the official Isaac Sim 5.1 Python
package stack still crashes in the RTX/Hydra startup path. PyTorch CUDA works
in the same env, so the failure is narrower than GPU visibility.

Observed current host:

- Ubuntu 25.10
- NVIDIA driver `595.58.03`
- RTX A4500
- clean Python 3.11 env with Torch `2.7.0+cu128` and Isaac Sim `5.1.0.0`

Re-image target:

- Ubuntu 22.04 or Ubuntu 24.04
- NVIDIA production driver `580.65.06`
- Python 3.11 conda env
- Torch `2.7.0+cu128`
- `isaacsim[all,extscache]==5.1.0`
- editable Isaac Lab plus this workspace's local repos

After re-image, verify in this order:

```bash
conda run -n "${CONDA_ENV:-SL}" python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
conda run -n "${CONDA_ENV:-SL}" python -c "from isaacsim import SimulationApp; app = SimulationApp({'headless': True}); print('ok'); app.close()"
TERM=xterm conda run -n "${CONDA_ENV:-SL}" python scripts/replay_unitree_lerobot_reference.py --headless --device cuda:0 --no_video --max_frames 4
```

Only after those pass should video replay or full Isaac training be treated as
a rendering-stack validation.
