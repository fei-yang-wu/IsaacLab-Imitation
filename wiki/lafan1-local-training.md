# LAFAN1 Local Training Pipeline (Reproducible)

This is the reference recipe for training a G1 LAFAN1 imitation policy locally with the
**pretrain → low-level** pipeline: first learn a DiffSR skill encoder from expert motion,
then train the low-level "oracle" IPMD policy conditioned on that encoder. Every parameter
in this doc is the validated default baked into `scripts/rlopt/run_local_pretrain_lowlevel.sh`
and the `G1ImitationLatentRLOptIPMDConfig` agent config, so the numbers below reproduce.

All commands are run from the repository root inside the `isaaclab` Pixi environment.

## TL;DR

```bash
# One-shot: builds the zarr cache, pretrains the skill encoder, then trains the
# low-level oracle IPMD policy to 2B frames with video + wandb logging.
bash scripts/rlopt/run_local_pretrain_lowlevel.sh
```

That single script chains both stages and wires the fresh skill checkpoint into the
low-level run automatically. The rest of this doc explains each stage and how to reproduce
them by hand.

## What "oracle low-level" means

The low-level policy is conditioned on a **skill latent** `z` produced by the pretrained
DiffSR encoder from the *expert* future window (the "oracle"). It is not yet driven by a
learned planner — that is a later stage (`train_skill_commander.py`). This pipeline validates
that a good skill encoder + IPMD low-level policy can track LAFAN1 motions closely.

## Prerequisites

- The full G1 LAFAN1 data prepared locally (40 retargeted trajectories):
  - `data/lafan1/manifests/g1_lafan1_manifest.json`
  - `data/lafan1/npz/g1/*.npz` (referenced by the manifest)
  - See the README "Data preparation" section if you don't have these.
- The `isaaclab` Pixi environment installed (`pixi install -e isaaclab`).
- A CUDA GPU with room for 4096 envs (≈40 GB is comfortable).

## Joint-order correctness

The loader canonicalizes every reference to the **robot articulation joint order**
(`G1_29DOF_ISAACLAB_JOINT_NAMES`, an interleaved L/R breadth-first layout — *not* the SDK
serial "all-left-leg → all-right-leg → waist" order) at zarr-build time. This matters because
the reference is written directly onto `robot.data.joint_pos` via `write_joint_state_to_sim`
and indexed by the reward with the robot's own `joint_ids`. You can verify a built zarr:

```python
import zarr
root = zarr.open("data/lafan1/g1_hl_diffsr/lafan1", mode="r")
print(list(root.attrs["joint_names"])[:6])
# -> ['left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint',
#     'left_hip_roll_joint', 'right_hip_roll_joint', 'waist_roll_joint']  # articulation order
```

## Stage 1 — Skill-encoder pretrain (DiffSR)

Default architecture: macro horizon `W=25`, skill latent `z_dim=256`, DiffSR feature/embed
`128/512`, deterministic latent bottleneck, `intermediate` encoder window (target frame
`s_{t+W}` hidden). Trains for 5000 updates (~3–4 min).

```bash
pixi run -e isaaclab python scripts/rlopt/train_hl_skill_diffsr.py \
    --headless --device cuda:0 \
    --task Isaac-Imitation-G1-Latent-v0 \
    --num_envs 4096 --seed 0 \
    --output_dir logs/hl_skill_diffsr/lafan1_w25_z256 \
    --horizon_steps 25 --encoder_window_mode intermediate \
    --z_dim 256 --diffsr_feature_dim 128 --diffsr_embed_dim 512 \
    --batch_size 8192 --num_updates 5000 --log_interval 100 \
    --eval_batches 4 --eval_batch_size 8192 \
    --train_split all --eval_split all --eval_trajectory_fraction 0.5 \
    --trajectory_split_seed 0 \
    --reconstruction_eval --window_probe_eval \
    --window_probe_train_batches 8 --window_probe_eval_batches 4 \
    env.lafan1_manifest_path=$PWD/data/lafan1/manifests/g1_lafan1_manifest.json \
    env.dataset_path=$PWD/data/lafan1/g1_hl_diffsr \
    env.refresh_zarr_dataset=true
```

Output: `.../checkpoints/best.pt` (lowest held-out loss). `refresh_zarr_dataset=true` (re)builds
`data/lafan1/g1_hl_diffsr` once; later stages set it to `false`.

**Expert terms the encoder sees / reconstructs** (`LATENT_POSTERIOR_INPUT_KEYS`):
`expert_motion` (58-d = reference `joint_pos` 29 + `joint_vel` 29), `expert_anchor_pos_b`
(3-d reference torso position in the robot's current torso frame) and `expert_anchor_ori_b`
(6-d rot6d torso orientation in that frame). Root velocities are not included.

**Healthy pretrain signal** (the latent must be used, not ignored):

| metric | expected |
|---|---|
| `train/loss_real_z_eval` | ≈ 4 (drops from ~11) |
| `train/loss_shuffled_z_eval` | ≈ 100+ |
| `train/loss_zero_z_eval` | ≈ 90+ |

A large gap between `real_z` and `shuffled/zero_z` means the decoder genuinely depends on the
skill code. If they are close, the encoder collapsed — do not proceed to Stage 2.

## Stage 2 — Low-level oracle IPMD

> **2026-07-19:** `Isaac-Imitation-G1-Latent-v0` now resolves to the SONIC
> surface (pelvis anchor, strict adaptive terminations, full-trajectory
> adaptive resets, SONIC actuators/rewards, 10-step histories). The commands
> below still run, but the expected-curve table further down was measured on
> the pre-migration surface, which is now
> `Isaac-Imitation-G1-Latent-Legacy-v0`. See
> `wiki/isaaclab3-cu130-runtime-migration.md`, "Training-gate resolution", for
> the SONIC-surface expectations (ep_len ~4 -> ~15 over a 50M-frame block).

The low-level agent is **IPMD** on `Isaac-Imitation-G1-Latent-v0`. All hl_skill/latent
hyperparameters are baked config defaults in `G1ImitationLatentRLOptIPMDConfig`
(`_default_use_latent_command=True`): `command_source="hl_skill"`, `latent_dim=258`
(256 skill `z` + 2 sin/cos phase), `hl_skill_horizon_steps=25`, `code_period=25`,
`command_phase_mode="sin_cos"`, learned-reward coeffs disabled, online finetune off. The
**only** per-run override is the skill checkpoint path.

```bash
pixi run -e isaaclab python scripts/rlopt/train.py \
    --headless --video --video_length 500 --video_interval 2500 \
    --device cuda:0 --num_envs 4096 \
    --task Isaac-Imitation-G1-Latent-v0 --algo IPMD --seed 0 \
    agent.collector.total_frames=2000000000 \
    agent.logger.backend=wandb agent.logger.project_name=g1-lafan1-hl-skill-2b \
    agent.logger.video=true \
    agent.ipmd.hl_skill_checkpoint_path=$PWD/logs/hl_skill_diffsr/lafan1_w25_z256/checkpoints/best.pt \
    env.lafan1_manifest_path=$PWD/data/lafan1/manifests/g1_lafan1_manifest.json \
    env.dataset_path=$PWD/data/lafan1/g1_hl_diffsr \
    env.refresh_zarr_dataset=false
```

**Expected learning curve** (4096 envs, seed 0; roughly reproducible, RL variance applies):

| frames | `r_ep` | `ep_len` |
|---|---|---|
| ~18M   | ~0.9  | ~49  |
| ~150M  | ~18   | ~390 |
| ~300M  | ~19   | ~396 |
| 2B     | converged | near episode cap |

`r_ep` climbs from <1 to ~18 within the first ~150M frames and then refines. Full 2B run is
~11 h on a single modern GPU.

## Overriding defaults

`run_local_pretrain_lowlevel.sh` exposes every parameter as an env-var default. Examples:

```bash
# Shorter smoke run, no wandb:
TOTAL_FRAMES=20000000 LOGGER_BACKEND=none bash scripts/rlopt/run_local_pretrain_lowlevel.sh

# Reuse an already-trained skill encoder, just (re)run the low-level policy:
SKIP_PRETRAIN=1 SKILL_CKPT=/abs/path/to/best.pt bash scripts/rlopt/run_local_pretrain_lowlevel.sh

# Different seed / device:
SEED=1 DEVICE=cuda:1 bash scripts/rlopt/run_local_pretrain_lowlevel.sh
```

Key knobs: `HORIZON_STEPS`, `Z_DIM`, `ENCODER_WINDOW_MODE`, `SKILL_UPDATES`, `LOW_LEVEL_ALGO`,
`TOTAL_FRAMES`, `NUM_ENVS`, `LOGGER_BACKEND`, `LOGGER_PROJECT_NAME`, `MANIFEST_PATH`,
`DATASET_PATH`. If you change `HORIZON_STEPS` or `Z_DIM`, the low-level `latent_dim` and phase
period must match — either override the corresponding `agent.ipmd.*` fields or update the
config defaults, because the encoder and low-level policy must agree on the latent layout.

## Outputs

```
logs/local_pretrain_lowlevel/<timestamp>_lafan1_h25_z256_pretrain_lowlevel/
  skill_encoder_h25_z256/checkpoints/best.pt   # stage 1 skill encoder
data/lafan1/g1_hl_diffsr/                        # shared zarr cache (canonical joint order)
logs/rlopt/ipmd/Isaac-Imitation-G1-Latent-v0/<run>/  # stage 2 low-level checkpoints + videos
```

## Troubleshooting

- **`Latent observation ('policy', 'latent_command') has shape (258,), expected (64,)`** —
  the low-level `latent_dim` isn't 258. Use `--algo IPMD` on `Isaac-Imitation-G1-Latent-v0`
  (which selects `G1ImitationLatentRLOptIPMDConfig`); don't override `agent.ipmd.latent_dim`
  back to the posterior default of 64.
- **Garbage / jittery replay, low reward plateau** — usually a joint-order mismatch. Delete
  the zarr and rebuild with `env.refresh_zarr_dataset=true`, and confirm the zarr
  `joint_names` are in articulation order (see "Joint-order correctness").
- **`No space left on device` during video/checkpoint writes** — the 2B run writes many
  videos + checkpoints; free space under `logs/` or lower `agent.save_interval`.
