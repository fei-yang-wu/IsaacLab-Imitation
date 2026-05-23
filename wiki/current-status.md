# Current Status

Last refreshed: 2026-05-12.

This page is intentionally dated. Update it when a branch lands or when the
active experiment direction changes.

## Branch Snapshot

- Current branch: `codex-submodule-lerobot-offline-pretrain`.
- The in-repo `./RLOpt` submodule is pinned to the offline dataset cache
  sampler feature commit.
- The in-repo `./ImitationLearningTools` submodule is pinned to the Unitree
  LeRobot streaming mapper feature commit.

## High-Value Repo Context

- `IsaacLab-Imitation` is the orchestration repo for Isaac Lab G1 imitation
  environments, task registration, RLOpt entrypoints, cluster scripts, data
  manifests, and experiment scripts.
- The active research focus is representation learning for IPMD-family inverse
  RL: learn useful state representations and reward structure from expert state
  trajectories, then adapt online with environment interaction.
- `./RLOpt` is the active edit target for algorithm/runtime work and the pinned
  dependency snapshot used by this repo.
- `scripts/rlopt/train.py` resolves `--algo` to task-specific
  `rlopt_<algo>_cfg_entry_point` registry entries.
- `ImitationRLEnv.sample_expert_batch(...)` is the env-owned expert sampling
  surface used by imitation algorithms.
- `ImitationRLEnv.get_offline_dataset_mapper_params()` is the env-owned surface
  that exports G1 action offsets and action scale for external offline dataset
  mappers.

## Current Task Surfaces

- `Isaac-Imitation-G1-v0`: vanilla G1 tracking task.
- `Isaac-Imitation-G1-Latent-v0`: latent-conditioned G1 task.
- `Isaac-Imitation-G1-Latent-VQVAE-v0`: latent VQ-VAE G1 task.
- `IPMD_BILINEAR`: routed through the bilinear config entrypoint.
  Current working assumption: run bilinear experiments on the latent-command
  task surface for now. The non-latent-command bilinear path is not a trusted
  surface until it is fixed and revalidated.

## 2026-05-11 LeRobot Streaming Offline Dataset

Current approach: use LeRobot as the remote storage and streaming format, then
convert Unitree WBT episodes into canonical TensorDict transitions before they
enter a local TorchRL replay cache. Training samples from the local cache only.

First dataset:

```text
unitreerobotics/G1_WBT_Brainco_Pickup_Pillow
```

Second target after the mapper is validated:

```text
unitreerobotics/G1_WBT_Brainco_Collect_Plates_Into_Dishwasher
```

Implementation split:

- `./ImitationLearningTools` adds `UnitreeG1WBT29DofMapper` and
  `StreamingTensorDictReplayCache`.
- `./RLOpt` adds `OfflineDatasetConfig`, background LeRobot-to-TorchRL cache
  construction, and offline sampler selection for IPMD/bilinear pretraining.
- This repo adds env-owned action mapper constants, the latent G1 bilinear default
  Unitree WBT dataset config, and replay/preview scripts.

The Unitree WBT dataset does not expose qvel, so the first mapper finite
differences velocities per episode. That is acceptable for the first low
dimensional pretrain path, but it should be treated as a mapper choice rather
than a claim about measured robot velocity.

Validation policy:

- Validate schema, widths, action scale, and env action-offset compatibility at
  mapper/cache/agent construction.
- Do not add per-iteration schema guards in the optimizer loop.

Debugging tools:

- `scripts/preview_unitree_lerobot_episode.py` pulls a bounded streaming subset
  and writes PNG/GIF/NPZ previews without Isaac Sim.
- `scripts/replay_unitree_lerobot_reference.py` replays
  `observation.state.robot_q_current` on the Isaac G1 model once RTX rendering
  works.

RTX rendering status:

- Clean Python 3.11 conda env with Torch `2.7.0+cu128` and Isaac Sim
  `5.1.0.0` still crashes in the RTX/Hydra startup path; the local validation
  env happened to be named `SL`.
- The current host is Ubuntu 25.10 with NVIDIA driver `595.58.03`.
- PyTorch CUDA sees the RTX A4500, so this is not basic CUDA visibility.
- Re-image target is Ubuntu 22.04 or 24.04 with NVIDIA driver `580.65.06`.

See [LeRobot Offline Pretraining](lerobot-offline-pretraining.md) for the
current command surface and re-image verification checklist.

## 2026-05-07 Offline Bilinear Pretrain Recovery

Goal: recover the bilinear SR warm-start path before broader offline IRL/GAIL
work. This phase uses env-owned expert batches, reconstructed reference actions,
and transition-aligned next policy observations.

What changed locally:

- `scripts/rlopt/train.py` now logs unexpected exceptions with traceback and
  re-raises so local and cluster failures do not disappear behind a printed
  one-line error.
- `./RLOpt` now preflights offline bilinear expert batches at
  construction when `bilinear.offline_pretrain.enabled=True`.
- Vanilla `IPMD_BILINEAR` no longer requires latent rollout tensors for bilinear
  SR metric logging; latent `z_*` metrics are only recorded for latent-command
  runs.
- G1 bilinear IPMD logs to W&B project `G1-Imitation-RLOpt-Pretrain` by
  default, keeping the pretrain ablation separate from older `RLOpt` runs.
- Offline bilinear pretrain now emits start/progress/complete log lines with
  update count, percent complete, elapsed time, ETA, update rate, and SR history
  size.
- G1 bilinear IPMD now disables the raw-state bypass for the pretrain ablation
  with `bilinear.policy_include_raw_state=False`; the policy MLP receives
  `F(s)z` only.
- `experiments/bilinear_pretrain/submit_cluster_ablation.sh` submits the
  five-way 4096-env comparison: scratch, pretrained+finetune,
  pretrained+frozen, random+frozen, and pretrained+offline-policy-BC+finetune.
- `experiments/bilinear_pretrain/submit_pretrain_update_sweep.sh` submits a
  pretrained+finetune update-count sweep. The default sweep is
  `OFFLINE_NUM_UPDATES="500 1000 2000 4000"`.
- `experiments/bilinear_pretrain/summarize_pretrain_wandb.py` summarizes
  `offline/sr/loss/dynamics_loss`, reconstruction metrics, and SR history size
  from W&B.

Verified locally:

- RLOpt focused test: `pytest tests/test_ipmd_components.py -k bilinear -q`
  passed with 8 tests.
- Isaac expert sampler test: `pytest source/isaaclab_imitation/test_reference_patch_env.py -q`
  passed with 13 tests.
- Bug smoke passed at `num_envs=128`, `max_iterations=1`,
  `offline_pretrain.num_updates=2`, `offline_pretrain.batch_size=512`.
- Functional smoke passed at `num_envs=128`, `max_iterations=2`,
  `offline_pretrain.num_updates=20`, `sample_eval_interval=10`.
- Modest performance smoke passed at `num_envs=1024`, `max_iterations=1`,
  `offline_pretrain.num_updates=2`, `offline_pretrain.batch_size=1024`, with
  about 7.1K FPS in the first rollout iteration.

Next scale ladder:

- Use `num_envs=128` for bug-finding smoke runs.
- Use `num_envs=1024-2048` for local performance/debug runs.
- Submit `num_envs=4096` on cluster only after local 128 and 1024/2048 pass.
- For cluster runs, capture GPU memory headroom before deciding whether to push
  beyond 4096 envs.

The first 4096-env cluster ablation batch was cancelled because the policy still
had a raw-state bypass:

- `scratch`: SLURM job `3108283`, cancelled.
- `pretrained_finetune`: SLURM job `3108290`, cancelled.
- `pretrained_frozen`: SLURM job `3108297`, cancelled.
- `random_frozen`: SLURM job `3108305`, cancelled.

Replacement feature-only 4096-env cluster ablation submitted on 2026-05-07 under
W&B project `G1-Imitation-RLOpt-Pretrain` and group
`g1_bilinear_sr_pretrain_feature_only_4096`:

- `scratch`: SLURM job `3108533`.
- `pretrained_finetune`: SLURM job `3108535`.
- `pretrained_frozen`: SLURM job `3108545`.
- `random_frozen`: SLURM job `3108563`.

The feature-only 4096-env jobs completed in Slurm. W&B marked the runs as
`crashed`, so `scripts/rlopt/train.py` now explicitly calls
`wandb.finish(exit_code=0)` on the success path before Isaac shutdown.

Observed 2000-update offline SR pretrain trace from W&B:

- Dynamics loss fell from about `12.1` at update 100 to `2.68` at update 2000.
- Reconstruction MSE fell from about `44.0` at update 100 to `4.75` at update
  2000.
- Reconstruction L1 fell from about `28.8` at update 100 to `8.63` at update
  2000.
- The SR history buffer reached its `10M` transition cap around update 1300.

Interpretation: `2000` updates is sufficient for the first ablation, but the
offline loss is not obviously saturated. The next targeted experiment is now the
100M-frame feature-only ablation. The earlier 20M-frame run was too short to
evaluate meaningful online sample efficiency.

Pretrained+finetune update-count sweep submitted on 2026-05-07 under W&B group
`g1_bilinear_sr_pretrain_update_sweep_feature_only_4096`:

- `500` offline updates: SLURM job `3108917`, cancelled.
- `1000` offline updates: SLURM job `3108920`, cancelled.
- `2000` offline updates: SLURM job `3108924`, cancelled.
- `4000` offline updates: SLURM job `3108934`, cancelled.

Replacement feature-only 4096-env, 100M-frame ablation submitted on 2026-05-07
under W&B group `g1_bilinear_sr_pretrain_feature_only_4096_100m`. The budget is
`max_iterations=1024`, which is `1024 * 4096 * 24 = 100,663,296` online frames:

- `scratch`: SLURM job `3108983`.
- `pretrained_finetune`: SLURM job `3108987`.
- `pretrained_frozen`: SLURM job `3108995`.
- `random_frozen`: SLURM job `3109015`.
- `pretrained_bc_finetune`: SLURM job `3109020`.

The `pretrained_bc_finetune` variant runs `2000` offline SR updates followed by
`2000` offline policy-BC updates against reconstructed expert actions. The BC
phase runs after the SR encoder is copied into the EMA encoder used by the
feature-only policy path.

Dance102 latent-command feature-only ablation submitted on 2026-05-07 after
establishing that the current bilinear path should be treated as latent-command
only. All runs use `Isaac-Imitation-G1-Latent-v0`,
`ipmd.use_latent_command=True`, `bilinear.policy_include_raw_state=False`,
the explicit Dance102 manifest, `num_envs=4096`, and `max_iterations=1024`:

- `scratch` / no offline pretrain: SLURM job `3110712`.
- `pretrained_finetune`: SLURM job `3110776`.
- `pretrained_frozen`: SLURM job `3110782`.
- `pretrained_bc_finetune`: SLURM job `3110795`.

Follow-up 500M-frame Dance102 latent-command comparison submitted on 2026-05-08
with the same setup. The budget is `max_iterations=5087`, which is
`5087 * 4096 * 24 = 500,072,448` online frames:

- `scratch` / no offline pretrain: SLURM job `3114583`.
- `pretrained_finetune`: SLURM job `3114586`.

## Context-Management Status

- `AGENTS.md` is the durable coding-agent rule file.
- `CLAUDE.md` is the Claude Code command and architecture shortcut file.
- `wiki/context-management.md` is the durable context policy.
- `wiki/ipmd-representation-learning.md` records the current algorithmic focus.
- `wiki/experiment-workflow.md` records the local/cluster/tracking workflow.
- This file holds dated status and should be refreshed or pruned as work
  changes.
