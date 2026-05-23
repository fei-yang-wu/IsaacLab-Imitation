# Experiment Workflow

This page records the practical workflow for local tests, cluster jobs, and
experiment tracking for IPMD-family G1 imitation runs.

The rule is simple: local smoke first, then full cluster job. Do not submit a
cluster job until the local command proves the task, algorithm, manifest, and
submodule pins are wired correctly.

## Local Validation Ladder

Start with the cheapest check that matches the change.
Set `CONDA_ENV` to the conda environment you want to use; `SL` is only the
example default:

```bash
CONDA_ENV="${CONDA_ENV:-SL}"
```

### 1. Docs or shell-only changes

```bash
git diff --check
bash -n docker/cluster/cluster_interface.sh
bash -n experiments/ipmd_stability/run_local_debug_ablations.sh
bash -n experiments/ipmd_stability/submit_cluster_ablations.sh
```

### 2. Expert-batch or env sampling changes

Pure pytest path:

```bash
conda run -n "${CONDA_ENV:-SL}" pytest source/isaaclab_imitation/test_reference_patch_env.py
```

Isaac Lab launcher path, needed when imports require Isaac Sim / Omniverse:

```bash
TERM=xterm conda run -n "${CONDA_ENV:-SL}" ./IsaacLab/isaaclab.sh -p -m pytest \
    source/isaaclab_imitation/test_reference_patch_env.py
```

### 3. Minimal train smoke

Use a small number of envs and one or a few rollout iterations to prove wiring:

```bash
TERM=xterm PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 TORCHDYNAMO_DISABLE=1 \
conda run -n "${CONDA_ENV:-SL}" python scripts/rlopt/train.py \
    --task Isaac-Imitation-G1-v0 \
    --num_envs 16 \
    --headless \
    --algo IPMD \
    --max_iterations 2 \
    --log_interval 1000 \
    --kit_args=--/app/extensions/fsWatcherEnabled=false \
    env.lafan1_manifest_path=./data/unitree/manifests/g1_unitree_dance102_manifest.json \
    env.dataset_path=/tmp/iltools_g1_lafan1_tracking_g1_unitree_dance102_manifest_6d26546fd54a \
    env.refresh_zarr_dataset=False \
    agent.logger.backend= \
    agent.logger.exp_name=ipmd_local_smoke
```

The empty `agent.logger.backend=` override disables the external metrics backend
for quick local smoke tests. Remove it when you want W&B tracking.

For latent IPMD, switch the task:

```bash
--task Isaac-Imitation-G1-Latent-v0
```

For bilinear IPMD, switch the algorithm and enable offline pretrain if that is
the surface under test:

```bash
--algo IPMD_BILINEAR \
agent.bilinear.offline_pretrain.enabled=true \
agent.bilinear.offline_pretrain.num_updates=10
```

For LeRobot-backed offline pretraining, keep the task latent and enable the
offline dataset cache explicitly:

```bash
--task Isaac-Imitation-G1-Latent-v0 \
--algo IPMD_BILINEAR \
agent.bilinear.offline_pretrain.enabled=true \
agent.offline_dataset.enabled=true
```

The default first dataset for the G1 bilinear config is
`unitreerobotics/G1_WBT_Brainco_Pickup_Pillow`. The full command surface and
re-image notes live in [LeRobot Offline Pretraining](lerobot-offline-pretraining.md).

The `--kit_args=--/app/extensions/fsWatcherEnabled=false` override is useful on
local machines where Isaac Kit file watcher startup fails under resource
pressure.

## Existing Experiment Scripts

`experiments/ipmd_stability/run_local_debug_ablations.sh` runs local IPMD reward
stability sweeps. Useful knobs:

```bash
TASK=Isaac-Imitation-G1-v0 \
NUM_ENVS=128 \
TIMEOUT_SECONDS=300 \
SEEDS=2024 \
COMBOS="A B C" \
experiments/ipmd_stability/run_local_debug_ablations.sh
```

`experiments/ipmd_stability/submit_cluster_ablations.sh` submits cluster sweeps
for IPMD and baseline GAIL/AMP/ASE variants:

```bash
DRY_RUN=1 \
TASK=Isaac-Imitation-G1-v0 \
NUM_ENVS=2048 \
ALGO=ipmd \
SEEDS="2024" \
COMBOS="A B" \
experiments/ipmd_stability/submit_cluster_ablations.sh
```

Use `DRY_RUN=1` first to inspect the generated commands.

`experiments/vqvae_temporal_ablation.sh` is the current VQ-VAE/FSQ temporal
ablation helper. It supports `local`, `cluster`, and `print` modes:

```bash
experiments/vqvae_temporal_ablation.sh print all
experiments/vqvae_temporal_ablation.sh local vqvae_p10_d64
experiments/vqvae_temporal_ablation.sh cluster vqvae_p10_d64
```

## Full Cluster Jobs

Cluster submission entrypoint:

```bash
./docker/cluster/cluster_interface.sh job [profile] [job args...]
```

The default profile is `base`. A typical full latent IPMD job:

```bash
./docker/cluster/cluster_interface.sh job \
    --task Isaac-Imitation-G1-Latent-v0 \
    --num_envs 4096 \
    --headless \
    --video \
    --algo IPMD \
    --kit_args=--/app/extensions/fsWatcherEnabled=false \
    agent.logger.exp_name=ipmd_latent_full_4096
```

A bilinear job with offline pretraining enabled:

```bash
./docker/cluster/cluster_interface.sh job \
    --task Isaac-Imitation-G1-Latent-v0 \
    --num_envs 4096 \
    --headless \
    --video \
    --algo IPMD_BILINEAR \
    --kit_args=--/app/extensions/fsWatcherEnabled=false \
    agent.logger.project_name=G1-Imitation-RLOpt-Pretrain \
    agent.ipmd.use_latent_command=true \
    agent.bilinear.policy_include_raw_state=false \
    agent.bilinear.offline_pretrain.enabled=true \
    agent.bilinear.offline_pretrain.num_updates=2000 \
    agent.logger.exp_name=ipmd_bilinear_offline_full_4096
```

Treat `IPMD_BILINEAR` as a latent-command experiment surface unless the user
explicitly asks for a vanilla debug run. Do not submit bilinear comparison jobs
on `Isaac-Imitation-G1-v0`; the vanilla/non-latent-command path is useful for
debugging only until it is explicitly fixed and revalidated.

For the current pretrain/scratch/frozen comparison, use:

```bash
experiments/bilinear_pretrain/submit_cluster_ablation.sh
```

By default this submits one seed on `Isaac-Imitation-G1-Latent-v0` for five
feature-only variants at `num_envs=4096`, `max_iterations=10173`, and logs them
to W&B project `G1-Imitation-RLOpt-Pretrain`. This is a 1B-frame budget because
each rollout iteration collects `4096 * 24` frames. The script sets
`agent.bilinear.policy_include_raw_state=false` so the policy sees `F(s)z`, not
`concat(F(s)z, s)`. The default seed is `42`. The default variants are `scratch`,
`pretrained_finetune`, `pretrained_frozen`, `random_frozen`, and
`pretrained_bc_finetune`. The BC variant runs offline policy BC after SR
pretraining using reconstructed expert actions. Set `DRY_RUN=1` to print the
commands, or set `SEEDS="2024 2025 2026"` for a three-seed sweep.

For the Dance102 action-label ablation, use the labeled manifest and script:

```bash
DRY_RUN=1 experiments/bilinear_pretrain/submit_dance102_action_label_ablation.sh
experiments/bilinear_pretrain/submit_dance102_action_label_ablation.sh
```

The default variants are `scratch`, `pretrained_finetune`,
`pretrained_bc_finetune`, `pretrained_labeled_bc_finetune`, and
`labeled_bc_finetune`. The labeled BC variants set
`env.reconstructed_reference_action=false`, so expert actions are read from the
locally generated action-label NPZ described by
`data/unitree/manifests/g1_unitree_dance102_rlopt_ipmd_500m_actions_manifest.json`.
The NPZ itself is intentionally not tracked.

To test whether the offline stage is long enough, run the pretrained+finetune
update-count sweep:

```bash
DRY_RUN=1 experiments/bilinear_pretrain/submit_pretrain_update_sweep.sh
experiments/bilinear_pretrain/submit_pretrain_update_sweep.sh
```

The default update counts are `500 1000 2000 4000`. Override them with:

```bash
UPDATE_COUNTS="1000 2000 4000 8000" \
experiments/bilinear_pretrain/submit_pretrain_update_sweep.sh
```

Summarize the offline SR traces in W&B with:

```bash
python experiments/bilinear_pretrain/summarize_pretrain_wandb.py \
    --group g1_bilinear_sr_pretrain_feature_only_4096
```

Cluster jobs append the default full G1 manifest unless the submitted command
already includes `env.lafan1_manifest_path=...`. The default is controlled by
`docker/cluster/.env.cluster`.

For Dance102 or other single-manifest debugging, pass the manifest explicitly:

```bash
env.lafan1_manifest_path=./data/unitree/manifests/g1_unitree_dance102_manifest.json
```

## RLOpt Submodule State

Cluster jobs should use the pinned `RLOpt/` submodule state from
`IsaacLab-Imitation` by default. If a task explicitly needs an unpinned local
experiment outside this repo, enable an overlay path in
`docker/cluster/.env.cluster`:

```bash
CLUSTER_RLOPT_LOCAL_PATH=/absolute/path/to/RLOpt
```

Leave that line commented out for submodule-first runs.

Every job writes a repo manifest to:

```text
<CLUSTER_ISAACLAB_DIR>/repo_sync_manifest.tsv
```

Use it to confirm the exact branch/SHA/dirty-state for `IsaacLab-Imitation` and
any overlaid repos.

## Tracking Experiments

Every `scripts/rlopt/train.py` run writes local metadata under:

```text
logs/rlopt/<algo>/<task>/<timestamp>/
```

Important files:

- `command.txt`: exact command used for the run.
- `params/env.yaml`: resolved environment config.
- `params/agent.yaml`: resolved RLOpt config.
- `rlopt.log`: durable training summaries from RLOpt.
- `videos/train/`: local rollout videos when `--video` is enabled.
- `models/`: checkpoints, when the agent saves them.

Use explicit experiment names:

```bash
agent.logger.project_name=<separate_wandb_project>
agent.logger.exp_name=<short_descriptive_name>
agent.logger.group_name=<optional_group_name>
```

Default RLOpt logging uses W&B. For cluster jobs, provide the key on the cluster
host and let `run_singularity.sh` inject it into the container:

```bash
printf '%s\n' 'your_wandb_api_key' > ~/.wandb_api_key
chmod 600 ~/.wandb_api_key
```

Then set:

```bash
CLUSTER_WANDB_API_KEY_FILE=.wandb_api_key
```

Do not rely only on W&B. For debugging, inspect `rlopt.log`, `command.txt`, and
the YAML configs first.

Useful IPMD metrics to scan in `rlopt.log`:

- `episode/return` and `episode/length`
- `r_step`
- `reward_diff`
- `exp_r`
- `env_r`
- `reward_l2`
- `reward_gp`
- `v_loss`
- `entropy`
- `grad_norm`
- `lr`
- `clip`

Interpretation rule: separate standing/stability from imitation quality. A run
can improve episode length or standing while still failing to imitate the
reference motion.

## Local To Cluster Promotion

Use this promotion checklist:

1. Run the smallest local test that exercises the changed path.
2. Inspect `logs/rlopt/.../command.txt`, `params/agent.yaml`, and `rlopt.log`.
3. Use a distinct `agent.logger.exp_name`.
4. Confirm the intended `RLOpt/` and `ImitationLearningTools/` submodule SHAs.
5. Run `DRY_RUN=1` for experiment scripts that support it.
6. Submit the cluster job.
7. Record the job id, repo manifest path, experiment name, task, algo, seed,
   manifest, and important overrides.

## Common Failure Modes

- Hydra receives `task_name=None`: inspect the generated cluster job command;
  scheduler wrappers must preserve one shell word per argument.
- Cluster job uses stale algorithm code: check `repo_sync_manifest.tsv` and
  whether `CLUSTER_RLOPT_LOCAL_PATH` was enabled.
- Dance102 smoke loads the full LAFAN cache: pass both
  `env.lafan1_manifest_path=...` and a matching explicit `env.dataset_path=...`.
- W&B panels look duplicated: inspect the actual local history/logs before
  changing logger code.
- Runtime succeeds but imitation quality is poor: inspect `reward_diff`,
  `exp_r`, videos, and reference comparison; do not treat standing alone as
  success.
