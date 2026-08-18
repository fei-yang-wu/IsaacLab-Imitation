# Experiment Workflow

This page records the practical workflow for local tests, cluster jobs, and
experiment tracking for IPMD-family G1 imitation runs.

The rule is simple: local smoke first, then full cluster job. Do not submit a
cluster job until the local command proves the task, algorithm, manifest, and
submodule pins are wired correctly.

## Finding the current launcher

Start with [`experiments/README.md`](../experiments/README.md). It explicitly
names the current campaign, links the latest dated status, and separates
release-facing entrypoints from historical launchers.

New experiment campaigns use
`experiments/campaigns/YYYY-MM-DD-short-purpose/`. The dated folder owns the
human-readable protocol snapshot and an optional thin wrapper; reusable code
stays in a topical directory. Do not infer that an older top-level script is
current merely because its path is still preserved.

## Local Validation Ladder

Start with the cheapest check that matches the change.
Use the default Pixi environment for lightweight checks and the `isaaclab`
environment for Isaac-backed runtime checks.

### 1. Docs or shell-only changes

```bash
git diff --check
bash -n docker/cluster/run_singularity.sh
bash -n experiments/campaigns/2026-07-22-latent-learning-ablation/run.sh
bash -n experiments/paper/run.sh
```

### 2. Expert-batch or env sampling changes

Pure pytest path:

```bash
pixi run test-rlopt
```

IsaacLab Pixi path, needed when imports require Isaac Sim / Omniverse:

```bash
pixi run -e isaaclab test-isaaclab
```

### 3. Minimal train smoke

Use a small number of envs and one or a few rollout iterations to prove wiring:

```bash
TERM=xterm PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 TORCHDYNAMO_DISABLE=1 \
pixi run -e isaaclab python scripts/rlopt/train.py \
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
for quick local smoke tests. Remove it for normal local runs: RLOpt defaults to
online W&B logging through `agent.logger.backend=wandb`. Use
`WANDB_MODE=offline` only when explicitly running without network sync.

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

## Classified Experiment Scripts

Use [`experiments/SCRIPT_INVENTORY.md`](../experiments/SCRIPT_INVENTORY.md) as the exhaustive live-file index. It classifies every retained shell and Python file as a front door, guarded launcher, workflow, qualification tool, library, audit/report, diagnostic, supporting study, or test.

The current collaborator-facing entrypoints are the dated campaign wrappers in `experiments/campaigns/` and the staging `experiments/paper/run.sh` surface. Reusable implementation stays in topical directories and should normally be reached through one of those front doors.

Completed and superseded paths are classified in [`experiments/PRUNED_SCRIPTS.md`](../experiments/PRUNED_SCRIPTS.md). Recover one from Git history only when designing a new dated protocol; do not restore old launchers merely to use them as an archive.

## Full Cluster Jobs

**2026-08-15: `docker/cluster/cluster_interface.sh` is a retired deprecation
shim.** It no longer submits anything — running it errors with a pointer to
the replacement. Cluster submission now goes through the repo-owned control
plane:

```bash
pixi run python -m imitation_experiments.pipeline.cluster plan \
    --campaign <path/to/campaign.yaml> --arm <arm> --seed <seed>
pixi run python -m imitation_experiments.pipeline.cluster submit \
    --plan <plan_dir> --confirm <PLAN_SHA>
```

Unlike the old entry point, the new CLI has no ad-hoc "just run these flags"
mode: it requires a `campaign.yaml` declaring the arm's stages, resources, and
preflight requirements (`imitation_experiments.pipeline.cluster.config`). See
`experiments/campaigns/2026-08-14-latent-quant-ice-repeats/campaign.yaml` for
a worked 14-arm example, real-ICE validated 2026-08-15 (jobs
5577564/5577565). A campaign not yet migrated to a `campaign.yaml` has no
working cluster submission path until one is written; write it, or run the
job locally, rather than reaching for the retired scripts.

Treat `IPMD_BILINEAR` as a latent-command experiment surface unless the user
explicitly asks for a vanilla debug run. Do not submit bilinear comparison jobs
on `Isaac-Imitation-G1-v0`; the vanilla/non-latent-command path is useful for
debugging only until it is explicitly fixed and revalidated.

The exploratory bilinear pretrain, action-label, and update-count launchers were pruned on 2026-07-23 because they were not part of a current campaign or paper dependency. The generic command above remains useful for bounded development, but a new comparison requires a new dated campaign and qualification record. See `experiments/PRUNED_SCRIPTS.md` for the historical paths.

Cluster jobs append the default full G1 manifest unless the submitted command
already includes `env.lafan1_manifest_path=...`. The default is controlled by
`docker/cluster/.env.cluster`.

For Dance102 or other single-manifest debugging, pass the manifest explicitly:

```bash
env.lafan1_manifest_path=./data/unitree/manifests/g1_unitree_dance102_manifest.json
```

### Georgia Tech PACE / ICE

PACE ICE is the instructional Georgia Tech cluster. The
[official PACE ICE page](https://pace.gatech.edu/ice-cluster/) describes ICE as
an instructional cluster with Phoenix-like hardware/software, and public ICE
[student-facing ICE guides](https://github.com/guru-desh/Intro-To-PACE-ICE)
list SSH access through `login-ice.pace.gatech.edu`. PACE Slurm examples use
explicit account, QOS, CPU, memory, and GPU GRES requests. The repo keeps this
in `docker/cluster/submit_job_slurm_pace.sh` so the lab-local Slurm submitter
does not inherit PACE assumptions.

For a dry run of the high-level skill pipeline on ICE/PACE with video enabled
and a 2B-frame low-level cap:

```bash
DRY_RUN=1 experiments/campaigns/2026-07-23-bones-phase5-language-h200/submit_hl_skill_pipeline_pace_2b.sh
```

For a real submission, set the PACE account first:

```bash
CLUSTER_SLURM_ACCOUNT=<pace-account> \
DRY_RUN=0 experiments/campaigns/2026-07-23-bones-phase5-language-h200/submit_hl_skill_pipeline_pace_2b.sh
```

The helper defaults to:

- `CLUSTER_LOGIN=login-ice.pace.gatech.edu`
- `CLUSTER_SLURM_SUBMIT_SCRIPT=pace`
- `CLUSTER_SLURM_PARTITION=ice-gpu`
- `CLUSTER_SLURM_GPU_GRES=gpu:l40s:1`
- `CLUSTER_SLURM_QOS=coe-ice`
- `CLUSTER_SLURM_CPUS_PER_TASK=24`
- `CLUSTER_SLURM_MEM=32G`
- `CLUSTER_SLURM_TIME_LIMIT=16:00:00`
- `CLUSTER_G1_MANIFEST_REFRESH_POLICY=auto`
- `CLUSTER_GIT_SYNC_FIRST=0`

Override any of those in the environment if `sinfo`, `pace-quota`, or PACE
support guidance says the current ICE allocation needs a different account, QOS,
GPU model, or CPU/GPU ratio. If L40S nodes are only available through backfill
for the active allocation, submit with `CLUSTER_SLURM_QOS=embers`; PACE's
[L40S announcement](https://pace.gatech.edu/2024/10/31/new-gpus-for-phoenix-v100s-being-replaced/)
notes that L40S availability/QOS has changed over time, so verify current ICE
policy before relying on a specific QOS.

Useful live checks on ICE:

```bash
pace-quota
sacctmgr -n -P show assoc where user=$USER format=Account,QOS%60
sinfo -o "%P %G %D %t" | grep -E "ice-gpu|l40s"
```

The pipeline entrypoint does not accept Hydra-style
`env.lafan1_manifest_path=...`, so the helper sets
`CLUSTER_APPEND_DEFAULT_G1_MANIFEST=0` and passes:

```text
--manifest-path /data/lafan1/manifests/g1_lafan1_manifest.json
--dataset-path /data/lafan1/g1_hl_diffsr
```

The container preflight still auto-checks `${CLUSTER_DATA_DIR}/lafan1/npz/g1`
and creates the manifest from the cluster NPZ tree when it is missing or stale.
The helper also skips git-clone-first sync by default on ICE because user
scratch quotas can be tight; incremental `rsync` uses the previous snapshot as
the base and excludes generated caches, logs, worktrees, and `.npz` data files.
The live `ice-gpu` partition reports `MaxTime=16:00:00`; keep the 2B frame cap
as a training cap, but do not assume ICE will grant a two-day walltime.

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

Default RLOpt logging uses online W&B. For local long runs, leave
`WANDB_MODE` unset or set `WANDB_MODE=online`; do not set
`WANDB_MODE=offline` unless the run is intentionally local-only. For cluster
jobs, provide the key on the cluster host and let `run_singularity.sh` inject
it into the container:

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
