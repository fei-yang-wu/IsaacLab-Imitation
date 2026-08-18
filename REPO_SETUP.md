# Repository Setup (Git Submodules)

This repo now tracks its dependent repos as submodules:

- `RLOpt/`
- `ImitationLearningTools/`

`IsaacLab-Imitation` itself remains the top-level repo. Isaac Lab itself is
not a submodule here: `pixi.toml` pins it as a regular PyPI dependency
(`isaaclab==3.0.0b2.post1` from NVIDIA's index) in the `isaaclab` Pixi
environment.

Optional local checkouts:

- `loco-mujoco/` only when explicitly using the `loco_mujoco` dataset loader

## 1. Clone with submodules

```bash
git clone --recurse-submodules git@github.com:GTLIDAR/IsaacLab-Imitation.git
cd IsaacLab-Imitation
```

If you already cloned before submodules were added:

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

## 2. Verify remotes (from current git config)

Run:

```bash
git remote -v
git -C RLOpt remote -v
git -C ImitationLearningTools remote -v
```

Expected default remotes:

- `IsaacLab-Imitation`: `origin -> git@github.com:GTLIDAR/IsaacLab-Imitation.git`
- `RLOpt`: `origin -> git@github.com:fei-yang-wu/RLOpt.git`
- `ImitationLearningTools`: `origin -> git@github.com:GTLIDAR/ImitationLearningTools.git`
- optional `loco-mujoco`: configured by your local checkout when using that loader

## 2b. Optional loco-mujoco loader

The G1 Isaac training path no longer depends on `unitree_rl_lab`; robot
configuration and URDF/mesh assets are packaged in this repo. If you want to
use the optional Loco-MuJoCo loader, keep a local checkout or install the
package in your Python environment:

```bash
cd ..
git clone https://github.com/robfiras/loco-mujoco.git
```

Optional extra remotes used in this workspace:

```bash
git -C RLOpt remote add gatech https://github.gatech.edu/GeorgiaTechLIDARGroup/RLOpt.git
```

## 3. Update submodules later

```bash
git submodule update --init --recursive
```

To move submodules to newer commits from their configured tracking branches:

```bash
git submodule update --remote --recursive
git add RLOpt ImitationLearningTools
git commit -m "Update submodule pins"
```

For feature work, it is also valid to check out an exact commit inside a
submodule and then commit the updated top-level gitlink. Keep `.gitmodules`
tracking branches unchanged unless the long-lived default branch changes.

## 4. Cluster note (no conda/venv needed for submission)

For cluster submission, you do not need a local conda/venv for IsaacLab Python packages.

- Job execution uses `/isaac-sim/python.sh` inside the container/Apptainer image.
- Local requirements for submission are mainly Apptainer and SSH access to the cluster.

**2026-08-15: `docker/cluster/cluster_interface.sh` and every
`submit_job_slurm_*.sh`/`submit_job_pbs.sh` are retired deprecation shims** —
running any of them errors with a pointer to the replacement instead of
submitting. Cluster submission goes through the repo-owned control plane:

```bash
pixi run python -m imitation_experiments.pipeline.cluster plan \
    --campaign <path/to/campaign.yaml> --arm <arm> --seed <seed> [--profile ice|skynet]
pixi run python -m imitation_experiments.pipeline.cluster submit \
    --plan <plan_dir> --confirm <PLAN_SHA>
```

A campaign declares its arms, stages, and resources in one `campaign.yaml`
consumed by `imitation_experiments.pipeline.cluster.config`; see
`experiments/campaigns/2026-08-14-latent-quant-ice-repeats/` for a worked
14-arm example (real-ICE validated 2026-08-15, jobs 5577564/5577565). Cluster
profiles (ssh alias, data/SIF/log paths, Slurm defaults, frozen env vars) live
in `source/imitation_experiments/imitation_experiments/pipeline/cluster/conf/profile_<name>.yaml`;
the `ice` profile is validated, `skynet` is experimental.

A campaign not yet migrated to a `campaign.yaml` has no working submission
path until one is written.

The helper defaults to `ice-gpu`, `gpu:l40s:1`, `coe-ice`, and 32G RAM; override
those with `CLUSTER_SLURM_PARTITION`, `CLUSTER_SLURM_GPU_GRES`,
`CLUSTER_SLURM_QOS`, or `CLUSTER_SLURM_MEM` if the active allocation differs.

By default, cluster jobs use the submodule states pinned by this top-level repo
(`RLOpt` and `ImitationLearningTools`) plus the pip-pinned `isaaclab` package.
Only set path overrides in `docker/cluster/.env.cluster` when a task explicitly
needs an unpinned local checkout outside this repo:

```bash
CLUSTER_RLOPT_LOCAL_PATH=/absolute/path/to/RLOpt
# Optional: syncs an arbitrary local Isaac Lab source checkout as an overlay.
# Isaac Lab is not a submodule of this repo, so there is no default path to
# fall back to -- set this only when testing unreleased Isaac Lab patches.
# CLUSTER_ISAACLAB_LOCAL_PATH=/absolute/path/to/IsaacLab
# CLUSTER_IMITATION_TOOLS_LOCAL_PATH=/absolute/path/to/ImitationLearningTools
```

These overrides are used when `CLUSTER_EXTRA_SYNC_SPECS` is not set. Only the uncommented overrides are synced as overlays. If none are set, the cluster job uses the submodule state from the main `IsaacLab-Imitation` checkout without extra repo sync. The paths are local paths on the submission machine.

Each `job` submission also writes a repo manifest to `<CLUSTER_ISAACLAB_DIR>/repo_sync_manifest.tsv` containing SHA/branch/dirty-state for all synced repos.
