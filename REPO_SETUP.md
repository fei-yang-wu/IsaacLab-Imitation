# Repository Setup (Git Submodules)

This repo now tracks its dependent repos as submodules:

- `IsaacLab/`
- `RLOpt/`
- `ImitationLearningTools/`

`IsaacLab-Imitation` itself remains the top-level repo.

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
git -C IsaacLab remote -v
git -C RLOpt remote -v
git -C ImitationLearningTools remote -v
```

Expected default remotes:

- `IsaacLab-Imitation`: `origin -> git@github.com:GTLIDAR/IsaacLab-Imitation.git`
- `IsaacLab`: `origin -> git@github.com:GTLIDAR/IsaacLab.git`
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
git -C IsaacLab remote add upstream git@github.com:isaac-sim/IsaacLab.git
git -C RLOpt remote add gatech https://github.gatech.edu/GeorgiaTechLIDARGroup/RLOpt.git
```

## 3. Update submodules later

```bash
git submodule update --init --recursive
```

To move submodules to newer commits from their configured tracking branches:

```bash
git submodule update --remote --recursive
git add IsaacLab RLOpt ImitationLearningTools
git commit -m "Update submodule pins"
```

For feature work, it is also valid to check out an exact commit inside a
submodule and then commit the updated top-level gitlink. Keep `.gitmodules`
tracking branches unchanged unless the long-lived default branch changes.

## 4. Cluster note (no conda/venv needed for submission)

For cluster submission, you do not need a local conda/venv for IsaacLab Python packages.

- Job execution uses `/isaac-sim/python.sh` inside the container/Apptainer image.
- Local requirements for submission are mainly Docker, Apptainer, and SSH access to the cluster.

Typical flow:

```bash
cd docker/cluster
# edit .env.cluster for cluster paths/login/script
bash cluster_interface.sh push base
bash cluster_interface.sh job --task Isaac-Imitation-G1-Latent-v0 --algo IPMD --headless
```

**Multiple clusters:** per-cluster env files and submit scripts are supported. Create `docker/cluster/.env.<name>` and optionally `docker/cluster/submit_job_slurm_<name>.sh`, then pass `-c <name>`:

```bash
bash cluster_interface.sh -c ice job --task Isaac-Imitation-G1-Latent-v0 --algo IPMD --headless
```

If no `-c` is given, the script auto-selects `submit_job_slurm_${CLUSTER_LOGIN}.sh` (from `.env.cluster`) when that file exists, uses `submit_job_slurm_pace.sh` for `*.pace.gatech.edu` logins, and otherwise falls back to `submit_job_slurm.sh`. You can force a submitter with `CLUSTER_SLURM_SUBMIT_SCRIPT=pace`.

For Georgia Tech ICE/PACE pipeline jobs, use:

```bash
DRY_RUN=1 experiments/submit_hl_skill_pipeline_pace_2b.sh
CLUSTER_SLURM_ACCOUNT=<pace-account> DRY_RUN=0 experiments/submit_hl_skill_pipeline_pace_2b.sh
```

The helper defaults to `ice-gpu`, `gpu:l40s:1`, `coe-ice`, and 32G RAM; override
those with `CLUSTER_SLURM_PARTITION`, `CLUSTER_SLURM_GPU_GRES`,
`CLUSTER_SLURM_QOS`, or `CLUSTER_SLURM_MEM` if the active allocation differs.

By default, cluster jobs use the submodule states pinned by this top-level repo.
Only set path overrides in `docker/cluster/.env.cluster` when a task explicitly
needs an unpinned local checkout outside this repo:

```bash
CLUSTER_RLOPT_LOCAL_PATH=/absolute/path/to/RLOpt
# Optional:
# CLUSTER_ISAACLAB_LOCAL_PATH=/absolute/path/to/IsaacLab
# CLUSTER_IMITATION_TOOLS_LOCAL_PATH=/absolute/path/to/ImitationLearningTools
```

These overrides are used when `CLUSTER_EXTRA_SYNC_SPECS` is not set. Only the uncommented overrides are synced as overlays. If none are set, the cluster job uses the submodule state from the main `IsaacLab-Imitation` checkout without extra repo sync. The paths are local paths on the submission machine.

Each `job` submission also writes a repo manifest to `<CLUSTER_ISAACLAB_DIR>/repo_sync_manifest.tsv` containing SHA/branch/dirty-state for all synced repos.
