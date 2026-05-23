# IsaacLab-Imitation

IsaacLab-Imitation is a multi-repo workspace for humanoid imitation learning on top of Isaac Lab. This repository
contains the Isaac Lab extension code for the imitation environments and pins the active `IsaacLab`, `RLOpt`, and
`ImitationLearningTools` dependency checkouts as git submodules.

The current focus is manager-based imitation environments for the Unitree G1 robot, with training flows built around
RLOpt and RSL-RL.

## What is in this repo

- `source/isaaclab_imitation`: the installable Isaac Lab extension package
- `scripts/rlopt`: training and playback entrypoints for RLOpt
- `scripts/rsl_rl`: training entrypoints for RSL-RL
- `scripts/zero_agent.py`, `scripts/random_agent.py`: smoke-test environment runners
- `IsaacLab/`, `RLOpt/`, `ImitationLearningTools/`: required submodule checkouts
- `source/isaaclab_imitation/isaaclab_imitation/assets/unitree`: vendored Unitree G1 URDF, meshes, and robot config
- `docker/cluster`: cluster submission utilities

Registered task IDs currently include:

- `Isaac-Imitation-G1-v0`
- `Isaac-Imitation-G1-Latent-v0`
- `Isaac-Imitation-G1-LafanTrack-v0` (legacy alias of `Isaac-Imitation-G1-v0`)

## Workspace setup

Clone with submodules:

```bash
git clone --recurse-submodules git@github.com:GTLIDAR/IsaacLab-Imitation.git
cd IsaacLab-Imitation
```

If you already cloned without submodules:

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

This workspace expects `IsaacLab`, `RLOpt`, and `ImitationLearningTools` to live under this repo as submodules. G1 robot
configuration and the required URDF/mesh assets are vendored in this repo under `source/isaaclab_imitation`, so
`unitree_rl_lab` is no longer required for training. `loco-mujoco` is optional and only needed when explicitly selecting
the `loco_mujoco` loader.

```text
/path/to/workspace-root/
  IsaacLab-Imitation/
  loco-mujoco/  # optional
```

More detail on remotes, submodules, and cluster sync lives in [REPO_SETUP.md](REPO_SETUP.md).

## Installation

Pick any conda environment name you want. The examples below use `SL`, but
`SkillLearning` or another Python 3.11 environment name is also fine:

```bash
CONDA_ENV="${CONDA_ENV:-SL}"
# Skip this line if the environment already exists.
conda create -y -n "$CONDA_ENV" python=3.11
conda activate "$CONDA_ENV"
```

Use the workspace installer:

```bash
./scripts/install_workspace.sh
```

The script does the following:

- verifies the active `python` is `3.11`
- installs `uv` with `conda install -y uv`
- initializes git submodules if `IsaacLab`, `RLOpt`, or `ImitationLearningTools` are incomplete
- installs `ImitationLearningTools` and `RLOpt` in editable mode
- installs LeRobot support when `INSTALL_LEROBOT=1` is set
- installs `isaacsim[all,extscache]==5.1.0`
- installs `torch==2.7.0` and `torchvision==0.22.0` from the CUDA 12.8 wheel index
- runs `./isaaclab.sh -i none` inside `IsaacLab`
- installs `source/isaaclab_imitation` in editable mode

If you need the manual submodule setup details or cluster notes, see [REPO_SETUP.md](REPO_SETUP.md).

To install optional LeRobot streaming dependencies for offline pretraining:

```bash
INSTALL_LEROBOT=1 ./scripts/install_workspace.sh
```

### LeRobot Reference Prep

Convert a Unitree LeRobot desired-command episode into an Isaac FK reference NPZ
at the native 30 Hz control rate. Use `observation.state.robot_q_current` only
when you explicitly want to inspect measured robot tracking instead of the
desired label sequence:

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

For multiple episodes, use `scripts/batch_csv_to_npz.py` with LeRobot jobs:

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

conda run -n "${CONDA_ENV:-SL}" python scripts/write_lafan1_npz_manifest.py \
    --npz_dir data/unitree/npz \
    --manifest_path data/unitree/manifests/g1_wbt_pillow_30hz.json \
    --dataset_name unitree_lerobot
```

NPZ manifests with a single FPS auto-sync the G1 env control rate. A 30 Hz
manifest uses 240 Hz physics with `env.decimation=8` unless timing is overridden.

### Large LeRobot Streaming

The current G1 WBT LeRobot collection list is tracked in
`data/unitree/g1_wbt_lerobot_repos.json`. The IPMD bilinear config uses that
list when `agent.offline_dataset.enabled=true`.

Probe the multi-repo streaming cache without launching Isaac:

```bash
conda run -n "${CONDA_ENV:-SL}" python scripts/validate_lerobot_streaming_cache.py \
    --repo_ids_file data/unitree/g1_wbt_lerobot_repos.json \
    --max_episodes_per_repo 1 \
    --min_ready_transitions 32 \
    --max_cache_transitions 20000 \
    --batch_size 16 \
    --drain
```

For training-scale runs, leave `agent.offline_dataset.max_episodes_per_repo=0`
and size the cache deliberately, for example:

```bash
agent.offline_dataset.enabled=true \
agent.offline_dataset.min_ready_transitions=100000 \
agent.offline_dataset.max_cache_transitions=5000000 \
agent.offline_dataset.max_episodes=0 \
agent.offline_dataset.max_episodes_per_repo=0
```

### Hugging Face And GitHub CLI

```bash
CONDA_ENV="${CONDA_ENV:-SL}"
conda activate "$CONDA_ENV"

# Hugging Face Hub CLI for LeRobot dataset access.
uv pip install --system -U "huggingface_hub[cli]"
hf auth login
hf auth whoami

# GitHub CLI is recommended for branch, push, PR, and CI workflows.
conda install -y -c conda-forge gh
gh auth login
gh auth setup-git --hostname github.com
gh auth status

# Optional: only for direct git push/pull to https://huggingface.co.
# This uses Git's plaintext store helper, scoped to Hugging Face only.
git config --global credential.https://huggingface.co.helper store

# If you are already logged in:
hf auth list
TOKEN_NAME=home-ubuntu
hf auth switch --token-name "$TOKEN_NAME" --add-to-git-credential

# If you are not logged in yet:
hf auth login --add-to-git-credential

# Remove the Hugging Face-scoped helper later if you no longer want it.
git config --global --unset credential.https://huggingface.co.helper
```

## Running training

Examples below assume you are running from the repository root.
Activate your conda environment first:

```bash
CONDA_ENV="${CONDA_ENV:-SL}"
conda activate "$CONDA_ENV"
```

Train a G1 imitation policy with RLOpt IPMD:

```bash
python scripts/rlopt/train.py \
    --task Isaac-Imitation-G1-Latent-v0 \
    --algo IPMD \
    --headless \
    env.lafan1_manifest_path=./data/lafan1/manifests/g1_lafan1_manifest.json
```

For imitation-based RL, the recommended starting point in this repo is RLOpt IPMD on
`Isaac-Imitation-G1-Latent-v0`. If you want a smaller single-motion setup for the
retargeted Unitree `dance102` clip, use:

```bash
python scripts/rlopt/train.py \
    --task Isaac-Imitation-G1-Latent-v0 \
    --algo IPMD \
    --headless \
    env.lafan1_manifest_path=./data/unitree/manifests/g1_unitree_dance102_manifest.json
```

The action-labeled Dance102 variant keeps the original NPZ intact and uses a
locally generated label NPZ plus a separate manifest. Generate those artifacts
with `scripts/rlopt/label_npz_with_policy.py` or provide your own matching
manifest before launching:

```bash
python scripts/rlopt/train.py \
    --task Isaac-Imitation-G1-Latent-v0 \
    --algo IPMD_BILINEAR \
    --headless \
    env.lafan1_manifest_path=./data/unitree/manifests/g1_unitree_dance102_rlopt_ipmd_500m_actions_manifest.json \
    env.reconstructed_reference_action=false \
    agent.bilinear.offline_pretrain.policy_bc_updates=2000
```

For the cluster ablation set comparing scratch, state-only SR pretraining,
reconstructed-action BC, and recorded-label BC:

```bash
DRY_RUN=1 experiments/bilinear_pretrain/submit_dance102_action_label_ablation.sh
experiments/bilinear_pretrain/submit_dance102_action_label_ablation.sh
```

Train with RLOpt PPO:

```bash
python scripts/rlopt/train.py \
    --task Isaac-Imitation-G1-v0 \
    --algo PPO \
    --headless \
    env.lafan1_manifest_path=./data/lafan1/manifests/g1_lafan1_manifest.json
```

Train ASE with the full local LAFAN1 G1 manifest:

```bash
python scripts/rlopt/train.py \
    --task Isaac-Imitation-G1-Latent-v0 \
    --num_envs 4096 \
    --algo ASE \
    --headless \
    --video \
    env.lafan1_manifest_path=./data/lafan1/manifests/g1_lafan1_manifest.json
```

Train latent-conditioned IPMD with the same manifest:

```bash
python scripts/rlopt/train.py \
    --task Isaac-Imitation-G1-Latent-v0 \
    --algo IPMD \
    --headless \
    env.lafan1_manifest_path=./data/lafan1/manifests/g1_lafan1_manifest.json
```

To run IPMD on the vanilla tracking task instead, disable latent commands explicitly:

```bash
python scripts/rlopt/train.py \
    --task Isaac-Imitation-G1-v0 \
    --algo IPMD \
    --headless \
    env.lafan1_manifest_path=./data/lafan1/manifests/g1_lafan1_manifest.json \
    ipmd.use_latent_command=False
```

If you want to reuse an existing cached Zarr dataset instead of rebuilding it on startup, add:

```bash
env.refresh_zarr_dataset=False
```

For manifest-driven G1 tasks, the cache path is derived from the resolved manifest path and contents, so LaFAN1 and
Unitree manifests do not share the same Zarr dataset by default.

Run the lightweight vanilla G1 IPMD training smoke routine:

```bash
scripts/rlopt/smoke_train_g1_ipmd.sh
```

This runs one 128-env rollout iteration on `data/lafan1/manifests/g1_lafan1_manifest_single.json`, rebuilds that
single-manifest Zarr cache, and records a short local training video. It disables the metrics backend by default so it
does not require W&B credentials. To test W&B video sync too, run:

```bash
LOGGER_BACKEND=wandb scripts/rlopt/smoke_train_g1_ipmd.sh
```

Useful overrides:

```bash
MAX_ITERATIONS=2 NUM_ENVS=256 MANIFEST=data/lafan1/manifests/g1_debug_manifest.json \
    scripts/rlopt/smoke_train_g1_ipmd.sh
```

Train with RSL-RL:

```bash
python scripts/rsl_rl/train.py \
    --task Isaac-Imitation-G1-v0 \
    --headless \
    env.lafan1_manifest_path=./data/lafan1/manifests/g1_lafan1_manifest.json
```

`Isaac-Imitation-G1-LafanTrack-v0` remains registered as a backward-compatible alias to
`Isaac-Imitation-G1-v0`, but new commands should prefer `Isaac-Imitation-G1-v0`.

Common flags:

- `--task`: selects the registered Isaac Lab environment
- `--num_envs`: overrides the environment count from config
- `--max_iterations`: caps training iterations
- `--video`: records periodic rollout videos during training
- `--device cuda:0`: pins execution to a specific GPU

Logs are written under `logs/`.

## Data preparation

Motion loading in this repo is manifest-driven and repo-local under `data/`.

Tracked manifests:

- `source/isaaclab_imitation/isaaclab_imitation/manifests/g1_lafan1_manifest.template.json`: tracked template for a
  full local G1 LAFAN1 manifest

Local manifests:

- `data/lafan1/manifests/g1_lafan1_manifest.json`: full local G1 LAFAN1 manifest
- `data/lafan1/manifests/g1_debug_manifest.json`: optional smaller local subset
- `data/unitree/manifests/g1_unitree_dance102_manifest.json`: single-motion Unitree
  `dance102` manifest pointing at `data/unitree/npz/g1/G1_Take_102.bvh_60hz.npz`

The full local G1 set is not shipped in git. When you prepare local motions under `data/lafan1/npz/g1/`, the full
manifest should live under `data/lafan1/manifests/g1_lafan1_manifest.json`.

The Unitree `dance102` manifest is useful for quick smoke tests and smaller imitation-based
RL runs before scaling up to the full LAFAN1 manifest.

See `data/README.md` for the expected local directory layout and the common local-data commands.

### Recommended full-dataset flow

The simplest way to get the full local G1 dataset from the public Hugging Face dataset
`lvhaidong/LAFAN1_Retargeting_Dataset` is the shell wrapper:

```bash
./scripts/download_g1_lafan1_data.sh
```

This downloads the G1 subset into `data/` and then runs the local NPZ + manifest preparation step.
To bake the G1 arms-up alignment trim into the generated NPZ files, pass
`--auto_trim_mode g1_shoulder_roll`.

The underlying Python entrypoint is:

```bash
conda run -n "${CONDA_ENV:-SL}" python scripts/setup_lafan1_dataset.py \
    --prepare-npz --headless
```

For the G1 retargeted set, the public CSV motions often begin with an arms-up
alignment pose. To bake a per-motion trim into the generated NPZ files, add:

```bash
conda run -n "${CONDA_ENV:-SL}" python scripts/setup_lafan1_dataset.py \
    --prepare-npz --headless \
    --auto_trim_mode g1_shoulder_roll
```

Both commands download the public retargeted LAFAN1 G1 CSV set, convert it to NPZ, and write:

```text
data/lafan1/raw/g1/
data/lafan1/npz/g1/
data/lafan1/manifests/g1_lafan1_manifest.json
```

The Hugging Face dataset stores the retargeted G1 motions at 30 FPS, so the wrapper passes `--input_fps 30`
automatically during conversion. Use `--robot_type h1`, `--robot_type h1_2`, or `--robot_type all` for other subsets.

### If You Already Have NPZ Files

If `data/lafan1/manifests/g1_lafan1_manifest.json` already exists, you do not need to regenerate it.

If you already have local NPZ files but no manifest yet, generate one directly:

```bash
conda run -n "${CONDA_ENV:-SL}" python scripts/write_lafan1_npz_manifest.py \
    --npz_dir data/lafan1/npz/g1 \
    --manifest_path data/lafan1/manifests/g1_lafan1_manifest.json
```

If you want to hand-edit a manifest instead of generating one, copy the tracked template:

```bash
mkdir -p data/lafan1/manifests
cp source/isaaclab_imitation/isaaclab_imitation/manifests/g1_lafan1_manifest.template.json \
   data/lafan1/manifests/g1_lafan1_manifest.json
```

For a smaller local subset:

```bash
conda run -n "${CONDA_ENV:-SL}" python scripts/write_lafan1_npz_manifest.py \
    --npz_dir data/lafan1/npz/g1 \
    --manifest_path data/lafan1/manifests/g1_debug_manifest.json \
    --select dance1_subject1 dance1_subject2 walk1_subject1
```

### If You Start From CSV Files

Prepare local CSV motions into NPZ plus a manifest with:

```bash
conda run -n "${CONDA_ENV:-SL}" python scripts/prepare_lafan1_from_csv.py \
    --csv_dir /absolute/path/to/csv_motions \
    --npz_dir /absolute/path/to/data/lafan1/npz/g1 \
    --manifest_path /absolute/path/to/data/lafan1/manifests/g1_lafan1_manifest.json \
    --recursive
```

If you want one replay MP4 per converted motion, add `--record_videos` and `--video_dir`.

To auto-trim the G1 arms-up alignment segment while rebuilding NPZ files, add:

```bash
conda run -n "${CONDA_ENV:-SL}" python scripts/prepare_lafan1_from_csv.py \
    --csv_dir /absolute/path/to/csv_motions \
    --npz_dir /absolute/path/to/data/lafan1/npz/g1 \
    --manifest_path /absolute/path/to/data/lafan1/manifests/g1_lafan1_manifest.json \
    --recursive \
    --auto_trim_mode g1_shoulder_roll \
    --overwrite
```

That trims each CSV before conversion, writes clean NPZ files suitable for
upload to Hugging Face, and records the source trim range in the manifest as
provenance.

If you already have NPZ files and only want a trimmed manifest without
rewriting those NPZ files, use:

```bash
conda run -n "${CONDA_ENV:-SL}" python scripts/prepare_lafan1_from_csv.py \
    --csv_dir /absolute/path/to/csv_motions \
    --npz_dir /absolute/path/to/data/lafan1/npz/g1 \
    --manifest_path /absolute/path/to/data/lafan1/manifests/g1_lafan1_manifest.json \
    --recursive \
    --assume_npz_exists \
    --auto_trim_mode g1_shoulder_roll
```

In that mode the per-motion trim is written into each manifest entry as
`frame_range`, leaving the NPZ payload unchanged.

### Direct NPZ Sync With Hugging Face

If you only want the prepared NPZ subtree, use:

```bash
conda run -n "${CONDA_ENV:-SL}" python scripts/setup_g1_lafan1_npz_dataset.py
```

That syncs `npz/g1` from the dataset repo `GeorgiaTech/g1_lafan1_50hz` into:

```text
data/lafan1/npz/g1/
```

Upload mode pushes the same local NPZ tree back to Hugging Face:

```bash
conda run -n "${CONDA_ENV:-SL}" python scripts/setup_g1_lafan1_npz_dataset.py \
    --mode upload --token "$HF_TOKEN"
```

## Playback and smoke tests

Run a zero-action smoke test:

```bash
python scripts/zero_agent.py \
    --task Isaac-Imitation-G1-v0 \
    env.lafan1_manifest_path=./data/lafan1/manifests/g1_lafan1_manifest.json
```

Run a random-action smoke test:

```bash
python scripts/random_agent.py \
    --task Isaac-Imitation-G1-v0 \
    env.lafan1_manifest_path=./data/lafan1/manifests/g1_lafan1_manifest.json
```

Play back an RLOpt checkpoint:

```bash
python scripts/rlopt/play.py \
    --task Isaac-Imitation-G1-v0 \
    --checkpoint /absolute/path/to/checkpoint.pt \
    env.lafan1_manifest_path=./data/lafan1/manifests/g1_lafan1_manifest.json
```

Compare an RLOpt policy checkpoint against the synchronized reference motion:

```bash
python scripts/compare_policy_reference.py \
    --task Isaac-Imitation-G1-Latent-v0 \
    --algo IPMD \
    --checkpoint /absolute/path/to/checkpoint.pt \
    env.lafan1_manifest_path=./data/lafan1/manifests/g1_lafan1_manifest.json \
    env.refresh_zarr_dataset=False
```

Replay all 40 local G1 LAFAN1 motions from the full manifest:

```bash
python scripts/replay_reference.py \
    --task Isaac-Imitation-G1-v0 \
    --motion_manifest data/lafan1/manifests/g1_lafan1_manifest.json \
    --motion_refresh_dataset \
    --reset_schedule round_robin \
    --num_envs 40 \
    --video \
    --video_length 500 \
    --headless
```

Notes:

- use `data/lafan1/manifests/g1_lafan1_manifest.json` to load the full local 40-motion set
- `Isaac-Imitation-G1-v0` is the canonical vanilla tracking task and expects `env.lafan1_manifest_path=...`
- `Isaac-Imitation-G1-Latent-v0` is the latent-conditioned variant for ASE or latent-enabled IPMD
- `Isaac-Imitation-G1-LafanTrack-v0` remains available as a legacy alias for the vanilla task
- `replay_reference.py` disables reward and termination terms by default, so long reference videos do not reset early
- pass `--keep_terminations` or `--keep_rewards` if you explicitly want the old RL-style behavior during replay
- `--num_envs 40` is the way to see all 40 loaded trajectories at once; using fewer environments still loads the manifest,
  but only that many trajectories are visible at a time

## Development workflow

This repo is easier to work on with terminal-first tooling than with heavy IDE indexing.

Recommended tools:

- `ruff` for linting and formatting
- `pyrefly` for type and import checking
- `pytest` for focused unit tests

Use your selected conda environment for all checks. Prefer `conda run` for
non-interactive commands so the repo uses that environment's Python and tooling:

```bash
CONDA_ENV="${CONDA_ENV:-SL}"
```

Install tools into that environment:

```bash
conda run -n "${CONDA_ENV:-SL}" uv pip install --system ruff pyrefly pytest
```

Useful commands:

```bash
conda run -n "${CONDA_ENV:-SL}" ruff check .
conda run -n "${CONDA_ENV:-SL}" ruff format --check .
conda run -n "${CONDA_ENV:-SL}" pyrefly check
```

Pure-Python pytest targets can run directly through the conda environment, for
example:

```bash
conda run -n "${CONDA_ENV:-SL}" pytest source/isaaclab_imitation/test_reference_patch_env.py
```

Tests that import Isaac Lab or Omniverse modules need Isaac Sim's Python
bootstrap before imports such as `pxr` are available. Run those tests through
the in-repo Isaac Lab submodule launcher:

```bash
TERM=xterm conda run -n "${CONDA_ENV:-SL}" ./IsaacLab/isaaclab.sh -p -m pytest source/isaaclab_imitation/test_reference_patch_env.py
```

`pyrefly` is configured by [source/isaaclab_imitation/pyproject.toml](source/isaaclab_imitation/pyproject.toml) and
already includes the import roots for this repo plus dependency checkouts such as `IsaacLab`, `RLOpt`, and
`ImitationLearningTools`.

For VS Code, prefer the Ruff extension and terminal-based `pyrefly` checks. Pylance is not the recommended workflow for
this workspace because the Isaac / Omniverse dependency tree is large, generated settings tend to drift, and static
analysis is more reliable here when driven from the checked-in repo configuration.

## Formatting and hooks

A pre-commit configuration is included:

```bash
pip install pre-commit
pre-commit run --all-files
```

Note that the current hook set is inherited from upstream Isaac Lab conventions. For day-to-day work in this repo,
`ruff` and `pyrefly` are the recommended feedback loop.

## Cluster note

For cluster submission, local Isaac Lab Python installation is not required on the submission machine if jobs run inside
the provided container or Apptainer image. See `docker/cluster` and [REPO_SETUP.md](REPO_SETUP.md) for the expected sync
layout and environment variables.

Cluster jobs submitted through `docker/cluster/cluster_interface.sh job ...` now auto-check the G1 dataset tree before
running the user workload. The container-side preflight in `docker/cluster/run_singularity.sh` verifies that the G1 NPZ
tree under `${CLUSTER_G1_DATA_ROOT:-${CLUSTER_DATA_DIR}/lafan1}` contains at least 40 motions. If the dataset is
incomplete, it downloads the G1 NPZ dataset from Hugging Face with `scripts/setup_g1_lafan1_npz_dataset.py` and
regenerates `g1_lafan1_manifest.json` with `scripts/write_lafan1_npz_manifest.py` only when the manifest is missing or
older than the NPZ files. You can override that behavior with `CLUSTER_G1_MANIFEST_REFRESH_POLICY`:
`auto` regenerates only when needed, `never` leaves the manifest untouched, and `always` regenerates on every job.

Submitted jobs also append a default full-dataset override:

```text
env.lafan1_manifest_path=${CLUSTER_G1_MANIFEST_PATH:-${CLUSTER_G1_DATA_ROOT:-${CLUSTER_DATA_DIR}/lafan1}/manifests/g1_lafan1_manifest.json}
```

That gives cluster training the 40-motion G1 manifest by default. If you want a different manifest, either set
`CLUSTER_G1_MANIFEST_PATH` in `docker/cluster/.env.cluster`, disable the behavior with
`CLUSTER_APPEND_DEFAULT_G1_MANIFEST=0`, or pass `env.lafan1_manifest_path=...` explicitly in the submitted job args.

Relevant cluster env vars:

- `CLUSTER_AUTO_SETUP_G1_DATA=1`: enable the automatic G1 dataset bootstrap before each job (default)
- `CLUSTER_G1_EXPECTED_MOTION_COUNT=40`: minimum motion count required for the G1 manifest check
- `CLUSTER_G1_DATA_ROOT=${CLUSTER_DATA_DIR}/lafan1`: override the G1 dataset root checked by the preflight helper
- `CLUSTER_G1_REPO_ID=GeorgiaTech/g1_lafan1_50hz`: override the Hugging Face dataset repo used for G1 NPZ download
- `CLUSTER_HF_TOKEN_FILE=/path/to/.hf_token`: recommended way to provide a Hugging Face read token for cluster-side dataset download
- `CLUSTER_HF_TOKEN=hf_xxx`: inline token override if you do not want to use a token file
- `CLUSTER_WANDB_API_KEY_FILE=/path/to/.wandb_api_key`: recommended way to provide a W&B API key from the cluster host into the container
- `CLUSTER_WANDB_API_KEY=...`: inline W&B API key override if you do not want to use a token file
- `CLUSTER_APPEND_DEFAULT_G1_MANIFEST=1`: append the default full-manifest override to submitted jobs
- `CLUSTER_G1_MANIFEST_PATH=${CLUSTER_G1_DATA_ROOT}/manifests/g1_lafan1_manifest.json`: override the default full-manifest job argument
- `CLUSTER_G1_MANIFEST_REFRESH_POLICY=auto`: control whether cluster preflight regenerates the manifest (`never` is the right setting for a Unitree manifest you want to preserve)

For private repos or authenticated Hugging Face access on the cluster, the recommended setup is:

```bash
printf '%s\n' 'hf_...' > ~/.hf_token
chmod 600 ~/.hf_token
```

Then set in `docker/cluster/.env.cluster`:

```bash
CLUSTER_HF_TOKEN_FILE=/home/<user>/.hf_token
```

For W&B, the same host-side pattern is recommended:

```bash
printf '%s\n' 'your_wandb_api_key' > ~/.wandb_api_key
chmod 600 ~/.wandb_api_key
```

Then set in `docker/cluster/.env.cluster`:

```bash
CLUSTER_WANDB_API_KEY_FILE=/home/<user>/.wandb_api_key
```

The W&B key file is read on the cluster host before `singularity exec`, then injected into the container as `WANDB_API_KEY`.
