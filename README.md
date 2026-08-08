# IsaacLab-Imitation

IsaacLab-Imitation is a multi-repo workspace for humanoid imitation learning on top of Isaac Lab. This repository
contains the Isaac Lab extension code for the imitation environments and pins the active `RLOpt` and
`ImitationLearningTools` dependency checkouts as git submodules. Isaac Lab itself is pinned as a regular Pixi/PyPI
dependency (`isaaclab==3.0.0b2.post1` from NVIDIA's index), not a submodule.

The current focus is manager-based imitation environments for the Unitree G1 robot, with training flows built around
RLOpt and RSL-RL.

## What is in this repo

- `source/isaaclab_imitation`: the installable Isaac Lab extension package
- `scripts/rlopt`: training and playback entrypoints for RLOpt
- `scripts/rsl_rl`: training entrypoints for RSL-RL
- `scripts/zero_agent.py`, `scripts/random_agent.py`: smoke-test environment runners
- `experiments/`: current-campaign navigation, reusable experiment tooling, and the staged paper-facing entrypoint
- `RLOpt/`, `ImitationLearningTools/`: required submodule checkouts
- `source/isaaclab_imitation/isaaclab_imitation/assets/unitree`: vendored Unitree G1 URDF, meshes, and robot config
- `docker/cluster`: cluster submission utilities

Registered task IDs currently include:

| Task ID | Actor command | Skill-encoder window |
| --- | --- | --- |
| `Isaac-Imitation-G1-v2` | latent, 258-D | single frame |
| `Isaac-Imitation-G1-Explicit-v2` | explicit, full body | none |
| `Isaac-Imitation-G1-Chunk-v2` | 10-frame packet held 10 steps | none |
| `Isaac-Imitation-G1-VQVAE-v0` | latent, 258-D | 8 past + current |
| `Isaac-Imitation-G1-CVAE-v0` | latent, 256-D | current + 9 future |
| `Isaac-Imitation-G1-PerStepVQ-v0` | latent, 64-D | current + 9 future |
| `Isaac-Imitation-G1-Sonic-v0` | latent, 258-D | single frame, SONIC recipe |

Each is one point in the same environment's configuration space, so an id is
the citable name of a protocol rather than a distinct implementation, and the
same selections are available directly:

```bash
--task Isaac-Imitation-G1-v2 \
    env.command_interface.actor=latent|explicit|chunk \
    env.command_interface.encoder=single|causal9|future10|future26 \
    env.command_interface.reference.selection=default|sonic|random80_adaptive20|frame0
```

`random80_adaptive20` chooses a trajectory uniformly and a start frame
uniformly within its first 50% on 80% of resets. The other 20% use the learned
SONIC failure distribution. Unlike SONIC's internal uniform-bin mixture, the
random branch gives every trajectory equal probability.

Register a new id when a protocol needs to be cited later; until then, override.

`Isaac-Imitation-G1-v0`, `-v1`, `-Latent-v0`, `-Strict-v0`, and `-LafanTrack-v0`
stay registered for reproducing recorded results and should not be cited for new
work. They are also the boundary for how motion data is configured: `-G1-v2` and
later use `env.data.*` (`env.data.manifest`, `env.data.cache_dir`,
`env.data.clips`), while the frozen ids keep the older flat fields
(`env.lafan1_manifest_path`, `env.dataset_path`, `env.motions`). Setting a flat
field on a v2 task fails with the replacement named, rather than silently
training on data it did not select.

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

This workspace expects `RLOpt` and `ImitationLearningTools` to live under this repo as submodules. G1 robot
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

Install Pixi if it is not already available:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
pixi --version
```

Install the default development environment:

```bash
pixi install
```

The default environment is intentionally Isaac-light. It installs Python 3.12,
PyTorch 2.11.0 / torchvision 0.26.0 from the CUDA 13.0 wheel index,
TensorDict / TorchRL, and the local editable `RLOpt` and
`ImitationLearningTools` submodules. It does not install Isaac Sim, Isaac Lab,
or `isaaclab_imitation`, so RLOpt and ILTools tests do not trigger TorchRL's
IsaacLab integration path.

Install the Isaac Lab environment only when you need Isaac-backed training,
playback, conversion, or tests:

```bash
pixi install -e isaaclab
```

The `isaaclab` environment adds `isaaclab[isaacsim,all]==3.0.0b2.post1` (Isaac Sim 6.0.1) from
NVIDIA's PyPI index plus editable `source/isaaclab_imitation`. Pixi owns both
Conda and PyPI dependencies through `pixi.toml`; do not install repo
dependencies with `conda`, `pip`, or `uv`.

The compatibility wrapper below initializes submodules and installs a selected
Pixi environment, defaulting to `default`:

```bash
./scripts/install_workspace.sh
PIXI_ENVIRONMENT=isaaclab ./scripts/install_workspace.sh
```

If you need the manual submodule setup details or cluster notes, see [REPO_SETUP.md](REPO_SETUP.md).

To install optional LeRobot streaming dependencies for offline pretraining:

```bash
pixi install -e lerobot
pixi install -e isaaclab-lerobot
```

### LeRobot Reference Prep

Convert a Unitree LeRobot desired-command episode into an Isaac FK reference NPZ
at the native 30 Hz control rate. Use `observation.state.robot_q_current` only
when you explicitly want to inspect measured robot tracking instead of the
desired label sequence:

```bash
TERM=xterm PYTHONUNBUFFERED=1 \
pixi run -e isaaclab-lerobot python scripts/viz/replay_unitree_lerobot_reference.py \
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

For multiple episodes, use `scripts/data/batch_csv_to_npz.py` with LeRobot jobs:

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
pixi run -e isaaclab-lerobot python scripts/data/batch_csv_to_npz.py \
    --headless \
    --device cuda:0 \
    --jobs_json data/unitree/lerobot_jobs.json \
    --output_fps 30

pixi run python scripts/data/write_lafan1_npz_manifest.py \
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
pixi run -e lerobot python scripts/audit/validate_lerobot_streaming_cache.py \
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
# Hugging Face Hub CLI for LeRobot dataset access.
pixi run -e lerobot hf auth login
pixi run -e lerobot hf auth whoami

# GitHub CLI is recommended for branch, push, PR, and CI workflows.
pixi run gh auth login
pixi run gh auth setup-git --hostname github.com
pixi run gh auth status

# Optional: only for direct git push/pull to https://huggingface.co.
# This uses Git's plaintext store helper, scoped to Hugging Face only.
git config --global credential.https://huggingface.co.helper store

# If you are already logged in:
pixi run -e lerobot hf auth list
TOKEN_NAME=home-ubuntu
pixi run -e lerobot hf auth switch --token-name "$TOKEN_NAME" --add-to-git-credential

# If you are not logged in yet:
pixi run -e lerobot hf auth login --add-to-git-credential

# Remove the Hugging Face-scoped helper later if you no longer want it.
git config --global --unset credential.https://huggingface.co.helper
```

## Running training

Examples below assume you are running from the repository root.
Use the Isaac Lab Pixi environment for Isaac-backed training:

```bash
pixi shell -e isaaclab
```

### Start here: the current default pipeline

One script runs the current default end to end — pretrain a **deterministic
continuous (det-SR)** skill encoder, then train the low-level tracker on the
**tuned recipe** conditioned on that frozen encoder:

```bash
bash scripts/rlopt/run_local_v2_pipeline.sh

# quick check first; DRY_RUN=1 prints both commands and runs nothing
DRY_RUN=1 bash scripts/rlopt/run_local_v2_pipeline.sh
TOTAL_FRAMES=10000000 bash scripts/rlopt/run_local_v2_pipeline.sh
```

Defaults: task `Isaac-Imitation-G1-v2`, agent
`rlopt_ipmd_tuned_cfg_entry_point`, encoder horizon 10 / `z_dim` 256 (published
command width **258**, including the `sin_cos` phase), `newton_mjwarp` with
njmax 288 / nconmax 200, 4096 envs, 50M frames, and **instantaneous
terminations** — the persistence window is opt-in via `TERMINATION_WINDOW=N`.

Budget guidance: ~10M frames for routine debugging, at most ~50M for a serious
local check, and do not run 100M locally. Local runs qualify code; the cluster
produces convergence and paper numbers.

Read [wiki/local-experiments.md](wiki/local-experiments.md) before a first run.
It covers what "the default" currently resolves to, how to evaluate a
checkpoint (strict **and** the full-horizon diagnostic), how to read MPJPE-L
against MPJPE-G, and the measurement traps that have cost real time — Newton is
not run-to-run deterministic at a fixed seed, and every per-minute rate is
gameable by anything that lengthens an episode.

### SONIC-compatible success-rate evaluation

Use the dedicated [SONIC success evaluation
protocol](wiki/sonic-success-evaluation.md) when reporting a checkpoint with
SONIC's published success-rate definition. It is a third pass, separate from
both task-strict qualification and the required non-terminating diagnostic.

A motion succeeds only when it reaches the end of its reference without any of
these failures: pelvis height error above 0.25 m, ankle/wrist height error above
0.25 m, or full pelvis orientation error above 1 rad. The launch must disable
`foot_pos_xyz` **and the interval push**, while retaining startup and reset
randomization. Evaluate mode actions deterministically, run every assigned clip
to completion, and preserve the exact task, agent entry point, encoder, command
interface, data, and physics contract used by the checkpoint.

Read the result as:

- SR: `aggregate.completed_tracking_success_rate`
- MPJPE-L: `successful_metrics.tracking_mpjpe_mm.mean`

Do not use `tracking_success_rate` from a capped rollout: an unfinished survivor
has not succeeded. MPJPE-L is micro-averaged over successful motions only, as in
SONIC's evaluator. The repo-local `sonic-success-eval` skill contains the
launch and validation checklist.

Train a G1 imitation policy with RLOpt IPMD:

```bash
python scripts/rlopt/train.py \
    --task Isaac-Imitation-G1-v2 \
    --algo IPMD \
    --headless \
    env.data.manifest=./data/lafan1/manifests/g1_lafan1_manifest.json
```

The task runs at **50 Hz control** (200 Hz physics, `sim.dt=0.005`,
`decimation=4`). That is a protocol decision every reward, termination
threshold, and recorded result is defined at, so it is declared by the task and
never inferred from data: clips are checked against it and a mismatch is
refused rather than silently retuning the physics rate. The conversion pipeline
in `scripts/data/` resamples sources to 50 Hz, so a mismatch means the wrong
manifest.

Train the IPMD learning-to-teach variant on a current-v2 command surface:

```bash
python scripts/rlopt/train.py \
    --task Isaac-Imitation-G1-Explicit-v2 \
    --algo IPMD_L2T \
    --headless \
    env.data.manifest=./data/lafan1/manifests/g1_lafan1_manifest.json
```

`IPMD_L2T` keeps rollout control and the ordinary IPMD/PPO objectives on a
privileged teacher that reads the critic observations. A second actor reads
the normal policy observations and learns from the teacher's executed actions;
it never controls training rollouts. The same algorithm entry point is
registered on `Isaac-Imitation-G1-v2`, `Isaac-Imitation-G1-Explicit-v2`, and
`Isaac-Imitation-G1-Chunk-v2`. Latent v2 runs retain the usual IPMD skill-command
checkpoint requirements.

Play an IPMD-L2T checkpoint with the deployable student policy:

```bash
python scripts/rlopt/play.py \
    --task Isaac-Imitation-G1-v2 \
    --algo IPMD_L2T \
    --agent rlopt_ipmd_l2t_tuned_cfg_entry_point \
    --checkpoint /absolute/path/to/checkpoint.pt \
    env.data.manifest=./data/lafan1/manifests/g1_lafan1_manifest.json
```

### LAFAN1 local pretrain + low-level pipeline (reproducible)

> **Superseded.** This section documents the PRE-v2 recipe, kept to reproduce
> runs that predate the v2 command interface. For a new experiment use
> `scripts/rlopt/run_local_v2_pipeline.sh` (see "Start here" above and
> [wiki/local-experiments.md](wiki/local-experiments.md)).

The pre-v2 recipe trains a G1 LAFAN1 policy in two stages — pretrain a
DiffSR skill encoder from expert motion, then train the low-level "oracle" IPMD policy
conditioned on that encoder. One script chains both stages with the validated defaults
(builds the zarr cache, wires the fresh skill checkpoint into the low-level run):

```bash
bash scripts/rlopt/run_local_pretrain_lowlevel.sh
```

Defaults: skill encoder `W=25`, `z_dim=256`, DiffSR `128/512`, 5000 updates; low-level
`--algo IPMD` on `Isaac-Imitation-G1-Latent-v0` to 2B frames, 4096 envs, video + wandb.
Every value is env-overridable, e.g. a quick smoke run:

```bash
TOTAL_FRAMES=20000000 LOGGER_BACKEND=none bash scripts/rlopt/run_local_pretrain_lowlevel.sh
```

Expected low-level curve: `r_ep` climbs from <1 to ~18 by ~150M frames and refines toward
convergence by 2B. The full per-stage commands, expected metrics, joint-order verification,
and troubleshooting are in [wiki/lafan1-local-training.md](wiki/lafan1-local-training.md).

For imitation-based RL, the recommended starting point in this repo is RLOpt IPMD on
`Isaac-Imitation-G1-v2`. If you want a smaller single-motion setup for the
retargeted Unitree `dance102` clip, use:

```bash
python scripts/rlopt/train.py \
    --task Isaac-Imitation-G1-v2 \
    --algo IPMD \
    --headless \
    env.data.manifest=./data/unitree/manifests/g1_unitree_dance102_manifest.json
```

Restrict it to a subset of the manifest's clips with
`env.data.clips='["dance1_subject1"]'`, and point the built Zarr cache somewhere
specific with `env.data.cache_dir=...` (omit it and the cache path is derived
from the manifest's identity, so jobs naming the same manifest share one).

Train with RLOpt PPO on the explicit command surface (PPO and SAC read the
vanilla input keys, so pair them with an explicit actor):

```bash
python scripts/rlopt/train.py \
    --task Isaac-Imitation-G1-Explicit-v2 \
    --algo PPO \
    --headless \
    env.data.manifest=./data/lafan1/manifests/g1_lafan1_manifest.json
```

The `ASE`, `GAIL`, `AMP`, and `IPMD_BILINEAR` algorithms still exist in RLOpt,
but their agent-config entry points were pruned from these tasks on 2026-08-01;
selecting one fails to resolve an agent config. `IPMD` (default), `IPMD_L2T`,
`PPO`, and `SAC` are the live pairings.

The broad command-space oracle ablation is archived. Current Phase-4/5 qualification retains only two internal helpers in `experiments/campaigns/2026-07-23-bones-phase5-language-local10/command_space_ablation/`: checkpoint evaluation and the low-level oracle submission adapter. They are dependencies of guarded paper workflows, not a collaborator-facing command-style sweep. Historical paths and recovery instructions are in [`experiments/PRUNED_SCRIPTS.md`](experiments/PRUNED_SCRIPTS.md).

For experiment navigation, begin with
[`experiments/README.md`](experiments/README.md). It points to the current
dated campaign, exhaustive live inventory, and paper-facing staging surface.

The paper-facing learned-planner comparison has exactly two rows: the learned
DiffSR latent interface and the ten-frame explicit vanilla command packet.
The stable public entrypoint is staged under
[`experiments/paper/`](experiments/paper/README.md); it remains blocked until
the documented Phase 4 and Phase 5 release gates pass. The authoritative
protocol is
[`wiki/causal-interface-paper-plan.md`](wiki/causal-interface-paper-plan.md).

The current ten-goal BONES-SEED language-planner baseline is trajectory-first.
It collects 100 complete frame-0 oracle-policy trajectories per motion in one
1,000-environment Newton process, with domain randomization retained, pushes
disabled, deterministic policy actions, and official SONIC terminations with
foot XYZ disabled. Samples include causal robot history, the oracle latent
target, current expert/achieved `root_qpos`, and a masked 30-frame expert
`root_qpos` lookahead. The first medium planner is trained only on this oracle
data for 10,000 updates and evaluated every 2,000 updates before any DAgger
stage. The canonical MiniLM language descriptions and launcher are documented
in
[`experiments/campaigns/2026-08-05-bones-language10-oracle-pretrain/`](experiments/campaigns/2026-08-05-bones-language10-oracle-pretrain/README.md).

```bash
MODE=smoke experiments/campaigns/2026-08-05-bones-language10-oracle-pretrain/run.sh
MODE=run experiments/campaigns/2026-08-05-bones-language10-oracle-pretrain/run.sh
```

The matched receding-horizon follow-up predicts three ordered H10 latent
commands at each 5 Hz publication and compares fresh-only, exponential overlap,
and clipped/gated overlap execution. It includes both future-publication
(transport-aware) targets and a deliberately stale current-publication-frame
diagnostic, plus the original H1 baseline. All seven rows use the same frozen
root-qpos encoder, tracker, language goals, deterministic action selection,
randomization-without-push protocol, 10k planner budget, and 10-by-100 SONIC
evaluation:

```bash
STAGES=materialize,train experiments/campaigns/2026-08-06-bones-language10-latent-receding/run.sh
STAGES=eval,aggregate experiments/campaigns/2026-08-06-bones-language10-latent-receding/run.sh
```

The completed 10-by-100 grid selects future-publication H3 fresh-only by the
predeclared SR-first rule: 0.401 SONIC SR versus 0.329 for H1. Future
clipped/gated is the quality Pareto point at 0.396 SR and 39.72 mm successful
MPJPE-L, versus 49.82 mm for fresh-only. Current-publication-frame overlap is
rejected. See the campaign README for the full seven-row and per-motion result.

The current H10 tracker is not used for an execute-5/10 Hz row: changing its
publication cadence would violate the tracker training contract. Qualify a
matched execute-5 tracker before adding that cadence comparison.

For the two-stage high-level skill workflow, use the pipeline entrypoint. It
first runs offline DiffSR skill-encoder pretraining, checks
`checkpoints/latest.pt`, then starts low-level IPMD training with
`agent.ipmd.command_source=hl_skill`. Defaults match the LaFAN1 latent setup:
`z_dim=256`, `horizon_steps=25`, `sin_cos` phase features, W&B logging, video
recording, and sparse checkpoints every 100M environment frames.

```bash
pixi run -e isaaclab hl-skill-pipeline
```

Useful local smoke/dry-run forms:

```bash
pixi run -e isaaclab hl-skill-pipeline --dry-run

pixi run -e isaaclab hl-skill-pipeline \
    --pretrain-updates 1 \
    --train-max-iterations 1 \
    --train-num-envs 16 \
    --no-train-video \
    --logger-backend none
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
./scripts/data/download_g1_lafan1_data.sh
```

This downloads the G1 subset into `data/` and then runs the local NPZ + manifest preparation step.
To bake the G1 arms-up alignment trim into the generated NPZ files, pass
`--auto_trim_mode g1_shoulder_roll`.

The underlying Python entrypoint is:

```bash
pixi run -e isaaclab python scripts/data/setup_lafan1_dataset.py \
    --prepare-npz --headless
```

For the G1 retargeted set, the public CSV motions often begin with an arms-up
alignment pose. To bake a per-motion trim into the generated NPZ files, add:

```bash
pixi run -e isaaclab python scripts/data/setup_lafan1_dataset.py \
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
pixi run python scripts/data/write_lafan1_npz_manifest.py \
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
pixi run python scripts/data/write_lafan1_npz_manifest.py \
    --npz_dir data/lafan1/npz/g1 \
    --manifest_path data/lafan1/manifests/g1_debug_manifest.json \
    --select dance1_subject1 dance1_subject2 walk1_subject1
```

### If You Start From CSV Files

Prepare local CSV motions into NPZ plus a manifest with:

```bash
pixi run -e isaaclab python scripts/data/prepare_lafan1_from_csv.py \
    --csv_dir /absolute/path/to/csv_motions \
    --npz_dir /absolute/path/to/data/lafan1/npz/g1 \
    --manifest_path /absolute/path/to/data/lafan1/manifests/g1_lafan1_manifest.json \
    --recursive
```

If you want one replay MP4 per converted motion, add `--record_videos` and `--video_dir`.

To auto-trim the G1 arms-up alignment segment while rebuilding NPZ files, add:

```bash
pixi run -e isaaclab python scripts/data/prepare_lafan1_from_csv.py \
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
pixi run -e isaaclab python scripts/data/prepare_lafan1_from_csv.py \
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
pixi run python scripts/data/setup_g1_lafan1_npz_dataset.py
```

That syncs `npz/g1` from the dataset repo `GeorgiaTech/g1_lafan1_50hz` into:

```text
data/lafan1/npz/g1/
```

Upload mode pushes the same local NPZ tree back to Hugging Face:

```bash
pixi run python scripts/data/setup_g1_lafan1_npz_dataset.py \
    --mode upload --token "$HF_TOKEN"
```

## Playback and smoke tests

### Physics backend smokes and benchmark (Isaac Lab 3.0)

The G1 vanilla task supports both physics backends via a launch-time preset:
`physics=physx` (default) or `physics=newton_mjwarp` (MuJoCo-Warp / Newton).
Routine checks from the repo root:

```bash
# 1-iteration training smokes, one per backend (Dance102 motions)
pixi run -e isaaclab smoke-vanilla-physx
pixi run -e isaaclab smoke-vanilla-newton

# Throughput benchmark across backends; writes JSON + per-run logs
# under logs/benchmarks/. Quick = 1024 envs x 5 iters; full ~10M frames each.
pixi run -e isaaclab bench-backends-quick
pixi run -e isaaclab bench-backends
```

Any train/play command accepts the same `physics=<preset>` override, e.g.
`... python scripts/rlopt/train.py --task Isaac-Imitation-G1-v0 ... physics=newton_mjwarp`.

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
python scripts/viz/compare_policy_reference.py \
    --task Isaac-Imitation-G1-Latent-v0 \
    --algo IPMD \
    --checkpoint /absolute/path/to/checkpoint.pt \
    env.lafan1_manifest_path=./data/lafan1/manifests/g1_lafan1_manifest.json \
    env.refresh_zarr_dataset=False
```

Replay all 40 local G1 LAFAN1 motions from the full manifest:

```bash
python scripts/viz/replay_reference.py \
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
- the playback and replay commands above target the frozen `-G1-v0` / `-Latent-v0`
  ids, which keep the flat `env.lafan1_manifest_path=...` field; on `-G1-v2` and
  later the same setting is `env.data.manifest=...` (see the task list at the top)
- `Isaac-Imitation-G1-LafanTrack-v0` remains available as a legacy alias for the vanilla task
- `replay_reference.py` disables reward and termination terms by default, so long reference videos do not reset early
- pass `--keep_terminations` or `--keep_rewards` if you explicitly want the old RL-style behavior during replay
- `--num_envs 40` is the way to see all 40 loaded trajectories at once; using fewer environments still loads the manifest,
  but only that many trajectories are visible at a time

## Development workflow

This repo is easier to work on with terminal-first tooling than with heavy IDE indexing.

Recommended tools:

- `ruff` for linting and formatting
- `ty` for type and import checking
- `pytest` for focused unit tests

Pixi owns the development tools in `pixi.toml`. Prefer `pixi run` for
non-interactive commands so the repo uses the checked-in environment
definition:

```bash
pixi run ruff check .
```

Useful commands:

```bash
pixi run lint
pixi run format-check
pixi run typecheck
pixi run check
```

RLOpt tests run in the default Pixi environment, which does not install
IsaacLab or `isaaclab_imitation`:

```bash
pixi run test-rlopt
```

Focused pure-Python pytest targets can run directly through the default
environment, for example:

```bash
pixi run pytest RLOpt/tests/test_ipmd_components.py
```

Tests that import Isaac Lab or Omniverse modules need Isaac Sim's Python
bootstrap before imports such as `pxr` are available. Run those tests through
the `isaaclab` Pixi environment:

```bash
pixi run -e isaaclab test-isaaclab
```

For a minimal IPMD training smoke on the Unitree Dance102 manifest:

```bash
pixi run -e isaaclab smoke-ipmd
```

`ty` is configured by [ty.toml](ty.toml) at the repo root, which
mirrors the module-resolution layout previously used by `pyrefly.toml`: it
points ty at the `isaaclab` Pixi environment and includes the import roots for
this repo plus dependency checkouts such as `IsaacLab`, `RLOpt`, and
`ImitationLearningTools`.

For VS Code, prefer the Ruff and ty extensions and terminal-based `ty` checks. Pylance is not the recommended workflow for
this workspace because the Isaac / Omniverse dependency tree is large, generated settings tend to drift, and static
analysis is more reliable here when driven from the checked-in repo configuration.

## Formatting and hooks

A pre-commit configuration is included:

```bash
pixi run pre-commit run --all-files
```

Note that the current hook set is inherited from upstream Isaac Lab conventions. For day-to-day work in this repo,
`ruff` and `ty` are the recommended feedback loop.

## Cluster note

For cluster submission, local Isaac Lab Python installation is not required on the submission machine if jobs run inside
the provided container or Apptainer image. See `docker/cluster` and [REPO_SETUP.md](REPO_SETUP.md) for the expected sync
layout and environment variables.

Cluster jobs submitted through `docker/cluster/cluster_interface.sh job ...` now auto-check the G1 dataset tree before
running the user workload. The container-side preflight in `docker/cluster/run_singularity.sh` verifies that the G1 NPZ
tree under `${CLUSTER_G1_DATA_ROOT:-${CLUSTER_DATA_DIR}/lafan1}` contains at least 40 motions. If the dataset is
incomplete, it downloads the G1 NPZ dataset from Hugging Face with `scripts/data/setup_g1_lafan1_npz_dataset.py` and
regenerates `g1_lafan1_manifest.json` with `scripts/data/write_lafan1_npz_manifest.py` only when the manifest is missing or
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
