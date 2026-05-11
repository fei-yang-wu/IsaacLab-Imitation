# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repo.

## Scope

Work in repo-owned files:
- `source/isaaclab_imitation/` — installable Isaac Lab extension
- `scripts/` — training, playback, data prep entrypoints
- `docker/` — cluster and container workflows
- Top-level configs: `pyrefly.toml`, `.pre-commit-config.yaml`

Treat `IsaacLab/`, `RLOpt/`, `ImitationLearningTools/` as **read-only dependencies**. Don't fix code inside them unless task explicitly requires; prefer wrappers, config, or docs instead.

For RLOpt: don't edit vendored submodule at `IsaacLab-Imitation/RLOpt/`. Use installed sibling repo at `/home/fwu91/Documents/Research/SkillLearning/RLOpt` as authoritative `rlopt` codebase. Don't add path overrides forcing submodule copy.

Before editing agent guidance or ownership rules, read
`wiki/context-management.md`. Keep `AGENTS.md` and `CLAUDE.md` short; put
longer status, rationale, and strategy in `wiki/`.

## Environment

Always use `SkillLearning` conda env:

```bash
conda run -n SkillLearning <command>
# or activate interactively:
conda activate SkillLearning
```

## Common commands

**Linting and formatting:**
```bash
conda run -n SkillLearning ruff check .
conda run -n SkillLearning ruff format --check .
conda run -n SkillLearning pyrefly check
# Apply formatting:
conda run -n SkillLearning ruff format .
```

**Smoke test (fast, no GPU training):**
```bash
conda run -n SkillLearning python scripts/zero_agent.py \
    --task Isaac-Imitation-G1-v0 \
    env.lafan1_manifest_path=./data/lafan1/manifests/g1_lafan1_manifest.json
```

**Run training (RLOpt IPMD on latent task — recommended):**
```bash
conda run -n SkillLearning python scripts/rlopt/train.py \
    --task Isaac-Imitation-G1-Latent-v0 \
    --algo IPMD \
    --headless \
    env.lafan1_manifest_path=./data/lafan1/manifests/g1_lafan1_manifest.json
```

**Quick single-motion run (dance102):**
```bash
conda run -n SkillLearning python scripts/rlopt/train.py \
    --task Isaac-Imitation-G1-Latent-v0 \
    --algo IPMD \
    --headless \
    env.lafan1_manifest_path=./data/unitree/manifests/g1_unitree_dance102_manifest.json
```

**Play back a checkpoint:**
```bash
conda run -n SkillLearning python scripts/rlopt/play.py \
    --task Isaac-Imitation-G1-v0 \
    --checkpoint /path/to/checkpoint.pt \
    env.lafan1_manifest_path=./data/lafan1/manifests/g1_lafan1_manifest.json
```

Add `env.refresh_zarr_dataset=False` to reuse cached Zarr dataset instead of rebuilding on startup.

For IPMD/Bilinear representation-learning work, use
`Isaac-Imitation-G1-Latent-v0` unless the user explicitly asks for a vanilla
debug run. Do not submit `IPMD_BILINEAR` comparison jobs on
`Isaac-Imitation-G1-v0`; that surface is not the trusted comparison path.
Unless specified otherwise, cluster training should target about 1B environment
frames per task/run with a two-day SLURM walltime.

## Architecture overview

### Package layout (`source/isaaclab_imitation/`)

```
isaaclab_imitation/
  envs/
    imitation_rl_env.py   # ImitationRLEnv — thin subclass of ManagerBasedRLEnv
    rlopt.py              # IsaacLabWrapper (GymWrapper for TorchRL/RLOpt), RLOpt config imports
  tasks/manager_based/imitation/
    imitation_env_cfg.py  # Base ManagerBasedRLEnvCfg with scene, observations, rewards, terminations
    lafan1_manifest.py    # Manifest loading and path normalization utilities
    mdp/                  # MDP terms: observations, rewards, events, terminations
    config/g1/
      imitation_g1_env_cfg.py        # G1 vanilla tracking env config (Isaac-Imitation-G1-v0)
      imitation_g1_latent_env_cfg.py # G1 latent-conditioned env config (Isaac-Imitation-G1-Latent-v0)
      agents/                        # Per-algo agent configs (PPO, SAC, IPMD, ASE, AMP, GAIL…)
      __init__.py                    # gym.register() calls for all task IDs
```

### Training flow

1. `scripts/rlopt/train.py` parses `--task` and `--algo`, loads registered env config and corresponding `rlopt_<algo>_cfg_entry_point` from gym registry.
2. `ImitationRLEnv` (subclassing `ManagerBasedRLEnv`) instantiated with env config.
3. `IsaacLabWrapper` wraps for TorchRL. RLOpt drives training loop.
4. Motion data loaded from JSON manifest (`env.lafan1_manifest_path`), normalized by `lafan1_manifest.py`, cached as Zarr dataset at startup.

### Task IDs

| ID | Config | Notes |
|----|--------|-------|
| `Isaac-Imitation-G1-v0` | `ImitationG1LafanTrackEnvCfg` | Vanilla motion tracking |
| `Isaac-Imitation-G1-Latent-v0` | `ImitationG1LatentEnvCfg` | Latent-conditioned (ASE/IPMD) |
| `Isaac-Imitation-G1-LafanTrack-v0` | same as v0 | Legacy alias, prefer v0 |

### Data flow

Motion data in `data/` (gitignored except manifests):
- `data/lafan1/npz/g1/` — converted NPZ files (30 Hz input, 50 Hz output)
- `data/lafan1/manifests/g1_lafan1_manifest.json` — local manifest (not tracked)
- `data/unitree/manifests/g1_unitree_dance102_manifest.json` — tracked, for quick tests
- `source/isaaclab_imitation/isaaclab_imitation/manifests/g1_lafan1_manifest.template.json` — tracked template

Each manifest entry requires `path` (or `file`) and `input_fps` fields.

## Key conventions

- **Hydra overrides**: pass config overrides as positional CLI args after script flags, e.g. `env.lafan1_manifest_path=...`, `ipmd.use_latent_command=False`.
- **`--task`** selects registered gym env; **`--algo`** selects RLOpt agent config via `rlopt_<algo>_cfg_entry_point`.
- **Type checking**: `pyrefly.toml` at repo root configures search paths for `pyrefly`. Don't modify `pyrightconfig.json` or VS Code Pylance settings.
- **Logs**: written under `logs/`; `outputs/` holds Hydra outputs. Both gitignored.
- **Cluster jobs**: managed via `docker/cluster/cluster_interface.sh`. See `REPO_SETUP.md` and `docker/cluster/.env.cluster` for env var config.
