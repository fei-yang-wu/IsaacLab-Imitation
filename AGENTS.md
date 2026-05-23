# AGENTS.md

This file defines how coding agents should work in the `IsaacLab-Imitation` workspace.

## Scope

- This guidance is for the top-level `IsaacLab-Imitation` repo only.
- Do not add or maintain agent guidance inside dependency submodules.
- Treat `IsaacLab/`, `RLOpt/`, and `ImitationLearningTools/` as dependency submodules unless a task explicitly requires changes there.
- For RLOpt or ImitationLearningTools work, use the in-repo submodules at `./RLOpt` and `./ImitationLearningTools`; do not route active work to sibling checkouts.
- `unitree_rl_lab` is not required for normal training; G1 robot configuration and URDF/mesh assets are owned by this repo. `loco-mujoco` is optional and only needed when explicitly selecting the `loco_mujoco` dataset loader.
- Prefer edits in files owned by this repo, especially:
  - `source/isaaclab_imitation/`
  - `scripts/`
  - `docker/`
  - `README.md`
  - `REPO_SETUP.md`
  - top-level config files such as `.pre-commit-config.yaml` and package config files such as `source/isaaclab_imitation/pyproject.toml`

## Environment

- Do not assume a fixed conda environment name. If the user has not specified
  one in the current thread, ask which environment they prefer before running
  conda commands.
- Use `$CONDA_ENV` in reusable commands. `SL` and `SkillLearning` are common
  examples, not requirements.
- Prefer `conda run -n "${CONDA_ENV:-SL}" ...` for non-interactive commands.
- If you need an interactive shell, activate with:

```bash
CONDA_ENV="${CONDA_ENV:-SL}"
conda activate "$CONDA_ENV"
```

- Use `uv` inside that environment for package installation and Python tooling.
- The documented workspace installer is:

```bash
CONDA_ENV="${CONDA_ENV:-SL}"
conda run -n "${CONDA_ENV:-SL}" ./scripts/install_workspace.sh
```

- The installer expects Python `3.11` and installs the local packages plus Isaac Sim / PyTorch dependencies.

## Repo Shape

- `source/isaaclab_imitation/`: installable Isaac Lab extension package for imitation environments.
- `scripts/rlopt/`: RLOpt train, test, and playback entrypoints.
- `scripts/rsl_rl/`: RSL-RL training entrypoints.
- `scripts/zero_agent.py`, `scripts/random_agent.py`: smoke-test runners.
- `docker/`: container and cluster-related workflows.
- `logs/`, `outputs/`: generated run artifacts; do not treat them as source.

## Working Rules

- Read `README.md` first when changing setup, training, or execution workflows.
- Read `wiki/context-management.md` before changing agent guidance, updating
  submodule pointers, or deciding which repository owns an edit.
- Keep changes aligned with the existing terminal-first workflow.
- Prefer minimal, targeted edits over broad refactors.
- Preserve Isaac Lab / Hydra CLI patterns already used in `scripts/`.
- Do not assume IDE-only workflows; command-line verification is the default here.
- Avoid committing generated artifacts, caches, checkpoints, or log directories.
- For IPMD/Bilinear representation-learning work, use the latent task surface
  `Isaac-Imitation-G1-Latent-v0` unless the user explicitly requests vanilla.
  Do not submit `IPMD_BILINEAR` comparison jobs on `Isaac-Imitation-G1-v0`; the
  vanilla bilinear path is debug-only.
- Unless the user specifies another budget, cluster training jobs should target
  about 1B environment frames per task/run and a two-day SLURM walltime.
- For simple G1 Dance102 cluster experiments, edit `docker/cluster/.env.cluster`
  and set `CLUSTER_G1_MANIFEST_PATH` to the Dance102 manifest before submitting.
  If that `CLUSTER_G1_MANIFEST_PATH` line is commented out, it means the job is
  using the default 40 trajectories.

## Validation

Run the smallest relevant checks from the repo root, using the user's selected
conda environment.

General checks:

```bash
CONDA_ENV="${CONDA_ENV:-SL}"
conda run -n "${CONDA_ENV:-SL}" ruff check .
conda run -n "${CONDA_ENV:-SL}" ruff format --check .
conda run -n "${CONDA_ENV:-SL}" pyrefly check
```

Run pure-Python pytest targets through the selected conda environment, not as a
bare `pytest` command. Prefer the smallest relevant target:

```bash
conda run -n "${CONDA_ENV:-SL}" pytest source/isaaclab_imitation/test_reference_patch_env.py
```

Tests that import Isaac Lab or Omniverse modules need Isaac Sim's Python
bootstrap before imports such as `pxr` are available. Run those tests through
the in-repo Isaac Lab submodule launcher:

```bash
TERM=xterm conda run -n "${CONDA_ENV:-SL}" ./IsaacLab/isaaclab.sh -p -m pytest source/isaaclab_imitation/test_reference_patch_env.py
```

If you changed formatting intentionally:

```bash
conda run -n "${CONDA_ENV:-SL}" ruff format .
```

For workspace setup changes, verify the installer or README commands still match:

```bash
conda run -n "${CONDA_ENV:-SL}" ./scripts/install_workspace.sh
```

For environment or training-entry changes, prefer a targeted smoke test over broad execution:

```bash
TERM=xterm conda run -n "${CONDA_ENV:-SL}" ./IsaacLab/isaaclab.sh -p scripts/zero_agent.py --task Isaac-Imitation-G1-LafanTrack-v0
```

Use heavier training or playback commands only when the task requires them.

## Submodule Boundary

- Do not “fix” code inside `IsaacLab/`, `RLOpt/`, or `ImitationLearningTools/` as part of routine top-level work.
- If a task explicitly requires RLOpt or ImitationLearningTools changes, edit the in-repo submodule and update the top-level submodule pointer.
- If a top-level change depends on submodule behavior, first see whether the issue can be solved from this repo through config, wrappers, scripts, or documentation.
- If a submodule edit is truly required, call it out explicitly in your summary.

## When Updating Docs

- Keep `README.md` and command examples consistent with actual scripts in this repo.
- Prefer absolute clarity about required submodules and optional local dependency checkouts such as `loco-mujoco`, and document the expected directory layout explicitly.
- When mentioning execution commands, show them from the repository root.
