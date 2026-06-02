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

- Pixi is the repo-owned environment manager. Do not install repo dependencies
  with `conda`, `pip`, or `uv`.
- Use `pixi run ...` for default-environment commands and
  `pixi run -e isaaclab ...` for Isaac Sim / Isaac Lab workflows.
- The default Pixi environment contains Python 3.11, PyTorch, TensorDict,
  TorchRL, editable `RLOpt`, and editable `ImitationLearningTools`.
- The `isaaclab` Pixi environment adds
  `isaaclab[isaacsim,all]==2.3.2.post1` from NVIDIA's PyPI index plus editable
  `source/isaaclab_imitation`.
- RLOpt tests should run in the default environment so TorchRL does not import
  IsaacLab or initialize Isaac Sim during lightweight testing.
- If you need an interactive shell, use:

```bash
pixi shell
pixi shell -e isaaclab
```

- The documented workspace installer is:

```bash
./scripts/install_workspace.sh
PIXI_ENVIRONMENT=isaaclab ./scripts/install_workspace.sh
```

- The installer is a compatibility wrapper around `pixi install`. Prefer direct
  `pixi install`, `pixi install -e isaaclab`, or `pixi install --all` when
  possible.

## Codex Worktrees

- Codex-created worktrees should live under this repo's `.codex/worktrees/`
  directory. Keep Claude-created worktrees under `.claude/` if that is the
  active Claude workflow.
- For Codex worktree commands, define a workspace-local `CODEX_HOME` from the
  main checkout and use `${CODEX_HOME}/worktrees` as the worktree root:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
export CODEX_HOME="${CODEX_HOME:-${REPO_ROOT}/.codex}"
mkdir -p "${CODEX_HOME}/worktrees"
```

- Create one worktree per task or agent run. Prefer descriptive branch and
  directory names:

```bash
TASK_NAME="ipmd-reward-fix"
git worktree add "${CODEX_HOME}/worktrees/${TASK_NAME}" -b "codex/${TASK_NAME}"
cd "${CODEX_HOME}/worktrees/${TASK_NAME}"
git submodule update --init --recursive
```

- Every new worktree must have its own Pixi environment prefix. Do not point a
  worktree at another worktree's `.pixi/envs`, because editable installs would
  resolve to the wrong branch's `RLOpt`, `ImitationLearningTools`, or
  `source/isaaclab_imitation`.
- Use the locked Pixi environments in each worktree. Pixi reuses the shared
  package cache for heavy packages such as PyTorch, IsaacLab, and Isaac Sim, so
  this creates a separate editable layer without redownloading the world:

```bash
pixi install --locked
pixi run test-rlopt

pixi install --locked -e isaaclab
pixi run -e isaaclab smoke-ipmd
```

- If only local source changed, editable installs are picked up immediately. If
  package metadata, entry points, compiled extensions, or local package wiring
  changed, refresh only the affected editable packages:

```bash
pixi reinstall rlopt iltools
pixi reinstall -e isaaclab rlopt iltools isaaclab-imitation
```

- Do not commit `.codex/worktrees/`, `.pixi/envs/`, generated logs, caches, or
  outputs from worktrees. Commit only the intended source changes from the
  worktree branch.

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

Run the smallest relevant checks from the repo root through Pixi.

General checks:

```bash
pixi run lint
pixi run format-check
pixi run typecheck
```

Run RLOpt pure-Python tests in the default environment, not the `isaaclab`
environment:

```bash
pixi run test-rlopt
```

Tests that import Isaac Lab or Omniverse modules need Isaac Sim's Python
bootstrap before imports such as `pxr` are available. Run those tests through
the `isaaclab` environment:

```bash
pixi run -e isaaclab test-isaaclab
```

If you changed formatting intentionally:

```bash
pixi run ruff format .
```

For workspace setup changes, verify the installer or README commands still match:

```bash
./scripts/install_workspace.sh
```

For environment or training-entry changes, prefer a targeted smoke test over broad execution:

```bash
pixi run -e isaaclab smoke-ipmd
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
