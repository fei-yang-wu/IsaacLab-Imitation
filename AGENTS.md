# AGENTS.md

This file defines how coding agents should work in the `IsaacLab-Imitation` workspace.

## Scope

- This guidance is for the top-level `IsaacLab-Imitation` repo only.
- Do not add or maintain agent guidance inside vendored submodules.
- Treat `IsaacLab/`, `RLOpt/`, and `ImitationLearningTools/` as dependencies unless a task explicitly requires changes there.
- Do not edit the vendored submodule at `IsaacLab-Imitation/RLOpt/`; for RLOpt work, use the sibling installed repo at `/home/fwu91/Documents/Research/SkillLearning/RLOpt`.
- The workspace also expects the upstream `IsaacLab`, `unitree_rl_lab`, and `loco-mujoco` repositories to be available either as git submodules in this repo or as sibling checkouts next to it.
- Prefer edits in files owned by this repo, especially:
  - `source/isaaclab_imitation/`
  - `scripts/`
  - `docker/`
  - `README.md`
  - `REPO_SETUP.md`
  - top-level config files such as `pyrefly.toml` and `.pre-commit-config.yaml`

## Environment

- Use the conda environment `SkillLearning` for development, linting, and tests.
- Prefer `conda run -n SkillLearning ...` for non-interactive commands.
- If you need an interactive shell, activate with:

```bash
conda activate SkillLearning
```

- Use `uv` inside that environment for package installation and Python tooling.
- The documented workspace installer is:

```bash
conda run -n SkillLearning ./scripts/install_workspace.sh
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
  submodule pointers, or deciding between vendored submodule and sibling-repo
  edit targets.
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

Run the smallest relevant checks from the repo root, using `SkillLearning`.

General checks:

```bash
conda run -n SkillLearning ruff check .
conda run -n SkillLearning ruff format --check .
conda run -n SkillLearning pyrefly check
```

Run pure-Python pytest targets through the `SkillLearning` environment, not as
a bare `pytest` command. Prefer the smallest relevant target:

```bash
conda run -n SkillLearning pytest source/isaaclab_imitation/test_reference_patch_env.py
```

Tests that import Isaac Lab or Omniverse modules need Isaac Sim's Python
bootstrap before imports such as `pxr` are available. Run those tests through
the Isaac Lab launcher. In the sibling-checkout layout:

```bash
TERM=xterm conda run -n SkillLearning ../IsaacLab/isaaclab.sh -p -m pytest source/isaaclab_imitation/test_reference_patch_env.py
```

If `IsaacLab` is checked out as an in-repo submodule, use
`./IsaacLab/isaaclab.sh` instead.

If you changed formatting intentionally:

```bash
conda run -n SkillLearning ruff format .
```

For workspace setup changes, verify the installer or README commands still match:

```bash
conda run -n SkillLearning ./scripts/install_workspace.sh
```

For environment or training-entry changes, prefer a targeted smoke test over broad execution:

```bash
TERM=xterm conda run -n SkillLearning ../IsaacLab/isaaclab.sh -p scripts/zero_agent.py --task Isaac-Imitation-G1-LafanTrack-v0
```

Use heavier training or playback commands only when the task requires them.

## Submodule Boundary

- Do not “fix” code inside `IsaacLab/`, `RLOpt/`, or `ImitationLearningTools/` as part of routine top-level work.
- For any `rlopt` Python import or code change, assume the sibling repo `../RLOpt` is authoritative and ignore the vendored `IsaacLab-Imitation/RLOpt` submodule.
- If a top-level change depends on submodule behavior, first see whether the issue can be solved from this repo through config, wrappers, scripts, or documentation.
- If a submodule edit is truly required, call it out explicitly in your summary.

## When Updating Docs

- Keep `README.md` and command examples consistent with actual scripts in this repo.
- Prefer absolute clarity about required sibling or submodule checkouts such as `IsaacLab`, `unitree_rl_lab`, and `loco-mujoco`, and document the expected directory layout explicitly.
- When mentioning execution commands, show them from the repository root.
