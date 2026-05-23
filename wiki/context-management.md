# Context Management

This repo should act as the orchestration layer for G1 imitation experiments.
It owns Isaac Lab environment wiring, task registration, RLOpt entrypoints,
cluster submission, data manifests, and experiment scripts. Algorithm and data
tooling work is owned by the in-repo dependency submodules, not by sibling
overlays.

The goal of context management here is to help coding agents quickly answer:

- Which repository owns the file I need to change?
- Which context document should I read before editing?
- Which validation command is appropriate for this change?
- When should I update a submodule pointer versus solving the issue in this repo?

## Context Layers

Use these surfaces for different kinds of context:

| Surface | Purpose | Update cadence |
| --- | --- | --- |
| `AGENTS.md` | Durable operating rules for coding agents in this repo. | Rarely |
| `CLAUDE.md` | Claude Code specific command shortcuts and architecture notes. | Occasionally |
| `wiki/` | Longer repo context, status, strategy, and workflow explanations. | Often |
| `README.md` | Human-facing setup and common usage. | When commands or setup change |
| `REPO_SETUP.md` | Submodule, remote, and cluster setup details. | When repo layout changes |
| `.github/instructions/` | Future path-specific GitHub Copilot instructions. | When a path needs stable rules |
| `.github/skills/` or `.agents/skills/` | Future reusable agent workflows with scripts or references. | When a workflow repeats |
| Issues or Projects | Work tracking, experiment decisions, and cross-repo roadmap. | Continuously |

Keep `AGENTS.md` short. It should route an agent to the right files and
validation commands. Put current status, historical reasoning, and experiment
strategy in `wiki/` instead.

## Ownership Boundaries

### Owned by `IsaacLab-Imitation`

Edit these directly when the task is about environment behavior, config wiring,
scripts, documentation, or cluster workflows:

- `source/isaaclab_imitation/`
- `scripts/`
- `docker/`
- `experiments/`
- `data/README.md` and tracked manifests under `data/unitree/`
- `README.md`, `REPO_SETUP.md`, `AGENTS.md`, `CLAUDE.md`, and `wiki/`

Expert-batch sampling and task configuration are repo-owned integration
surfaces because they expose Isaac Lab data and registry wiring to RLOpt.

### Owned by `RLOpt`

Use the in-repo submodule at `./RLOpt` for algorithm/runtime changes:

- IPMD, ASE, PPO, SAC, GAIL, AMP implementations
- latent-command runtime behavior
- reward estimator implementation and cadence
- bilinear model and offline pretraining internals
- RLOpt tests and package metadata

When RLOpt changes are required, make them in `./RLOpt`, push them to the RLOpt
remote, and update the top-level submodule pointer in this repo.

### Owned by `ImitationLearningTools`

Use the in-repo submodule at `./ImitationLearningTools` for reusable data
loading, dataset conversion, replay-buffer construction, and LeRobot streaming
utilities. Push ILT changes to its remote and update the top-level submodule
pointer here.

### Owned by dependencies

Treat these as dependencies unless a task explicitly targets them:

- `IsaacLab/`
- optional sibling `loco-mujoco/` for the Loco-MuJoCo loader

Prefer integration fixes in this repo before changing a dependency.

## Submodule-First Workflow

This repo uses submodules as the active dependency checkouts and the
reproducible experiment snapshots:

- `IsaacLab/`
- `RLOpt/`
- `ImitationLearningTools/`

The practical workflow is:

1. Change algorithm code in `./RLOpt` or reusable data code in
   `./ImitationLearningTools`.
2. Push those submodule commits to their remotes.
3. Run integration from this repo.
4. Update the submodule pointer in this repo so the experiment state is
   self-contained.
5. For cluster jobs, leave local path overlays disabled unless a task explicitly
   needs an unpinned local experiment.

If `CLUSTER_RLOPT_LOCAL_PATH` is commented out, cluster submissions use the
submodule-pinned `RLOpt` state from this repo.

## Agent Startup Flow

For a coding agent starting in this repo:

1. Read `AGENTS.md`.
2. If using Claude Code, read `CLAUDE.md` for command shortcuts.
3. For context or ownership tasks, read this wiki page before patching.
4. For IPMD/inverse-RL representation-learning tasks, read
   `wiki/ipmd-representation-learning.md`.
5. For local/cluster run planning, read `wiki/experiment-workflow.md`.
6. Inspect live files before editing. Do not rely only on previous memory.
7. Determine the owner repo before patching:
   - `IsaacLab-Imitation` for env/config/script/cluster/docs.
   - `./RLOpt` for algorithm runtime.
   - `./ImitationLearningTools` for reusable dataset tooling.
8. If conda commands are needed and the user has not already named an
   environment, ask which environment they prefer. Use `$CONDA_ENV` in reusable
   command examples; `SL` and `SkillLearning` are examples only.
9. Prefer construction-time validation and fail-fast errors. Avoid defensive
   runtime guards in algorithmic hot paths.
10. Run the smallest relevant validation command.

## Where To Put New Context

Use this decision table:

| New context | Put it in |
| --- | --- |
| Stable coding-agent rule | `AGENTS.md` |
| Claude-specific command or caveat | `CLAUDE.md` |
| Repo status, design rationale, experiment strategy | `wiki/*.md` |
| Setup or user command changed | `README.md` or `REPO_SETUP.md` |
| Path-specific instruction for Copilot or other GitHub agents | `.github/instructions/*.instructions.md` |
| Repeatable multi-step workflow | `.github/skills/<name>/SKILL.md` or `.agents/skills/<name>/SKILL.md` |
| Concrete unfinished work | GitHub issue or project item |

Do not put long project history in `AGENTS.md` or `CLAUDE.md`. Long startup
context makes agents slower and increases the chance they follow stale plans.

## External CLI Setup

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

## Future Work

The following sections are non-operative planning notes. Do not treat them as
active instructions until the referenced files exist.

### Candidate GitHub Instructions

If this repo adds GitHub-native instruction files, start with these narrow
surfaces:

- `.github/instructions/rlopt-entrypoints.instructions.md` for `scripts/rlopt/`
  and RLOpt config entrypoint rules.
- `.github/instructions/g1-config.instructions.md` for
  `source/isaaclab_imitation/isaaclab_imitation/tasks/manager_based/imitation/config/g1/`.
- `.github/instructions/cluster.instructions.md` for `docker/cluster/`.
- `.github/instructions/context-docs.instructions.md` for `AGENTS.md`,
  `CLAUDE.md`, and `wiki/`.

Each instruction file should be short and path-specific. Do not duplicate the
entire repo architecture in every file.

### Candidate Agent Skills

Only create skills for workflows that repeat often and need scripts or a strict
checklist:

- `ipmd-smoke-verification`: local G1/IPMD smoke run and log inspection.
- `cluster-submit`: local validation, repo overlay sync, and SLURM job submit.
- `training-log-diagnosis`: inspect `rlopt.log`, W&B metadata, and run folders.
- `context-refresh`: update wiki status after a branch lands.

Skills should own workflows. They should not become another copy of the wiki.

## Validation Map

Pick the smallest relevant check:

| Change type | First check |
| --- | --- |
| Docs only | `git diff --check` |
| Shell scripts | `bash -n <script>` |
| Pure Python helper or env sampling tests | `conda run -n "${CONDA_ENV:-SL}" pytest source/isaaclab_imitation/test_reference_patch_env.py` |
| Isaac Lab imports or runtime env behavior | `TERM=xterm conda run -n "${CONDA_ENV:-SL}" ./IsaacLab/isaaclab.sh -p -m pytest source/isaaclab_imitation/test_reference_patch_env.py` |
| Cluster script behavior | `bash -n docker/cluster/cluster_interface.sh` |
| Training entrypoint or config routing | targeted `scripts/rlopt/train.py` smoke run |

Do not submit cluster jobs until the relevant local check passes.
