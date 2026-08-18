# AGENTS.md

This file defines how coding agents should work in the `IsaacLab-Imitation` workspace.

Always talk in ASD-STE100 Simplified Technical English. Always read CONTEXT.md files, and use their ubiquitous language

## Scope

- This guidance is for the top-level `IsaacLab-Imitation` repo only.
- Do not add or maintain agent guidance inside dependency submodules.
- Treat `IsaacLab/`, `RLOpt/`, `ImitationLearningTools/`, and `external/Isaac-GR00T/` as dependency submodules unless a task explicitly requires changes there. `external/` holds adjacent upstream code used verbatim, not hard workspace dependencies. `external/Isaac-GR00T/` is upstream NVIDIA code (pinned commit, own `gr00t` Pixi environment); never edit it — adapters live in `RLOpt` and `source/imitation_experiments/`.
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
- The default Pixi environment contains Python 3.12, PyTorch, TensorDict,
  TorchRL, editable `RLOpt`, and editable `ImitationLearningTools`.
- The `isaaclab` Pixi environment adds
  `isaaclab[isaacsim,all]==3.0.0b2.post1` (Isaac Sim 6.0.1) from NVIDIA's PyPI index plus editable
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
- `source/imitation_experiments/`: installable, editable experiment library —
  the shared planner, evaluation, audit, data, and provenance implementation,
  organized as `imitation_experiments.{data,planner,lowlevel,evaluation,audit,
  provenance,pipeline,capacity}` with its tests in
  `source/imitation_experiments/tests/`. Launchers invoke it with
  `python -m imitation_experiments.<subpackage>.<module>`. New shared
  experiment Python goes here with a test, never into a campaign directory.
- `scripts/rlopt/`: RLOpt train, test, and playback entrypoints.
- `scripts/rsl_rl/`: RSL-RL training entrypoints.
- `scripts/data/`, `scripts/audit/`, `scripts/viz/`, `scripts/bench/`:
  standalone dataset-preparation, audit, visualization, and benchmark tools
  (see `scripts/README.md`).
- `scripts/zero_agent.py`, `scripts/random_agent.py`: smoke-test runners.
- `docker/`: container and cluster-related workflows.
- `experiments/README.md`: human-facing pointer to current, historical, and
  paper-facing experiment surfaces.
- `experiments/campaigns/YYYY-MM-DD-short-purpose/`: dated protocol/status
  indexes plus the frozen shell launchers a collaborator invokes. Campaign
  directories are thin: README, configs, and `.sh` wrappers that call the
  library. They contain no Python implementation.
- `experiments/paper/`: stable release-facing entrypoints only — `run.sh`, the
  Phase-4 and Phase-5 submit plus aggregate scripts, the release-bundle
  builder, and the reference-buffer workflow. Do not add exploratory launchers,
  diagnostics, shared implementation, or tests there.
- Every script in `experiments/paper/` must be **self-contained, current, and
  runnable as-is**. Concretely:
  - It runs from the repository root with no undocumented prerequisite step and
    no hand-editing before use.
  - Every configurable parameter lives either inside the script as an explicit
    named constant, or in a config file next to it. Prefer Hydra: a
    `@hydra.main` entrypoint with its YAML under `experiments/paper/conf/`, so
    parameters are discoverable, overridable on the command line, and recorded
    in the run directory. Do not leave required values to be supplied only by
    environment variables or by editing the source.
  - It stays updated when the code it drives changes. A stale script here is a
    defect, not history — move superseded launchers out rather than leaving
    them to rot.
  - It fails loudly on a missing input or an unmet gate instead of silently
    doing less work.
  `reference_buffer_workflow.py` plus `conf/reference_buffer.yaml` is the
  reference implementation of this shape.
- The shared planner modules live in the `imitation_experiments` package and
  import each other absolutely (`from imitation_experiments.planner...`).
  Never mutate `sys.path` under `experiments/` and never rely on a fixed
  `parents[N]` depth to find the repository root: use `REPO_ROOT` and the
  helpers in `source/imitation_experiments/imitation_experiments/paths.py`.
- `logs/`, `outputs/`: generated run artifacts; do not treat them as source.

## Working Rules

- Read `README.md` first when changing setup, training, or execution workflows.
- Read `wiki/context-management.md` before changing agent guidance, updating
  submodule pointers, or deciding which repository owns an edit.
- Keep changes aligned with the existing terminal-first workflow.
- Prefer minimal, targeted edits over broad refactors.
- Be rigorous. Do not treat a preliminary result as a fact. A preliminary
  result is a sign that tells you where to look next. It is not a conclusion,
  and it is not evidence for or against a research claim.
- A result is preliminary until it meets all of these conditions:
  - The protocol is the frozen one for that comparison.
  - The compared arms differ in one variable only.
  - The run is complete. A partial aggregate, an unfinished grid, a cancelled
    job, or a missing cell keeps the result preliminary.
  - The measured difference is larger than the known evaluation noise. Isaac
    evaluation is not deterministic; treat a relative difference below about
    15% in the high-error regime as unresolved.
  - Repeated seeds support the difference.
- Before you cite a stored result, find out how it was produced. Read its
  campaign README, its aggregate manifest, and the status of the jobs that
  made it. An artifact on disk is not proof that its protocol was complete.
- State the qualification with the number, in the same sentence. Say
  "preliminary", "one seed", "partial grid", or "frames not matched" where the
  number appears, not in a later paragraph.
- Do not build an argument, a recommendation, or a paper claim on a
  preliminary result. Say what experiment would settle the question instead.
- Ask the user when the status of a result is unclear. Invoke the
  `result-rigor` skill before citing a stored number, and the
  `experiment-campaign` skill when starting or extending a campaign.
- Before using a newly coined project term, abbreviation, variant label,
  metric shorthand, or overloaded word, define it in plain language first and
  say exactly what changes relative to the baseline. Do not make the user infer
  a term's meaning from code, configuration, or a results table.
- When a coined term matters again in a later turn or conversation, briefly
  restate its meaning before relying on the shorthand. Keep doing so until the
  user has clearly adopted the term or explicitly asks to omit the reminder.
- Avoid committing generated artifacts, caches, checkpoints, or log directories.
- Use W&B tags to identify each run's environment or environments, primary
  change, and other main features.
- Organize W&B runs with concise, functional group names such as
  `planner-ablation`, not names based on timestamps or incidental
  implementation details. Ask the user to confirm the proposed group name
  before launching the run.
- `Isaac-Imitation-G1-v2` changed IN PLACE on 2026-08-04, by explicit decision:
  its rewards are now `G1V2TunedRewardsCfg` (-37.3% MPJPE-G / -34.7% EE-G over
  three seeds against two control seeds, ranges disjoint) and its DiffSR macro
  state is the `root_qpos` frame (qpos + root pose, 380-wide encoder input)
  instead of full-body (670). A v2 checkpoint from before that date needs
  `env.expert_macro_state_terms=[expert_motion,expert_anchor_pos_b,expert_anchor_ori_b]`
  plus its original reward overrides to reproduce; pairing an old encoder with
  the new default fails loudly, never silently. Invoke the
  `g1-encoder-interface` skill before changing or pairing an encoder.
- G1 latent task versioning (2026-07-31 onward): "the default" is always the
  highest-numbered `Isaac-Imitation-G1-vN` id. When the recipe's
  config needs a breaking change, register a new `-G1-vN+1` with the new
  kwargs instead of mutating the existing vN; once superseded, a vN stays
  registered with its exact old kwargs forever (for reproducibility) and
  simply stops being cited as the default. Update the versioning-convention
  comment in `config/g1/__init__.py`'s "Current defaults" section and this
  line when the default moves.
- Unless the user specifies another budget, cluster training jobs should target
  about 10B environment frames per task/run and a two-day SLURM walltime.
- Never shrink a run's frame budget or `max_iterations` to fit a scheduler
  walltime. Submit every segment of a chained run with the full frame target
  and let the walltime end it: sbatch delivers SIGTERM before the kill, the
  trainer writes a final resume checkpoint, and the next segment (chained
  `afterany`, `--checkpoint <tracker tree>`) continues the same global budget
  through the checkpoint's `cumulative_env_frames`. The chain stops at the
  target no matter how the walltime divides it. Write checkpoints to
  persistent storage, never node-local disk.
- Prefer cluster for large training and paper-scale batch evaluation. Prefer the
  local workstation for inference, playback, metric inspection, and video
  rendering because a fresh Isaac Lab container is expensive to initialize on
  each cluster job.
- For simple G1 Dance102 cluster experiments, edit `docker/cluster/.env.cluster`
  and set `CLUSTER_G1_MANIFEST_PATH` to the Dance102 manifest before submitting.
  If that `CLUSTER_G1_MANIFEST_PATH` line is commented out, it means the job is
  using the default 40 trajectories.

## Focused Causal Interface Comparison

- Before changing or running the paper-facing comparison, read
  `wiki/current-status.md` for the living project state, then read
  `wiki/causal-interface-paper-plan.md` and
  `wiki/whole-body-vla-literature-review.md` so named SOTA methods,
  literature-inspired diagnostics, and native reproductions stay distinct.
  Then read
  `wiki/lafan1-from-scratch-comparison.md`. Read
  `wiki/bones-seed-phase5-data-preparation.md` before Phase 5. Keep job IDs
  and chronology in the from-scratch page rather than in this file.
- The direct vanilla tracker receiving a fresh expert command at 50 Hz is the
  low-level ceiling, not a high-level planner row. EE chunks, alternative raw
  command styles, Future-CVAE, and token variants are diagnostics or appendix
  studies unless the user explicitly changes the paper scope. Do not start a
  combinatorial command-style sweep.
- The explicit packet is current plus nine future frames. Its term-major shape
  is `expert_motion=10*58=580`, `anchor_pos=10*3=30`, and
  `anchor_ori=10*6=60`: `[580, 30, 60]`, 670 values total. Re-express
  anchors against the current robot anchor and consume slots 0 through 9 once
  each before per-environment renewal.
- The streamed and direct vanilla paths must use the exact same ordered actor
  inputs and frozen tracker weights. Load only the policy state dict, require a
  strict restore, freeze the module in evaluation mode, and record the
  checkpoint SHA and input-key provenance. A phase-complete, asynchronous
  equivalence certificate covering all actor inputs and actions is mandatory.
- Direct actor command terms and the corresponding critic command entries have
  the same numerical values. They are separate observation groups, and the
  critic may contain additional privileged state. Keep command-side expert
  noise disabled; do not describe command noise as an actor/critic difference.
- Planner inference uses only the causal robot history and explicit task input:
  nine past frames plus current, 93 values per frame, for a `10 x 93`
  observation. Future reference data is allowed only for oracle commands,
  labels, and metrics. Never use
  `current_achieved_macro_transition_batch` as a deployable planner input.
- M3 planner collection and evaluation keep the normal 10-second, 500-control-
  step episode and the frozen random reference-start range 0-200 for both
  interfaces. Do not extend a planner episode to the outer sample-collection
  budget. The outer collector may continue across resets until it has the exact
  row count.
- For definition of success rate, use the same termination as in SONIC.
  No push, keep domain randomization on.
- BONES-SEED oracle demonstrations may be collected in one balanced
  multi-environment run per interface because motion identity is a supervised
  label there. Planner-driven collection and evaluation must still receive an
  explicit goal independent of the live reference rank. Do not choose or
  change the language goal from a trajectory reassignment after reset.
- Publish planner commands on a per-environment renewal schedule. Global
  timestep modulo logic is invalid when environments reset asynchronously.
  Use the same planner backbone, training stages, exact positive sample budget,
  optimizer budget, seed, evaluation starts, and low-level protocol for both
  main rows.
- Local smoke tests and 10M-frame blocks are qualification only. About 50M
  total frames is the maximum useful serious local low-level check, not a
  default target. Do not run a 100M local block. Stop earlier once the code is
  visibly doing what the protocol intends, and do not keep extending local
  training merely to demonstrate convergence. Keep resets,
  rewards, terminations, and other environment details on the frozen protocol
  unless the user explicitly changes it. Use clusters for long convergence,
  final verification, and paper numbers.
- A DiffSR qualification must prove that the selected skill checkpoint's
  `skill_encoder_state_dict` is tensor-identical to the encoder embedded in
  the latent low-level checkpoint. Run
  `validate_latent_skill_checkpoint_binding.py` before Isaac evaluation and
  require the binding record in later planner submission gates. Prefer the
  exact skill checkpoint path recorded by low-level training even when another
  checkpoint happens to contain identical runtime encoder weights. Invoke the
  `planner-submission-gate` skill for the full ordered gate sequence.

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

Run the experiment-library and standalone-script tests in the default
environment after touching `source/imitation_experiments/` or `scripts/`:

```bash
pixi run test-experiments
pixi run test-scripts
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

- Do not “fix” code inside `external/*`, `RLOpt/`, or `ImitationLearningTools/` as part of routine top-level work.
- If a task explicitly requires RLOpt or ImitationLearningTools changes, edit the in-repo submodule and update the top-level submodule pointer.
- If a top-level change depends on submodule behavior, first see whether the issue can be solved from this repo through config, wrappers, scripts, or documentation.
- If a submodule edit is truly required, call it out explicitly in your summary.

## When Updating Docs

- Keep `README.md` and command examples consistent with actual scripts in this repo.
- Prefer absolute clarity about required submodules and optional local dependency checkouts such as `loco-mujoco`, and document the expected directory layout explicitly.
- When mentioning execution commands, show them from the repository root.
