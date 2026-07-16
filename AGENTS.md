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
- Prefer Skynet for large training and paper-scale batch evaluation. Prefer the
  local workstation for inference, playback, metric inspection, and video
  rendering because a fresh Isaac Lab container is expensive to initialize on
  each cluster job. Render on Skynet only when video is produced inside an
  already-running training job or local inference is genuinely infeasible.
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
- The main planner comparison has exactly two rows:
  1. DiffSR latent commands published at 5 Hz.
  2. An explicit packet containing ten consecutive vanilla full-body commands,
     published at 5 Hz and consumed slot-by-slot by the same frozen vanilla
     50 Hz tracker.
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
- In M3, disable only the tracking-error terminations `anchor_pos`,
  `anchor_ori`, and `ee_body_pos`. Keep `base_too_low` active. Treat MPJPE,
  root, joint, and EE errors as metrics; define survival as completing the
  episode without `base_too_low`. Low-level oracle qualification remains strict
  and keeps all original termination terms.
- Every low-level oracle evaluation and M3 planner evaluation must also include
  a full-horizon diagnostic pass with all early terminations disabled, including
  `base_too_low`, so MPJPE is measured over the intended evaluation horizon
  rather than a termination-truncated rollout. Render and retain a video from
  that same non-terminating pass for visual inspection. This diagnostic is in
  addition to, not a replacement for, strict oracle qualification and the
  standard M3 survival pass with `base_too_low` active.
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
- The secondary planner-scaling study reports both performance at matched
  actual parameter counts and the smallest tested model reaching a fixed
  survival plus oracle-normalized MPJPE target. Limit it to the current
  flow-matching Transformer, one diffusion chunk Transformer, and one
  deterministic chunk predictor. Use repeated seeds, retain absolute MPJPE,
  keep demonstration-only and rollout-fine-tuned curves separate, and report
  planner latency because equal parameter counts do not imply equal inference
  cost. Do not interpolate through a non-monotonic curve or expand this into a
  broad architecture sweep.
- Do not use legacy scene-grid-offset LAFAN1 data or stale caches for
  paper-facing runs. Audit the manifest with
  `scripts/audit_g1_lafan1_body_frames.py` and preserve data/checkpoint
  hashes.
- Local smoke tests and 10M-frame blocks are qualification only. About 50M
  total frames is the maximum useful serious local low-level check, not a
  default target. Do not run a 100M local block. Stop earlier once the code is
  visibly doing what the protocol intends, and do not keep extending local
  training merely to demonstrate convergence. Keep resets,
  rewards, terminations, and other environment details on the frozen protocol
  unless the user explicitly changes it. Use Skynet for long convergence,
  final verification, and paper numbers.
- The paper-facing LAFAN1 no-language launcher is
  `experiments/interface_baselines/submit_phase4_no_language_skynet.sh`. It is
  fixed to planner seeds `0 1 2`, all 40 corrected motions, and planner sample
  budgets `1k/10k/50k`; do not repurpose it for a wider command or budget
  sweep. It must remain blocked until manifest-bound LAFAN1 oracle audits pass
  for both low-level controllers. Each audit and the equivalence certificate
  must also bind the matching content-specific dataset cache. BONES-SEED
  qualification artifacts cannot satisfy this gate. The launcher refuses a
  stale output root and records the array job and workspace hashes in
  `cluster_submission.json`. Aggregate only a complete passing task grid with
  `aggregate_phase4_no_language_results.py`; it refuses overwrite and emits a
  Markdown table plus `aggregation_manifest.json`. Use
  `experiments/interface_baselines/run_lafan1_diffsr_low_level_skynet.sh` for
  the matched corrected-LAFAN1 latent low-level prerequisite; it intentionally
  submits no EE/full-body variants or paper-facing planner stages. Use
  `experiments/interface_baselines/submit_lafan1_low_level_qualification_skynet.sh`
  for the strict two-controller gate. Do not bypass that gate with a local or
  cross-dataset checkpoint.
- Measure planner inference latency only around the high-level planner's root
  forward call at command publication. Synchronize CUDA, exclude one warmup
  call, and do not include simulator stepping, the low-level policy, metric
  collection, or file I/O. Tiny local latency checks validate instrumentation
  only. Paper summaries require a post-warmup measurement unless every
  environment honestly terminates before the second planner publication;
  retain that failure result and mark its latency as unavailable. If this
  happens on the first control step, temporal-difference metrics are also
  unavailable. Never reject the failure or invent measurements.
- Build the final paper reproducibility index with
  `experiments/interface_baselines/build_paper_release_bundle.py` only after
  both Phase-4 and Phase-5 paper aggregates exist. It must verify the complete
  aggregate and source-artifact hash chains; do not use it with smoke or
  diagnostic outputs and do not weaken its fixed grids to make an incomplete
  release pass.
- Phase 5 uses BONES-SEED language annotations. Prepare corrected data into a
  fresh output tree with recorded input/output hashes and exact commands; do
  not repair source data in place or reuse a cache built from replaced NPZs.
  Require `scripts/audit_bones_seed_phase5.py --require-body-names` to pass
  before a Phase-5 run.
- The provenance-complete 100-motion Phase-5 tree is available inside Skynet
  jobs at `/data/bones_seed_phase5/bones_seed_100`. Use its fresh manifest,
  preparation record, MiniLM table, and the separate fresh latent and vanilla
  Zarr cache paths exactly as listed in
  `wiki/bones-seed-phase5-data-preparation.md`; do not use the older in-place
  100-motion manifest or an environment-default cache for final paper jobs.
- Use `experiments/interface_baselines/run_bones_seed_low_level_skynet.sh` for
  the paired Phase-5 low-level candidate jobs. Its default block is fixed at
  4096 environments, 10,173 iterations (about 1B frames), seed 0, and a two-day
  walltime. Always run it first with `DRY_RUN=1`. It uses one verified workspace
  archive per job and extracts it on compute-local storage because full-tree
  copies can block on Skynet NFS. It must not submit planners;
  evaluate both resulting oracle checkpoints and regenerate the matching
  streamed-vanilla equivalence certificate first.
- After both low-level jobs finish, use
  `experiments/interface_baselines/submit_bones_seed_low_level_qualification_skynet.sh`
  with the three final container-visible checkpoint paths. Its fixed gate is
  100 environments, 1000 steps from frame 0, seed 0, and 0.8 oracle success for
  both controllers. It also certifies all ten streamed-vanilla phases and
  asynchronous renewal. Each audit must record and match its own dataset path:
  `vanilla_seed0` for direct vanilla and `latent_seed0` for DiffSR. Do not
  weaken or bypass `--require_pass`.
- A DiffSR qualification must prove that the selected skill checkpoint's
  `skill_encoder_state_dict` is tensor-identical to the encoder embedded in
  the latent low-level checkpoint. Run
  `validate_latent_skill_checkpoint_binding.py` before Isaac evaluation and
  require the binding record in later planner submission gates. Prefer the
  exact skill checkpoint path recorded by low-level training even when another
  checkpoint happens to contain identical runtime encoder weights.
- Phase 5 uses the same optional language token in the shared planner for both
  main rows. Train one planner per interface across motions, but pass the goal
  name explicitly at deployment and evaluate it against a matching explicit
  motion selection. Never choose the goal embedding from trajectory rank, expert
  history, or the reference cursor. Use
  `experiments/interface_baselines/run_bones_seed_language_smoke.sh` for the
  tiny local code gate; its two rows, one update, and twenty steps are not a
  performance result.
- Use `experiments/interface_baselines/run_bones_seed_multigoal_language_comparison.sh`
  for the shared multi-goal Phase-5 workflow. It must collect balanced rows per
  explicit goal, merge before shared training, evaluate each goal against the
  same named motion, and pass
  `audit_bones_seed_multigoal_language_comparison.py`. The cluster launcher mode
  is `bones-seed-multigoal-language`; it requires a complete fresh preparation
  record, passing vanilla and latent oracle audits for the exact checkpoint and
  manifest and dataset paths, and a matching streamed-vanilla equivalence
  certificate. Every explicit vanilla collection/evaluation command must pass
  `VANILLA_DATASET_PATH`; never rely on the environment default. Dry-run it
  before submission and keep it blocked until those gates pass.
- Paper planner-rollout collection uses ten parallel environments only within
  one explicit goal task. Every environment receives the same language goal,
  is restricted to the same named motion, and contributes to one exact
  per-goal row budget. A goal/reference mismatch must fail immediately. Keep
  pretrained and final closed-loop evaluation at one environment per goal.
- The paper-facing three-seed Skynet entrypoint is
  `experiments/interface_baselines/submit_bones_seed_multiseed_pipeline_skynet.sh`.
  It fixes planner seeds `0 1 2`, preflights all three before submitting the
  first, and calls the guarded single-seed launcher
  `submit_bones_seed_multigoal_pipeline_skynet.sh`. Each seed uses the
  dependency chain `prepare -> rollout array -> finetune -> final-eval array ->
  summarize`. The launcher refuses an existing output root by default; permit
  one only for an intentional audited resume. Each paper run must retain
  `cluster_submission.json` with the workspace archive hash and all five Slurm
  job IDs. Array indices are explicit goal indices; never infer them from
  reference rank. Keep the default dry run, all 100 goals, and chunked sample
  writing. Do not run it while qualification is merely pending or running.
- A user-approved preliminary exception was submitted on 2026-07-15 only to
  obtain early DiffSR Phase-5 planner behavior before qualification. Use
  `submit_bones_seed_diffsr_preliminary_skynet.sh` only for explicitly labeled
  `latent_skill` diagnostics. Its outputs must record
  `preliminary_unqualified=true`, cannot run the paper audit, cannot satisfy a
  qualification gate, and cannot enter the paired or multi-seed paper
  aggregate. The first chain covers ten goals at seed 0 under jobs
  `3506446 -> 3506447 -> 3506448 -> 3506449 -> 3506450`. This exception does
  not authorize an unqualified explicit baseline or the guarded 100-goal,
  three-seed paper launch.
- After at least three complete paper-protocol seeds, use
  `experiments/interface_baselines/aggregate_bones_seed_multiseed_results.py`.
  It fixes the exact seed set `0 1 2` by default and must reject failed audits,
  smoke runs, duplicate or substituted seeds, changed stage
  artifacts, and runs with different protocols, source hashes, data, or
  checkpoints. Report paired latent-minus-explicit differences by goal within
  seed; do not pool all goal rows as if they were independent training seeds.
  Use its generated `multiseed_results.md` as the paper-table draft; do not
  manually transcribe unaudited numbers into a separate table. It refuses to
  overwrite an existing aggregation directory and writes
  `aggregation_manifest.json` with hashes for every aggregate output.
- Keep the full paper metric set in final summaries: success, survival,
  root-relative MPJPE, root, joint and EE errors, action change, and velocity
  and acceleration errors. Record planner parameters and output bandwidth, and
  normalize tracking success by the matching qualified low-level oracle. The
  existing G1 interval-push event must be recorded and identical for both
  interfaces; do not alter it solely for evaluation.
- Define M3 survival identically for both planners: `base_too_low` means a fall;
  `time_out` and `reference_finished` are successful episode ends. Tracking
  errors do not terminate M3 and remain continuous metrics. Preserve
  per-environment termination causes and report both demonstration-only and
  rollout-finetuned planner results with exact unique sample counts. Keep the
  older strict definition only for low-level oracle qualification.

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
