# Working in IsaacLab-Imitation

Write as a concise technical colleague. Investigate the existing system before
asking questions; ask when an unresolved intent changes the work materially.

## Context and ownership

Read root CONTEXT.md and the context for the directory you change. Read README.md
for setup or execution changes. Consult the wiki for the relevant workstream,
not as a mandatory startup reading list. Verify dated claims against live code.

This repo owns environment wiring, experiment orchestration, and entrypoints.
Shared experiment code belongs in source/imitation_experiments/ with relevant
tests. Campaign directories contain configs, documentation, and thin launchers.
Use absolute imports and imitation_experiments.paths.REPO_ROOT.

Algorithms belong in the in-repo RLOpt submodule; reusable data tools belong in
ImitationLearningTools. Prefer top-level integration fixes when appropriate.
Preserve unrelated changes. If a submodule edit is needed, report it and update
the parent pointer after its commit is available. Never edit
external/Isaac-GR00T; keep adapters in owned code.

## Environment and validation

Use Pixi, not pip, uv, or conda, for repository dependencies. Run commands from
the repository root:

- Default environment: pixi run ...; use it for RLOpt and pure-Python tests.
- Isaac environment: pixi run -e isaaclab ....
- Give each worktree its own environment; install with pixi install --locked.
  Keep agent worktrees under .codex/worktrees/ or .claude/.
- Run affected tests first. Broaden for shared contracts, cross-package changes,
  failures, or release verification. Documentation needs git diff --check;
  shell changes need bash -n. Do not run an unrelated full suite.
- Available suites: test-rlopt, test-experiments, test-scripts, and
  -e isaaclab test-isaaclab. Environment/entrypoint changes may need smoke-ipmd.
  test-rlopt and test-scripts list files explicitly in pixi.toml.

## Experiments and results

Keep the requested protocol, budget, and comparison intact. Use clusters for
long training and large evaluations; local training is for qualification.
The default cluster training budget is about 10B environment frames. Segment
by walltime without shrinking the total; retain resume checkpoints on persistent
storage and account for cumulative_env_frames.

Use python -m imitation_experiments.pipeline.cluster for plan, submit, status,
logs, and cancel. Inspect the concrete plan before authorized submission.
campaign.yaml owns job settings; frozen stage env files override manual env
files. Submission packs the working tree. Do not revive retired shell launchers.
Use the cluster-job-submission skill for operational details.

Trace results to their campaign, recorded config, artifacts, and completion
evidence. Qualify partial grids, unmatched frames, and single seeds beside the
numbers. Separate observations from causal claims; use matched protocols and
repeats for conclusions. Report data without an unsolicited ranking or
recommendation. The result-rigor skill covers detailed metric checks.

## Contracts

- Keep historical task registrations and agent recipes reproducible. Breaking
  recipe changes get a new vN; inspect registration for the current default.
- Bind encoder and tracker by weights and interface, not dimensions alone.
  Preserve the selected checkpoint's original observation/reward contract.
- Keep batched final_obs and detached device-tensor logs; do not add per-step
  Python clone loops or CPU synchronization.
- Record reference_prefetch_mode: next_and_reset makes reset weights one
  control step stale; next preserves the sequential reset distribution.
- Planner inputs are causal robot history plus explicit task input. Future
  references are only for oracle commands, labels, and metrics. Publish on
  per-environment renewal schedules because resets are asynchronous.
- Streamed/direct equivalence requires identical ordered actor inputs and
  strictly restored, frozen policy weights.

For paper work, start with wiki/final-paper-experiment-design.md and the
specific campaign. Keep protocol details there, not in this file.
Update the relevant campaign/topic record after meaningful work; keep
wiki/current-status.md concise and index new wiki pages in wiki/README.md.
Do not commit logs, datasets, caches, checkpoints, or generated artifacts.
