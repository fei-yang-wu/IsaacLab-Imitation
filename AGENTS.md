# AGENTS.md

This file defines how coding agents work in the `IsaacLab-Imitation` workspace.

Always talk in ASD-STE100 Simplified Technical English. Always read CONTEXT.md
files, and use their ubiquitous language. Always follow Google Developer
Documentation Style Guide for langauge style.

Last refreshed 2026-08-30. A rule here is a live constraint. If a rule is only
history, it belongs in `wiki/`, not in this file.

## Scope

- This guidance is for the top-level `IsaacLab-Imitation` repo only. Do not add
  agent guidance inside dependency submodules.
- `IsaacLab/`, `RLOpt/`, `ImitationLearningTools/`, and `external/Isaac-GR00T/`
  are dependency submodules. `external/` holds upstream code used verbatim.
  Never edit `external/Isaac-GR00T/`; adapters live in `RLOpt` and
  `source/imitation_experiments/`.
- For RLOpt or ImitationLearningTools work, use the in-repo submodules at
  `./RLOpt` and `./ImitationLearningTools`, not sibling checkouts.
- `unitree_rl_lab` is not required for training; this repo owns the G1 robot
  configuration and its URDF/mesh assets. `loco-mujoco` is optional and only
  needed for the `loco_mujoco` dataset loader.

## Environment

- Pixi owns the environment. Do not install repo dependencies with `conda`,
  `pip`, or `uv`.
- `pixi run ...` uses the default environment (Python 3.12, PyTorch,
  TensorDict, TorchRL, editable `RLOpt` and `ImitationLearningTools`).
  `pixi run -e isaaclab ...` adds Isaac Sim 6.0.1 / Isaac Lab 3.0.0b2.post1 and
  editable `source/isaaclab_imitation`.
- Run RLOpt tests in the default environment so TorchRL does not import Isaac
  Lab or start Isaac Sim.
- Interactive shells: `pixi shell`, `pixi shell -e isaaclab`.
- `./scripts/install_workspace.sh` is a compatibility wrapper around
  `pixi install`. Prefer `pixi install`, `pixi install -e isaaclab`, or
  `pixi install --all`.
- One Pixi environment prefix per git worktree. A worktree that points at
  another worktree's `.pixi/envs` resolves editable installs to the wrong
  branch. Keep agent worktrees under `.codex/worktrees/` (Codex) or `.claude/`
  (Claude), install with `pixi install --locked`, and never commit them.
  Refresh editable packages only when packaging metadata changes:
  `pixi reinstall rlopt iltools`, or
  `pixi reinstall -e isaaclab rlopt iltools isaaclab-imitation`.

## Repo Shape

- `source/isaaclab_imitation/`: the installable Isaac Lab extension package
  (environments, command terms, MDP terms, task configs).
- `source/imitation_experiments/`: the shared experiment library
  (`data`, `planner`, `lowlevel`, `evaluation`, `audit`, `provenance`,
  `pipeline`, `capacity`, `reporting`) with its tests. Launchers call
  `python -m imitation_experiments.<subpackage>.<module>`. New shared
  experiment Python goes here with a test, never into a campaign directory.
- `scripts/rlopt/`, `scripts/rsl_rl/`: training, evaluation, and playback
  entrypoints. `scripts/data/`, `scripts/audit/`, `scripts/viz/`,
  `scripts/bench/`: standalone tools (see `scripts/README.md`).
- `experiments/campaigns/YYYY-MM-DD-short-purpose/`: dated campaigns. A
  campaign directory is thin: a README, configs, and launchers that call the
  library. No Python implementation.
- `experiments/paper/`: release-facing entrypoints only. Every script there
  must run from the repository root as-is, keep its parameters in named
  constants or in Hydra configs under `experiments/paper/conf/`, stay current
  with the code it drives, and fail loudly on a missing input.
  `reference_buffer_workflow.py` plus `conf/reference_buffer.yaml` is the
  reference shape.
- `logs/`, `outputs/`: generated artifacts. Not source.
- Import shared modules absolutely (`from imitation_experiments.planner...`).
  Never mutate `sys.path` under `experiments/`, and never find the repository
  root by a fixed `parents[N]`; use `REPO_ROOT` and
  `source/imitation_experiments/imitation_experiments/paths.py`.

## Research Rigor

- A preliminary result tells you where to look next. It is not a conclusion.
- A result stays preliminary until all of these hold: the protocol is the
  frozen one for that comparison; the arms differ in one variable; the run is
  complete; the difference is larger than evaluation noise (Isaac evaluation is
  not deterministic — treat a relative difference below about 15% in the
  high-error regime as unresolved); repeated seeds support it.
- Before citing a stored number, find out how it was produced: its campaign
  README, its aggregate manifest, and the state of the jobs that made it. An
  artifact on disk does not prove its protocol completed.
- State the qualification in the same sentence as the number: "preliminary",
  "one seed", "partial grid", "frames not matched".
- Never build a recommendation or a paper claim on a preliminary result. Say
  which experiment would settle it.
- Ask the user when the status of a result is unclear. Invoke the
  `result-rigor` skill before citing a stored number, and the
  `experiment-campaign` skill when starting or extending a campaign.
- Define a new term, abbreviation, variant label, or metric shorthand in plain
  language before using it, and say what changes against the baseline. Restate
  it in later turns until the user adopts it.
- Replace a vague verb with a precise clause, not with a paragraph. Words like
  "runs", "uses", "handles", "involves", or "touches" hide what happened; the
  fix is the specific short statement, not an expansion into file paths.
  Not: "emastack-20b never runs that loss." (vague)
  Not: three sentences naming every entrypoint and config field. (padded)
  Yes: "That is emastack-20b's loss, but it reused the existing pretrained
  encoder."
- Name a setting by its override path when the setting is the subject of the
  sentence, and give its semantics once. Not: "0.1 reset ratio". Yes:
  "`adaptive_uniform_ratio=0.1` — 10% of reset starts are uniform, 90% are
  drawn by per-bin failure rate." Read the code to confirm semantics; do not
  paraphrase a campaign comment. Give the full path
  (`env.command_interface.reference.selection.adaptive_uniform_ratio`) and the
  consuming file the first time, then use the short field name.

## Runs and Budgets

- Unless the user sets another budget, a cluster training run targets about
  10B environment frames.
- Never shrink a run's frame budget or `max_iterations` to fit a walltime.
  Submit every segment with the full frame target and let the walltime end it:
  sbatch sends SIGTERM before the kill, the trainer writes a resume
  checkpoint, and the next segment continues the same global budget through
  `cumulative_env_frames`. ICE allows about 16 hours per GPU job, so a 10B run
  is several chained segments. Write checkpoints to persistent storage, never
  to node-local disk.
- Prefer the cluster for long training and paper-scale batch evaluation. Prefer
  the local workstation for inference, playback, metric inspection, and video
  rendering, because a fresh Isaac Lab container is expensive per job.
- Local runs are for qualification: stop once the code visibly does what the
  protocol intends. Do not extend a local run only to show convergence. Use
  the cluster for convergence, final verification, and paper numbers.
- Keep resets, rewards, terminations, and other environment details on the
  frozen protocol unless the user changes it explicitly.
- Do not commit generated artifacts, caches, checkpoints, or log directories.

## Cluster Submission

- `python -m imitation_experiments.pipeline.cluster` is the control plane:
  `plan` validates, preflights, and freezes a plan; `submit` uploads the
  workspace archive and calls sbatch with the printed `PLAN_SHA`; `status`,
  `logs`, and `cancel` follow the job. Invoke the `cluster-job-submission`
  skill for the full sequence.
- A campaign declares its whole job in `campaign.yaml`, including per-stage
  environment variables. `run_singularity.sh` sources ONLY the frozen
  per-stage env file when it is present, so editing `docker/cluster/.env.cluster`
  does not affect a control-plane job. That file now applies only to manual
  invocations of `run_singularity.sh`.
- `docker/cluster/cluster_interface.sh` and every `submit_job_slurm_*.sh` are
  deprecation shims since 2026-08-15. Do not revive them.
- `submit` packs the working tree, not `git HEAD`. It warns and records
  `drift=true` when the tree is dirty. Commit before submitting when the run
  must be reproducible from a SHA.

## Weights & Biases

- Tag each run with its environment, its primary change, and its main
  features. Use concise functional group names such as `planner-ablation`.
  Ask the user to confirm a group name before launching.
- The control plane pins one W&B run id per `(arm, seed)` output tree. The
  first job of a chain appends a random token to the declared id and writes
  `<output_root>/wandb_run_id`; every resume reads it, so a chained run stays
  one W&B run. To move a chain onto a fresh id, delete that file. W&B refuses
  a run id that was ever deleted, and the refusal kills the job.
- A run id must stay at or below 31 characters. RLOpt adds a
  `logdir:<19-char timestamp>_wandb-<run id>` tag and W&B caps a tag at 64.
- The W&B run name carries the launch timestamp
  (`<exp_name>-YYYY-MM-DD_HH-MM-SS`) so two launches of one arm are
  distinguishable. The id does not.
- Shared mode is retired (2026-08-18). The evaluation sidecar publishes to its
  own companion run in the same group and repeats the trainer's `env_frames`
  key so both runs share one x-axis. Do not reintroduce `WANDB_MODE=shared`,
  `WANDB__PRIMARY`, or `WANDB__LABEL`: shared mode had to be set at run
  creation, could never be added later, refused asynchronously (a sidecar
  dropped every point while looking healthy), and made
  `wandb.log(step=...)` a no-op.

## Contracts You Must Not Silently "Fix"

These look like defects and are not. Changing them changes results or
performance.

- **Terminal observations are batched.** `ImitationRLEnv` publishes
  `extras["final_obs"]` as `{"_env_ids": LongTensor[k], "obs": nested [k, ...]
  tensors}`. The per-environment object array is the legacy format, still
  accepted on read. Restoring the per-environment clone loop costs about 60% of
  collection throughput.
- **The env log payload is detached, not converted.** `IsaacLabWrapper`
  keeps device tensors and lets the trainer convert once per iteration.
  A per-step `.cpu().item()` is a pipeline stall for values that are dropped.
- **`env.data.reference_prefetch_mode`**: `next` overlaps the sequential rows
  with physics and keeps SONIC's reset distribution exact. `next_and_reset`
  also pre-stages reset rows, which makes a reset draw see sampler failure
  weights that are one control step stale. Choose per campaign and record it.
- **Encoder and tracker are one pair.** `Isaac-Imitation-G1-v2` changed in
  place on 2026-08-04: its rewards are `G1V2TunedRewardsCfg` and its DiffSR
  macro state is the `root_qpos` frame (380-wide encoder input) instead of
  full body (670). A v2 checkpoint from before that date needs
  `env.expert_macro_state_terms=[expert_motion,expert_anchor_pos_b,expert_anchor_ori_b]`
  plus its original reward overrides. Invoke the `g1-encoder-interface` skill
  before changing or pairing an encoder.
- **Tracker optimizer geometry**: new campaigns pass
  `--agent rlopt_ipmd_tuned_fullbatch_cfg_entry_point` (full batch, 3 epochs =
  3 optimizer steps per iteration, +18.9% frames per unit wall-clock).
  `rlopt_ipmd_tuned_cfg_entry_point` is FROZEN, not deprecated: it is still the
  only way to reproduce the 46.5B/50B chains and every campaign before
  2026-08-30. Never redirect it. The promotion rests on `mb_full_e3` in
  `2026-08-30-optimizer-ablation-5b` at 3.47B of a 5B budget with no matched
  control, and its half-minibatch sibling at the same 3 epochs COLLAPSED, so
  read the class docstring before citing the recipe.
- **G1 task versioning**: "the default" is the highest-numbered
  `Isaac-Imitation-G1-vN`. A breaking recipe change registers `vN+1`; a
  superseded `vN` keeps its exact old kwargs forever and simply stops being
  cited. Update the versioning comment in `config/g1/__init__.py` and this line
  when the default moves.
- **DiffSR binding**: a qualification must prove the selected skill
  checkpoint's `skill_encoder_state_dict` is tensor-identical to the encoder
  inside the latent tracker checkpoint. Run
  `validate_latent_skill_checkpoint_binding.py` before Isaac evaluation and
  keep the record for planner gates. Invoke the `planner-submission-gate`
  skill for the ordered gate sequence.

## Paper Work

- Read `wiki/final-paper-experiment-design.md` first: it is the current
  paper-facing contract and it supersedes
  `wiki/causal-interface-paper-plan.md` where they disagree. Then read
  `wiki/current-status.md` for live state, and
  `wiki/whole-body-vla-literature-review.md` so named SOTA methods, borrowed
  diagnostics, and native reproductions stay distinct.
  Read `wiki/bones-seed-phase5-data-preparation.md` before Phase 5.
- Planner inference uses only causal robot history plus the explicit task
  input: nine past frames plus current, 93 values per frame, a `10 x 93`
  observation. Future reference data is allowed only for oracle commands,
  labels, and metrics. `current_achieved_macro_transition_batch` is never a
  deployable planner input.
- Publish planner commands on a per-environment renewal schedule. Global
  timestep modulo logic is invalid when environments reset asynchronously.
- Both main planner rows share the backbone, training stages, positive sample
  budget, optimizer budget, seed, evaluation starts, and low-level protocol.
- Planner collection and evaluation keep the 10-second, 500-control-step
  episode and the frozen random reference-start range 0-200 for both
  interfaces. The outer collector may continue across resets to reach an exact
  row count.
- Success rate uses SONIC's termination definition. No push, domain
  randomization on.
- BONES-SEED oracle demonstrations may be collected in one balanced
  multi-environment run per interface. Planner-driven collection and
  evaluation must still receive an explicit goal that does not depend on the
  live reference rank.
- A streamed and a direct path must use the same ordered actor inputs and the
  same frozen tracker weights: load only the policy state dict, require a
  strict restore, freeze in evaluation mode, and record the checkpoint SHA and
  input-key provenance.
- Do not start a combinatorial command-style sweep. EE chunks, alternative raw
  command styles, Future-CVAE, and token variants are diagnostics or appendix
  studies unless the user changes the paper scope.

## Validation

Run validation within the blast radius of the change by default. A full test
suite is not required for every code change. First identify the changed
package, its direct consumers, and the tests that exercise the changed
behavior. Run those targeted tests and report them.

Expand to a full suite only when the change crosses package boundaries,
changes a shared contract, fixture, registration path, dependency or package
metadata, affects many consumers, or when a targeted check fails. Also run the
full relevant suite before a release or a broad integration submission when
the change needs that confidence. A source-code change alone is not a reason
to run every suite.

Examples:

- For an RLOpt algorithm change, run the directly affected RLOpt test files and
  the top-level integration tests for any changed boundary. Use
  `test-rlopt` for shared RLOpt behavior or a cross-package change.
- For `source/imitation_experiments/`, run the affected package tests first.
  Use `test-experiments` when the change affects shared schemas, provenance,
  the control plane, or multiple subpackages.
- For Isaac Lab environment behavior, run the affected environment/config
  tests and a targeted smoke. Use `test-isaaclab` when registration, shared
  environment contracts, or simulator-wide behavior is affected.
- For scripts, shell wrappers, and documentation, run the matching syntax or
  formatting check; do not start an unrelated Python test suite.

Run commands from the repository root.

```bash
pixi run lint
pixi run format-check
pixi run typecheck
pixi run test-rlopt        # RLOpt, default environment
pixi run test-experiments  # source/imitation_experiments
pixi run test-scripts      # standalone scripts
pixi run -e isaaclab test-isaaclab   # tests that import Isaac Lab or pxr
```

`test-rlopt` and `test-scripts` name their test files explicitly in
`pixi.toml`. A new test file does not run until it is added there.

For environment or training-entry changes, prefer a targeted smoke test:

```bash
pixi run -e isaaclab smoke-ipmd
```

If you changed formatting on purpose: `pixi run ruff format .`.

## Submodule Boundary

- Do not fix code inside `external/*`, `RLOpt/`, or `ImitationLearningTools/`
  as part of routine top-level work.
- First check whether the top-level repo can solve the problem through config,
  wrappers, scripts, or documentation.
- If a submodule edit is truly required, edit the in-repo submodule, update the
  top-level submodule pointer, and call the edit out in your summary.

## Documentation

- Read `README.md` before changing setup, training, or execution workflows, and
  keep its commands consistent with the actual scripts.
- Read `wiki/context-management.md` before changing agent guidance, updating a
  submodule pointer, or deciding which repository owns an edit.
- Update `wiki/current-status.md` after a meaningful decision, qualification,
  submission, failure, or paper result. Keep chronology in the phase-specific
  pages so that file does not grow without bound.
- Index every `wiki/*.md` page in `wiki/README.md`.
- Show commands from the repository root.
