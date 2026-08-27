# CONTEXT.md — Repository Bounded Context

This file defines the repository-wide ubiquitous language. Read it before you
read code. Use these terms with these exact meanings in code, documents, and
discussion. Each owned directory has its own `CONTEXT.md` with more local
terms:

- [`source/isaaclab_imitation/CONTEXT.md`](source/isaaclab_imitation/CONTEXT.md)
- [`source/imitation_experiments/CONTEXT.md`](source/imitation_experiments/CONTEXT.md)
- [`scripts/CONTEXT.md`](scripts/CONTEXT.md)
- [`experiments/CONTEXT.md`](experiments/CONTEXT.md)
- [`docker/CONTEXT.md`](docker/CONTEXT.md)

Rules for these files:

- Keep each file short. Define terms; do not tell history. History goes in
  `wiki/`.
- A term has one meaning. If a meaning changes, update the definition here in
  the same change.
- Before you coin a new project term, add it to the correct `CONTEXT.md`.

## What this repository is

`IsaacLab-Imitation` is the orchestration layer for G1 humanoid imitation
experiments. It owns Isaac Lab environment wiring, task registration, RLOpt
entrypoints, experiment scripts, and cluster submission. Algorithms live in
the `RLOpt` submodule. Reusable data tooling lives in the
`ImitationLearningTools` submodule. See `wiki/context-management.md` for
ownership boundaries.

## Core domain language

- **G1** — the Unitree G1 humanoid robot. This repo owns its configuration
  and URDF/mesh assets.
- **Low-level policy / tracker** — the 50 Hz RL policy that outputs joint
  actions to track a commanded motion.
- **High-level planner / planner** — the model that publishes commands to the
  frozen tracker at 5 Hz. It sees only causal robot history plus an explicit
  task input.
- **Oracle** — a frozen tracker driven by fresh expert reference commands at
  50 Hz. It is the low-level ceiling, not a planner row.
- **Reference** — the dataset-backed expert motion the environment scores
  against. Rewards, terminations, and MPJPE are always measured against it.
- **Command interface** — what the actor receives. Exactly one of: latent,
  explicit (vanilla), or chunk (packet). See the env `CONTEXT.md`.
- **DiffSR** — the diffusion-based skill representation. Its encoder turns
  reference windows into latent skill commands; "latent" rows use it.
- **Explicit packet** — ten consecutive vanilla full-body commands published
  at 5 Hz and consumed slot-by-slot: `[580, 30, 60]`, 670 values.
- **Two-row comparison** — the paper's main planner comparison: DiffSR latent
  at 5 Hz versus the explicit packet at 5 Hz, same frozen vanilla tracker.
- **root_qpos frame** — the 38-D per-frame state: joint positions plus root
  pose. The v2 DiffSR macro state (380-wide encoder input, 10 frames).
- **Macro state / macro transition** — planner-timescale state built from
  stacked frames; contrast with the per-step 50 Hz observation.
- **M3** — planner-driven closed-loop evaluation protocol: 10 s episodes,
  tracking-error terminations disabled, `base_too_low` active.
- **Survival** — an M3 episode that ends without `base_too_low`. `time_out`
  and `reference_finished` are successful ends.
- **Qualification** — the strict oracle gate (all original terminations,
  fixed success threshold) that must pass before planner submission.
- **Equivalence certificate** — the recorded proof that the streamed-vanilla
  and direct-vanilla paths feed identical ordered actor inputs to identical
  frozen weights.
- **Manifest** — the file that lists the NPZ motions a dataset build uses.
  Audits bind checkpoints to a manifest path and hash.
- **LAFAN1** — the motion-capture dataset for the no-language comparison
  (Phase 4).
- **BONES-SEED** — the SONIC-derived G1 motion set with language annotations
  (Phase 5). "Selected-ten" / "language10" is the ten-motion local
  development subset.
- **Phase 4 / Phase 5** — the paper stages: Phase 4 is LAFAN1 no-language;
  Phase 5 is BONES-SEED language-conditioned.
- **SONIC** — the upstream whole-body tracking formulation this project
  reproduces and extends (FSQ commands, tracking terminations).
- **New common eval subset** — the frozen 124-rank in-distribution population
  used to score every local tracker under one protocol and to anchor external
  baseline context through SONIC. Its artifact ID is
  `sonic_capability124_v1`. It was calibrated from SONIC results, so it is not
  held out or unbiased.
- **JEPA** — ONLY the LeWorldModel/LeJEPA-style well-posed objective:
  deterministic transition modeling in token space with one online encoder on
  both branches and SIGReg as the anti-collapse constraint. No EMA copy, no
  stop-gradient. A prediction term whose target comes from an EMA encoder
  copy is NOT JEPA (user decision, 2026-08-23).
- **EMA trick** — the lagged self-target stabilization (momentum encoder +
  stop-grad, BYOL family). An optimization-dynamics trick, not a well-posed
  loss; categorize it with training tricks. The production `sigreg_ebm`
  recipe reads: DiffSR endpoint grounding + EMA token-prediction trick +
  SIGReg. Historical `jepa_*` arm and run names predate this convention and
  keep their spelling.
- **SIGReg** — LeJEPA's sketched isotropic-Gaussian regularizer
  (Epps-Pulley): forces every 1-D projection of the token batch toward
  standard normal, preventing collapse without negatives.
- **IPMD** — the RLOpt policy-mirror-descent algorithm family used for
  low-level training.
- **IPMD-L2T** — the IPMD learning-to-teach variant. On the latent G1 task, a
  privileged teacher controls rollouts from the explicit reference command,
  while a deployable student receives the DiffSR latent command and learns the
  teacher's executed actions.
- **Latent task IDs** — `Isaac-Imitation-G1-vN`. The default is always the
  highest N; superseded versions stay registered unchanged. See the env
  `CONTEXT.md`.

## Infrastructure language

- **Pixi** — the only environment manager. `pixi run ...` for the default
  environment; `pixi run -e isaaclab ...` for Isaac Sim workflows.
- **Skynet** — the SLURM cluster for large training and paper-scale batch
  evaluation.
- **ICE** — the Georgia Tech PACE ICE cluster (H200 nodes, 300 GB storage
  cap, node-local `/tmp`). Write checkpoints to `/data`, never node-local.
- **Workspace archive** — the verified tar of this repo that a cluster job
  extracts on compute-local storage; its hash is recorded in
  `cluster_submission.json`.
- **Smoke test** — a tiny run that gates code correctness only. It is never
  a performance result.
