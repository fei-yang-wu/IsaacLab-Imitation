# IsaacLab-Imitation Wiki

Repo-owned context that is too detailed or too changeable for `AGENTS.md`.

Pages are grouped by what they are for. **Live contracts** bind current work.
**Active work** describes systems being built or run now. **Evidence** pages
are the record of finished or superseded work: read them to learn how a number
was produced, not as instructions to follow. Index every `wiki/*.md` page here
so no page becomes orphaned.

## Live contracts

- [Final Paper Experiment Design](final-paper-experiment-design.md): the
  paper-facing contract for the three result sections, the headline
  tracker/planner lock, and the approved claim boundary against SONIC. It
  supersedes the paper plan below where the two disagree.
- [Canonical Paper-Facing Metrics](canonical-paper-metrics.md): the frozen
  definition of a paper row (success rate, success-only micro MPJPE-L,
  MPJPE-G), the boards, and which SONIC number a result may be compared with.
- [SONIC-Compatible Success Evaluation](sonic-success-evaluation.md): the
  checkpoint-evaluation pass behind every reported success rate.
- [Causal High-Level Interface Paper Plan](causal-interface-paper-plan.md):
  provenance, gates, the causal `10 x 93` state, and the frozen evaluation
  machinery. Still authoritative for everything the design page does not
  restate.
- [Results: Interface Ablations](results-interface-ablations.md): the
  publication-facing results section for the 72-arm command-interface study --
  what saturates, what separates, and which orderings the evaluation noise
  leaves unresolved.
- [Interface Ablation Study](interface-ablation-study.md): the setup behind
  those results. Shared training and evaluation protocol, and every arm's
  single changed field with its override path and its 2B row.
- [Latent-Learning Star v2](latent-learning-star-v2.md): the same ablation
  rebased on `diffntp_chunk`, which beat the v1 hub on all three metrics.
  Family spine, code shape, window, cadence; 62 rows, 45 of them untrained.
- [Project Live Status](current-status.md): live research state, gates,
  running and failed jobs, and the immediate work queue.
- [Context Management](context-management.md): how agent context is organized
  across this repo and its submodules, and which repository owns an edit.
- [Experiment Workflow](experiment-workflow.md): local checks, cluster
  submission, and experiment-tracking conventions.

## Active work

- [Linear Closure Problem Statement](linear-closure-problem-statement.md):
  self-contained, shareable statement of the skill-latent linear-closure
  problem — the bilinear score, the closure definition, hard-linear and
  chord-penalty relaxations, and the Q1-Q6 discussion questions.
- [Tracker Pareto Program](tracker-pareto-program.md): plan of record for
  improving SR, MPJPE-L, and MPJPE-G together — lever evidence, the
  pareto-stack campaign design, the feature menu, and the push-termination
  attribution thread.
- [Progress Report](progress-report.md): results-facing summary in three fixed
  sections; update it whenever a result changes.
- [GR00T Planner Deployment](gr00t-planner-deployment.md): batched
  paper-evaluation design and the local native deployment path for the
  language planner.
- [Skill Encoder JEPA Plan](skill-encoder-jepa-plan.md): design-only plan for
  chunk-to-chunk encoding with JEPA plus SIGReg. Nothing measured yet.
- [Skill Encoder JEPA Related Work](skill-encoder-jepa-related-work.md):
  method-level comparison of that plan against latent-action, trajectory-code,
  JEPA, and deployed-latent humanoid systems.
- [Language-Conditioned Skill Commander (System 2)](system2-skill-commander.md):
  the commander that maps state plus language goal to a skill code.
- [BONES Seed Language Planner Memory](bones-seed-language-planner-memory.md):
  durable snapshot of the demo8 merged language-planner experiment.
- [Closed-Loop Skill Commander Eval](closed-loop-skill-commander-eval.md): the
  oracle-drive, rollout-finetune, and M3 evaluation recipe.
- [BONES-SEED Phase-5 Data Preparation](bones-seed-phase5-data-preparation.md):
  language-data workflow, provenance requirements, audits, and cache
  invalidation. Read before Phase 5.
- [Running Experiments Locally](local-experiments.md): the default local
  pipeline, what the defaults resolve to, and the measurement traps.
- [Tracker Runtime v2 Architecture](tracker-runtime-v2-architecture.md): the
  adopted asynchronous stage/mailbox specification for deployed inference.
- [EC Reference-Streaming Oracle Run](ec-reference-streaming-oracle-run.md):
  runbook for the native oracle producer, C++ tracker, and MuJoCo plant.
- [Whole-Body VLA and Latent-Action Literature Review](whole-body-vla-literature-review.md):
  primary-source comparison of SONIC, HuMI, WholeBodyVLA, LeVERB, GR00T, and
  LAPA.
- [IPMD Representation Learning](ipmd-representation-learning.md): research
  focus and methodological constraints for inverse-RL reward learning.

## Evidence and history

- [SONIC Release Checkpoint (Tier 2)](sonic-release-checkpoint-tier2.md):
  running NVIDIA's public SONIC G1 tracker inside our environment, and what it
  scores there.
- [New common eval subset](sonic-v1_1-subsets.md): the frozen 124-rank
  population selected so public `sonic_v1_1` scores on the 23.7 mm scale, with
  its selection contract, confirmation command, and paper claim boundary.
- [Sim2Sim Backend Verification](sim2sim-backend-verification.md): why
  Newton-trained checkpoints collapse under PhysX (2026-08-03).
- [Residual Sim2Sim Dynamics Gap and Randomization](sim2sim-dynamics-gap-and-randomization.md):
  companion analysis and randomization plan (2026-07-21).
- [IsaacLab 3 CU130 Runtime Migration](isaaclab3-cu130-runtime-migration.md):
  migration chronology; its one-shot launchers were pruned.
- [LAFAN1 From-Scratch Interface Comparison](lafan1-from-scratch-comparison.md):
  matched low-level protocol, chronology, and checkpoint history.
- [LAFAN1 Local Training Pipeline](lafan1-local-training.md): the reproducible
  pretrain-then-low-level recipe for the LAFAN1 dataset.
- [LAFAN1 Motion Tracking Evaluation](lafan1-motion-tracking-evaluation.md):
  historical record; its runner and summarizer were pruned.
- [LAFAN1 Latent-Learning Ablation Plan](latent-learning-ablation-plan.md):
  the twelve-arm qualification grid and its H200 follow-up.
- [Ablation Experiment Plan](ablation-experiment-plan.md): archived 2026-07-21
  interface and language-conditioning plan, superseded by the current campaign.
- [Command-Space Ablation](command-space-ablation.md): historical two-level
  machinery for full-body versus end-effector command spaces; diagnostics or
  appendix only.
- [Fair Interface Baselines](fair-interface-baselines.md): local runner for the
  planner rows and the direct-vanilla low-level ceiling. The 50 Hz ceiling row
  itself was dropped by the 2026-08-17 design lock.
- [Embodied-Control Tracker Runtime](embodied-control-tracker-runtime.md):
  the original inference-pipeline design, superseded as the architecture
  reference by tracker runtime v2.
- [Hierarchical Spectral Planning](hierarchical-spectral-planning.md): early
  design for hierarchical planning over spectral skills.
- [Dance102 No-Language Planner Debug](dance102-no-language-planner-debug.md):
  single-trajectory closed-loop debugging run (2026-06).
- [LeRobot Offline Pretraining](lerobot-offline-pretraining.md): Unitree WBT
  LeRobot ingestion and TorchRL cache ownership.
- [Isaac Consumer Data Plan](isaac-consumer-data-plan.md): branch split between
  off-machine action labeling and this repo's data-consumer work.
