# IsaacLab-Imitation Wiki

This wiki holds repo-owned context that is too detailed or changeable for
`AGENTS.md` and `CLAUDE.md`.

Start with:

- [Project Live Status](current-status.md): the living source of truth for the
  current research direction, completed gates, active or failed jobs,
  preliminary evidence, and immediate work queue.
- [Context Management](context-management.md): how coding-agent context should
  be organized across this orchestration repo, dependency submodules, and
  future reusable agent workflows.
- [IPMD Representation Learning](ipmd-representation-learning.md): current
  research focus, ownership boundaries, and methodological constraints for
  representation learning with inverse RL / adversarial reward learning.
- [Language-Conditioned Skill Commander (System 2)](system2-skill-commander.md):
  high-level commander mapping current state plus language goal to a skill code
  by distilling the frozen skill encoder; approved approach, milestone status,
  and grounded code reference map.
- [Closed-Loop Skill Commander Eval](closed-loop-skill-commander-eval.md):
  practical oracle-drive, rollout-finetune, and closed-loop evaluation recipes.
- [Experiment Workflow](experiment-workflow.md): local tests, final cluster job
  submission, and experiment tracking conventions.
- [Causal High-Level Interface Paper Plan](causal-interface-paper-plan.md):
  authoritative paper contract for the focused DiffSR latent versus exact
  streamed-vanilla packet comparison, causal `10 x 93` state, equivalence
  gates, phased execution, and claim limits.
- [Whole-Body VLA and Latent-Action Literature Review](whole-body-vla-literature-review.md):
  primary-source comparison of SONIC, HuMI, WholeBodyVLA, LeVERB, GR00T,
  LAPA, and the planner families that motivate the paper experiments.
- [Fair Interface Baselines](fair-interface-baselines.md): operational local
  runner for the two main planner rows and the separate direct-vanilla
  low-level ceiling; older command variants are documented as diagnostics.
- [LAFAN1 From-Scratch Interface Comparison](lafan1-from-scratch-comparison.md):
  matched low-level training protocol, changing experiment chronology,
  corrected-data requirements, and checkpoint history.
- [BONES-SEED Phase-5 Data Preparation](bones-seed-phase5-data-preparation.md):
  fresh-output language-data workflow, provenance requirements, audits, and
  cache invalidation for the shared language-conditioned experiment.
- [Command-Space Ablation](command-space-ablation.md): historical two-level
  oracle and planner machinery for full-body versus end-effector command
  spaces; use for diagnostics or explicitly scoped appendix work, not the main
  paper grid.
- [LeRobot Offline Pretraining](lerobot-offline-pretraining.md): Unitree WBT
  LeRobot ingestion, TorchRL cache ownership, replay/debug commands, and the
  current RTX re-image note.
- [Isaac Consumer Data Plan](isaac-consumer-data-plan.md): current branch split
  between off-machine action labeling and this repo's Isaac/data-consumer work.

Index every `wiki/*.md` file here so future pages do not become orphaned.
