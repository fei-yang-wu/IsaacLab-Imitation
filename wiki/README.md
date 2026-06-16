# IsaacLab-Imitation Wiki

This wiki holds repo-owned context that is too detailed or changeable for
`AGENTS.md` and `CLAUDE.md`.

Start with:

- [Context Management](context-management.md): how coding-agent context should be
  organized across this orchestration repo, dependency submodules, and future
  reusable agent workflows.
- [IPMD Representation Learning](ipmd-representation-learning.md): current
  research focus, ownership boundaries, and methodological constraints for
  representation learning with inverse RL / adversarial reward learning.
- [Language-Conditioned Skill Generator (System 2)](system2-language-skill-generator.md):
  high-level generator mapping current state + language goal to a skill code by
  distilling the frozen skill encoder; approved approach, milestone status
  (M0 done), and a grounded code reference map.
- [Experiment Workflow](experiment-workflow.md): local tests, full cluster job
  submission, and experiment tracking conventions.
- [LeRobot Offline Pretraining](lerobot-offline-pretraining.md): Unitree WBT
  LeRobot ingestion, TorchRL cache ownership, replay/debug commands, and the
  current RTX re-image note.
- [Isaac Consumer Data Plan](isaac-consumer-data-plan.md): current branch split
  between off-machine action labeling and this repo's Isaac/data-consumer work.
- [Current Status](current-status.md): dated snapshot of the current branch and
  high-value repo context.

Index every `wiki/*.md` file here so future pages do not become orphaned.
