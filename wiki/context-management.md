# Context management

Keep instructions short and close to the decisions they support.

| Surface | Purpose |
| --- | --- |
| AGENTS.md | Repository operating rules and key contracts |
| CONTEXT.md | Shared terms; directory files add local concepts |
| README.md | Setup and common commands |
| wiki/current-status.md | Current research state with links |
| Topic and campaign pages | Protocol, history, results, and provenance |
| .agents/skills/ | Specialist workflows and reusable helpers |
| User-level Codex AGENTS.md | Personal working style across projects |

Read root context and the relevant directory context before edits. Consult
specific topic pages as needed; do not load every wiki or skill at startup.
Check live code before treating a dated note as a current default.

## Ownership

IsaacLab-Imitation owns environment integration, task registration, experiment
orchestration, and entrypoints. Shared experiment code belongs in
source/imitation_experiments/.

RLOpt owns algorithms and pretraining internals. ImitationLearningTools owns
reusable dataset tooling. Use the in-repo submodules. Prefer an integration fix
here when it solves the problem; when a dependency change is needed, retain its
commit and update the parent pointer. Do not edit external/Isaac-GR00T.

## Maintenance

Store a rule once. Keep transient job state, benchmark values, recipe
promotions, and debugging history out of startup instructions. Use code and
campaign records as the authority for exact recipes and configuration.

Keep skill descriptions narrow enough to avoid triggering unrelated tasks.
Skills should add non-obvious procedures, not generic advice or another copy
of AGENTS.md. Retire superseded aliases and redundant style packages.
Installed plugin skills are managed by their package manager.

Update the affected campaign or topic record after meaningful work. Update
current-status when the project state changes; update the progress report
when its results summary changes. Notion synchronization is a separate task
when requested. Index new wiki pages in wiki/README.md.

Validate in proportion to the change: git diff --check for documentation,
bash -n for shell scripts, and affected tests for code. Expand checks for
shared contracts, integration changes, or failures. Use Pixi environments
and commands from AGENTS.md.
