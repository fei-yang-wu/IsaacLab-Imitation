# Campaigns and release entrypoints

A campaign is experiments/campaigns/YYYY-MM-DD-purpose/ with a README,
campaign.yaml, and thin launchers calling imitation_experiments modules.
Implementation belongs in the shared library.

- Frozen campaign: protocol fixed; append status and provenance rather than
  silently rewriting its launchers.
- Current campaign: explicitly designated in experiments/README.md.
- Release surface: experiments/paper/ entrypoints runnable from the repository
  root, with named constants or conf/ YAML and errors for missing inputs.
- Gate: required audit or preflight for the selected workflow.
- Dry run: validation without submission.
- SCRIPT_INVENTORY.md: launcher classification; PRUNED_SCRIPTS.md: recovery
  references for removed launchers.

The paper contract is wiki/final-paper-experiment-design.md, which supersedes
wiki/causal-interface-paper-plan.md where they differ. Campaigns record the
specific protocol and submission provenance. Never infer a live instruction
from a historical launcher.
