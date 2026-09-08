---
name: experiment-campaign
description: Create or extend dated experiment campaigns with reproducible configs, budgets, and launchers.
---

# Campaigns

Use experiments/campaigns/YYYY-MM-DD-purpose/ with README.md, campaign.yaml,
and thin shell launchers. Put shared Python in imitation_experiments with
relevant tests. Start from a campaign with the same workflow, and inspect
the current control-plane schema.

Resolve the arm list, budget, and W&B group from the request and prior context.
Ask about material research choices that remain open. Default training budget
is about 10B environment frames; do not repeat already answered approvals.
Prepare a concrete plan before submission through cluster-job-submission.

campaign.yaml owns stages, resources, paths, and environment variables. Use
named vars for overrides and container-visible paths for args/preflight.
Retain checkpoints on persistent storage and use content-specific output roots.

Keep the full frame target across walltime segments. Resume using
cumulative_env_frames; use afterany for continuation and afterok when downstream
work requires successful pretraining. Verify checkpoint/resume behavior for
the selected trainer rather than assuming a termination signal saved it.

Preserve one W&B run identity per arm/seed across segments via the control
plane's wandb_run_id file. Use a concise group and environment/feature tags.
Use local training for wiring checks; cluster runs establish convergence.

The README records the question, changed variables, protocol, commands, data
and checkpoint identities, output artifacts, and validity windows. Qualify
results and match budgets for comparisons. Append status to submitted
campaigns without silently changing their frozen protocol.
