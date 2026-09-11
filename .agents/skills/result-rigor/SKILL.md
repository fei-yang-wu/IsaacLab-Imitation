---
name: result-rigor
description: Audit experiment results, comparison validity, and metric provenance for research reports.
---

# Result checks

Trace each result to the campaign protocol, recorded resolved config, result
JSON, and aggregate/audit manifest. Check actual frames, evaluated population,
completion, and any invalidation window. Verify scheduler state when claiming
a job is currently running or complete. A timed-out segment can belong to a
successfully resumed chain; inspect cumulative_env_frames and the final result.

Keep single seeds, partial grids, unmatched frames, and different reductions
explicit beside the numbers. Causal claims need controlled comparisons and
repeat evidence. Isaac evaluation is stochastic; use protocol-specific repeats.
The project's approximate 15% high-error heuristic is a caution, not a
universal significance test.

Report results without unsolicited rankings or recommendations. Investigate
unclear provenance; ask only when the remaining question requires user intent
or unavailable evidence.

## Metric pitfalls

- Recorded config overrides launcher intent; entrypoints can modify Hydra values.
- Reset-time snapshots are not episode averages. Sampling only at reset can
  make error appear zero; check sampling cadence and reset exclusions.
- Missing body caches can omit MPJPE. Confirm expected metric keys and population.
- Tracking terminations can hide falls. Keep strict SONIC success separate from
  M3 fall-only survival; use the protocol's termination definition.
- Report success rate with success-only MPJPE-L/G and their sample counts.
  Do not mix success-only, all-environment, and full-horizon reductions.
- Planner reports need root-relative error and fall-only survival together.
- Match seeds, frame budgets, board, randomization, starts, episode caps, and
  checkpoint identity. Do not treat goals within one seed as independent seeds.
- Pin stochastic planner inference. For training curves, summarize a fixed
  frame window with spread rather than selecting one sampled W&B point.

Use sonic-success-eval for exact scoring commands and planner-submission-gate
for planner qualification.
