# Paper-facing reproduction entrypoint

This directory is the stable public surface intended for the eventual
open-source release. It contains no diagnostic command styles, recovery jobs,
or historical launchers.

It holds only the guarded entrypoints — [`run.sh`](run.sh), the Phase-4 and
Phase-5 submit plus aggregate scripts, and the release-bundle builder. Shared
implementation, diagnostics, and tests live in the campaign that owns them, so
this surface stays stable as campaigns come and go. See
[`../README.md`](../README.md) for the campaign layout.

Current state: **staging**.

- Phase 4 remains blocked on its corrected-LAFAN1 low-level gate.
- The first Phase-5 three-seed preparation failed; its default data budget and
  storage workflow still need the revisions recorded in
  [`wiki/current-status.md`](../../wiki/current-status.md).
- The final release bundle requires complete audited aggregates from both
  phases.

Accordingly, [`run.sh`](run.sh) reports the intended commands but refuses to
launch them while `PAPER_WORKFLOW_STATE=staging`. Internal development
continues through the canonical guarded scripts in
[`experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/`](../campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/).

The eventual public commands are:

```bash
experiments/paper/run.sh phase4-lafan1
experiments/paper/run.sh phase5-bones-seed
experiments/paper/run.sh bundle \
  --phase4_aggregate /path/to/phase4/aggregate \
  --phase5_aggregate /path/to/phase5/aggregate \
  --output_dir /path/to/new/release
```

Before changing the state to `ready`, require:

1. the current paper protocol and public commands agree;
2. both phase launchers pass their dry runs and fixed gates;
3. both audited aggregates exist;
4. the bundle builder passes without weakened checks;
5. README paths and a clean-checkout reproduction are verified.

The authoritative scope is exactly two planner rows: DiffSR latent commands
and the ten-frame explicit vanilla packet. The direct vanilla tracker is a
low-level ceiling, not a third planner row.

