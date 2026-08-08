# CONTEXT.md — `experiments/` (campaigns and release surface)

Bounded context: the human-facing experiment surface. Implementation lives
in `source/imitation_experiments/`; this tree holds protocol front doors.
Read the repository root [`CONTEXT.md`](../CONTEXT.md) first, then
[`README.md`](README.md) for the current-work table.

## Ubiquitous language

- **Campaign** — a dated directory `campaigns/YYYY-MM-DD-short-purpose/`
  that freezes one experiment protocol: a README, configs, and shell
  launchers. Thin by rule: no Python implementation.
- **Frozen campaign** — a campaign whose protocol is fixed. Append-only:
  status and provenance updates are allowed; launchers are never silently
  rewritten.
- **Current** — an explicit decision recorded in the `README.md` table,
  not whichever directory sorts last.
- **Release surface** — `paper/`: the small, stable set of paper-facing
  entrypoints (submit, aggregate, release-bundle, reference-buffer
  workflow). Nothing exploratory lands here.
- **Paper script standard** — every script in `paper/` is self-contained,
  current, and runnable as-is from the repo root: parameters are named
  constants or Hydra YAML under `paper/conf/`, it fails loudly on missing
  inputs or unmet gates, and a stale script is a defect.
  `reference_buffer_workflow.py` plus `conf/reference_buffer.yaml` is the
  reference shape.
- **Gate** — a precondition a launcher enforces before submission:
  passing oracle audits, the equivalence certificate, encoder binding, a
  fresh preparation record, a non-stale output root. Never weaken or
  bypass a gate to make a run pass.
- **Dry run** — the default preflight mode of guarded launchers
  (`DRY_RUN=1`). Always run it first.
- **Script inventory** — `SCRIPT_INVENTORY.md` classifies every live file;
  `PRUNED_SCRIPTS.md` records removed paths and their Git recovery.

## Rules

- Never mutate `sys.path` here and never locate the repo root by a fixed
  `parents[N]` depth; use `imitation_experiments.paths`.
- Launchers call the library with
  `python -m imitation_experiments.<subpackage>.<module>`.
- Paper protocol authority is `wiki/causal-interface-paper-plan.md` and the
  AGENTS.md comparison rules; a campaign README records how one instance
  of that protocol was run.
- Retain `cluster_submission.json` (workspace hash, job IDs) for every
  paper run.

## Validation

```bash
bash -n <launcher>.sh
pixi run test-experiments
```
