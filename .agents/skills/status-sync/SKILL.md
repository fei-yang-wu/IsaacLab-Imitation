---
name: status-sync
description: Record a finished result in the right places and keep them consistent — the campaign README, wiki/current-status.md, wiki/progress-report.md, and the Notion project mirror. Use after a campaign completes, a qualification passes or fails, a job chain fails, a protocol or budget changes, a result is invalidated, or when the user asks to update the status, the progress report, or Notion.
---

# Status sync

Four surfaces record project state. They have different jobs. Update them in
this order, and stop early when the change does not reach the next surface.

| surface | holds | update when |
| --- | --- | --- |
| campaign README | the protocol, the run commands, the arm table, validity windows | every result inside that campaign |
| `wiki/<topic>.md` | job IDs, chronology, and the detailed protocol of one workstream | the workstream moves |
| `wiki/current-status.md` | **where we are now** — the newest decisions and the live state | a meaningful code decision, qualification result, cluster submission, job failure, or paper result |
| `wiki/progress-report.md` | the results-facing summary in three fixed sections | any qualification result, campaign completion, or invalidation |
| Notion project mirror | a human-friendly mirror of `current-status.md` | `current-status.md` changes meaningfully |

## Rules that apply to all of them

- Carry the qualification with the number, in the same sentence:
  "preliminary", "one seed", "partial grid", "frames not matched". See the
  `result-rigor` skill before you write any number down.
- Every table must be backed by a machine-readable artifact
  (`summary.json`, audit JSON, aggregate manifest), never by hand
  transcription.
- An invalidation is a result. Record what became invalid, the date window,
  and what still stands.
- Keep chronology and job IDs in the topic page, not in `current-status.md`.
  `current-status.md` answers "where are we now"; it must not grow without
  bound.
- A historical path in an old page is not a live submission instruction. When
  a page names a retired launcher, either fix it or mark the page as a
  historical log.

## wiki/progress-report.md — the three sections

1. Latent encoder experiments and ablations.
2. Interface design experiments.
3. Hardware — compute-hardware findings and robot / sim2sim readiness.

Update the affected section **and** its "Verified" date. Do not add a fourth
section; put a new workstream in its own wiki page and link it.

## wiki/current-status.md

Update after a meaningful code decision, a qualification result, a cluster
submission, a job failure, or a paper result. Verify changing external state —
Slurm jobs above all — before you treat any line as current:

```bash
pixi run python -m imitation_experiments.pipeline.cluster status --submission <plan_dir>
```

Set "Last verified" to the date you actually checked.

## Notion mirror

The repo wiki stays authoritative; the Notion project page says so explicitly.

- Project page: "Causal Interface Comparison (Latent vs Explicit Planner
  Commands)" — `39f2af4e-204e-81a6-b03d-c5a6f373836e`
- Projects data source: `collection://2522af4e-204e-8130-9e85-000b3a8c0489`
- Tasks data source: `collection://2522af4e-204e-81b5-8c1e-000b14b7a587`
  (the relation property "Project" takes the project page URL; Status is one of
  Not Started / In Progress / Done / Archived)

Update the existing project page content and its Summary, and update or add
tasks in the Tasks data source. Do not create a second project.

The Notion MCP server needs an authorized connector. When it is not
authorized in this session, say so and leave the Notion step for the user
instead of inventing a workaround.

## Navigation surfaces

`experiments/README.md` names the current campaign and separates
release-facing entrypoints from historical launchers. When a campaign becomes
the current one, or stops being current, update that page too.
