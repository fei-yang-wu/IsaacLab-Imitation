---
name: status-sync
description: Update project status, campaign records, or the Notion mirror when requested or when research state changes.
---

# Status records

Update only surfaces affected by the work:

- Campaign README: protocol, commands, artifacts, validity windows.
- Topic wiki: detailed decisions, job IDs, chronology.
- wiki/current-status.md: concise current state and links.
- wiki/progress-report.md: results summary in its existing three sections
  (latent encoders, interface design, hardware).
- experiments/README.md: when the current campaign changes.

Use artifact-backed results with inline qualifications. Record invalidations
as well as successes. Set verification dates only for what was actually
checked; inspect live scheduler state before claiming current job status.

## Notion, when included in the request

The wiki is authoritative. Update the existing project rather than creating
another:

- Project: 39f2af4e-204e-81a6-b03d-c5a6f373836e
- Projects source: collection://2522af4e-204e-8130-9e85-000b3a8c0489
- Tasks source: collection://2522af4e-204e-81b5-8c1e-000b14b7a587

Fetch the page and schema before editing. Preserve unrelated content; update
Summary and associated tasks as needed. If connector access is unavailable,
report that limitation and complete the local documentation.
