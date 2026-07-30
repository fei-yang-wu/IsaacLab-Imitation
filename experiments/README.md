# Experiments

Start here before running anything under `experiments/`.

The complete live-file classification is in
[`SCRIPT_INVENTORY.md`](SCRIPT_INVENTORY.md). The 64 paths removed in the
2026-07-23 cleanup and their recovery instructions are in
[`PRUNED_SCRIPTS.md`](PRUNED_SCRIPTS.md).

The repository organizes everything by campaign:

- `campaigns/YYYY-MM-DD-short-name/` records when a concrete experiment
  campaign was frozen and gives collaborators one front door for that work.
- Each campaign owns its implementation in topic-named group subdirectories,
  so a script's path states which protocol froze it and when.
- `paper/` is the one exception: a small, stable set of release-facing
  entrypoints that must not move as campaigns come and go.

A group directory has exactly one home, in the newest campaign that uses it;
other campaigns reference it rather than copying it. This keeps chronology
visible without creating several drifting versions of the same launcher.

## Current work

| Priority | Campaign | State |
| --- | --- | --- |
| Primary | [`2026-07-29-sonic-official-fsq`](campaigns/2026-07-29-sonic-official-fsq/README.md) | Official-window SONIC low-level reproduction with the sample-efficient `[0, 200]` reset sampler: one 64-D normalized FSQ command (32 levels per coordinate) recomputed from current + nine future frames every 50 Hz step. ICE H200 job `5549500` submitted 2026-07-29; first 1,259,962,368-frame segment is pending resources under a 5B cumulative cap. |
| Primary | [`2026-07-29-latent-holdout-horizon`](campaigns/2026-07-29-latent-holdout-horizon/README.md) | Command-interface ablation at fixed latent space: one shared frozen h10 encoder, hold in {5, 1} against the hold=10 control curve. Submitted to ICE 2026-07-29 as `5548369` (hold=5) and `5548370` (hold=1), one 16h segment each, no resume chain. |
| Primary (local, preliminary) | [`2026-07-23-bones-phase5-language-local10`](campaigns/2026-07-23-bones-phase5-language-local10/README.md) | Local latent-only ten-goal Phase-5 run on the workstation: one shared language-conditioned planner, demonstration-pretrained then rollout-finetuned, 150+150 rows per goal, 500-step episodes. No scheduler involved. |
| Primary (preliminary) | [`2026-07-23-bones-phase5-language-h200`](campaigns/2026-07-23-bones-phase5-language-h200/README.md) | Prepared as a guarded, latent-only H200 pilot: ten shared language goals, 150 demonstration plus 150 planner-rollout rows per goal. Dry-run is ready; no Slurm jobs submitted in this campaign yet. |
| Primary | [`2026-07-22-latent-learning-ablation`](campaigns/2026-07-22-latent-learning-ablation/README.md) | All twelve local 10M qualification arms passed; the H200 submission remains deliberately gated and was not submitted as of the recorded 2026-07-22 status. |
| Supporting | [`2026-07-22-bones-h10-scale`](campaigns/2026-07-22-bones-h10-scale/README.md) | The wall-clock screen selected the H200 16,384 x 12 profile used by the latent-ablation campaign. |
| Previous | [`2026-07-21-interface-scale`](campaigns/2026-07-21-interface-scale/README.md) | Historical tracker-scale campaign. Read its result and data-loss notes; do not treat its launchers as the current submission surface. |

“Current” is an explicit decision in this file, not whichever directory sorts
last. Update this table when the project changes focus.

Changing scheduler state is recorded in
[`wiki/current-status.md`](../wiki/current-status.md). Query the scheduler
before treating any dated job state as live.

## Paper-facing surface

[`paper/`](paper/README.md) is reserved for the stable, public reproduction
entrypoint. It is intentionally marked as staging while Phase 4 and Phase 5
remain incomplete. Exploratory launchers, one-off recovery scripts, and
diagnostics do not belong there.

The authoritative two-row paper protocol remains
[`wiki/causal-interface-paper-plan.md`](../wiki/causal-interface-paper-plan.md).

## Directory map

| Path | Purpose |
| --- | --- |
| `campaigns/` | Dated experiment decisions, status, wrappers, **and** the implementation each campaign owns. |
| `campaigns/<date>-<slug>/<group>/` | Implementation grouped by topic inside the campaign that owns it. |
| `paper/` | Stable release-facing entrypoints only; no implementation, diagnostics, or tests. |

Implementation lives in a date-stamped subdirectory of the campaign that owns
it, grouped by topic. There are no top-level topical directories: a bare
`interface_baselines/` gave no signal about which protocol it belonged to or
whether it was still current.

| Group directory | Owning campaign | Contents |
| --- | --- | --- |
| `interface_baselines/` | `2026-07-23-bones-phase5-language-local10` | Shared causal-planner, qualification, audit, summarization, guarded submission, and the tests for all of it. |
| `command_space_ablation/` | `2026-07-23-bones-phase5-language-local10` | Two shared low-level qualification helpers still called by current paper workflows. |
| `interface_baselines/` | `2026-07-23-lafan1-planner-capacity` | Capacity-scaling aggregation and its tests. |
| `latent_ablation/` | `2026-07-22-latent-learning-ablation` | Latent-learning objective and bottleneck ablations. |

A group directory sits in the newest campaign that uses it, so shared code has
exactly one home. Older campaigns reference it rather than copying it. Because
the shared planner modules import each other as bare siblings, the coupled set
must stay in one directory; only genuinely standalone modules may live
elsewhere.

`paper/` holds the guarded entrypoints and nothing else: `run.sh`, the Phase-4
and Phase-5 submit plus aggregate scripts, and the release-bundle builder. The
tests that span this boundary reach the paper modules through
`campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/conftest.py`.

Every retained script must have a row in `SCRIPT_INVENTORY.md`. Completed
one-off launchers do not remain beside live code merely as an archive: their
paths and reasons are recorded in `PRUNED_SCRIPTS.md`, while their source stays
recoverable from Git history.

Never submit a script solely because it exists. Begin with a dated campaign or
the paper front door, inspect its recorded status, and dry-run any guarded
launcher before allowing scheduler mutation.

## Adding a campaign

Use an ISO date plus a descriptive slug:

```text
experiments/campaigns/2026-07-23-short-purpose/
```

The date is the protocol-freeze or first-submission date, not every rerun date.
Each campaign must contain a `README.md` that records:

- research question and scope;
- current status and last verified date;
- canonical launcher paths;
- frozen data, checkpoint, and protocol identities;
- dry-run or local qualification command;
- submission command and safety gate;
- result/job pointers;
- superseding campaign, when applicable.

Keep the campaign directory itself thin: a `README.md` plus the wrappers a
collaborator invokes. Put shared Python, shell, tests, and configuration in a
topic-named group subdirectory such as `interface_baselines/`. If the code you
need already exists in another campaign's group directory, reference it there
rather than copying it. After a real submission, append status and provenance;
do not silently rewrite the frozen protocol.
