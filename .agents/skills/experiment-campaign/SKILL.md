---
name: experiment-campaign
description: Create and run a new dated experiment campaign in this repo — directory shape, campaign.yaml, thin submit wrapper, README contract, frame budget and walltime segmentation, and W&B naming. Use when the user asks to start a new campaign, add an ablation arm, set up a grid or sweep, scale an arm up, resubmit a chain, or asks where a launcher for an experiment should live.
---

# Experiment campaign

One campaign is one dated directory:

```
experiments/campaigns/YYYY-MM-DD-short-purpose/
    README.md          # protocol, status, and the result table
    campaign.yaml      # arms, stages, resources, dataset paths, preflight
    submit.sh          # thin wrapper that only PLANS
    <other>.sh         # optional thin wrappers for collect / eval
```

**A campaign directory holds no Python.** Shared implementation goes in
`source/imitation_experiments/imitation_experiments/<subpackage>/` with a
test, and the campaign calls it with
`python -m imitation_experiments.<subpackage>.<module>`. A `.py` file inside a
campaign directory is a defect.

Reference shape to copy:
`experiments/campaigns/2026-08-14-latent-quant-ice-repeats/`.

## Before you create one

Confirm with the user:

1. The **arm list** and the single variable each arm changes.
2. The **frame budget** per arm. Default, unless the user says otherwise:
   about 10B environment frames per task/run.
3. The **W&B group name**. Use a concise functional name such as
   `planner-ablation` or `latent-bottleneck-10b`. Never a timestamp or an
   incidental implementation detail. Ask before you launch.

## campaign.yaml

`campaign.yaml` is the single declaration of arms, resources, dataset paths,
and preflight requirements. See the `cluster-job-submission` skill for the
schema, the four CLI verbs, and the per-cluster profile table.

Rules that decide whether the campaign is repeatable:

- Every configurable value is a named `vars` field with a sane default, and is
  overridable with `--set vars.key=value`. Never hand-edit the YAML for a
  one-off run.
- Every path in `args` and `preflight` is a **container-visible** absolute
  path. Preflight maps it to the real remote path and fails the plan when the
  path is not visible under the job binds.
- Checkpoints and pretrain output go to persistent storage (`/data` bind),
  never to node-local disk. A Slurm TIMEOUT wipes node-local output.
- Shared blocks use YAML anchors. The `mul`, `ceil_div`, `floor_div`, and
  `concat` resolvers are registered in `pipeline.cluster.config`.

## Frame budget and walltime — do not confuse them

**Never shrink a run's frame budget or `max_iterations` to fit a scheduler
walltime.** Submit every segment of a chained run with the **full** frame
target and let the walltime end it.

The mechanism:

- `slurm.py` emits `#SBATCH --signal=TERM@300`; the trainer routes SIGTERM
  through the SIGINT handler and writes a final resume checkpoint at the
  current global step, then re-raises so the interrupted run exits nonzero.
- Checkpoints carry `cumulative_env_frames`. The next segment seeds
  `frames_processed` from it, trims the remaining budget, and offsets the log
  and save cadence. The chain therefore stops at the target no matter how the
  walltime divides it.
- Chain the segments with `dependency_kind: afterany`, because a TIMEOUT
  predecessor must still release its successor. A `pretrain -> lowlevel1`
  edge stays `afterok`.
- A segment that finds the budget already complete runs 0 iterations and exits
  cleanly. This is harmless.
- `--only-stage` accepts a comma-separated list, so a lowlevel chain can run
  against an encoder that is already on disk.

W&B continuity across segments: set `WANDB_RUN_ID: <campaign>-<arm>-s<seed>`
on every chained stage and use one segment-less `exp_name`. Put the segment
identity in the run tags. Without a run id, each segment starts a new run at
step 0.

## Local first

Local smoke and 10M-frame blocks are qualification only. About 50M frames is
the maximum useful serious local low-level check. Do not run a 100M local
block, and do not extend local training only to show convergence. Use the
cluster for long convergence, final verification, and paper numbers.

Prefer the local workstation for inference, playback, metric inspection, and
video rendering, because a fresh Isaac Lab container is expensive to start on
each cluster job.

## The README contract

The campaign README is the human-facing protocol and status snapshot. It must
carry:

- What the campaign compares, and the one variable each arm changes.
- The exact run commands, copied from a real invocation.
- Data identity: dataset path, manifest, and content hash or persist id.
- A result table backed by machine-readable artifacts (`summary.json`, audit
  JSON), never by hand transcription.
- Every **validity window**. When a launcher bug made a set of arms
  non-comparable, write the date window and say which seeds may not be
  aggregated. The 08-14 campaign's LayerNorm window is the worked example.
- The qualification of every number, in the same sentence as the number:
  "preliminary", "one seed", "partial grid", "frames not matched". See the
  `result-rigor` skill.

## Tags

Use W&B tags to name each run's environment or environments, its primary
change, and its other main features.

## Related skills

- `cluster-job-submission` — the plan/submit/status/logs/cancel CLI and the
  campaign.yaml schema.
- `result-rigor` — when a campaign result may be cited as a result.
- `planner-submission-gate` — the audits a planner campaign must pass before
  submission.
- `status-sync` — where a finished campaign's result is recorded.
