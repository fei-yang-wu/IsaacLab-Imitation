---
name: result-rigor
description: Decide whether a measured number may be cited as a result, and how to qualify it. Covers the preliminary-until checklist, the evaluation-noise floor, how to trace an artifact back to the protocol that produced it, which file is the authority for a run's real configuration, and the metric traps that produce confident wrong numbers. Use before citing a stored result, writing a claim or a paper row, updating a progress table, comparing two arms, or when the user asks "is this real", "is this significant", or "where did this number come from".
---

# Result rigor

A preliminary result is a sign that tells you where to look next. It is not a
conclusion, and it is not evidence for or against a research claim.

## The preliminary-until checklist

A result stays **preliminary** until every one of these is true:

1. The protocol is the frozen one for that comparison.
2. The compared arms differ in **one** variable only.
3. The run is complete. A partial aggregate, an unfinished grid, a cancelled
   job, or a missing cell keeps the result preliminary.
4. The measured difference is larger than the known evaluation noise.
5. Repeated seeds support the difference.

## The noise floor

Isaac evaluation is not deterministic.

- In the high-error regime, treat a relative difference below about **15%** as
  unresolved.
- Measured planner MPJPE run-to-run spread is 0.2–6.4%.
- Measured low-level evaluation standard deviation is about 12%; use repeats.

A 46.95 against a 46.33 is not a resolved difference. Say so, in the sentence
that states the numbers.

## How to state a qualified number

Put the qualification **with** the number, in the same sentence — not in a
later paragraph.

- Good: "46.95 mm, one seed, 20 episodes per motion, preliminary."
- Bad: "46.95 mm." ... "(All results in this section are preliminary.)"

Do not build an argument, a recommendation, or a paper claim on a preliminary
result. Instead, say which experiment would settle the question.

Ask the user when the status of a result is unclear.

## Trace the artifact before you cite it

An artifact on disk is not proof that its protocol was complete. For each
number you intend to cite:

1. Read the campaign README that owns it — protocol, arm definition, and any
   recorded validity window.
2. Read the aggregate manifest or audit JSON, not a transcribed table.
3. Read the run's own `summary.json` / result JSON, and check the completion
   fields (for a low-level pass: `done_rate == 1.0`, `time_out_rate == 0.0`,
   `stop_reason == "all_envs_done"`, `num_evaluated_envs` equals the request).
4. Check the state of the jobs that produced it:

```bash
pixi run python -m imitation_experiments.pipeline.cluster status --submission <plan_dir>
ssh <ice|skynet> 'sacct -j <jobid> --format=JobID,State,Elapsed,ExitCode -P'
```

A `TIMEOUT` or `CANCELLED` job in the chain means the aggregate is partial
until proven otherwise.

## The launcher is not the authority

Isaac entrypoints re-assign the environment config **after** Hydra, so a flag
on the command line can be silently overridden. Trust the run's recorded
`summary.json` / resolved config, never the launcher script or the shell
history.

Check the frames actually processed as well as the frames requested. For a
chained run, the global count is `cumulative_env_frames` in the checkpoint;
segment-local step numbers are not the budget.

## Metric traps that produce confident wrong numbers

- **Post-reset snapshot.** `CommandTerm.reset` logs the metric buffer as it
  stands, so an instantaneous metric read at termination reports the value at
  termination, not the episode mean — and after a reset it measures reset
  placement noise. Detect it with a `--randomization none` pass.
- **`metric_interval` equal to the episode length** samples once, at step 0,
  on the reset placement, where every tracking error is 0 by construction. Use
  about 10 steps.
- **Saturated survival.** With `foot_pos_xyz` active, episodes end on a foot
  tracking termination before the robot can fall, so survival reads 1.000.
  Planner evaluation uses `--fall_only_success`, which also disables that term.
- **Missing metric.** MPJPE needs `env.data.runtime_cache_body_names=[...]`.
  Without it the metric silently does not appear at all.
- **Unmatched frames.** Two arms at different frame counts are not a
  comparison. Say "frames not matched" when they are not.
- **Unpinned stochastic inference.** The GR00T flow sampler starts from
  `randn`. Two unseeded identical runs already differed enough to be mistaken
  for an effect. Pin the seed.
- **Truncated episodes.** A short step cap that ends a third of the episodes
  produces numbers that cannot be compared with an uncapped run.

## Two metrics, always, for a planner arm

Report root-relative MPJPE **and** fall-only survival. Either metric alone can
invert the ranking: a re-encoded FSQ arm had perfect survival and the worst
MPJPE — a degenerate but safe policy. See the `sonic-success-eval` and
`gr00t-planner` skills for the exact commands.

## Define the term before you use it

Before you use a newly coined project term, abbreviation, variant label, or
metric shorthand, define it in plain language and say exactly what changes
against the baseline. Restate it briefly when it comes back in a later turn,
until the user adopts it or asks you to stop.
