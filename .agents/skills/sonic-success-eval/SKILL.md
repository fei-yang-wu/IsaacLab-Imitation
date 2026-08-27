---
name: sonic-success-eval
description: Evaluate IsaacLab-Imitation low-level G1 checkpoints with the official SONIC motion-success criterion and success-only MPJPE-L, pick the right board and randomization profile for a paper-facing row, and apply the repo's planner metric standard (root-relative MPJPE plus fall-only survival). Use when the user asks for SONIC success rate, paper-compatible SR, push-disabled or foot-position-disabled evaluation, checkpoint comparison by completed motions, MPJPE-L computed only on successful trajectories, which SONIC paper number a result may be compared against, or which two numbers a planner evaluation must report.
---

# SONIC success evaluation

Run a deterministic-policy, randomized-environment, push-disabled, full-clip
checkpoint evaluation with SONIC's released evaluation thresholds. Keep this
pass distinct from task-strict qualification and the non-terminating
diagnostic.

Read `wiki/canonical-paper-metrics.md` first: it fixes which board, which
randomization profile, and which SONIC paper number a row may be compared
against. Then read `wiki/sonic-success-evaluation.md` for the pinned upstream
links and result-field definitions.

## Reconstruct the checkpoint contract

Inspect the training launch, saved environment/agent config, and checkpoint
metadata. Preserve all of the following:

- task id and physics backend;
- algorithm and `--agent_entry_point`;
- actor command width and encoder interface;
- exact encoder checkpoint, horizon, latent mode, phase mode, and
  frozen/finetuned state;
- dataset or reference-array path plus content identity;
- tracked body order.

Invoke `g1-encoder-interface` before pairing an encoder when the interface is
not already proven. Never rebuild an old checkpoint from today's task defaults.
Change only the evaluation settings below.

## Apply the SONIC criterion at launch

Use `imitation_experiments.lowlevel.evaluate_checkpoint` with deterministic
policy actions (`--action_sampling mode`), start frame 0, and a fixed
sequential assignment.

**Choose the randomization profile by what the row is for** (frozen
2026-08-17):

- `--randomization none` for a paper-facing **quality** row
  (`sonic_sr_clean_v1`). This is the headline. Randomization costs the released
  SONIC checkpoint 2.75 mm of MPJPE-L and 0.001 of success rate, so a quality
  number measured under randomization understates the tracker.
- `--randomization no_push` for the **robustness** partner (`sonic_sr_v1`),
  which retains startup and reset randomization while removing the interval
  push. Every scoreboard row recorded before 2026-08-17 is one of these.

Never place a clean row and a `no_push` row in the same table column. Pass
`env.events.push_robot=null` explicitly as an auditable second guard. Apply
these overrides exactly:

```text
env.events.push_robot=null
env.terminations.anchor_pos.params.threshold=0.25
env.terminations.anchor_pos.params.down_threshold=0.25
env.terminations.anchor_ori.params.threshold=1.0
env.terminations.ee_body_pos.params.threshold=0.25
env.terminations.ee_body_pos.params.down_threshold=0.25
env.terminations.foot_pos_xyz=null
env.terminations.base_too_low=null
```

These mean:

- fail when pelvis Z error exceeds 0.25 m;
- fail when any ankle or wrist Z error exceeds 0.25 m;
- fail when full pelvis orientation error exceeds 1 rad;
- do not fail on foot XYZ error;
- retain startup and reset randomization;
- apply no interval velocity push;
- use the deterministic/mode policy action.

Setting both adaptive position thresholds to 0.25 disables the crouching
relaxation without swapping the task's predicate. The orientation predicate
compares squared angular error, so a threshold of 1.0 is an angle limit of
1 rad.

## Run the complete motions

Use enough `--steps` for the longest evaluated clip. Prefer the maximum clip
length plus one; a deliberately generous cap is fine because the evaluator
stops when all environments are done. Never report an SR from a run whose
`stop_reason` is `max_steps` or whose `done_rate` is below 1.

For the standard large local pass, use 4096 environments. When comparing
checkpoints, keep `num_envs`, seed, reset schedule, start frame, data identity,
and trajectory-rank assignment identical.

Use the command template in `wiki/sonic-success-evaluation.md`; add the exact
checkpoint-specific agent and data overrides after the fixed CLI arguments.

## Validate before reporting

Require all of these:

- `aggregate.num_evaluated_envs` equals the requested environment count;
- `aggregate.done_rate == 1.0`;
- `aggregate.time_out_rate == 0.0`;
- `metadata.action_sampling == "mode"`;
- `metadata.randomization_profile` matches the intended row (`none` for a
  quality row, `no_push` for a robustness row), and `randomization_kept` agrees
  with it on all three of `startup`, `reset`, and `push`;
- `metadata.push_perturbation.enabled == false`;
- `stop_reason == "all_envs_done"`;
- no non-SONIC failure term is active;
- checkpoint comparisons have the same ordered trajectory-rank hash.

Report:

- SR from `aggregate.completed_tracking_success_rate`;
- numerator from `successful_metrics.tracking_mpjpe_mm.num_successful_envs`;
- denominator from `aggregate.num_evaluated_envs`;
- MPJPE-L from `successful_metrics.tracking_mpjpe_mm.mean` (frame-weighted
  micro mean over successful episodes);
- MPJPE-G from `successful_metrics.mpjpe_g_mm.micro_mean`. It is mandatory:
  MPJPE-L flatters a policy that holds its pose while drifting.

`python -m imitation_experiments.evaluation.summarize_paper_boards <json>`
prints the three-number row and refuses to print an incomplete one. Add
`--ranks_json` to restrict a row to an explicit rank subset of a larger run.

Do not report `aggregate.tracking_success_rate` as SR unless every evaluated
environment finished. It intentionally exposes survivor status and can count
unfinished clips under a short cap. A clip that reaches `reference_finished`
on the same step as a failure remains a failure.

## Score on the canonical testbed

New work goes on `bones_testbed4096_v1` (`paper_testbed4096_v1` clean,
`paper_testbed4096_robust_v1` for the robustness partner). Ranks come from
`imitation_experiments.evaluation.protocol.TESTBED4096_RANKS`, never from a
copied literal.

The legacy block (ranks 12288-16383) stays registered only so pre-2026-08-17
rows remain interpretable. Never put a legacy row and a testbed row in one
table column: they are different populations, and the testbed is about 2.85 mm
harder at matched randomization.

Released-SONIC calibration on the testbed: clean **SR 0.9912 / 28.75 mm /
MPJPE-G 135.73 mm**; `no_push` **0.9905 / 31.06 mm / 192.93 mm**.

## Say which SONIC number the row may be compared against

SONIC's headline **22.3 mm at 100% success is its 123-clip hardware deployment
set scored in simulation**. That set is never enumerated (Figure S2 has no
names or IDs), and SONIC's project page shows it deploying squatting,
kneeling, hand crawling and elbow crawling on real hardware. **Do not compare
anything in this repo against 22.3 mm**, and do not build a board that excludes
crouch or ground motion in order to approach it — a 2026-08-17 attempt to do
exactly that was deleted, because the filter selected for ease, not
deployability.

Compare a 4,096-clip row against SONIC's large held-out rows instead:
test-content **98.7% / 23.2 mm**, test-repetition 99.6%, PHUMA 97.0%. State in
the same sentence that the populations differ.

No board in this repo is held out from training: every tracker trains on the
full 129,785-clip tree with no rank filter. Never call a board "held out".

## Preserve the other evaluation passes

This pass does not replace:

- task-strict oracle qualification with the checkpoint's frozen task terms;
- the full-horizon diagnostic with all early terminations disabled and its
  required retained video;
- the planner metric standard below.

Label artifacts explicitly as `sonic_eval` to prevent accidental aggregation
with either protocol.

## Planner metric standard (2026-08-12 onward)

A **planner** evaluation is a different pass from the SONIC criterion above,
and it reports **two** numbers, always:

1. **Root-relative MPJPE (mm)** — the headline tracking metric.
2. **Success = survival = the robot did not fall** (`base_too_low` only).
   A tracking-error termination is not a failure.

Either number alone can invert the ranking of two arms. A re-encoded FSQ
planner arm had perfect survival (1.000, zero falls) and the worst MPJPE
(102.2 against 57.1 mm) — a degenerate but safe policy. This is the case that
makes both numbers mandatory.

Command shape:

```bash
pixi run -e isaaclab python scripts/rlopt/eval_skill_commander_closed_loop.py \
    --fall_only_success --disable_tracking_terminations \
    --metric_interval 10 \
    'env.data.runtime_cache_body_names=[...14 tracked bodies...]' ...
```

- `--fall_only_success` also disables `foot_pos_xyz`, which is **not** in
  `TRACKING_TERMINATION_NAMES`. Without it a large share of episodes end on a
  foot-tracking termination before the robot can fall, so survival reads a
  saturated 1.000 and the step count measures that termination. Measured on
  the 30-goal set: with `foot_pos_xyz` on, 0/30 falls and 317 mean steps; with
  it off, 2/30 falls and 406 mean steps.
- MPJPE needs `env.data.runtime_cache_body_names=[...]` passed to the
  evaluation, or the metric silently does not appear at all.
- `metric_interval` must be far below the episode length. Setting it to the
  episode length samples once, at step 0, on the reset placement, where every
  tracking error is 0 by construction. Use about 10 steps.
- Do **not** import the MPJPE helper from
  `imitation_experiments.lowlevel.evaluate_checkpoint`. That module is a script
  with module-level argparse; importing it mid-run re-parses `sys.argv` and
  aborts the evaluation with a misleading "--checkpoint is required".
  `_tracking_mpjpe_mm` in the closed-loop script computes it from the
  environment accessors.
- Normalize against the matching tracker oracle. On the frozen 4,096-motion
  board: `root_qpos_explicit` 19.21 mm, `z256_scaled` 23.27 mm, `fsq64_sonic`
  25.44 mm.

See the `gr00t-planner` skill for the full planner protocol (2000-step cap,
`physics=newton_mjwarp`, explicit goal) and `result-rigor` before citing any
of these numbers.
