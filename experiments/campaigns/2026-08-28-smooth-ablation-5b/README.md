# 2026-08-28 — Smoothness ablation at 5B

The 2026-08-26 `action_rate_l2` weight sweep measured that the weight axis
buys acceleration by spending accuracy, and concluded the next attempt should
change the objective. This campaign runs the first objective-level cells, plus
the two 2026-08-28 SONIC parity corrections as isolated variables.

## Base

`diffntp_chunk_h1_ee_wide`'s frozen encoder (round-4 winner), tracker from
scratch to 5B in the `merged64_pen_ramp_5b` regime: `action_rate_l2` -0.03,
SONIC selection with the failure-share ramp 0.8 -> 0.2 over the first 1B.
Two deliberate changes against that arm, carried by every arm here so the
star stays internally matched:

- 256-D command (not 64-D)
- 20,480 envs (not 16,384) — the measured safe H200 step; 24,576 OOM'd the
  Newton graph launch (job 5580202, 2026-08-18 campaign)

Every arm also trains at the corrected `feet_acc` -2.5e-6 (in-tree default
since 2026-08-28).

## Arms (seed 0, W&B group `smooth-ablation-5b`)

| arm | the one variable | question |
|---|---|---|
| `base` | — | the 256-D + env20k + corrected-feet_acc control |
| `energy` | `env.rewards.energy_consumption.weight=-1.0e-4` | SONIC's whole-body mechanical-power penalty, at its released weight, on our stack |
| `sigma` | `agent.ppo.log_std_init=log(0.05)`, clamped `[0.001, 0.5]` | does bounding exploration noise let `action_rate_l2` smooth the MEAN instead of fighting sampling noise? Ours is an unbounded init-1.0 Gaussian; sigma drifted to ~0.16 by 5B in `merged64_pen_ramp_5b` |
| `feetacc_weak` | `env.rewards.feet_acc.weight=-2.5e-7` | isolates the 2026-08-28 10x parity correction (this arm trains at the old wrong value) |
| `ar0` | `env.rewards.action_rate_l2.weight=0.0` | the penalty's own effect at matched frames, schedule, and batch shape — the decomposition `merged64_pen_ramp_5b` asked for |

## Schedule

One 15:59 H200 segment carries the full 5B (`merged64_pen_ramp_5b` finished
5B in 10:43 at 16,384 envs). Segment 2 is a safety resume that PINS the
landed ratio 0.2 (`select_hold`), so a restart never re-runs the ramp — the
ramp keys off `common_step_counter`, which restarts per segment.

## Scoring

Final checkpoints on `bones_testbed4096_v1`, clean and robust, plus the
milestone curve. Two metric requirements beyond the standard row:

1. `tracking_acceleration_distance_mps2` is an acceleration TRACKING ERROR
   (reference-relative), not a smoothness measure. Report it for SONIC
   comparability, but do not attribute smoothness with it alone.
2. Surface the action-space measures (`action_l2`, `action_delta_l2`,
   eval-side `action_rate_l2`) that `evaluate_checkpoint.py` computes —
   no scored board row has ever carried them. The reference-free jerk /
   high-frequency-power metric is still to be added to the evaluator.

## Status

- 2026-08-28: campaign written; all five arms plan clean offline
  (`--skip-preflight`). Not submitted.
