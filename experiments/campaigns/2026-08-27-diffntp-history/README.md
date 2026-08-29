# 2026-08-27 — Proprioceptive history on the generative-NTP encoders

Ten-step history on every actor observation term, from scratch, on the two
round-4 winners, with the same `action_rate_l2` -0.03 penalty as the 50B
promotion chains.

## Arms (seed 0, W&B group `diffntp-history`)

| arm | encoder | jobs |
|---|---|---|
| `diffntp_chunk_hist` | `diffntp_chunk_h1_ee_wide` | 5592715 |
| `diffntp_pair_hist` | `diffntp_pair_h1_ee_wide` | 5592716 |

Change against the 50B chains' `std1` stage: `history_length=10` on
`projected_gravity`, `base_ang_vel`, `joint_pos_rel`, `joint_vel_rel`, and
`last_action`. The actor input widens; the command channel is unchanged. This
is the `smooth_block` history set from the 2026-08-15 bottleneck campaign,
never before run with a generative-NTP encoder.

## Budget: ONE segment, cap not reached by design

Each arm is a single 15:59 segment carrying the full 20B frame target. The
walltime ends it far short of the cap — expected and accepted by the user
(2026-08-27). Each arm reports whatever depth one segment buys (~7B at
~120k fps).

DO NOT read these against the 50B chains as a budget axis: they differ in
frames AND stop mid-schedule, before the `sonic` reset switch. The honest
comparison is against the 50B chains' own checkpoint at the same frame count,
inside the `random80_adaptive20` phase.

## Verdict (2026-08-27): REFUTED

Matched over the 3,485 clips all five rows survived (`bones_testbed4096_v1`,
`randomization=none`, one seed):

| row | SR | L (own -> matched) | G (own -> matched) | ee |
|---|---:|---|---|---:|
| `diffntp_pair` @2.0B (no history) | **0.9182** | 24.51 -> 22.83 | 96.22 -> 81.77 | 282 |
| `diffntp_pair_hist` @2.0B | 0.8909 | 26.66 -> 25.30 | 101.10 -> 92.55 | 374 |
| `diffntp_pair_hist` @4.0B | 0.8789 | 27.90 -> 27.11 | 104.08 -> 98.34 | 419 |

At MATCHED 2.0B frames the ten-step histories cost -0.027 SR, +11% local and
+13% global error against the identical arm without them, and raise wrist
(`ee_body_pos`) failures 33% (374 vs 282). Outside the noise band on SR.

It does not recover with budget: the 4.0B row is worse than its own 2.0B row
on all three metrics, and the ICE milestone curve
(`bones_milestone_testbed256_v1`, 8 points) peaks at 2.00B and flattens.

### Smoothness (2026-08-28): the one axis history helped

The same rows carry SONIC's other two tracker metrics. On the arm's own board
population, at MATCHED 2.0B frames:

| row | SR | acc | vel |
|---|---:|---:|---:|
| `diffntp_pair` (no history) | 0.9182 | 7.10 m/s² | 0.243 m/s |
| `diffntp_pair_hist` | 0.8909 | **6.42 m/s²** | 0.249 m/s |
| `diffntp_pair_hist` @4.0B | 0.8789 | 7.09 m/s² | 0.259 m/s |

Ten-step history bought -9.6% acceleration at 2.0B while costing 0.027 SR, and
the advantage is gone by 4.0B (7.09 vs the parent's 7.10). Acceleration is not
part of the REFUTED verdict above, which rests on success rate and tracking
error; this is the one axis where the history arm was ahead, and it did not
hold with budget. One seed, and the 4.0B parent row does not exist, so the
4.0B comparison is against a 2.0B parent.

Robust rows (`bones_testbed4096_v1`, `randomization=no_push`, added
2026-08-28) agree with the clean verdict and erase the smoothness advantage:

| row | SR | L | G | vel | acc |
|---|---:|---:|---:|---:|---:|
| `diffntp_pair_hist` @2.0B | 0.8738 | 29.15 mm | 157.19 mm | 0.287 m/s | 7.12 m/s² |
| `diffntp_pair_hist` @4.0B | 0.8628 | 29.63 mm | 162.97 mm | 0.298 m/s | 7.82 m/s² |

Under domain randomization the arm loses on every axis with budget, including
acceleration, so the 2.0B smoothness gain measured on the clean board does not
survive the perturbed protocol.

Reading: capacity dilution. The actor input widens by ten frames of
proprioception the policy does not need, because the DiffSR-grounded latent
command already carries the temporal structure. The arm still beats the
deterministic-head hub on SR, so it is not broken — it loses to its own
parent.

QUALIFICATION: one seed, and these arms ran a single segment, so they never
reached the SONIC reset phase. The verdict covers the standard-regime portion
of training only. Not worth more budget: the deficit is present at matched
frames and grows with depth.

## Scoring

`./eval.sh` scores mirrored checkpoints. It reads the command width, hold,
code width, phase mode, macro terms, stride, anchor AND the five
`history_length=10` overrides back out of `campaign.yaml`, so an evaluation
cannot drift from training. The history overrides are load-bearing: they widen
the actor input and the policy restore is strict, so an eval that omits them
cannot load the checkpoint.

```bash
ARMS=diffntp_pair_hist ROWS=robust ./experiments/campaigns/2026-08-27-diffntp-history/eval.sh
```

## Status

- 2026-08-28: added `eval.sh`; scored the robust rows. See the smoothness
  table above for the acceleration reading on the existing clean rows.
- 2026-08-27: both arms died at ~4.0-4.5B when the ICE 300 GB quota filled
  mid-save (their 4.5B checkpoints were truncated and deleted). Scored at the
  surviving 2.0B and 4.0B points; see the verdict above. Not resubmitted.
- 2026-08-27 02:52: SUBMITTED seed 0, jobs 5592715 and 5592716.
