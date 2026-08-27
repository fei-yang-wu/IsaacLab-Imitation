# 2026-08-27 — Does the candidate leader survive a 64-D command channel?

One arm. It decides the code width for the whole planned ablation set, because
every arm of that set inherits the leader's width.

## The arm

`leader64_h1_nophase`, seed 0, 2B frames. Continuous 64-D code, hold 1, no
phase channel, generative next-chunk objective (`--jepa_ntp_head diff_chunk`),
EE and wide-anchor rewards, `action_rate_l2` at −0.03.

The command channel is **64 values instead of 258** — the reason to want this
result. A planner has to emit that vector.

## Its control, and the confound

`diffntp_chunk_h1_ee_wide` from `2026-08-22-pareto-stack`:
**0.9297 / 23.45 mm / 74.85 mm** on `bones_milestone_testbed256_v1`.

Three fields move against it at once:

| # | change | why it is here |
|---|---|---|
| 1 | width 256 → 64 (command 258 → 64) | the question |
| 2 | phase channel dropped | at `code_period=1` the hold clock pins the phase constant, so the star's hold-1 arms already ran phase-free in all but two wasted command slots; `hold1_live_phase` measured a live clock as a null there |
| 3 | `action_rate_l2` restored at −0.03 | matches the 50B promotion chains; the measured efficient point of the 2026-08-26 smoothness sweep |

A shortfall therefore does not say which field caused it. The decomposition arm
to run in that case is **256-D + phase + action-rate**, which isolates (3); (2)
is the one change with a mechanism argument for being free.

## What makes this open rather than a formality

Width is free at hold 10: `bn_cont64` 0.9180 / 22.80 / 204.74 against `ctrl`
0.9102 / 23.44 / 199.87, every difference inside the evaluation band. But
width × hold is the one interaction this program has caught, and it points the
other way: `ix_fsq64_hold1` LOST success rate against its hold-10 partner
(0.8789 vs 0.9023) while the 256-D arm GAINED at hold 1 (0.9180 vs 0.9102).
That cell is quantized. Continuous 64-D at hold 1 has never been run.

## Reading it

Score on `bones_milestone_testbed256_v1`, the same board as the 2B study, and
compare against the control row above. The evaluation band is ΔSR 0.016,
MPJPE-L 1.3%, MPJPE-G 6.7%; inside it, the width is free and the ablation set
should be submitted at 64-D.

## Status

- 2026-08-27 17:47: submitted. Jobs 5593641 (pretrain) → 5593642 → 5593643,
  W&B group `leader64-gate`. Nothing measured.
