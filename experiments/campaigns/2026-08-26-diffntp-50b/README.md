# 2026-08-26 — Promote the generative-NTP winners to 50B

`diffntp_chunk` and `diffntp_pair` from `2026-08-22-pareto-stack` round 4,
trained from zero to 50B on the `ln_hold1_sonicreset` leader's exact reset
schedule, with the action-rate penalty restored at -0.03.

## Why these two

Round-4 clean rows, matched over the 3,521 clips all six arms survived:

| arm | SR | L (own -> matched) | G (own -> matched) |
|---|---:|---|---|
| hub (mlp, conditional mean) | 0.9060 | 27.16 -> 25.62 | 104.72 -> 91.40 |
| `diffntp_chunk` (executed frame) | 0.9163 | 24.18 -> 22.87 | 85.74 -> **74.57** |
| `diffntp_pair` (s, z joint) | **0.9182** | 24.51 -> 23.06 | 96.22 -> 82.86 |

`chunk` is the best global-error arm measured in the program; `pair` is the
best success rate. Both dominate the mlp-head recipe the `emastack-20b`
promotion is training.

## Schedule (the leader's history, replicated)

| stages | reset regime | cap |
|---|---|---|
| `std1`-`std3` | `random80_adaptive20`, termination curriculum ON | 10B |
| `ramp1` | `selection=sonic`, `adaptive_uniform_ratio` 0.5 -> 0.1 over 2.5B frames | 30B |
| `hold1`-`hold3` | `sonic`, ratio pinned 0.1 | 30B |
| `focus1`-`focus4` | `sonic`, ratio pinned 0.05 | 50B |

The ramp keys off `common_step_counter`, which restarts every segment, so it
must complete inside `ramp1` and later stages must pin — leaving it enabled
would re-sweep it.

## The action-rate penalty: a stated risk

`env.rewards.action_rate_l2.weight=-0.03` (the `tuned_lowlevel` recipe zeroes
this term; the 2026-08-02 HP search zeroed it for training speed).

The 2026-08-26 `smooth-finetune` campaign measured this weight as a FINETUNE
at 46.5B and REFUSED promotion — every weight cost SR, local and global error
for its smoothness gain:

| arm | SR | L | G | acc m/s^2 |
|---|---:|---:|---:|---:|
| parent @46.5B | 0.9773 | 21.95 | 92.31 | 5.45 |
| `ar003` (-0.03) | 0.9756 | 22.53 | 99.45 | 4.84 |
| `ar01` (-0.1) | 0.9734 | 24.19 | 102.94 | 4.43 |

-0.03 was the efficient point of that screen (-11% acceleration for +0.58 mm
local). These chains train the penalty FROM ZERO, a different regime, which
`smooth-finetune/ar01scratch` is measuring independently (4.15B/10B at
submission time). The user launched without waiting for it on 2026-08-27.

CONSEQUENCE FOR READING THE ROWS: these arms differ from the round-4 screen
rows by TWO variables — budget and the action-rate penalty. A 50B row that
underperforms cannot be attributed to the generative head without the
`ar01scratch` result to separate the penalty's from-zero cost.

## Status

- 2026-08-27 02:49: SUBMITTED seed 0. `diffntp_chunk_50b` jobs
  5592690-5592700, `diffntp_pair_50b` jobs 5592702-5592713 (11 stages each).
  Encoders pinned to the round-4 pretrains; nothing measured.
