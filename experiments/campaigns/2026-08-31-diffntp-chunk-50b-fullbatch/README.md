# 2026-08-31 — the headline arm at 50B, on the promoted optimizer geometry

`diffntp_chunk_h1_ee_wide` (2026-08-22 pareto-stack round 4) trained from zero
to 50B. **This is the first 50B run of the headline arm.**

Status: **SUBMITTED 2026-08-31**, seed 0, 11 chained stages, jobs
**5601421-5601431**. W&B group `diffntp-chunk-50b-fullbatch`, project
`g1-bones-seed`, run id `d50fb-chunk-s0`.

## The 08-26 attempt is not reusable

`2026-08-26-diffntp-50b` submitted this arm on 08-27 (jobs 5592690-5592700).
`std1` was CANCELLED after 9h06m, every downstream stage shows 0:00:00 and
never started, and `/data/diffntp_50b` no longer exists on ICE. Only a local
2.5B mirror survives, under `logs/diffntp_50b_mirror/`. This chain therefore
starts from zero; `seg_first` keeps its fresh-start value.

## What changed against `2026-08-26-diffntp-50b`

Two fields, both optimizer geometry, on user direction:

| | 08-26 (frozen) | this chain |
|---|---|---|
| agent entry point | `rlopt_ipmd_tuned_cfg_entry_point` | `rlopt_ipmd_tuned_fullbatch_cfg_entry_point` |
| envs | 16,384 | 20,480 |
| minibatch | `3/4 x batch` (368,640 at 20,480) | whole batch |
| epochs | 5 | 3 |
| optimizer steps / iteration | 10 | 3 |

The campaign passes **no** `agent.loss.mini_batch_size` and **no**
`agent.loss.epochs`. The config owns both, and a campaign override would
silently restore the 3/4 minibatch — that is the single easiest way to
invalidate this run, so do not add one.

Stage caps are expressed in FRAMES and `iters_*` divides by the batch, so the
schedule boundaries are unchanged by the environment count. Verified on the
frozen plan: 10,000,465,920 / 30,000,414,720 / 50,000,363,520.

## Schedule (the `ln_hold1_sonicreset` leader's history, unchanged)

| stages | reset regime | cap |
|---|---|---|
| `std1`-`std3` | `random80_adaptive20`, termination curriculum ON | 10B |
| `ramp1` | `selection=sonic`, `adaptive_uniform_ratio` 0.5 -> 0.1 over 2.5B | 30B |
| `hold1`-`hold3` | `sonic`, ratio pinned 0.1 | 30B |
| `focus1`-`focus4` | `sonic`, ratio pinned 0.05 | 50B |

The ramp keys off `common_step_counter`, which restarts every stage, so it must
complete inside `ramp1` and later stages pin the landed value.

## Why this geometry

`mb_full_e3` in `2026-08-30-optimizer-ablation-5b`, seed 0, at the full 5B:

* best MPJPE-L of that campaign, 28.54 +- 0.08 mm against a 30.3-44.3 field;
* highest completion share, `reference_finished` 0.558;
* +18.9% frames per unit wall-clock (4.21B against `mb_full_e5`'s 3.54B in a
  matched 7h24m);
* **the only arm of eight that did not degrade in the second half of
  training.** Six of eight turned over after ~2.3B and two went non-finite.
  That property is why it was chosen for a chain that runs to 50B.

## What this costs, and what is not established

* **The headline is no longer optimizer-matched** to `sonic-reset-50b` or
  `emastack-50b`, which used the frozen tuned geometry. A comparison against
  those rows now carries an optimizer difference as well as a recipe one.
* The 5B evidence is **one seed with no matched control**: `ctrl` in that
  campaign was cancelled by SIGTERM at 2.14B and never reached a comparable
  frame count. Every comparison behind this promotion is arm-vs-arm.
* The geometry grid did **not** order by update density — 3 steps 28.54, 5
  steps 40.37, 6 steps 44.34, 10 steps 32.71 — so the mechanism behind
  `mb_full_e3`'s win is unexplained. It is one good cell, not a trend.
* `env.rewards.action_rate_l2.weight=-0.03` is inherited from the 08-26 plan
  and remains the stated risk recorded there: as a FINETUNE at 46.5B every
  action-rate weight cost SR and both error metrics. This chain trains it from
  zero, a different regime, which is the point — but it is untested at 50B.

## Follow

```bash
pixi run python -m imitation_experiments.pipeline.cluster status \
  --campaign diffntp-chunk-50b-fullbatch
pixi run python -m imitation_experiments.pipeline.cluster logs \
  --campaign diffntp-chunk-50b-fullbatch --stage std1 -n 40
```

Score with `2026-08-30-optimizer-ablation-5b/eval.sh` (same interface, same
encoder) by pointing `MIRROR` and `ARMS` at this tree, and carry the
`sonic_v1_1` row for whichever board is used.
