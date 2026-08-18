# 2026-08-17 — Language planners on the 10B trackers

Two GR00T-head planner arms, one per interface, both driving a tracker from
`2026-08-15-latent-bottleneck-10b` at 10.0B frames. Matched budgets, so the
command interface is the only intended difference.

| arm | tracker | command | planner emits |
|---|---|---|---|
| `fsq64_10b` | `fsq64_hold10` | 64-D FSQ + phase, hold 10 | 3 latents x 64 dims per call |
| `ln_hold1_10b` | `cont_det_ln_hold1` | 256-D continuous + phase, hold 1 | 30 latents x 256 dims per call |

The head must emit at the cadence its tracker was trained on. That makes the
regression target 40x wider on the hold-1 arm (7,680 values against 192), which
is inherent to the interface and is the main reason the arm could lose despite
a better tracker.

## Why hold 1 is worth training now

The 2026-08-13 hold-1 arm scored 85.77 mm and the campaign recorded it as
refuted. That test drove a **hold-10-trained** tracker whose `sin_cos` phase
channel swept 0 -> 0.9 during training and sat pinned at slot 0 under a
per-step planner. `cont_det_ln_hold1` was trained at hold 1 with
`code_period=1`, where the phase term is `(period - steps)/period = 0` on every
step (`RLOpt/rlopt/agent/imitation/latent_learning.py:2173`): the channel is the
constant `(0, 1)` in training and in deployment. Per-step latents are this
tracker's native distribution.

## Tracker ceilings (2026-08-17, measured before any planner)

`eval_oracle_ceiling.sh`, M3 planner protocol: 150 environments (30 goals x 5),
2000-step cap, fall-only success, Newton/MJWarp, push off, **sensor noise on**,
`robot_heading` frame. Reported as the episode mean, the same reduction as the
planner table below.

| tracker / cadence | all 30 | 29 (drop rank 28) | 28 (2026-08-13 set) | fall-free (30) |
|---|---:|---:|---:|---:|
| `cont_det_ln_hold1` 10B, hold 1 | 23.42 mm | **20.60 mm** | 18.44 mm | 0.973 |
| `fsq64_hold10` 10B, hold 10 | 28.07 mm | **25.16 mm** | 22.36 mm | 0.973 |
| `fsq64_sonic` 4.5B, hold 10 (the 46.95 mm arm's tracker) | 29.83 mm | 25.66 mm | 22.32 mm | 0.947 |

Under SONIC terminations instead of fall-only, same 150 episodes: SONIC SR
0.9400 (`cont_det_ln_hold1`) against 0.8933 (`fsq64_hold10`) on all 30, every
failure `ee_body_pos`.

## The motion set is 29, not 28

The 2026-08-13 set dropped two motions whose oracle fell 4/5 on the 4.5B
tracker. On the 10B trackers only one of them is still tracker-limited:

| rank | motion | oracle falls, 4.5B | `fsq64_10b` | `ln_hold1_10b` |
|---|---|---:|---:|---:|
| 22 | `panic_run_away_180` | 4/5 | **0/5** | **1/5** |
| 28 | `walk_big_dog_ff_225_stop` | 4/5 | 4/5 | 3/5 |

So rank 22 returns to the set and rank 28 stays out (`EXCLUDE_RANKS=28`). Rank
22 is still the hardest surviving motion — oracle MPJPE 81-103 mm against a
board median near 20 mm — and both trackers fail it under SONIC thresholds
(5/5 and 4/5, `ee_body_pos`), so it is a fall-free-but-imprecise motion, which
the fall-only planner protocol admits.

Ranks are the ORIGINAL manifest positions throughout: they index the reference
arrays, so renumbering after a drop would silently score different motions.

## Reduction convention

The 46.95 mm headline is the **episode mean** — the mean over episodes of each
episode's MPJPE. The same run's transition-weighted `metric_means` is 53.62 mm
and its success-only mean is 48.27 mm. All three are in every summary; compare
like with like, and say which one a number is.

## Pipeline

```bash
./collect.sh fsq64_10b 0        # per-seed rollout collection, one row per control step
./collect.sh fsq64_10b 1
./merge_seeds.sh fsq64_10b      # symlink both seeds into collection_merged/
WANDB_GROUP=<confirmed> ./train.sh fsq64_10b     # prepare table + train the head
./eval.sh fsq64_10b             # 29 x 20, 2000-step cap, exponential ensembling 0.5
```

Budget per arm, matched to the 889,044-row arm that produced 46.95 mm: 29 goals
x 31 environments x 500 steps x 2 seeds, about 899k rows before early reference
ends. Training is 12k updates, batch 64, `state_dropout_prob` 0, warm-started
trunk — identical for both arms.

`consume_slots` is 1 on the FSQ arm (re-plan every publication, 10 control
steps) and 10 on the hold-1 arm (re-plan every 10 control steps out of a
30-slot horizon). Both give temporal ensembling three overlapping predictions
per published latent.

## Known deviations from upstream, unresolved

The 2026-08-13 parity check found a loss mask-normalization defect (~38x
gradient inflation) and an undisclosed `attend_text_every_n_blocks=1` masking
change in the head. Both are still present. Keeping them makes these arms
comparable with the 46.95 mm row; fixing them first would not. Either choice is
defensible, but it must be made before the first arm trains, not after.
