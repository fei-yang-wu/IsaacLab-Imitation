# 2026-08-17 — canonical paper-facing metrics

## Purpose

Decide, once, what a paper-facing tracking number in this repo means, and build
the population it is measured on. Protocol definition:
`wiki/canonical-paper-metrics.md`. This directory holds the launcher that
regenerates the released-SONIC reference rows.

## Question that started it

The SONIC paper advertises 22.3 mm MPJPE-L at 100% success. Our 4,096-clip
board scored the released SONIC checkpoint at 28.65 mm. Where does the
difference come from?

## What we found, in the order we found it

1. **Domain randomization is a real but minor term.** Our board ran `no_push`,
   which keeps startup and reset randomization. Turning randomization fully off
   moves the released checkpoint from 28.66 mm to **25.90 mm** and success rate
   from 0.9934 to 0.9946. Quality is randomization-sensitive; success is not.
   (The `no_push` row was measured twice, 2026-08-07 and 2026-08-17: 28.65 and
   28.66 mm, so MPJPE-L repeats to about 0.01 mm on this board.)

2. **22.3 mm is not the number to chase.** It is the paper's **123-clip
   hardware deployment set** scored in simulation, never enumerated (Figure S2
   has no names or IDs). Its comparable large-set row is test-content
   98.7% / 23.2 mm.

3. **A "hardware-plausible" board was built and then deleted.** A rule that
   kept only upright, moderate-speed clips produced 21.92 mm at SR 1.0000 on
   123 clips, which looked like a reproduction of the paper's headline. It was
   not. SONIC's project site shows real-robot deployment of **Squatting,
   Kneeling, Hand Crawling, Elbow Crawling** and Boxing — exactly the class
   that rule excluded. The rule was selecting for *ease*, not deployability.
   Board, rule, and the claim built on them are removed. Do not rebuild one
   without evidence about what a real G1 can actually run.

4. **Nothing here is held out from training.** Trackers train on the full
   129,785-clip tree with no rank filter, so ranks 12288-16383 are training
   data. Earlier notes calling that block "held out" were wrong.

5. **The residual against 23.2 mm is still unexplained.** Retargeting is ruled
   out — BONES-SEED is the SONIC keyword-filtered corpus and the G1 retargeting
   is upstream's, so both sides use the same retargeted data. Backend is ruled
   out in the wrong direction: PhysX is 2 mm *worse* than Newton/MJWarp here.
   What remains is population composition, which we cannot check against a set
   the paper does not publish.

## The testbed this produced

`bones_testbed4096_v1`: 4,096 clips sampled from the whole corpus after
dropping scene-dependent clips (crate, door, lever, horse, …), length and
below-floor artifacts, and the easiest quarter by a reference-only difficulty
index. Squat, kneel, crawl and boxing stay in. Selection code:
`imitation_experiments.evaluation.clip_features.select_testbed_ranks`.

Released checkpoint: **SR 0.9912 / MPJPE-L 28.75 mm / MPJPE-G 135.73 mm**
clean, and **SR 0.9905 / 31.06 mm / 192.93 mm** under `no_push`. The clean row
is 2.85 mm harder than the legacy block at matched randomization, which is the
difficulty band working as intended.

## First testbed rows (2026-08-17)

Clean protocol (`paper_testbed4096_v1`), 10.0B-frame latent arms from the
`latent-bottleneck-10b` campaign:

| arm | SR | MPJPE-L | MPJPE-G |
| --- | --- | --- | --- |
| cont_det_ln_hold1 | 0.9236 | 24.23 mm | 120.0 mm |
| fsq64_hold10 | 0.9048 | 26.00 mm | 147.2 mm |
| released SONIC | 0.9912 | 28.75 mm | 135.7 mm |

Robustness partner (`paper_testbed4096_robust_v1`): 0.9126 / 25.86 mm,
0.8923 / 28.35 mm, 0.9905 / 31.06 mm respectively.

On the 3,664 clips all three complete, the success-rate confound is removed:
cont_det_ln_hold1 **23.40 mm**, fsq64_hold10 25.89 mm, released SONIC
27.79 mm; MPJPE-G 109.5 / 146.2 / 115.7 mm. The ranking is unchanged, so the
tracking-precision advantage is not an artifact of scoring an easier subset.

The testbed costs our arms far more success rate than it costs SONIC's
(-2.4 and -2.7 points against -0.3), which is the difficulty band exposing
exactly the clips our trackers drop. SONIC completes 282 clips that
cont_det_ln_hold1 fails; the reverse holds for 5.

## Videos

`render_clips.sh` renders the motions where the released checkpoint
completes and `cont_det_ln_hold1` does not, one clip per run at one
environment, both sides on the identical pinned rank and the identical clean
protocol. Three renders per clip: the protocol run (which ends AT the
termination, so the clip is the failure moment), the same run with early
terminations off, and the released checkpoint. Output:
`logs/testbed4096/failure_videos/`.

### Successes — `SET=successes`

Ten families, each represented by its median clip by minimum reference pelvis
height, so the sample is typical rather than picked. Both sides complete every
one. Output: `logs/testbed4096/success_videos/`.

| clip | ours MPJPE-L | SONIC MPJPE-L | ours MPJPE-G | SONIC MPJPE-G |
| --- | --- | --- | --- | --- |
| `medium_big_light_two_hands_walk` r22323 | 15.7 mm | 27.7 mm | 30.3 | 74.6 |
| `walk_ff_stop_315` r10543 | 16.5 mm | 18.7 mm | 59.9 | 139.0 |
| `small_heavy_one_hand_put_down` r75358 | 17.3 mm | 34.6 mm | 20.6 | 39.4 |
| `jog_arc_cw_stop` r26867 | 17.7 mm | 25.5 mm | 89.7 | 72.2 |
| `jump_ff_180` r35154 | 17.9 mm | 20.0 mm | 59.1 | 85.9 |
| `big_light_two_hands_walk` r53256 | 19.9 mm | 35.5 mm | 44.1 | 90.2 |
| `turn_run_270` r59632 | 28.3 mm | 26.7 mm | 187.7 | 145.3 |
| `injured_leg_jog_loop` r16372 | — | 36.9 mm | — | 108.9 |
| `dance_hiphop_tls_step` r103805 | 41.7 mm | 35.6 mm | 86.1 | 82.3 |
| `crouch_ff_loop_270` r56107 | 60.3 mm | 55.7 mm | **968.3** | 100.4 |

Our arm wins the loco-manipulation carries by 12-18 mm and the plain locomotion
by 2-8 mm; it loses dance, turn-run, and the crouch loop. **`crouch_ff_loop_270`
is the mandatory-MPJPE-G case in one clip**: local error is within 5 mm of the
released checkpoint while global error is nine times worse. The pose is right
and the robot is somewhere else. Anyone reading MPJPE-L alone would call that
row a tie.

### Failures — `SET=failures`

What the first four show — the deficit is not one failure mode:

| clip | ours, protocol | ours, full horizon | released SONIC |
| --- | --- | --- | --- |
| `kneeling_stop` r15135 | `ee_body_pos` at 24 steps | 340.3 mm over 154 steps | 40.6 mm |
| `high_jump` r31600 | `ee_body_pos` at 31 steps | 57.3 mm over 139 steps | 45.5 mm |
| `reach_jump` r49681 | `ee_body_pos` at 39 steps | 33.2 mm over 203 steps | 29.2 mm |
| `dance_take_the_l` r65247 | `ee_body_pos` at 41 steps | 71.1 mm | 45.1 mm |

`reach_jump` is a **transient threshold trip**: with the gate off the arm
tracks the whole clip to within 4 mm of the released checkpoint. `kneeling_stop`
is a **genuine collapse**: 340 mm means the policy is not tracking at all after
the gate would have fired. Those two need different fixes, and the success rate
alone cannot tell them apart.

The `dance_take_the_l` row is capped by `--video_length 260`, so neither side
reaches `reference_finished` there; do not read SONIC's row as a failure.

Worth running next: the same full-horizon pass over all 282 deficit clips,
split by whether the arm recovers. That separates gate-tuning from capability.

A render costs about a minute, including Isaac startup. One render failed on a
transient Omniverse CDN fetch of the ground-plane USD; rerunning the same rank
fixed it, and the script skips clips that already have a result.

## Run it

```bash
./experiments/campaigns/2026-08-17-paper-metric-canon/run_sonic_reference_rows.sh
./experiments/campaigns/2026-08-17-paper-metric-canon/run_sonic_reference_rows.sh --report
```

About 12 minutes per row on one RTX PRO 6000. Results land in
`logs/sonic_release_4096/`.

## What this does not settle

- Whether our own trackers keep their ranking on the testbed. Every arm was
  scored on the legacy block under `no_push` and needs re-scoring.
- Run-to-run Isaac noise for a single arm. Population noise between two 4,096
  blocks is under 0.1 mm, which is a different quantity from seed noise.
- The 2.7 mm gap against SONIC's test-content row.

## Paper-facing decision: two boards (2026-08-26)

User directive: the paper reports **two** evaluation populations and no
others.

1. **`bones_testbed4096_v1`** — the canonical 4,096-clip board, the deciding
   population. Report SR over all 4,096, and report success-only MPJPE / vel /
   acc over the **matched intersection** of exactly the rows in the table.
2. **`sonic_capability124_v1`** — the 124-clip new common eval subset, the
   SONIC-facing calibration board (`wiki/sonic-v1_1-subsets.md`). It was
   selected by reading public SONIC's own results, so it favors that anchor:
   never call it held out, and never make the headline SONIC comparison on it.

### The matched population is a property of the TABLE, not of the board

Success-only errors averaged over unequal success sets are not comparable, so
each table reduces to the clips every row in it completes. That intersection
changes when a row is added or removed, so it must be frozen per table and
named, exactly like the 124-clip rank list.

For the 2026-08-26 smoothness table (rows: public `sonic_v1_1`,
`ln_hold1_sonicreset` @46.5B, `ar003`, `ar01`, `ar01shake4`):

- artifact `matched3932_smoothness_2026-08-26.json`, 3,932 unique ranks;
- file SHA-256
  `c5b8c3c873f25bdf521417d31d028bc673d2d8b0bac12f7864692d110bb12e40`.

Regenerate and RENAME whenever the row set changes; do not edit in place.

### PAPER NUMBERS, measured 2026-08-26

Every row below: `Isaac-Imitation-G1-v2`, Newton/MJWarp, seed 0, deterministic
`mode` actions, frame-0 starts, sequential ranks from the registry, SONIC
thresholds, `foot_pos_xyz` and `base_too_low` disabled, `robot_heading` macro
anchor. **One seed, one evaluation per row.** SR is over all 4,096 clips;
MPJPE / vel / acc are success-only micro means over the matched intersection.

**Table A — clean (`randomization=none`), matched 3,932**
artifact `matched3932_smoothness_2026-08-26.json`, SHA-256 `c5b8c3c8…`

| row | frames | SR (of 4,096) | MPJPE-L | MPJPE-G | vel m/s | acc m/s^2 | `ee_body_pos` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| public `sonic_v1_1` | - | 0.9888 | 26.25 mm | 177.41 mm | 0.193 | 3.34 | 41 |
| `ln_hold1_sonicreset` | 46.50B | 0.9773 | 21.95 mm | 92.31 mm | 0.205 | 5.45 | 77 |
| `ar003` (`action_rate_l2` -0.03) | 48.50B | 0.9756 | 22.53 mm | 99.45 mm | 0.206 | 4.84 | 74 |
| `ar01shake4` | 48.50B | 0.9753 | 24.02 mm | 106.94 mm | 0.209 | 4.38 | 75 |
| `ar01` (`action_rate_l2` -0.1) | 48.50B | 0.9734 | 24.19 mm | 102.94 mm | 0.209 | 4.43 | 79 |

**Table B — robust (`randomization=no_push`), matched 3,859**
artifact `matched_robust_smoothness_2026-08-26.json`, SHA-256 `33011687…`

| row | SR (of 4,096) | MPJPE-L | MPJPE-G | vel m/s | acc m/s^2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| public `sonic_v1_1` | 0.9895 | 28.26 mm | 209.53 mm | 0.203 | 3.39 |
| `ln_hold1_sonicreset` @46.5B | 0.9688 | 23.74 mm | 157.32 mm | 0.244 | 6.23 |
| `ar003` | 0.9666 | 24.71 mm | 174.86 mm | 0.245 | 5.52 |
| `ar01` | 0.9624 | 26.37 mm | 168.69 mm | 0.249 | 5.03 |
| `ar01shake4` | 0.9624 | 26.25 mm | 179.70 mm | 0.249 | 4.98 |

**Table C — `sonic_capability124_v1`, clean, matched 122 of 124.** The
SONIC-facing calibration board; see `wiki/sonic-v1_1-subsets.md` for its
selection contract and its bias toward the anchor.

| row | SR (of 124) | MPJPE-L | MPJPE-G | vel m/s | acc m/s^2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| public `sonic_v1_1` | 1.0000 | 23.39 mm | 147.44 mm | 0.159 | 2.75 |
| `ln_hold1_sonicreset` @46.5B | 1.0000 | 18.79 mm | 91.90 mm | 0.168 | 4.47 |
| `ar003` | 0.9919 | 19.02 mm | 78.65 mm | 0.169 | 3.94 |
| `ar01` | 0.9919 | 20.66 mm | 96.43 mm | 0.174 | 3.68 |
| `ar01shake4` | 0.9839 | 21.24 mm | 146.15 mm | 0.178 | 3.72 |

### How to state the SONIC comparison

The trade is two-directional and BOTH halves must appear wherever either does:

- SONIC leads **success rate** (0.9888 vs 0.9773 clean, 47 more clips of
  4,096) and **smoothness** (3.34 vs 5.45 m/s^2 acceleration distance, and
  0.193 vs 0.205 m/s velocity distance).
- We lead **accuracy**: 21.95 vs 26.25 mm local (-16%) and 92.31 vs 177.41 mm
  global (-48%).

Do not claim "our tracker beats SONIC." Do not cite a success-only MPJPE
across unequal success sets — that is why every table above reduces to a
matched intersection. Do not make the headline SONIC comparison on the
124-clip board: it was selected by reading SONIC's own results.

Velocity and acceleration distance are OUR measurements under one shared
definition (the 14 tracked links, world-frame link linear-velocity error, and
its finite difference at the 0.02 s control step), in SI units. They are not
SONIC's reported figures and must not be presented as such.

This supersedes the older claim-boundary table in
`wiki/final-paper-experiment-design.md`, which used contiguous ranks
12288-16383, `fsq64_hold10`, and the `sonic_release` checkpoint.
