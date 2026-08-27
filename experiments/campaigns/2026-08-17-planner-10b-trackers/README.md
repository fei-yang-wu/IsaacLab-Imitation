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

## The motion set: the original 28

User directive, 2026-08-17: train and evaluate on the **28-motion set behind
the 46.95 mm row** — the language30 dataset minus rank 22
(`panic_run_away_180`) and rank 28 (`walk_big_dog_ff_225_stop`). Both were
dropped in 2026-08-13 because the 4.5B tracker's oracle fell 4/5 on them.
`arms.sh` enforces this with `EXCLUDE_RANKS="22 28"`, and the goal features the
head trains against are the existing 30-goal cache, which covers all 28.

On the 10B trackers rank 22 is no longer tracker-limited (oracle falls 0/5 on
`fsq64_10b`, 1/5 on `ln_hold1_10b`), but it stays out so the motion set matches
the row these arms are compared against.

### A 30-motion successor exists but is not used here

`data/bones_seed_language30_v2` (`persist_id bones_seed_language30_v2@7a6d5c49`)
was built and screened on 2026-08-17: out `walk_big_dog_ff_225_stop` (still
tracker-limited on both 10B trackers, 4/5 and 3/5) and
`looking_in_the_mirror_amateur` (least discriminating at ~10.6 mm, and Object
Interaction was over-represented); in `walk_ff_loop_360_001_A049`
("walk backwards", the corpus's largest category had 3 of 30 and no backward
locomotion at all) and `reaching_up_001_A033` ("reaching up", an overhead
upper-body reach the set lacked). Reference arrays and MiniLM goal embeddings
are built; the eight screened candidates were all fall-free 5/5 on both
trackers.

Switching to it needs the GR00T goal features rebuilt, which needs Hugging Face
access to the gated `nvidia/Cosmos-Reason2-2B` backbone — unavailable on this
workstation.

## Reduction convention

The 46.95 mm headline is the **episode mean** — the mean over episodes of each
episode's MPJPE. The same run's transition-weighted `metric_means` is 53.62 mm
and its success-only mean is 48.27 mm. All three are in every summary; compare
like with like, and say which one a number is.

## Pipeline

```bash
./collect.sh fsq64_10b                       # ONE run, one row per control step
WANDB_GROUP=<confirmed> ./train.sh fsq64_10b # prepare table + train the head
./eval.sh fsq64_10b                          # 28 x 20, 2000-step cap, DR off
VIDEO=1 EPISODES_PER_GOAL=1 ./eval.sh fsq64_10b   # one rendered clip per motion
```

Evaluation runs with **domain randomization off** (`DR=off`, the default here):
`--deterministic_tracking` disables pushes and randomization and starts the
episode exactly on the reference, and the evaluator prefixes every metric with
`deterministic_tracking/` so an unperturbed number cannot be pooled with a
perturbed one. The 46.95 mm row was measured WITH randomization, so a DR=off
number is not directly comparable with it; `DR=on` reproduces that protocol.

One collection run per arm, never several merged: `prepare_gr00t_dataset` keys
rows by `(env_id, episode_id, control_step)` and each run numbers environments
from zero, so a second run of the same arm collides and the prepare step
refuses it. Budget: 28 goals x 93 environments x 500 steps, about 1.09M rows
before early reference ends, against the 889,044 rows of the 2026-08-13 arm
that reached 46.95 mm. Training is 12k updates, batch 64,
`state_dropout_prob` 0, warm-started trunk — identical for both arms.

Goal features come from the existing 30-goal cache
`outputs/gr00t_language30/goal_features/goal_features.pt`, which covers all 28
motions unchanged.

`consume_slots` is 1 on the FSQ arm (re-plan every publication, 10 control
steps) and 10 on the hold-1 arm (re-plan every 10 control steps out of a
30-slot horizon). Both give temporal ensembling three overlapping predictions
per published latent.

## Results (2026-08-18, DR off)

Both heads trained to 12k updates on ~1.09M rows each (28 goals x 93
environments x 500 steps, one row per control step), W&B group
`planner-hold1-vs-hold10`. Evaluation: 28 x 20 = 560 episodes,
`--deterministic_tracking` (pushes AND domain randomization off, start exactly
on the reference), 2000-step cap, exponential ensembling 0.5.

| arm | fall-free | MPJPE-L ep-mean | success-only micro | SONIC-thresh pass |
|---|---:|---:|---:|---:|
| `fsq64_10b` | **1.000** | **45.49 mm** | 48.85 mm | 0.904 |
| `ln_hold1_10b` | **1.000** | 46.62 mm | 49.14 mm | 0.898 |

Zero falls in 1,120 episodes across both arms. The MPJPE difference (2.5%) is
far inside evaluation noise: the interfaces are level on this protocol, one
seed. Per motion the split is 15/13 for `ln_hold1_10b`, with the largest gaps
on slow walks (ln better by 10-17 mm) and on `exercise_3`/`stoop_down`
(fsq better by 18-30 mm).

These DR-off numbers are NOT comparable with the 46.95 mm row, which was
measured with randomization on (`DR=on` reproduces that protocol; rows below).
Videos (one episode per motion, tiled):
`logs/planner_10b/isaac_eval/*__droff__video/videos/play/rl-video-step-0.mp4`.

## Results, second pass (2026-08-18 evening): DR=on, ensembling off, async

Same heads (update 12k), same 28 x 20 = 560 episodes, seed 0, one seed each —
preliminary.

| arm | protocol | fall-free | ep-mean | success micro | SONIC-thresh |
|---|---|---:|---:|---:|---:|
| `fsq64_10b` | DR=on, exp ens | 0.9893 | 50.48 mm | 53.15 mm | 0.9089 |
| `ln_hold1_10b` | DR=on, exp ens | 1.000 | 52.90 mm | 55.44 mm | 0.8732 |
| `fsq64_10b` | DR=off, ENSEMBLE=none, sync | 1.000 | 52.30 mm | 56.30 mm | 0.8768 |
| `ln_hold1_10b` | DR=off, ENSEMBLE=none, sync | 1.000 | 42.83 mm | 44.89 mm | 0.9107 |
| `fsq64_10b` | DR=off, async lead 5 | 0.9982 | 69.40 mm | 69.89 mm | 0.7946 |
| `ln_hold1_10b` | DR=off, async lead 5 | 1.000 | 41.45 mm | 43.64 mm | 0.9125 |

- DR=on `fsq64_10b` sits 7.5% above the 46.95 mm row — inside evaluation
  noise, unresolved; a settled comparison needs repeated seeds. The two 10B
  arms stay level under DR=on (4.8% apart).
- The `ENSEMBLE=none` sync rows are the matched companions for the async rows
  (async refuses ensembling). Ensembling helps `fsq64_10b` (45.49 vs 52.30)
  and hurts `ln_hold1_10b` (46.62 vs 42.83, ~8%, inside noise) — one seed.
- Async (first live D1 run, relaxed gate): `ln_hold1_10b` is level with its
  sync companion (41.45 vs 42.83 mm; 2,380 deadline misses, ~8.5% of
  renewals; round-trip p50 2418 ms). `fsq64_10b` degrades 33% (69.40 vs
  52.30 mm; 8,517 misses, ~30% of renewals; round-trip p50 163 ms, p95
  948 ms against a 10-control-step renewal with lead 5). The 30-slot hold-1
  horizon leaves a 20-step consumable tail that absorbs late responses; the
  3-slot FSQ horizon does not. Caveat: deadlines count sim control steps
  while round-trips are wall-clock, and the service shares the evaluator's
  GPU, so these miss rates are execution-mode-shaped, not a real-time 50 Hz
  certificate.

## Async evaluation infrastructure (phase D1, built 2026-08-18)

The head can now run as a separate service process while Isaac evaluates —
the deployment-shaped execution mode, with real request latency in the loop:

- `imitation_experiments/planner/gr00t_service_protocol.py` — wire format:
  zmq multipart, JSON header + raw float32 bytes. Torch-free on the wire, so
  the service keeps the upstream torch 2.9 pin while Isaac keeps 2.11.
- `imitation_experiments/planner/gr00t_batch_service.py` — one process, the
  trained head + all goal features on GPU; serves batched
  `act(states, goal_ids) -> chunks` on a zmq REP socket (`gr00t` Pixi env).
- `imitation_experiments/planner/gr00t_async_sampler.py` — drop-in for the
  sync sampler implementing the design's request loop: lead-time requests
  (`--gr00t_lead_steps` before a needed renewal), swap-at-expiry with
  time-aligned slot skip (`floor(elapsed/hold)` — the training join puts
  slot k at `request_step + hold*k`), deadline miss = consume the previous
  chunk's tail then re-publish the last slot, counted, never blocking, never
  fabricated. The one blocking point is an environment's first chunk after
  reset (`startup_syncs`, counted separately). Ensembling / `fresh` /
  multi-sample are refused in async mode.
- Evaluator flags `--gr00t_service <endpoint>` and `--gr00t_lead_steps`; the
  summary's `gr00t_planner` block then carries
  `planner_execution: async_service`, miss/startup counts, and service
  round-trip latency quantiles.
- `./eval_async.sh <arm>` — starts the service, waits for its ready record,
  runs the evaluator, tears down.

Per the relaxed D1 gate (user, 2026-08-18): an async row is always labelled
async and reported NEXT TO its D0 sync companion on the same seed, with the
gap and the deadline-miss statistic stated — but no numeric equivalence bound
blocks anything. Async and sync rows are never pooled.

Contract tests: `tests/test_gr00t_async_service.py` (wire round-trip,
request batching, epoch fencing across resets, time-aligned swap, miss-hold
semantics) and `tests/test_gr00t_async_chunk.py` (the chunk routes' frame
math, staged swap, per-boundary miss accounting, anchor re-expression) —
default environment, no GPU. Live D1 runs against a real service: done on
2026-08-18 for both latent arms and on 2026-08-19 for both chunk routes.

Async at the winning cadence (2026-08-19, 560 episodes, seed 0):

| row | MPJPE-L | fall-free | SONIC-thresh | deadline misses |
|---|---:|---:|---:|---:|
| sync, 30 slots | 38.41 mm | 1.000 | 0.9143 | — |
| async service, 30 slots, lead 5 | 38.78 mm | 1.000 | **0.9464** | 19,996 |

Nearly every renewal missed its deadline (round-trip p50 2.4 s against a
5-step lead, which is about 2.5 s at this environment count) and the row is
still level with its sync companion, because 30 slots of runway make the miss
path cheap: the live plan keeps being consumed and the late reply swaps in at
the next renewal. The FSQ arm's 3-slot horizon has no such runway and paid
33% for the same execution mode. Long plans buy latency tolerance — the same
property that makes 30 slots win in the first place. A lead of about 10 steps
would remove most of these misses and is untested.

Reading a deadline-miss count: it is measured in SIM control steps against
wall-clock round-trips, on a GPU shared with the simulator. At 560
environments the simulator is far slower than real time (a 979 ms round trip
is about 1.7 control steps), so a miss count here is execution-mode-shaped
and is NOT a real-time 50 Hz certificate.

## Hold-1 planner: what limits it, and the cadence that improves it (2026-08-19)

All rows below are DR off, no ensembling, seed 0, one seed — preliminary. The
sweep arms use 5 episodes per goal (140 episodes) and are comparable with each
other, NOT with the 560-episode rows of record; the 140-vs-560 repeatability
of one config was 1.3% (95.68 vs 94.42 mm on the explicit legacy row).

### The tracker is not the limit; the planner is

`ORACLE=1 ./eval.sh ln_hold1_10b` runs the identical protocol with the frozen
encoder publishing instead of the head:

| row | MPJPE-L ep-mean | fall-free | SONIC-thresh |
|---|---:|---:|---:|
| oracle latent (the ceiling) | **17.19 mm** | 1.000 | 1.0000 |
| planner, `consume_slots` 10 | 42.83 mm | 1.000 | 0.9107 |

So the planner contributes 25.6 mm of the 42.8 mm, about 60% of the error.

### Consumption cadence is the lever, and longer is better

`FORCE_CONSUME_SLOTS` changes how much of each 30-slot prediction is consumed
before the head is called again. Nothing else changes.

| consume slots | MPJPE-L (140 ep) | head calls | published-vs-oracle z cosine |
|---:|---:|---:|---:|
| 1 (re-plan every control step) | 50.76 mm | 1055 | 0.677 |
| 3 | 47.61 mm | 352 | — |
| 10 (the shipped setting) | 44.18 mm | 106 | 0.661 |
| **30 (consume the whole plan)** | **38.73 mm** | **36** | 0.517 |

Confirmed at the full 560 episodes, seed 0:

| row | MPJPE-L | fall-free | SONIC-thresh |
|---|---:|---:|---:|
| 10 slots (shipped) | 42.83 mm | 1.000 | 0.9107 |
| **30 slots** | **38.41 mm** | 1.000 | 0.9143 |
| 30 slots, clean observations | 37.60 mm | 1.000 | 0.9107 |

4.41 mm better than the shipped setting (10.3%), same direction as the 12% at
140 episodes, and the single fall seen there did not reproduce.

How solid that is, tested on the rows themselves rather than against the
blanket 15% heuristic. Pair the two runs per MOTION — the right unit, because
the spread between motions (sd 29 mm) dwarfs the spread between episodes of
one motion (sd 8 mm), so counting 560 episodes as 560 independent samples
would overstate the case:

| statistic | value |
|---|---|
| paired per-motion difference | **-4.41 mm** (95% bootstrap CI -8.96 to -1.10) |
| paired t, 28 motions | -2.12, two-sided p = 0.043 |
| sign test | 19/28 motions, p = 0.087 |
| episode sd, SE of a 560-episode mean | 30.2 mm, 1.28 mm |

Physics non-determinism is already inside both runs, so this interval covers
it. What a repeat still adds is the run-level component — every episode in a
run shares one checkpoint, one process, one GPU state — which no within-run
statistic can see. Two same-configuration re-measurements bound that at about
1% (94.42 vs 95.68 mm, 38.41 vs 38.73 mm), small next to a 10.3% effect.
Repeating both arms would double the paired sample to 56 differences and is
the cheap way to make p unambiguous before the number goes in a paper.

The ordering is the REVERSE of per-publication accuracy: the best row
publishes the least oracle-like latents. A fresh head call draws fresh flow
noise, so re-planning often makes the published sequence jitter, while the
slots of one draw are mutually consistent. The tracker is more sensitive to
discontinuity in the command stream than to per-step command error. Consuming
the whole plan is also 3x cheaper in head calls and leaves a 30-step (0.6 s)
re-plan budget, which is what makes the async row's deadline pressure vanish.

**Averaging flow draws is actively harmful here**: `SAMPLES=4` at 10 slots
scores 98.85 mm against 44.18 mm, 2.2x worse. Averaging k draws at one state
collapses toward the conditional mean, and the mean of several valid 256-D
latents is off-manifold (and shorter): the tracker receives a command its
encoder would never emit. Variance reduction is the wrong instinct for a
continuous latent interface — the tracker wants a COHERENT command, not an
average one. `samples_per_publication` was designed for the FSQ arm, where
averaging happens pre-snap and the lattice restores a valid code.

Two knobs that do NOT matter, measured the same way:

* ODE steps 16 against the default 4: 43.65 vs 44.18 mm (1.2%, noise).
* Observation corruption off (`OBS_NOISE=off`): 44.62 vs 42.83 mm at 560
  episodes — slightly worse clean. The tracker trains with noise and gains
  nothing from having it removed, so the SONIC evaluator's clean-observation
  contract does not bias the comparison below.

### Why the head is only so good: horizon decay is small, covariate shift is not

`imitation_experiments.planner.head_slot_error` scores the head open-loop
against its own table, per horizon slot (1024 rows):

| slot | 0 | 9 | 29 |
|---|---:|---:|---:|
| cosine | 0.777 | 0.760 | 0.733 |

Decay across the consumed window is 2.2%, which is why freshness buys so
little. The head's closed-loop published cosine is 0.661 against 0.777
open-loop: a 15% drop caused by the state distribution, because the
collection was recorded with the ORACLE driving the tracker while at
evaluation the planner drives. A DAgger-style second collection under planner
control is the principled fix; it needs disk this workstation does not
currently have (the hold-1 collection alone is 28 GB, and a lookahead-bearing
one exceeded the free space and failed mid-write on 2026-08-19).

## SONIC comparison under this campaign's protocol (2026-08-19)

`./eval_sonic_row.sh planner|oracle` runs the released NVIDIA SONIC v1.1
decoder on THIS motion set and protocol: 28 motions x 20 episodes, 2000-step
cap, fall-only success, `--randomization none --reference_start_frame 0`
(the SONIC-side equivalent of `--deterministic_tracking`). The head is the
`sonic_hold1` GR00T head from 2026-08-12, which predicts SONIC's own 64-D FSQ
token — so all three rows share the head recipe and budget and differ in the
(encoder, tracker) pair.

| system | tracker ceiling | planner row (560 ep) | planner-induced |
|---|---:|---:|---:|
| ours, 10B `cont_det_ln_hold1` | **17.19 mm** | 42.83 mm (slots 10) | +25.6 mm |
| ours, same, slots 30 | 17.19 mm | 38.41 mm | +21.2 mm |
| ours, slots 30, clean observations | 17.19 mm | **37.60 mm** | +20.4 mm |
| SONIC release v1.1 | 22.70 mm | 38.49 mm | **+15.8 mm** |

Our tracker is 24% better than SONIC's. At the shipped cadence SONIC won end
to end; at 30 slots the two are level (37.60 against 38.49 on the same clean
observation contract, 2.3% apart and inside noise). The gap that remains is
in the planner-induced column: SONIC's latent space converts planner error
into tracking error more gently (+15.8 mm against our +20.4), which is a
property of the interface rather than of planner quality. That is the reason
to keep measuring the ceiling next to every planner row.

With the fsq64 ceiling measured on the same protocol (20.47 mm, fall-free
1.000, every episode passing the SONIC threshold), all three latent spaces
can be ranked two ways at once — and the two rankings are opposites:

| latent space | tracker ceiling | best 12k-head row (560 ep) | planner-induced |
|---|---:|---:|---:|
| ours, continuous 256-D hold 1 | **17.19 mm** | 38.41 mm | +21.2 mm |
| ours, FSQ-64 hold 10 | 20.47 mm | 44.75 mm | +24.3 mm |
| SONIC FSQ v1.1 | 22.70 mm | 38.49 mm | **+15.8 mm** |

SONIC has the WORST ceiling of the three and the best end-to-end row: its
latent space absorbs planner error better than either of ours.

**But MPJPE is not the whole comparison.** Scored on SONIC's own threshold
criterion — the fraction of episodes that never violate anchor_pos 0.25,
anchor_ori 1.0 or ee_body_pos 0.25 — the two systems separate sharply:

| row | MPJPE-L (fall-only) | SR (SONIC thresholds) | failures |
|---|---:|---:|---|
| ours, 30 slots | 38.41 mm | **0.9143** | — |
| SONIC release planner | 38.49 mm | 0.6232 | ee_body_pos 185, anchor_ori 26 |
| ours, oracle | 17.19 mm | 1.0000 | — |
| SONIC oracle | 22.70 mm | 1.0000 | — |

Both sides mean the same thing by that rate: ours counts episodes with zero
tracking-failure events (terminations disabled), SONIC's counts episodes that
reached the reference end under `--termination_contract sonic`. Level MPJPE,
47% relative gap in threshold success, and SONIC's failures are almost all
wrist height — the same `ee_body_pos` term that dominates our 4096 board.
Both oracles pass 100%, so this is the planner violating thresholds, not the
tracker.

Do NOT quote SONIC's MPJPE under its own termination contract (34.13 mm)
next to a fall-only row: violating episodes truncate at 402 steps against
496, so that number is survivor-biased. The comparable figure is 38.49 mm. A latent
interface should therefore be scored by how much tracking error it produces
per unit of planner error, not by its oracle ceiling alone — an arm can win
the ceiling and lose the deployment.

The comparison is system against system: SONIC's decoder is a different
tracker trained on its own corpus, and it runs in a different evaluator
binary. It does not isolate the interface — that needs one tracker, which is
what the explicit-vs-latent rows below do.

## Explicit-interface arms (added 2026-08-18 evening)

To test "latent interface beats explicit under deployment-shaped execution",
an explicit chunk head `explicit_10b` trains on the SAME fsq64_10b collection
and optimizer budget (`conf/train_explicit_10b.yaml`, `target: chunk`, the
table's 1,088,277 x 30 x 38 `chunk_target`), W&B project `g1-bones-seed-tamp`,
group `planner-explicit-vs-latent`. Two routes (`./eval_explicit.sh
<native|encoded>`, `ASYNC=1` for the D1 service run):

### Correction (2026-08-19): the first explicit rows were confounded

The rows below encode `explicit_10b` onto the hold-1 tracker, but that head
trained on the **fsq64** collection — the only one carrying the 30-frame
lookahead. It therefore drove a tracker whose rollout states it never saw,
and the 2.2x deficit it showed was that mismatch, not the interface.

Pairing the same head with the tracker that produced its training states
reverses the result (560 episodes, DR off, no ensembling, same 12k budget):

| row on the `fsq64_hold10` 10B tracker | MPJPE-L | per-motion |
|---|---:|---|
| latent `fsq64_10b`, no ensembling | 52.30 mm | — |
| **explicit re-encoded, 10-frame cursor** | **44.75 mm** | better on 21/28 |
| explicit re-encoded, per-publication | 45.03 mm | — |
| latent `fsq64_10b`, exponential ensembling | 45.49 mm | — |

So on a matched pairing the EXPLICIT interface is level with or better than
the latent one. A plausible mechanism, consistent with everything else in
this campaign: the explicit route's prediction passes through the frozen
encoder, which projects it onto the manifold of latents the tracker was
trained on. A latent head has no such guardrail and can publish off-manifold
commands — the failure that makes averaged latents catastrophic (98.85 mm).

Not yet available: the matched explicit row on the HOLD-1 tracker, which
needs a hold-1 collection carrying the lookahead. That collection exceeded
the workstation disk on 2026-08-19 and failed mid-write.

- **native (row B)**: the head's 30-frame root_qpos packet drives the 7.6B
  `root_qpos_explicit` tracker through the chunk actor term. Caveats stated
  with the row: tracker frames unmatched (7.6B vs 10B), and the head's
  packets live in the `robot_heading` frame while that tracker is
  robot-frame — `--gr00t_packet_frame heading` pins the heading frame on the
  term so the re-expression is exact.
- **encoded (row C)**: the same packet through the frozen ln_hold1_10b
  encoder onto the 10B `cont_det_ln_hold1` tracker — the tracker-matched
  latent-vs-explicit row against the `ln_hold1_10b` rows. Precedent from the
  2026-08-13 campaign: the explicit head there also trained on a latent
  tracker's rollout collection.

Controls that separate the interface result from the new execution code
(560 episodes each, encoded route on the 10B hold-1 tracker):

| control | MPJPE-L | reads as |
|---|---:|---|
| legacy per-publication path (pre-2026-08-18 code) | 94.42 mm | the gap is not the cursor's doing |
| cursor at `consume_frames=1` | 94.62 mm | the cursor reproduces that path to 0.2% |
| cursor at `consume_frames=10` | 108.67 mm | consuming a stale explicit window costs 15% |

The first two are the same protocol expressed through different code, so
their agreement is the equivalence certificate for the cursor and, through
the shared `_PacketStore`, for the async chunk routes. The explicit route
prefers per-publication re-planning, the OPPOSITE of the latent arm's
preference — see the cadence section above.

Not wired: `--packet_source expert` (the pin test) reached the GR00T encoded
route as a no-op until 2026-08-19; it now passes through, and a 30-frame head
makes it fail loudly because the pin only supports the encoder's native
10-frame horizon.

New execution machinery (`imitation_experiments/planner/gr00t_async_chunk.py`,
contract tests `tests/test_gr00t_async_chunk.py`):

- `--gr00t_packet_consume_frames N` gives chunk_encoded a receding-horizon
  cursor: re-plan every N steps, serve intermediate publications from the
  cached packet at its age, re-expressed from the request anchor into the
  current one — the latent hold-1 arm's cadence instead of a head call per
  publication. Required with `--gr00t_service` on that route.
- Async chunk routes mirror the latent D1 protocol: lead-time requests,
  staged replies swapped at the boundary, miss = slide into the packet tail
  (20 steps on horizon 30) then hold, startup blocks and is counted apart.
  The native route republishes the live packet time-shifted and re-pins the
  request-time anchor (`ChunkActorCommand.pin_anchor_pose`).

## Known deviations from upstream, unresolved

The 2026-08-13 parity check found a loss mask-normalization defect (~38x
gradient inflation) and an undisclosed `attend_text_every_n_blocks=1` masking
change in the head. Both are still present. Keeping them makes these arms
comparable with the 46.95 mm row; fixing them first would not. Either choice is
defensible, but it must be made before the first arm trains, not after.
