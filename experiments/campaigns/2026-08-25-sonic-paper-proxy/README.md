# 2026-08-25 — a population that faces a named SONIC paper column

## Purpose

Build an evaluation population on which the publicly released SONIC checkpoint
reproduces a number the SONIC paper actually reports, so the paper's own
baseline rows can be cited as a faithful proxy on our protocol instead of
re-derived.

Population definition: `imitation_experiments.evaluation.sonic_paper_proxy`.
Board and profiles: `sonic_proxy_testrep4096_v1`,
`paper_sonic_proxy_testrep4096_v1` (clean) and its `_robust_v1` partner in
`imitation_experiments.evaluation.protocol`.

## The question that started it

Aim for the paper's in-distribution MPJPE-L, about 23 mm, on hardware-
deployable trajectories. Our clean legacy-block row for the released checkpoint
is 25.90 mm at SR 0.9946. Where does the difference come from, and what
population closes it?

## What the paper actually reports

SONIC (arXiv 2511.07820v3, Science Robotics 11(117)) does not report one
number. For the robot-motion tracking path, which is what the released
checkpoint runs:

| paper location | split | SR | MPJPE-L |
| --- | --- | ---: | ---: |
| Table 4(c), robot encoder, 128 GPU | test-repetition | 99.8% | 22.5 mm |
| Table 4(c), robot encoder, 128 GPU | test-content | 99.6% | 23.8 mm |
| Table 4(a), "FSQ (ours)", 128 GPU | test-repetition | 99.6% | 25.5 mm |
| Table 4(a), "FSQ (ours)", 128 GPU | test-content | 99.3% | 26.6 mm |
| Table 4(b), FSQ-32-32, 32 GPU | test-repetition | 99.3% | 26.3 mm |
| Fig. 2(d-g), MuJoCo baseline comparison | test-content | 98.7% | 23.2 mm |
| Fig. 2(k-l), sim leg of sim-to-real | 123-clip deployment set | 100% | 22.3 mm |

The spread across rows that nominally share a configuration is 22.5-27.5 mm.

Splits (Table 2): test-content is 6,998 clips over 182 sub-categories entirely
absent from training; test-repetition is 6,306 clips, new takes of trained
sub-categories. Neither is enumerated.

## Finding 1 — the released checkpoint is the paper's 16M model, not its 42M one

Paper Table S1 gives the action decoder as
`[4096, 4096, 2048, 2048, 1024, 1024, 512, 512]`. The release's own
`config.yaml:152` gives `decoders.g1_dyn` as
`[2048, 2048, 1024, 1024, 512, 512]` — six layers, top width 2048 — and the
critic matches that smaller stack. Encoders agree with Table S1 at
`[2048, 1024, 512, 512]`.

Parameters on the tracking path (g1 encoder plus action decoder, biases in):

| | encoder | decoder | total |
| --- | ---: | ---: | ---: |
| paper Table S1 | 4.23 M | 37.39 M | **41.6 M** |
| released `last.pt` | 4.23 M | 10.18 M | **14.4 M** |

The paper scales "from 1.2M to 16M to 42M parameters", and Table S1's dims
reproduce the 42M headline to three significant figures. The release is the
16M rung. Its config also carries `num_learning_iterations: 100000` where the
paper trains 50k, so it is a different run as well as a different size.

The paper's own size ladder on test-content is 1.2M at 98.0% / 27.7 mm and 42M
at 99.6% / 23.8 mm. A 16M model belongs between them, which is where every
measurement of the released checkpoint in this repo sits.

**No clip selection will make `last.pt` produce 22.5 mm.** It already produces
the number the paper reports for a model of its size.

### Correction (2026-08-25, later the same day): the 42M weights ARE public

An earlier version of this page said the 42M weights were not released. Wrong.
The HF repo carries three checkpoints, and their `config.yaml` files settle it:

| HF path | `decoders.g1_dyn` hidden dims | reading |
| --- | --- | --- |
| `sonic_release/` | `[2048, 2048, 1024, 1024, 512, 512]` | 16M rung |
| `sonic_v1_1/` | `[4096, 4096, 2048, 2048, 1024, 1024, 512, 512]` | **Table S1, 42M** |
| `low_latency/` | `[4096, 4096, 2048, 2048, 1024, 1024, 512, 512]` | Table S1, deployment variant |

So the paper's architecture is public as `sonic_v1_1`, and this campaign scores
it. `low_latency` is unscored; it shares v1.1's architecture and is a
deployment variant, not a separate quality point.

## Finding 2 — clip selection is not the lever

Every clip of the two stored clean 4,096-blocks was joined to its BONES-SEED
`category` through the selection JSON (129,779 of 129,785 join) and reweighted
to each Table 2 column:

| block | raw | reweighted to test-repetition | reweighted to test-content |
| --- | --- | --- | --- |
| ranks 12288-16383 | 0.9946 / 25.90 mm | 0.9950 / 25.80 mm | 0.9961 / 26.01 mm |
| ranks 20480-24575 | 0.9937 / 25.86 mm | 0.9940 / 25.50 mm | 0.9954 / 25.70 mm |

Mixture matching moves the number by 0.1-0.4 mm. A random block from this
corpus is already mixture-neutral. Both reweighted test-repetition figures land
on Table 4(a)'s 99.6% / 25.5 mm.

To reach 23.8 mm by pruning instead, you have to delete whole capability
classes:

| prune | n | SR | MPJPE-L |
| --- | ---: | ---: | ---: |
| full block | 4096 | 0.9946 | 25.90 |
| − Stunts, Unusual Locomotion | 4019 | 0.9955 | 25.63 |
| − those, Advanced Locomotion | 3826 | 0.9956 | 24.62 |
| − those, Sports | 3707 | 0.9965 | 24.43 |
| − those, Dancing | 3402 | 0.9971 | 23.78 |

Figure S2 shows hip-hop dance, high jump and kick running on the real G1.
Deleting Dance to hit 23.8 is the same "selecting for ease" error the
2026-08-17 campaign made and deleted. Do not rebuild it.

## Finding 3 — deployability is already applied, and it is a keyword list

SONIC filters "physically implausible motions (e.g. stair climbing, seated
activities)" from 700 h down to 611 h before training. That filter is the
40-keyword list in `gear_sonic/data_process/filter_and_copy_bones_data.py`
(`bed`, `chair`, `climb`, `sitting`, `stair`, `ladder`, `table`, `handstand`,
…), which `scripts/data/select_bones_seed_sonic.py` already applies verbatim:
142,220 clips to 129,785. Surviving crawl (1,874) and kneel (1,283) clips are
motions SONIC deploys on hardware.

So the corpus already IS SONIC's deployable corpus by SONIC's own definition.
There is no further deployability filter to apply.

## Two hypotheses checked and dropped

- **MPJPE-L frame convention.** SONIC's `mpjpe_l` comes from
  `smpl_sim.smpllib.smpl_eval.compute_metrics_lite`
  (`im_eval_callback.py:598`), where it is `jpos - jpos[:, root_idx]` on both
  sides — root translation only, no rotation cancelled. Identical to our
  `contracts/tracking_metrics.py`. Their `body_names`
  (`commands/terms/motion.yaml:52`) are our 14 links verbatim, `anchor_body` is
  `pelvis`, and `extend_config` is empty. Their rewards and terminations use a
  heading-aligned root frame (`get_heading_q`), matching our
  `reroot_body_positions`. Frames are fully aligned; this explains nothing.
- **Reference time base.** `convert_soma_csv_to_motion_lib.py` downsamples by
  integer stride, and `int(120/50) = 2` would leave 60 fps of samples stamped
  50 fps, a 1.2x slowdown. Dropped: `torch_humanoid_batch.py:342`
  `interploate_pose` is a proper time-based lerp/slerp resample driven by the
  PKL's stored `fps`, the converter's default is `--fps 30` (120 to 30 is an
  exact stride 4), and the production converter `process_bones_to_motionlib.py`
  is not in the public release, so the stored `fps` is not observable. The
  120 fps source is confirmed: `move_duration_frames` over all 142,220 rows
  totals 124,554,605 frames against a published 288 hours, i.e. 120.13 fps.

## The board this produced

`sonic_proxy_testrep4096_v1`: 4,096 clips drawn from the whole corpus at the
Table 2 test-repetition main-category mixture — Locomotion 42.7%, Gestures
17.9%, Others 14.2%, Injured 8.4%, Dance 7.7%, Action/Tool 5.1%, Props 4.0%.
Realized shares match the target to four decimals.

SONIC's two remaining Table 2 groups are dropped and the rest renormalized:
Acting/Roleplay and Combat are essentially absent from the public BONES-SEED
release (16 "Martial Arts" and 4 "Magic" clips against SONIC's 50,162 combat
clips) and hold 20 and 0 clips in that split. Injured is carved out by name
prefix, recovering 10,558 clips against SONIC's 11,081 across all splits,
because BONES-SEED spreads injured motions across six `category` values.

No difficulty band, unlike `bones_testbed4096_v1`: the paper's splits are not
difficulty-filtered. 120 clips overlap the legacy block.

This board is **not** a comparison board for our own arms. Score those on
`bones_testbed4096_v1`. This one exists to face a named paper column.

## Result (2026-08-25)

Released checkpoint `sonic_release/last.pt`, SHA-256 `e6bdab3f…`, clean
protocol (`paper_sonic_proxy_testrep4096_v1`): Newton/MJWarp, seed 0,
deterministic `mode` actions, frame-0 starts, randomization off, SONIC
thresholds, `foot_pos_xyz` and `base_too_low` off.

| | SR | MPJPE-L (micro, success-only) | MPJPE-G |
| --- | ---: | ---: | ---: |
| **released `last.pt`, clean** | **0.9924** | **25.63 mm** | 150.94 mm |
| SONIC Table 4(a), test-repetition | 0.996 | 25.5 mm | — |
| released `last.pt`, `no_push` | 0.9917 | 28.16 mm | 201.94 mm |

Randomization costs 2.53 mm of MPJPE-L and 0.0007 of success rate here, matching
the 2.3-2.8 mm and sub-0.001 band measured on the testbed and the legacy block.
Quality is randomization-sensitive, success is not, which is why the clean row
is the one that faces the paper.

**0.13 mm and 0.36 points of success rate.** The harness reproduces the paper
row that belongs to the model the checkpoint actually is. The offline
reweighting prediction was 25.5-25.8 mm, so the measurement also confirms that
mixture matching, not clip pruning, is what the board does.

Run integrity (clean row): `steps_run` 9,006 of a 10,000 cap so no clip truncated,
`done_rate` 1.000, 4,065 clips completed and 31 failed, scored ranks equal to
the frozen `SONIC_PROXY_TESTREP4096_RANKS`
(`trajectory_ranks_sha256` `7ad425d7…`). Result:
`logs/sonic_paper_proxy/sonic_release_sonic_proxy_testrep4096_v1_rand_none.json`.

What this row does **not** license: comparing it to Table 4(c)'s 22.5 mm, or
reading the 3 mm between them as a protocol defect. That is the 16M-to-42M
model gap.

## Our best tracker on this board (2026-08-25)

`ln_hold1_sonicreset` at 30B frames, the leader of the legacy scoreboard
(0.9707 / 21.75 mm there), scored by `score_arms_proxy.sh`. Same clean
protocol, same 4,096 clips, same 14 links, one seed. Encoder binding verified
in-run: `live_vs_checkpoint_encoder_max_abs` 0.0.

All rows: clean protocol, 4,096 clips, seed 0, one seed each.
`sonic_v1_1` SHA-256 `af24831a…`, `sonic_release` SHA-256 `e6bdab3f…`.

| row | SR | MPJPE-L | MPJPE-G |
| --- | ---: | ---: | ---: |
| **ours, `ln_hold1_sonicreset` 30B** | 0.9722 | **20.09 mm** | **104.08 mm** |
| `sonic_v1_1` (42M, Table S1 arch) | **0.9932** | 24.35 mm | 206.00 mm |
| `sonic_release` (16M) | 0.9924 | 25.63 mm | 150.94 mm |
| ours, `no_push` | 0.9658 | 22.18 mm | 178.54 mm |
| `sonic_v1_1`, `no_push` | 0.9927 | 26.72 mm | 247.37 mm |
| `sonic_release`, `no_push` | 0.9917 | 28.16 mm | 201.94 mm |

Success-only MPJPE across different success rates scores different subsets, so
the table above does not rank anything by itself. Matched subsets:

| pair, clips both complete | MPJPE-L | MPJPE-G |
| --- | ---: | ---: |
| ours vs `sonic_v1_1`, 3,972 clips | **20.03** vs 23.93 mm | **103.84** vs 185.41 mm |
| ours vs `sonic_release`, 3,968 clips | **20.03** vs 25.32 mm | **103.92** vs 140.24 mm |
| `sonic_v1_1` vs `sonic_release`, 4,057 clips | **24.29** vs 25.60 mm | 206.20 vs **150.39** mm |

Read from this:

1. **Our advantage survives the bias correction** against both checkpoints
   (20.09 to 20.03 mm), so it is not an artifact of scoring an easier subset.
2. **The two families trade, they do not rank.** We track 3.9 mm tighter than
   v1.1 locally and 82 mm tighter globally; v1.1 completes 96 clips we fail
   while we complete 10 it fails. Our failure mix is 80 `ee_body_pos`, 26
   `anchor_pos`, 18 `anchor_ori`, with `ee_body_pos` far below the 274-406 the
   10B arms show on the testbed — the SONIC-reset sampler working.
3. **v1.1 is not a strict improvement on the release.** It is 1.3 mm better on
   MPJPE-L at level completion (11 clips gained, 8 lost) and **56 mm worse on
   MPJPE-G**. The bigger decoder buys local precision and spends global
   anchoring. Anyone quoting "the newer SONIC is better" should say on which
   metric.
4. **Where each lands against the paper.** `sonic_release` sits 0.13 mm from
   Table 4(a)'s 99.6% / 25.5 mm; `sonic_v1_1` sits 1.85 mm from Table 4(c)'s
   99.8% / 22.5 mm. The 16M row is the tighter reproduction, which is expected:
   architecture alone does not make v1.1 the paper's snapshot, and its changed
   encoder orientation contract says it is not.

What none of this shows: that `sonic_v1_1` is better or worse than the model
the paper evaluated. The paper's number is on a split we do not have, so the
comparison is not available. What IS measured is that v1.1 is the stronger of
the two public checkpoints on MPJPE-L and success rate on identical clips.

## Run it

```bash
./experiments/campaigns/2026-08-25-sonic-paper-proxy/run_proxy_rows.sh
./experiments/campaigns/2026-08-25-sonic-paper-proxy/run_proxy_rows.sh --report
```

About 12 minutes per row on one RTX PRO 6000. Results land in
`logs/sonic_paper_proxy/`.

## The deployment-set reconstruction, and why 22.3 mm is unreachable by selection

Asked 2026-08-25: select a subset of hardware-deployable motions on which a
public checkpoint reproduces the paper's 100% / 22.3 mm deployment row.

Built `sonic_deploy123_v1`: a balanced 123-clip draw across the ten families
SONIC names as deployed — Figure S2's caption gives hip-hop dance, stage bow,
high jump, kick, crouch walk and grovel; the project page adds squatting,
kneeling, hand crawling, elbow crawling and boxing (hand and elbow crawl merge,
BONES-SEED names do not distinguish them). Criterion fixed from the paper
before any score was read; one draw, seed 20260825. Clean protocol.

| | SR | MPJPE-L | MPJPE-G |
| --- | ---: | ---: | ---: |
| `sonic_v1_1` (42M) | 0.9675 | 36.76 mm | 381.50 mm |
| ours, `ln_hold1_sonicreset` 30B | 0.8862 | 38.49 mm | 210.39 mm |
| `sonic_release` (16M) | 0.9919 | 42.72 mm | 221.21 mm |
| SONIC paper, its 123 deployment clips | 1.000 | **22.3 mm** | — |

On the 107 clips all three complete: ours 38.14, `sonic_v1_1` 36.46,
`sonic_release` 43.02 mm.

**The answer is no, and the sign is the interesting part.** Selecting the
motions SONIC deploys on hardware makes the board *harder*, not easier — every
tracker lands at 36-43 mm against 20-26 mm on the 4,096-clip board. Crawl,
crouch, kneel and dance were already the worst families in the per-category
breakdown (30-46 mm); a board made of nothing else is the hardest population we
have built.

So there is no subset of deployable motions that yields 22.3 mm. The only
selections that reach 22-23 mm are the ones that *delete* these families, which
is the 2026-08-17 ease-selection error — and this run is its mirror image:
that deleted board scored 21.92 mm precisely because it excluded squat, kneel
and crawl.

What this leaves unexplained is SONIC's own 22.3 mm at 100% on 123 clips
spanning these families. It is not reachable from our corpus by any
family-based draw, so either their deployment clips are much easier instances
within those families, or something outside clip selection separates them.
Their set is not enumerated, so this cannot be settled.

`sonic_deploy123_v1` is registered as a DIAGNOSTIC board with profile
`diag_sonic_deploy123_v1`. Do not publish a row from it: 123 clips carries
large population noise, and the family list is a partial reading of a caption
that says "representative".

## Can a smaller subset match 22.3 mm with v1.1? Yes, and that is why it means nothing

Asked 2026-08-25, after the deployment-family board came out at 36.76 mm: search
for some other subset on which `sonic_v1_1` reads 22.3 mm. Done offline on the
stored per-clip results, no new rollout. Tool:
`imitation_experiments.evaluation.subset_sensitivity`.

**How hard is the target?** On the 4,068 clips `sonic_v1_1` completes on
`sonic_proxy_testrep4096_v1`, **51.9% of individual clips already sit below
22.3 mm**, and random 123-clip subsets give 24.38 +/- 1.07 mm. The target is a
z of -1.93, reached by **2.2% of random draws**. It is a two-sigma draw, not an
extreme one.

**How hard is it to find a "principled" rule that lands there?** `RULE_GRID_V1`
is a bounded grid over four axes the canonical testbed rule already uses —
difficulty cap, frame band, pelvis-height floor, root-speed cap — 512
combinations. Every one retains at least 123 clips. Results span
16.51 to 26.48 mm, and **67 of 512 (13.1%) land within 0.5 mm of 22.3**.

**The closest rule.** Frames 100-300 (2.1-6.0 s) and `root_speed_max <= 1.5`
m/s, no difficulty band, no height floor: **MPJPE-L 22.30 mm**, MPJPE-G
108.91 mm, SR 1.0000 by construction (the pool is success-only).

**It is the 2026-08-17 ease-selection rule in different coordinates.** The
subset keeps **12 of the 123 deployment-family clips** — 5 hip-hop dance,
5 crouch walk, 2 kneel — and **zero** crawl, squat, boxing, high jump, kick or
grovel. The 08-17 rule selected on "upright and moderate speed"; this one
selects on "short and slow". Same species, same objection.

Conclusion: a subset match to 22.3 mm carries no evidential weight. It is
reachable by chance at 2.2%, by a modest rule search at 13.1%, and the rule
that gets closest achieves it by removing the motions SONIC actually deploys.
**No board from this search is registered, and no number from it may be
published.**

The one part worth keeping as a hypothesis: every close rule is a SHORT-clip
rule (2-6 s against the board's 5.8 s median). If SONIC's 123 hardware demos
are short takes — which is what you would run repeatedly on a robot — that
could explain part of their 22.3 mm without any ease selection on their part.
Their set is not enumerated, so this stays a hypothesis.

Run the guard on any future claim of the form "population P reproduces X":

```python
from imitation_experiments.evaluation.subset_sensitivity import (
    load_clip_scores, null_distribution, rule_grid)
```

## New common eval subset (2026-08-25)

The user directed one transparent exception to the earlier population-search
conclusion: construct a small in-distribution capability subset on which public
`sonic_v1_1` reads on the 23.7 mm scale. This is calibration by trial and
error, not evidence that we recovered SONIC's unpublished split.

The human-facing name is **new common eval subset**. The unchanged artifact ID
`sonic_capability124_v1` freezes 124 ranks at SHA-256
`19b83597f0e7bf86fb462ae691b1dad455bb6b8cc130a9a4c702062aa75de147`.
The direct clean run completed all 124 motions: **1.0000 SR / 23.79 mm
MPJPE-L / 173.92 mm MPJPE-G**, one evaluation. The 30B
`ln_hold1_sonicreset` tracker, seed 0, completed 123 of 124: **0.9919 /
19.44 / 108.58 mm**, preliminary. On the 123 shared successes, SONIC is
23.43 / 148.37 mm and ours is 19.44 / 108.58 mm. The only local-tracker
failure is rank 6364, `kneeling_loop_003_A244`, on `anchor_pos`.

The ordered-rank hash matches in both result files. The local tracker's binding
audit passed all 18 encoder tensors. The 46.5B mid-chain checkpoint of the running 50B chain reads **1.0000 /
19.44 / 122.08 mm** on its own 124 successes and 18.99 / 97.56 mm on the 123
clips all three rows complete; it is a progress read, never the 50B row.
`score_arms_capability124.sh` is the launcher for our own arms on this
subset. Artifacts are under
`logs/sonic_capability124_v1/`; the selection contract and exact rerun command
are in `wiki/sonic-v1_1-subsets.md`. Motion-name review of all 124 clips and
full-horizon videos of the nine ambiguous names passed without a fall or a
scene dependency; the ranks remain unchanged. Repeated runs remain open. Never
call this subset held out, unbiased, or SONIC's evaluation set.

## How to cite SONIC after this

**Do not cite SONIC's numbers for the SONIC row — measure them.** Both public
checkpoints run in our environment, on identical clips, under one protocol,
with an actor verified bitwise against their released ONNX. Use `sonic_v1_1`
as the headline baseline (strongest public checkpoint, and it carries Table
S1's architecture) and keep `sonic_release` as the harness-calibration row.

The justification paragraph:

> We compare against the strongest publicly released SONIC checkpoint
> (`sonic_v1_1`, carrying the architecture specified in the paper's Table S1),
> evaluated in our environment under an identical protocol on an identical clip
> set. We validate the harness by reproducing SONIC's own reported figures for
> both public checkpoints — `sonic_release` to 0.13 mm of Table 4(a) and
> `sonic_v1_1` to 1.85 mm of Table 4(c) — on a population matched to the
> category composition of their test-repetition split.

Say in the same sentence that the populations differ: same corpus, our
sampling, and our board is in-distribution for every tracker here.

For BeyondMimic, GMT and Any2Track, which we have not run, use the
anchored-ratio rule in `wiki/canonical-paper-metrics.md`. Decided 2026-08-25:
no native baselines will be trained.

One factual guard: **the SONIC paper reports no PHC numbers.** PHC (Luo et al.
2023b) is cited only for the tracking formulation and as the success convention
the paper departs from. PHUMA (Lee et al. 2025) is a 68,000-motion *dataset*,
not a method. The baselines carrying quotable numbers are BeyondMimic
(81.6/85.8/73.4% success, 39.1 mm MPJPE-L) and Any2Track (31.1/38.4/58.6%
success); GMT appears only as bars in Figure 2(d-g).
