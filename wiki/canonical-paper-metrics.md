# Canonical paper-facing evaluation metrics

Status: frozen 2026-08-17. Supersedes the per-campaign metric choices that
produced the mixed table in `wiki/current-status.md`.

This page defines what a paper-facing number in this repo means, which
population it is measured on, and what it may be compared against. Registry:
`imitation_experiments.evaluation.protocol` (protocols, boards, profiles) and
`imitation_experiments.evaluation.clip_features` (the testbed rule and the
difficulty index). Reduce a result file with
`python -m imitation_experiments.evaluation.summarize_paper_boards`.

## The three numbers, always together

A paper row is **success rate, success-only micro MPJPE-L, and success-only
micro MPJPE-G**. None of the three may be published alone.

- **Success rate** uses SONIC's published definition verbatim: a motion fails
  when root height error or end-effector height error exceeds 0.25 m, or root
  orientation error exceeds 1 rad. `foot_pos_xyz` and `base_too_low` are off.
  An episode succeeds by reaching the end of its reference clip.
- **MPJPE-L** is root-position-subtracted mean per-joint position error over
  the 14 tracked links (pelvis, hips, knees, ankles, torso, shoulders, elbows,
  wrists), in millimetres, frame-weighted over successful episodes only
  ("micro"). SONIC defines its metric over the same 14 links.
- **MPJPE-G** is the same error in the world frame, unaligned. It is mandatory
  because MPJPE-L flatters a policy that holds its pose while drifting: the
  released SONIC checkpoint scores 28.75 mm local against 135.73 mm global on
  the testbed.

Success-only MPJPE is **not comparable across different success rates**. A
19 mm figure at SR 0.60 is the easy 60% of the board, not a better tracker.
Print the success rate in the same sentence as the millimetres.

## The canonical testbed

`bones_testbed4096_v1` — 4,096 clips drawn from the whole 129,785-clip corpus,
frozen as `TESTBED4096_RANKS`. New work is scored here.

| profile | randomization | what it answers |
| --- | --- | --- |
| `paper_testbed4096_v1` | none | headline quality and success |
| `paper_testbed4096_robust_v1` | `no_push` | cost of startup and reset randomization |

Common to both: Newton/MJWarp, seed 0, deterministic `mode` actions, frame-0
starts, episode horizon = reference length, `episode_length_s` large enough
that no clip truncates.

### How the population is chosen

`TESTBED_CLIP_RULE_V1`, applied to reference kinematics and clip names only —
no policy, rollout, or score touches the selection:

1. **Drop scene-dependent clips.** A motion whose reference is conditioned on
   an object or terrain the evaluation scene does not contain (crate, door,
   handle, lever, switch, wall, car, horse, ladder, …) scores a behavior the
   robot has no reason to be able to make. 6,121 clips, 4.7% of the corpus.
2. **Drop artifacts.** Under 100 or over 1,500 frames; reference pelvis below
   the floor.
3. **Drop the easiest quarter of what remains** by `difficulty_index` — the
   mean of four percentile ranks over the retained corpus: how low the pelvis
   goes, how fast the root travels, how fast the joints move, how high the feet
   rise. Equal weights on purpose; fitting them to a tracker's error would make
   the board a function of that tracker. Validated, not fitted: Spearman +0.46
   against the released checkpoint's per-clip MPJPE-L, deciles rising
   monotonically from 16.3 mm to about 30 mm. The dropped quarter sits at
   16-20 mm with saturated success for every tracker measured so far.
4. **Sample 4,096** from the 93,173 survivors at `random.Random(20260818)`.

**Squatting, kneeling, crawling and boxing stay in.** SONIC's project site
shows all of them deployed on real hardware, so an upright-only board would
measure ease, not capability. An earlier "hardware-plausible" board that
excluded them was built and deleted on 2026-08-17; see the campaign README.

### What the testbed is not

- **Not held out from training.** Every tracker in this repo trains on the full
  129,785-clip tree with no rank filter. Every board here is in-distribution.
  A held-out claim requires a training-side split we do not have.
- **Not SONIC's evaluation set.** Same underlying corpus (BONES-SEED is the
  SONIC keyword-filtered set, selection SHA `e714bbff…`, and the G1 retargeting
  is upstream's), but our population and our sampling.

## Legacy boards

| board | what it is |
| --- | --- |
| `bones_scoreboard4096_v1` | ranks 12288-16383. Every pre-2026-08-17 row sits here, under `no_push`. Kept so those rows stay interpretable. |
| `bones_heldout4096_v1` | ranks 20480-24575. A disjoint block of the same shape, for showing a claim survives a different population. |

130 clips are shared between the testbed and the legacy block. Rows from
different boards are different populations and must not share a table column.

## Which SONIC number to compare against

Two separate questions: which row is the SONIC **baseline** for a paper claim,
and which row the **released checkpoint** should reproduce. They are different
models.

**Baseline.** SONIC's 42M flagship on its own held-out splits: test-repetition
**99.8% / 22.5 mm**, test-content **99.6% / 23.8 mm** (Table 4(c), robot
encoder). Its Figure 2 MuJoCo baseline-comparison row is 98.7% / 23.2 mm on
test-content. Cite one of these, and say in the same sentence that the
populations differ.

**Not the headline.** The advertised **22.3 mm at 100% success is the 123-clip
hardware deployment set scored in simulation** — never enumerated (Figure S2
carries no names or IDs) and shown on the project page to include squatting,
kneeling and crawling. Nothing in this repo compares against it, and 2026-08-25
established that nothing can: a 123-clip board built from the ten families
SONIC names as deployed scores the released checkpoints at **36.76 mm
(`sonic_v1_1`) and 42.72 mm (`sonic_release`)**, and our 30B arm at 38.49 mm.
Selecting deployable motions makes a board HARDER, because crawl, crouch,
kneel and dance are the worst families in the corpus. The only selections that
reach 22-23 mm delete those families, which is the 2026-08-17 ease-selection
error. See `experiments/campaigns/2026-08-25-sonic-paper-proxy/README.md`.

A follow-up search confirmed the same thing from the other side. On the 4,068
clips `sonic_v1_1` completes, 51.9% of individual clips already sit below
22.3 mm, random 123-clip subsets give 24.38 +/- 1.07 mm (the target is a
z of -1.93, reached by 2.2% of draws), and **67 of a 512-rule grid (13.1%)
land within 0.5 mm of it**. The closest rule keeps only short, slow clips and
retains 12 of 123 deployment-family clips with zero crawl, squat, boxing, high
jump, kick or grovel. **A subset match to a target number is therefore not
evidence of anything**, and `evaluation.subset_sensitivity` exists to measure
that before a claim of the form "population P reproduces X" is believed.

**Released checkpoint.** `sonic_release/last.pt` is **not** the 42M model. Its
action decoder is `[2048, 2048, 1024, 1024, 512, 512]` against Table S1's
`[4096, 4096, 2048, 2048, 1024, 1024, 512, 512]`, i.e. 14.4M parameters on the
tracking path against 41.6M — the paper's 16M rung. Its comparable rows are
Table 4(a): test-repetition 99.6% / 25.5 mm and test-content 99.3% / 26.6 mm.
Every clean 4,096-clip row we have for it lands at 25.5-26.0 mm, so the harness
reproduces the paper for the model that actually exists publicly. Do not read
the 25.90-vs-23.2 mm difference as a protocol defect; it is a model-size
difference. Details and the two dropped explanations (metric frame, reference
time base) are in `wiki/sonic-release-checkpoint-tier2.md`.

The population that lets a released-checkpoint row name its paper column is
`sonic_proxy_testrep4096_v1` — see
`experiments/campaigns/2026-08-25-sonic-paper-proxy/README.md`. It is a
calibration board, not a comparison board: score our own arms on
`bones_testbed4096_v1`.

## Baselines we have not run: the anchored-ratio rule (2026-08-25)

Decided 2026-08-25: we do **not** train native BeyondMimic, GMT or Any2Track
baselines. Their numbers may enter the paper only as ratios against a shared
SONIC anchor, never as absolute values in a column beside ours.

**Never do this.** A table cell holding BeyondMimic's 39.1 mm beside our
20.09 mm stacks four unmatched axes at once: simulator (their comparison is
MuJoCo, ours is Isaac Lab / Newton), training corpus (BeyondMimic and Any2Track
on LaFAN1, GMT on AMASS, ours on BONES-SEED), evaluation population (their
unpublished splits, our board), and model. The SONIC paper says so itself:
"this comparison should be interpreted primarily as evidence of cross-dataset
generalization and scaling effects rather than as a fully data-matched
benchmark."

**Do this instead.** SONIC appears on both sides, so quote each system relative
to its own SONIC row:

| system | measured where | MPJPE-L relative to its SONIC anchor |
| --- | --- | ---: |
| BeyondMimic | SONIC Fig. 2(d-g), MuJoCo, test-content | 39.1 / 23.2 = **1.69x** |
| ours, `ln_hold1_sonicreset` 30B | our env, proxy board, matched subset | 20.03 / 23.93 = **0.84x** |

Success rate transports the same way: BeyondMimic 81.6 / 98.7 = 0.83x of its
anchor; ours 0.9722 / 0.9932 = 0.98x of ours.

**Why the transport is licensed, and by how much.** The anchor is only useful
if a SONIC row is stable across simulator and population. That is measured, not
assumed: on `sonic_proxy_testrep4096_v1`, `sonic_release` lands 0.13 mm from
its Table 4(a) row and `sonic_v1_1` lands 1.85 mm from its Table 4(c) row.
So the anchor transports to within about **8% of MPJPE-L**. A 1.69x against a
0.84x is a 2x separation and survives that band; anything inside about 1.1x of
another row does not, and must not be claimed.

Always label each row with its simulator, training corpus, evaluation
population and model size, and keep the borrowed rows in a visually separate
block. The standing wording rule in
`wiki/whole-body-vla-literature-review.md` still holds for these three: they
are not our baselines and we have run none of them. It no longer holds for
SONIC's tracker, where we run the released weights under their own contract
with an actor verified bitwise against their ONNX.

## Released-SONIC reference rows (2026-08-17)

Regenerate with
`experiments/campaigns/2026-08-17-paper-metric-canon/run_sonic_reference_rows.sh`.
Checkpoint `sonic_release/last.pt`, SHA-256 `e6bdab3f…`.

| board | randomization | SR | MPJPE-L (micro) | MPJPE-G |
| --- | --- | --- | --- | --- |
| **testbed 4,096** | **none** | **0.9912** | **28.75 mm** | 135.73 mm |
| testbed 4,096 | `no_push` | 0.9905 | 31.06 mm | 192.93 mm |
| legacy block 4,096 | none | 0.9946 | 25.90 mm | 117.98 mm |
| legacy block 4,096 | `no_push` | 0.9934 | 28.66 mm | 175.59 mm |
| second block 4,096 | none | 0.9937 | 25.86 mm | 131.49 mm |

Read from this table:

1. **The testbed is harder than the legacy block by 2.85 mm at matched
   randomization**, and costs 0.0034 of success rate. That is the difficulty
   band doing its job. Expect every existing number in this repo to look worse
   when re-scored here; that is a change of population, not of tracker.
2. **Randomization costs 2.3-2.8 mm of MPJPE-L and under 0.001 of success
   rate** (testbed 28.75 -> 31.06; legacy block 25.90 -> 28.66). Quality is
   randomization-sensitive; success is not. That is why the headline runs clean
   and the robustness row runs `no_push`. It also costs 42% of MPJPE-G on the
   testbed (135.73 -> 192.93 mm): randomization mostly buys global drift.
3. **Two noise floors, different quantities.** Population noise between two
   disjoint blocks: 25.90 against 25.86 mm. Run-to-run noise on the identical
   protocol: the `no_push` row was measured on 2026-08-07 and 2026-08-17,
   giving SR 0.9937 / 28.65 mm / MPJPE-G 172.08 mm and SR 0.9934 / 28.66 mm /
   MPJPE-G 175.59 mm. MPJPE-L repeats to about 0.01 mm and success rate to
   about 0.0003, but **MPJPE-G moves by 2%** — treat a small global-error
   difference as unresolved without repeats. None of this licenses reading a
   sub-millimetre difference between two *checkpoints* as real; that needs
   seeds.

## Migrating existing numbers

Every scoreboard row recorded before 2026-08-17 was measured on the legacy
block under `no_push`. Those rows stay valid **as legacy robustness rows on
that board**. They are not headline numbers, they are not on the testbed, and
they must not sit beside a testbed row in the same column. Re-scoring an arm on
the testbed costs about 12 minutes locally.
