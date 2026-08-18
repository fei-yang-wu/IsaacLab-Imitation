# Canonical paper-facing evaluation metrics

Status: frozen 2026-08-17. Supersedes the per-campaign metric choices that
produced the mixed table in `wiki/current-status.md`.

This page defines what a paper-facing number in this repo means, which
population it is measured on, and what it may be compared against. Registry:
`imitation_experiments.evaluation.protocol` (protocols, boards, profiles) and
`imitation_experiments.evaluation.clip_features` (the deployable-clip rule).
Reduce a result file with
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
  released SONIC checkpoint scores 25.90 mm local against 117.98 mm global on
  the canonical block.

Success-only MPJPE is **not comparable across different success rates**. A
19 mm figure at SR 0.60 is the easy 60% of the board, not a better tracker.
Print the success rate in the same sentence as the millimetres.

## The three boards

| profile | board | randomization | what it answers |
| --- | --- | --- | --- |
| `paper_deployable123_v1` | `bones_deployable123_v1` (123 clips) | none | headline quality on hardware-plausible motion |
| `paper_scoreboard4096_v1` | `bones_scoreboard4096_v1` (ranks 12288-16383) | none | breadth over the full motion population |
| `paper_scoreboard4096_robust_v1` | same 4,096 block | `no_push` | cost of startup and reset randomization |

`bones_heldout4096_v1` (ranks 20480-24575) is the falsification partner of the
canonical block, not a reporting row. Use it to show a claim survives a
different population.

Common to all three: Newton/MJWarp, seed 0, deterministic `mode` actions,
frame-0 starts, episode horizon = reference length, `episode_length_s` large
enough that no clip truncates.

## Which SONIC number to compare against

SONIC's headline **22.3 mm MPJPE-L at 100% success is its 123-clip hardware
deployment set scored in simulation**, not a large held-out benchmark. Its
large-set rows are test-content **98.7% / 23.2 mm**, test-repetition 99.6%, and
PHUMA 97.0%.

So: compare `paper_deployable123_v1` against 22.3 mm, and
`paper_scoreboard4096_v1` against 23.2 mm. Comparing a 4,096-clip random block
against 22.3 mm compares two different motion populations and understates every
tracker in this repo by roughly 3.7 mm.

## The deployable-clip rule

`DEPLOYABLE_CLIP_RULE_V1` keeps a clip when, from the **reference motion
alone**:

| axis | bound | intent |
| --- | --- | --- |
| minimum pelvis height | >= 0.65 m | never squats or goes to the ground |
| peak horizontal pelvis speed | <= 2.0 m/s | no sprint or lunge |
| 99th-percentile joint speed | <= 6.0 rad/s | no whipped limb |
| peak ankle height | <= 0.35 m | no high kick or jump |
| length | 150-600 frames | three to twelve seconds at 50 Hz |

1,869 of the canonical block's 4,096 clips pass. The published board is 123 of
them, drawn with `random.Random(20260817)` and frozen as
`DEPLOYABLE123_MOTIONS`.

Nothing in the rule reads a policy, a rollout, or a score, so no checkpoint
influenced the selection. The thresholds were nonetheless *chosen* on the
canonical block, so they were validated unchanged on the disjoint held-out
block before being frozen. Do not retune them; add a `_v2` rule instead.

## Released-SONIC reference rows (2026-08-17)

Regenerate with
`experiments/campaigns/2026-08-17-paper-metric-canon/run_sonic_reference_rows.sh`.
Checkpoint `sonic_release/last.pt`, SHA-256 `e6bdab3f…`.

| board | randomization | SR | MPJPE-L (micro) | MPJPE-G |
| --- | --- | --- | --- | --- |
| deployable-123 | none | **1.0000** | **21.92 mm** | 63.83 mm |
| canonical 4,096 | none | 0.9946 | 25.90 mm | 117.98 mm |
| canonical 4,096 | `no_push` | 0.9934 | 28.66 mm | 175.59 mm |
| held-out 4,096 | none | 0.9937 | 25.86 mm | 131.49 mm |
| deployable-123 | `no_push` | 1.0000 | 25.20 mm | 121.47 mm |

Read from this table:

1. **Our simulation reproduces SONIC's published headline.** 21.92 mm at
   SR 1.0000 against the paper's 22.3 mm at 100%. On the full block, 25.90 mm
   at 0.9946 against the paper's test-content 23.2 mm at 98.7%.
2. **The 4 mm difference between the two boards is motion population, not the
   tracker.** On the canonical block the minimum reference pelvis height has
   Spearman -0.61 against per-clip MPJPE-L; its bottom quintile scores 36.3 mm
   and its top quintile 18.5 mm.
3. **Both noise floors on a 4,096-clip figure are small, and they are
   different quantities.** Population noise between two disjoint blocks:
   25.90 against 25.86 mm. Run-to-run noise on the identical protocol: the
   `no_push` row was measured twice, on 2026-08-07 and 2026-08-17, giving
   SR 0.9937 / 28.65 mm / MPJPE-G 172.08 mm and SR 0.9934 / 28.66 mm /
   MPJPE-G 175.59 mm. So MPJPE-L repeats to about 0.01 mm and success rate to
   about 0.0003, while **MPJPE-G moves by 2%** — treat a small global-error
   difference as unresolved without repeats. None of this licenses reading a
   sub-millimetre difference between two *checkpoints* as real; that still
   needs seeds.
4. **Startup plus reset randomization costs 2.75 mm of MPJPE-L and 0.001 of
   success rate.** Quality is randomization-sensitive; success is not. That is
   why the headline board runs clean and the robustness row runs `no_push`.

## Migrating existing numbers

Every scoreboard row in this repo recorded before 2026-08-17 was measured under
`paper_scoreboard4096_robust_v1` (`no_push`). Those rows stay valid as
robustness rows. They are **not** headline quality numbers and must not be set
beside a clean-randomization figure in the same table. Re-scoring an arm on
`paper_scoreboard4096_v1` costs about 12 minutes locally.
