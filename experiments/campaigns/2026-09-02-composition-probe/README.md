# 2026-09-02 -- skill-composition probes

Does a phi that is affine in `z` give a tracker that composes skills better
than the concat phi? Three tests on the same 60 clip pairs, one job per
(arm, test), scored offline. Plan of record: the four-test design of
2026-09-02 (held mix, handover, off-manifold, chains); this campaign runs
the first three. Chains and the additive form come after these rows.

| test | settings | question |
|---|---|---|
| `held_alpha` | a in {0, .25, .5, .75, 1} held from step 0, 300 steps | is behaviour monotone in a (speed, stride frequency, arm swing), fall-free at every a? |
| `handover` | switch at {150, 160, 170} with ramp {0, 10, 50} | fall-free handover rate, settling time to the source's speed, peak action step, gait distance to the source robot |
| `extrapolate` | a in {-0.5, 1.25, 1.5, 2.0} held | graceful degradation off the segment |

Arms: `lstm` (concat phi) and `lstm_affine` (affine phi) from
`2026-09-02-lstm-hub64-10b` at their newest checkpoint (10B). Same trunk,
same recipe, same budget; the two encoders are different pretrains, so
encoder-init noise rides on every difference. `combo` joins when its 10B
finishes.

Pairs (`pairs.json`): six kinds x 10 (walk->jog, walk->turn, walk->stand,
stand->wave, crouch->walk, jog->turn), clips >= 320 frames that both LSTM
arms complete on the 4,096 board at 4.5B, seed 0. A clip may recur across
pairs; the driver packs pairs into processes so no rank repeats inside one
process (the evaluator pins one rank per environment).

Mechanics: `scripts/rlopt/eval_composition_probe.py` (the `eval*.py` name
selects the container's CU130 torch branch; the first submission under
another name died on an NCCL symbol) ->
`imitation_experiments.evaluation.composition_probe` -> one
`evaluate_checkpoint` process per setting through `scripts/rlopt/run_evaluator.py`,
`--latent_blend_layout pairs`, reference-relative terminations off, Newton.
Summaries: `/data/eval/composition_probe/<arm>/<test>/<arm>_<setting>_chunk<k>.json`
plus `index.json`. Score:

```bash
pixi run python -m imitation_experiments.evaluation.composition_metrics \
    --results logs/composition_probe_mirror --out logs/composition_probe_mirror/table.md
```

Submit:

```bash
for arm in lstm_affine_held lstm_held lstm_affine_handover lstm_handover lstm_affine_extrapolate lstm_extrapolate; do
  pixi run python -m imitation_experiments.pipeline.cluster plan \
    --campaign experiments/campaigns/2026-09-02-composition-probe/campaign.yaml --arm $arm --seed 0
  # then the printed submit line
done
```

Resolution: fall-free rates on 60 pairs resolve differences of roughly 15%
relative and no less; one seed per arm; encoder-init noise unmeasured on
these metrics.

## Rows: held mix and extrapolation (2026-09-03, partial)

Jobs 5627479 / 5627480 / 5627484 / 5627486. The first chunk of the first
setting died in the Kit startup crash on three of the four jobs (alpha 0 on
both held jobs, alpha -0.5 on `lstm` extrapolate), so those rows have 40 of
60 pairs; refills 5627647-5627649 are running. Handover jobs still running.
One seed per arm, 300 steps, Newton, means over pairs.

| test | alpha | arm | n | fall-free | speed post | stride Hz post | arm swing post | action delta post | gait dist to source | code dist |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| held | 0 | lstm | 40 | 0.975 | 0.549 | 0.783 | 0.618 | 0.911 | 0.266 | 11.1 |
| held | 0 | lstm_affine | 40 | 0.975 | 0.549 | 0.771 | 0.617 | 0.933 | 0.267 | 8.4 |
| held | 0.25 | lstm | 60 | 0.933 | 0.451 | 0.772 | 0.511 | 0.952 | 0.208 | 11.8 |
| held | 0.25 | lstm_affine | 60 | 0.950 | 0.469 | 0.786 | 0.509 | 0.983 | 0.205 | 8.6 |
| held | 0.5 | lstm | 60 | 0.950 | 0.376 | 0.767 | 0.554 | 1.178 | 0.135 | 12.5 |
| held | 0.5 | lstm_affine | 60 | 0.950 | 0.400 | 0.758 | 0.527 | 1.191 | 0.135 | 8.8 |
| held | 0.75 | lstm | 60 | 0.933 | 0.456 | 0.772 | 0.632 | 1.088 | 0.078 | 13.2 |
| held | 0.75 | lstm_affine | 60 | 0.933 | 0.459 | 0.767 | 0.622 | 1.035 | 0.078 | 9.2 |
| held | 1 | lstm | 60 | 0.933 | 0.557 | 0.733 | 0.697 | 0.961 | 0.041 | 13.7 |
| held | 1 | lstm_affine | 60 | 0.933 | 0.555 | 0.744 | 0.706 | 0.965 | 0.041 | 9.2 |
| extrapolate | -0.5 | lstm | 40 | 0.625 | 0.607 | 0.750 | 1.173 | 2.054 | 0.349 | 11.6 |
| extrapolate | -0.5 | lstm_affine | 60 | 0.650 | 0.611 | 0.750 | 1.047 | 1.978 | 0.344 | 8.5 |
| extrapolate | 1.25 | lstm | 60 | 0.883 | 0.728 | 0.756 | 0.773 | 1.671 | 0.074 | 13.7 |
| extrapolate | 1.25 | lstm_affine | 60 | 0.917 | 0.684 | 0.747 | 0.808 | 1.456 | 0.070 | 9.3 |
| extrapolate | 1.5 | lstm | 60 | 0.767 | 0.981 | 0.758 | 0.858 | 2.553 | 0.117 | 13.7 |
| extrapolate | 1.5 | lstm_affine | 60 | 0.800 | 0.777 | 0.719 | 0.857 | 2.134 | 0.106 | 9.1 |
| extrapolate | 2.0 | lstm | 60 | 0.050 | 0.799 | 0.697 | 1.124 | 3.864 | 0.217 | 13.5 |
| extrapolate | 2.0 | lstm_affine | 60 | 0.300 | 0.860 | 0.739 | 1.038 | 3.888 | 0.189 | 8.8 |

Per-pair monotonicity across the five alphas (fraction of consecutive
alpha steps moving toward the alpha=1 value), pairs with all five alphas:
`lstm` speed 0.76 / stride 0.93 (n=42 with a gait) / arm swing 0.86, fall-free
at every alpha 55/60; `lstm_affine` speed 0.77 / stride 0.96 (n=47) / arm
swing 0.82, fall-free at every alpha 56/60. Fall-free at alpha 0.5 by kind is
identical on both arms: 10/10 on every kind except stand->wave 7/10.
Source-robot speeds in the post window average 0.54 m/s (the source clips
include turns, stands and crouches), so "speed post" at alpha 1 matches the
source and the alpha 0.5 mix runs slower than both ends.

## Rows (2026-09-02 evening, jobs 5627479-5627486)

Rows are per (pair, setting) episodes; "post" is the window after the ramp
(held mixes: the whole 300 steps). Fall-free = no step with uprightness below
0.5. Gait distance = mean |joint_t - joint_s| (rad) against the source robot at
the same step. Settled = target speed within 15% of the source's post-window
speed for 25 consecutive steps. Stride monotonicity counts only pairs with a
finite stride estimate at every alpha. The a=0 held rows and the concat
a=-0.5 row are missing one 20-pair chunk (the job's first evaluator process
died in the Kit startup flake; resubmitted as 5627651-53). One seed per arm,
one tracker per phi, so encoder-init noise is not separated from the phi.

Held mix (test 1):

| arm | alpha | n | fall-free | speed post (m/s) | stride Hz | arm swing (rad) | action delta | gait distance to source |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| affine | 0.00 | 40 | 0.975 | 0.549 | 0.771 | 0.617 | 0.933 | 0.267 |
| affine | 0.25 | 60 | 0.950 | 0.469 | 0.786 | 0.509 | 0.983 | 0.205 |
| affine | 0.50 | 60 | 0.950 | 0.400 | 0.758 | 0.527 | 1.191 | 0.135 |
| affine | 0.75 | 60 | 0.933 | 0.459 | 0.767 | 0.622 | 1.035 | 0.078 |
| affine | 1.00 | 60 | 0.933 | 0.555 | 0.744 | 0.706 | 0.965 | 0.041 |
| concat | 0.00 | 40 | 0.975 | 0.549 | 0.783 | 0.618 | 0.911 | 0.266 |
| concat | 0.25 | 60 | 0.933 | 0.451 | 0.772 | 0.511 | 0.952 | 0.208 |
| concat | 0.50 | 60 | 0.950 | 0.376 | 0.767 | 0.554 | 1.178 | 0.135 |
| concat | 0.75 | 60 | 0.933 | 0.456 | 0.772 | 0.632 | 1.088 | 0.078 |
| concat | 1.00 | 60 | 0.933 | 0.557 | 0.733 | 0.697 | 0.961 | 0.041 |

Monotonicity in alpha (test 1):

| arm | pairs | speed monotone fraction | stride monotone fraction | arm-swing monotone fraction |
|---|---:|---:|---:|---:|
| affine | 60 | 0.769 | 0.956 (n=47) | 0.819 |
| concat | 60 | 0.761 | 0.935 (n=42) | 0.858 |

Extrapolation (test 3a):

| arm | alpha | n | fall-free | speed post | action delta | gait distance |
|---|---:|---:|---:|---:|---:|---:|
| affine | -0.50 | 60 | 0.650 | 0.611 | 1.978 | 0.344 |
| affine | 1.25 | 60 | 0.917 | 0.684 | 1.456 | 0.070 |
| affine | 1.50 | 60 | 0.800 | 0.777 | 2.134 | 0.106 |
| affine | 2.00 | 60 | 0.300 | 0.860 | 3.888 | 0.189 |
| concat | -0.50 | 40 | 0.625 | 0.607 | 2.054 | 0.349 |
| concat | 1.25 | 60 | 0.883 | 0.728 | 1.671 | 0.074 |
| concat | 1.50 | 60 | 0.767 | 0.981 | 2.553 | 0.117 |
| concat | 2.00 | 60 | 0.050 | 0.799 | 3.864 | 0.217 |

Handover, pooled over switch step 150/160/170 (test 2):

| arm | ramp (steps) | n | fall-free | settled rate | median settle (steps) | peak action step after switch | gait distance post |
|---|---:|---:|---:|---:|---:|---:|---:|
| affine | 0 | 180 | 0.933 | 0.21 | 26 | 4.54 | 0.067 |
| affine | 10 | 180 | 0.933 | 0.23 | 28 | 4.34 | 0.061 |
| affine | 50 | 180 | 0.939 | 0.18 | 37 | 3.52 | 0.050 |
| concat | 0 | 180 | 0.928 | 0.22 | 41 | 5.26 | 0.064 |
| concat | 10 | 180 | 0.928 | 0.22 | 37.5 | 4.77 | 0.059 |
| concat | 50 | 180 | 0.922 | 0.20 | 47 | 3.72 | 0.050 |
