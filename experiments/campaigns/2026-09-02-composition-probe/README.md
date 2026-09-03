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

## Rows (2026-09-03, complete)

Jobs 5627479-5627486 plus refills 5627647-5627649 for the first-chunk Kit
crashes. 60 pairs per row (180 per pooled handover row), one seed per arm,
300 steps, Newton, fall = uprightness below 0.5.

Handover (switch at 150 / 160 / 170 pooled):

| arm | ramp | n | fall-free | settled (15% band, 25 steps) | settling median (steps, settled only) | peak action step after switch | action delta post | speed post / source | gait dist to source |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| lstm | 0 | 180 | 0.928 | 0.217 | 41 | 5.26 | 1.43 | 0.70 / 0.72 | 0.064 |
| lstm_affine | 0 | 180 | 0.933 | 0.206 | 26 | 4.54 | 1.35 | 0.69 / 0.71 | 0.067 |
| lstm | 10 | 180 | 0.928 | 0.222 | 37.5 | 4.77 | 1.35 | 0.70 / 0.72 | 0.059 |
| lstm_affine | 10 | 180 | 0.933 | 0.228 | 28 | 4.34 | 1.29 | 0.70 / 0.71 | 0.061 |
| lstm | 50 | 180 | 0.922 | 0.200 | 47 | 3.72 | 1.22 | 0.69 / 0.73 | 0.050 |
| lstm_affine | 50 | 180 | 0.939 | 0.183 | 37 | 3.52 | 1.22 | 0.69 / 0.71 | 0.050 |

Held mix:

| alpha | concat fall-free | affine fall-free | concat speed / stride / swing / action delta | affine speed / stride / swing / action delta | code dist concat / affine |
|---:|---:|---:|---|---|---|
| 0 | 0.950 | 0.950 | 0.54 / 0.77 / 0.60 / 0.82 | 0.54 / 0.76 / 0.60 / 0.83 | 11.2 / 8.5 |
| 0.25 | 0.933 | 0.950 | 0.45 / 0.77 / 0.51 / 0.95 | 0.47 / 0.79 / 0.51 / 0.98 | 11.8 / 8.6 |
| 0.5 | 0.950 | 0.950 | 0.38 / 0.77 / 0.55 / 1.18 | 0.40 / 0.76 / 0.53 / 1.19 | 12.5 / 8.8 |
| 0.75 | 0.933 | 0.933 | 0.46 / 0.77 / 0.63 / 1.09 | 0.46 / 0.77 / 0.62 / 1.04 | 13.2 / 9.2 |
| 1 | 0.933 | 0.933 | 0.56 / 0.73 / 0.70 / 0.96 | 0.56 / 0.74 / 0.71 / 0.97 | 13.7 / 9.2 |

Extrapolation:

| alpha | concat fall-free | affine fall-free | concat speed / action delta | affine speed / action delta |
|---:|---:|---:|---|---|
| -0.5 | 0.683 | 0.650 | 0.60 / 1.83 | 0.61 / 1.98 |
| 1.25 | 0.883 | 0.917 | 0.73 / 1.67 | 0.68 / 1.46 |
| 1.5 | 0.767 | 0.800 | 0.98 / 2.55 | 0.78 / 2.13 |
| 2.0 | 0.050 | 0.300 | 0.80 / 3.86 | 0.86 / 3.89 |

Per-pair monotonicity across the five held alphas (60 pairs each): speed
0.74 vs 0.75, stride 0.94 (n=44) vs 0.96 (n=48), arm swing 0.84 vs 0.80;
fall-free at every alpha 55/60 vs 56/60. Handover fall-free by kind at ramp
0 (30 handovers per kind): identical on both arms except walk->turn 29/30 vs
30/30; stand->wave 21/30 on both, walk->stand 27/30 on both. The settling
criterion (within 15% of the source robot's post-switch speed for 25
consecutive steps) is met by about a fifth of handovers on either arm; the
median settling time among those is 26-37 steps for affine and 37-47 for
concat. Resolution: 60 pairs resolve about 15% relative on a rate; the alpha
2.0 extrapolation (0.05 vs 0.30) is the only gap above it. One seed per arm,
encoder-init noise unmeasured on these metrics.

## `combo` added (2026-09-03): a second affine tracker

Jobs 5629658 / 5629659 / 5629660, all COMPLETED with no failed chunk
(30 / 54 / 24). `combo` is the affine past-5 encoder on an MLP actor with
ten-step history, weight decay and critic decay, 10B. Same 60 pairs.

Handover, pooled over switch steps:

| arm (phi) | ramp | n | fall-free | settled | settling median | peak action step | action delta post | gait dist |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lstm (concat) | 0 | 180 | 0.928 | 0.217 | 41 | 5.26 | 1.43 | 0.064 |
| lstm_affine (affine) | 0 | 180 | 0.933 | 0.206 | 26 | 4.54 | 1.35 | 0.067 |
| combo (affine, MLP+hist) | 0 | 180 | 0.939 | 0.250 | 27 | 9.61 | 4.32 | 0.067 |
| lstm | 10 | 180 | 0.928 | 0.222 | 37.5 | 4.77 | 1.35 | 0.059 |
| lstm_affine | 10 | 180 | 0.933 | 0.228 | 28 | 4.34 | 1.29 | 0.061 |
| combo | 10 | 180 | 0.950 | 0.233 | 32 | 7.89 | 3.97 | 0.061 |
| lstm | 50 | 180 | 0.922 | 0.200 | 47 | 3.72 | 1.22 | 0.050 |
| lstm_affine | 50 | 180 | 0.939 | 0.183 | 37 | 3.52 | 1.22 | 0.050 |
| combo | 50 | 180 | 0.950 | 0.228 | 55 | 6.24 | 3.62 | 0.050 |

Fall-free rate, held mix:

| alpha | lstm | lstm_affine | combo |
|---:|---:|---:|---:|
| 0 | 0.950 | 0.950 | 0.950 |
| 0.25 | 0.933 | 0.950 | 0.950 |
| 0.5 | 0.950 | 0.950 | 0.950 |
| 0.75 | 0.933 | 0.933 | 0.933 |
| 1 | 0.933 | 0.933 | 0.933 |

Fall-free rate, extrapolation:

| alpha | lstm | lstm_affine | combo |
|---:|---:|---:|---:|
| -0.5 | 0.683 | 0.650 | 0.700 |
| 1.25 | 0.883 | 0.917 | 0.933 |
| 1.5 | 0.767 | 0.800 | 0.750 |
| 2.0 | 0.050 | 0.300 | 0.300 |

Monotonicity across the five held alphas (60 pairs each): speed 0.74 / 0.75 /
0.73, stride 0.94 / 0.96 / 0.95, arm swing 0.84 / 0.80 / 0.82 for lstm /
lstm_affine / combo; fall-free at every alpha 55 / 56 / 56 of 60. Code
distance between the two skills is 11.2-13.7 under concat and 8.5-9.2 under
both affine arms at every alpha. `combo`'s action-space numbers are on a
different scale (its actor reads a ten-step history), so its action delta and
peak step are comparable only to itself.
