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

Mechanics: `scripts/rlopt/composition_probe.py` ->
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
