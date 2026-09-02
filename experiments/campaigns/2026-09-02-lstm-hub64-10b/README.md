# 2026-09-02 -- LSTM actor on the new default hub

Three arms, W&B project `g1-bs-pareto`, group `lstm-hub64-10b`, seed 0.
The recipe is the user's 2026-09-02 base (see the `campaign.yaml` header):
64-D merged hub with the past-5 concat phi (`p5_concat`, the default since
2026-09-02), recurrent actor (`agent.ppo.rnn_hidden_size=256`), weight decay
1e-2 plus linear critic decay to 1e-5, 100% uniform resets
(`random_trajectory_sampling_ratio=1.0`), `action_rate_l2` -0.03, 16,384
environments, sin/cos phase pair kept.

| arm | change | budget |
|---|---|---|
| `lstm` | the base | 10B, two chained segments |
| `lstm_affine` | `p5_affine` encoder (phi affine in z) | 10B, two chained segments |
| `lstm_nophase` | `command_phase_mode=none`, command 64 | 500M, one segment |

Nothing here is a one-variable comparison against `latent64-probe-10b`:
five fields move at once (encoder, actor, optimizer extras, resets, env
count). Within the campaign, `lstm_affine` vs `lstm` moves the phi
parameterization plus encoder-init noise, and `lstm_nophase` vs `lstm`
moves the phase pair and the budget.

Scoring: `bones_testbed4096_v1` clean, the star-v2 curves evaluator
arguments, with `agent.ppo.rnn_hidden_size=256` repeated at eval. The
recurrent-state carry in `evaluate_checkpoint` was flagged as qualification
debt in `2026-08-28-smooth-ablation-5b`; qualify it before scoring these.

Submit:

```bash
for arm in lstm lstm_affine lstm_nophase; do
  pixi run python -m imitation_experiments.pipeline.cluster plan \
    --campaign experiments/campaigns/2026-09-02-lstm-hub64-10b/campaign.yaml \
    --arm $arm --seed 0
  # then the printed submit line
done
```
