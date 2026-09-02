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

## `lstm_nophase` result (2026-09-02)

Job 5614457 (12,288 envs) finished its 500M budget clean. Matched frames
against `lstm` (16,384 envs), W&B/log rows:

| frames | `lstm_nophase` ep_len / r_step | `lstm` ep_len / r_step |
|---|---|---|
| 310-330M | 46-49 / 0.14 | 99-147 / 0.17 |
| 480-500M | 44-65 / 0.15 | 173-179 / 0.19 |

Same stall signature as every no-phase 64-D hold-1 arm on record, now on a
recurrent actor, the past-5 encoder, uniform resets, and the optimizer
extras. Six stalls across four recipes without the constant (0, 1) pair;
none with it. One seed; env count differs from `lstm`.

## Latent blend probe (2026-09-02, `blend_probe.sh`, v6)

Env 0 tracks the walk (rank 2389 `walk_forward_loop`, 0.84-0.92 m/s), env 1
the jog (76357 `jog_ff_loop_180`, 2.21 then 1.73 m/s; or 30608
`jog_arc_cw_loop`, 1.12 m/s steady); from `start` env 0 receives
`(1 - a) z_walk + a z_jog` with `a` ramping linearly over the ramp window,
then held. Reference-relative terminations off (they fire on any gait
change by construction, v4); uprightness is `-projected_gravity_z` per step.
Both arms at 4.5B, one seed, one run per cell, PhysX. No robot fell in any
run (upright min >= 0.97, fallen steps 0).

| run (arm, walk -> jog, ramp) | speed pre / ramp / post (m/s) | action delta pre / ramp / post, max in ramp | upright min | code distance pre / ramp / post |
|---|---|---|---|---|
| `lstm_affine`, 2389 -> 76357 fast, 150-200 | 0.68 / 1.05 / 1.07 | 0.892 / 1.332 / 1.319, 5.82 | 0.98 | 7.44 / 7.43 / 8.52 |
| `lstm`, 2389 -> 76357 fast, 150-200 | 0.72 / 0.93 / 0.89 | 0.753 / 1.801 / 1.893, 4.65 | 0.97 | 9.98 / 11.40 / 12.97 |
| `lstm_affine`, 2389 -> 30608 arc, 200-300 | 0.87 / 0.70 / 0.71 | 0.837 / 1.271 / 1.528, 3.83 | 0.98 | 9.50 / 9.94 / 6.32 |
| `lstm`, 2389 -> 30608 arc, 200-300 | 0.88 / 0.73 / 1.06 | 0.991 / 1.551 / 1.324, 5.21 | 0.97 | 11.02 / 12.17 / 14.76 |

Speed is the target robot's planar base velocity; "post" is from ramp end to
the last step (300 or 450). The jog references run at 1.7-2.2 (fast) and
1.12 (arc) m/s in the post windows. Videos:
`logs/latent64_probe_mirror/blend_v6/<arm>_r<walk>-<jog>_s<start>_r<ramp>_fall_only/video/`.
Earlier passes: v2/v3 blended into a jog clip that stops at frame 300 and a
walk clip that stands for its first 2 s (both arms held 450/450, speeds
uninformative); v4 kept the board terminations and every run ended on
`ee_body_pos` 10-20 steps after the ramp.
