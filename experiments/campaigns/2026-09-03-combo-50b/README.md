# 2026-09-03 -- `combo` at 50B

The stacked 64-D arm promoted from 10B to the headline frame budget, on the
user's direction. Recipe unchanged from `2026-09-01-latent64-probe-10b` arm
`combo`; only the budget and the segment count move.

| | |
|---|---|
| encoder | past-5 **affine** phi (`p5_affine`), 64-D merged hub, hold 1, sin_cos phase |
| actor | MLP, ten-step history on the five policy terms (critic single frame) |
| optimizer | full-batch / 3-epoch entry point, `weight_decay=1e-2`, linear critic decay to 1e-5 |
| resets | `sonic`, uniform share ramped 0.8 -> 0.2 (see below), 5M-30M termination curriculum |
| rewards | motion_ee_pos 1.0, motion_global_anchor_pos_wide 1.0, tracking_reward_points 4.0, action_rate_l2 -0.03 |
| scale | 16,384 environments x 24 = 393,216 frames per batch, 127,157 iterations |

What it is promoted on, one seed, `bones_testbed4096_v1`: at 10B `combo`
read 0.9214 SR / 22.64 MPJPE-L / 88.52 MPJPE-G clean and 0.9146 / 24.84 /
124.52 robust, the best tracking rows of the 10B group (control 0.9292 /
23.25 / 93.09 clean at 9.5B, `lstm` 0.9121 / 21.24 / 103.11, `lstm_affine`
0.9062 / 22.27 / 110.63). It stacks four levers at once, so no row here
attributes to one of them.

Seven chained 15:59 segments, `afterany`, each carrying the FULL 50B cap:
`_apply_critic_lr_schedule` reads `collector.total_frames`, which is the
segment's own `--max_iterations`, so a smaller cap on a later segment would
push the critic learning rate back up. At the ~150k fps this arm held on
`coe-gpu`, one segment buys about 8.6B, so six reach the cap and the seventh
is slack.

Disk: `agent.save_interval=1000000000` (1B, against the 500M of the 10B
campaigns) -- 50 checkpoints x 244 MB = 12.2 GB. The ICE home quota was at
265 GB of 300 GB at submission, so 500M would not have fit.

```bash
pixi run python -m imitation_experiments.pipeline.cluster plan \
    --campaign experiments/campaigns/2026-09-03-combo-50b/campaign.yaml --arm combo --seed 0
# then the printed submit line
```

## Reset schedule (user, 2026-09-03)

The reset mix ramps from 80% uniform / 20% failure-drawn to 20% / 80%, so
late training concentrates on the frames where the tracker terminates.

| segment | selection | uniform share | curriculum |
|---|---|---|---|
| 1 | `sonic` | 0.8 -> 0.5 over its first 4B | 5M-30M |
| 2 | `sonic` | 0.5 -> 0.2 over its first 4B | off |
| 3-7 | `sonic` | pinned 0.2 | off |

Two mechanics force this shape:

* The ramp drives `SonicAdaptiveResetSampler.uniform_sampling_rate` and is
  legal only under `selection=sonic`. `random80_adaptive20` puts the
  adaptive sampler inside a random-trajectory wrapper that takes 80% of
  resets, so a ramp there rescales the remaining 20% only; the config
  refuses the combination (the `cont_det_hold1_resetramp` near-no-op of
  2026-08-28).
* `_advance_adaptive_ratio_curriculum` reads `common_step_counter`, which
  restarts every segment, so a ramp must finish inside the segment that
  starts it. A segment buys about 8.6B at ~150k fps, hence two 4B ramp
  stages rather than one 10B ramp. The mix reaches 20/80 at roughly 13B and
  holds it for the remaining 37B.

Second change against the 10B `combo` row, worth stating with any
comparison: `sonic` also drops the random first-half start frames that
`random80_adaptive20` supplied, because SONIC picks rank and frame jointly
from its bin distribution.
