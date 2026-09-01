# Past-chunk conditioning at 64-D: concat vs affine phi (2026-08-30)

Two arms, 5B environment frames each, seed 0. Both give `phi` a five-frame
past chunk; they differ only in how `phi` depends on `z`.

| arm | `phi` conditions on | `phi` parameterization | `phi` source width |
|---|---|---|---:|
| `p5_concat` | `s[t-5..t]` | concat (production) | 228 |
| `p5_affine` | `s[t-5..t]` | affine, `F(s)^T (A z + b)` | 228 |

## Why a past chunk

The macro state is `root_qpos` and carries no velocities, so `s_t` alone
cannot express which way the motion is already going, while the encoder sees a
whole future window. That forces `z` to carry both the momentum a dynamics
model should infer from the past and whatever genuinely distinguishes this
future. Give `phi` the past and the first part moves out of `z`.

## Why five frames and not ten

The Tier B linear probe over 2,998 held-out windows predicts `s[t+10]` from the
past at R2 0.751 with two frames and 0.730 with eleven, so the extrapolation
information saturates almost immediately. Five frames sit above the two-frame
reading and cost 228 values of `phi` source instead of 418 at past-10, against
38 for `s_t` alone.

The past-10 form was already measured at 500M in
[`../2026-08-30-encoder-interface-500m/`](../2026-08-30-encoder-interface-500m/README.md):
against its matched `h10` control it was a null on tracking at two
checkpoints, with MPJPE-G 42 to 50 mm worse on the robust row. This campaign
carries the idea to the production 5B budget at a cheaper width, and adds the
affine axis.

## Why the merged head

At 64-D and hold 1 the two-head `diffntp_chunk` form has never trained.
`leader64_h1_nophase` was cancelled at 0.84B with episode length plateaued at
50-62 and MPJPE-L flat near 51 mm, while its 256-D control was at 166 / 42.6 mm
by 0.17B, and its encoder pretrain was provably healthy, so the stall was in
the tracker. Both arms therefore take the confirmed star-v2 hub objective:
`--jepa_ntp_head diff_chunk --jepa_ntp_chunk_span boundary_next
--jepa_endpoint_coeff 0`.

This means the affine axis here is NOT directly comparable with the 256-D
`linear-closure-affine` pair: head and width both move.

## The two axes compose without a code change

`_build_diffsr` widens `phi`'s `obs_dim` to
`state_dim * (source_history_steps + 1)` and passes `phi_parameterization`
separately, so the past chunk enters `F(s)` while the affine constraint binds
only the `z` path. Validation requires `transition_objective=jepa_ntp` and
`jepa_context_chunks=0`; the hub satisfies both. The anchor stays on `s_t`, so
the encoder's own input and the deployed 66-wide command are untouched and
either encoder is drop-in.

## What this pair can and cannot attribute

Between the two arms exactly one flag moves, so concat versus affine at 64-D
with a past chunk is a clean single-variable comparison.

There is **no matched no-past control at 64-D** in this campaign, by user
decision on 2026-08-30. The nearest reference is `merged64_pen_ramp_5b` in
`../2026-08-22-pareto-stack/` (0.9543 SR / 23.67 mm MPJPE-L / 89.33 mm MPJPE-G
clean at 5B). It differs from these arms by the past chunk **and** by encoder
pretrain provenance, a 16,384-environment rollout, and the frozen
`rlopt_ipmd_tuned` optimizer geometry. It is a reference row, not a control, so
a past-chunk effect cannot be attributed against it.

## Optimizer geometry

Both arms pass `--agent rlopt_ipmd_tuned_fullbatch_cfg_entry_point`, the
geometry promoted on 2026-08-30: full batch, 3 epochs, 3 optimizer steps per
iteration, +18.9% frames per unit wall-clock. That class sets
`loss.mini_batch_size` to a sentinel that `scripts/rlopt/train_impl.py` clamps
to the whole rollout, so this campaign must never pass
`agent.loss.mini_batch_size`; a literal would silently become a partial batch.
The promotion itself rests on one 3.47B window of a 5B budget with no matched
control, and the same 3 epochs at half the minibatch collapsed, so read the
class docstring before citing the recipe.

## No interpolation probe stage

`imitation_experiments.capacity.probe_latent_interpolation` loads
`diffsr_state_dict`, which is the endpoint head. `--jepa_endpoint_coeff 0`
leaves that head untrained, and the merged objective's head is
`jepa["ntp_diffsr"]`, which the probe cannot load yet. Add a chunk-head option
to the probe before running it on these encoders; a number from the endpoint
head here would be measured on untrained weights.

## Protocol

Everything else is the star-v2 64-D hub with the `merged64_pen_ramp_5b`
tracker regime:

- 20,480 environments x 24 rollout steps = 491,520 frames per batch, 5B budget.
- Rewards `motion_ee_pos` 1.0, `motion_global_anchor_pos_wide` 1.0,
  `action_rate_l2` -0.03; `feet_acc` -2.5e-6 from the in-tree default.
- Reset selection `sonic` with the failure share ramping 0.8 to 0.2 over the
  first 1B. Segment 1 carries the ramp; segment 2 pins the landed 0.2, because
  the ramp keys off `common_step_counter`, which restarts per segment.
- Termination curriculum 5M to 30M in segment 1, off in segment 2.
- Frozen encoder during RL, hold 1, `sin_cos` phase, `robot_heading` anchor,
  380-value `root_qpos` macro state, stride 1.
- Checkpoints every 250M cumulative frames.

## Submission

Submitted 2026-08-31 on ICE, H200, seed 0, W&B group `past-chunk-affine-64d`.
Each arm is a three-job chain: pretrain, then two 5B tracker segments.

| arm | pretrain | lowlevel1 | lowlevel2 | W&B run id |
|---|---:|---:|---:|---|
| `p5_concat` | 5599861 | 5599862 | 5599863 | `pca64-p5concat-s0` |
| `p5_affine` | 5599864 | 5599865 | 5599866 | `pca64-p5affine-s0` |

`lowlevel1` depends `afterok` on its pretrain; `lowlevel2` depends `afterany`
on `lowlevel1`, so a walltime kill still continues the chain from the resume
checkpoint. Both submissions recorded `drift=true`: the working tree was dirty
at submission, so neither run is reproducible from a git SHA.

## Running

```bash
python -m imitation_experiments.pipeline.cluster plan \
    --campaign experiments/campaigns/2026-08-30-past-chunk-affine-64d/campaign.yaml \
    --arm p5_concat --seed 0
python -m imitation_experiments.pipeline.cluster submit --plan-sha <PLAN_SHA>
```

Then mirror and score:

```bash
./experiments/campaigns/2026-08-30-past-chunk-affine-64d/mirror.sh
./experiments/campaigns/2026-08-30-past-chunk-affine-64d/eval.sh
```

Per AGENTS.md every result table also carries the `sonic_v1_1` row for the same
board.
