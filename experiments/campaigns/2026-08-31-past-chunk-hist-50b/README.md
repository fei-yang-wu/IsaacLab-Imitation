# 2026-08-31 — past-chunk phi plus proprioceptive history, 64-D, 50B

Two arms, seed 0, 50B environment frames each, on the 64-D merged hub. Both
reuse a finished encoder from
[`../2026-08-30-past-chunk-affine-64d/`](../2026-08-30-past-chunk-affine-64d/README.md),
so this campaign has no pretrain stage.

| arm | encoder (reused) | phi parameterization | jobs | W&B run id |
|---|---|---|---|---|
| `p5h10_concat` | `/data/past_chunk_affine_64d/p5_concat_seed0/encoder` | concat | 5605194-5605201 | `pch50-concat-s0` |
| `p5h10_affine` | `/data/past_chunk_affine_64d/p5_affine_seed0/encoder` | affine | 5605202-5605209 | `pch50-affine-s0` |

Status: **SUBMITTED 2026-09-01** on ICE, H200, seed 0. W&B project
`g1-bs-pareto`, group `past-chunk-hist-50b`. Both submissions recorded
`drift=false`, so each chain is reproducible from the commit it was submitted
at.

Both encoders read a five-frame past chunk in `phi`
(`--source_history_steps 5 --source_anchor current`, 228-value source) and
carry the merged hub objective (`--jepa_ntp_head diff_chunk
--jepa_ntp_chunk_span boundary_next --jepa_endpoint_coeff 0`, `--z_dim 64`).
Pretrain jobs 5599861 and 5599864 both COMPLETED. Reusing them means the two
arms differ from each other in exactly the flag their names carry, with no
encoder-initialization noise between this campaign and that one.

## Three tracker changes against `past-chunk-affine-64d`

**Ten-step proprio history** on the five policy terms `projected_gravity`,
`base_ang_vel`, `joint_pos_rel`, `joint_vel_rel`, `last_action`. The critic
stays single-frame, as in `2026-08-27-diffntp-history`.

That campaign REFUTED this knob at 256-D on the two-head form: at matched 2.0B
it cost 0.027 SR, +11% MPJPE-L, +13% MPJPE-G and +33% wrist terminations, and
its one gain (-9.6% acceleration) was gone by 4.0B and absent under `no_push`.
This is a re-test in a different cell. `phi`'s past chunk moves reference
momentum out of `z`, and the history covers the actor's own state estimate.
The actor is not velocity-blind without it — `joint_vel_rel` and `base_ang_vel`
are already policy terms at one frame — so the history buys filtering and lag
compensation, not new state.

**No phase channel**, command 66 -> 64. At `code_period=1` with
`command_phase_source='hold'` the `sin_cos` pair is constant
("Informationally dead at `code_period=1`", `RLOpt/rlopt/agent/ipmd/ipmd.py`),
so this drops two constant values. The recorded phase failures
(`g5_phase_none_h10` 0.3784, v1 `use_phase_none` 0.3679) are hold-10 rows where
the slot clock varies and do not apply at hold 1. It is still a second
difference from the `p5_*` rows: a negative result cannot be split between the
history and the width.

**Weight decay and critic decay**, adopted from
`2026-08-30-optimizer-ablation-5b` on user direction:
`agent.optim.weight_decay=1.0e-2`, `agent.ipmd.critic_lr_schedule=linear`,
`agent.ipmd.critic_lr_final=1.0e-5`. Neither was promoted.

## What the optimizer evidence does and does not say

Final 5B MPJPE-L in that campaign, seed 0, 300M window, arm-vs-arm because
`ctrl` was cancelled at 2.14B:

| arm | MPJPE-L | geometry |
|---|---:|---|
| `mb_full_e3` | 28.54 ± 0.08 | full batch, 3 epochs |
| `wd_1e2` | 30.33 | `ctrl`, 10 steps/iteration |
| `wd_1e4` | 31.55 | `ctrl` |
| `mb_half_e5` | 32.71 | half batch, 5 epochs |
| `critic_lin` | 33.66 | `ctrl` |
| `mb_full_e5` | 40.37 | full batch, 5 epochs |
| `mb_half_e3` | 44.34 | half batch, 3 epochs |

Every `wd_*` and `critic_lin` arm ran at `ctrl`'s geometry, so **weight decay
crossed with the full-batch geometry has never been run**. This stack is a new
cell, not a validated recipe.

The reason to carry `wd_1e2` at this budget is stability, not that row: only
`mb_full_e3` and `wd_1e2` never degraded, while six of eight arms turned over
after ~2.3B and two went non-finite (`wd_1e1` at 939M, `floor_late` at 1.93B).
This chain runs 20x past that turnover point. `critic_lin` led the field at
3.6B and finished fifth through six consecutive declining windows, so the
record does not show it protecting against the late instability.

**Critic decay horizon.** `_apply_critic_lr_schedule`
(`RLOpt/rlopt/agent/ipmd/ipmd.py`) computes progress as
`metadata.frames_processed / collector.total_frames`, and `total_frames` comes
from the segment's own `--max_iterations`. Every stage here passes the SAME 50B
cap, so the decay is one monotone 1e-3 -> 1e-5 ramp across the chain and a
resume continues it. A stage with a smaller cap would make the critic learning
rate jump back up at that boundary.

## Reset regime: fixed, no curriculum, no ramp

`random80_adaptive20` for the whole 50B, `enable_termination_curriculum=false`
from the first frame, no `sonic` switch and no failure-share ramp (user
direction 2026-08-31). Every stage is identical except the resume flag.

`past-chunk-affine-64d` ramped the sonic failure share 0.8 -> 0.2 over its
first 1B, so **`p5_concat`'s finished 5B row is not a matched control for these
arms at any frame count**. It is the nearest reference row and the recipe
ancestor. `p5_affine` has no tracker row at all: job 5599865 died in two
seconds on `printf: write error: Disk quota exceeded` and its `lowlevel2` then
failed for a missing checkpoint.

## Protocol

- `--agent rlopt_ipmd_tuned_fullbatch_cfg_entry_point`: full batch, 3 epochs,
  3 optimizer steps per iteration. The campaign passes no
  `agent.loss.mini_batch_size` and no `agent.loss.epochs`; the class owns both
  and a literal minibatch would silently become a partial batch.
- 20,480 environments x 24 rollout steps = 491,520 frames per batch, 50B cap in
  every segment, 101,725 iterations.
- Rewards `motion_ee_pos` 1.0, `motion_global_anchor_pos_wide` 1.0,
  `action_rate_l2` -0.03, `tracking_reward_points` 4.0; `feet_acc` -2.5e-6 from
  the in-tree default.
- Frozen encoder, hold 1, `robot_heading` anchor, 380-value `root_qpos` macro
  state, stride 1, `reference_prefetch_mode=next`.
- Checkpoints every 500M cumulative frames.
- Eight chained segments of 15:59 per arm, `afterany`, so a walltime kill
  continues from the resume checkpoint. `p5_concat` reached 5B in 8h29m at this
  environment count without the history, so 50B is roughly seven segments; a
  segment that starts after the budget is met exits immediately with
  "0 iterations remain".

## Running

```bash
python -m imitation_experiments.pipeline.cluster plan \
    --campaign experiments/campaigns/2026-08-31-past-chunk-hist-50b/campaign.yaml \
    --arm p5h10_concat --seed 0
python -m imitation_experiments.pipeline.cluster submit --plan-sha <PLAN_SHA>
```

Evaluation must repeat the five `history_length=10` overrides and
`command_phase_mode=none`, or the actor input width will not match the
checkpoint. `2026-08-27-diffntp-history/eval.sh` is the reference for reading
history overrides back into an evaluation.

Per AGENTS.md every result table carries the `sonic_v1_1` row for the same
board.
