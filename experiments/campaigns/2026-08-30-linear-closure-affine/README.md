# Linear closure of the skill latent: the affine spectral arm (2026-08-30)

## The question

The skill latent `z` is the action space of the high-level MDP: a planner emits
`z`, the frozen tracker executes it. We want that space **closed under linear
combination** — for two valid skills `z1`, `z2`, the point `a*z1 + (1-a)*z2`
should be both executable and have a predictable meaning. Nothing in the
current recipe supplies that, because the path from `z` to the grounding score
runs through an MLP: `phi(s, a*z1 + (1-a)*z2)` has no relation to
`a*phi(s, z1) + (1-a)*phi(s, z2)`.

## The change

One flag on encoder pretraining:

```
--diffsr_phi_parameterization affine
```

`phi(s, z) = F(s)^T (A z + b) / sqrt(E)`. `F(s)`, `mu`, and the encoder `E`
stay as nonlinear as they are today; only the path from `z` to the score is
constrained to be affine. `A` and `b` are learned like every other weight — the
constraint is on the functional form, not the values.

Because `eps_pred = phi(s,z)^T mu(y_t, t)` and `mu` never sees `z`, the learned
denoising score field is affine in `z` at every diffusion time. So for weights
summing to one, the score of the mixed latent is the mixture of the scores,
which is the score of the **geometric mixture** of the endpoint conditionals:

```
p(y | s, z_a)  proportional to  p(y | s, z1)^a * p(y | s, z2)^(1-a)
```

An interpolated skill therefore means "futures compatible with both endpoints",
a product/intersection, not a coin flip and not an average. Both DiffSR heads
are affine — the endpoint head for `p(s[t+10] | s_t, z)` and the `diff_chunk`
head for `p(s[t+11..t+20] | s_t, z)` — because a single config field builds
both.

Implementation: `BilinearSR` in `RLOpt/rlopt/agent/ipmd/module.py` gained
`phi_parameterization="affine"`, which takes the matrix-valued `F(s)` of the
legacy `bilinear` branch and replaces the Mish `ResidualMLP` action net with a
single `nn.Linear(z_dim, embed_dim)`. Exactness is asserted in fp64 by
`RLOpt/tests/test_ipmd_components.py::test_affine_phi_is_affine_in_action` and
`::test_affine_eps_prediction_is_affine_in_action`, with the nonlinear
`bilinear` branch as a negative control.

## What the tracker consumes — unchanged

The sampler publishes `[z (256); sin_cos phase (2)]` = 258-D as the
`latent_command` observation term, exactly as every other latent arm. `phi`,
`F(s)`, `A`, and the diffusion heads are pretraining-only; the policy sees
`pi(a | s_robot, z)` and never a score. The arm changes **where a motion lands
in R^256**, not the shape of the interface.

Note for later: `z` is not on a sphere. The encoder's output layer is a bare
`nn.Linear` (`RLOpt/rlopt/agent/hl_skill_encoder.py`); `--encoder_layer_norm`
gates hidden layers only. So straight-line interpolation is already in-domain
and no convexity fix is required.

## Arms

Both arms pretrain their own encoder in this campaign, so the comparison holds
pretrain provenance fixed and moves exactly one variable.

| arm | pretrain | tracker |
|---|---|---|
| `affine` | `--diffsr_phi_parameterization affine` | 5B, EMA action filter |
| `concat` | `--diffsr_phi_parameterization concat` (production) | 5B, EMA action filter |

Everything else is the round-4 `diffntp_chunk` recipe (`--transition_objective
jepa_ntp --jepa_loss sigreg_ebm --jepa_ntp_head diff_chunk`, intermediate
window, 380-D `root_qpos` macro state, `robot_heading` anchor, 50k updates at
batch 8,192) and the smooth-ablation-5b tracker regime: 20,480 envs, 5B frames,
`action_rate_l2` -0.03, failure-share ramp 0.8 -> 0.2 over the first 1B, and
the trained-in EMA action filter `env.actions.joint_pos.ema_alpha=0.65` — the
current smoothness lead. The filter is the shared regime, not the variable, and
it lives in the env action term, so eval repeats it.

The smooth-ablation-5b `ema` row is a **third** comparison point, not a matched
one: it reuses the pareto-stack encoder, so it differs from `concat` here by
pretrain provenance as well as by nothing else. Cite it as context, not as the
control.

## Sequence

1. **Local smoke.** `NUM_UPDATES=500 ./pretrain_affine_arm.sh affine` — the
   matrix-valued `F(s)` branch had never been trained before this campaign, so
   this run exists to prove it trains and fits in memory, not to produce a
   number.
2. **Local capacity qualification** (optional, before cluster spend).
   `./pretrain_affine_arm.sh affine` then `./pretrain_affine_arm.sh concat`,
   50k updates each, SEQUENTIALLY — two concurrent pretrain arms OOM this
   workstation (each holds ~47 GiB for the macro cache). Read
   `train/jepa_endpoint_loss_eval` and `train/jepa_ntp_loss_eval` out of each
   `metrics.jsonl`. The gap is the capacity price of forcing `g` to be affine,
   measured before any cluster time is spent.
3. **Cluster.** Commit first so the submission records `drift=false`.
   ```
   python -m imitation_experiments.pipeline.cluster plan \
       --campaign experiments/campaigns/2026-08-30-linear-closure-affine/campaign.yaml \
       --arm affine --seed 0
   python -m imitation_experiments.pipeline.cluster submit --plan <plan dir> --confirm <PLAN_SHA>
   ```
   Then the same for `--arm concat`. Three chained stages per arm:
   `pretrain -> lowlevel1 -> lowlevel2`.
4. **Mirror and score.** `./mirror.sh` pulls each arm's encoder AND tracker
   checkpoints (the encoders differ, so a crossed pair would measure a
   mismatch). Run
   `imitation_experiments.audit.validate_latent_skill_checkpoint_binding`
   before citing rows, then `./eval.sh` for the clean and robust rows on
   `bones_testbed4096_v1`.

## Decision rules

- **Capacity (Q3).** If the affine encoder's eval losses are far above the
  concat control's, the affine constraint costs prediction quality and the
  scoreboard rows say where that lands. Report both numbers in the same
  sentence as any tracking claim.
- **Tracking.** Score `affine` against `concat` on the 4096 board, clean and
  robust, alongside the standing `sonic_v1_1` row for the same board. Isaac
  evaluation is not deterministic; a relative difference below about 15% in the
  high-error regime is unresolved, and one seed settles nothing.
- **Closure itself is NOT settled by these rows.** The scoreboard measures
  tracking on real motions, where no interpolation happens. The guarantee is a
  property of the grounding; whether the tracker executes chords of `z`-space
  is a separate question that needs the alpha-sweep probe on the resulting
  (encoder, tracker) pair. Do not report a scoreboard win as evidence of
  linear closure.

## Results

### Pretraining capacity (both arms complete, 50k updates, one seed)

No measurable capacity penalty. Averaged over the 26 milestones from update
25,000 to 50,000, affine is at 0.936x concat's endpoint eval loss and 1.001x
its next-chunk eval loss. The next-chunk term is the clean read at 1.001; the
endpoint term nominally favours affine but the milestone-to-milestone standard
deviation over that window is 0.028 (affine) and 0.023 (concat) against a mean
separation of 0.015, and the final milestone inverts. Read it as a null.

The cost is wall-clock: 1:37:08 against 0:49:38, a 1.96x slowdown, and 4.13 GB
checkpoints against 1.35 GB. Both follow from `F(s)`'s 1024x256 output layer.

### Interpolation probe (both encoders, 2,000 cross-motion pairs)

The guarantee holds after full training, and the matched control quantifies
what the production recipe loses.

| alpha | affine score gap | concat score gap |
|---|---|---|
| 0.25 | 1.81e-07 | 7.69e-02 |
| 0.50 | 1.72e-07 | 1.09e-01 |
| 0.75 | 1.81e-07 | 7.43e-02 |

Affine is at float32 roundoff; concat is 10.9% mean and 52.4% max at the
midpoint. Interpolant geometry is near-identical between arms (midpoint norm
ratio 0.724 vs 0.718, nearest-real-neighbour ratio 0.972 vs 1.076), so affine
did not buy affinity by distorting the latent distribution. Denoising transfer
is smooth and monotone on both.

### Tracking at 1B, bracketed (4,096 board, clean, one seed) -- PRELIMINARY

Scored at 1.25B of a 5B budget while both arms were still running. Three
checkpoints per arm because checkpoint variance here exceeds evaluation noise.

| arm | frames | SR | MPJPE-L | MPJPE-G | jerk | adelta |
|---|---|---|---|---|---|---|
| affine | 0.75B | 0.8682 | 28.80 | 145.31 | 155.5 | 0.802 |
| affine | 1.00B | 0.8823 | 28.12 | 130.57 | 152.1 | 0.780 |
| affine | 1.25B | 0.8862 | 27.56 | 118.81 | 154.3 | 0.787 |
| concat | 0.75B | 0.8894 | 27.90 | 138.33 | 170.7 | 0.846 |
| concat | 1.00B | 0.9006 | 26.59 | 129.83 | 159.8 | 0.803 |
| concat | 1.25B | 0.9041 | 26.11 | 116.04 | 169.3 | 0.857 |
| sonic_v1_1 | released | 0.9888 | -- | 26.73 | -- | acc 3.45 |

Concat leads on success rate and local error at all three checkpoints. The
gaps (about 0.018 SR, about 1.5 mm MPJPE-L) are no larger than each arm's own
movement across the bracket, but the sign never flips across three independent
checkpoints, which the training curves could not resolve. Affine is smoother:
jerk 152-155 against 160-171, and its jerk is flat across the bracket while
concat's climbs. `ee_body_pos` dominates failures for both, as it does on this
board generally. MPJPE-G is still falling steeply for both and is not settled.

## Status

- 2026-08-30: RLOpt `affine` parameterization implemented and unit-tested.
  Both pretrain arms COMPLETE. Both interpolation probes COMPLETE. Both 5B
  tracker chains submitted and running. 1B tracking scored, preliminary.

## Job log

| date | arm | stage | job | outcome |
|---|---|---|---|---|
| 08-30 | affine | pretrain | 5598584 | COMPLETED, 1:37:08, 50k updates |
| 08-30 | concat | pretrain | 5598585 | COMPLETED, 0:49:38, 50k updates |
| 08-30 | affine | probe | 5598898 | COMPLETED, 44s (a100) |
| 08-30 | concat | probe | 5598899 | COMPLETED, 34s (a100) |
| 08-30 | affine | lowlevel1/2 | 5598889/90 | running |
| 08-30 | concat | lowlevel1/2 | 5598891/92 | running |

Two earlier probe attempts failed and are kept here because both were
plumbing, not science: 5598872/5598875 died on `No module named
'imitation_experiments'` (the container PYTHONPATH did not carry the
experiment library), and 5598895/5598896 died on `No module named 'torch'`
(a `scripts/` entrypoint must call `configure_cu130_bridge` before importing
Torch under Kit's Python).
