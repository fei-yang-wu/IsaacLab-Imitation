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

## Status

- 2026-08-30: RLOpt `affine` parameterization implemented, unit-tested,
  campaign frozen and plan-validated for both arms. Nothing submitted.

## Job log

| date | arm | stage | job | outcome |
|---|---|---|---|---|
| | | | | |
