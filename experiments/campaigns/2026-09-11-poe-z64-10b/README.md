# 2026-09-11 -- product-of-experts score `<z, E(s, s')>`, merged 64-D, 10B

One arm, `z64_poe`, one seed. The star-v2 hub encoder recipe with the DiffSR
score reparameterized so the latent enters linearly and the expert fields are
a joint network on the transition pair, then the `z64_merged` 10B tracker
stage of `2026-09-01-latent64-probe-10b` verbatim.

## The parameterization

The DiffSR denoiser predicts noise on the target as `eps = phi(s, z)^T mu`.
This arm sets

| flag | value | effect |
|---|---|---|
| `--diffsr_phi_parameterization identity` | `phi(s, z) = z` | no `g`, no `F(s)`, no bias |
| `--diffsr_mu_conditioning pair` | `mu(s, s', t)` | one network on the transition pair, `feature_dim x d_target` |
| `--diffsr_feature_dim 64` | `= z_dim` | forced by identity |

so `eps(s', t | s, z) = sum_k z_k E_k(s, s', t)` with `E = mu`. Each
coordinate of `z` is the log-weight of one expert field over the pair, the
score is exactly linear in `z`, and `z_mix = a z_1 + b z_2` samples the
tempered product `p_1^a p_2^b / Z` for any `a, b`, not only `a + b = 1`.

Against the existing heads: the affine head (`p5_affine`, the combo-50b
encoder) is the factorized special case `E_k = sum_e A_ek F_e(s) mu(s')`
plus a `z`-independent bias; this arm drops both the factorization and the
bias. Score-parameterized: the expert fields are free networks, not the
gradient of a scalar energy, so the product-of-experts reading holds for
sampling and composition but the energy itself is not an evaluable object.
An energy-parameterized variant (scalar `E_k`, `eps = -sigma_t grad_{s'}
<z, E>` by autograd) is a follow-up, not this arm.

Implementation: `RLOpt/rlopt/agent/ipmd/module.py` (`identity` phi,
`mu_conditioning`), `RLOpt/rlopt/agent/hl_skill_diffsr.py`
(`diffsr_mu_conditioning` config field, threaded through `_build_diffsr` so
the endpoint head and the `diff_*` NTP heads both get it),
`scripts/rlopt/train_hl_skill_diffsr.py` (CLI). Six tests in
`RLOpt/tests/test_ipmd_components.py`: identity returns `z`, dimension
check, pair mu sees `s`, eps linear in `z` for weights summing to 1.75,
loss and sampling run, bad conditioning rejected.

## What is held fixed

Everything else is the hub pretrain (jepa_ntp + sigreg_ebm, `diff_chunk`,
`boundary_next`, endpoint coefficient 0, deterministic 64-D, LayerNorm,
intermediate window, horizon 10, 50,000 updates at batch 8,192, SIGReg 1.0,
latent L2 1e-3) and the probe's tracker (20,480 x 24, `random80_adaptive20`,
5M-30M curriculum, ee + wide rewards, `action_rate_l2` -0.03, full-batch
entry point, 10B, checkpoint every 500M). The pair against `z64_merged`
therefore moves the score parameterization alone, plus encoder-init noise
(0.0064 SR / 0.46 mm / 10.8 mm, `2026-08-30-encoder-interface-500m`).

Control row, `z64_merged` at 9.5B: 0.9292 / 23.25 / 93.09 on
`bones_testbed4096_v1` clean, one seed. The sibling `z64_merged_noreg`
(`2026-09-11-z64-merged-noreg-10b`) removes the regularizers on the concat
head; read all three together.

## Local qualification

```bash
./experiments/campaigns/2026-09-11-poe-z64-10b/smoke_local.sh
# PRETRAIN_UPDATES=500 PRETRAIN_BATCH=4096 ./smoke_local.sh   # longer look
```

Short pretrain on the workstation, a check that the checkpoint config carries
`identity` + `pair` + `feature_dim 64`, then one tracker iteration at 32
environments binding that encoder. Logs under `logs/poe_z64_smoke/`.

## Submit

```bash
./experiments/campaigns/2026-09-11-poe-z64-10b/submit.sh z64_poe 0
pixi run python -m imitation_experiments.pipeline.cluster submit --plan <dir> --confirm <PLAN_SHA>
```

Three chained stages (pretrain, lowlevel1 afterok, lowlevel2 afterany).
W&B project `g1-bs-pareto`, group `latent64-probe-10b`, run id
`l64p-z64poe-s0`. Output on ice-shared,
`/storage/ice-shared/vip-vwt/scratch-fwu91/latent64_probe_10b/z64_poe_seed0/`.
The plan resolved and passed the eight ICE preflight checks on 2026-09-11;
the resolved pretrain script carries `--diffsr_phi_parameterization identity
--diffsr_mu_conditioning pair --diffsr_feature_dim 64`. NOT submitted from
this machine: `submit` packs the working tree, and the RLOpt change above is
uncommitted in the submodule. Commit RLOpt (and bump the pointer) or submit
from a tree that carries the change.

## Score

Same as the control: `eval_checkpoint_tree.py --final_only` with the probe's
evaluator arguments and this arm's encoder file. Then the linear-closure
probes of `2026-08-30-linear-closure-affine` on the encoder: interpolation
gap (expected exactly zero on the score, as for affine) and, new here,
tempering with `a + b != 1`, which the affine head cannot do.

## Status

- 2026-09-11: implemented, 11 targeted RLOpt tests pass, plan resolved with
  preflight OK, local smoke launched. Not submitted.
