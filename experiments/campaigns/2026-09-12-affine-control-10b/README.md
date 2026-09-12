# 2026-09-12 -- the affine control, matched to the PoE family's tracker

One arm, `p5_affine_ctrl`, one seed, 10B. Jobs 5768198 (lowlevel1), 5768199
(lowlevel2, afterany). No pretrain stage: it binds the existing `p5_affine`
encoder.

## Why the affine head is THE control

Write the affine score out:

```
eps = (A z + b)^T M(s, s', t),    M := F(s) mu(s', t) / sqrt(embed)
    = <z, A^T M>  +  <b, M>
```

That is exactly the `linear_bias` PoE form, a projection of `z` plus a
command-independent base, with one restriction: `E = A^T M` and the base
`b^T M` are both built from the separable product `F(s) mu(s')`, so every
expert row lives in the same rank-`embed_dim` subspace and nothing models a
joint `(s, s')` interaction outside that product. The PoE arms'
pair-conditioned `mu` lifts exactly that restriction. Affine is the NESTED
special case, not a rival family.

## What it fixes

Every PoE arm takes `--source_history_steps 5`; `z64_merged` takes the
current state alone. So every PoE-against-`z64_merged` comparison reported
before 2026-09-12 confounds the source history with the score form. Two
controls share the PoE source:

| control | encoder | phi | source | tracker |
|---|---|---|---|---|
| `enc_hist` | `p5_concat` | concat | past 5 | matched, trained to 9.5B |
| `p5_affine_ctrl` (this) | `p5_affine` | affine | past 5 | matched, this arm |

Both encoders were pretrained with the regularizers ON, so they pair with
the `*_reg` PoE arms of `2026-09-12-poe-reg-ablation`, not with the
regularizer-off ones.

## Held fixed

The `z64_merged` tracker stage verbatim, verified in the resolved script:
20,480 environments x 24, `random80_adaptive20` with the 5M-30M curriculum,
ee + wide rewards, `action_rate_l2` -0.03, full-batch entry point, 10B cap
(20,346 iterations), a checkpoint every 500M, command 66.

Binding the existing encoder also removes encoder-initialization noise
against `combo`, which shares the file.

## Do not read against its own campaign

`p5_affine`'s tracker in `2026-08-30-past-chunk-affine-64d` ran
`selection=sonic` with the 0.8 -> 0.2 ramp at a 5B cap. That is a different
reset regime and budget from this arm.

## Status

- 2026-09-12: submitted, nine ICE preflight checks. `enc_hist`'s own curve
  (18 unscored checkpoints) submitted the same hour as job 5768192. Both are
  wired into `2026-09-12-latent64-probe-live-eval`.
