# 2026-09-11 -- merged 64-D hub encoder with no latent regularizer, 10B tracker

One arm, `z64_merged_noreg`, one seed. The star-v2 hub encoder recipe
(merged `diff_chunk` head, `boundary_next` span, endpoint coefficient 0,
concat phi, deterministic 64-D latent, LayerNorm, intermediate window,
horizon 10, 50,000 updates at batch 8,192) pretrained with BOTH latent
regularizers at zero, then the `z64_merged` tracker stage of
`2026-09-01-latent64-probe-10b` verbatim for 10B frames.

## What moves

| flag | default (hub) | this arm | what it is |
|---|---:|---:|---|
| `--jepa_sigreg_coeff` | 1.0 | 0 | SIGReg: sketched isotropic-Gaussian test on the online token `z1` |
| `--reg_coeff` | 1e-3 | 0 | the deterministic latent's L2, `mean(z^2)` (`hl_skill_encoder.py`) |

With the `diff_chunk` head the pretrain objective is then a single term: the
diffusion next-chunk loss over the raw frames `s[t+H .. t+2H]` in the
executed chunk's heading frame. No self-target, no EMA token, no isotropy
pull, nothing shaping the marginal of `z` except what the conditional
diffusion model needs from it.

## Control

`z64_merged` in `2026-09-01-latent64-probe-10b`, same W&B group
(`latent64-probe-10b`, project `g1-bs-pareto`), same tracker stage: 20,480 x
24 environments, `random80_adaptive20` resets with the 5M-30M termination
curriculum, ee + wide rewards, `action_rate_l2` -0.03, full-batch entry
point, checkpoint every 500M. Its row at 9.5B (walltime end): 0.9292 /
23.25 / 93.09 on `bones_testbed4096_v1` clean, one seed.

This arm is a fresh pretrain, so the pair also carries encoder-initialization
noise: 0.0064 SR / 0.46 mm MPJPE-L / 10.8 mm MPJPE-G across three replicates
of one recipe (`2026-08-30-encoder-interface-500m`).

## Run

```bash
./experiments/campaigns/2026-09-11-z64-merged-noreg-10b/submit.sh z64_merged_noreg 0
pixi run python -m imitation_experiments.pipeline.cluster submit --plan <dir> --confirm <PLAN_SHA>
```

Three chained stages: `pretrain` (about 45 min on H200), `lowlevel1`
(afterok), `lowlevel2` (afterany, resumes the tree; the 10B cap stays).
Output on ice-shared,
`/storage/ice-shared/vip-vwt/scratch-fwu91/latent64_probe_10b/z64_merged_noreg_seed0/`,
because personal scratch stood at 269 of 300 GB on submission day.

## Score

Same board and protocol as the control: `eval_checkpoint_tree.py --final_only`
on the tracker tree with the probe's evaluator arguments and this arm's
encoder file, `bones_testbed4096_v1` clean, seed 0. Also run the interpolation
gap and midpoint-norm probes of `2026-08-30-linear-closure-affine` on the
encoder: without SIGReg the chord geometry is the open question.

## Status

- 2026-09-11: planned, eight ICE preflight checks passed, submitted:
  pretrain 5761655, lowlevel1 5761656 (afterok), lowlevel2 5761657
  (afterany). Resolved batch scripts checked: `--jepa_sigreg_coeff 0
  --reg_coeff 0`, `diff_chunk`, `boundary_next`, endpoint 0, deterministic
  64-D, 50,000 updates; tracker at 20,346 iterations binding this arm's own
  encoder file. Working tree was dirty at submit (drift recorded).
