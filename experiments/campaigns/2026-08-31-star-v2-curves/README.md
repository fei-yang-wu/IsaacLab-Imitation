# 2026-08-31 — Star v2 convergence curves

Score every star-v2 arm's whole budget axis ON THE CLUSTER, where the
checkpoints already live, to produce the convergence figures.

One Slurm job per arm, not one per checkpoint: `eval_checkpoint_tree.py` keeps
a single Isaac Sim start and swaps the policy weights across the milestones.
Measured locally, that is 30.9 s per cell against 47.4 s, and on the cluster it
also collapses 25 container starts into one.

Board: **`bones_testbed4096_v1`** clean, `--randomization none` — the SAME
board the ablation tables report, by user decision 2026-08-31. Curves and
tables therefore come from one corpus and the 2B point of a curve IS the table
row, instead of two boards that disagree by more than a rounding step.

The cost is real: 4,096 clips per cell instead of 256, so a 25-point arm takes
roughly 90 minutes rather than 15. That is why the stage limit is 8 hours and
why this runs on the cluster rather than locally.

## This config is generated, not written

```bash
pixi run python -m imitation_experiments.reporting.build_curve_eval_campaign \
    --campaign experiments/campaigns/2026-08-30-latent-star-v2/campaign.yaml \
    --tree-root /storage/ice-shared/vip-vwt/scratch-fwu91/archived_data/latent_star_v2_checkpoints \
    --out experiments/campaigns/2026-08-31-star-v2-curves/campaign.yaml
```

Every per-arm interface field — code width, hold, phase, macro terms, stride,
anchor, horizon, and the posterior-versus-pretrained route — is copied from the
training campaign, so an evaluation cannot silently disagree with the run it
scores. Hand-writing these for 44 arms is exactly how that drift happens; the
runner this replaces had `horizon_steps` and the command source hardcoded.
Re-run the generator instead of editing this file.

## Where the checkpoints are

`/storage/ice-shared/vip-vwt/scratch-fwu91/archived_data/latent_star_v2_checkpoints`,
not scratch. Scratch keeps only the newest three checkpoints per arm under the
300 GB quota; the archive is a complete superset, refreshed by
`/home/hice1/fwu91/scratch/sync_archive.sh`. **Re-run that sync before
re-submitting**, or the newest milestones will be missing from the curves.

## Concurrency

Each arm gets a PRIVATE `CLUSTER_ISAAC_SIM_CACHE_DIR`. A shared cache is what
crashed the second of two concurrent evaluation jobs inside Kit startup on
2026-08-27; with one directory per arm there is nothing to contend for. That
makes these safe to run in parallel — 44 serialized jobs would be about 18
hours, against roughly one in parallel.

Ramp rather than trusting it: submit a few arms, confirm they survive Kit
startup together, then submit the rest.

`eval_checkpoint_tree.py` skips already-scored cells unless `--rescore`, so
re-submitting as arms deepen only scores the new milestones.

## Status

- 2026-08-31: campaign generated for all 44 arms. Pilot batch submitted —
  `hub` 5604546, `g2_token` 5604553, `g1_post_ae` 5604554 — to test the
  private-cache assumption before the remaining 41.
