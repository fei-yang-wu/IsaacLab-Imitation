# 2026-07-22 BONES-SEED h10 GPU/LR scale screen

This supporting campaign selected the production geometry used by the current
latent-learning ablation. It trained one shared h10 encoder, screened H100 and
H200 geometry plus actor learning rates, and selected the H200
16,384-environment x 12-step profile.

Last recorded 2026-07-22:

- shared encoder job `5526697` completed;
- the five 500M-frame screen arms were intentionally cancelled after the
  wall-clock comparison;
- production job `5526830` used the selected H200 geometry.

These are dated records, not live scheduler claims. Check
[`wiki/current-status.md`](../../../wiki/current-status.md) before acting.

The canonical launcher remains
[`experiments/campaigns/2026-07-23-bones-phase5-language-h200/submit_bones_seed_h10_gpu_lr_ablation_ice.sh`](../2026-07-23-bones-phase5-language-h200/submit_bones_seed_h10_gpu_lr_ablation_ice.sh).
This campaign wrapper preserves that one source of truth.

Print the shared-encoder submission:

```bash
DRY_RUN=1 STAGE=pretrain \
experiments/campaigns/2026-07-22-bones-h10-scale/submit.sh
```

Print only the selected H200 arm:

```bash
DRY_RUN=1 \
STAGE=train \
ARM_FILTER=h200_e16384_lr1e3 \
experiments/campaigns/2026-07-22-bones-h10-scale/submit.sh
```

Do not change `DRY_RUN` until the persistent encoder checkpoint, output names,
current allocation, and scheduler state have been rechecked.

