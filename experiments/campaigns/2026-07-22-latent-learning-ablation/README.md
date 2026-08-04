# 2026-07-22 LAFAN1 latent-learning ablation

Status recorded 2026-07-22: all twelve local 10M-frame qualification arms
passed. No H200 jobs from this study had been submitted. The approved
production geometry is one H200 with 16,384 environments x 12 rollout steps
and minibatches of 24,576.

This is the primary current experiment campaign. It compares reconstruction
families and DiffSR bottlenecks under the same corrected-LAFAN1, held-h10
controller protocol. It is additive to the frozen two-row paper comparison;
it does not change the paper grid.

Authoritative design and current scientific caveats:
[`wiki/latent-learning-ablation-plan.md`](../../../wiki/latent-learning-ablation-plan.md).

Canonical implementation:
[`experiments/latent_ablation/`](latent_ablation/).

## Entry point

Print the local qualification plan:

```bash
experiments/campaigns/2026-07-22-latent-learning-ablation/run.sh local
```

Run the local gate into a fresh output root:

```bash
MODE=run \
OUTPUT_ROOT=/absolute/path/to/local_10m_gate \
experiments/campaigns/2026-07-22-latent-learning-ablation/run.sh local
```

Validate all qualification records and print the exact H200 commands without
submitting:

```bash
MODE=validate \
LOCAL_QUALIFICATION_ROOT=/absolute/path/to/local_10m_gate \
TRAINING_PROFILE=experiments/campaigns/2026-07-22-latent-learning-ablation/latent_ablation/training_profile.h200.approved.env \
experiments/campaigns/2026-07-22-latent-learning-ablation/run.sh h200
```

Real submission requires `MODE=submit`, the same complete local qualification
root, and the approved profile. Re-read
[`wiki/current-status.md`](../../../wiki/current-status.md) and inspect the
rendered commands before changing `MODE`.

