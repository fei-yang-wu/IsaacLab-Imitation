# BONES-SEED 129k v2 scale run

This campaign trains the current `Isaac-Imitation-G1-v2` deterministic DiffSR
encoder on all 129,785 SONIC-filtered BONES-SEED motions, then trains the tuned
IPMD low-level tracker locally on the 96 GB RTX PRO 6000. The macro-state is
the `root+qpos` interface: 29 joint positions, root position, and 6D root
orientation (38 values per frame; 380 values for the h10 window). Joint
velocity is deliberately absent.

The stages stay separate so the low-level run cannot begin before the encoder's
held-out curve is inspected. The default encoder budget is the collaborator's
50,000 updates at batch 8,192. That is 409.6 million sampled windows, or about
9.6 draws per training transition after the deterministic 90/10 trajectory
split.

The controller geometry is 24,576 environments by 6 rollout steps. The v2
screen measured about 63.2 GiB at this geometry; 24,576 by 24 already OOMed.
This is therefore a wall-clock convergence probe, not the long-rollout v2
production geometry. The default controller cap is 1 billion frames.

From the repository root, print and validate the complete plan without
starting training:

```bash
STAGE=plan \
  experiments/campaigns/2026-08-04-bones129k-v2-scale/run.sh
```

After confirming the W&B group, start only pretraining:

```bash
STAGE=pretrain \
CONFIRM_RUN=bones129k-v2-scale \
WANDB_GROUP=bones129k-v2-scale \
  experiments/campaigns/2026-08-04-bones129k-v2-scale/run.sh
```

Metrics are written to:

```text
/mnt/storage/fwu91/bones_seed_full/runs/bones129k_root_qpos_v2_e24576_r6_1b_seed0/encoder/metrics.jsonl
```

Monitor `train/loss_real_z_eval` against `train/loss_zero_z_eval` and
`train/loss_shuffled_z_eval`, plus the latent diversity diagnostics. Accept the
50k checkpoint only after the held-out real-z loss has flattened while staying
clearly below both controls and the latent has not collapsed. If the last
several thousand updates still improve materially, keep the low-level stage
blocked and continue pretraining from a copied, explicitly versioned output.

Once accepted, start the low-level stage:

```bash
STAGE=lowlevel \
CONFIRM_RUN=bones129k-v2-scale \
WANDB_GROUP=bones129k-v2-scale \
  experiments/campaigns/2026-08-04-bones129k-v2-scale/run.sh
```

The launcher pins the manifest SHA-256, persisted-buffer content identity,
motion count, and exact replay-buffer keys. It opens the 95 GiB CPU memmap and
materializes a compact 6.7 GiB root+qpos source cache in VRAM once per process;
encoder batches therefore do not scatter-read the entire replay schema.
