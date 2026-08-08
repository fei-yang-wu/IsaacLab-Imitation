# BONES-129k scaled SONIC-FSQ64 tracker-capacity comparison

This campaign pretrains one shared root-qpos skill encoder on all 129,785
BONES-SEED motions, then trains two 5B-frame low-level controllers against the
same frozen checkpoint. The controllers differ only in actor/critic capacity.

Shared pretrain contract:

- 380-value root-qpos input: ten frames x 38 values.
- Direct 64-value FSQ command, 32 levels per coordinate.
- Encoder MLP `[2048, 1024, 512, 512]`, SiLU, no hidden LayerNorm.
- DiffSR `feature_dim=256`, `embed_dim=1024`.
- DiffSR state/skill/phi and next-state hidden widths `[1024, 1024, 512]`.
- Endpoint transition objective, 50,000 updates, batch size 8,192.

Controller arms:

| arm | actor and critic hidden widths | purpose |
|---|---|---|
| `tuned` | `[1024, 1024, 512]` | tracker-capacity control |
| `sonic` | `[2048, 2048, 1024, 1024, 512, 512]` | SONIC-sized tracker |

Both controllers hold one FSQ64 code for ten control steps and append sin/cos
phase, yielding a 66-value command. They use 16,384 environments, rollout 24,
gamma 0.97, the tuned G1-v2 rewards, and `random80_adaptive20` resets.

W&B project/group: `g1-bones-seed/skill-encoding-ablation`.

## Submitted jobs

Submitted to ICE on 2026-08-06 after one-update pretrain and one-iteration
controller smokes passed. Pretraining completed on H200. Both pending controller
jobs were retargeted in place to H100 after H200 saturation, preserving job IDs
and queue age; both started together at 18:18 EDT.

| stage | job ID | GPU | node |
|---|---:|---|---|
| shared scaled FSQ64 pretrain | `5570673` | H200 | completed |
| tuned tracker, 5B frames | `5570680` | H100 | `atl1-1-03-012-28-0` |
| SONIC tracker, failed before iteration 1 | `5570681` | H100 | `atl1-1-03-013-8-0` |
| SONIC tracker, H200 retry 1 | `5570936` | H200 | `atl1-1-03-013-26-0` |

The H100 SONIC arm failed with a Warp CUDA out-of-memory error. Retry 1 keeps
the exact controller and encoder contract, changes only the allocation to an
H200 on `coe-gpu`, and writes to `sonic_tracker_h200_retry1/rlopt_train`.

Both controller jobs become eligible together after the shared pretrain exits
successfully. Exact source contract and scheduler metadata are recorded in
`cluster_submission.json`.

Run local pretrain and controller smokes:

```bash
MODE=smoke experiments/campaigns/2026-08-06-bones129k-sonic-fsq-scale/run.sh
```

Validate ICE inputs and fresh output paths using the printed smoke root:

```bash
MODE=validate LOCAL_SMOKE_ROOT=/absolute/smoke/root \
  experiments/campaigns/2026-08-06-bones129k-sonic-fsq-scale/run.sh
```

Submit one shared pretrain plus two dependent controllers (H200 by default):

```bash
MODE=submit CONFIRM_SUBMIT=skill-encoding-ablation \
  LOCAL_SMOKE_ROOT=/absolute/smoke/root \
  experiments/campaigns/2026-08-06-bones129k-sonic-fsq-scale/run.sh
```

Both controller jobs use `afterok:<shared-pretrain-job>`, so they become
eligible together and load tensor-identical encoder weights.

## Matched local evaluation

`monitor_and_eval.sh` follows the earlier Claude-style checkpoint pull loop.
It waits until both new trackers cross a frame target, pulls the nearest
checkpoint for both plus the old continuous-z256 controls, verifies both
encoder hashes, and evaluates every available arm on the same 4,096 sequential
BONES-SEED motions. Each checkpoint gets a deterministic, randomized-no-push
SONIC-compatible pass and a full-horizon pass with tracking terminations
disabled.

The monitor also includes the later `old_z256_critic_no_latent` ablation from
ICE job `5571183` when that run has reached each target. Its absence does not
change the matched frontier of the two FSQ64 tracker arms.

```bash
# One idempotent pull/evaluate cycle.
experiments/campaigns/2026-08-06-bones129k-sonic-fsq-scale/monitor_and_eval.sh

# The command used by each two-hour poll: newest checkpoint shared by both
# FSQ64 arms, plus matched controls.
experiments/campaigns/2026-08-06-bones129k-sonic-fsq-scale/monitor_and_eval.sh --latest

# Persistent two-hour loop. It first waits until two hours after the newest
# completed local evaluation, then invokes the same newest-shared-frontier
# cycle. A file lock prevents duplicate loops.
experiments/campaigns/2026-08-06-bones129k-sonic-fsq-scale/monitor_and_eval.sh --poll-latest

# Status without downloads or evaluation.
experiments/campaigns/2026-08-06-bones129k-sonic-fsq-scale/monitor_and_eval.sh --report

# Focus on one target or keep polling all configured targets.
THRESHOLDS=250000000 \
  experiments/campaigns/2026-08-06-bones129k-sonic-fsq-scale/monitor_and_eval.sh
experiments/campaigns/2026-08-06-bones129k-sonic-fsq-scale/monitor_and_eval.sh --watch
```

The watch and poll-latest intervals default to 7,200 seconds (two hours).
`--poll-latest` sleeps in chunks no longer than 60 seconds, writes
`poll_latest.last_cycle` after every attempted cycle, and exits after all three
tracked ICE jobs leave the queue and their final shared frontier evaluates.
`THRESHOLDS` remains available for an exact-frame replay.

Generated checkpoints, JSON summaries, hashes, logs, and validation markers
live under `logs/bones129k_sonic_fsq_scale_eval/` and are not source artifacts.
