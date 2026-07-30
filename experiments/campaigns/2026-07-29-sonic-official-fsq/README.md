# 2026-07-29 — SONIC official-window FSQ

This campaign replaces the removed cached-packet SONIC implementation, which
created ten independent future-frame tokens and consumed the cached packet one
token per step. The
public SONIC actor instead encodes the complete 10-frame G1 reference window
into two 32-D token blocks and runs that encoder as part of every policy
forward pass.

The new task is `Isaac-Imitation-G1-Latent-SonicOfficialFSQ-v0`:

- full SONIC environment, 10-step proprioceptive history, pelvis anchor,
  curriculum, rewards, events, actuators, and robot preset, with this repo's
  sample-efficient `[0, 200]` non-full-trajectory reset sampler;
- current plus nine consecutive 50 Hz future frames as one encoder input;
- one 64-D FSQ command recomputed every control step (`code_period=1`);
- co-trained encoder with PPO gradient plus the 0.01 future-window
  reconstruction auxiliary;
- release optimizer geometry: 4096 envs x 24 steps, 5 epochs, 4 minibatches,
  SiLU networks, actor LR 2e-5, critic LR 1e-3, grad clip 0.1.

The public config uses 32 FSQ levels on each of the 64 scalar coordinates
(`max_num_tokens=2`, `fsq_level_list=32`). This campaign fixes that value; it
has no higher-level-count variant.

## Validation and submission status

The exact task completed a two-environment, one-rollout, one-optimizer-update
Isaac Lab smoke on 2026-07-29. The complete default-environment RLOpt suite
also passed (107 tests), including overflow-safe coverage for the `32**64`
product space and normalized per-coordinate code publication.

Initial job `5549447` was cancelled with zero runtime after an inheritance
audit found it still used SONIC's full-trajectory adaptive-failure reset
sampler. Replacement ICE job `5549500` was submitted on 2026-07-29 with the
reset contract pinned both in the task and on its command line: start range
`[0, 200]`, `random_reset_full_trajectory=false`, and adaptive-failure ratio
`50`. Its workspace archive SHA-256 is
`83a1cdfc34010de25062b35900269a0f1cbda629fb130cd00266a927f28e4e94`.

Job `5549500` requests one H200, 4,096 environments x 24 rollout steps, and
12,817 PPO iterations = 1,259,962,368 environment frames. At the last
submission-time check it was pending for H200 resources. This is the first
segment under the 5B cumulative cap, not a claim that 5B has completed.

```bash
# Safe dry run.
MODE=print ./experiments/campaigns/2026-07-29-sonic-official-fsq/run.sh

# Remote input/log-dir validation, still no submission.
MODE=validate ./experiments/campaigns/2026-07-29-sonic-official-fsq/run.sh

# Deliberate initial submission.
MODE=submit CONFIRM_SUBMIT=sonic-official-fsq32 \
  ./experiments/campaigns/2026-07-29-sonic-official-fsq/run.sh
```

The launcher is fixed to corrected LAFAN1 (manifest SHA-256
`d972c37c...c945db8`), seed 0, persistent `/data` checkpoints, a 5B cap, and a
single H200 segment per invocation. It refuses to overwrite an existing
initial-run directory. Resume requires both `TRAIN_CHECKPOINT` and the exact
`COMPLETED_FRAMES` credit.

Official references: [NVIDIA GEAR-SONIC release config](https://huggingface.co/nvidia/GEAR-SONIC/blob/main/low_latency/config.yaml) and [NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl).
