# BONES-129k — command hold period, on the old z256 recipe

One low-level arm, one variable against the scoreboard row `old_z256` = ICE job
`5567801` (`reset80_diffsr`, W&B group `bones129k-ablation`), whose 4,096-motion
row is SR 0.9058 / 24.52 mm at 5B frames.

```
agent.ipmd.latent_steps_min=1
agent.ipmd.latent_steps_max=1
agent.ipmd.latent_learning.code_period=1      # control: 10, 10, 10
```

The control holds one published latent for 10 control steps (200 ms at 50 Hz).
This arm re-encodes every control step, as the released SONIC tokenizer does.
Holding for 200 ms means the tracker acts on a command derived from a macro
window up to 200 ms stale; hold 1 removes that staleness.

## The old z256 recipe, unchanged

Frozen root-qpos DiffSR encoder (the exact file the control loaded, SHA-256
`d191d865…f8c5e7`), **robot** anchor frame, macro stride 1, h10, z256 plus
sin/cos phase, the tuned entry point's default tracker capacity, and the
default critic channels `[actor, reference]` — so the critic still reads the
latent here. Actor input 351, critic input 544. 16,384 x 24, minibatch 294,912,
gamma 0.97, seed 0, `random80_adaptive20` resets, curriculum 5M → 30M, 10B cap,
checkpoints every 50M (so a 5B checkpoint exists for the scoreboard).

No pretrain job: the encoder is reused as-is.

The sin/cos phase channel is **kept** even though `code_period=1` makes it
constant. Dropping it would shrink the published command from 258 to 256 and
move the actor input width, so the hold would stop being the only variable. Two
constant inputs are the cheaper price.

## Two caveats to read before comparing

1. **Bandwidth.** Hold 1 publishes 50 commands/second instead of 5. As a
   low-level ceiling that is the question being asked. It is **not** a
   planner-interface row — the paper's planner comparison publishes at 5 Hz,
   and a 50 Hz latent stream is not something the high level can produce.
2. **Cost.** The skill encoder now runs on every control step rather than every
   tenth, so expect lower frames per second than the control. Compare at equal
   **frames**; if you compare at equal wall-clock, say so.

## The gates

The smoke asserts the **resolved** agent config (`params/agent.yaml`, written
by the run — not the command line) has all three hold knobs at 1 and still has
`command_phase_mode: sin_cos` and `code_latent_dim: 256`. All three knobs are
checked because holding is enforced in two places, and setting only one of them
silently keeps the other. It then asserts actor 351 and critic 544 with
`latent_command` present in **both** groups, so this arm is not accidentally the
critic ablation.

## Running it

```bash
MODE=print ./experiments/campaigns/2026-08-09-bones129k-hold1/run.sh
MODE=smoke ./experiments/campaigns/2026-08-09-bones129k-hold1/run.sh
MODE=validate LOCAL_SMOKE_ROOT=<smoke-dir> ./experiments/campaigns/2026-08-09-bones129k-hold1/run.sh
MODE=submit LOCAL_SMOKE_ROOT=<smoke-dir> CONFIRM_SUBMIT=hold-period \
  ./experiments/campaigns/2026-08-09-bones129k-hold1/run.sh
```

## Status

Submitted 2026-08-09 to ICE. W&B: project `g1-bones-seed`, group
`hold-period`. Full record in `cluster_submission.json`.

| stage | job | note |
|---|---:|---|
| tracker | `5573633` | no pretrain job; frozen old z256 encoder |

Gates cleared: local smoke at source contract `706643d08e0f8c77…`, encoder
identity (`d191d865…`, robot frame, stride 1, h10, z256, width 380), remote
reference identity, and a fresh output path.
