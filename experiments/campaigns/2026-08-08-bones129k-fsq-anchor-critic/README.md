# BONES-129k — FSQ latent, scaled nets, expert-heading frame, critic without the latent

One low-level arm on ICE that combines four ingredients, each of which was
introduced and measured separately during 2026-08.

**This is a combined arm, not a single-variable ablation.** It cannot attribute
a gain to any one ingredient. Say so in every comparison that uses it.

## The four ingredients

| ingredient | setting | origin |
|---|---|---|
| SONIC-FSQ bottleneck | `--latent_mode sonic_fsq`, 64 coordinates x 32 levels | `2026-08-07-bones129k-latent-mode-stride5` |
| Scaled capacity | encoder `[2048, 1024, 512, 512]` SiLU, no layer norm; DiffSR feature 256 / embed 1024 with `[1024, 1024, 512]` heads; tracker actor and critic `[2048, 2048, 1024, 1024, 512, 512]` SiLU | same campaign |
| Expert-heading macro frame | `env.expert_macro_anchor_mode=expert_heading` | `2026-08-08-bones129k-anchor-frame` (ICE 5573233 → 5573234) |
| Critic without the actor latent | `env.command_interface.critic_channels=[reference]` | `2026-08-08-bones129k-anchor-critic-no-latent` (ICE 5573413) |

The FSQ quantizer publishes its output directly as the command, so `--z_dim`
must equal the number of FSQ coordinates. The published command is
64 code values plus a 2-value sin/cos phase = **66 wide**, held 10 control
steps. The actor input is 159 (93 non-command inputs plus the command); the
critic input is 286 and contains no `latent_command`.

## Macro window

Stride **1**, ten slots, root-qpos macro state (380 wide), endpoint DiffSR
objective. The scaled FSQ row this arm borrows its capacity from used stride 5
(SONIC's 0.9 s cadence); stride 1 was chosen here on 2026-08-08 so the
expert-heading and critic ingredients stay comparable to the arms that measured
them.

The macro state is 380 wide at **every** stride and under **both** anchor
modes, so a mispaired encoder produces no shape error. Three things prevent it:

1. The pretrain records `macro_frame_stride`, `macro_anchor_mode`,
   `horizon_steps`, `z_dim`, and `latent_mode` into the skill checkpoint.
2. The low level compares each against the live environment and refuses a
   mismatch.
3. The local smoke asserts (1) and exercises (2) negatively: it reruns the
   tracker with `env.expert_macro_anchor_mode=robot` against the
   `expert_heading` encoder and requires the run to fail with
   `anchor mode does not match`.

The smoke also parses the tracker's observation tables and requires that the
critic lost `latent_command` **and nothing else** — a width check alone would
pass if another term changed at the same time.

## Training contract

16,384 environments x 24 rollout steps, minibatch 294,912, gamma 0.97, seed 0,
Newton/MJWarp, `random80_adaptive20` resets, termination curriculum 5M → 30M,
10B frame cap (25,432 iterations).

Checkpoints land every **250M** frames — not the 50M of the anchor campaigns —
so this arm produces a checkpoint exactly at the 5B mark that the
`2026-08-08-bones129k-4096-scoreboard` protocol scores.

ICE caps one allocation at 16 h, so the tracker is expected to TIMEOUT before
the cap. Checkpoints are written under persistent `/data`, so a TIMEOUT loses
no training.

## Running it

```bash
MODE=print ./experiments/campaigns/2026-08-08-bones129k-fsq-anchor-critic/run.sh
MODE=smoke ./experiments/campaigns/2026-08-08-bones129k-fsq-anchor-critic/run.sh
MODE=validate LOCAL_SMOKE_ROOT=<smoke-dir> ./experiments/campaigns/2026-08-08-bones129k-fsq-anchor-critic/run.sh
MODE=submit LOCAL_SMOKE_ROOT=<smoke-dir> CONFIRM_SUBMIT=fsq-anchor-critic \
  ./experiments/campaigns/2026-08-08-bones129k-fsq-anchor-critic/run.sh
```

## Status

Submitted 2026-08-08 to ICE. W&B: project `g1-bones-seed`, group
`fsq-anchor-critic`. Full record in `cluster_submission.json`.

| stage | job | state at submit |
|---|---:|---|
| encoder pretrain | `5573502` | RUNNING, H200 `atl1-1-03-018-2-0` |
| tracker | `5573503` | PENDING `afterok:5573502` |

Gates cleared before submission: local smoke at source contract
`9a764d9bfc58f405…` (encoder records `sonic_fsq`, `expert_heading`, stride 1,
h10, z64, input width 380; matched tracker completed one PPO update with actor
159 and critic 286; mismatched `robot` tracker refused), remote reference
identity (129,785 / 47,491,234 / `bones_seed_sonic_full_129785@e714bbff`), and
two fresh output paths.

## Nearest reference rows

Scored on the frozen 4,096-motion scoreboard (ranks 12288–16383, 5B frames):

| row | SONIC SR | success-only MPJPE-L |
|---|---:|---:|
| `critic_no_latent` (z256, robot frame, tuned cells) | 0.9062 | 24.39 mm |
| `old_z256` (z256, robot frame, tuned cells) | 0.9058 | 24.52 mm |
| `fsq64_sonic` | 0.8943 | 25.74 mm |
| `fsq64_tuned` | 0.8711 | 27.81 mm |
| released SONIC checkpoint | 0.9937 | 28.65 mm |

The stride-5 scaled FSQ rows and the `expert_heading` arm were still training
when this arm was submitted.
