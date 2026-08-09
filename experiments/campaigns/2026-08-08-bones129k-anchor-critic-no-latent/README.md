# 2026-08-08 — expert-heading anchor frame, critic without the actor latent

One low-level job. One variable against the running arm of
`experiments/campaigns/2026-08-08-bones129k-anchor-frame/`.

| row | job | W&B run | `critic_channels` |
|---|---|---|---|
| control | ICE `5573234` | `9lraqu2e` | `[actor, reference]` |
| arm | ICE `5573413` | `bones129k_expert_heading_critic_no_latent_seed0` | `[reference]` |

The critic keeps the noise-free reference channel and the privileged state, and
loses only the 258-D actor latent: 544 critic inputs become 286. The actor is
untouched at 351 inputs. Both rows log to `g1-bones-seed`, group
`latent-anchor-frame`.

## No pretrain job

This arm loads the **same file** the control loads,
`/data/bones129k_anchor_frame/expert_heading_encoder/checkpoints/latest.pt`,
SHA-256 `be6d533f1d1ca4aa6b1e819af1d3ef63eb033125018c8309c7448384b6a9583e`.
`validate` re-hashes it on ICE before submitting, so the encoder cannot drift
between the two rows and the comparison stays single-variable. The recorded
encoder contract is `expert_heading`, stride 1, horizon 10, z256, 380-wide
input.

## Why this arm

Removing the latent from the critic was already measured once, on the plain
robot-anchored baseline: run `mp85ex1f` against control `r09s1pc7`. On the
frozen 4,096-motion protocol at 5B frames it was a tie — SR 0.9060 versus
0.9058, success-only MPJPE-L 24.39 versus 24.52 mm.

The expert-heading frame changes the premise. Its latent is a pure function of
(trajectory, cursor) and no longer carries the robot's tracking drift, so the
critic's copy of it is closer to redundant with the reference channel than it
was in the baseline. This arm measures whether dropping it now helps, hurts, or
still does nothing.

Read it against what the control is actually doing at matched frames
(1.2B–1.49B window): `ee_body_pos` termination share 0.169, the lowest of any
arm we have run, paid for with `anchor_ori` 0.084 against the robot-anchored
control's 0.023, plus MPJPE-G 343 mm against 268 mm. The open question is
orientation drift, not EE tracking.

## Contract

Byte-identical to the control arm apart from the axis: `Isaac-Imitation-G1-v2`,
Newton MJWarp, seed 0, 16,384 environments x 24 rollout steps, minibatch
294,912, gamma 0.97, `random80_adaptive20` resets, termination curriculum
5M -> 30M, tuned entry-point tracker capacity (not overridden), macro state
`root_qpos` at stride 1 in the `expert_heading` frame, command 256 + sin/cos
phase held 10 control steps.

Frame cap 10B = 25,432 iterations. ICE allocations end at 16 hours, so the job
is **expected to TIMEOUT before the cap** — that is the accepted plan, not a
failure. Checkpoints land every 50M frames under persistent `/data`, so a
TIMEOUT loses no training and a continuation can resume from the newest intact
checkpoint.

## Running it

```bash
MODE=print ./experiments/campaigns/2026-08-08-bones129k-anchor-critic-no-latent/run.sh
MODE=smoke ./experiments/campaigns/2026-08-08-bones129k-anchor-critic-no-latent/run.sh
MODE=validate LOCAL_SMOKE_ROOT=<smoke-dir> ./experiments/campaigns/2026-08-08-bones129k-anchor-critic-no-latent/run.sh
MODE=submit LOCAL_SMOKE_ROOT=<smoke-dir> CONFIRM_SUBMIT=latent-anchor-frame \
  ./experiments/campaigns/2026-08-08-bones129k-anchor-critic-no-latent/run.sh
```

`smoke` runs one real local PPO iteration and then asserts the axis actually
took effect: the policy group still contains `latent_command`, the critic group
does not, the remaining critic terms are unchanged, and the two widths are 351
and 286. A width check alone would pass if some other term had also moved.

Gates cleared before this submission: local smoke at source contract
`7b68f5c0478a740f…`, remote reference identity (129,785 / 47,491,234 /
`bones_seed_sonic_full_129785@e714bbff`), remote encoder SHA-256 equal to the
control's, and a fresh output path. Full record in `cluster_submission.json`.

## Evaluating it

Score the resulting checkpoint on the frozen 4,096-motion scoreboard, not on
training curves:
`experiments/campaigns/2026-08-08-bones129k-4096-scoreboard/run.sh`.
