# BONES-129k — a dedicated wrist tracking reward

One low-level arm, one variable against the running ICE job `5573413`
(`expert_heading_critic_no_latent`):

```
env.rewards.motion_ee_pos.weight=2.0      # control: 0.0, inert
```

`motion_ee_pos` already exists in `G1SonicRewardsCfg` and is deliberately
switched off. It is rerooted wrist position error against the reference, std
0.1, over the two wrist yaw links, pelvis-anchored — the same geometry as
`motion_foot_pos`, on the hands. This arm turns it on; it introduces no new
reward code.

Everything else is the control's contract: expert-heading macro frame, stride
1, z256 command held 10 control steps, `critic_channels=[reference]`,
tuned-entry-point tracker capacity, 16,384 x 24, gamma 0.97, 10B cap,
`random80_adaptive20` resets — and the same frozen encoder file, matched by
SHA-256 `be6d533f…`.

## Why

On the frozen 4,096-motion scoreboard (ranks 12288–16383, 5B frames,
released-SONIC thresholds, `foot_pos_xyz` and `base_too_low` disabled), the
best arm's failures are almost entirely one termination:

| termination | rate | environments |
|---|---:|---:|
| `ee_body_pos` | 7.76% | 318 |
| `anchor_ori` | 1.39% | 57 |
| `anchor_pos` | 0.42% | 17 |

`ee_body_pos` is a Z-only height error over `G1_EE_BODY_NAMES` — both ankles
**and** both wrists. Rerunning the same evaluation with the termination
narrowed to one pair at a time (2026-08-09,
`logs/bones129k_ee_attribution/`):

| narrowed to | `ee_body_pos` | SR |
|---|---:|---:|
| wrists only | 231 | 0.9197 |
| ankles only | 208 | 0.9182 |

231 + 208 > 318, so many environments fail on both pairs. This is end-effector
height in general, not a wrist-only defect, and this arm does not claim
otherwise.

The wrists are the target because they are where a cheap reward change has
room: no termination bounds them horizontally, their only positional reward is
2 of 5 points in `tracking_reward_points`, and the dedicated term is inert. The
feet, by contrast, already carry `motion_foot_pos` at weight 2.0 plus a 3D
termination — they were given exactly this treatment once `foot_pos_xyz` was
found to dominate terminations.

For scale: the released SONIC checkpoint fails the same way on the same ranks,
26 times against our 318.

## Declared counter-evidence

An earlier screen (s13) added a local wrist term and improved root-relative EE
error while root drift **rose** and MPJPE-G got 28% worse. That was measured
before the v2 tuned rewards raised both global anchor terms from 0.5 to 2.0, so
the drift counterweight is now 4x stronger. If root drift rises again here,
that is the result, not a bug.

## The gate

The smoke asserts `motion_ee_pos` carries weight 2.0 in the reward table and
that every other term is untouched (`motion_body_pos` 2.0, `motion_foot_pos`
2.0, both anchors 2.0, `tracking_reward_points` 4.0, `action_rate_l2` 0.0),
plus actor 351 / critic 286 with no `latent_command` in the critic. The table
lists zero-weight terms too, so the weight value is what is asserted, not
presence.

## Running it

```bash
MODE=print ./experiments/campaigns/2026-08-09-bones129k-ee-reward/run.sh
MODE=smoke ./experiments/campaigns/2026-08-09-bones129k-ee-reward/run.sh
MODE=validate LOCAL_SMOKE_ROOT=<smoke-dir> ./experiments/campaigns/2026-08-09-bones129k-ee-reward/run.sh
MODE=submit LOCAL_SMOKE_ROOT=<smoke-dir> CONFIRM_SUBMIT=ee-reward \
  ./experiments/campaigns/2026-08-09-bones129k-ee-reward/run.sh
```

## Status

Submitted 2026-08-09 to ICE. W&B: project `g1-bones-seed`, group `ee-reward`.
Full record in `cluster_submission.json`.

| stage | job | state at submit |
|---|---:|---|
| tracker | `5573515` | submitted; no pretrain job (shared frozen encoder) |

Gates cleared: local smoke at source contract `220f147a74c98dc0…`, shared
encoder identity (`be6d533f…`, expert_heading, stride 1, h10, z256, width 380),
remote reference identity, and a fresh output path.
