# 2026-08-08 — BONES-129k macro-window anchor frame

One variable: the frame the DiffSR macro window is expressed in. One new arm on
ICE H200, measured against an existing control that is **not** resubmitted.

| row | job | `env.expert_macro_anchor_mode` | encoder |
|---|---|---|---|
| control | ICE `5567801` (`reset80_diffsr`, group `bones129k-ablation`) | `robot` (the historical split) | frozen, sha256 `d191d865…f8c5e7` |
| arm | this campaign | `expert_heading` | fresh pretrain, same recipe |

## The defect this tests

The historical convention splits the macro window across two frames:

| path | anchor | what slot 0 reads |
|---|---|---|
| DiffSR pretrain (`context="expert"`) | the **expert's** full pose at window slot 0 | exactly `(0,0,0)` + identity, in every sample |
| live encoder input (`context="rollout"`) | the **robot's** full live anchor pose | the live tracking error, nonzero and growing |

The two distributions differ by exactly the live tracking error. A frozen
encoder is therefore queried off its pretraining manifold precisely when
tracking is worst, and being frozen it never adapts. Nine of the 380 encoder
inputs are constant in pretraining and not constant at rollout.

`expert_heading` uses one convention for both: the expert's slot-0 heading
(yaw-only, swing-twist) frame with an xy-only origin. Per slot,

- `pos_b = R_head0ᵀ (p_k − (x₀, y₀, 0))` — xy relative and rotated, **z absolute**
- `ori_b = 6D(R_head0ᵀ R_k)` — roll/pitch absolute, yaw relative to slot-0 heading

Pretrain and rollout inputs then match by construction, and the latent becomes
a pure function of (trajectory, cursor) — robot-independent, which also makes
planner oracle latent targets deterministic rather than contaminated by the
collecting policy's drift.

## Why yaw-only, and how that relates to SONIC

The frame cancels global yaw and xy and nothing else, so absolute height and
roll/pitch relative to gravity survive. That is exactly the invariance group of
the re-rooted tracking reward (`reroot_body_positions`: heading-only delta, xy
from the robot, z from the reference).

This is *not* copied from SONIC's encoder. Two different SONIC conventions,
both verified:

1. **Encoder input — full orientation, not yaw.** The released checkpoint's
   anchor-ori term is `matrix_from_quat(quat_inv(robot_anchor_quat) @
   ref_root_quat)[..., :2]`, recovered bitwise by ONNX probe. It can afford a
   full-orientation, robot-anchored term because its tokenizer is online
   (hold 1, PG into the encoder), so the term is always on-distribution.
2. **Reward / termination re-rooting — yaw-only.**

A frozen offline encoder cannot buy (1). So the criterion here is the reward's
invariance group, not SONIC's encoder convention.

## Declared risk (user decision)

The actor input keys are **unchanged** from the control. No observation term
replaces the drift signal the robot-anchored latent used to carry implicitly,
so the policy cannot directly observe xy/yaw drift. Expect MPJPE-L to hold and
MPJPE-G / push recovery to degrade. That is the measurement, not a defect —
and keeping the actor contract fixed is what makes this arm single-variable.
Training pressure against drift survives through the IPMD reward-input channel
and the anchor rewards/terminations, which are unchanged.

## Matched contract

Reproduced exactly from the control (verified against `5567801`'s arm
overrides and the control encoder checkpoint's recorded config):

- Task `Isaac-Imitation-G1-v2`, Newton MJWarp, seed 0.
- Macro state `root_qpos` (`expert_motion_qpos` + anchor pos + anchor ori),
  38/frame, 380 over the 10-slot window, `frame_stride=1`.
- Encoder: z256, `deterministic`, `intermediate` window, `endpoint` objective,
  MLP `[1024, 512, 512]`, mish, LayerNorm **on**; DiffSR feature 128, embed
  512, `g`/`mu` `[512]`; 50,000 updates at batch 8,192.
  These were the trainer defaults when the control was pretrained; they are
  passed explicitly here so a later default change cannot move this arm.
- Command: 256 code + sin/cos phase = 258, held 10 control steps.
- Tracker/critic capacity: the tuned entry point's default, **not** overridden
  — the control did not override it either.
- 16,384 envs × 24 rollout steps, minibatch 294,912, gamma 0.97, 10B frame cap
  (25,432 iterations), checkpoints every 50M, `random80_adaptive20` resets,
  termination curriculum 5M → 30M.
- Reference arrays `g1_bones_seed_sonic_full_129785_e714bbff_v1`
  (129,785 trajectories, 47,491,234 frames), resident.

ICE caps one allocation at 16 h, so the controller will not reach 10B in a
single segment — same as the control, which is what keeps the comparison fair.

## Running it

```bash
MODE=print ./experiments/campaigns/2026-08-08-bones129k-anchor-frame/run.sh
MODE=smoke ./experiments/campaigns/2026-08-08-bones129k-anchor-frame/run.sh
MODE=validate LOCAL_SMOKE_ROOT=<smoke-dir> ./experiments/campaigns/2026-08-08-bones129k-anchor-frame/run.sh
MODE=submit LOCAL_SMOKE_ROOT=<smoke-dir> CONFIRM_SUBMIT=latent-anchor-frame \
  ./experiments/campaigns/2026-08-08-bones129k-anchor-frame/run.sh
```

The macro state is 380 wide in **both** modes, so a mispaired encoder produces
no shape error and no warning — only a silently off-distribution command. Three
things prevent that:

1. `env.expert_macro_anchor_mode` is recorded into the skill checkpoint at
   pretrain time (deliberately no CLI flag — two sources for one value is how
   they drift apart).
2. The low level compares the checkpoint's mode against the live environment's
   and refuses the pairing, exactly as it already does for `horizon_steps` and
   `macro_frame_stride`.
3. The local smoke asserts (1) and exercises (2) **negatively**: it reruns the
   tracker with `env.expert_macro_anchor_mode=robot` against the
   `expert_heading` encoder and requires the run to fail with
   `macro-window anchor mode does not match`.

A checkpoint written before the field existed reads back as `robot`, which is
what it was — so the control's encoder stays loadable and is refused only if
paired with an `expert_heading` environment.

## Status

Submitted 2026-08-08 to ICE. W&B: project `g1-bones-seed`, group
`latent-anchor-frame`. Full record in `cluster_submission.json`.

| stage | job | state at submit |
|---|---:|---|
| encoder pretrain | `5573233` | RUNNING, H200 `atl1-1-03-017-9-0` |
| tracker | `5573234` | PENDING `afterok:5573233` |

Gates cleared before submission: local smoke at source contract
`7ff03c839ec801ad…` (checkpoint records `macro_anchor_mode='expert_heading'`,
stride 1, encoder width 380; matched tracker completed one PPO update;
mismatched `robot` tracker refused with `Skill encoder macro-window anchor
mode does not match the environment`), remote reference identity
(129,785 / 47,491,234 / `bones_seed_sonic_full_129785@e714bbff`), and two
fresh output paths.
