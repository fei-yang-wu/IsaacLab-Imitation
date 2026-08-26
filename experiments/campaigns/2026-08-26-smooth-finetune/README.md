# 2026-08-26 — Smoothness arms: restore the action-rate penalty

The 46.5B `ln_hold1_sonicreset` leader beats public `sonic_v1_1` on SR and
both MPJPE columns of the new common eval subset, but reads 4.67 m/s^2
acceleration distance against SONIC's 2.89 — visible joint shiver in every
video. The recipe trains with `env.rewards.action_rate_l2.weight=0.0` (zeroed
by the 2026-08-02 HP search for training speed); SONIC v1.1 trains with -0.1.
`anti_shake_ang_vel` is already active at -5.0e-3 in the inherited set.

## Arms (seed 0, W&B group `smooth-finetune`)

| arm | change | budget | resumes |
|---|---|---|---|
| `ar01` | `action_rate_l2` -0.1 (SONIC's value) | +2B (cap 48.5B) | pinned 46.5B file |
| `ar003` | `action_rate_l2` -0.03 | +2B | pinned 46.5B file |
| `ar01shake4` | -0.1 AND `anti_shake_ang_vel` -0.02 (4x) | +2B | pinned 46.5B file |
| `ar01scratch` | -0.1 from zero, base leader schedule | 10B | nothing |

The finetune arms resume
`/data/sonic_reset_50b/.../models/model_step_46500151296.pt` BY FILE — the
50B chain is still writing newer checkpoints into that tracker directory —
and keep its exact reset regime (`selection=sonic`,
`adaptive_uniform_ratio=0.05` static), so the reward axis is the only change
against the 46.5B parent. `ar01scratch` runs the leader's own 0-10B history
(`random80_adaptive20` + termination curriculum).

## Gate

Promote a weight only when, against the 46.5B parent:

- acceleration distance on `sonic_capability124_v1` drops toward SONIC's
  scale (score with
  `2026-08-25-sonic-paper-proxy/score_arms_capability124.sh`);
- SR and MPJPE-L on `bones_testbed4096_v1` stay inside evaluation noise.

Every finetuned checkpoint is a NEW arm: rescore, never splice into the
parent's row. The HP search zeroed this weight because it slows quality per
hour EARLY in training; that objection is weakest at the finetune stage, and
`ar01scratch` measures what it costs from zero.

## Status

- 2026-08-26 17:42: submitted from commit 63041d9, no drift.
  `ar01` 5592178-79, `ar003` 5592180-81, `ar01shake4` 5592182-83,
  `ar01scratch` 5592184-86. Nothing measured.
