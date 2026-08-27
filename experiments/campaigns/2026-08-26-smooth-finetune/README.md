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

## Verdict (2026-08-26): NO PROMOTION

All three finetune arms finished their 2B budget and were scored on both
paper boards. The gate was: acceleration distance drops toward SONIC's scale
AND SR / MPJPE-L on `bones_testbed4096_v1` stay inside noise. **The second
half fails.** On the deciding board, matched over the 3,932 clips every row
completes:

| arm | SR | MPJPE-L | MPJPE-G | acc m/s^2 |
| --- | ---: | ---: | ---: | ---: |
| parent @46.5B | 0.9773 | 21.95 mm | 92.31 mm | 5.45 |
| `ar003` -0.03 | 0.9756 | 22.53 mm | 99.45 mm | 4.84 |
| `ar01shake4` | 0.9753 | 24.02 mm | 106.94 mm | 4.38 |
| `ar01` -0.1 | 0.9734 | 24.19 mm | 102.94 mm | 4.43 |

Every arm pays SR, local error, and global error for its smoothness gain, and
none reaches SONIC's 3.34 m/s^2. `ar003` is the efficient point (-11% acc for
+0.58 mm L); `ar01shake4` is dominated and the 4x anti-shake is dead. The
user's read on 2026-08-26: keep the parent as the paper row.

The 124-clip board is more favorable to the finetunes (`ar003` improves
MPJPE-G 91.90 -> 78.65 mm at +0.23 mm L) but it is the calibration board, not
the deciding one, and it was selected from SONIC's own results.

Open, not refuted: the acceleration gap to SONIC is real and unclosed. A
future attempt should change the objective rather than the weight -- the
weight sweep is now measured and it buys acc by spending accuracy.

## Status

- 2026-08-26 22:16: all three finetune arms COMPLETE at 48.5B and scored on
  both boards; verdict above. `ar01scratch` still training (3.15B of 10B).
  INCIDENT: `ar003`'s first segment (5592180) went non-finite at 47.82B and
  `IPMD._abort_on_nonfinite` aborted it before a poisoned checkpoint was
  written -- the guard's first real save. `lowlevel2` resumed from the last
  good 47.5B checkpoint and trained cleanly to 48.5B; all 41 float tensors of
  every final checkpoint verified finite. `ar003`'s history is therefore NOT
  identical to the other two arms.
- 2026-08-26 17:42: submitted from commit 63041d9, no drift.
  `ar01` 5592178-79, `ar003` 5592180-81, `ar01shake4` 5592182-83,
  `ar01scratch` 5592184-86. Nothing measured.
