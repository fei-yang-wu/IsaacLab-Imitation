# 2026-08-04 — Eval-time tracking screen

Goal: lower **evaluation-time** MPJPE and end-effector tracking error on the G1
low level. Seven arms at 500M frames, scored by evaluating the checkpoint, not
by reading the training curve.

```bash
DRY_RUN=1 ./submit_eval_tracking_screen_ice.sh       # plan
DRY_RUN=0 ./submit_eval_tracking_screen_ice.sh       # submit all
./score_eval_tracking_screen.sh                      # score whatever reached 500M
```

Arms and their rationale live in `arms.sh`, sourced by both scripts so the table
has one definition.

## Baseline

The pre-screen checkpoint (`model_step_1900118016`, the 2026-08-03 aligned 5B
run), 10 envs from frame 0, seed 0:

| pass | MPJPE mm | EE (world) m | root drift m | survival |
|---|---|---|---|---|
| strict, DR off | 20.21 | 0.0535 | 0.0547 | 425.1 |
| strict, DR on | 25.22 | 0.1631 | 0.1521 | 420.2 |
| **full-horizon, DR off** | **59.68** | **0.1339** | **0.1295** | 500.0 |

Read the full-horizon row as tracking quality: the strict pass scores MPJPE only
over frames a live episode reached, so it is biased toward whatever survived.

## Full-dataset baseline — the numbers to beat

All 40 LAFAN1 clips, one env each, DR off, seed 0, frame 0
(`model_step_1900118016`):

| pass | MPJPE mm | EE m | root m | survival |
|---|---|---|---|---|
| strict | **20.98** | **0.1041** | 0.1001 | 449.0 |
| full-horizon | **41.19** | **0.2009** | 0.1939 | 500.0 |

Survival by motion class (strict), and the reason a single average misleads:

| class | survived full | mean survival |
|---|---|---|
| dance | 8/8 | 500.0 |
| walk | 11/12 | 485.0 |
| sprint | 1/2 | 492.5 |
| fight | 2/5 | 434.2 |
| run | 1/4 | 407.2 |
| jump | 1/3 | 387.0 |
| **fallAndGetUp** | **0/6** | 365.8 |

**18 of 40 clips fail**, and the cause distribution is lopsided:

| cause | count |
|---|---|
| `foot_pos_xyz` | **13** |
| `ee_body_pos` | 2 |
| `anchor_ori` | 2 |
| `anchor_pos` | 1 |

`foot_pos_xyz` is 72% of all failures across the whole dataset.

Note the scope of `s8`: the crouching allowance fires only when the *reference*
root is low, so it targets the 6 fallAndGetUp clips. `run`, `jump` and `fight`
failures happen at normal or high root height and will need something else —
worth knowing before reading s8's result as a general fix.

## foot_pos_xyz is a tripwire, not the cause

Before spending 500M-frame runs on the allowances, they were applied at
EVALUATION time to the unchanged baseline checkpoint (40 clips, DR off):

| config | MPJPE | EE | survival | clips full | causes |
|---|---|---|---|---|---|
| baseline | 20.98 | 0.1041 | 449.0 | 24/40 | foot 13, ee 2, ori 2, pos 1 |
| + low-root (s8) | 20.88 | 0.1024 | 447.6 | 24/40 | **foot 7**, ee 4, ori 5, pos 1 |
| + low-root + swing (s9) | 21.16 | 0.1066 | 450.6 | 25/40 | **foot 5**, ee 4, ori 5, pos 1 |

The allowances do what they were designed to do — foot terminations fall
13 → 7 → 5 — but **the failures migrate to other terms** rather than
disappearing. `anchor_ori` goes 2 → 5, `ee_body_pos` 2 → 4, and net survival is
flat: 449.0 → 447.6 → 450.6, with clips surviving 24 → 24 → 25.

So `foot_pos_xyz` is the first tripwire, not the cause. On these dynamic clips
the robot is genuinely losing the reference; removing one detector lets the
failure continue until another catches it.

**Caveat on how far this generalises.** This applies the allowance to a policy
*trained without* it. Training with it could produce a different policy — one
that learns to recover from a foot excursion instead of being terminated at it.
So this does not refute s8/s9 outright; it refutes the "just relax the tripwire"
reading, and it means s8/s9 should be expected to be roughly neutral on survival
unless training changes the policy's recovery behaviour.

The consequence is that termination definition is probably not the binding
constraint, which re-elevates the round-1 reward arms (s1/s2/s4): improving
tracking capability on dynamic motions is the thing that would actually move
both metrics.

## The horizon curve: precision is fine, failures are not

Per-step root-relative MPJPE over a 500-step rollout, DR off, tracking
terminations off (`scripts/audit/sim2sim_backend_eval.py`, 10 envs, frame 0):

| steps | mean MPJPE |
|---|---|
| 0–50 | 11.24 mm |
| 50–150 | 11.05 mm |
| 150–300 | 11.66 mm |
| **300–499** | **35.19 mm** |

**Tracking is flat at ~11 mm for the first 6 seconds, then diverges.** That is
not gradual drift accumulation; it is late-episode failure. The strict pass
agrees: `done_rate` 0.50 and `survival_steps_mean` 425/500, so roughly half the
environments fail, concentrated after step ~300, and with terminations disabled
those failures run away and dominate the horizon mean.

So the 59.7 mm full-horizon figure is not a precision number. It is
`~11 mm of real tracking` blended with `a diverged tail`. Two consequences:

1. **Precision is already good.** Sharpening a reward kernel to chase 20 → 15 mm
   is chasing a term that contributes little to the headline number. The
   round-1 arms (s1/s2/s4) are therefore less likely to move the eval metric
   than their gradient analysis suggests.
2. **The lever is survival.** Anything that reduces late failures moves the
   full-horizon number far more. `foot_pos_xyz` is 66% of non-timeout
   terminations, which is why the foot reward is in the default and why the
   remaining failure modes are worth attacking directly.

Precision and survival are coupled — better tracking means fewer threshold
crossings — so the reward arms are not worthless. But read them against
survival and the full-horizon number, not against strict MPJPE alone.

## Where the error actually is

**Root drift accumulates, and it is the dominant eval-time failure.** It grows
54.7 mm → 129.5 mm between the strict pass and the full horizon, and world-frame
EE error tracks it almost exactly (133.9 vs 129.5 mm).

Decomposing the DR-off strict pass:

| quantity | value |
|---|---|
| MPJPE-L (root-relative) | 20.2 mm |
| root drift (world) | 54.7 mm |
| root drift, horizontal only | 44.2 mm |
| EE error (**world frame**) | 53.5 mm |
| tracked-body error (world) | 54.3 mm |

`ee_pos_error_m` in the evaluator is world-frame — `actual_pos - ref_pos`, no
root subtraction. So **world-frame EE error is almost entirely root drift**, not
the wrists mistracking relative to the body. Any attempt to improve "EE
tracking" that does not reduce drift is working on the wrong term.

The same holds for global MPJPE: `mpjpe_g_mm` ≈ root drift + pose error, and the
drift is the larger half.

## Which reward terms still have gradient

Measure, do not assume. IsaacLab logs
`Episode_Reward/<term> = weight · mean(kernel) · ep_len/500`, so the kernel value
is recoverable from a live run. Inverted from the control at 260M:

| term | kernel | implied err | gradient |
|---|---|---|---|
| `motion_body_pos` | **0.970** | 0.052 | −1.13 |
| `motion_global_anchor_ori` | 0.933 | 0.106 | −0.62 |
| `tracking_reward_points` | 0.870 | 0.037 | **−25.94** |
| `motion_body_lin_vel` | 0.866 | 0.379 | −0.66 |
| `motion_foot_pos` | 0.849 | 0.040 | −13.73 |
| `motion_body_ori` | 0.767 | 0.206 | −1.97 |
| `motion_body_ang_vel` | 0.692 | 1.905 | −0.27 |
| `motion_global_anchor_pos` | **0.599** | 0.215 | −1.43 |

`motion_body_pos` — the term whose error *is* MPJPE — is the most saturated in
the config and supplies ~23× less gradient than `tracking_reward_points`. Its
exp kernel at std 0.30 is flat by the precision we care about, so the policy is
paid almost nothing for improving. That is a mechanical explanation for the
plateau and needs no new term to fix, only a narrower kernel.

**Read this at the training operating point, not the evaluation one.** An
earlier version of this analysis used eval-time errors and concluded
`motion_global_anchor_pos` was 96.7% saturated. It is not — at training, where
domain randomization and exploration noise make errors much larger, it is the
*least* saturated term at 0.599. That mistake is why `s5` is only weakly
motivated; see `arms.sh` for the correction.

## Scoring

`score_eval_tracking_screen.sh` pulls each 500M checkpoint and runs
`evaluate_checkpoint --randomization none` in two passes:

- **strict** — every termination active. The protocol number, but MPJPE is
  scored only over frames a surviving episode reached, so a policy that dies
  early can post a flattering value.
- **full_horizon** — every early termination off including `base_too_low`,
  fixed length, so every arm is scored over identical frames.

On the pre-screen checkpoint these read 25.22 and 68.08 mm. The choice is not
cosmetic; quote the full-horizon number as tracking quality.

**The training curve cannot substitute for this**, for three measured reasons:

| effect | size |
|---|---|
| domain randomization live during training | 20.21 → 25.22 mm |
| terminal-step vs episode-mean logging (runs before 2026-08-04) | 30.9 → 64.8 mm |
| exploration noise vs MODE actions | the residual |

## Control

Job 5561149, `lafan1_v2_foot_reward_5b_seed0_e12288_r24`, is the current default
and passes 500M on its way to 5B. It is the matched control and costs nothing —
**do not submit a separate one.**

## Caveats

- `tracking_reward_points.weight=4.0` is carried forward unscreened: it was
  tuned when that term tracked 3 points without the feet, and now tracks
  SONIC's 5.
- `motion_foot_pos.weight=2.0` and `motion_ee_pos`'s std are likewise
  considered starting points, not tuned values.
- One seed per arm. The 2026-08-02 campaign measured ~2% seed spread on
  per-minute rates and larger node-to-node variation; treat differences below
  a few percent as unresolved.
