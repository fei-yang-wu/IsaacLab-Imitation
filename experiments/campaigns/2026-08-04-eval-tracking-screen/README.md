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

## Clip statistics do not predict failure

Correlating per-clip reference statistics against survival on the baseline
(40 clips):

| statistic | correlation with survival |
|---|---|
| mean \|joint velocity\| | **+0.012** |
| mean \|joint acceleration\| | **+0.027** |
| root speed | −0.270 |
| min root height | +0.233 |

Joint velocity and acceleration are uncorrelated with survival. The dance clips
carry the *highest* joint velocities in the dataset (1.35–1.74) and all eight
survive the full horizon; fallAndGetUp sits mid-range and all six fail. Within
run, jump, fight and sprint, some clips survive and some fail.

So "dynamic motions fail" is **not** supported — it was an over-reading of the
first 10-env sample, where the split happened to be dance against fallAndGetUp.
Only fallAndGetUp is a uniform class. Failure is per-clip and is not predicted
by any simple statistic tried here, which rules out a cheap clip-level
intervention (difficulty weighting, speed-gated thresholds) and points back at
general tracking capability as the lever.

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

## Results at 500M

Scored with `--randomization none`, 40 clips, seed 0, frame 0. Compared against
the **control at matched 500M** (job 5561149), not against the 1.9B baseline.

| arm | MPJPE strict | EE strict | survival | clips full | MPJPE full-horiz | EE full-horiz |
|---|---|---|---|---|---|---|
| baseline (1.9B, pre-screen) | 20.98 | 0.1041 | 449.0 | 24/40 | 41.19 | 0.2009 |
| **control** (std 0.30) | 22.03 | 0.0785 | 444.6 | 25/40 | 43.51 | 0.2272 |
| s1 (std 0.10) | 19.61 | 0.0791 | 442.1 | 24/40 | 43.42 | 0.1902 |
| **s2 (std 0.05)** | **17.89** | **0.0722** | 439.1 | 23/40 | 43.10 | 0.2034 |
| s3 (velocity kernels) | 23.03 | 0.0842 | 442.4 | 24/40 | 46.34 | 0.2145 |
| **s4** (body 0.10, w2) | 18.17 | 0.0820 | 442.6 | 23/40 | **39.81** | — |
| s5 (anchor 0.10) | 23.47 | 0.0821 | 444.2 | 24/40 | 44.18 | — |
| **s6** (anchor 0.10, w2) | 24.98 | **0.0599** | 443.8 | 24/40 | 43.35 | — |
| **s7** (wrist reward) | 19.11 | 0.0781 | **445.5** | 24/40 | 40.60 | — |
| s8 (foot allowance) | 22.37 | 0.0762 | 443.4 | 24/40 | 44.44 | — |

### The body-pos kernel sweep is bracketed: std 0.05 is the optimum

| std | MPJPE | EE | survival | full-horizon |
|---|---|---|---|---|
| 0.30 (control) | 22.03 | 0.0785 | 444.6 | 43.51 |
| 0.10 | 19.61 | 0.0791 | 442.1 | 43.42 |
| **0.05** | **17.89** | **0.0722** | 439.1 | 43.10 |
| 0.025 (s10) | 18.21 | 0.0746 | 444.1 | 40.61 |

The sweep turns at 0.025, so 0.05 is a real interior optimum for strict MPJPE
rather than an edge of the range tried. Going sharper than 0.05 trades MPJPE
back for survival (439.1 → 444.1) and full-horizon (43.10 → 40.61) — the same
precision-versus-survival tension the whole screen shows.

**The two goal metrics are won by different arms, with opposing trade-offs.**
s2 sharpens the root-relative body term and takes MPJPE (−18.8%) while EE moves
little; s6 upweights the global root term and takes EE (**−23.7%**) while MPJPE
gets worse. That is precisely the error decomposition: MPJPE-L is
*root-relative*, so the body kernel owns it, and world-frame EE is mostly root
drift, so the anchor term owns that. s12 combines one from each.

**s8 landed neutral (+1.5%)**, confirming the eval-time prediction made before
the run that `foot_pos_xyz` is a tripwire rather than the cause. **s3 and s5 are
negatives** — sharpening velocity kernels, or the anchor kernel without also
raising its weight, both degrade MPJPE.

**Sharpening `motion_body_pos` works, and is monotone**: 22.03 → 19.61 → 17.89
strict MPJPE, i.e. **−18.8%** at std 0.05, with EE **−8.0%**. Both are well
outside the ~2% seed spread the 2026-08-02 campaign measured.

**s3 is a clean negative**: sharpening the velocity kernels degrades every
axis — MPJPE +4.5%, EE +7%, full-horizon +6.5%. That refutes the "velocity is
the derivative channel of position tracking" hypothesis it was built on. Only
the *position* kernel helps.

Three things to read carefully:

- **Full-horizon barely moves** (43.51 → 43.42 → 43.10). That is exactly what
  the failure-dominance finding predicts: the full-horizon pass is governed by
  the clips that fail, not by precision on the ones that do not. Sharpening the
  kernel buys precision, not survival.
- **There is a survival cost**: 444.6 → 442.1 → 439.1, clips-full 25 → 24 → 23.
  Small, but consistently in the wrong direction, and it is the reason to find
  where the sweep turns rather than assuming sharper is always better.
- **The control already beat the 1.9B baseline on EE** (0.0785 against 0.1041,
  −25%) despite four times less training. That is the SONIC alignment plus the
  foot reward, not the kernel change.

### Training-seed variance is ~10%, not ~2% — headline moderated

| run | MPJPE | EE | survival |
|---|---|---|---|
| control (train-seed 0) | 22.03 | 0.0785 | 444.6 |
| s2 (train-seed 0) | 17.89 | 0.0722 | 439.1 |
| s2 (train-seed 1) | 19.79 | 0.0752 | 441.8 |

Both seeds beat the control, but the spread **within** s2 is 1.90 mm — about
10%, and far larger than the ~2% the 2026-08-02 campaign measured on per-minute
rates. So the honest effect is

- **MPJPE −14.5% mean (range −10.2% to −18.8%)**
- **EE −6.1% mean**

not the −18.8% a single seed suggested. The direction is solid — two independent
training seeds both beat the control on both metrics — but any single-seed arm
in the tables above carries roughly ±10% of uncertainty, which is comparable to
the differences between several of the arms.

**Read the single-seed rankings with that in mind.** Differences of a few
percent between arms (s2 vs s4 vs s10, for instance) are inside seed noise and
should not be treated as an ordering. The large effects — the ~4 mm control-to-
sharpened-kernel gap, s6's 23.7% EE gain, s3's and s5's degradations — survive
it; the fine ordering does not.

A matched control at train-seed 1 and an s1 repeat are running to put error bars
on the comparison rather than on the arm alone.

### Best setting: std 0.05 with weight 2.0 (s11)

At the full 500M checkpoint:

| arm | MPJPE | EE | survival | clips | full-horizon |
|---|---|---|---|---|---|
| control (0.30, w1) | 22.03 | 0.0785 | 444.6 | 25/40 | 43.51 |
| s2 (0.05, w1) | 17.89 | **0.0722** | 439.1 | 23/40 | 43.10 |
| s4 (0.10, w2) | 18.17 | 0.0820 | 442.6 | 23/40 | 39.81 |
| s10 (0.025, w1) | 18.21 | 0.0746 | 444.1 | 23/40 | 40.61 |
| **s11 (0.05, w2)** | **17.90** | 0.0777 | **443.9** | **24/40** | **37.28** |

**s11 is the best overall setting**: it matches s2's MPJPE (−18.8%) while
keeping essentially the control's survival (443.9 against 444.6) and posting the
best full-horizon number in the screen (37.28, **−14.3%**). Sharpening the
kernel alone buys precision at a survival cost; adding the weight back recovers
the survival without giving up the precision.

**Correction.** An earlier reading of this pair at the 300M checkpoint concluded
that kernel width and weight were "interchangeable, not additive", because s11
trailed s2 there (19.35 against 18.51). That does not hold at 500M — the two
converge on MPJPE and s11 wins survival and full-horizon outright. The 300M
comparison was made before either had converged, which is exactly the
scoring-at-an-unreached-mark error the 2026-08-02 campaign documented.

### The precision–survival trade-off

At 300M (control and s2 re-scored at the same checkpoint for a like-for-like
read):

| arm | MPJPE | EE | survival |
|---|---|---|---|
| control | 22.35 | 0.0775 | 433.2 |
| s2 (std 0.05, w1) | **18.51** | 0.0748 | 427.5 |
| s11 (std 0.05, w2) | 19.35 | 0.0808 | **437.0** |

Adding weight on top of the sharpened kernel makes MPJPE *worse* (−13.4% vs
s2's −17.2%) and survival better. Combined with s2-vs-s4 at 500M, the picture is
consistent: kernel width and term weight are largely **interchangeable, not
additive**, and moving either trades precision against survival rather than
buying both.

That means there is no single "best" setting — there is a frontier, and where to
sit on it depends on whether the downstream consumer cares more about tracking
precision or about episodes surviving. For a planner-training checkpoint,
survival is arguably worth more than the last 1 mm of MPJPE.

## Final screen table — no single config dominates

| arm | MPJPE | EE | survival | clips | full-horizon |
|---|---|---|---|---|---|
| control | 22.03 | 0.0785 | 444.6 | 25/40 | 43.51 |
| **s11** (body 0.05, w2) | **17.90** | 0.0777 | 443.9 | 24/40 | 37.28 |
| **s12** (body 0.05 + anchor-pos w2) | 21.19 | **0.0540** | 440.2 | 23/40 | 46.73 |
| **s13** (body 0.05 + wrist reward) | 18.93 | 0.0988 | **450.2** | 24/40 | **33.33** |
| s2 (body 0.05) | 17.89 | 0.0722 | 439.1 | 23/40 | 43.10 |
| s6 (anchor-pos w2) | 24.98 | 0.0599 | 443.8 | 24/40 | 43.35 |

Pick by what the downstream consumer needs:

- **strict MPJPE** → s11 (−18.8%), survival preserved
- **EE** → s12 (−31.2%), beating s6
- **full-horizon MPJPE and survival** → s13 (−23.4%, 450.2)

**s13 corrects an earlier call of mine.** I described the wrist reward as
"refuted" because s7 alone moved EE by 0.5%. Combined with the sharpened body
kernel it produces the best full-horizon *and* best survival in the screen,
while making world-frame EE worse. The wrist term does something real — it just
is not visible in the metric I judged it by, and world-frame EE was the wrong
lens because it is dominated by root drift.

For a **planner-training checkpoint**, s13 is the strongest candidate: survival
and full-horizon are what a planner consumes, and strict MPJPE at 18.93 is
within noise of the best.

## How to improve EE further

EE is **world-frame**, and decomposing it settles what to work on:

| component | control (500M) |
|---|---|
| EE, world-frame | 0.0783 m |
| root position drift | 0.0704 m |
| **EE, root-relative** (`ee_pos_error_local_m`) | **0.0331 m** |
| root orientation error | 0.055 rad |

Root-relative EE (33.1 mm) is almost exactly the orientation lever
(0.6 m mean lever arm × 0.055 rad = 33 mm). Since EE-L subtracts root *position*
but not root *orientation* — the same caveat as MPJPE-L — the root-relative EE
error is essentially **all root orientation**, and genuine wrist articulation
error is near zero.

So the EE budget is:

1. **Root position drift** — the largest component. `s6` cut it 43% and took EE
   down 23.7%. `s12` and `s15` carry that forward.
2. **Root orientation lever** — ~33 mm, and **no arm has moved it**: `root_ori`
   sits at 0.055–0.063 rad in every single run. `motion_global_anchor_ori` is
   kernel 0.933 / weight 0.5, exactly the state `motion_global_anchor_pos` was
   in before s6. `s14` applies the same recipe.
3. **Wrist articulation** — **refuted as a lever**. `s7` rewards wrist position
   directly and moved EE by 0.5% (0.0785 → 0.0781). Do not add wrist-specific
   rewards.

The order matters: as position drift falls, the orientation lever becomes the
dominant term. At s6's 40 mm drift the 33 mm lever is already nearly half the
budget, which is why s14 and s15 are the arms that matter for EE now.

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

## Round 9: the seed repeat, and what it invalidates

s15 was retrained at seed 1 with the evaluation seed held at 0, so the only
difference is the training seed.

| arm | strict MPJPE-G | vs control | full-horizon MPJPE-G | vs control | surv |
|---|---|---|---|---|---|
| control | 0.0758 | — | 0.2230 | — | 444.6 |
| s15 seed 0 | 0.0439 | −42.1% | 0.1740 | −22.0% | 438.7 |
| s15 seed 1 | 0.0533 | −29.6% | 0.2303 | **+3.3%** | 441.2 |
| s16 | 0.0334 | −55.9% | 0.1879 | −15.8% | 430.7 |

**The training-seed spread on full-horizon MPJPE-G is ~28% — larger than every
between-arm difference this campaign has reported on that pass.** s15's −22.0%
does not reproduce; its seed-1 repeat is marginally *worse* than control. So no
arm has a demonstrated full-horizon improvement, and the full-horizon column of
rounds 1–8 cannot rank arms. Those rankings were read off single seeds.

What survives:

- **The strict-pass gain is real.** Both s15 seeds sit far below control
  (−42.1%, −29.6%) against a 19.5% spread. The reward change does something.
- **Survival is the low-variance metric** — 438.7 against 441.2 across the same
  two seeds, well under 1%. Prefer it for ranking until MPJPE has repeats.
- **s16's strict win is confounded by survival.** It has the lowest survival in
  the screen (430.7, below control's 444.6), and strict MPJPE is scored only
  over frames a surviving episode reached, on a per-step curve that is flat to
  ~300 steps and then diverges. Dying earlier deletes the worst frames.

Why full-horizon is the noisy one: with every termination off, a fallen robot
keeps accumulating error for the rest of the horizon, so the mean is set by how
many environments fall and how far they drift — a heavy-tailed quantity. The
pass is still the right frame-matched comparison; it just needs repeats before
a difference under ~30% means anything. `per_environment` records survival and
termination terms but **not** per-environment metrics, so the tail cannot be
decomposed from a saved run.

### It is training-seed variance, not evaluation noise

The cheap fix would have been averaging evaluation seeds. It is not available.
Each checkpoint re-evaluated at evaluation seeds 0/1/2, full-horizon MPJPE-G:

| checkpoint | eval 0 | eval 1 | eval 2 | spread |
|---|---|---|---|---|
| control | 0.2230 | 0.2100 | 0.2126 | 6.1% |
| s15 seed 0 | 0.1740 | 0.1743 | 0.1736 | **0.4%** |
| s15 seed 1 | 0.2303 | 0.2366 | — | 2.7% |

Evaluation is nearly deterministic per checkpoint — 0.4% on s15 seed 0 — while
the gap between s15's two *training* seeds is **29.2%** on eval-seed means
(0.1739 against 0.2335). So one evaluation seed per checkpoint is plenty, and
the only way to resolve an arm is to retrain it. Against the control's eval-seed
mean of 0.2152, s15 seed 0 is −19.2% and s15 seed 1 is +8.5%.

Next: two more seeds of s15 and of the control before promoting anything.
