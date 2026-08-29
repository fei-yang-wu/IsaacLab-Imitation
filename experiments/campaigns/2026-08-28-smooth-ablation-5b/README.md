# 2026-08-28 — Smoothness ablation at 5B

The 2026-08-26 `action_rate_l2` weight sweep measured that the weight axis
buys acceleration by spending accuracy, and concluded the next attempt should
change the objective. This campaign runs the first objective-level cells, plus
the two 2026-08-28 SONIC parity corrections as isolated variables.

## Base

`diffntp_chunk_h1_ee_wide`'s frozen encoder (round-4 winner), tracker from
scratch to 5B in the `merged64_pen_ramp_5b` regime: `action_rate_l2` -0.03,
SONIC selection with the failure-share ramp 0.8 -> 0.2 over the first 1B.
Two deliberate changes against that arm, carried by every arm here so the
star stays internally matched:

- 256-D command (not 64-D)
- 20,480 envs (not 16,384) — the measured safe H200 step; 24,576 OOM'd the
  Newton graph launch (job 5580202, 2026-08-18 campaign)

Every arm also trains at the corrected `feet_acc` -2.5e-6 (in-tree default
since 2026-08-28).

## Arms (seed 0, W&B group `smooth-ablation-5b`)

| arm | the one variable | question |
|---|---|---|
| `base` | — | the 256-D + env20k + corrected-feet_acc control |
| `energy` | `env.rewards.energy_consumption.weight=-1.0e-4` | SONIC's whole-body mechanical-power penalty, at its released weight, on our stack |
| `sigma` | `agent.ppo.log_std_init=log(0.05)`, clamped `[0.001, 0.5]` | does bounding exploration noise let `action_rate_l2` smooth the MEAN instead of fighting sampling noise? Ours is an unbounded init-1.0 Gaussian; sigma drifted to ~0.16 by 5B in `merged64_pen_ramp_5b` |
| `feetacc_weak` | `env.rewards.feet_acc.weight=-2.5e-7` | isolates the 2026-08-28 10x parity correction (this arm trains at the old wrong value) |
| `ar0` | `env.rewards.action_rate_l2.weight=0.0` | the penalty's own effect at matched frames, schedule, and batch shape — the decomposition `merged64_pen_ramp_5b` asked for |

## Schedule

One 15:59 H200 segment carries the full 5B (`merged64_pen_ramp_5b` finished
5B in 10:43 at 16,384 envs). Segment 2 is a safety resume that PINS the
landed ratio 0.2 (`select_hold`), so a restart never re-runs the ramp — the
ramp keys off `common_step_counter`, which restarts per segment.

## Scoring

Final checkpoints on `bones_testbed4096_v1`, clean and robust, plus the
milestone curve. Two metric requirements beyond the standard row:

1. `tracking_acceleration_distance_mps2` is an acceleration TRACKING ERROR
   (reference-relative), not a smoothness measure. Report it for SONIC
   comparability, but do not attribute smoothness with it alone.
2. Surface the action-space measures (`action_l2`, `action_delta_l2`,
   eval-side `action_rate_l2`) that `evaluate_checkpoint.py` computes —
   no scored board row has ever carried them. The reference-free jerk /
   high-frequency-power metric is still to be added to the evaluator.

## Final 5B rows: base / energy / ar0 (2026-08-29, one seed)

`bones_testbed4096_v1` (the deciding board). The `sonic_v1_1` row is the same
board's all-clip figure; it predates the jerk/adelta metrics.

| arm | row | SR | L | G | acc | jerk | adelta |
|---|---|---:|---:|---:|---:|---:|---:|
| `sonic_v1_1` (public 42M) | clean | 0.9888 | 26.73 | 187.65 | **3.45** | — | — |
| `base` | clean | 0.9446 | 26.93 | **86.01** | 4.75 | 204.0 | 0.835 |
| `energy` | clean | 0.9268 | 26.98 | 86.79 | 4.73 | 202.5 | **0.818** |
| `ar0` | clean | **0.9587** | **24.24** | 93.09 | 5.62 | 293.1 | 1.257 |
| `sonic_v1_1` | robust | 0.9895 | 28.97 | 228.52 | **3.56** | — | — |
| `base` | robust | 0.9336 | 30.36 | 140.38 | 5.45 | 243.0 | 1.039 |
| `energy` | robust | 0.9041 | 29.93 | 144.80 | 5.51 | 249.3 | 1.056 |
| `ar0` | robust | 0.9402 | 27.50 | 185.45 | 6.70 | 359.0 | 1.648 |

`sonic_capability124_v1` (calibration board — never mix with the table above;
this board reads ~0.82x the 4,096-clip acceleration):

| arm | SR | L | G | vel | acc | jerk | adelta |
|---|---:|---:|---:|---:|---:|---:|---:|
| `sonic_v1_1` | **1.0000** | 23.89 | 174.23 | 0.165 | **2.89** | — | — |
| `base` | 0.9516 | 22.11 | **59.20** | 0.172 | 3.77 | **160.7** | 0.673 |
| `energy` | 0.9597 | 22.79 | 67.11 | 0.180 | 3.86 | 164.1 | **0.672** |
| `ar0` | 0.9758 | **20.50** | 78.48 | 0.185 | 4.65 | 239.8 | 1.036 |

Readings, all one seed:

- **The penalty decomposition at 5B** (`base` vs `ar0`, everything else
  matched): `action_rate_l2` -0.03 costs 0.014 SR and +2.7 mm L, and buys
  -30% jerk (293 -> 204), -34% adelta, and -7 mm G on the deciding board.
  The 3.75B intermediate had shown the penalty AHEAD on L too; that flipped
  because `base`'s local error regressed 22.78 -> 26.93 over the last 1.25B
  while `ar0`'s stayed flat (24.21 -> 24.24) — an UNRESOLVED late-training
  regression specific to the penalty arm on this base. The 20-checkpoint
  milestone curve for `base` is the diagnostic before promoting anything.
- **`energy` converges to `base`'s smoothness, not below it** (jerk 202.5 vs
  204.0, adelta 0.818 vs 0.835) while costing ~0.018 SR and more wrist
  failures — its 3.75B smoothness lead did not survive. On top of an
  action-rate penalty the energy term buys nothing measurable at 5B.
- **Against `sonic_v1_1`**: the acceleration gap ratio on the 124 board is
  now 1.30 (`base` 3.77 vs 2.89) against 1.62 for the 46.5B leader (4.67).
  Success rate remains SONIC's axis (1.0000 vs our 0.95-0.98 there, 0.9888
  vs 0.94-0.96 on 4,096); global error remains ours (59-93 vs 174-188).

## Preliminary rows (2026-08-29, mid-training, one seed)

Matched 3.75B checkpoints (`feetacc_weak` at 1.00B — its node runs at 39k fps
against the siblings' ~133k, healthy curve, no restarts), clean protocol,
scored with the 2026-08-29 smoothness metrics. PRELIMINARY: none of these is
the 5B row.

| arm | SR | L | G | acc | jerk | adelta |
|---|---:|---:|---:|---:|---:|---:|
| `base` | 0.9463 | 22.78 | 83.72 | 4.60 | 194.4 | 0.809 |
| `energy` | 0.9189 | 27.72 | 93.67 | 4.46 | **181.0** | **0.741** |
| `sigma` | 0.0151 | — | — | — | (35.5) | (0.063) |
| `feetacc_weak` @1.0B | 0.9014 | 25.99 | 132.25 | 5.13 | 224.1 | 0.882 |
| `ar0` | **0.9553** | 24.21 | 98.73 | 5.52 | 285.4 | 1.229 |

- **`sigma` is dead and the verdict needs no more budget**: episode length
  pinned at 15-22 from the first iteration through 3.85B. SONIC's tight
  exploration contract (init 0.05, clamped) does not train FROM SCRATCH under
  our optimizer stack — its sigma init belongs to a warm-started release
  policy. Its jerk 35.5 is a frozen robot, not a smooth one. The noise-aware
  hypothesis survives; testing it needs a sigma ANNEAL or a floor applied
  late, not a hard clamp at birth.
- **`base` vs `ar0` (the penalty decomposition, preliminary): from scratch,
  the -0.03 penalty currently costs 0.009 SR and buys -32% jerk, -34% adelta,
  AND -1.4 mm L AND -15 mm G.** Opposite sign on L/G from the 2026-08-26
  FINETUNE sweep, where every weight cost accuracy. If it holds at 5B:
  the penalty hurts as a late finetune but helps trained-in.
- `energy` is the smoothest trainable arm (jerk 181.0) and pays for it in
  SR and L — directionally the same trade as the action-rate weight axis.
- `base` already beats `merged64_pen_ramp_5b`'s FINAL row on jerk (194 vs
  216), adelta (0.81 vs 0.91) and G (83.7 vs 89.3) at 1.25B fewer frames.

## Added arm (2026-08-29): `hist`

Base + ten-step proprio history on all five actor observation terms (the
`smooth_block` set). On the `diffntp_pair` base, history bought -9.6%
acceleration at matched 2B before evaporating with budget — but that base had
no action-rate penalty, and the 08-29 harvest showed `diffntp_pair_hist` as
the smoothest penalty-free arm in action space (adelta 1.29). This cell asks
whether history STACKS with the trained-in penalty. Its eval repeats the
history overrides (strict restore on the widened actor input); `eval.sh`
handles that per-arm.

`hist` runs at **16,384 envs, not 20,480**: the first submission (5597331,
commit `753a692`) died in 7 minutes on a Warp CUDA OOM at graph launch — the
ten-step histories widen the observation buffers past what 20,480 envs leave
on the H200, the same signature as the 24,576-env attempt (job 5580202).
Resubmitted at 16,384 as jobs 5597345 -> 5597346
(`--set vars.train_num_envs=16384`, minibatch 294,912). Reading caveat: the
`hist` vs `base` comparison therefore carries batch shape as a second
variable, like the 2026-08-18 env20k precedent.

## Added arms (2026-08-29, round 2): `ema` and `lcp`

The two objective-level levers from the smoothness plan that needed code,
both landed with inert defaults so no other arm changes:

- **`ema`** — trained-in first-order low-pass on the joint-target action term
  (`EMAJointPositionAction`, new in `imitation/mdp/actions.py`; the G1 action
  config now uses the EMA cfg class with `ema_alpha=1.0` = identity).
  The arm sets `env.actions.joint_pos.ema_alpha=0.65`, ~8.4 Hz at 50 Hz —
  deliberately milder than the locomotion literature's 3-5 Hz so fast clips
  survive. The policy trains THROUGH the filter; the filter state resets to
  the default pose on episode reset. Eval must repeat the alpha (the filter
  lives in the env, not the checkpoint); `eval.sh` handles it per-arm.
- **`lcp`** — Lipschitz-constrained policy (arXiv:2410.11825): RLOpt PPO now
  carries `ppo.lcp_coeff` (default 0.0, inert) adding
  `coeff * mean(||d loc / d obs||^2)` to the actor loss via the
  vector-Jacobian ones-trick on the policy MEAN, computed on the first
  `ppo.lcp_batch_cap=65536` rows of each mini-batch (create_graph memory
  does not fit the full 368k-row production mini-batch). Smooths the policy
  FUNCTION: no phase lag, no bandwidth cap. The arm sets
  `agent.ppo.lcp_coeff=0.005` — a first screen point, not a tuned value.
  The checkpoint is a plain policy; eval needs no override. This is an
  in-repo RLOpt submodule edit (ppo.py + the IPMD hook now deferring to it).

## Added arm (2026-08-29, round 3): `lstm`

Recurrent actor (`agent.ppo.rnn_hidden_size=256`), user-requested from prior
experience that recurrent policies train more stably. New config-driven path
in RLOpt PPO (inherited by IPMD):

- Actor becomes normalize/concat -> `LSTMModule` -> the tuned MLP head. The
  critic stays feed-forward, so GAE is untouched.
- The env gains `InitTracker` and the LSTM's `TensorDictPrimer`
  (`make_tensordict_primer()`), so the collector carries recurrent states
  across steps and zero-fills them on reset.
- PPO mini-batches switch from shuffled flat rows to time-contiguous
  `[env, T]` sequences sampled along the env dimension. TorchRL loss modules
  run their forward under `set_recurrent_mode(True)`, so this gives true
  BPTT over the 24-step rollout window with `is_init` resetting hidden state
  at episode boundaries.
- The pre-existing `PPORecurrent` class was NOT used: it pairs the LSTM with
  the flat shuffled mini-batches, which recurrent mode would treat as one
  arbitrary sequence.

Runs at 16,384 envs (BPTT activation memory plus the `hist` OOM caution), so
it carries batch shape as a second variable against `base`, like `hist`.

QUALIFICATION DEBT: `evaluate_checkpoint`'s manual step loop has not been
verified to carry recurrent states correctly (the primer keys must survive
`step_mdp` and reset). Qualify that before citing any `lstm` row.

## Sigma arm terminated (2026-08-29)

`sigma` FAILED at 3.85B on the trainer's "interrupted before reaching its
frame budget" guard (job 5597007, 9:43 elapsed — not walltime; consistent
with a non-finite abort under the clamped exploration contract). The arm was
already dead on merit (episode length pinned 15-22 from iteration one). Its
dependent segment 5597008 was cancelled to free the H200. Verdict stands:
SONIC's sigma contract does not train from scratch on our optimizer stack;
test the noise hypothesis with an anneal or late floor instead.

## Status

- 2026-08-29: round-2/3 arms SUBMITTED from commits `78ed349` (top) +
  `8378891` (RLOpt): `ema` 5597424 -> 5597425, `lcp` 5597426 -> 5597427,
  `lstm` 5597429 -> 5597430 (16,384 envs via `--set`). Both new code paths
  smoke-qualified together (EMA+LCP one run; LSTM one run, recurrent-state
  keys confirmed in the policy in_keys). `base` and `energy` finished 5B in
  segment 1.
- 2026-08-29 04:10: SUBMITTED, seed 0, all five arms, from commit `8e11d2e`
  (drift=true: the RLOpt merged-head working-tree change rides along, as in
  every pareto-stack round-5/6 submission). Jobs (lowlevel1 -> lowlevel2):
  `base` 5597003 -> 5597004, `energy` 5597005 -> 5597006, `sigma`
  5597007 -> 5597008, `feetacc_weak` 5597009 -> 5597010, `ar0`
  5597011 -> 5597012. Nothing measured.
- 2026-08-28: campaign written; all five arms plan clean offline
  (`--skip-preflight`).
