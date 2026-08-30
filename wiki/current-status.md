# Project Live Status

Experiment navigation now starts at `experiments/README.md` and its exhaustive `SCRIPT_INVENTORY.md`. One-shot launchers named in the chronology below may have been pruned on 2026-07-23; `experiments/PRUNED_SCRIPTS.md` is the authoritative deletion and recovery catalog. A historical path is not a live submission instruction.

Last verified: 2026-08-27. New latent/interface work uses
`Isaac-Imitation-G1-v2`; frozen v0/v1 aliases remain only for reproducing the
historical runs recorded below. The current contract for what goes in the paper
is [final-paper-experiment-design.md](final-paper-experiment-design.md).
Sections are newest first; a section is a record of its date, not a standing
instruction, unless it says so.

This is the living memory for the active research project. Read it first when
returning to the project or starting a new agent session. It answers **where we
are now**. The detailed protocol and experiment history remain in the linked
phase documents.

Human-facing launcher navigation now starts at
[`experiments/README.md`](../experiments/README.md). It marks the
2026-08-18 BONES-SEED latent-design ablation as the primary prepared campaign
and reserves `experiments/paper/` for the eventual stable release entrypoint.
Dated campaign folders index canonical scripts rather than copying their
implementation.

Update this page after a meaningful code decision, qualification result,
cluster submission, job failure, or paper result. Verify changing external
state such as Slurm jobs before treating a status below as current. Keep old
chronology in the phase-specific pages instead of allowing this page to grow
without bound.

## The command-interface star is rebased on `diffntp_chunk` (2026-08-30)

User decision: the 72-arm interface ablation is re-run against a new hub,
because `diffntp_chunk_h1_ee_wide` beats the v1 hub `ctrl` on all three
canonical metrics — 0.9163 / 24.07 / 84.69 against 0.9023 / 24.49 / 212.3 on
`bones_testbed4096_v1` clean, a 2.5x difference in global error. Every v1
main effect is conditional on a hub that is no longer competitive, and three
of them are already known to be conditional: the objective ordering inverts
(the v1 star's worst converged objective is this hub's ancestor), the hold
effect was measured on one objective only, and the width and quantization
verdicts came from a loss with no marginal regularizer.

Design page: [latent-learning-star-v2.md](latent-learning-star-v2.md). Axes:
A what the encoder predicts (predictive generative / predictive mean /
other predictive targets / reconstruction / posterior-in-RL; contrastive
excluded), B continuous versus discrete, C encoder input and window size,
D publication cadence, E frozen versus finetuned versus learned-in-RL,
F loss detail and seed. Census: 62 rows, 16 already MEASURED at the hub's
cell inside `2026-08-22-pareto-stack`, 1 trained-unscored, 45 to train —
about 118 ICE segments. The page carries a four-step cut line down to 33
new arms that keeps every family claim.

Scored while assembling the table (4,096 clean, one seed, first rows for
both): `diffntp_chunktok` 0.9189 / 23.55 / 88.92 and `diffntp_chunkra`
0.9194 / 24.74 / 106.45. `chunktok` beats the hub on success rate and local
error and is 5% worse globally, all inside the band.

Four gates before any v2 arm is submitted: pin
`env.rewards.feet_acc.weight=-2.5e-7` against the 2026-08-28 in-place
correction; decide the rot6d convention (it distorts every re-anchored and
EMA-token target, though not the hub's); score `diffntp_merged` so
`merged64`'s lead becomes attributable; smoke one pretrain per quantizer
family. Two v2 arms are already written into the pareto-stack
`campaign.yaml` and NOT submitted: `diffntp_chunk_nosig_h1_ee_wide` and
`diffntp_chunk_h1_ee_wide_s1`.

## Linear-closure program: the affine spectral EBM is the only arm (2026-08-29)

Decision: the linear-closure thread keeps exactly one design. The chord
(Jensen-gap) penalty, the residual split `g(z) = Gz + h(z)`, and the
mixture-target interior coverage from
[linear-closure-problem-statement.md](linear-closure-problem-statement.md)
are dropped, not deferred.

The arm: `z = E(s_{t+1:t+H})`, `M(z) = M0 + sum_k z_k M_k`,
`f(s_t, y, z) = psi(y)^T M(z) phi(s_t)`,
`p(y | s, z) ∝ q0(y | s) exp(f)`. The networks `psi`, `phi`, and `E` stay
nonlinear; only the path from `z` to the energy is affine. This makes the
score exactly linear along chords, so
`p(y | s, z_alpha) ∝ p1^alpha * p2^(1-alpha)`: a mixed latent means the
product (intersection) of the endpoint skills, which settles Q2 of the
problem statement. `A(s, z)` is convex in `z`, so `log p` is concave along
every chord.

Consequences recorded with the decision:

- The `z` domain must be convex, so the arm removes the final LayerNorm
  (or replaces it with a ball constraint). Straight-line interpolation is
  the only combination with exact semantics; spherical interpolation has no
  guarantee under this model.
- In the repo's bilinear head this is `g(z) = Gz + c` with
  `F(s, y) = vec(psi(y) phi(s)^T)`; the tanh bound on `g` must go.
- `q0(y | s)` absorbs skill-independent predictability. Because the visible
  boundary pair explains 98.0-99.7% of the mid frames (Tier A below), a
  single endpoint target gives the `M_k` a weak gradient; the target choice
  waits on the Tier B suffix verdict.
- The guarantee covers model validity only. Encoder realizability needs
  joint training with `E`, and executability of the frozen tracker on mixed
  latents stays open: the alpha-sweep probe (survival, jerk, lawfulness)
  remains the falsifier.

## Endpoint-collapse probe: does the diffntp_chunk z summarize the window or just the boundary? (2026-08-29)

Question: the diffntp_chunk code z_t is trained to predict the endpoint
`s[t+10]` and the next chunk `s[t+11..t+20]`. Because the macro state is close
to a boundary-plus-velocity sufficient statistic, z could collapse to a
function of the last visible frames `s[t+8], s[t+9]` and the paper's
"z summarizes the intermediate states" claim would be hollow. Campaign:
`experiments/campaigns/2026-08-29-endpoint-collapse-probe/`.

**Tier A (offline probes, round-4 `diffntp_chunk_h1_ee_wide_seed0` encoder,
2,500 reference windows, one seed, correlational): partial collapse.**

- Frame-sufficiency is non-discriminative on-manifold: the last visible frame
  alone reaches MLP R2 0.914 vs 0.914 for the whole visible window, but
  smooth motion makes every subset predictive of every other.
- The visible boundary pair (s_t, s[t+9]) linearly explains 98.0-99.7% of
  every mid frame; z recovers 28-61% of the small residual (peak at slots
  6-8). z carries real mid-window information, but the pool it draws from is
  tiny.
- Sensitivity per unit input RMSE: last-frame replacement moves z 4.3x more
  than mid-frame replacement (232 vs 54). On-manifold mid replacement still
  moves z by 1.14x its own norm, so z is not mid-blind.
- Integrated gradients (batch-permuted baseline): 53% of attribution on the
  last visible slot, 66% on the last two, 26% spread over slots 1-7.

Tooling that made this measurable: window builder validated against the
compiled sampler primitives to 1e-6
(`imitation_experiments.capacity.probe_skill_window_usage`); the trainer now
logs `train/jepa_endpoint_loss[_eval]` and `train/jepa_ntp_loss[_eval]`
separately (they were merged into `jepa_objective` before).

**Tier B (causal falsifier, running):** new `encoder_window_mode=suffixN`
(encoder sees only the last N slots of the intermediate window; `suffix9` ==
production `intermediate`). Arms suffix1/suffix9 launched locally 2026-08-29
evening at the full 50k-update round-4 recipe; suffix2/suffix5 follow. Flat
eval losses in N kill the summarization claim; improvement with N measures
it.

**Found while verifying frame conventions (not fixed, needs a decision):**
the data plane's `quat_to_rot6d_flat` emits the INTERLEAVED 6-D layout
(`R[..., :2].reshape` = r00,r01,r10,r11,r20,r21) but RLOpt's
`_rot6d_to_matrix` parses two concatenated columns. Consequences: (a)
`_reanchor_heading_frames` extracts a wrong (sign-flipped, roll/pitch-mixed)
yaw from data-plane frames, so every `jepa_ntp` target that routes through
re-anchoring (chunk_anchor='next' arms, context>0 arms, and the z2 EMA-target
token of every jepa arm) is a deterministically distorted view, and (b)
`analyze_reference_latent_scale.py` builds windows with a full-rotation
anchor and rows-concatenated rot6d, both off the sampler convention, so its
absolute geometry claims are suspect. The production diffntp_chunk loss path
(context 0, chunk_anchor='executed') is NOT affected.

## Global-error decomposition: the G gap to SONIC is root translation drift, and it accumulates (2026-08-29)

Question: local tracking is nearly tied with `sonic_v1_1` (L 26.9 vs 26.7 mm
on the 4,096 clean board) but our global error is 2.2x better (86 vs 188 mm).
What produces the gap?

**The decomposition is exact and closes arithmetically.** MPJPE-L is
root-relative and MPJPE-G is world-frame (`contracts/tracking_metrics.py`),
so per link the world error is the root translation error plus the
root-relative residual. If the two are uncorrelated, G^2 = drift^2 + L^2.
Measured (success-only micro, 4,096 clean, one seed each):

| arm | G | root drift | L | sqrt(drift^2+L^2) | residual |
|---|---:|---:|---:|---:|---:|
| `sonic_v1_1` | 187.7 | **186.0** | 26.7 | 187.9 | -0.3 |
| `base` @5B | 86.0 | **78.9** | 26.9 | 83.4 | +2.6 |
| `ar0` @5B | 93.1 | 86.2 | 24.2 | 89.5 | +3.6 |
| `energy` @5B | 86.8 | 78.0 | 27.0 | 82.5 | +4.2 |

The identity closes to 0.2% for SONIC and 3-5% for ours (small positive
correlation between drift direction and pose error). **The entire G gap is
where the pelvis is, not body shape**: root drift 186 vs 79 mm (2.4x), while
local pose and heading are near-tied (heading 0.074 vs 0.053 rad; its
lever-arm contribution, ~0.35 m x theta ~ 18-26 mm, lives INSIDE L on both
sides).

**SONIC's drift accumulates; ours is bounded.** Final-step vs episode-mean
root drift: `sonic_v1_1` 1.77 (172.9 -> 305.8 mm on the 124 board; 1.73 on
4,096), `base` 1.47 (61.0 -> 89.7 mm). A ratio near 2 is what an uncorrected
random walk gives over a 10 s episode; SONIC slides while tracking locally.

**Config-level mechanism (verified in the released `sonic_v1_1` config, not
inferred):**

1. **SONIC's actor is blind to drift.** `motion_anchor_pos_b` /
   `motion_anchor_ori_b` — the world-frame anchor error — appear ONLY in its
   CRITIC observation group (asymmetric). The policy cannot observe planar
   drift, so it cannot servo it; its anchor rewards (w=0.5, std=0.3) can
   only shape open-loop reproduction. Our v2 policy observes
   `expert_anchor_pos_b`/`ori_b` directly (with U(-0.25,0.25) noise).
2. **SONIC's anchor termination is height-based** (`exceeded_anchor_height`,
   threshold 0.15, its own training config), so planar drift is never
   terminal during its training. Ours trains under a planar `anchor_pos`
   termination (0.25 m) plus explicit world-frame rewards
   (`motion_global_anchor_pos` + `_wide`).

The attribution to actor-visible drift feedback is a HYPOTHESIS until
ablated. The falsifying experiment is one arm: train `base` with
`expert_anchor_pos_b`/`ori_b` removed from the policy group (critic keeps
them, SONIC-style). Prediction: G rises toward the SONIC drift regime while
L stays put. Not yet run.

Infrastructure: `evaluate_checkpoint` per-env rows now carry
`root_pos_xyz_error_final_m` (drift at the last active step), giving the
accumulation ratio; SONIC rows already had the analogous
`anchor_pos_err_final_m` from the command term. Same-quantity check done:
both drift columns are the world-frame pelvis/anchor position error norm.

## Smooth-ablation campaign CLOSED: all code arms scored (2026-08-30)

`lcp` and `lstm` finished 5B and were scored on both boards at final +
penultimate checkpoints; with `ema`'s gate that completes every surviving
arm (only `feetacc_weak`'s queued resume outstanding). Close-out table and
verdicts in `experiments/campaigns/2026-08-28-smooth-ablation-5b/README.md`.
Headlines: `base@4750049280` 0.9570/23.68/86.68 is the program's best
all-round row; `lcp` (0.005, untuned) and `lstm` converge to base's
smoothness without beating it (`lstm` functional — the recurrent path is
fixed and trains competitively — with its best relative axis on robust G);
only `ema` beats the penalty's smoothness (jerk 166.8; cap124 acc 3.49 vs
`sonic_v1_1` 2.89). Live decisions: `alpha=0.8` arm and `alpha=0.65` at
10B decide the ema promotion.

## EMA gate scored: smoothness records, promotion deferred to the alpha sweep (2026-08-30)

`ema` finished 5B and was scored at both final and penultimate checkpoints
(both boards, `ema_alpha=0.65` repeated at eval). Verdict, one seed: the
trained-in filter holds the program's smoothness records (jerk 166.8 on the
4,096 clean board, 132.5 / acc 3.49 on the 124 board — `sonic_v1_1`
acceleration ratio 1.21, the closest yet) but is NOT promoted as-is: it
costs 0.029 clean SR against `base@4750049280` (the base row of record; its
5.0B save is an arm-specific anomaly) and DOUBLES robust MPJPE-G (272.8 vs
140.4) — the 8.4 Hz bandwidth cap visibly costs perturbation recovery.
`ema` was still improving into its cap (4.75B -> 5.0B better on every
axis), so budget is a live confounder on the SR gap. Follow-ups in order:
`alpha=0.8`, `alpha=0.65` at 10B, alpha-anneal. Full table in
`experiments/campaigns/2026-08-28-smooth-ablation-5b/README.md`. Also
2026-08-30: `ar0` and `energy` discarded by user directive (their closing
contribution: at matched 4.75B the from-scratch action-rate penalty costs
~0 SR and buys -34% jerk — the penalty is free trained-in; and the cap-hit
checkpoint anomaly is base-specific, with checkpoint-to-checkpoint variance
near convergence exceeding eval-repeat noise, so rows of record must name
their checkpoint).

## EMA action filter is the smoothness program's lead lever (2026-08-29)

User decision after the matched-1.0B mid-training rows
(`bones_testbed4096_v1` clean, one seed): `ema` (trained-in
`EMAJointPositionAction` low-pass, `ema_alpha=0.65` ~ 8.4 Hz at 50 Hz) read
0.9055 SR / 26.96 L / 128.29 G / acc 4.40 / jerk 163.5 / adelta 0.812 —
the lowest jerk measured on this board, below `base`'s 5B final (204.0),
at NO success-rate cost against the matched-frames MLP reference
(`feetacc_weak` 0.9014). The `sonic_v1_1` acceleration ratio falls to 1.28
at one-fifth of the budget. `lcp` also delivers (adelta 0.789, jerk 189.4);
`lstm` trails on every axis at 1.0B. The 5B final row is the gate; if the
profile holds, `ema` is the promotion candidate and the follow-up axes are
the alpha sweep and ema x lcp. Eval discipline: the filter lives in the env
action term, so every eval of an ema checkpoint must repeat the alpha.

## Smooth-ablation 5B finals: base / energy / ar0 scored on both boards (2026-08-29)

First three arms of `2026-08-28-smooth-ablation-5b` finished 5B and scored;
full tables with the `sonic_v1_1` comparison rows (standing user directive:
every table carries the same-board sonic_v1_1 row) are in the campaign
README. Headlines, one seed:

- Penalty decomposition at 5B (`base` vs `ar0`, matched): `action_rate_l2`
  -0.03 costs 0.014 SR and +2.7 mm L, buys -30% jerk / -34% adelta / -7 mm G.
- UNRESOLVED: `base`'s local error regressed 22.78 -> 26.93 mm over its last
  1.25B while `ar0` stayed flat — a late-training regression specific to the
  penalty arm; milestone curve is the diagnostic before promotion.
- `energy` converges to `base`'s smoothness (jerk 202.5 vs 204.0) at an SR
  cost: on top of an action-rate penalty the energy term buys nothing at 5B.
- Gap to `sonic_v1_1` acceleration on the 124 board: `base` 3.77 vs 2.89
  (ratio 1.30) against the 46.5B leader's 4.67 (1.62). SR remains SONIC's
  axis, global error remains ours.
- Eval-infra postmortem: three background eval waiters deadlocked matching
  their own `pgrep -f evaluate_checkpoint` guard strings (~70 min idle GPU).
  Process-name grep guards are retired from local pipelines; sequential
  chains + skip-existing-rows idempotence instead.

## Smoothness is now measured directly: jerk + action metrics in board rows (2026-08-29)

`evaluate_checkpoint.py` now writes four smoothness measures per environment,
so they join the success-only row every paper metric reports on:

- `action_delta_l2` / `action_l2` (existed board-wide only since 08-17;
  never per-environment)
- `action_jerk_l2` — NEW, `||a_t - 2a_{t-1} + a_{t-2}||`, buzz not authority
- `body_acc_mps2` / `body_jerk_mps3` — NEW, the robot's OWN acceleration
  magnitude and its finite difference over the 14 tracked links.
  Reference-FREE: unlike `tracking_acceleration_distance_mps2` (an error
  against the reference, where copying a jerky clip scores 0), these measure
  what "smooth" physically means.

`summarize_paper_boards` renders them as `jerk=... m/s3 adelta=...`; rows
scored before 2026-08-29 simply omit them. `evaluate_sonic_release` does NOT
carry them yet, so `sonic_v1_1` has no jerk/adelta row.

Best checkpoints rescored on `bones_testbed4096_v1` clean (success-only
micro; `logs/smoothness_rescore/`; SR/L/acc reproduce the existing rows
within the known eval noise):

| arm | SR | acc | **jerk m/s³** | **adelta** |
|---|---:|---:|---:|---:|
| `ar01` @48.5B | 0.9722 | 4.48 | **188.0** | 0.918 |
| `merged64_pen_ramp_5b` @5B | 0.9541 | 4.85 | 215.8 | **0.911** |
| `ar003` @48.5B | 0.9761 | 4.94 | 234.5 | 1.154 |
| `ln_hold1_sonicreset` @46.5B | 0.9775 | 5.57 | 306.1 | 1.557 |
| `diffntp_chunk_h1_ee_wide` @2B | 0.9163 | 6.99 | 412.0 | 1.720 |

The reference-free jerk separates mature arms 63% (188 -> 306) where the
tracking-error acc separates them 24% (4.48 -> 5.57): jerk is the
discriminative smoothness axis. The headline tracker is 63% jerkier than
`ar01` and 42% jerkier than `merged64_pen_ramp_5b`.

A free harvest of the board-wide `action_delta_l2` already present in every
`evaluate_checkpoint` JSON
(`logs/smoothness_rescore/action_metrics_harvest_2026-08-29.txt`, 62 arms,
all-transition means so survival-mix-weighted) adds three structural facts:

- **Every arm trained with an action-rate penalty occupies the top of the
  board** (0.915-1.171); the smoothest penalty-free arm is 1.29.
- **Hold-10 and explicit-command arms are the jitteriest families**
  (fsq64/jepa hold-10 2.0-2.8, `ctrl_*` explicit 2.35-2.80) despite hold-10
  winning MPJPE-G — the smoothness cost of the wide-command interface was
  invisible until now.
- The leader's action jitter GROWS with depth: 1.59 @20B -> 1.62 @30B ->
  1.71 @49B.
- `diffntp_pair_hist` (ten-step history) is the smoothest penalty-free arm
  (1.29) — the history smoothness signal was real in action space too.

Broken artifact flagged while harvesting:
`logs/bottleneck_10b_4096/fsq64_hold10_sonicreset_f30000021504.json` records
0/4096 with `steps_run=6` — an eval-configuration failure, not a score. Do
not cite it.

## Smoothness ablation SUBMITTED: 5 arms x 5B, env 20,480 (2026-08-29)

`experiments/campaigns/2026-08-28-smooth-ablation-5b/`, W&B group
`smooth-ablation-5b`, commit `8e11d2e`, jobs 5597003-5597012 (two chained
segments per arm). Base: `diffntp_chunk_h1_ee_wide`'s frozen encoder, tracker
from scratch to 5B in the `merged64_pen_ramp_5b` regime but 256-D and 20,480
envs. One variable per arm: `base` (control), `energy` (SONIC's
`energy_consumption` at -1.0e-4), `sigma` (SONIC's exploration-noise contract,
init 0.05 clamped [0.001, 0.5]), `feetacc_weak` (the old wrong -2.5e-7,
isolating the parity fix), `ar0` (`action_rate_l2` 0.0 — the penalty
decomposition `merged64_pen_ramp_5b` asked for). Nothing measured.

## Reward parity fixes: `feet_acc` 10x correction + `energy_consumption` term (2026-08-28)

Two changes to `G1SonicRewardsCfg` (inherited by the v2 tuned recipe), both
approved by the user and both verified against the released SONIC configs at
`/mnt/hsstorage/fwu91/sonic_v1_1/config.yaml` and
`/mnt/hsstorage/fwu91/sonic_release/config.yaml`:

1. **`feet_acc` weight corrected -2.5e-7 -> -2.5e-6.** Both released SONIC
   configs carry -2.5e-06. The 2026-08-04 "align the reward definition with
   SONIC" commit (`5a36dfc`) read the exponent backwards and WEAKENED the
   ankle-acceleration penalty 10x while claiming to strengthen parity. Their
   `joint_acc_l2` is a re-export of isaaclab's (`gear_sonic/envs/manager_env/
   mdp/__init__.py` does `from isaaclab.envs.mdp import *`), so the function
   was always identical; only the weight was wrong. Every arm trained between
   2026-08-04 and 2026-08-28 used the weak value. This is an IN-PLACE v2
   recipe change: rows trained after 2026-08-28 differ from earlier rows on
   this weight.
2. **`energy_consumption` added, OFF by default.** SONIC's whole-body
   mechanical-power penalty (`|applied_torque * joint_vel|` summed over
   joints, ported verbatim from `gear_sonic`'s `rewards.py`), weighted
   -1.0e-4 in both released configs, was missing from our reward set
   entirely. It now exists in `imitation/mdp/rewards.py` with config default
   weight 0.0 so existing rows stay comparable; enable per campaign with
   `env.rewards.energy_consumption.weight=-1.0e-4`. Newton fills
   `applied_torque` for implicit actuators via a post-actuator callback, so
   the term works on both backends. Qualified with a 1-iteration smoke on
   `Isaac-Imitation-G1-v2` (term registers and runs, exit 0).

Also corrected while in the file: our `anti_shake_ang_vel` body set uses
`torso_link` where SONIC uses `head_link` — deliberate (the bundled 29-DoF
asset has no head body), already documented, unchanged.

## qvel in the encoder input costs smoothness: three matched pairs agree (2026-08-28)

Every arm whose DiffSR macro state is the FULL-BODY frame
(`env.expert_macro_state_terms=[expert_motion,...]`, 670-wide, qpos+qvel) was
scored against its `root_qpos` (380-wide) partner on `bones_testbed4096_v1`.
Added today: robust rows for `qvel_h1_ee_wide` and `trip_qvel_h1_ee_wide`, the
frame-matched 10.0B rows for the full-body leader, and its 8.5B robust row
(the 2026-08-19 attempt died on a CUDA OOM caused by a concurrent job and left
a log with no row).

Clean rows, one seed:

| pair | encoder input | frames | SR | L | G | vel | **acc** |
|---|---|---:|---:|---:|---:|---:|---:|
| `cont_det_ln_hold1` | root_qpos 380 | 10.0B | **0.9226** | 24.19 | **119.42** | 0.223 | **5.78** |
| `cont_det_ln_hold1_fullbody_env20k` | full_body 670 | 10.0B | 0.9099 | 24.45 | 126.34 | 0.238 | 6.33 |
| `jepa_h1_ee_wide` | root_qpos 380 | 2.0B | 0.9060 | 26.99 | **103.61** | 0.252 | **7.02** |
| `qvel_h1_ee_wide` | full_body 670 | 2.0B | 0.9070 | 26.02 | 121.54 | 0.263 | 7.68 |
| `trip_h1_ee_wide` | root_qpos 380 | 2.0B | **0.9004** | 27.35 | 127.28 | 0.260 | **6.95** |
| `trip_qvel_h1_ee_wide` | full_body 670 | 2.0B | 0.7891 | 33.75 | 128.87 | 0.274 | 7.14 |

Robust (`no_push`) rows agree: 6.57 vs 7.14 (leader, 10.0B), 7.89 vs 8.43
(`jepa`/`qvel`), 7.89 vs 8.10 (`trip`/`trip_qvel`).

**qvel in the encoder input makes the motion LESS smooth, not more.** All three
matched pairs move the same way on acceleration (+9.5%, +9.4%, +2.7% clean) and
on velocity error, and two of three are clearly worse on global error. This is
the opposite of the intuition that giving the encoder velocity information
should help the tracker produce velocity-consistent motion, and it holds at
both 2B and 10B and under both randomization profiles.

It also reproduces the 2026-08-19 star's `in_fullbody670` result (-0.0271 SR)
on a third axis: that arm reads acc 8.54 clean / 9.05 robust, the worst of the
qvel group.

`trip_qvel_h1_ee_wide` is a separate and larger failure — SR 0.7891 against its
triplet partner's 0.9004, with 714 wrist terminations. Triplet context and qvel
do not combine.

QUALIFICATION: one seed per arm. The leader pair now shares a frame count
(10.0B) after today's rescore; the previous comparison used its 8.5B row
against a 10B control and understated nothing — acc moved 6.36 to 6.33 between
those two checkpoints, so the arm was already flat.

Artifacts: `logs/qvel_fullbody_eval/`, `logs/pareto_stack_eval/*qvel*`.

## `merged64_pen_ramp_5b` scored at 5B; `diffntp_pair_hist` robust rows added (2026-08-28)

The 5B merged-head run at 64-D finished its single segment (job 5594864,
COMPLETED, 10:43:26) at 5,000,134,656 frames, so its 0 -> 1B failure-share
ramp completed and 4B ran at the landed ratio. Scored locally on
`bones_testbed4096_v1`:

| row | SR | MPJPE-L | MPJPE-G | vel | acc |
|---|---:|---:|---:|---:|---:|
| `merged64_pen_ramp_5b` clean @5B | 0.9543 | 23.67 mm | 89.33 mm | 0.211 m/s | **4.84 m/s²** |
| `merged64_pen_ramp_5b` robust @5B | 0.9458 | 26.46 mm | 155.73 mm | 0.255 m/s | 5.53 m/s² |
| `ln_hold1_sonicreset` clean @46.5B | 0.9773 | 22.42 mm | 103.31 mm | 0.210 m/s | 5.57 m/s² |
| `ln_hold1_sonicreset` robust @46.5B | 0.9688 | 24.38 mm | 182.25 mm | 0.252 m/s | 6.41 m/s² |

At a ninth of the headline row's frames this arm is 0.023 SR behind on clean
while landing 13.5% lower global error and 13% lower acceleration; the robust
pair reads the same way. 4.84 m/s² is the lowest acceleration this program has
measured on the 4,096-clip board. For scale, the public `sonic_v1_1` 42M
checkpoint reads 3.45 m/s² all-clip on that board (3.34 matched-3,932); the
16M `sonic_release` checkpoint has no acceleration row at all, since the
metric landed on 2026-08-26 and every `sonic_release` artifact predates it.
Do not attribute either figure to "the released checkpoint".

It is NOT an attribution: three fields separate it from its own 2B parent
`diffntp_merged64_h1_ee_wide` (0.9207 / 24.54 / 91.12 / acc 6.90) — frames,
`action_rate_l2` 0 -> -0.03, and the reset schedule. The 20-point milestone
curve flattens on local error and acceleration by about 3.0B and on global
error by about 4.0B, so the last 1-2B bought very little and a longer chain is
not the obvious next move. The decomposition worth running is the
action-rate-only arm at the parent's schedule.

Separately, the `diffntp_pair_hist` robust rows were added and agree with the
REFUTED verdict: 0.8738 / 29.15 / 157.19 / acc 7.12 at 2.0B falling to
0.8628 / 29.63 / 162.97 / acc 7.82 at 4.0B. On the clean board the ten-step
history had bought -9.6% acceleration at 2.0B (6.42 against the parent's 7.10)
while costing 0.027 SR; that smoothness edge is gone by 4.0B and absent under
`no_push` at both depths. `experiments/campaigns/2026-08-27-diffntp-history/`
now carries an `eval.sh` that reads the five `history_length=10` overrides back
out of its `campaign.yaml`, because the widened actor input is part of the
strict policy restore.

All rows are one seed. Artifacts: `logs/pareto_stack_eval/merged64_pen_ramp_5b_*`
and `logs/diffntp_history_eval/`.

## Additive chunk+EMA cell submitted: `diffntp_chunktok_h1_ee_wide` (2026-08-28)

Round 6 of `2026-08-22-pareto-stack`, jobs 5594207-09, seed 0, 2B screen.
Loss = endpoint + `diff_chunk` + `1.0 * ||P(z1) - z2_EMA||^2` + SIGReg — the
question is whether the EMA-trick latent-dynamics term stacks with chunk
generation on MPJPE-G or substitutes. New RLOpt knob `jepa_token_pred_coeff`
(default 0 reproduces every existing arm; positive requires a diffusion
head). Reads against `diffntp_chunk` (0.9163 / 24.07 / 84.69) and the hub.
Nothing measured. Same day: `diffntp_merged64_h1_ee_wide` (jobs 5594211-13)
— the merged head at 64-D (z 64 / command 66), the width axis on the merged
objective, one variable against `diffntp_merged`.

## Merged-head cell submitted: `diffntp_merged_h1_ee_wide` (2026-08-27)

Round 5 of `2026-08-22-pareto-stack`, jobs 5594018-20, seed 0, 2B screen.
One diffusion head denoises `s[t+H..t+2H]` (418-d) in place of
`diffntp_chunk`'s endpoint + next-chunk pair; the separate endpoint term is
dropped (`jepa_endpoint_coeff=0`). New RLOpt knobs `jepa_ntp_chunk_span`
{next, boundary_next} and `jepa_endpoint_coeff` in
`RLOpt/rlopt/agent/hl_skill_diffsr.py`; `boundary_next` requires the
executed anchor; the endpoint head stays in the checkpoint, gradient-free.
Reads against the round-4 table (`diffntp_chunk` 0.9163 / 24.07 / 84.69
clean). Nothing measured.

## `diffntp_token_h1_ee_wide` re-score and showcase clips on ICE (2026-08-27)

`experiments/campaigns/2026-08-27-diffntp-token-showcase/`. Jobs 5593843
(`clean`), 5593844 (`robust`), 5593845 (`video`), chained `afterany` so they
run one at a time. Nothing measured yet.

The arm is round 4 of `2026-08-22-pareto-stack`. It ran 2B frames in ONE
segment, so `model_step_2000289792.pt` is both its last and its final
checkpoint; there is no longer chain behind it.

**A board mix-up worth fixing.** The `diffntp_*` rows in
[results-interface-ablations.md](results-interface-ablations.md) §5.6 come from
`bones_milestone_testbed256_v1` (256 clips), not from the canonical
`bones_testbed4096_v1`. For this arm the two disagree by more than a rounding
step: 0.9258 / 24.11 / 84.20 on 256 clips against 0.9121 / 24.44 / 86.29 on
4,096. The campaign README quotes the 4,096-clip row.

**Robust row measured**: 0.9050 / 26.96 / 135.88 on 4,096 clips, one seed.
Domain randomization costs this arm 0.007 SR, 2.5 mm local and 50 mm global
against its clean row.

**First RTX render submitted to the cluster, and it took two fixes.**
`docker/cluster/run_singularity.sh` sent `scripts/viz/*.py` to the bare
`/isaac-sim/python.sh` branch, which carries Kit but no torch; the rendering
entrypoints now select the evaluator interpreter, the same class as
`scripts/rlopt/eval*.py`. That alone was not enough: the interpreter only
exports the CU130 site-packages path, and something in the process has to put
it on `sys.path`. `eval_checkpoint_tree.py` calls `configure_cu130_bridge`;
`render_paper_policy_video.py` did not, so `AppLauncher.__init__` still died on
`No module named 'torch'` (ICE job 5593845) and the renderer now makes the same
call before constructing the launcher.

**That failure exited 0 and Slurm recorded COMPLETED.** The evaluator branch of
`run_singularity.sh` carries no workload success marker -- by design, it is the
PhysX trainer's contract -- so for an evaluation or render job the written
output file is the only trustworthy success signal. A green `status` table is
not one.

The standing preference is still to render locally; this ran on ICE by request.

## All jobs cancelled on both hosts (2026-08-27)

The user stopped every job on ICE and on the workstation to start fresh.
`scancel -u $USER` removed 25 ICE jobs: 4 running and 21 pending on
`(Dependency)`. `squeue` is now empty. The workstation had 11
`evaluate_checkpoint` processes; all 11 exited on `SIGTERM`.

**No local result was lost.** Each of the 11 processes had zero accumulated CPU
time since launch, one thread, no child process, and a `futex_do_wait` wait
channel. The GPU read 0% utilization and 555 MiB of 97,887 MiB, which no
4,096-environment Isaac evaluation can produce. They were hung, not slow. Two
had been hung for more than 2 days.

**Depth reached by each cancelled chain.** The last checkpoint under each output
root, in environment frames. A chain can resume from it, because the checkpoint
carries `cumulative_env_frames`.

| output root | arm | last checkpoint | declared cap |
| --- | --- | ---: | ---: |
| `/data/diffntp_50b` | `diffntp_chunk_50b` | 2.50B | 50B |
| `/data/diffntp_50b` | `diffntp_pair_50b` | 7.00B | 50B |
| `/data/leader64_gate` | `leader64_h1_nophase` | 0.75B | 2B |
| `/data/smooth_finetune` | `ar01scratch` | 9.00B | 10B |

The other three `smooth-finetune` arms (`ar01`, `ar003`, `ar01shake4`) had
already reached their 48.5B cap and are `COMPLETED`; the cancel did not touch
them. `ln_hold1_sonicreset`'s 50B chain had already stopped at 49.00B before
this cancel and was not in the queue.

**Three campaigns were failing before the cancel, and the cause is unread.**
`sacct` over the 24 hours to the cancel shows `diffntp-history` arms 5592715
and 5592716 both `FAILED` after about 10.5 hours at 4.00B each,
`diffntp_pair_50b` losing five consecutive segments to `FAILED`, and
`emastack-20b` losing four. Read those job logs before resubmitting any of the
three.

**Before relaunching into any of these output trees.** Each root still holds its
`wandb_run_id` file, which pins the chain to its W&B run. A fresh run that
reuses the tree resumes that run. Delete the file to move to a new id. Never
delete the run inside W&B: the service refuses an id that was ever deleted, and
the refusal kills the job.

## Budget-axis curves for the whole interface study (2026-08-27)

Every ablation axis the study measures now has a metric-against-frames curve,
not only an endpoint: success rate, success-only micro MPJPE-L, and
success-only MPJPE-G at eight budgets from 250M to 2B frames, all on
`bones_milestone_testbed256_v1`.

**576 points, 72 arms, five campaigns.** `interface-design-study` (29 arms) and
`interface-combos` (5) were already scored. The 2026-08-26 sweep added
`pareto-stack` (27 arms, 216 cells) and `posterior-interface` (9 arms, 72
cells) locally in 3h45m. `bn_vq_ema` is the one collapsed arm: 0 successes at
every budget, so it carries a success rate and no MPJPE.

`python -m imitation_experiments.reporting.curve_table <eval dirs> --row
milestone --csv <out>` reduces the scored cells to one tidy table. The setup and
every arm's single changed field are in
[interface-ablation-study.md](interface-ablation-study.md); the publication-facing
reading of the numbers is [results-interface-ablations.md](results-interface-ablations.md).
The interactive board is published at
https://claude.ai/code/artifact/4b4a4e86-0d51-4571-be54-685ddafba264 . It keeps a
collapsed arm with an empty success-only MPJPE rather than dropping the row,
because a dropped row makes a collapse look like missing data.

The curve board is 256 clips and the paper row is `bones_testbed4096_v1`. A
curve is internally consistent; anchor its endpoint against the 4096-clip clean
row instead of putting both scales on one axis.

**The posterior campaign gained a budget axis.** Only its 2B checkpoint had
been mirrored; ICE still held all eight per arm.
`2026-08-20-posterior-interface/eval.sh` now takes `ROWS=milestone` and has a
`mirror.sh`.

**Scoring a tree in one process.** `scripts/rlopt/eval_checkpoint_tree.py`
keeps a single Isaac Sim start and swaps the policy weights across a tree's
milestones, planning the cells with
`imitation_experiments.evaluation.score_tree`. An arm fixes the interface, so
only the weights change. On `hold1_seed1`: 30.9 s/cell against 47.4 s/cell,
and on the cluster it also collapses eight container starts into one. The rows
agree within Isaac's nondeterminism -- mean success-rate difference +0.0010,
largest 0.0156; MPJPE-L +0.06% mean; MPJPE-G -1.30% mean, scattered in sign.

Two traps this cost:

- The rollout steps inside `torch.inference_mode()`, so the expert data plane's
  reference-row buffers are allocated as inference tensors during the first
  cell. A later cell's reset writes them with `index_copy_`, which torch
  refuses outside inference mode. Later cells now reset inside inference mode;
  the first cell keeps the single-cell path untouched.
- `run_singularity.sh` sent every non-`train*.py` entrypoint to Kit's Python,
  which has no torch, and appended `--assert-kitless`, which only the train
  entrypoints define. `scripts/rlopt/eval*.py` now resolves its backend the
  same way training does and does not get the training-only flag.

## Hold 5 fills the middle of the hold axis (2026-08-27)

The study had hold 10 and hold 1 and nothing between, at either code width.
`use_hold5` (256-D) and `ix_fsq64_hold5` (64-D SONIC FSQ) joined
`2026-08-19-interface-design-study` as one-field changes from their hubs and
went to ICE at the full 2B budget: jobs 5592720-22 and 5592723-25, pretrain
then two chained lowlevel segments. The star is now 18 core + 11 supporting +
2 interaction probes.

Both chains finished the full 2B in one lowlevel segment (the second segment
exited in six minutes on the met budget), and `2026-08-27-hold5-curve-eval`
scored both budget axes on ICE, one job per arm, about 16 minutes each for
eight cells: jobs 5593234 and 5593236. The 29 star arms stay on the local
mirror -- their ICE trees were cleaned under the 300 GB quota.

At 2B, monotone in MPJPE-G at both code widths, and the SR cost of hold 1 only
appears at 64-D:

| width | hold 10 | hold 5 | hold 1 |
|---|---|---|---|
| 256-D | 0.9102 / 23.44 / 199.87 | 0.9102 / 24.79 / 146.92 | 0.9180 / 25.76 / 140.94 |
| 64-D FSQ | 0.9023 / 28.86 / 177.70 | 0.9023 / 28.52 / 141.36 | 0.8789 / 30.65 / 134.10 |

SR / MPJPE-L / MPJPE-G, `bones_milestone_testbed256_v1`, one seed. A 256-clip
board and one seed: read the hold-5 rows as filling in the shape of the axis,
not as separating arms whose ends already sit this close.

**What the ICE evaluation path cost.** Four failed submissions, each a wrong
assumption about the cluster, all fixed:

- `score_tree` read only the mirror's `f<frames>` layout. On ICE the trainer
  layout stands, so the frame count comes from the file name -- valid only for
  a tree that ran as ONE segment. A tree whose checkpoints span several run
  directories is now refused, not guessed.
- The CU130 runtime Python has no Kit extension cache: `AppLauncher` dies on a
  missing `EXP_PATH`. The evaluators construct `AppLauncher` at import, so they
  need Kit AND torch, which only Kit's Python with the CU130 site-packages
  gives -- the PhysX training branch's interpreter.
- That interpreter needs `configure_cu130_bridge` before any torch import, the
  same call `train_physx.py` makes.
- Two eval jobs started eleven minutes apart both hit the shared Isaac Sim
  cache; the second crashed inside Kit startup. Re-run alone it passed. Serialize
  eval jobs, or give each its own `CLUSTER_ISAAC_SIM_CACHE_DIR`, until this is
  understood.

## Stroboscopic motion-sequence figures (2026-08-26)

`--shot sequence` on `scripts/viz/render_paper_policy_video.py` produces the
graphics-paper motion composite: one image, one scene, the robot at several
poses along the path it walked. It runs the clip twice — once unrendered to
learn the travel path and auto-frame a locked camera to it, then again to
capture poses — cuts each pose out by differencing against a background plate
rendered with the robot hidden, and layers them. `--sequence_poses` (default 6),
`--sequence_alpha_min`, `--sequence_threshold`.

Findings worth keeping, each of which cost a render cycle to establish:

- **A chase camera cannot make this figure.** It holds the robot in the middle
  of every frame, so the poses stack in one place. The locked camera is the
  whole feature; the palette needed no change.
- **Order poses by distance travelled, not by time.** Even time spacing bunches
  poses wherever the robot slows down, which is exactly the interesting part.
- **Layer along the shadow direction, not chronologically.** Shadows all run
  downwind of the key light, so the pose a shadow falls ON must be drawn AFTER
  the pose it comes FROM or that shadow eats its feet. Sorting by projection
  onto the shadow direction is correct whichever way the robot walks.
- **Re-apply the style after locking the camera.** The three-point rig is placed
  relative to the camera azimuth, and the sequence shot moves the camera after
  the style was applied.
- **Difference keying alone fails on this robot.** Its white shell sits within a
  few levels of the backdrop, so differencing finds the outline and the dark
  joints but drops the middle of a limb. Filling external contours fixes it, and
  over-filling is safe because frame and plate are identical outside the robot.
- **Pitch must exceed half the vertical field of view** or the horizon enters
  the frame; the wide lens a sequence needs makes this bite where `hero_low`
  never did. Derived from the lens at runtime, not hard-coded.
- **Do not median several renders to denoise a still frame.** This renderer
  converges tile by tile, so a per-pixel median across successive renders bakes
  tile seams in.

### The horizontal banding was the floor slab (2026-08-26)

Figures carried visible horizontal lines. **Cause: the studio floor slab's
geometry.** Not the compositing, not the locked camera, not temporal AA. Both
of its dimensions matter and both are now pinned with a comment in
`_spawn_studio_rig` — do not "tidy" either:

- **Extent.** The original 20 km slab spans a depth range wide enough that
  depth quantisation prints as thin dark lines lying on the floor, crowding
  toward the near field. Scored 24–130 on the seam metric for a grazing locked
  camera against 0.67 for a chase shot; 400 m scores 0.33. Keep it past the
  largest fog end (110 m) and no bigger.
- **Thickness.** A thicker slab bands at *shallow* angles instead: 0.5 m took
  `hero_low` from 0.4 to 100. 2 cm is clean everywhere.

Now 9 of 10 style×shot combinations score under 1.0, the tenth
(`studio_dark`/`ground_high`, 3.67) being a smooth lighting falloff rather than
a seam.

**How the diagnosis went wrong first, which is the transferable part.** Three
successive "fixes" were built on a metric that could not see the defect: it
averaged columns over a strip the robot walks through, so it scored the robot,
not the banding. That produced a confident and wrong story (temporal
accumulation on a static camera), a settle-and-retry loop that gamed the blind
spot by optimising a region while the rest of the frame got worse, and a
row-debanding post-process that treated the symptom. Two rules earned the hard
way: **a metric that does not cover the whole artifact is worse than no metric**,
because it licenses fixes that make things worse, and **amplify and look at the
image before theorising** — one ×12 view of the residual showed lines lying on
the ground plane and pointed straight at the floor geometry.

Also measured along the way, all now moot but worth not repeating:
`RenderCfg.samples_per_pixel` is a **no-op** under RT2 (that is the RT1
`/rtx/directLighting/...` path; RT2 uses `rtx.rtpt.*`), and
`ManagerBasedRLEnv.render(recompute=True)` **skips** `sim.render()` rather than
forcing it.

## Paper figure renders: `studio_light` + `hero_low` is the chosen look (2026-08-26)

**Decision: paper figures and clips use `--style studio_light --shot hero_low`.**
The user picked it off the contact sheet on 2026-08-26 and both are now the
script defaults; the contact sheet marks that cell RECOMMENDED. It is a
seamless near-white cyclorama with the robot at eye level on a 35 lens: no
horizon to crop around, and it sits on a white page without a visible frame
edge. Use it unless a figure needs something else on purpose.

`scripts/viz/render_paper_policy_video.py` no longer hard-codes one look.
`--style {light,dark,studio_light,studio_dark,photoreal}` picks the palette and
`--shot {ground_high,hero_low,orbit_hero}` picks the framing; `--preview`
renders the whole style-by-shot matrix from one frozen pose in a single Isaac
launch and writes a labelled contact sheet, so a look is chosen before any clip
is rendered. `--stills_every` / `--stills_steps` write lossless PNG, which
figures should use instead of extracting frames from the CRF-23 MP4.
`--style light --shot ground_high` reproduces what the script rendered before
the presets existed, so earlier renders stay reproducible.

Four defects were found and fixed while calibrating, and each is a trap for the
next person touching this file:

- `safe_set_attribute_on_usd_prim(..., camel_case=True)` lowercases the whole
  name, so `inputs:diffuseColor` was written as `inputs:diffusecolor` and every
  floor-color change was silently a no-op. Pass `camel_case=False`.
- `/OmniverseKit_Persp` is authored in the stage's **session layer**. Lens
  writes through the default edit target lose to it and read back unchanged;
  they need `Usd.EditContext(stage, stage.GetSessionLayer())`.
- The rendering `.kit` leaves `/rtx/sceneDb/ambientLightIntensity` at 1.0, a
  flat term that puts a hard floor under every pixel. A dark studio is
  unreachable while it stands, however far the lights are dimmed, so it is now
  a per-style value.
- `init_state.rot` is **(x, y, z, w)** in Isaac Lab 3.0 and a `DistantLight`
  emits along -Z. The old key-light quaternion was a hand-tuned magic number;
  lights are now specified as elevation/azimuth in degrees.

The horizon seam that a low camera exposes is closed with RTX distance fog
(`/rtx/fog/*`) tinted to the dome color, not with backdrop geometry. Fog
`start` must sit well beyond the camera-to-subject distance or it hazes the
robot itself. Light budgets were calibrated against a measured reference
(original rig: floor at about sRGB 200 near, 180 far, no clipped pixels)
rather than by eye; the light presets sit on the tonemap shoulder, where large
intensity cuts move the output very little.

## Paper boards frozen at two, smoothness arms not promoted (2026-08-26)

**Paper-facing decision (user).** The paper reports two evaluation
populations and no others: `bones_testbed4096_v1`, the deciding board, and
`sonic_capability124_v1`, the SONIC-facing calibration board. Success-only
errors reduce to the clips every row of a given table completes; that
intersection is a property of the TABLE and must be frozen and named per
table. The smoothness table's is
`experiments/campaigns/2026-08-17-paper-metric-canon/matched3932_smoothness_2026-08-26.json`,
3,932 ranks, SHA-256 `c5b8c3c8…`.

**`ln_hold1_sonicreset` stays the headline tracker row.** On the deciding
board, matched 3,932, one seed, one evaluation:

| row | SR (of 4,096) | MPJPE-L | MPJPE-G | vel m/s | acc m/s^2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| public `sonic_v1_1` | 0.9888 | 26.25 mm | 177.41 mm | 0.193 | 3.34 |
| ours @46.5B | 0.9773 | 21.95 mm | 92.31 mm | 0.205 | 5.45 |

SONIC leads success rate and smoothness; we lead local and global accuracy by
a wide margin. Cite both directions together.

**The smoothness campaign is a NULL on the deciding board.** Every
`action_rate_l2` arm trades SR and accuracy for its acceleration gain and none
reaches SONIC's 3.34 m/s^2 — see the campaign README for the table and the
`ar003` non-finite incident, which `IPMD._abort_on_nonfinite` caught cleanly
(its first real save).

**Two caveats attached to the headline row.** 46.5B is a MID-CHAIN checkpoint
of the 50B run, chosen because it was the newest file when scoring started;
the chain will stop near 49.0B (no insurance segment, by user decision), so
the final paper row should be re-scored from that last checkpoint. And a
6th row added to the table changes the matched population, hence every
success-only number in it.

## Smoothness arms submitted (2026-08-26)

The acceleration-distance gap (ours 4.67 vs SONIC 2.89 m/s^2 on the new
common eval subset) traces to `env.rewards.action_rate_l2.weight=0.0` in the
tuned recipe; SONIC v1.1 trains with -0.1. `2026-08-26-smooth-finetune`
submits three +2B finetune arms off the pinned 46.5B leader checkpoint
(`ar01` -0.1, `ar003` -0.03, `ar01shake4` -0.1 plus 4x anti-shake) and one
10B from-scratch arm (`ar01scratch`), jobs 5592178-86, W&B group
`smooth-finetune`. Gate: acceleration distance falls on
`sonic_capability124_v1` while SR/MPJPE-L hold on `bones_testbed4096_v1`.
Nothing measured.

## 50B leader chain: mid-chain progress row at 46.5B (2026-08-26)

`ln_hold1_sonicreset`'s 50B chain is **not finished**: it stands at 46.67B of
50B. Segment `lowlevel14` (job 5590009) hit its 15:59 walltime at 46.60B and
`lowlevel15` (5590010) resumed from it. Node `atl1-1-03-017-16-0` carries five
jobs, so the chain now logs 43-50k fps instead of its usual ~129k; at that rate
segment 15 reaches only ~49.1B, and it was the last declared segment. Segment
`lowlevel16` was declared and planned as second insurance, but the user held
the submission on 2026-08-26, so nothing is queued behind 5590010 yet.

The newest checkpoint, `model_step_46500151296.pt`, was scored on the new
common eval subset `sonic_capability124_v1`, clean protocol, seed 0, one
evaluation. This is a PROGRESS read, not the 50B promotion row.

| row | clips | SR | MPJPE-L | MPJPE-G |
| --- | ---: | ---: | ---: | ---: |
| public `sonic_v1_1` | 124 / 124 | 1.0000 | 23.79 mm | 173.92 mm |
| ours @30B | 123 / 124 | 0.9919 | 19.44 mm | 108.58 mm |
| ours @46.5B | 124 / 124 | 1.0000 | 19.44 mm | 122.08 mm |

Success-only errors above use each row's own successful clips. On the 123 clips
all three complete: SONIC 23.43 / 148.37 mm, ours @30B 19.44 / 108.58 mm, ours
@46.5B **18.99 / 97.56 mm**.

Read the matched pair carefully. 30B to 46.5B moves MPJPE-L by 2.3% and
MPJPE-G by 10.1%, both inside the unresolved band for one seed and one
evaluation; the only firm change is that rank 6364 `kneeling_loop_003_A244`,
the 30B row's single `anchor_pos` failure, now completes. The subset was
selected by reading public SONIC's own results, so it favors that anchor;
never call it held out or unbiased.

Velocity/acceleration distance (added 2026-08-26): `evaluate_sonic_release`
now accumulates `tracking_velocity_distance_mps` and
`tracking_acceleration_distance_mps2` per step over the same 14 links, the
`evaluate_checkpoint` definition. On the 124 clips (all rows complete all
clips): SONIC 0.165 m/s / 2.89 m/s^2, ours @46.5B 0.175 m/s / 4.67 m/s^2 —
velocity distance near-equal (6%, unresolved), acceleration distance clearly
higher for ours (rollout less smooth at the link level), while we lead on both
MPJPE columns. The SONIC re-run doubles as the board's first repeat: 23.89 vs
23.79 mm L, so repeat noise is ~0.1 mm.

Artifacts: `logs/sonic_capability124_v1/ln_hold1_sonicreset_46b5_clean.json`
and its `_encoder_binding.json` (18 of 18 encoder tensors identical to the
selected encoder). The launcher is
`experiments/campaigns/2026-08-25-sonic-paper-proxy/score_arms_capability124.sh`,
which verifies the frozen rank-list SHA-256 before it evaluates anything.

**Control plane: a stage may now depend on a live Slurm job.** `depends_on:
"job:<id>"` resolves to `afterany:<id>` at submit time and survives
`--only-stage` selection, so an insurance segment appended to a chain that is
already running queues behind it instead of starting beside it and training the
same output tree twice. `sonic-reset-50b` gained `lowlevel16` with
`depends_on: ${vars.chain_after}`, default `lowlevel15`. The stage is planned,
not submitted.

## The released SONIC checkpoint is the paper's 16M model (2026-08-25)

Goal was an evaluation population on which the public SONIC checkpoint
reproduces a SONIC paper number, so the paper's baseline rows can be cited as a
faithful proxy. The population turned out not to be the lever.

**The public `sonic_release/last.pt` is not the paper's 42M flagship — but
`sonic_v1_1/last.pt` is.** Both are on HF; their `config.yaml` files settle it
(`sonic_release` `g1_dyn` `[2048,2048,1024,1024,512,512]`, `sonic_v1_1` and
`low_latency` `[4096,4096,2048,2048,1024,1024,512,512]` = Table S1). An earlier
line here said the 42M weights were unreleased; that was wrong.

**`sonic_release/last.pt` is not the paper's 42M flagship.** Its action decoder is
`[2048, 2048, 1024, 1024, 512, 512]` (release `config.yaml:152`) where Table S1
specifies `[4096, 4096, 2048, 2048, 1024, 1024, 512, 512]` — 14.4M parameters
on the tracking path against 41.6M. It is the paper's 16M rung, at 100k
iterations against the paper's 50k. So its comparable rows are Table 4(a),
test-repetition 99.6% / 25.5 mm, **not** Table 4(c)'s 99.8% / 22.5 mm, the
Figure 2 MuJoCo 98.7% / 23.2 mm, or the 123-clip deployment 100% / 22.3 mm.

Measured on the new `sonic_proxy_testrep4096_v1` board, clean protocol:
`sonic_release` **0.9924 / 25.63 mm / 150.94 mm** against Table 4(a)'s
0.996 / 25.5 mm (0.13 mm), and `sonic_v1_1` **0.9932 / 24.35 mm / 206.00 mm**
against Table 4(c)'s 0.998 / 22.5 mm (1.85 mm). The harness reproduces the
paper's own rows at the sub-2 mm level for both public checkpoints.

v1.1 is **not** a strict improvement on the release: on the 4,057 clips both
complete it is 1.3 mm better on MPJPE-L (24.29 vs 25.60) and 56 mm WORSE on
MPJPE-G (206.20 vs 150.39), at level completion. The bigger decoder buys local
precision and spends global anchoring.

Also settled, both from SONIC's released source: their `mpjpe_l` is
`smpl_sim` `compute_metrics_lite`, root translation only, and their 14
`body_names` are ours verbatim — the metric was never the gap. A reference
time-base hypothesis (their stride converter at 120 to 50 fps) was raised and
dropped: the load-time resampler is a proper time-based lerp/slerp and the
production converter is not public.

Clip selection moves the number by 0.1-0.4 mm. Reaching 23.8 mm by pruning
requires deleting Dance, Advanced Locomotion, Sports, Stunts and Unusual
Locomotion — the ease-selection error of 2026-08-17, still not to be rebuilt.
SONIC's own deployability filter is the 40-keyword list we already apply.

**Our best on the same board.** `ln_hold1_sonicreset` @30B, clean protocol:
**0.9722 SR / 20.09 mm / 104.08 mm**. On the clips both complete the tracking
gap survives against both public checkpoints — 20.03 vs 23.93 mm against
`sonic_v1_1` (3,972 clips; MPJPE-G 103.84 vs 185.41) and 20.03 vs 25.32 mm
against `sonic_release` (3,968 clips) — so it is not a subset artifact. v1.1
completes 96 clips we fail; we complete 10 it fails. They trade — tracking precision to us, robustness to them — which is
the same shape as the 2026-08-07 reading, now at a 5.3 mm margin instead of
3 mm and with MPJPE-G moving to our side for the first time. One seed. The
50B chain (`sonic-reset-50b`, resubmitted 2026-08-24) has no local mirror yet.

**22.3 mm is not reachable honestly, twice over.** A 123-clip board built from
the ten motion families SONIC names as deployed scores `sonic_v1_1` at
36.76 mm, `sonic_release` at 42.72 and our 30B arm at 38.49 — selecting
deployable motions makes a board HARDER, since crawl, crouch, kneel and dance
are the worst families. And a search for any other matching subset found that
2.2% of random 123-clip draws and 13.1% of a 512-rule grid already hit 22.3 mm,
with the closest rule ("short and slow") dropping all but 12 of the 123
deployment-family clips. New guard `evaluation.subset_sensitivity` measures
this for any future "population P reproduces X" claim.

**New common eval subset, preliminary.** The frozen artifact ID
`sonic_capability124_v1` identifies this transparent, policy-conditioned
population; it is not a held-out board. The direct clean `sonic_v1_1` run completed
124/124 clips at **1.0000 SR / 23.79 mm MPJPE-L / 173.92 mm MPJPE-G**. The
30B `ln_hold1_sonicreset` seed-0 tracker completed 123/124 at **0.9919 /
19.44 / 108.58 mm**, one evaluation. On the 123 shared successes, SONIC is
23.43 / 148.37 mm and ours is 19.44 / 108.58 mm. The ordered-rank hash matches
and the 18-tensor encoder binding audit passes. Motion-name review of all 124
clips and full-horizon videos of the nine ambiguous names passed; the ranks did
not change. Repeats are still open. Do not use this result as an unbiased
tracker ranking.

New: `evaluation.sonic_paper_proxy`, board `sonic_proxy_testrep4096_v1` and
profiles `paper_sonic_proxy_testrep4096_v1` / `_robust_v1`, campaign
`experiments/campaigns/2026-08-25-sonic-paper-proxy/`. It is a calibration
board; our own arms still score on `bones_testbed4096_v1`.
[canonical-paper-metrics.md](canonical-paper-metrics.md) and
[sonic-release-checkpoint-tier2.md](sonic-release-checkpoint-tier2.md) are
updated.

## IPMD reward estimation (IRL): PARKED, future work needed (2026-08-25)

The reward-estimation campaign is complete and parked by user decision. Two
findings, both one seed on the 4,096 board:

1. **The stack is a null on tracking.** At a matched 10B budget every
   explicit row lands at 0.9558-0.9604 SR / 17.47-17.92 mm and every latent
   row at 0.9368-0.9377 SR / 23.61-24.23 mm — each inside evaluation noise
   of its no-reward-estimation counterpart (headline `cont_det_ln_hold1`
   0.9368 / 23.61). Reward estimation neither helps nor hurts the tracker,
   on either interface, at four regularizer settings. Expected, since PPO
   trains on the task reward; the useful part is that the machinery is
   proven safe to carry.
2. **The estimator itself is degenerate, and saturation is why.** The pure
   diff objective has no interior optimum: it is minimized by separating
   policy and expert as far as the output activation permits. Unregularized
   it pins at the tanh rails; with the R1 grad penalty it still rails
   (input gradients vanish exactly at the rails); with logit regularization
   it holds a soft ceiling (reward_diff -1.958) rather than a shaped
   reward. Goal-conditioned pairing makes separation *easier*, not harder,
   because the expert side has identically zero tracking error.
   `use_estimated_rewards_for_ppo=true` was therefore never worth trying.

Reviving this needs a different objective (a binding WGAN-GP-style
interpolate penalty or a density-ratio/logistic loss), harder negatives, and
a calibration target that reports saturation early instead of after 10B
frames — see the campaign README's "PARKED — future work needed" section.
Also open: `irl_pair_latent_ln_hold1` diverged to NaN inside one
25-iteration window at ~9.33B (scored at its 9.0B pre-divergence
checkpoint); unattributed, suspect pairing x harsh grad penalty x latent
route. Everything stays off by default, so no paper-path row is affected.
Campaign: `experiments/campaigns/2026-08-22-reward-estimation/`.

## IPMD reward estimation (IRL): normalized input fixed, 10B run submitted (2026-08-22)

The parked IPMD direct-reward-estimation stack is live again with a fixed
input. On the v2 surface the `reward_input` group is now `RewardInputUnitCfg`:
29 joint positions normalized per joint by the soft limits, the relative root
(anchor) position mapped from [-1 m, +1 m] (perfect tracking = 0.5), and the
relative root orientation as rot6d mapped from [-1, 1] — every feature in
[0, 1], shared helpers in `isaaclab_imitation.envs.reward_input_normalization`.
The composed env's data plane serves the expert side through the same helpers
and the same pinned joint order (lazily — the articulation does not exist at
cache-build time); the frozen v0/v1 surfaces and `ImitationRLEnvLegacy` keep
the raw 58-wide pairing. The old policy-side/expert-side joint-order and
width mismatch (robot pos+vel vs reference order) is gone with it.

Local qualification (workstation): 2-iteration smoke and a 30-iteration
1,024-env explicit run — estimator updates end-to-end, `reward_diff`
-0.92 -> -2.0, `exp_r` -> 1.0, no NaN. Note: the declared vanilla
coefficients (pure diff loss, zero regularizers) saturate the tanh output
early; a non-saturated estimate needs a logit-reg or grad-penalty follow-up
arm. Contract tests pass (16).

Campaign `2026-08-22-reward-estimation`: tuned explicit root_qpos tracker at
10B frames with `agent.reward_estimation=true`, PPO still on the task reward
(`use_estimated_rewards_for_ppo=false`). Submitted to ICE 2026-08-23 UTC as
jobs 5588194 -> 5588195 -> 5588196 (afterany chain); W&B project
`g1-reward-estimation`, group `irl-explicit-10b`, run id `irl-expl-s0`.

## Dyn arms scored at last; tracker pareto program opened (2026-08-22)

The three online-finetune (`dyn`) arms of `2026-08-15-latent-bottleneck-10b`
were scored on the 4,096 board for the first time — they had been excluded on
the mismatched-encoder premise retired 2026-08-20. Frame-matched at 10B, one
seed, `--skill_encoder_source checkpoint` (provenance 0.0 divergence in every
row): `cont_det_hold1_dyn` cuts MPJPE-G 17.9% (181.90 -> 149.41 mm) at level
SR/L; dyn on top of the reset ramp adds nothing (150.61 -> 148.56); dyn on
`fsq64_hold10` is directionally worse everywhere. Reading: dyn and
failure-driven resets are substitutes for global drift, and dyn helps only the
continuous hold-1 interface. Rows in `logs/bottleneck_10b_4096/`; the
scoreboard script now carries the dyn rows.

New plan of record: [Tracker Pareto Program](tracker-pareto-program.md) —
lever evidence for SR/L/G, the designed-but-not-submitted
`2026-08-22-pareto-stack` campaign, the graded feature menu (unscreened
reward terms, symmetry augmentation, L2T distillation, asymmetric critic),
and the push-termination attribution thread with its two protocol-neutral
measurements (pushed diagnostic row + steps-since-push histogram, now wired
into `evaluate_checkpoint` behind `--randomization all`).

## 20B SONIC-reset leaders scored; interface-combos screen COMPLETE; 30B chase running (2026-08-21)

Both `2026-08-18-sonic-reset-20b` chains completed at exactly 20,000,145,408
frames and were scored on the frozen 4,096 scoreboard:
**`ln_hold1_sonicreset` 0.9558 SR / 22.15 mm** (base `cont_det_ln_hold1` at
10B: 0.9368 / 22.86) and **`fsq64_hold10_sonicreset` 0.9468 / 24.57 mm** (base
0.9197 / 24.93). One seed; the gain confounds the reset sampler with the
second 10B of frames by design (continuation, no 20B random80 control).
`ee_body_pos` failures drop ~30-40% on both interfaces. Full table and
qualification in the campaign README.

Same day, user directive to chase SONIC's SR (0.9937): both arms continue
20B -> 30B under the identical regime, budget the only change —
`experiments/campaigns/2026-08-21-sonic-reset-30b/`, ln jobs 5587505-07,
fsq 5587509-11, appending to the same W&B `-r1` runs (run-id state files
hand-seeded on ICE before submission; the 20B chain predates the per-chain
id mechanism). If the SR curve is flat by ~25B, cancel the tail segments.

New campaign `experiments/campaigns/2026-08-21-interface-combos/` (W&B group
`interface-combos-2b`, confirmed): five combination/follow-up cells of the
interface design study at the byte-identical 2B screen, seed 0 —
`jepa_ebm_hold1_256d`, `jepa_ebm_hold1_fsq64`, `recon_endpoint`,
`recon_full_window`, `hold1_live_phase`. Supporting RLOpt changes:
`reconstruction_target` (`input_window`/`endpoint`/`full_window`) on the
skill-encoder config, and `command_phase_source=episode` +
`command_phase_period` for a live hold-1 phase clock (hl_skill sampler only).
Tests in `RLOpt/tests/test_hl_skill_recon_phase.py`. Note recorded in the
campaign README: `sigreg_ebm` already IS endpoint DiffSR + chunk NTP + SIGReg,
so the "endpoint plus JEPA auxiliary" cell needs no separate arm. The
chunk-triplet machinery (`--jepa_context_chunks 1`,
`--jepa_target_encoder_mode online`, copy-baseline gate metrics) is
implemented and tested but untrained; phase 0 (robot-pose recording) turned
out to already exist as the achieved-pose ring in `expert_data_plane.py`.

**Screen results, same day (all five at exactly 2B, one seed, clean board):**
no combination beats the star's `ctrl` on SR — no 10B promotion earned.
`recon_endpoint` 0.8992 / 24.16 mm / 373.1 mm partially repairs `obj_recon`'s
drift (was 446.3 mm) while `recon_full_window` repairs none of it (453.0 mm).
`hold1_live_phase` is a null against `use_hold1` on every axis — the phase
channel matters only at long holds. The JEPA x hold-1 cells post the two best
MPJPE-G in the program (142.0 mm continuous, 129.6 mm fsq64) without moving
SR. Full table + milestone curves in the campaign README and
`logs/interface_combos_eval/`.

## EC deployment tier: the async plan protocol, and where it stops (2026-08-20)

User directive: prepare for hardware deployment and build the asynchronous
path in Embodied-Control. EC is the last test before the robot, and its
sim2sim dynamics gap from the training environment is the POINT of the tier,
not a defect to close.

**What now exists (EC submodule, `native/ec_native` plus its Python wrappers).**

- **Multi-slot latent plans.** One planner reply may carry `plan_slots`
  consecutive latents, each held `hold_steps` control ticks; the controller
  walks the plan without calling the planner again, and the lead time counts
  down to PLAN exhaustion instead of hold expiry. Before this, one reply served
  exactly one hold and `lead_ticks < hold_steps` was enforced, so the leading
  Isaac row (hold 1, whole 30-slot plan) could not be expressed on the
  deployment runtime at all. New response tag 4, `plan_slots = 1` reproduces the
  old behaviour, `MAX_VALUES` raised 4,096 -> 16,384 to fit 30 x 256.
- **Plan-aware staleness.** A plan is not stale inside the horizon it was
  predicted for; the watchdog counts only the time past that horizon, so a
  planner that stops replying still trips `command_stale_ms`.
- **The hold-1 scheduling trap.** At hold 1 the countdown steps 1 -> 0, so an
  equality test on the lead never fires again once a plan is exhausted: the
  controller starved after its first plan and damped. Fixed to `<=` plus a
  guard against re-asking while a reply is already in hand. Regression test:
  `test_native_latent_plan_keeps_requesting_at_hold_one`.
- **Sensor noise where the hardware has it.** The DDS plant now serves SONIC's
  noise ON THE WIRE (joint pos/vel, gyro, plus an IMU tilt that stands in for
  SONIC's additive projected-gravity term), and the in-process MuJoCo backend
  perturbs the controller's view only. Metrics read the clean state in both.
- **Ground truth for the wire tier.** The plant logs its own true state,
  because the G1 wire protocol carries no root pose: MPJPE on that tier cannot
  come from the controller.
- **Rig plumbing:** optional DDS domain (0 stays the robot), a gantry that
  holds the reference start pose until the first controller command, an init
  ramp that can hold the current pose instead of the default stance, a
  skippable motion-switcher handshake (a simulated plant has no motion service,
  and its 5 s CheckMode timeout starved the controller), and latched
  state-fault reason bits.

**Verified.** With the same native controller, the same async plan protocol and
the real `ln_hold1_10b` head in its own process, an in-process paced MuJoCo
episode tracks a full 400-tick motion: 0 deadline misses, 0 damp ticks, 16 head
calls for 397 control ticks, planner round trip about 51 ms, control tick max
0.79 ms, wake late max 0.30 ms, base height 0.75-0.77 m. The async plan cadence
therefore holds at 50 Hz real time with a GPU planner off-process.

**Open, and isolated.** The same configuration through the DDS plant falls
within about 0.7 s of arming. The isolation chain: (1) the same bundle, motion
and MJCF track at 17.24 mm through the Python lockstep sidecar; (2) the same
planner drives 400 lockstep steps without falling through `ec lowlevel run`;
(3) the same controller and planner survive the full episode on in-process
paced MuJoCo; (4) only the DDS wire tier falls. The controller's view of the
joints was verified bit-exact against the plant's true state, so it is not a
state-ordering bug, and the fall reproduces with sensor noise off and with the
plant timestep matched to the bundle. Remaining suspects, in order: loop delay
added by the writer/plant round trip, the plant's servo law against the
backend's, and the command-direction joint mapping. This is the next thing to
settle before any hardware conversation.

**The board: 28 motions x 5 episodes, one seed, all preliminary.** Same bundle
(`cont_det_ln_hold1_seed0` at 10.0B), same head (`ln_hold1_10b` update 12k,
30-slot plan, hold 1), same reference start frame, SONIC observation noise on.
Scored from the clean simulator state; MPJPE-L is the episode mean.

| row | fall-free | MPJPE-L all | MPJPE-L upright |
|---|---:|---:|---:|
| EC oracle, reference in the loop (ceiling) | 0.643 | 26.27 mm | 16.45 mm |
| EC planner, lockstep (sync) | 0.764 | 45.26 mm | 39.56 mm |
| EC planner, paced native async (lead 5) | 0.664 | 127.97 mm | 43.13 mm |
| Isaac oracle (for scale) | 1.000 | 17.19 mm | 17.19 mm |
| Isaac planner, sync / async (for scale) | 1.000 | 38.41 / 38.78 mm | same |

MPJPE-L "all" is not comparable across the EC rows: the lockstep runner ends an
episode at its safety damp while the paced row keeps logging on the floor, so
the async row's 128 mm is dominated by post-fall frames. The upright column is
the comparable one, and there asynchrony costs about 3.6 mm (39.56 -> 43.13),
the same order as the Isaac pair (38.41 -> 38.78).

**The finding: the deployment bottleneck is the tracker, not the planner.** On
this board the tracker falls on 36% of episodes WITH THE REFERENCE IN THE LOOP,
against 0% in Isaac. Per motion, six clips fall in every row including the
oracle (`jump_around`, `rock_out`, `exercise_3`, both `big_heavy`/`big_light`
lifts, `feeding_birds`); the planner rows inherit those falls and add almost
nothing. Two clips are genuinely async-attributable: `cellphone_typing`
(oracle 20%, sync 20%, async 100%) and `triumph` (oracle 80%, sync 0%, async
100%). Three clips go the other way, where the planner survives what the oracle
does not (`talking_with_adult` 60% -> 0%, `injured_R_leg` 60% -> 0%,
`big_heavy_high_to_low` 80% -> 0% in the sync row).

Asynchrony itself is cheap at this scale: 44 late plans out of 2,830 head calls
across 140 paced episodes (1.6%), no faults, no damp ticks.

Artifacts: `logs/ec_dds_board/{async,sync}_ln_hold1_28x5/score.json` and
`logs/ec_dds_board/oracle_ln_hold1_28x5.json`.

**Also recorded (corrected 2026-08-21):** the bundle carries Isaac's SOFT joint
limits, uniformly about 0.27-0.29 rad tighter than the MJCF's hard ranges on all
29 joints, so the plant can legitimately reach a pose the controller's hardware
guard rejects. (An earlier note here read the MJCF's `actuatorfrcrange` -
motor torque in N.m - as a joint range and claimed a "-139 deg" limit; that was
a misread.) Fix: the guard should use the robot's hard limits plus its margin,
or the plant should clamp to the soft limits.

**SONIC parity check (2026-08-21).** Our tracker's PD constants ARE SONIC's
deployment constants, verified numerically against
`gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp`:
stiffness = armature x (2 pi 10 Hz)^2, damping = 2 x 2 x armature x (2 pi 10
Hz), action_scale = 0.25 x effort / stiffness, including the 2x doubling on the
ankles. All five motor classes match to four decimals, as do armature and
effort limits. SONIC's own sim path (`deploy.sh sim`) is the same architecture
as our plant: the deployment controller talking DDS on loopback to a MuJoCo
bridge. Three setup deltas remain, and they are alignment, not research:

1. Torque law form. They compute `tau = tau_ff + kp (q_des - q) + kd (dq_des -
   dq)` and clip the TOTAL to the motor effort limit, applied through motor
   actuators (`mj_data.ctrl`). Our plant uses position-servo actuators plus a
   separate `qfrc_applied` term, so the effort clamp covers only part of the
   torque and the total can exceed the motor limit.
2. Sim rate. Theirs is `sim_frequency = 200` (dt 0.005); our plant defaults to
   0.002.
3. Their deploy MJCF still carries the old hip-roll torque limit (88 N.m) while
   their policy constants use the newer 7520_22 (139 N.m), which is what our
   asset has.

They also document a sim-to-real gain delta of their own: waist pitch KD is
reduced by 10 on the real robot because it is over-damped in sim
(`gear_sonic/utils/mujoco_sim/configs.py:102-105`). Treat the residual gap as
physics, not as something to engineer away.

## Posterior interface submitted: learning the code through the policy (2026-08-20)

`experiments/campaigns/2026-08-20-posterior-interface/`, W&B group
`posterior-interface`. **9 arms training on ICE**, jobs `5584268`-`5584276`
plus `5584278`, every arm `drift=0`.

The counterpart to the 2026-08-19 star: there the skill encoder is pretrained
offline for 50,000 updates and frozen, here it is learned during RL through
`command_source=posterior`. A separate campaign on purpose — it differs from
that study's `ctrl` in the whole command-generation path, so `ctrl` is a
cross-campaign reference row rather than a one-field control.

A 3 x 3: the learning signal that shapes the code x the latent space it passes
through.

| | AE (`identity`, 256-D) | FSQ (64-D) | VQ (`vq_ema`, K=512) |
|---|---|---|---|
| reconstruction only | `post_recon_ae` | `post_recon_fsq` | `post_recon_vq` |
| policy gradient only | `post_pg_ae` | `post_pg_fsq` | `post_pg_vq` |
| both | `post_pgrecon_ae` | `post_pgrecon_fsq` | `post_pgrecon_vq` |

Hold 10, 2B frames, seed 0 — the star's own settings, so the route is the
variable. **No pretrain stage**, which is the property under test rather than a
caveat: a policy-gradient route does not get a 50,000-update head start, and
what that costs is the result.

`post_*_vq` is the direct test of whether `bn_vq_ema`'s 0-of-4096 collapse was
the quantizer or the offline path that hosted it: the posterior quantizer is a
different implementation, so the star's bottleneck ordering is a hypothesis
here, not a result.

### The input view is aligned with the control, and the knob was not obvious

These arms encode the **same 38-value `root_qpos` frame** `ctrl` does. The knob
is `env.command_interface.encoder.components`:

```
env.command_interface.encoder.components=[joint_qpos,root_pos,root_ori]
```

`EncoderViewCfg.components` defaults to the full-body trio — that is where the
reference joint velocity enters — and for a LATENT actor that view is the ONLY
source of expert terms in the policy group, because
`policy_command_terms() = actor terms + encoder-view terms` and a latent actor
contributes none. So that one field decides what the posterior can read.

The agent half cannot be set from the CLI at all: `sync_input_keys` re-assigns
`posterior_input_keys` during `__post_init__`, after Hydra applies overrides, so
an override is silently discarded and the run trains against the default view
while appearing to honour the flag. Hence the new
`rlopt_ipmd_posterior_root_qpos_cfg_entry_point`. A contract test pins the two
halves together; drift gives either `KeyError: 'expert_motion_qpos'` or a silent
fall back to the wider view the star measured as WORSE.

Ruled out empirically, recorded so nobody retries them:
`env.command_observation_terms` (reaches the env and keeps the term, but the
term has no producer — keeping BOTH `expert_motion` and `expert_motion_qpos`
still raises `KeyError` on the qpos one, which isolates this to publication
rather than pruning), and `reference.critic_components` (that is the critic's
view).

### Compile caches are now per job

`post_recon_vq` died six minutes in with
`InductorError: FileNotFoundError ... .tmp -> ....py` while its eight siblings
ran: nine concurrent jobs racing on the shared `~/.cache/torchinductor` rename.
`slurm.py` now exports `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` under
the job's own bootstrap root, which removes the shared name and is cleaned up
with that root. The resubmitted arm (`5584278`) trains with zero InductorError.
Regression test in `tests/test_cluster_run_id_block.py`.

## Co-trained vs frozen skill encoder, frame-matched at 6.0B (2026-08-20)

The first real measurement of SONIC's co-training choice in our recipe. It
needed no new training: two `*_dyn` arms of `2026-08-15-latent-bottleneck-10b`
already had 6.0B checkpoints, and both of their frozen controls have a 6.0B
checkpoint too — 6,000,082,944 against 6,000,476,160 frames, 0.007% apart.
Scored on the canonical testbed, clean and `no_push`, 8 runs, no failures.
Artifacts in `logs/dyn_vs_frozen_6b/`.

### The clean pair: FSQ 64x32, hold 10

`fsq64_hold10` against `fsq64_hold10_dyn` differ in `dyn_args` and nothing
else — neither sets `curriculum_args` — so this is a genuine one-variable
comparison.

| profile | encoder | SR | MPJPE-L | MPJPE-G |
|---|---|---:|---:|---:|
| clean | frozen | 0.8970 | 25.88 mm | 153.5 mm |
| clean | co-trained | 0.8962 | 25.84 mm | **143.1 mm** |
| `no_push` | frozen | 0.8843 | 28.24 mm | 249.7 mm |
| `no_push` | co-trained | 0.8823 | 28.50 mm | **233.1 mm** |

**Co-training changes essentially nothing on success rate or local error**
(-0.0007 and -0.0020 SR; -0.2% and +0.9% MPJPE-L) and gives a **small
consistent improvement in global error, -6.8% clean and -6.6% robust**. The
sign agrees across two independent randomization profiles, which is mild
corroboration, but ~7% is inside the ~15% band and this is one seed on one
interface at 6.0B.

Read against the 10M-era datum: `sonic_fsq_pg` versus `fsq_recon` suggested
policy gradient into the encoder "did not help" (21.73 against 24.63 ep_len).
At 6.0B on the deciding board the honest statement is narrower — it does not
help success or local error, and may modestly help global drift.

### The other pair is confounded — do not attribute it

`cont_det_hold1_resetramp_dyn` scored 0.8806 against its control's 0.9041
(-0.0234 SR, +6.4% MPJPE-L), but that difference **cannot be assigned to
co-training**, for two independent reasons:

1. The two arms differ in **two** fields, not one: the dyn arm sets both
   `curriculum_args` and `curriculum_hold_args` while its control sets only
   `curriculum_args`. `curriculum_hold_args` is the fix for the reset-schedule
   re-sweep, so one arm has it and the other does not.
2. This arm carries the documented contamination window: its 5.0B-7.0B history
   was trained under a re-swept reset schedule, and **the 6.0B checkpoint sits
   inside that window**.

Recorded so the number is not later mistaken for a co-training effect.

### What this changes about priorities

Co-training looks like a small effect on the axis that matters least to the
headline, not a structural difference. That lowers the value of the three
deferred `use_cotrain_*` arms relative to the **posterior** route
(`command_source=posterior`, `train_posterior_through_policy`, `recon_coeff`,
`kl_coeff`), which learns the code THROUGH the policy rather than fine-tuning a
pretrained one and remains genuinely untested since July.

### Method note: the co-trained arms were always scoreable

`IPMD.load_model` restores the fine-tuned encoder from
`hl_skill_command_sampler_state_dict` (`ipmd.py:3036-3046`), and
`agent.ipmd.hl_skill_checkpoint_path` only seeds the sampler at construction.
Measured: the live encoder is **0.0 max abs** from the checkpoint's embedded
encoder in every configuration tried, including one that passed a different
arm's pretrained file and one that forced `--skill_encoder_source pretrained`.

So the belief that a `*_dyn` arm "would score a mismatched pair", which kept
both arms off every board, was never measured and is wrong.
`evaluate_checkpoint.py` now takes `--skill_encoder_source
{auto,checkpoint,pretrained}` and records the resolved source plus
`live_vs_checkpoint_encoder_max_abs` in `summary.json` — provenance, not a
behaviour change, so the question is settled by a logged number next time.

## Interface design study: all 29 arms trained and scored (2026-08-20)

**Complete.** 29/29 ICE jobs COMPLETED, 29 clean rows + 29 robust rows +
232 milestone rows scored locally, zero evaluation failures. One seed per arm
at a matched 2B frames, scored on `paper_testbed4096_v1`.

| arm | axis | SR | dSR | MPJPE-L | MPJPE-G | `ee_body_pos` |
|---|---|---:|---:|---:|---:|---:|
| **`ctrl`** | control | 0.9023 | — | 24.49 mm | 212.3 mm | 313 |
| `bn_cont128` | bottleneck | 0.9050 | +0.0027 | 24.27 mm | 212.7 mm | 319 |
| `bn_cont64` | bottleneck | 0.9021 | -0.0002 | 24.15 mm | 216.9 mm | 300 |
| `bn_no_ln` | bottleneck | 0.8984 | -0.0039 | 23.17 mm | 211.2 mm | 333 |
| `bn_gaussian` | bottleneck | 0.8813 | -0.0210 | 28.49 mm | 214.0 mm | 407 |
| `bn_sonic_fsq64` | bottleneck | 0.8701 | -0.0322 | 29.07 mm | 173.2 mm | 456 |
| `bn_sonic_fsq32` | bottleneck | 0.8501 | -0.0522 | 32.21 mm | 205.0 mm | 528 |
| `bn_sonic_fsq64_l8` | bottleneck | 0.8352 | -0.0671 | 33.81 mm | 191.4 mm | 571 |
| `bn_sonic_fsq16` | bottleneck | 0.6716 | -0.2307 | 43.48 mm | 252.0 mm | 1247 |
| `bn_gumbel_multicat` | bottleneck | 0.6660 | -0.2363 | 45.21 mm | 322.0 mm | 1229 |
| `bn_categorical` | bottleneck | 0.5454 | -0.3569 | 55.35 mm | 640.8 mm | 1689 |
| `bn_gumbel` | bottleneck | 0.3457 | -0.5566 | 69.06 mm | 1299.2 mm | 1973 |
| `use_hold1` | code use | 0.8921 | -0.0103 | 26.44 mm | 150.4 mm | 352 |
| `use_phase_none` | code use | 0.3679 | -0.5344 | 66.48 mm | 1281.6 mm | 2089 |
| `in_window_full` | encoder input | 0.8989 | -0.0034 | 23.65 mm | 235.0 mm | 322 |
| `in_anchor_robot` | encoder input | 0.8914 | -0.0110 | 25.15 mm | 219.2 mm | 381 |
| `in_anchor_expert_heading` | encoder input | 0.8772 | -0.0251 | 32.57 mm | 410.9 mm | 374 |
| `in_fullbody670` | encoder input | 0.8752 | -0.0271 | 27.29 mm | 232.2 mm | 441 |
| `in_stride5` | encoder input | 0.6558 | -0.2466 | 44.22 mm | 406.4 mm | 1203 |
| `ix_fsq64_hold1` | interaction | 0.8496 | -0.0527 | 30.45 mm | 136.9 mm | 532 |
| `obj_recon` | objective | 0.8931 | -0.0093 | 27.50 mm | 446.3 mm | 384 |
| `obj_phi_bilinear` | objective | 0.8906 | -0.0117 | 26.71 mm | 200.1 mm | 380 |
| `obj_jepa_sigreg_ebm` | objective | 0.8875 | -0.0149 | 27.70 mm | 212.8 mm | 391 |
| `obj_state_occupancy` | objective | 0.8826 | -0.0198 | 25.78 mm | 183.8 mm | 393 |
| `obj_semimarkov` | objective | 0.8621 | -0.0403 | 28.12 mm | 184.5 mm | 482 |
| `obj_jepa_infonce` | objective | 0.8542 | -0.0481 | 30.33 mm | 211.0 mm | 466 |
| `obj_jepa_ntp` | objective | 0.8218 | -0.0806 | 36.77 mm | 177.8 mm | 629 |
| `obj_endpoint_delta` | objective | 0.8123 | -0.0901 | 39.52 mm | 238.8 mm | 675 |
| `bn_vq_ema` | bottleneck | **0.0000** | -0.9023 | n/a | n/a | 4078 |

Read every row as **directional**: one seed, and the campaign's own repeat
floor is not measured. Differences of a few thousandths of SR between the top
arms are ties.

### Paper framing (decided 2026-08-20)

**Main story: the 26 hold-10 arms, led by success-only MPJPE-L**, with success
rate alongside. On that set the two agree at Spearman **+0.957**, so leading on
MPJPE-L tells the same story as leading on success — which is what makes the
choice safe. The hold-10 set is also the internally consistent one: 27 of 29
active arms share the hold-10 control, so the main figure has one cadence and
one hub.

Ranked by MPJPE-L, four arms sit nominally ahead of `ctrl` — `bn_no_ln`
(-5.4%), `in_window_full` (-3.4%), `bn_cont64` (-1.4%), `bn_cont128` (-0.9%) —
and all four are inside the ~15% band. The claim is therefore "**the control is
at the top of a five-way tie**", not "nothing beats the control".

**Cadence moves to the ablation** (width x hold 2x2). Report it honestly: hold
10 and hold 1 tie on success and local error, and hold 1 is decisively better
on global error at both widths (-29% at 256-D, -21% at 64-D). "Hold 10 is
better" is not supported.

**MPJPE-G remains a column in every table**, per the frozen
`canonical-paper-metrics.md` rule that none of the three may be published
alone. Leading the narrative on MPJPE-L is a framing choice and is compatible
with that rule; dropping G would not be, and would hide the drift failure mode
this very campaign exposed (`obj_recon` +12.3% on L against +110% on G). Within
the main set L and G still disagree (+0.430), and that disagreement is
ablation and discussion material.

**Consistency trap for the write-up:** the planner section's best row
(38.41 mm) runs on a hold-1 tracker while Section 1 now leads on hold 10. Say
so explicitly — cadence is reported separately and the deployed pair takes
hold 1 for its global-drift advantage. Stated it is a finding; unstated it is
something a reviewer will catch.

### What the study found

**The control configuration is the design.** No arm beats `ctrl` on success
rate by more than noise. The three that come closest are its own near-ties:
`bn_cont128` (+0.0027), `bn_cont64` (-0.0002) and `bn_no_ln` (-0.0039, and
5.4% BETTER on MPJPE-L). Continuous code width barely matters between 64 and
256 — so the width is free to choose on other grounds, such as what a planner
can emit.

**Discrete costs about 0.03 SR and 19% MPJPE-L at this budget.** The best
quantized arm, `bn_sonic_fsq64` — SONIC's own 64x32 token space — reaches
0.8701 / 29.07 mm against the continuous 0.9021-0.9050 / ~24.2 mm. Within FSQ
the bit budget orders cleanly at fixed levels (320 bits 0.8701 > 160 bits
0.8501 > 80 bits 0.6716), and levels matter separately (64x8 = 192 bits scores
0.8352, below 64x32 = 320 bits).

**Learned codebooks are worse than the fixed FSQ lattice at equal bits.**
`bn_gumbel_multicat` and `bn_categorical` sit at 64x32 — FSQ-64's exact 320-bit
budget — and score 0.6660 and 0.5454 against FSQ's 0.8701. The single-codebook
arms are worse still: `bn_gumbel` (9 bits) 0.3457, and **`bn_vq_ema` collapses
completely — 0 successes in 4,096 clips, 5.7-step survival, 99.6% of episodes
ending on `ee_body_pos`.** That matches the 2026-07-22 10M gate, where single
`vq` was one of two near-flat arms; it is now confirmed at 2B on the canonical
board.

**Endpoint DiffSR beats every other objective — but read WHERE it wins.**
`obj_endpoint_delta` is the worst objective outright (-0.0901, +61.4%).
Reconstruction is the interesting case and must not be summarised as simply
"worse":

| | `ctrl` (endpoint) | `obj_recon` |
| --- | ---: | ---: |
| SR | 0.9023 | 0.8931 (-0.0093) |
| MPJPE-L | 24.49 mm | 27.50 mm (+12.3%) |
| MPJPE-G | 212.3 mm | **446.3 mm (+110%)** |

On success rate the two are effectively tied and reconstruction is the **best
of all eight alternative objectives**. Its +12.3% on local error sits inside
the ~15% unresolved band at one seed and is mid-pack among objectives. The one
resolved difference is **global error, where it is the worst objective by a
wide margin** — the next worst is `endpoint_delta` at 238.8 mm and every other
objective is within +/-16% of the control. The `no_push` partner says the same
more starkly: reconstruction edges the control on success (0.8835 against
0.8828) at 604.7 mm global error against 332.4.

Not an unconverged artifact: its curve improves 598.7 -> ~400 mm and flattens,
never approaching 212.

So the defect is specific — **the reconstruction latent yields a policy that
gets the pose right and ends up in the wrong place.** That is precisely the
drift case MPJPE-L flatters and `canonical-paper-metrics.md` made MPJPE-G
mandatory to expose, now produced by an objective choice of ours rather than by
an external checkpoint. A plausible mechanism: reconstructing the encoder's own
input window is a purely local objective with nothing tying the code to where
the robot ends up, while endpoint DiffSR predicts the successor state, which is
where translation lives.

This indicts THAT reconstruction target, not reconstruction in general. The
study carries one reconstruction cell (continuous 256; there is no recon x FSQ
arm in the star), and its decoder targets the exact input window only. A
future-window or endpoint target is the obvious follow-up and would test the
mechanism directly.

**The phase channel is not optional.** `use_phase_none` — the control with the
2-wide `sin_cos` slot clock removed — falls to 0.3679 / 66.48 mm. At hold 10
the tracker needs the "where am I inside the chunk" signal.

**Two 2026-08-09 findings reproduce on the current recipe.** `in_stride5`
collapses to 0.6558, as stride 5 did under every latent mode then; and adding
joint velocities to the encoder input (`in_fullbody670`) HURTS (-0.0271,
+11.4%), so the 380-value `root_qpos` frame is the better input.

**The 64-D hold-1 "dead zone" is not dead at 2B.** `ix_fsq64_hold1`, the
interaction probe, trains to 0.8496 / 30.45 mm. It is worse than FSQ-64 at
hold 10 (0.8701) but nowhere near the collapse the 100M grid implied. What the
100M grid measured was a budget artifact, not a dead cell.

**MPJPE-G reorders the board again, and both hold-1 arms win it.**
`ix_fsq64_hold1` has the best global error in the whole study at 136.9 mm and
`use_hold1` the second best at 150.4 mm, while both are mid-table on local
error. Publishing at 50 Hz costs a little success rate and buys a large
reduction in global drift.

### The budget axis earned its place

Ranking the 28 healthy arms by success rate at 1.0B and again at 2.0B agrees at
Spearman **+0.847**. The disagreement is concentrated among the top arms, whose
SR spread is only 0.8906-0.9023 — near-ties reshuffling, not real reordering.

**Nine arms had not converged at 2B**, still gaining more than 0.01 SR over
their last 0.5B: `obj_endpoint_delta` (+0.0469), `bn_sonic_fsq64_l8` (+0.0391),
`bn_sonic_fsq32` (+0.0234), `in_stride5` (+0.0234), `in_fullbody670` (+0.0195),
`bn_sonic_fsq16` (+0.0156), `in_window_full` (+0.0156), `bn_categorical`
(+0.0117), `use_hold1` (+0.0117). **The quantized arms dominate that list while
the continuous arms have flattened**, so the discrete-versus-continuous gap
above is an upper bound on the gap at convergence and must be reported as
"at 2B", never as a property of the interface.

### Provenance

Jobs `5583773`-`5583802` (low-level) on the encoders from `5583651`-`5583738`.
Mirror `logs/interface_design_study_mirror/`, rows
`logs/interface_design_study_eval/`. Reproduce with the campaign's
`mirror.sh` then `eval.sh`.

## Low-level section reframed as a design study; MPJPE-G reorders the board (2026-08-19)

**Decision: the paper's low-level section is a DESIGN STUDY of learned command
interfaces — the ablation is the contribution, not support for the planner.**
`final-paper-experiment-design.md` records a 2026-08-17 discussion and is not a
binding contract; its 2x2 objective x bottleneck grid is one cell block inside
a much wider space, and its headline "lock" is not one.

Scope set with the user: all four encoder-objective families (successor,
reconstruction, JEPA, co-trained); reconstruction keeps **one** target, the
exact encoder input window, so no new decoder code is needed; every arm trains
to a matched **2B** frames; and **no arm is promoted to a longer budget** —
promoting winners selects on the outcome and leaves rows of mixed budgets.
**Budget is an axis instead**, read off training curves and checkpoint
milestones.

Campaign: `experiments/campaigns/2026-08-19-interface-design-study/`. A **star**
of 34 arms — `ctrl` is the hub and every other arm changes exactly ONE field —
over four axes plus one width-x-hold interaction probe.

**SUBMITTED to ICE on 2026-08-19**, W&B group `interface-design-study`. The
first submission (87 jobs, `5583651`-`5583738`) lost every low-level stage to a
control-plane bug; the encoders survived and the low-level stages were
re-submitted alone. **All 29 arms are now training** (`5583773`-`5583802`), the
control arm verified at iteration 128/5087 and 50.3M of 2.00B frames.

### The bug: every low-level job died at zero seconds

21 of 21 `lowlevel1` jobs failed with **exit 141 at 00:00:00 and no error
message**, while all 29 pretrains completed normally. Exit 141 is 128+13 =
SIGPIPE. The cause was in the W&B run-id block that
`pipeline/cluster/slurm.py` renders into every batch script:

```bash
resolved_run_id="${run_id_base}-$(tr -dc 'a-z0-9' < /dev/urandom | head -c 6)"
```

`head -c 6` closes the pipe after six bytes while `tr` is still streaming an
endless source, so `tr` takes SIGPIPE, `set -o pipefail` surfaces 141, and
`set -e` kills the job before it prints anything. It fires only on a stage that
declares `WANDB_RUN_ID` — every low-level stage, no pretrain — which is exactly
what made it look like a per-stage configuration problem rather than a shared
bug. It has been latent since the 2026-08-18 run-id pinning and would break any
campaign whose chain has no `wandb_run_id` state file yet.

Fixed by generating the token from a bounded read, `od -An -tx1 -N3
/dev/urandom | tr -d ' \n'`, which cannot leave a writer on a closed pipe.
`source/imitation_experiments/tests/test_cluster_run_id_block.py` executes the
rendered block under `set -euo pipefail` rather than pattern-matching it, and
was confirmed to fail with `assert 141 == 0` against the old form.

Diagnosis notes worth keeping: the archive extracted fine (`gzip -t` clean,
`tar` verified by hand on the very node that failed, 28 TB free on its `/tmp`),
and the pretrain that preceded the first failure wrote a real 416 MB encoder in
27:55 — so neither the workspace nor the encoder was ever at fault.

### Recovery

`plan --only-stage lowlevel1` re-submitted the low-level stage alone against
the encoders already on disk, so no pretrain was repeated. No `wandb_run_id`
state file existed anywhere — the block died before writing one — so every
chain minted a fresh id on the fixed path. `lowlevel2` was not re-submitted: a
2B run at ~104k fps is about 5.3 h against a 15:59 walltime, so the insurance
segment is not needed and can be added later if an arm runs long.

Follow with
`python -m imitation_experiments.pipeline.cluster status --campaign interface-design-study`.

29 arms are active, **every one passed its wiring smoke**, and every plan
resolved offline with ICE preflight passing. 101 contract tests hold the star
property, the `latent_dim == code_latent_dim + phase_dim` width contract, the
`sonic_fsq` level-list contract, W&B id uniqueness, the deferred set, the
one-seed rule, and the no-duplicate-cell rule. The W&B group
`interface-design-study` is confirmed.

**One seed per arm** (user decision). No arm carries a repeat, so a single-arm
difference inside the roughly 15% evaluation band is directional and not
resolved — for tier 1 exactly as much as for tier 2. The budget axis partly
compensates: each arm is scored at five checkpoints, so a gap that holds across
the whole curve is stronger evidence than the same gap at one endpoint. Five
points on one run are not five seeds; any conclusion the paper leans on wants a
second seed before publication.

Five arms are **deferred to tier 4** by user decision and are never planned:
the three co-trained arms, which wait on an encoder-from-checkpoint eval path,
and `use_phi` / `use_z_phi`. They stay defined so the design is intact and
re-enabling is a one-field change. `plan_all.sh` refuses tier 4.

### What the wiring gate caught

Three findings that would each have cost real GPU time:

1. **A duplicated cell.** `obj_jepa_ntp` omitted `--jepa_loss` and a separate
   `obj_jepa_sigreg` passed it explicitly — but sigreg is the parser default,
   so the two arms measured loss 0.26313257217407227 apiece, to every digit.
   They were the same cell. A star with a duplicate spends a full budget twice
   and reports it as two independent measurements. The duplicate is removed,
   the surviving arm names its energy explicitly, and a contract test now
   resolves parser defaults before comparing arms.
2. **The smoke batch changed the answer.** At batch 256 a 512-entry VQ codebook
   cannot hit enough codes and reports a fully collapsed code; at the
   production batch of 8,192 it never does. A 400-update probe watched it
   recover from `z_dim_std_mean` 0.134 to 1.365 as the loss fell 39.2 to 4.7.
   The smoke now runs at the production batch so batch is not a confound
   between the gate and the run it qualifies.
3. **`z_effective_rank` cannot detect a collapse on its own.** On the fully
   collapsed VQ code it reported 11.57 rather than 1, because the covariance is
   degenerate. The gate judges a quantized arm on the trainer's own
   `code_perplexity` (1.0 = a single code) and a continuous arm on the MEAN
   per-dimension spread — never the minimum, since a discrete codebook has dead
   dimensions this early (`bn_gumbel_multicat` measured `z_dim_std_min` 0.0 at a
   healthy `z_dim_std_mean` 0.317).

### The eight 10B arms, re-scored on the canonical testbed

Every prior row in this repo sat on the retired `bones_scoreboard4096_v1` block
under `no_push`, and **no row anywhere carried MPJPE-G**. All eight scorable
arms of `2026-08-15-latent-bottleneck-10b` are now on
`paper_testbed4096_v1` (clean) and `paper_testbed4096_robust_v1` (`no_push`),
16 rows, no failures, about 1.4 minutes per row.

| arm | frames | SR | MPJPE-L | MPJPE-G | `ee_body_pos` |
|---|---:|---:|---:|---:|---:|
| `jepa_sigreg_ebm_hold10_256d` | 10.00B | 0.9146 | **22.60 mm** | 151.77 mm | 280 |
| `cont_det_hold1` | 10.00B | 0.9187 | 23.37 mm | 107.10 mm | 279 |
| `cont_det_ln_hold1` | 10.00B | **0.9226** | 24.19 mm | 119.42 mm | 274 |
| `cont_det_hold1_resetramp` | 10.00B | 0.9192 | 24.96 mm | 107.84 mm | 285 |
| `fsq64_hold10` | 10.00B | 0.9062 | 25.99 mm | 146.53 mm | 324 |
| `jepa_sigreg_ebm_hold10_fsq64` | 10.00B | 0.9004 | 26.48 mm | 123.15 mm | 358 |
| `jepa_ntp_hold10_256d` | 8.50B | 0.8884 | 26.54 mm | 161.76 mm | 391 |
| `jepa_pure_256d_hold1` | 10.00B | 0.8865 | 28.98 mm | **86.52 mm** | 406 |
| released SONIC (reference) | - | 0.9912 | 28.75 mm | 135.73 mm | - |

**The local and global metrics do not rank the same arms.** Over these eight,
Spearman is **-0.071** between MPJPE-L and MPJPE-G, against +0.786 between
MPJPE-L and success rate. `jepa_pure_256d_hold1` is LAST on local error and
FIRST on global error by 20 mm; `jepa_sigreg_ebm_hold10_256d` is the exact
inverse. This is the drift-versus-pose split
`canonical-paper-metrics.md` made MPJPE-G mandatory for, now measured on our
own arms rather than only on the released SONIC checkpoint. Every leaderboard
in this repo before today ranked on the local metric alone.

Read with care: eight arms is a small rank sample, MPJPE-G repeats only to
about 2% run-to-run, and success-only figures at different success rates carry
a selection bias. What the numbers support is "the two metrics disagree", not a
new winner. The `ctrl` hub of the design study is unaffected — it is chosen
from proven configurations, not from this ordering.

LayerNorm is still not established: `cont_det_hold1` (LN off) beats
`cont_det_ln_hold1` on both error metrics (23.37 vs 24.19 mm local, 107.10 vs
119.42 global) and loses 0.0039 of success rate. The design study carries
`bn_no_ln` with two seeds to settle it.

**Run-to-run repeat, measured here.** Two arms were scored twice on the
identical protocol, because their first rows predated the per-environment
acceleration fix. The repeat moved success rate by 0.0010 and 0.0014, MPJPE-L
by 0.04 and 0.01 mm, and MPJPE-G by 0.45 and 0.67 mm (0.4%). That is looser
than the ~0.01 mm / ~0.0003 floor `canonical-paper-metrics.md` records for the
released SONIC checkpoint, and it is the right floor to use for OUR arms: it
puts the whole `cont_det_*` cluster inside noise of each other on success
rate.

### Supporting changes

- **`paper_milestone_testbed256_v1`**, a new profile: 256 clips drawn stride-16
  from `TESTBED4096_RANKS`, the same population and the same
  `sonic_sr_clean_v1` protocol as the headline board, for scoring intermediate
  checkpoints. `bones_milestone256_v1` could not serve this — it is a strided
  draw from the RETIRED legacy block, so a curve on it cannot be read against a
  testbed row.
- **Acceleration distance now accumulates per environment**
  (`lowlevel/evaluate_checkpoint.py`). It previously existed only as a
  board-wide, all-transition mean, so a success-only acceleration row was not
  computable — SONIC publishes velocity AND acceleration distance, and the
  success-only reduction is what every other paper metric uses.
- **`summarize_paper_boards` rows** now carry MPJPE-G, velocity, acceleration,
  per-termination counts and true env frames (recovered from `f<frames>`
  checkpoint tree names when metadata omits them). A metric a result file
  predates stays `None` rather than silently reporting the all-transition mean
  in a success-only column.
- **Online MPJPE needed no work**: `TrainHealth/mpjpe_{l,g}_mm_transition_ewma`
  already reaches the trainer through `mdp/commands/reference.py:628-642` and
  `envs/rlopt.py:101-116`.
- **Disk**: `outputs/gr00t_language30` (97 G) and `outputs/gr00t_language10`
  (41 G) moved to `/mnt/storage` behind symlinks after a file-count and
  byte-count verification. `/mnt/hsstorage` went from 20 G free (a collection
  had already died mid-write) to 157 G. `outputs/planner_10b` stayed on NVMe
  because the planner campaign still reads it.

## Planner cadence, the ceiling decomposition, and the SONIC row (2026-08-19)

One seed, DR off, no ensembling, 28 motions; sweep arms are 140 episodes and
rows of record are 560. Campaign:
`experiments/campaigns/2026-08-17-planner-10b-trackers/`.

**The tracker is not what limits the hold-1 planner.** The oracle latent row
on the identical protocol is 17.19 mm (fall-free 1.000, every episode passes
the SONIC threshold) against the planner's 42.83 mm, so the planner owns
about 60% of the error.

**Consuming the whole plan beats re-planning often.** Sweeping how many of the
head's 30 slots are consumed before the next head call: 50.76 mm (1 slot),
47.61 (3), 44.18 (10, the shipped setting), **38.73 mm (30)** — with 36 head
calls instead of 106. Confirmed at 560 episodes: **38.41 mm** against the
shipped 42.83, fall-free 1.000 — 4.41 mm paired per motion, 95% interval 1.1
to 9.0 mm, p = 0.043, better on 19 of 28 motions. The ordering is the reverse of per-publication accuracy
(z cosine 0.517 at 30 slots against 0.661 at 10): a fresh head call draws
fresh flow noise, so frequent re-planning jitters the command stream, while
one draw's slots are mutually consistent. Confirmation at 560 episodes is
running. Null results measured the same way: 16 ODE steps (43.65 vs 44.18)
and clean observations (44.62 vs 42.83, slightly worse clean).

**Why the head is the limit.** Open-loop per-slot cosine decays only from
0.777 to 0.760 across the consumed window, but the closed-loop published
cosine is 0.661 — a 15% covariate-shift gap, because the collection was
recorded with the oracle driving and the planner drives at evaluation. The
fix is a planner-driven (DAgger) collection; it is blocked on disk, not on
code. The workstation disk hit 100% on 2026-08-19 and a lookahead-bearing
hold-1 collection failed mid-write.

**SONIC, on this protocol at last.** `eval_sonic_row.sh` runs the released
v1.1 decoder on the same 28 motions, 20 episodes, fall-only, unperturbed:

| system | tracker ceiling | planner row (560 ep) | planner-induced |
|---|---:|---:|---:|
| ours, 10B `cont_det_ln_hold1` | **17.19 mm** | 42.83 mm (10 slots) | +25.6 mm |
| ours, same, 30 slots | 17.19 mm | 38.41 mm | +21.2 mm |
| ours, 30 slots, clean observations | 17.19 mm | **37.60 mm** | +20.4 mm |
| SONIC release v1.1 | 22.70 mm | 38.49 mm | **+15.8 mm** |

Our tracker is 24% better. At the shipped cadence SONIC won end to end; at 30
slots the two are level (37.60 vs 38.49 on the same clean-observation
contract). The residual difference is in the planner-induced column: SONIC's
latent space absorbs planner error better.

On SONIC's OWN success criterion the two are not level at all: episodes that
never violate its thresholds are 0.9143 for us against 0.6232 for the SONIC
row (same semantics both sides), and SONIC's failures are almost entirely
wrist height (`ee_body_pos` 185 of 560). Both oracles pass 100%, so it is the
planner violating thresholds rather than the tracker. SONIC's MPJPE measured
under its own terminating contract (34.13 mm) is survivor-biased — violating
episodes truncate at 402 steps against 496 — so the comparable number stays
38.49 mm. All three rows
share the GR00T head recipe and budget and differ in the (encoder, tracker)
pair, but SONIC's decoder is its own tracker in its own evaluator binary, so
this compares systems, not interfaces.

**Explicit-vs-latent: the first reading was confounded.** The explicit head
trained on the fsq64 collection and was first scored driving the HOLD-1
tracker, states it had never seen; that mismatch, not the interface, produced
its 2.2x deficit. Paired with the tracker that produced its training states,
the explicit route wins: 44.75 mm against the latent arm's 52.30 on the fsq64
tracker (better on 21/28 motions), both 12k updates, 560 episodes. The likely
mechanism is that encoding a predicted explicit window projects it onto the
manifold of valid latents, a guardrail the latent head does not have. The
matched explicit row on the hold-1 tracker still does not exist: it needs a
lookahead-bearing hold-1 collection, which the full disk refused.

## Planner 10B heads: DR-on rows, ensembling split, first live async run (2026-08-18)

Local evaluation of both `planner_10b` heads (update 12k), 28 motions x 20
episodes, seed 0, one seed each — all preliminary. Full summaries under
`logs/planner_10b/isaac_eval/`.

| arm | protocol | fall-free | MPJPE-L ep-mean | success-only micro | thresh pass |
|---|---|---:|---:|---:|---:|
| `fsq64_10b` | DR=on, exp ens (46.95 protocol) | 0.9893 | 50.48 mm | 53.15 mm | 0.9089 |
| `ln_hold1_10b` | DR=on, exp ens | 1.000 | 52.90 mm | 55.44 mm | 0.8732 |
| `fsq64_10b` | DR=off, no ens, sync | 1.000 | 52.30 mm | 56.30 mm | 0.8768 |
| `ln_hold1_10b` | DR=off, no ens, sync | 1.000 | 42.83 mm | 44.89 mm | 0.9107 |
| `fsq64_10b` | DR=off, no ens, async lead 5 | 0.9982 | 69.40 mm | 69.89 mm | 0.7946 |
| `ln_hold1_10b` | DR=off, no ens, async lead 5 | 1.000 | 41.45 mm | 43.64 mm | 0.9125 |

Findings, all one-seed:

- **DR=on (the 46.95 mm protocol)**: `fsq64_10b` 50.48 mm against the
  2026-08-13 arm's 46.95 mm (7.5%, inside evaluation noise, unresolved), and
  the two 10B arms are level with each other (4.8% apart).
- **Temporal ensembling splits by interface**: it helps `fsq64_10b`
  (45.49 with, 52.30 without, DR off) and hurts `ln_hold1_10b`
  (46.62 with, 42.83 without, ~8%, inside the noise band). A per-step
  head appears to prefer its raw chunk. Repeats would settle it.
- **First live D1 async run** (relaxed gate: labelled, next to the sync
  companion, never pooled). `ln_hold1_10b` async is level with its sync
  companion (41.45 vs 42.83 mm; 2,380 deadline misses ~8.5% of renewals).
  `fsq64_10b` async degrades hard: 69.40 vs 52.30 mm (33%), 8,517 misses
  (~30% of renewals), service round-trip p50 163 ms against a 10-control-step
  renewal. Caveat: the deadline is counted in sim control steps while the
  service round-trip is wall-clock, and the evaluator and service share one
  GPU — so miss rates are execution-mode-shaped, not a real-time 50 Hz
  certificate. The hold-1 arm's 30-slot horizon (20-step consumable tail)
  absorbs latency; the FSQ arm's 3-slot horizon does not.

## BONES-SEED latent-design ablation prepared and qualified (2026-08-18)

The paper method ablation is now the controlled 2 x 2 grid from the prior
design discussion: endpoint DiffSR versus exact encoder-window reconstruction,
crossed with a continuous 256-D bottleneck versus the SONIC FSQ 64 x 32
bottleneck. All four arms fix `root_qpos`, horizon 10 with the endpoint hidden,
hold 10, `robot_heading`, the same encoder trunk, 50,000 offline updates, a
frozen encoder, and a 10B low-level frame target. This replaces the old
autoencoder comparison, which also changed online training, input width,
bottleneck, and completion state.

The new offline reconstruction objective and checkpoint resume path passed 62
focused RLOpt tests. The continuous and FSQ reconstruction arms each completed
one real Isaac/Newton pretrain update and one 128-frame frozen-encoder IPMD
iteration with the correct 258-D and 66-D command widths. These are wiring
qualifications, not results. All four seed-0 control-plane plans resolve with
one pretrain plus four full-target `afterany` low-level segments. No ICE job was
submitted. The proposed W&B group is `latent-design-ablation`; it must be
confirmed before submit. Campaign front door:
`experiments/campaigns/2026-08-18-bones129k-latent-design-ablation/`.

## Training throughput: collection rebuilt, 20k-env leader on ICE (2026-08-18)

Profiling with the opt-in phase timers (`agent.trainer.profile_iterations=true`)
found collection dominated by Python plumbing rather than physics. Measured on
the tuned latent recipe at 8,192 envs on the local RTX PRO 6000, medians over
iterations 21-80 of matched runs on an exclusive GPU:

| phase | before | after |
| --- | ---: | ---: |
| collect | 2413 ms | 945 ms |
| learn | 1153 ms | 1158 ms |
| iteration total | 3566 ms | 2107 ms |

What changed, all value-identical: terminal observations are published as one
batched `{_env_ids, obs}` dict instead of a per-environment object array whose
clone loop issued thousands of small device copies per step; the wrapper's
terminal-obs reader no longer runs a 16,384-iteration Python fill per key on
every step, including steps with no reset; the env log payload is detached
instead of `.cpu().item()`-ed per scalar per step; recorder hooks are gated on
active recorder terms. Same-seed curves match at iteration 80 (ep_len 34.92 vs
35.14, r_ep 3.808 vs 3.798).

Two measured non-results, recorded so they are not retried blindly:

- **`torch.compile` does not help the PPO update here.** Compiling the loss
  module and GAE gives `update/ppo_terms` 867 -> 855 ms; `max-autotune` ties
  `default` (854 ms) and its CUDA-graphs variant crashes on latent-sampler
  tensor reuse. The step is GEMM-bound at roughly 45% of TF32 peak. Compile
  stays opt-in and off. The only reproducible win was GAE, 80 -> 57 ms.
- **The reference data plane never waits.** Sequential rows are gathered on a
  worker and copied H2D while physics runs (`wait_ms` about 0.01). Reference
  streaming is about 1.3 ms of a 36.8 ms step. Switching
  `reference_prefetch_mode` to `next_and_reset` also overlaps the reset-row
  gather: step 36.8 -> 34.5 ms, at the cost of reset draws seeing sampler
  failure weights one control step stale.

Per-step split after the change (8,192 envs, n=600): physics 16.7 ms,
`_reset_idx` 5.0, `write_data` 3.3, reward 2.4, command 1.5, obs 1.3,
termination 1.2, reference streaming 1.3. What remains is upstream
Newton/MJWarp and Isaac Lab manager code.

**ICE job 5580308** (`cont_det_ln_hold1_fullbody_env20k`, seed 0) carries this
into production: 20,480 envs x 24 = 491,520 frames per batch, 20,346
iterations = 10.0B frames, `next_and_reset`. First readings are about 131.5k
fps against roughly 104k fps for the 16,384-env 10B leaders, which projects
10B in about 21 h instead of 27 h. That comparison is not frame-matched or
batch-matched: the batch is 1.25x larger and the code differs, so it is
"this configuration sustains 131.5k fps", not "the optimizations bought 26%".
Attribution needs a 16,384-env run on current code.

An earlier attempt (5580302) died before its first iteration: W&B refuses a run
id that was ever deleted (`error 410`). Run ids are now generated once per
`(arm, seed)` output tree and recorded in `<output_root>/wandb_run_id`.

## Final paper experiment design locked (2026-08-17)

The three-section paper design is codified in
[final-paper-experiment-design.md](final-paper-experiment-design.md). Headline
pipeline: the 10B `fsq64_hold10` tracker plus the GR00T language planner with
temporal ensembling, pending one parity re-run of the planner against the 10B
checkpoint. The approved claim against SONIC is pipeline-level parity, not a
tracker win. The direct 50 Hz ceiling row is dropped, the explicit planner row
is the 38-D single-frame `root_qpos` command, and no explicit 10B re-run will
be made (frame counts are printed per row instead). That page supersedes the
two-row main-grid decision in `causal-interface-paper-plan.md` where they
disagree.

## Continuation: SONIC resets, 10B -> 20B, both interfaces (2026-08-18)

`experiments/campaigns/2026-08-18-sonic-reset-20b/`: the two leading trackers
resume from their 10B final checkpoints and train a second 10B with exactly one
change — `selection` moves from `random80_adaptive20` to `sonic`
(`random_trajectory_sampling_ratio` 0.8 -> 0.0, so every reset comes from the
SONIC joint rank+frame failure sampler), with a landing ramp
`adaptive_uniform_ratio` 0.5 -> 0.1 inside segment 5 and 0.1 pinned after.
`ln_hold1_sonicreset` (continuous leader) is ICE jobs 5580042-5580045;
`fsq64_hold10_sonicreset` (SONIC token space, the planner-facing interface) is
5580046-5580049. Submitted 2026-08-18, W&B group `sonic-reset-20b`. The earlier
`cont_det_hold1_resetramp` arm never touched the branch ratio — its ramp moved
the failure-weighted share 4% -> 16% only, which is why it read inconclusive.

W&B note: this chain was created in shared mode and keeps it to its end.
Shared mode discards `wandb.log(step=...)`, so it and the 08-17 `*_dyn` runs
plot on a log-call index; read the `env_frames` metric RLOpt declares as the
x-axis, not `_step`. Shared mode is retired for new runs (see below), but a
chain never changes mode mid-flight.

## Low-level leaderboard: the latent interface catches the explicit baseline (2026-08-17)

Eight arms of `experiments/campaigns/2026-08-15-latent-bottleneck-10b/` scored
on the frozen 4,096-motion scoreboard (ranks 12288-16383, frame-0, seed 0, mode
actions, `no_push`, Newton/MJWarp, released-SONIC thresholds, `foot_pos_xyz` and
`base_too_low` disabled) — the same protocol as the 2026-08-09 table below, run
locally in about four minutes per arm.

| arm | frames | SR | succ MPJPE-L | `ee_body_pos` |
|---|---:|---:|---:|---:|
| released SONIC | - | 0.9937 | 28.65 mm | 26 |
| **`cont_det_ln_hold1`** | **10.00B** | **0.9368** | **22.86 mm** | 217 |
| `root_qpos_explicit` (explicit baseline) | 7.60B | 0.9358 | 19.21 mm | 212 |
| `cont_det_hold1` | 10.00B | 0.9343 | 22.60 mm | 235 |
| `cont_det_hold1_resetramp` | 10.00B | 0.9307 | 23.84 mm | 248 |
| `jepa_sigreg_ebm_hold10_256d` | 10.00B | 0.9282 | 22.26 mm | 234 |
| `fsq64_hold10` | 10.00B | 0.9197 | 24.93 mm | 260 |
| `jepa_sigreg_ebm_hold10_fsq64` | 10.00B | 0.9197 | 25.83 mm | 270 |
| `jepa_ntp_hold10_256d` | 8.50B | 0.9077 | 25.71 mm | 307 |
| `critic_no_latent` (best pre-08-15 latent) | 5.00B | 0.9062 | 24.39 mm | 318 |
| `jepa_pure_256d_hold1` | 10.00B | 0.9050 | 27.94 mm | 336 |

Every new arm beats the best previous latent arm on both axes, and the top four
reach the explicit baseline's success rate. Two limits on that reading: the
explicit row has 7.60B frames against these 10.00B, so nothing here is
frame-matched; and the explicit row still wins success-only MPJPE-L by 15-18%.
"The latent interface has caught up on falls" is what this table supports; "the
latent interface wins" is not.

The best arm is a continuous deterministic 256-D latent, held one control step,
with encoder LayerNorm on. Its pair without LayerNorm (`cont_det_hold1`) is
0.0025 SR behind and 0.26 mm ahead, which is inside noise on both axes, so
LayerNorm is not established as the cause — the two continuous hold-1 arms
together are simply the strongest interface tested.

`ee_body_pos` is still the dominant failure in every row, and its spread across
arms (217-336) is wider than either tracking metric's, exactly as the
2026-08-09 attribution found.

Not on this board: the two `*_dyn` arms (their encoder is fine-tuned inside the
tracker checkpoint, so the pretrained-encoder path would score a mismatched
pair) and the categorical arms (no bundle exporter for a learned codebook).
Campaign README carries the EC/MuJoCo screen, the checkpoint neighbourhood
check, and the board-correlation study.

### The CPU screen board changed (2026-08-17)

The ten-motion `selected10_repeats5_v1` sidecar board does not predict this
scoreboard's survival axis: over the eight arms above, its fall-free rate ranks
them at Spearman **-0.238** against scoreboard success rate. Cause: 3,545 of the
4,096 scoreboard motions are passed by every arm, and the ten language clips are
quiet standing motions scored on a pelvis-height fall only.

Replacement, approved and built the same day: profile
`sidecar_ec_strat64_v1` — 64 motions drawn from the scoreboard ranks, seven or
eight from each "how many arms fail this motion" bucket, three noise draws each
(192 episodes, ~3.6 min CPU), success judged by the released SONIC thresholds,
and every figure re-weighted to the population by the board's per-case
`population_weight`. Measured against the same eight arms: survival **+0.571**
(from -0.238), quality **+0.929** (from +0.690).

The survival axis is better but still only moderate, and the residual is a
**backend** disagreement rather than a sampling one: `fsq64_hold10` scores 0.55
weighted success in MuJoCo under sensor noise against 0.92 in Isaac, the largest
per-arm gap in the set. Use the new board as the screen; the 4,096 Isaac board
stays the deciding board until that backend gap is explained.

## GR00T language planner — best result to date (2026-08-19)

**`ln_hold1_10b` on the 10B continuous 256-D hold-1 tracker, consuming its
whole 30-slot plan: 38.41 mm MPJPE-L, 1.000 fall-free, 0.9143 SONIC-threshold
success** over 28 motions x 20 episodes (560), DR off, fall-only, Newton,
2000-step cap, seed 0. Clean-observation twin 37.60 mm; service-backed async
twin 38.78 mm at 0.9464 threshold success.

Against the released NVIDIA SONIC v1.1 on the SAME protocol and motion set:
MPJPE is level (38.49 mm theirs) but SONIC's own threshold criterion is not —
**0.9143 ours against 0.6232 theirs**, their failures almost all wrist height
(`ee_body_pos`, 185 of 560). Tracker ceilings: 17.19 mm ours, 22.70 mm
SONIC's; both oracles pass every episode, so the threshold gap is planner
behaviour, not decoder quality.

How solid the cadence gain is: it beats the shipped 10-slot row (42.83 mm) by
4.41 mm, or 10.3%. Paired per motion across the 28 motions — the right unit,
because between-motion spread (29 mm) dwarfs within-motion spread (8 mm) —
that difference has a 95% bootstrap interval of 1.1 to 9.0 mm and a two-sided
p of 0.043, with 30 slots ahead on 19 of 28 motions. So the improvement is
supported by the data in hand, not merely directional. Both rows are single
runs; repeating each would double the paired sample and is worth doing before
the number goes in a paper. Two same-configuration re-measurements put
run-to-run drift near 1% (94.42 vs 95.68, 38.41 vs 38.73), small next to the
effect.

These are DR-OFF numbers and are not comparable with the 2026-08-13 headline
(46.95 mm, DR on); that row's DR-on successor is 50.48 mm (`fsq64_10b`).

Full rows, the cadence sweep, the ceiling decomposition and the SONIC
comparison:
`experiments/campaigns/2026-08-17-planner-10b-trackers/README.md`. The older
recipe and its refuted alternatives remain in
[`progress-report.md`](progress-report.md) section 2 and
`experiments/campaigns/2026-08-12-gr00t-language30-compositionality/PLAN.md`.

## Default BONES-129k H200 training recipe (2026-08-05)

The default single-H200 BONES-SEED 129k controller geometry is now **16,384
environments x 24 rollout steps**, with minibatch **294,912**, gamma **0.97**,
Newton/MJWarp, seed 0, and checkpoints every 50M frames. New long runs use a
10B-frame cap. ICE still limits one allocation to 16 hours, so slower online
latent learners may require a continuation from the newest intact checkpoint
under persistent `/data`; a TIMEOUT is not evidence that the 10B cap was
reached.

The common environment/data contract is `Isaac-Imitation-G1-v2`, the resident
129,785-motion reference arrays at
`/data/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1`, the
root-qpos macro interface, tuned rewards/curriculum, and the
`random80_adaptive20` reset sampler. That sampler chooses a trajectory
uniformly and a frame uniformly within its first half on 80% of resets; the
remaining 20% use the learned SONIC failure distribution.

Log these runs to W&B project **`g1-bones-seed`**, with concise functional
groups such as **`bones129k-ablation`**. Tags must identify the common
`bones-seed`, `129785`, `v2`, `newton`, `rollout24`, `gamma097`, and
`reset80-adaptive20` features plus the arm-specific command scheme (for
example `sonic_fsq32`, `vqvae_k32`, `autoencoder`, `root_qpos_explicit`, or
`reset80_diffsr`). The guarded launcher and exact five-arm contracts live in
`experiments/campaigns/2026-08-05-bones129k-latent-sampler/`.

The first production submission uses ICE jobs `5567801` (frozen DiffSR reset
baseline), `5567802` (SONIC-style FSQ), `5567803` (EMA VQ-VAE), `5567804`
(continuous autoencoder), and `5567809` (explicit root-qpos). The common
submitted workspace archive SHA-256 is
`e20e93be390a9985df0472893f20ce2b68050dd12a89366743a9dfc66f951d05`;
the complete arm-to-tag mapping and smoke provenance are retained in the
campaign's `cluster_submission.json`.

## BONES-129k L2T run (2026-08-09)

Learning-to-track (L2T) uses a privileged teacher with the explicit reference
command and a deployable student with the 258-value DiffSR latent command.
The command split passed 146 RLOpt tests, five Isaac configuration tests, and a
real local Newton iteration over the full reference arrays. ICE H200 job
`5573723` used the pinned 380-input root-qpos encoder and W&B project/group
`g1-bones-seed` / `l2t` (run `2znme7lg`). It completed normally at
1,000,341,504 frames, but that submission was incorrectly capped at 1B instead
of the required 10B frames. Treat it as an incomplete qualification, not the
requested result. It was not used as a resume point. Corrected ICE H200 job
`5574140` started from scratch on 2026-08-09 with 25,432 iterations
(10,000,269,312 actual frames), a fresh persistent output tree, and the same
verified data and encoder contracts. Its W&B run is `ycmodfu3` in
`g1-bones-seed` / `l2t`. The local launcher defaults to this 10B budget.
The workspace archive SHA-256 is
`d153c3d784ae322a0d7790b40cd3005e9d2f5b0bc7352c417e0165bd2e3e9449`.
The launcher and full provenance record are in
`experiments/campaigns/2026-08-09-bones129k-l2t/`.

The incomplete 1B run's student was evaluated on the canonical selected-ten
language motions at frame 0. Under the SONIC-compatible randomized-no-push
criterion it completed 0/10 motions; eight motions fired `ee_body_pos`, two
fired `anchor_ori`, and one fired `anchor_pos`, with one double termination.
The separate non-terminating diagnostic covered all 5,137 transitions: nine
motions fell below 0.4 m, step-weighted pre-fall MPJPE-L was 135.98 mm, and
full-horizon MPJPE-L including post-fall frames was 328.75 mm.

The checkpoint's privileged teacher completed 10/10 of the same motions under
the SONIC-compatible pass, with 14.71 mm success-only MPJPE-L and no tracking
failure. Its non-terminating pass had no falls and 14.00 mm frame-weighted
full-horizon MPJPE-L. The teacher consumes the explicit reference and
privileged robot state, so it is a training ceiling and is not deployable. The
teacher-student gap isolates the incomplete 1B result to student distillation
or the latent interface. The checkpoint, fresh ten-motion manifest, metrics,
and both sets of ten videos are recorded in the campaign README.

A 2026-08-07 fidelity audit against the SONIC paper (arXiv 2511.07820) and
the released `gear_sonic` BONES-SEED config found that `5567802` matches
SONIC's latent-learning objective exactly — hold-1 per-step re-encoding,
policy gradient into the encoder, reconstruction MSE at the released
coefficient 0.01, FSQ 64 coordinates x 32 levels, encoder MLP
[2048, 1024, 512, 512] SiLU, no phase — but not its encoder input: SONIC
reads 10 future frames spaced 0.1 s (a 0.9 s span) of 14-body keypoint
positions plus a 6D root-orientation difference, where `5567802` reads 10
consecutive 0.02 s frames of joint qpos/qvel plus anchor pose. The corrected
`sonic_fsq32_v2` arm (new `future10_stride5` encoder view with
`frame_stride` support, components `[keypoint_pos, root_ori]`, SONIC's own
14-body list; 480-value encoder input) was submitted as ICE job `5571455`.
The audit details and the deliberately retained deltas (our common controller
recipe; recon on a dedicated Adam(2e-5) instead of SONIC's single summed
loss) are recorded in the campaign README.

## What the 4,096-motion scoreboard says we are losing to (2026-08-09)

Superseded as a leaderboard by the 2026-08-17 section above; the rows here stay
because the failure attribution and the stride-5 result are still current.

`experiments/campaigns/2026-08-08-bones129k-4096-scoreboard/` scores every
finished BONES-129k arm under ONE protocol: 4,096 environments, ranks
12288-16383 pinned, frame-0 starts, seed 0, mode actions, `no_push`,
released-SONIC thresholds, `foot_pos_xyz` and `base_too_low` disabled.

**The explicit root_qpos baseline beats every latent arm on both metrics.**
Scored 2026-08-09: SR 0.9358, success-only MPJPE-L 19.21 mm at 7.6B frames,
against the best latent arm's 0.9062 / 24.39 mm at 5B. The frame counts are not
matched and the gap favors explicit, but 21% lower MPJPE is not a 52%-more-
training artifact to wave away. Treat "latent beats explicit" as unproven.

| arm | frames | SR | succ MPJPE-L | `ee_body_pos` | `anchor_ori` | `anchor_pos` |
|---|---:|---:|---:|---:|---:|---:|
| `root_qpos_explicit` | 7.60B | 0.9358 | 19.21 mm | 212 | 50 | 13 |
| `critic_no_latent` | 5.00B | 0.9062 | 24.39 mm | 318 | 57 | 17 |
| `old_z256` | 5.00B | 0.9058 | 24.52 mm | - | - | - |
| `skill_state_occupancy` | 5.00B | 0.9050 | 25.11 mm | 322 | 53 | 21 |
| `fsq64_sonic` | 5.00B | 0.8943 | 25.74 mm | - | - | - |
| `stride5_det64` | 5.00B | 0.7063 | 35.93 mm | 1079 | 110 | 28 |
| `stride5_fsq64` | 5.00B | 0.6785 | 37.59 mm | - | - | - |
| `stride5_gumbel64` | 4.75B | 0.5020 | 49.96 mm | - | - | - |
| released SONIC | - | 0.9937 | 28.65 mm | 26 | - | 4 |

Two readings beyond the headline. First, against the released SONIC checkpoint
we win on precision and lose on falls. Second, **stride 5 is bad under every
latent mode tested** — deterministic, FSQ, and Gumbel all collapse relative to
their stride-1 counterparts, so the 0.9 s SONIC cadence is not transferable to
our recipe as-is.

The falls are almost one termination in every row, explicit included:

`ee_body_pos` is a Z-only height error over `G1_EE_BODY_NAMES` — both ankles
and both wrists. Narrowing it to one pair at a time on the best arm
(`logs/bones129k_ee_attribution/`) gives wrists-only 231 (SR 0.9197) and
ankles-only 208 (SR 0.9182); 231 + 208 > 318, so many environments fail on
both. This is end-effector height in general, not a wrist-only defect.

Consequence for planning work: reward-kernel tuning that buys precision does
not touch this, exactly as the 2026-08-04 v2 tuned-reward screen already
recorded. Attack falls.

## Four arms against the explicit root_qpos baseline (2026-08-09)

Four single-variable arms, each against a stated control, all on ICE:

| campaign | axis | job(s) | control |
|---|---|---:|---|
| `2026-08-09-bones129k-ee-reward` | `env.rewards.motion_ee_pos.weight=2.0` (was inert) | `5573515` | `5573413` |
| `2026-08-09-bones129k-encoder-finetune` | `agent.ipmd.hl_skill_finetune_enabled=true` | `5573516` | `5573413` |
| `2026-08-09-bones129k-fullbody-encoder` | encoder input `root_qpos` 380 -> `full_body` 670 (adds reference joint velocity) | pretrain + dependent tracker | `5573413` |
| `2026-08-08-bones129k-fsq-anchor-critic` | combined: `sonic_fsq` 64x32, scaled nets, expert-heading frame, critic `[reference]` | `5573502` -> `5573503` | not single-variable; see its README |
| `2026-08-09-bones129k-hold1` | command hold 10 -> 1 control steps, on the old z256 recipe | `5573633` | `5567801` (`old_z256`) |

The hold-1 arm publishes 50 commands/second instead of 5 and runs the encoder
every control step. It is a **low-level ceiling**, not a planner-interface row:
the paper's planner comparison publishes at 5 Hz and cannot produce a 50 Hz
latent stream. Its throughput will also be lower than its control, so compare
it at equal frames.

The comparison target itself — the explicit 38-D `root_qpos` command
(`5567809`) — had no scoreboard row until 2026-08-09. It now scores SR 0.9358
and 19.21 mm, ahead of every latent arm on both metrics. Its only surviving
checkpoint on ICE is at 7,600,078,848 frames, 52% more training than the 5B
latent rows, so the comparison favors it; the scoreboard prints each row's
frame count and the explicit row is not frame-matched. A frame-matched rerun
needs a 5B explicit checkpoint, which no longer exists on ICE — the honest fix
is to retrain the explicit arm and checkpoint at 5B, not to reinterpret this
row. The scoreboard supports explicit arms as of this change (an encoder field
of `-` selects the explicit command-interface overrides).

Two corrections worth keeping:

- The **released** SONIC checkpoint's tokenizer reads reference
  `joint_pos(29) + joint_vel(29) + anchor_ori 6D` per window frame, not the
  480-value 14-body keypoint spec. That keypoint spec is the paper's
  `sonic_bones_seed.yaml` experiment config. Joint velocity, not keypoints, is
  the encoder-input difference between the released model and our v2 default —
  which is what the `fullbody-encoder` arm tests.
- `env.data.macro_cache_device` served only the `root_qpos` macro terms and
  refused anything else. It now also serves `full_body` by carrying
  `qvel[:, 6:]` alongside `qpos[:, 7:]` (about 5.5 GB more on the 129k set),
  so the encoder-input arm is not confounded by losing the fast macro path.

## Skill-transition factorization ablation (2026-08-06)

Three fixed-duration skill objectives are running as a matched 5B-frame H200
ablation in W&B project `g1-bones-seed`, group
`skill-encoding-ablation`. This user-selected 5B cap is an explicit exception
to the default 10B recipe above. All arms retain the 380-wide root-qpos encoder
input, 256-D code plus phase, ten-step hold, and common 16,384 x 24 controller
geometry.

The encoder pretrains are `5570344` (`state_occupancy`), `5570351`
(`semimarkov_chain`), and `5570358` (`endpoint_delta`). Their dependent
controllers are respectively `5570359`, `5570368`, and `5570370`; every
controller uses an arm-specific `afterok` dependency, so no controller can
start without its matching 50,000-update encoder. At 14:10 EDT, all three
pretrains were running on H200 GPUs and all three controllers were pending on
their dependencies. Source contract SHA-256 is
`f8db5faa403aa4f7ca40b1749a630fa769cb727cb4c0d955e2df7645edc77644`;
workspace archive SHA-256 is
`c8452d8261cc8f47a07ed33daf70a5810198830aff451c2bda0856aaec0b41cc`.
The guarded launcher and full submission record live in
`experiments/campaigns/2026-08-06-bones129k-skill-encoding/`.

## BONES language-motion screen and selected ten (2026-08-05)

The earlier language demonstrations used two historical sets. `demo8` was
`Neutral_stoop_down`, two one-hand heavy-object transfers, a two-hand light
pickup, standing mug drinking, two door-opening sequences, and seated book
reading. The later `local10` used `Neutral_stoop_down`, `avoid_bump`, axe
cutting, a two-hand heavy transfer, a short light-object pickup, body check,
`burning_loop`, casual greeting, cellphone typing, and coughing. The later set
contained known poor planner cases (`burning_loop`, `avoid_bump`). The old
random-start evaluation could also produce near-end rollouts for short clips;
the new campaign does not inherit that protocol.

A replacement pool of 30 diverse motions was evaluated in one local process
against the full 129,785-motion reference arrays. Each motion received 128
stable environments (3,840 total), deterministic actions, startup/reset
randomization, no push, frame-0 starts, and SONIC terminations. The rollout-24,
gamma-0.97 3.5B checkpoint completed all environments in 69 seconds with
aggregate SR 0.9047. Twenty-four candidates reached 1.0 per-motion SR.

The selected ten are `Neutral_stoop_down`, lift-crate-and-walk, standing mug
drinking, fishing, cellphone typing, feeding birds, slow clockwise arc walking,
driving away a mosquito, casual greeting, and surrender/raised hands. All ten
scored 1.0 SR over 128 rollouts; their mean MPJPE-L is 16.73 mm and mean
MPJPE-G is 0.131 m. Exact names, full-store ranks, language goals, metrics, and
the reusable one-process launcher live in
`experiments/campaigns/2026-08-05-bones-language10-screen/`.

That original set is now the frozen v1 baseline rather than the next-run
selection. After its planner evaluation, the task-active v2 selection retained
`Neutral_stoop_down`, lift-crate, feeding-birds, slow arc walking, and
mosquito-drive-away. It replaced drinking, fishing, phone typing, greeting,
and surrender with slow straight walking, one-hand object carrying while
walking backward, opening/traversing/closing a door, injured-torso diagonal
walking, and a one-hand heavy-object high-to-low transfer. Every v2 motion had
1.0 low-level SONIC SR in the 128-rollout candidate screen.

The v2 anti-holding screen defines a hold frame as root speed below 0.03 m/s,
root angular speed below 0.10 rad/s, and joint RMS speed below 0.15 rad/s; only
segments of at least 0.20 seconds count. It rejects hold fraction above 20% or
any uninterrupted hold above 1.50 seconds. The selected set averages 2.65%
hold frames, peaks at 8.71%, and has no hold longer than 1.18 seconds. Fishing
is intentionally removed despite robust planner SR because it is 38.9% held
and contains a 4.44-second hold. Exact metrics and canonical task descriptions
are frozen in `selected10_taskactive_v2.json`. The five replacements are not
yet planner-qualified; they require a fresh data and training run.

The selected-ten development protocol is now trajectory-first. Collection,
oracle-pretrained planner evaluation, and any later planner-driven stage start
at frame 0; the older random 0--200 Phase-5 start does not apply. One local
process assigns 100 environments to each of the ten motions (1,000 total) and
runs one complete oracle-policy trajectory per environment until an official
SONIC tracking failure or `reference_finished`. Foot-position XYZ and
base-height terminations are disabled, pushes are disabled, the remaining
startup/reset domain randomization stays enabled, and policy actions are
deterministic.

The saved planner dataset is trajectory-keyed rather than row-budgeted. It
retains the ten-frame causal robot history, 256-D oracle latent target,
trajectory/rank/episode/control-step identity, termination causes and tracking
success, current expert and achieved 38-D `root_qpos`, and a valid-masked
30-frame expert root-qpos lookahead. The lookahead and root-qpos fields make
future direct-root-qpos and temporal-ensemble planner ablations possible
without recollecting simulation. Motion rank is metadata only: deployment
must supply the language goal explicitly and the evaluator fails on any
goal/reference mismatch.

The current first planner stage uses only those oracle-policy trajectories—no
DAgger or planner-driven rows. One medium flow Transformer trains for 10,000
updates with a trajectory-wise 80/20 split. Checkpoints at 2k, 4k, 6k, 8k, and
10k are each evaluated on all ten explicit goals, reporting official SONIC SR
and success-only MPJPE-L. Plateau is provisionally flagged only after two
consecutive 2k intervals move SR by less than one percentage point and
success-only MPJPE-L by less than 1 mm.

The generalized v2 collector and pipeline passed a real end-to-end smoke on
2026-08-05. Oracle collection completed all ten motions at SR 1.0 with 5,137
valid control transitions and 519 planner publications; the stored tensor
contract includes `[N,30,38]` expert root-qpos lookahead plus its validity mask.
A medium planner trained for 20 smoke updates and deterministic explicit-goal
evaluation completed all 20 rollout jobs. The 10- and 20-update smoke SRs were
both 0.4; that is only a wiring result. The canonical launcher is
`experiments/campaigns/2026-08-05-bones-language10-oracle-pretrain/run.sh`.
This selected-ten workflow supersedes row budgets for the local experiment;
the older 100-goal paper Phase-5 protocol below remains frozen historical
scope until explicitly revised.

The full seed-0 collection then completed exactly 1,000/1,000 assigned
trajectories at SONIC SR 1.0, with 100 per motion, no incomplete trajectories,
and 513,700 valid control transitions. It wrote seven sample shards. The
medium 10,000-update planner and all 50 milestone evaluations completed under
`logs/bones_language10_oracle_pretrain_seed0`. The 2k/4k/6k/8k/10k SONIC SR
curve is 0.295/0.293/0.307/0.306/0.339; success-only MPJPE-L is
47.79/53.39/52.60/49.08/46.68 mm. No checkpoint meets the plateau rule. The
10k result is best, and its 8k-to-10k gain is still 3.3 SR points and 2.40 mm.
Success remains concentrated in fishing (1.00), lift-crate (0.99), and slow
arc walking (0.96), while most other goals are near zero despite offline target
cosine 0.983. Treat this as a closed-loop precision/covariate gap; stop before
planner-driven DAgger until the next experiment is chosen explicitly.

The 10k full-horizon diagnostic rendered every goal with all early
terminations disabled, deterministic policy inference, the normal startup/reset
randomization retained, and only interval pushes removed. All ten remained
upright through 5,137 combined steps. Step-weighted full-horizon MPJPE-L is
58.28 mm, EE XYZ error is 0.322 m, and MPJPE-G is 300.10 mm. Visual inspection
confirms pose/root-command drift rather than falling: fishing stays close,
while surrender misses the raised-hands pose. Videos and per-motion metrics are
in `logs/bones_language10_oracle_pretrain_seed0/nonterminating_video/`
`update_0010000_randomized_no_push`; reproduce them with campaign
`MODE=video`.

The 10k H1 planner completed an optimizer-preserving continuation to 20k; the
extended `--num_updates` value is a total target, and milestone numbers continue
from 12k rather than restarting at one. The pre-resume latest checkpoint is
retained as `latest_pre_resume_0010000.pt`. The earlier checkpoint did not record
RNG state, so this is optimizer-preserving but not a bit-exact stochastic
continuation. The full 2k-through-20k SR curve is
0.295/0.293/0.307/0.306/0.339/0.305/0.307/0.312/0.322/0.344; success-only
MPJPE-L is 47.79/53.39/52.60/49.08/46.68/44.09/48.98/41.96/45.22/45.48 mm.
The curve does not satisfy the formal plateau heuristic because it oscillates,
but 20k gains only 0.5 SR points over 10k despite a clear held-out latent-fit
improvement. Do not extend again merely on offline loss.

The controlled H3 receding-horizon campaign completed at
`experiments/campaigns/2026-08-06-bones-language10-latent-receding/`. Both
future-publication and current-publication target materializations completed
with 49,000 matched rows and 2,900 expected tail exclusions. The runtime H3
bootstrap and exponential overlap passed a real kit-less Newton rollout; both
10k planners, all 70 deterministic randomized-no-push rollouts, and the strict
aggregate passed. SR-first ranking is future/fresh-only 0.401,
future/clipped-gated 0.396, future/exponential 0.394, current/fresh-only 0.359,
H1 0.329, current/clipped-gated 0.317, and current/exponential 0.283.
Future/clipped-gated gives the best successful MPJPE-L among the leading rows
(39.72 mm versus 49.82 mm for future/fresh-only), but redistributes success
strongly by motion. Use future-publication targets; retain fresh-only as the
SR winner and clipped/gated as the quality Pareto alternative. Reject stale-
frame overlap. The grid stays at 5 Hz/H10. Execute-5/10 Hz remains deferred
because the frozen tracker was trained with a ten-step held code and phase.

## Stable LAFAN1 5B convergence run submitted (2026-07-29)

ICE job `5548933`, continued by job `5549304`, is the matched-scale follow-up
to the 500M diagnostic below.
It runs `Isaac-Imitation-G1-Latent-v0`
(`ImitationG1LatentStableEnvCfg`) on one H200 with 16,384 environments x 12
rollout steps, minibatch 24,576, seed 0, and 25,431 PPO iterations =
4,999,938,048 environment frames. It uses corrected LAFAN1 manifest SHA-256
`d972c37c...c945db8`, the existing read-only dataset cache, and the same frozen
h10 DiffSR encoder (`5c84ff72...264ea`) used in the 500M comparison.

Run name: `stable_lafan1_diffsr_det_h10_e16384_s12_5b_seed0_20260729`; W&B
project/group: `g1-sonic-env-latent-det-ice` /
`stable-e16384-s12-5b-seed0`, run `yuf0st77`. The persistent ICE run directory
is `logs/rlopt/ipmd/Isaac-Imitation-G1-Latent-v0/`
`2026-07-29_17-06-39_wandb-yuf0st77`. Checkpoints are requested every 100M frames and
the cluster persistent-project-log bind was confirmed in the startup log. The
submitted workspace archive SHA-256 is
`560a780d23e7e4c0a1e1ea0594776bbad010601c0c70bb1b93d062a277a52be6`.
The job resolved to `ImitationG1LatentStableEnvCfg` and the SONIC actor-input
contract, completed its first PPO update cleanly at about 92.9k FPS, and used
about 73.9 GB of H200 memory.
The first persistent checkpoint landed successfully at 100,073,472 frames:
`models/model_step_100073472.pt`, 25,036,449 bytes, SHA-256
`38059555818226b6b9cc3c74b306d10a7b925151838563e8725f5338c32d4f6e`.
At that point the training logger reported mean episode length 187.91, return
13.82, and 91.6k FPS.
The matched 500M checkpoint also landed successfully:
`models/model_step_500170752.pt`, 25,036,449 bytes, SHA-256
`5a6a03059187f4cc5d81e16a9540f96f8284e4122ec6502c60ff81930ddd5a43`.
At 500,170,752 frames the logger reported mean episode length 334.96, return
26.65, and 90.6k FPS.

Job `5548933` reached 1,000,144,896 frames, then failed while writing that
checkpoint because the ICE Lustre quota had just crossed its 300 GB limit; the
resulting 732,224-byte file was rejected as truncated. The newest intact
resume point is `model_step_900071424.pt`, 25,036,449 bytes, SHA-256
`2082b79a7dd7bf7b203af5deca04fc4a98660d15c8a810c7570d35eb01d51246`.
The quota was brought back under limit by thinning only redundant periodic
checkpoint series from completed runs while retaining 500M-spaced and final
checkpoints. The 900M file was also copied locally and passed a full
`torch.load`, including policy, value, and optimizer state.

Resume job `5549304` uses the exact original workspace archive above rather
than the subsequently changed shared worktree, and loads the checkpoint through
its container-visible persistent-log path. It runs 20,853 additional
`16384 x 12` iterations = 4,099,866,624 frames, so the credited total is
exactly 4,999,938,048. Its checkpoint interval is 512,483,328 frames, exactly
one eighth of the segment, so the eighth checkpoint is the actual endpoint;
the initially healthy `5549277` continuation was cancelled before its first
checkpoint after confirming that a round 500M cadence would otherwise leave
only a 4.9B-credited final checkpoint. W&B run `h874loew`, persistent directory
`2026-07-29_21-18-25_wandb-h874loew`. Its first resumed metric arrived at
10,027,008 segment frames at 90.1k FPS, proving that the Stable config,
checkpoint restore, and optimizer continuation all entered training.

Job `5549304` reached its final iteration and therefore computed the full
4,999,938,048 credited frames, but failed while serializing the endpoint
checkpoint. `model_step_4099866624.pt` is a zero-byte file and must not be
used. The last intact checkpoint is `model_step_3587506176.pt`, SHA-256
`d7b18bf5...e9f4`, corresponding to 4,487,577,600 credited frames after adding
the original 900,071,424-frame resume point.

The first requested model-inference diagnostic used that intact checkpoint on
`walk1_subject1`, starting every environment at frame 0: seed 0, 10
environments, 700 control steps, corrected manifest/cache, frozen h10 encoder,
Newton/MJWarp, zero flow noise, and the exact submitted source snapshot. The
non-terminating, unperturbed pass retained all 7,000 transitions and measured
**26.622 mm root-relative MPJPE**, with 1.0 survival and tracking success. The
secondary strict-termination pass kept the Stable environment's configured
pushes and randomization; all ten environments again completed 700 steps with
no termination and measured **26.322 mm MPJPE**. Artifacts and the 14-second
full-horizon video are under
`logs/interface_baselines/lafan1_stable_4488m_walk1_frame0_700_20260730/`.

The most relevant earlier one-motion reference is 30.482 mm from the former
Strict environment under the same motion/frame-0/700-step/seed-0 full-horizon
geometry. The new Stable result is 3.860 mm (12.66%) lower, but this is not a
single-variable convergence claim: the evaluated checkpoint was trained on the
Stable/SONIC reward and actor-input recipe, whereas the reference checkpoint
used the former Strict environment.

The remaining broader convergence diagnostic is the all-40 corrected-motion,
1,000-step deterministic pass plus its strict secondary pass.

## Stable latent reset/phase follow-ups submitted (2026-07-30)

Two H200 follow-ups were submitted on ICE from the exact original Stable-run
workspace archive (SHA-256
`560a780d23e7e4c0a1e1ea0594776bbad010601c0c70bb1b93d062a277a52be6`).
Both use `Isaac-Imitation-G1-Latent-v0`, corrected LAFAN1, the same frozen h10
DiffSR encoder, 16,384 environments x 12 rollout steps, minibatch 24,576,
seed 0, Newton/MJWarp, and 25,431 PPO updates = 4,999,938,048 new frames.
They log to the existing `g1-sonic-env-latent-det-ice` W&B project.

- Job `5551147`, group `stable-fulltraj-continuation`, resumes the intact
  4,487,577,600-credited-frame checkpoint and trains for another
  4,999,938,048 frames (about 9.488B credited total). The command remains the
  258D z256+sin/cos phase vector held for ten control steps. Relative to the
  source checkpoint's training contract, only reset sampling changes:
  `random_reset_full_trajectory=true`, reset bounds `0/0`, and adaptive-failure
  max/mean ratio `200`. Episode length remains 500 control steps, so the result
  is a full-trajectory-reset continuation/domain-adaptation experiment rather
  than a clean from-scratch environment comparison.
- Job `5551148`, group `stable-phase-ablation`, trains from scratch for
  4,999,938,048 frames. It removes only the two appended sin/cos phase values:
  command width `258 -> 256` and `command_phase_mode=sin_cos -> none`.
  Encoder, z256 code, horizon/hold/code period 10, reset sampler, environment,
  geometry, optimizer, seed, and data remain matched to the previous Stable
  run. A one-update local Isaac/Newton smoke passed with the expected 256D
  latent observation and actor/critic widths before submission.

At submission both jobs were pending on `ice-gpu` for H200 capacity. A
scheduler preflight showed that `coe-gpu` accepts the same `coe-ice` QOS, has
the same 16-hour limit, and offered a materially earlier H200 opportunity, so
both still-unstarted jobs were moved in place to `coe-gpu` on 2026-07-30. No
training state was lost in the partition move.

The first submissions (`5551147` and `5551148`) subsequently received H200s but
failed before entering the container or creating W&B runs. The staged wrapper
inherited the repository's legacy `ice_runtime.tar` default, while this
campaign uses the verified shared immutable SIF; that tar archive does not
exist. No frames or checkpoints were produced. Corrected replacements
`5551339` (full-trajectory continuation) and `5551340` (no phase) explicitly
pin `CLUSTER_USE_SHARED_SIF=1`, the 14 GB
`isaaclab-runtime-3.0.0b2-cu130.sif`, the immutable CU130 runtime root, and
cache-copy suppression. They were pending on `coe-gpu` for priority immediately
after resubmission.

Persistent staging and logs are under
`/home/hice1/fwu91/scratch/Research/IsaacLab/`
`isaaclab_stable_followups_20260730/`. The staging archive, encoder, and resume
checkpoint hashes were reverified before `sbatch`.

## LAFAN1 Stable-vs-Strict 500M inference diagnostic (2026-07-29)

ICE already retained both requested checkpoints, contrary to the earlier
checkpoint-loss generalization:

- Stable/SONIC recipe, `Isaac-Imitation-G1-Latent-v0`: completed job `5542378`,
  checkpoint `model_step_500072448.pt` from run
  `2026-07-27_14-07-04_wandb-3xz1v8k1`.
- Former Strict recipe, `Isaac-Imitation-G1-Latent-Strict-v0`: checkpoint
  `model_step_500170752.pt` from run
  `2026-07-22_19-42-12_wandb-gha4nlhl`.

Both commands reference the same h10 DiffSR skill checkpoint; its encoder SHA-256
is `5c84ff7261c5a3aca732e370ca39f889d68a5d39fb498fa9fde72c653eb264ea`.
The local copies and inference artifacts live under
`logs/downloaded_checkpoints/lafan1_stable_vs_strict_500m_20260729/` and
`logs/interface_baselines/lafan1_stable_vs_strict_500m_20260729/`.

The decisive matched inference pass used all 40 corrected LAFAN1 motions, one
environment per motion, seed 0, 1,000 steps, deterministic tracking, and all
early terminations disabled. Both rows therefore contain exactly 40,000
body-frame samples:

| Recipe at ~500M frames | Root-relative MPJPE | Root XYZ | Joint RMSE | EE position | Velocity | Acceleration | Action change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stable/SONIC | 111.17 mm | 0.899 m | 0.303 rad | 0.914 m | 0.654 m/s | 11.884 m/s2 | 1.692 |
| Strict | 129.84 mm | 0.911 m | 0.275 rad | 0.928 m | 0.576 m/s | 8.009 m/s2 | 1.085 |

Stable reduced MPJPE by 18.67 mm (`14.38%`) in this single deterministic pass,
but it increased joint error, temporal errors, and action change. Treat this as
a diagnostic trend, not a win: prior repeated inference showed roughly 12%
relative MPJPE variation in this error regime, and the retained checkpoints
also have different training geometry (Stable `4096 x 24`, Strict
`16384 x 12`). A same-geometry Strict legacy-reset run stopped near 300M and
cannot provide a matched 500M row.

The strict-termination pass is retained as a secondary diagnostic. It reported
33.32 mm / 0.19 success for Stable and 40.79 mm / 0.24 success for Strict, but
its unequal termination-truncated sample counts make those MPJPE values
unsuitable for the headline comparison. Videos were retained from the same
non-terminating full-horizon passes.

A second Stable checkpoint at 500,170,752 frames from the new `16384 x 12`
run removes the training-geometry confound. On the same 40-motion,
40,000-sample non-terminating inference pass it measured **111.996 mm MPJPE**,
only 0.74% above the earlier `4096 x 24` Stable result and 13.74% below the
Strict row. Its other metrics were 0.919 m root XYZ, 0.295 rad joint RMSE,
0.929 m EE position, 0.627 m/s velocity, 10.787 m/s2 acceleration, and 1.428
action change. The strict pass measured 33.257 mm over its valid,
termination-truncated transitions and 0.23 tracking success. Artifacts,
including the verified 20-second full-horizon video, are under
`logs/interface_baselines/lafan1_stable_e16384_s12_500m_20260729/`.

## Latent hold-out horizon ablation submitted (2026-07-29)

New campaign
[`experiments/campaigns/2026-07-29-latent-holdout-horizon/`](../experiments/campaigns/2026-07-29-latent-holdout-horizon/README.md).
It ablates the **command interface at a fixed latent space**: every arm is
frozen against one shared h10 DiffSR encoder, and only the number of 50 Hz
control steps a published latent is held for moves — hold in {5, 1} against the
hold=10 control. This is the GR00T-style "predict H, execute k" axis, and the
complement of the 2026-07-22 latent-learning ablation (which moved the
bottleneck at fixed hold=10).

**Live jobs (ice-gpu H100, submitted 2026-07-29 ~14:21):** `5548369` (hold=5)
and `5548370` (hold=1). One 16h segment each, no resume chain (user-decided);
28,883 iterations ~ 4.26B frames at the measured ~76k fps. Both booted clean on
`ImitationG1LatentStrictEnvCfg` with checkpoints written to
`/data/holdout_store/<run_tag>/rlopt_train`.

Three facts worth carrying forward:

1. **The ablation is provably single-variable.** The launcher's emitted command
   was token-diffed against the control's recorded `command.txt`: of 34 Hydra
   overrides, 31 are byte-identical and only `latent_steps_min`,
   `latent_steps_max`, and `latent_learning.code_period` move. The encoder
   window (`hl_skill_horizon_steps=10`) is checkpoint-bound
   (`hl_skill_diffsr.py:579-584`), so it cannot drift silently.
2. **`code_period` must move with the hold.** It feeds `phase_period`
   (`ipmd.py:1248,1315`) and the sampler computes
   `phase = (phase_period - latent_steps)/phase_period`
   (`hl_skill_diffsr.py:1078`). Holding it at 10 while hold=5 would emit
   0.5..0.9 instead of 0..0.8 and desynchronize the clock. Command width stays
   258 for every arm, which is the confound the 2026-07-22 hold isolation had
   (it switched `phase-mode` to `none`, changing 258 -> 256).
3. **The hold=10 control's policy checkpoints are gone.** Only `params/` and
   `command.txt` survived under `/data/ckpt_store/<control_run_tag>/`; the
   TIMEOUT SIGKILL wiped the rest. Its **encoder survived** on the `/data` bind,
   which is what makes this campaign possible. Consequence: the control is a
   **curve-only** comparison (W&B `g1-lafan1-strict`, group
   `scaled-e12288-5b-resumable-jointfix`, ~4.56B frames, ep_len 413.8), and
   there is no hold=10 policy for oracle evaluation or a planner row until one
   is retrained.

Expectation: the 2026-07-22 isolation showed hold=1 collapsing (ep_len 2.76 vs
46.38 at 30M). hold=1 is submitted as a declared-risk arm; a collapse is a
citable negative result, not a bug to tune away. Note hold=1 also costs 2580
floats per 200 ms against the explicit packet's 670 — more bandwidth than the
baseline it is meant to beat.

## Reconstruction-family arms resubmitted with persistent checkpoints (2026-07-29)

The Study A `vqvae` and `fsq_recon` arms (h10 held code, sin/cos phase,
`code_period=10`, 66-d command) were resubmitted because the original
2026-07-22 H200 runs lost every checkpoint to the ICE TIMEOUT wipe — the Study
A launcher had no `agent.logger.log_dir`, so checkpoints lived in the
per-submission workspace. Only W&B curves survive from those runs, and no
run directory for any of the twelve ablation arms remains on ICE scratch.

Fix: `submit_lafan1_reconstruction_ablation_ice.sh` now writes
`agent.logger.log_dir=/data/latent_ablation_store/<exp_name>/rlopt_train`
for every arm, following the persistent-bind convention. New jobs also override
`SAVE_INTERVAL` to 100M (the launcher's 25M default is what filled scratch and
forced the 07-26 thinning).

**Live jobs (submitted 2026-07-29 ~15:10):** `5548489` vqvae (ice-gpu H200,
running) and `5548504` fsq_recon (coe-gpu H200; first attempt `5548500` was
pinned to the single unavailable ice-gpu H200 node and was cancelled/resubmitted
on coe-gpu). Approved H200 profile, 16,384 x 12, 23,071 iterations ~ 4.54B
frames in one segment, seed 0.

## Grouped-VQ capacity ablation prepared (2026-07-26)

New campaign
[`experiments/campaigns/2026-07-26-groupvq-capacity-ablation/`](../experiments/campaigns/2026-07-26-groupvq-capacity-ablation/),
documented as "Study C" in
[`wiki/latent-learning-ablation-plan.md`](latent-learning-ablation-plan.md).
It fixes the DiffSR spectral bottleneck at the grouped product codebook
(`gumbel_multicat`, hard straight-through) and sweeps only its two capacity
axes around the `G=64, C=128` anchor that previously tracked the continuous
deterministic latent: `G` in {16, 32, 64, 128} at `C=128`, and `C` in
{16, 64, 128, 512} at `G=64` — seven arms, seed 0, corrected LAFAN1, approved
H200 geometry, 5B cap.

Status: CPU pre-flight passed for all seven grid points (build, quantize,
checkpoint round-trip), and all seven passed the local 10M wiring gate in
`logs/groupvq_ablation/local_10m_gate_20260726/`.

**Live jobs (coe-gpu H100, submitted 2026-07-26 ~19:15):** `5540442`
g16_c128, `5540443` g32_c128, `5540445` g64_c128, `5540446` g128_c128,
`5540448` g64_c16, `5540449` g64_c64, `5540450` g64_c512. All seven confirmed
RUNNING with clean logs and falling DiffSR pretrain loss. Each needs one
continuation segment: 12,288 x 12 at ~80k FPS covers about 4.0B of the 5B cap
in one 14h segment.

Getting there took three submission rounds and exposed three defects, two
pre-existing:

1. `submit_hl_skill_pipeline_pace_2b.sh` computed `REPO_ROOT` from a fixed
   `..`, which broke when the 07-23 reorg moved it into
   `experiments/campaigns/<dated>/`. Now marker-based. This had also silently
   broken the 2026-07-22 campaign's launchers.
2. **Concurrent dataset-cache rebuild.** Seven arms sharing
   `/data/lafan1_corrected_8e95d557/g1_hl_diffsr` each ran with
   `env.refresh_zarr_dataset=true`; they rebuilt it underneath each other,
   four arms died on `FileNotFoundError`, and the cache was truncated to
   56 KB. Rebuilt to the full 1.2 GB / 40 motions by one-time job `5540413`
   from the intact NPZ source and the hash-matching manifest. Arms now always
   pass `refresh=false` and the cache is owned by
   `groupvq_ablation/build_lafan1_cache_ice.sh`. **The 07-22 launcher has the
   same pattern across twelve overlapping arms and is the likely explanation
   for its checkpoint-less `continuous_ae` arm.**
3. **`atl1-1-03-010-15-0` has a dead GPU** (`No devices were found`,
   `no CUDA-capable device is detected`) while Slurm still advertises it as
   `mix` with no drain reason, so it keeps accepting and killing jobs. Both
   launchers exclude it via `CLUSTER_SLURM_EXCLUDE`; worth a PACE ticket.

An initial six-arm submission to ice-gpu H200 (`5539991`-`5539999`) sat
PENDING for two hours without starting and was cancelled: every H200 GPU on
ice-gpu and coe-gpu was allocated (one free cluster-wide) and the sixth ice-gpu
H200 node has been admin-drained since 07-24. coe-gpu had about 40 free H100s,
so the grid was moved there at 12,288 envs x 12 steps, minibatch 18,432 --
the 07-22 `h100_e12288_lr1e3` geometry, because an 80 GB H100 cannot hold the
16,384-environment point.

Consequence: **all seven arms including `g64_c128` are re-run on H100.** The
finished 4.53B H200 `lafan1_diffsr_gumbel_multicat_b448_h10_z256_seed0` run
differs in env count and minibatch, so it is not a row of this grid and the
07-22 study remains a separate table.

Save interval for these arms is 100M frames, not the 25M of the 07-22 study,
because ICE scratch had roughly 20-40 GB of headroom. The 07-22 runs were
thinned to the same 100M granularity to make room (see below), so
plateau-checkpoint selection now resolves to 100M across both studies.

Two corrections to earlier entries on this page:

- **The 2026-07-22 latent-learning H200 ablation did run.** Eleven of its
  twelve arms reached 4,525,129,728 frames on ICE (the twelfth,
  `continuous_ae`, has no checkpoints in its run dir and needs a separate
  look). Any statement that no H200 jobs from that study were submitted is
  stale.
- **07-22 checkpoints were thinned on 2026-07-26.** ICE scratch hit 300/300 GB
  and blocked submission. With user approval, intermediate checkpoints were
  deleted from the eleven finished runs, keeping every ~100M-frame checkpoint
  plus each run's final one (181 -> 84 per run, 1,062 files, ~42 GB). No run
  directory, metric CSV, or final checkpoint was removed.

Caveat to carry into any table: bandwidth, per-group code dim, and encoder head
width all move together across this grid (encoder parameters span 2.4M-18.8M),
and grouped code usage/perplexity are currently pooled over groups rather than
reported per group.

## Data-loss incident: Slurm TIMEOUT destroys node-local output (2026-07-22)

**All three 2026-07-21 night segments produced zero retained checkpoints.**
Jobs `5525663` (BONES-SEED-91 h10), `5525664` (LAFAN1 h10) and `5525687`
(LAFAN1 history) each ran the full 15:59 walltime at ~80k fps to ~4.5B frames,
then ended in Slurm `TIMEOUT`. TIMEOUT is a hard SIGKILL: it kills the job step
before `run_singularity.sh`'s `sync_project_logs_back` copies the container's
node-local `$TMPDIR` workspace to shared storage, and the epilog then wipes
`/tmp`. A rescue job pinned to `atl1-1-03-013-8-0` and `atl1-1-03-013-13-0`
confirmed nothing survived. ~48 GPU-hours lost, encoders included. By contrast
the separate EE-chunk run `isaaclab_20260721_222745` finished its 5B *normally*
in 15.4h and synced back fine -- normal exit was always safe, which is why this
stayed hidden.

Root cause of the missing safety net: `scripts/rlopt/train_impl.py`
unconditionally reassigned `agent_cfg.logger.log_dir` to
`logs/rlopt/<algo>/<task>/<timestamp>`, so the `--train-override
agent.logger.log_dir=...` the launchers had been passing was **silently
discarded and had never once taken effect** (verified: no run-scoped checkpoint
directory has ever existed on ICE).

The first recovery attempted on 2026-07-22 routed the two 5B launchers through
`/data/ckpt_store` and `/data/pretrain_store`. That was durable, but it was an
unnecessarily launcher-specific layout rather than a repair of the cluster
runtime's normal logging contract. Jobs `5526545`, `5526549`, and `5526551`
used that workaround and were intentionally cancelled after 2-4 minutes.

Final cluster-wide fix, verified 2026-07-22:

- Every Apptainer/Singularity profile now binds a persistent shared project
  log root directly at `/workspace/isaaclab/project/logs`. Normal submissions
  derive it from the stable (pre-timestamp) `CLUSTER_ISAACLAB_DIR`; direct
  `run_singularity.sh` invocations fall back to their persistent workspace's
  `logs` directory. Checkpoint durability no longer depends on
  `sync_project_logs_back` or any shell exit handler.
- RLOpt therefore keeps its original layout with no log-directory override:
  `logs/rlopt/<algo>/<task>/<timestamp>/models/model_step_<N>.pt`.
- ICE job `5526584` tested the latest `Isaac-Imitation-G1-Latent-v0` surface
  with 64 environments, two rollout iterations, and `agent.save_interval=1`.
  While the job was still running, the central tree received valid 29,133,293
  byte `model_step_128.pt` and `model_step_256.pt` archives under
  `logs/rlopt/ipmd/Isaac-Imitation-G1-Latent-v0/2026-07-22_14-49-29/models/`.
  Both archives passed ZIP integrity checks, and the first loaded successfully
  through PyTorch with policy, value, reward-estimator, optimizer, and skill
  sampler state present; the job completed in 5:16 with exit code 0.

Supporting fixes retained:

- `train_impl.py` now honors an explicit `agent.logger.log_dir` override as the
  log root; the config default is the literal `"logs"`, so runs that do not
  override keep byte-identical behavior. Verified end-to-end with a local
  1-iteration PhysX smoke run writing into an override directory.
- Each segment's iteration count is now capped to finish *before* the wall
  (`SEGMENT_TRAIN_SECONDS` 14.5h x conservative `ASSUMED_FPS` 70k = 24,780
  iterations) so jobs exit cleanly and get a final save instead of being
  SIGKILLed.

No replacement 5B low-level job is active after those cancellations. A future
resubmission should use the ordinary central log tree through the general bind;
the `/data/ckpt_store` launcher workaround is not the desired final layout.

## BONES-SEED h10 GPU/LR wall-clock ablation (2026-07-22)

A short wall-clock convergence screen is active on ICE using the post-fix
91-motion SONIC-filtered BONES-SEED manifest and one shared 50k-update h10
encoder. Encoder job `5526697` runs first on H100. Five 500M-frame controller
jobs depend on its successful completion, so every arm consumes the exact same
encoder checkpoint from the centralized project log tree:

| Job | GPU | Envs x rollout | Actor LR / adaptive cap |
| --- | --- | --- | --- |
| `5526698` | H100 | 12288 x 12 | `1e-3` |
| `5526756` | H200 | 12288 x 12 | `1e-3` |
| `5526757` | H200 | 16384 x 12 | `1e-3` |
| `5526703` | H100 | 12288 x 12 | `6e-4` |
| `5526704` | H100 | 12288 x 12 | `3e-4` |

The critic remains at `1e-3`; minibatch size is always rollout batch / 8, and
all other PPO/environment settings are fixed. Checkpoints are written every
25M frames. Compare arms by sustained wall-clock time to matched episodic
return and episodic-length levels, not final sample count alone. The W&B
project is `g1-bones-seed-h10-gpu-lr-ablation-ice`. Launcher:
`experiments/campaigns/2026-07-23-bones-phase5-language-h200/submit_bones_seed_h10_gpu_lr_ablation_ice.sh`.

PACE submission now accepts the same restricted
`CLUSTER_SLURM_DEPENDENCY=afterok:<job>[:<job>...]` contract as the general
Slurm wrapper; arbitrary dependency expressions remain rejected.

The original H200 jobs `5526700` and `5526702` started within the same second
and exposed a second concurrency bug: the standard RLOpt run directory used a
timestamp with only one-second resolution, so both wrote into
`2026-07-22_15-42-35`. They were cancelled at 16m48s and replaced by the jobs
listed above. RLOpt training now allocates the W&B run ID before logger setup,
exports it as `WANDB_RUN_ID`, and names W&B-backed run directories
`<timestamp>_wandb-<run-id>`. Non-W&B cluster runs use
`<timestamp>_slurm-<job-id>`, with a random local fallback. RLOpt's logging
manager recognizes both the legacy bare timestamp and the new suffixed form.
Replacement jobs created distinct centralized directories
`2026-07-22_16-00-03_wandb-37ozgk4i` and
`2026-07-22_16-00-28_wandb-r74i1tcr`.

After the initial wall-clock comparison, all five 500M-frame ablation arms
were cancelled at the user's request (`5526698`, `5526703`, `5526704`,
`5526756`, `5526757`). The H200 16384-env arm sustained about 90.4k FPS,
roughly 21% above the H100 12288-env arm, although this does not establish
better sample efficiency. Production job `5526830` is now running from
scratch on one H200 with 16384 envs x 12 steps, minibatch 24576, actor/critic
LR `1e-3`, the shared completed h10 encoder, checkpoints every 25M frames,
and 25431 iterations = 4,999,938,048 effective frames. Its ICE walltime is
15:59:00; at the measured screen throughput the estimated total including
initialization is about 15.5 hours, leaving only modest scheduler headroom.

## Protocol revision: no curriculum, h10 encoders, history ablation (2026-07-21 night)

The H100 h25 resubmissions (`5525266`/`5525267`) were cancelled ~40 min in
after the user traced a confusing early-metric kink to the termination
curriculum: while `G1SonicTerminationCurriculumCfg` anneals thresholds over
50M -> 300M frames, episode length/return dip as goalposts move. Decisions:

- **Curriculum removed from the default surface.**
  `ImitationG1LatentStrictEnvCfg` (behind `Isaac-Imitation-G1-Latent-v0`)
  now has `curriculum = None`; thresholds are strict from frame 0. The
  anneal remains available on the opt-in SONIC surface only. Caveat noted at
  the time: the 2026-07-19/20 investigation added the anneal precisely
  because strict-from-scratch spends the early budget on ~5-step episodes.
- **Encoder horizon moved from h25 to h10** (matches the 5 Hz planner
  publication interval: one latent per 10-step chunk at 50 Hz). Both 5B
  low-level jobs retrain their skill encoders at h10 with the full previous
  pretrain contract (50k updates, 0.9/0.1 split, groups/categories 64/128,
  gumbel_hard=true).
- **Active jobs:** `5525663` (BONES-SEED-91 h10, run tag
  `bones_seed_91_strict_h10_..._nocur_...`, ~77k fps) and `5525664`
  (corrected LAFAN1 h10, `lafan1_strict_h10_..._nocur_...`, ~83k fps), both
  5B, scaled config, Newton, H100.
- **History ablation:** new task `Isaac-Imitation-G1-Latent-History-v0` =
  strict surface + `G1SonicLatentObservationCfg` (10-step proprio
  histories, SONIC actor input set) paired with
  `G1ImitationLatentSonicRLOptIPMDConfig` on the local optimizer contract,
  so ONLY the observation/history contract differs from `Latent-v0` — a
  low-cost recurrent-policy stand-in. Running on corrected LAFAN1 as job
  `5525687` (run tag `lafan1_history_strict_h10_..._nocur_...`, W&B group
  `history10-h10-e12288-5b-jointfix-nocur`); compare against `5525664`.
  Its skill encoder is NOT retrained: it is the checkpoint `5525664`'s
  in-job pretrain produced, pulled off the compute node mid-run and staged
  to `pretrain_store/` for both run tags (sha256 `b3e23e0a...`, see the
  `PROVENANCE.txt` beside it), so the ablation pair shares tensor-identical
  encoders and differs only in policy-side history. (A first submission
  `5525682` that would have retrained its own encoder was cancelled during
  pretrain for exactly this reason.)
- **Planned next (not yet submittable):** planner training on top of the
  h10 low-level controllers once they finish and pass oracle audits — a
  language-conditioned planner for BONES-SEED (per the Phase-5 multigoal
  language workflow) and the standard no-language planner for LAFAN1. The
  h10 encoders trained here are the planner-side prerequisite; planner jobs
  must wait for qualified low-level checkpoints and the streamed-vanilla
  equivalence gates where applicable.

## Post-fix 5B resubmissions on Blackwell (2026-07-21 evening)

With the joint-order fix merged (PR #24, `900c66c`), both from-scratch 5B
low-level runs were resubmitted on `Isaac-Imitation-G1-Latent-v0` at the
scaled config (12288 envs x 12 steps, minibatch 18432, njmax=320/nconmax=40,
Newton) as ICE jobs `5525240` (BONES-SEED-91,
`experiments/submit_bones_seed_sonic_5b_resumable_ice.sh`) and `5525245`
(corrected LAFAN1, new `experiments/submit_lafan1_5b_resumable_ice.sh`).
The first submissions (`5525240` BONES, `5525245` LAFAN1) targeted the
Blackwell partition (`ice-bw-gpu`, 3x16 `rtx_pro_6000_blackwell`) as a
first Blackwell-stack test. **Result: the stack works on Blackwell**
(kernels, Newton solver init, 50k-update pretrain, and training start were
all clean) **but the cards are 48 GB (47.38 GiB visible), not 96 GB**, and
the 12288-env scaled config hit CUDA OOM at the first advantage pass in both
jobs. Both were resubmitted on `ice-gpu` H100 80 GB as jobs `5525266`
(BONES) and `5525267` (LAFAN1), reusing the completed Blackwell pretrains
(~35 min for 50k updates each) via the new pretrain store.

Two ICE plumbing facts learned in the process:

- Although `ice-bw-gpu` allows 18h, every ICE QoS
  (`coe-ice`/`coc-ice`/`pace-ice`) sets `MaxTRESMins gres/gpu=960`, capping
  any 1-GPU job at 16h regardless of partition — the initial 17:59:00
  submissions pended with `QOSMaxGRESMinutesPerJob` and were reduced in
  place via `scontrol update job`. Launchers default to 15:59:00.
- Every archive submission runs in its own `isaaclab_<timestamp>/` workspace
  dir and `run_singularity.sh` syncs job logs back into that dir only, so
  nothing accumulates under the stable `isaaclab/` root and a naive
  fixed-path resume scan never finds prior segments. The only host path all
  jobs share read-write is `CLUSTER_DATA_DIR` (bound at `/data`). Both 5B
  launchers therefore scan all `isaaclab*/logs/rlopt/ipmd/<TASK>/*/command.txt`
  (exp_name-filtered), keep cumulative-frame state in
  `<data>/resume_store/<RUN_TAG>/`, and stage the resume checkpoint and the
  skill-encoder pretrain into `/data`-visible stores that the next segment's
  container can read.

Pretrain was prolonged to the full previous contract: 50k skill-encoder
updates (the 5000 in the earlier launcher was a qualification-only value),
0.9/0.1 trajectory split (diffsr default), categorical groups/categories
64/128, and `--gumbel-hard` passed explicitly because the pipeline's argparse
default (False) silently overrides the diffsr-side default (True).

Both launchers keep checkpoints in the shared central location
(`logs/rlopt/ipmd/Isaac-Imitation-G1-Latent-v0/<timestamp>/`); resume
detection filters run dirs by `agent.logger.exp_name=<RUN_TAG>_oracle_low_level`
recorded in each run dir's `command.txt`, which restores per-run isolation --
required because these two jobs share one task id, and because the
invalidated pre-fix sanity checkpoints (jobs `5524387`/`5524390`) live in the
same tree.

Also removed: the `Isaac-Imitation-G1-Latent-Strict-v0` gym registration and
every repo reference to it. `Isaac-Imitation-G1-Latent-v0` is the single name
for the latest preferred latent surface; `Latent-Sonic-v0` and
`Latent-Legacy-v0` remain explicit opt-ins.

## Open Blocker: Newton joint-order leak (2026-07-21)

Cross-backend verification found that the expert command observations and the
action offset are resolved from the *live* articulation order instead of the
pinned canonical list. PhysX and Newton differ in 27 of 29 joint slots, and the
pinned list is the PhysX order, so both leaks are no-ops under PhysX and active
under Newton. Every Newton-trained checkpoint therefore encodes a
Newton-specific joint permutation, including the `reward_input` term that feeds
the IPMD reward and discriminator.

Confirmed in both directions on `L1_strict/model_step_992870400.pt`: removing
the mismatch on PhysX raises survival from 67/500 to 323/500 steps; injecting it
on Newton drops survival from 500/500 to 111/500. Joint tracking error is
~0.43-0.52 rad whenever mismatched and ~0.11-0.24 rad when matched.

**Fixed on 2026-07-21.** The command terms and the action offset are pinned,
the causal planner frame is pinned, and a latent double-scatter in
`batch_csv_to_npz.py` (live since 2026-07-16, no data affected) was removed.
The index contract now reports no leaks on either backend and the regression
test covers every command term.

**Existing Newton checkpoints are invalidated** and now fail on Newton too
(113/500 steps). `compare_policy_reference.py --emulate_joint_order_from` is a
diagnostic-only shim that restores them exactly; retraining the low-level
controllers is the real remedy.

Source reference NPZ/Zarr data is name-bound and **safe**. Policy-produced
artifacts are not: rollout NPZ state arrays, planner sample rows, and the
skill-encoder latent space are all Newton-permuted with no ordering metadata.

Two further bugs were found and fixed on 2026-07-21 while chasing the residual
gap:

- **Stale derived state after reset (both backends).** Both reset events called
  `asset.update(dt=0.0)`, which does not advance `_sim_timestamp`, so Isaac
  Lab's lazily cached body-frame buffers were never recomputed. `base_lin_vel`
  and `base_ang_vel` are policy observations, so the first observation after
  every reset came from the pre-reset state — stale under PhysX, zeros under
  Newton — throughout all training to date.
- **PhysX solver iterations.** The USD spawn copied the URDF importer's
  `articulation_props` and overrode the asset's requested 32/1 with a generic
  8/4. The override is removed; the asset now governs, verified on the live
  stage.

Neither closed the transfer gap, which is the point: with ordering matched,
Newton survives fully at 0.126 rad joint error while PhysX falls at 5.36 s with
0.242 rad. The gap is a genuine dynamics difference.

**Decision: if we randomize, we randomize for every experiment**, so the
protocol is re-frozen on the randomized event config rather than randomizing a
subset. This invalidates existing qualification artifacts, so sequence it with
the retraining already forced by the joint-order fix.

See [Sim2Sim Backend Verification](sim2sim-backend-verification.md) for the
audit tooling, evidence tables, and recorded-data status, and
[Sim2Sim Dynamics Gap and Randomization](sim2sim-dynamics-gap-and-randomization.md)
for the gap analysis and the randomization tiers.

## Research Question

We are testing whether a learned latent skill command is a better high-level
planner interface than the explicit action/state chunks used by current
humanoid VLA systems.

The main questions are:

1. Can a causal high-level planner command a frozen whole-body controller
   without future expert state leaking into its input?
2. Does the latent interface make the planner easier to learn or more
   data-efficient than an explicit full-body chunk?
3. Does the latent interface require a smaller planner to reach the same
   closed-loop performance?
4. Does language-conditioned planning work across diverse BONES-SEED motions?

## Frozen Main Comparison

The main paper comparison has exactly two planner rows:

| Interface | High-level output | Publication rate | Frozen low-level consumer |
| --- | --- | ---: | --- |
| DiffSR latent | 256-value latent code | 5 Hz | DiffSR latent tracker at 50 Hz |
| Explicit packet | Ten consecutive vanilla full-body commands, 670 values | 5 Hz | The same qualified vanilla tracker used by the direct ceiling, at 50 Hz |

The planner input is ten causal robot frames (`10 x 93`) plus an explicit task
input. Phase 5 adds the same 384-value MiniLM language embedding to both rows.
Future reference state is allowed only in oracle targets, labels, and metrics.
It is never a deployed planner input.

The direct vanilla tracker receiving a fresh expert command at 50 Hz is a
low-level ceiling, not a planner baseline. End-effector chunks and other
command styles are diagnostics or appendix work; do not start a combinatorial
command sweep.

The authoritative protocol is
[Causal High-Level Interface Paper Plan](causal-interface-paper-plan.md).

## What Is Implemented and Verified

### Causal planner path

- Both main rows use the same ordered `10 x 93` achieved-robot history.
- The planner does not use `current_achieved_macro_transition_batch`, future
  reference state, reference rank, or reference cursor as deployed input.
- Language goals are supplied explicitly and checked against the selected
  named motion.
- Commands renew independently per environment, including after asynchronous
  resets.
- M3 disables tracking-error terminations but keeps `base_too_low`; a fall is
  defined identically for both interfaces.
- Evaluators retain success, survival, MPJPE, root, joint, end-effector,
  smoothness, velocity, acceleration, action-change, termination-cause, and
  planner-latency metrics.

### Explicit tracker equivalence

- The direct and streamed vanilla paths load the same strict frozen policy
  state and use the same ordered actor inputs.
- The streamed packet consumes slots 0 through 9 exactly once.
- The BONES-SEED certificate passed all packet phases, asynchronous renewal,
  and policy immutability. Maximum command and action differences were
  `3.02e-7` and `1.31e-6`.

### Planner families and scaling tools

Three continuous planner families are implemented with matched Transformer
parameters:

- flow matching;
- clean-target diffusion;
- deterministic chunk prediction.

The scaling reports keep demonstration-only and rollout-fine-tuned results
separate and record actual parameters, output bandwidth, and measured planner
latency. They answer both performance at the same size and the smallest tested
size that reaches a fixed performance target.

### Reproducibility gates

- Data, checkpoints, caches, language tables, workflow sources, and stage
  artifacts are hash-bound.
- Phase 4 and Phase 5 have guarded launchers, exact seed grids, stage records,
  strict aggregators, and no-overwrite behavior.
- Final paper release assembly is intentionally blocked until both complete
  audited aggregates exist.

## Current Experiment Status

### Newton joint-order bug found on an unmerged branch (2026-07-21)

**Status: all today's Newton jobs cancelled; fix not yet reviewed/merged.**

A parallel, unmerged branch (`sim2sim-verification-transfer-547ca5`, mirrored
to `origin/fix/migration`, last commit 2026-07-21 13:56 EDT) found that G1's
expert-command observation terms (`expert_motion` in `policy`, `critic`,
`expert_state`, `expert_goal`, `expert_window`, `reward_input`, plus
`expert_state.joint_pos`/`joint_vel`) and the action-offset default in
`randomize_joint_default_pos` were built from the *live* per-backend joint
enumeration instead of the pinned canonical `G1_29DOF_ISAACLAB_JOINT_NAMES`
order. This is a no-op under PhysX (which the canonical list already matches)
but active under Newton, where 27 of 29 joint slots differ from PhysX's
ordering. Every job submitted today used `physics=newton_mjwarp`.

Their own behavioral confirmation (`L1_strict/model_step_992870400.pt`,
Newton-trained, ~993M frames, `walk1_subject1`, seed 0): evaluating with a
mismatched joint/command order collapses survival and roughly doubles joint
tracking error (0.517/0.431 rad, 67/500 and 111/500 survived) versus a matched
order (0.240/0.110 rad, 323/500 and 500/500 survived) in both transfer
directions. Because `reward_input.expert_motion` carries the same leak, they
flag training itself as suspect beyond cross-backend deployment, not only a
deployment-transfer issue — though the same checkpoint evaluated Newton-native
(matched to how it trained) still survived 500/500, suggesting the permutation
is at minimum a consistent, learnable relabeling within one backend rather
than pure noise.

All 5 jobs running at the time this was discovered were cancelled rather than
left to keep training on the pre-fix ordering:
`5524182`/`5524183`/`5524338` (SONIC VRAM ablation v1/v2/v5),
`5524342` (BONES-SEED-91 5B resumable, segment 1, at 1.29B/5B frames),
`5524390` (LAFAN1 hardcoded-default sanity, at 900M/1B frames — 90% done).
`5524387` (the LAFAN1 *scaled* sanity check, exactly reproducing bn931wny's
config) had already completed before cancellation, reaching `ep_len=288.68` /
`r_ep=15.76` — beating bn931wny's `244.18`/`13.11` — but this result is
likewise pre-fix and should be treated as provisional. Between the two
completed/near-complete sanity arms, 8192 envs x 12 rollout steps reached
comparable quality to 4096 x 24 while finishing faster (~4.5h at ~65k fps vs.
~5.5h+ at ~46k fps for the same 1B frames) — a reasonable default scale
choice to carry forward once re-validated post-fix.

Full detail, the audit tool
(`scripts/audit/dump_backend_index_contract.py`), and the fix commits are on the
unmerged branch; see `wiki/sim2sim-backend-verification.md` there. Not yet
reviewed for merge into `main`, and not yet reconciled against the
Strict/legacy-default reversal above (both branches diverged from a shared
ancestor and have not been compared for conflicts).

### Interface-ablation tracker arms submitted (2026-07-21, late evening)

**Status: all four arms submitted to ICE as jobs `5525739` (FB chunk,
running), `5525740` (EE chunk), `5525741` (FSQ), `5525742` (SONIC joint).**
Re-invoke each launcher to chain the next 16h segment; each refuses once its
5B cap is reached.

Per-step renewal decision (user, 2026-07-21): SONIC re-encodes its latent
every control step over the sliding future window, unlike our held-z
contract (which is the planner-friendly design). New pipeline knob
`--latent-hold-steps` (defaults to `--horizon-steps`, preserving every
existing run's behavior) sets `agent.ipmd.latent_steps_min/max`
independently of the encoder window. Both SONIC-flavored variants submit
with `--latent-hold-steps 1 --phase-mode none` (phase would be a constant at
hold=1; SONIC has no phase channel), so their latent command dim is 256, not
258. The main latent arm keeps the held-z contract.

Per the user's ablation-study decisions (plateau qualification instead of a
survival gate, task-index planner input, frame-0/~700-step eval, 5B budget
for every tracker, Study 1 on LAFAN1 — see
[Ablation Experiment Plan](ablation-experiment-plan.md)), four additional
LAFAN1 tracker arms are ready to join the running latent 5B job (`5525267`):

- `Isaac-Imitation-G1-Strict-v0` (new): vanilla observation/agent contract on
  the strict latent surface's protocol deltas (pelvis anchor, strict SONIC
  terminations, [0, 200] starts), so explicit-interface trackers differ from
  the latent arm only in command space.
- `experiments/submit_lafan1_chunk_tracker_5b_resumable_ice.sh`: FB-chunk and
  EE-chunk arms (`agent.command_space`, held 10-step chunks via
  `env.command_hold_steps=10`), plain `train.py`, resumable segments.
- `experiments/submit_lafan1_latent_variant_5b_resumable_ice.sh`:
  `VARIANT=fsq` (FSQ skill encoder; `FSQ_LEVELS` defaults to the
  SONIC-release token space, 64 dims x 32 levels ~= 320 bits per 5 Hz
  command, with an overflow-safe `FSQQuantizer` fix in RLOpt) and
  `VARIANT=sonic_joint` (`agent.ipmd.hl_skill_finetune_enabled=true`, PG +
  recon encoder finetuning; resume-safe via
  `hl_skill_command_sampler_state_dict`).

A 1-iteration PhysX smoke of the new Strict task with `ee_trajectory` passed
(actor in_keys confirmed as the expert-window EE terms).

### Phase 3: low-level protocol and causal planner code

**Status: complete as a code and local behavior gate.**

One-motion closed-loop experiments establish that a causal planner can command
both the latent and explicit interfaces. These runs are diagnostics, not paper
evidence across motions.

### SONIC default and policy-contract decision (2026-07-20)

**Status: code default, not yet re-validated at the new scale.**

With ICE H100 (and now H200) single-GPU access, the compute-scale objection
that paused the full SONIC surface on 2026-07-20 no longer applies at the
intended budget: 100k PPO iterations at 8192 envs x 12 rollout steps is
~9.83B (~10B) frames, matching the release's own convergence criterion
("after 100K iterations") on a single GPU instead of 64+. Decision:

- `Isaac-Imitation-G1-Latent-v0` (the SONIC surface) is the confirmed
  default latent task, not a paused/candidate one.
- `Isaac-Imitation-G1-Latent-Strict-v0` (legacy scaffolding + pelvis anchor +
  annealed strict terminations), briefly floated as the 2026-07-20 candidate
  default, is now DEPRECATED — kept only to reproduce runs already started
  on it.
- The default policy contract for the SONIC task is now the exact public
  release optimizer (`sonic_release_optimizer=True`: actor lr 2e-5, joint
  grad clip 0.1, init std 0.05, 6-layer SiLU MLPs, running input
  normalization), not the locally-validated small-scale contract — the
  release contract needs release-scale iteration counts to leave the flat
  regime, and 100k iterations now supplies that on one GPU.

Submitted 2026-07-20: the VRAM/throughput ablation
(`experiments/submit_sonic_latent_vram_ablation_ice.sh`, corrected LAFAN1,
2B-frame cap each) as ICE jobs `5523769` (v1, 8192 envs x 12 steps — njmax
95/nconmax 18), `5523770` (v2, 12288x12 — 143/27), `5523771` (v3, 16384x12 —
190/36), and `5523772` (v4, 12288x24 — 143/27); and the BONES-SEED
SONIC-latent job (`experiments/submit_bones_seed_100_sonic_latent_ice.sh`,
91/100-motion SONIC-exclusion-filtered manifest, L1 scale 8192x12, 3B-frame
cap, njmax 288/nconmax 32) as ICE job `5523773`.

**VRAM ablation result (2026-07-20): v3 and v4 failed within 10 minutes,
closed as-is (no resubmission).**

- v3 (16384 envs, `5523771`): genuine CUDA OOM, not a solver issue — 79.18 GB
  capacity, 76.82 GB already in use, failed allocating another 3 GB. The
  SONIC release network (6-layer [2048,2048,1024,1024,512,512] with running
  input normalization) plus rollout buffer at 16384 envs exceeds one H100's
  80 GB.
- v4 (12288 envs, rollout=24, `5523772`): contact-solver overflow, not VRAM —
  the proportional njmax/nconmax extrapolation (143/27) was too low for the
  longer 24-step rollout; the log shows repeated `nefc overflow` requests up
  to 196, and the run hard-crashed rather than NaN'ing. Confirms the
  extrapolation caveat noted in the ablation script: njmax/nconmax scaling by
  env count alone does not hold when rollout length also changes.
- v1 (8192 envs) and v2 (12288 envs) ran cleanly for 8.5+ h with no
  overflow/OOM. Per user direction, this is treated as sufficient signal for
  this ablation round: **12288 envs x 12 rollout steps fits one H100 and is
  the largest validated point; 16384 envs does not fit at this policy size.**
  No further arms were resubmitted.

**Correction (2026-07-21): v1/v2 "success" was contaminated by njmax
under-provisioning, not a clean result.** Log audit found `nefc overflow`
warnings throughout both "successful" arms: v1 (njmax=95) logged **7.4
million** overflow events over ~9.5h; v2 (njmax=143) logged 59,027. Peak
requested njmax was ~230-245 in BOTH arms regardless of env count (245 at
8192 envs, 232 at 12288), while the BONES-SEED job running concurrently at
njmax=288 logged zero overflow. This means njmax/nconmax is a per-step
contact-complexity budget driven by the SONIC env's domain
randomization/push events and early strict-from-scratch falling — NOT
something that scales with `num_envs`, contradicting the original
proportional-scaling assumption. All four VRAM-ablation arms were cancelled
and resubmitted with a fixed njmax=320/nconmax=40 (headroom above the
288/32 that measured zero overflow) as ICE jobs `5524182` (v1),
`5524183` (v2), `5524184` (v3), `5524185` (v4) — v3 (16384 envs) is expected
to OOM again since that failure was VRAM-related, not njmax-related.

**ICE partition walltime caps (2026-07-21): confirmed hardcoded, not a QoS
setting.** `scontrol show partition ice-gpu` shows `MaxTime=16:00:00`; `sinfo`
confirms every GPU-bearing PACE partition is capped the same way:
`ice-gpu`/`coc-gpu`/`coe-gpu`/`pace-gpu` at 16h, `ice-bw-gpu` at 18h. None of
the attached QoS (`coe-ice`, `coc-ice`, `pace-ice`) define a `MaxWall`
override, so the partition cap governs regardless of QoS choice — there is
no "long" GPU QoS on this cluster (unlike Skynet). Incidental find: H200s
are already in `ice-gpu` (`gres/gpu:h200=48`), so H200 access needs no
separate partition/QoS, just `--gres=gpu:h200:1`.

**Resumable BONES-SEED-91 SONIC-latent job (2026-07-21), 5B-frame cap.**
Since RLOpt's `save_model`/`load_model` restores weights + optimizer state
but not the frame/iteration counter (`frames_processed` resets to 0 on every
fresh `agent.train()` call), a walltime-capped job needing >16h of training
must be split into segments, and per-segment checkpoint filenames
(`model_step_<N>.pt`) are local to that segment rather than a global total.
`experiments/submit_bones_seed_sonic_5b_resumable_ice.sh` tracks true
cumulative frames itself in a remote state file keyed by the last-counted
checkpoint (crediting each segment's own contribution exactly once), and
computes the next segment's `--max_iterations` from the remaining budget;
`train_hl_skill_pipeline.py` gained `--train-checkpoint` to pass a low-level
checkpoint through to `train.py --checkpoint` for the resume case.
Re-invoking the script drives the chain forward; it refuses to resubmit once
5B frames are reached.

**v5 arm added, then the whole SONIC-default premise questioned (2026-07-21).**
v5 (`5524338`) re-tests the code's own hardcoded default shape (4096 envs x
24 rollout, mini_batch_size 24576 to match `rlopt_ipmd_cfg.py`'s literal
`4096 * 24 // 4`) under the SONIC release-optimizer contract at the
validated-safe njmax=320/nconmax=40, per the hypothesis that this exact
shape explains why earlier runs performed well. v3 and v4 were also
resubmitted at njmax=320/40 but both hit genuine CUDA OOM again (`5524184`,
`5524185`) — 12288 envs x 24 rollout doubles the collector buffer versus
v2's 12288x12 (which fits), so both are real VRAM-ceiling results, not a
solver misconfiguration; v1 (`5524182`) and v2 (`5524183`) are running
cleanly.

Pulling the actual W&B config for the run the user was comparing against
(`bn931wny`, project `g1-lafan1-strict`, group `ice3-l1-novideo`) revealed
the "L1" baseline never used the SONIC surface or release-optimizer contract
at all: `env_name=Isaac-Imitation-G1-Latent-Strict-v0`, `num_envs=8192`,
`collector.frames_per_batch=98304` (12 rollout steps),
`loss.mini_batch_size=12288`, `policy.num_cells=[512,256,128]` with
`activation_fn=elu` and `normalize_input=False` — the legacy/local optimizer
contract, not the release SiLU/[2048...512] one. It reached
`episode/length=244.18` and `episode/return=13.1`, well above anything the
new SONIC release-optimizer contract has produced so far.

**Reverted (2026-07-21): the 2026-07-20 "make SONIC the default" decision is
undone.** `Isaac-Imitation-G1-Latent-v0` resolves to the Strict/legacy
surface again (`_LATENT_STRICT_TASK_KWARGS`); `Isaac-Imitation-G1-Latent-Strict-v0`
is its back-compat alias. `Isaac-Imitation-G1-Latent-Sonic-v0` is opt-in
only and no longer aliased as `v0`.
`G1ImitationLatentSonicRLOptIPMDConfig.sonic_release_optimizer` reverts to
`False`. Every downstream script/pixi task that references the
`Isaac-Imitation-G1-Latent-v0` alias (interface_baselines scripts,
`smoke-ipmd`, etc.) automatically now gets the Strict/legacy surface again
by design — that is the whole point of using the floating alias rather than
a hardcoded surface name. The one exception fixed explicitly:
`experiments/submit_sonic_latent_vram_ablation_ice.sh` now targets
`Isaac-Imitation-G1-Latent-Sonic-v0` directly, since it specifically studies
the SONIC surface regardless of which surface is "default".

**Default-reversal sanity check submitted (2026-07-21).**
`experiments/submit_lafan1_strict_default_sanity_ice.sh` runs two arms on
`Isaac-Imitation-G1-Latent-v0` (corrected LAFAN1, ~1B frames,
njmax=320/nconmax=40): `scaled_e8192_r12` (ICE job `5524387`) exactly
reproduces bn931wny's config (8192 envs x 12 steps x minibatch 12288) as the
actual correctness check; `hardcoded_default_e4096_r24` (ICE job `5524390`)
tests the code's literal default shape (4096 envs x 24 steps, minibatch
24576) as a second, unvalidated data point on whether scale matters. Neither
has reported results yet.

Both running BONES-SEED jobs (`5523773` 3B and `5524188` 5B segment 1) were
cancelled and resubmitted with `TASK=Isaac-Imitation-G1-Latent-Strict-v0`
(matching the actual L1 config above) instead of the SONIC default; the
policy contract follows automatically since `Latent-Strict-v0`'s task kwargs
already route to the legacy-style `G1ImitationLatentRLOptIPMDConfig`, not
the Sonic one. `experiments/submit_bones_seed_100_sonic_latent_ice.sh` (the
3B one-shot) is now marked superseded/reference-only.
`experiments/submit_bones_seed_sonic_5b_resumable_ice.sh` is the live
launcher; its first segment under the corrected task is ICE job `5524342`.

### Non-paper BONES-SEED SONIC latent training

**Status: debugged locally; no active ICE job.**

Jobs `5523561`, `5523570`, `5523578`, and `5523588` are stopped or failed and
are not training results. The h25/z256 encoder checkpoint was retained, but the
Newton low-level run at the official flat-locomotion value `njmax=95` produced
NaN returns. A first-rollout finite-value trace ruled out MPJPE, the skill
encoder, and the actor: the latent and initial policy outputs were finite, then
Newton state and six independent reward terms became non-finite after contact
constraint overflow. The failing sample was `ab_bicycle_001_A359` near frame
20. In its first 200 frames, 25 of 32 body origins are below 5 cm; corrected
LAFAN1 has at most 3. An 8,192-environment LAFAN1 control had zero overflows,
while BONES-SEED at `njmax=95` had 951 in one rollout and requested up to 236
constraint rows.

A reduced debug manifest containing only `ab_bicycle_001_A359` and
`crawl_ff_loop_180_R_001_A214` isolated the interacting Newton capacities
without altering either motion. At 2,048 environments, `njmax=264` and
`nconmax=31` still overflowed (268 constraint rows requested), whereas
`272/32` and `288/32` each passed 30 rollouts across seeds 0, 1, and 2 with no
constraint/contact overflow or NaN. The retained setting is `288/32` to keep
20 rows of headroom above the observed request. Relative to the borderline
`264/31` setting, it reduced steady throughput by 0.87%; at 2,048 environments
GPU memory increased by 96 MiB (2.3%) compared with `95/18`.

The full 100-motion Newton run at `288/32` then completed 20,054,016 local
frames in 186.4 seconds with no overflow or NaN. Steady throughput was about
108--110 thousand frames/s, observed GPU use was 34,916 MiB, and the final
metrics included mean episode length 19.97 and mean episode reward 0.5444.
This qualifies the capacity change for local testing; it is not a training
result or a paper qualification. A separate PhysX local run was also finite
through 20,054,016 frames. No replacement was submitted at the user's request.

### Phase 4: corrected LAFAN1, no language

**Status: low-level prerequisites active; planner paper grid not submitted.**

Last verified on 2026-07-16:

| Purpose | Slurm job | State at last check |
| --- | ---: | --- |
| Corrected-LAFAN vanilla low level | `3500993` | Running |
| Corrected-LAFAN DiffSR low level | `3503434` | Running |
| Strict paired qualification | `3503441` | Waiting on both jobs |

The guarded Phase 4 planner grid remains blocked until both controller audits
and the matching streamed-vanilla certificate pass. The future planner grid is
fixed to seeds `0, 1, 2`, all 40 corrected motions, and sample budgets
`1k/10k/50k`.

Detailed chronology:
[LAFAN1 From-Scratch Interface Comparison](lafan1-from-scratch-comparison.md).

### Phase 5: BONES-SEED language study

**Status: low-level qualification passed; first planner preparation attempt
failed before training.**

The corrected, provenance-complete 100-motion BONES-SEED tree and separate
latent/vanilla caches passed their audits. Qualification job `3512041`
completed successfully:

| Controller | Strict success | Required |
| --- | ---: | ---: |
| Direct vanilla | 0.90 | 0.80 |
| DiffSR latent | 0.84 | 0.80 |

The selected skill checkpoint is tensor-bound to the encoder embedded in the
qualified latent checkpoint. The persistent qualification root is:

```text
logs/interface_baselines/bones_seed_100_low_level_qualification_seed0_retry_20260716
```

The first guarded three-seed planner chains were:

| Seed | Prepare | Rollout | Fine-tune | Final eval | Summarize |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `3512092` | `3512093` | `3512094` | `3512095` | `3512096` |
| 1 | `3512097` | `3512098` | `3512099` | `3512100` | `3512101` |
| 2 | `3512113` | `3512114` | `3512115` | `3512116` | `3512117` |

All three prepare jobs failed after about 2 hours 16 minutes. Each had written
98 explicit demonstration chunks, but no latent chunks or complete prepare
stage record. The two shared failure signals were:

1. repeated `OSError: [Errno 28] No space left on device` from compute-local
   job storage; and
2. the fixed collection limit ended with four motions below the old
   1,000-row-per-goal target:
   `ab_bicycle_001_A359`, `crawl_ff_loop_180_R_001_A214`,
   `jump_sideway_135_001_A021`, and
   `sitting_legs_bend_arms_front_loop_001_A030`.

The dependent rollout arrays show `DependencyNeverSatisfied`, so no incomplete
preparation data reached planner training or evaluation. These failed chains
are not paper results and must not be resumed without auditing the partial
artifacts.

**2026-07-23 latent-only H200 language pilot submitted.** The new H200
controller is intentionally being tested outside the paper comparison: it was
trained on the 91-motion SONIC-filtered manifest and has no matched qualified
vanilla checkpoint. The campaign therefore sets `INTERFACES=latent_skill` and
does not run a full-body baseline, controller comparison, or GPU/parameter
ablation. It uses ten goals common to the 91-motion and fresh 100-motion trees,
150 demonstration rows plus 150 planner-rollout rows per goal, ten same-goal
rollout environments, and the fixed 500-step evaluation.

The guarded dependency chain is:

| Stage | Slurm job |
| --- | ---: |
| Prepare | `3560697` |
| Rollout array | `3560698` |
| Fine-tune | `3560699` |
| Final eval array | `3560700` |
| Summarize | `3560701` |

Output root: `logs/interface_baselines/bones_seed_h200_language_preliminary_seed0_20260723`.
The persistent record is on Skynet at the corresponding `cluster_submission.json`.
At submission verification, all five jobs were `PENDING`; no stage had begun.
The H200 checkpoint SHA-256 is
`6765a324a840b33a84f9a0b5a817c60303979bbec7a36ebc31242086d61d1572` and the
encoder binding passed all 14 tensor checks. This run is
`preliminary_unqualified=true` and cannot enter the paper aggregate.

**2026-07-23 local ten-goal baseline campaign added.** To obtain the two basic
planner baselines without waiting on cluster queues, the campaign
`experiments/campaigns/2026-07-23-bones-phase5-language-local10/` runs the same
shared Phase-5 workflow entirely on the local workstation: latent-only, the
same ten goals and frozen H200 checkpoints as the H200 pilot, a derived
ten-motion subset manifest (`data/bones_seed_phase5_local10/`, source-hash
recorded) so Isaac only loads the needed references, 150 demonstration plus
150 rollout rows per goal, and 500-step episodes. Its two deliverables are the
demonstration-pretrained planner and the rollout-finetuned planner under
`logs/interface_baselines/bones_seed_phase5_local10_seed0/latent_skill/`. As
part of this, the canonical stage driver
`run_bones_seed_multigoal_language_comparison.{py,sh}` (now under
`experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/`)
was made location-independent: it resolves its sibling workflow scripts and the
repository root from its own path instead of the hardcoded pre-reorganization
`experiments/interface_baselines/` prefix, which had left the pipeline
unrunnable after the 2026-07-23 script reorganization. Like the H200 pilot,
this local run is `preliminary_unqualified=true` and is not paper evidence.

Data-budget interpretation is important: one saved row is one 5 Hz planner
decision containing a ten-frame 50 Hz command chunk. The failed configuration
requested 100,000 demonstration rows plus 100,000 rollout rows per interface,
then fine-tuned on 200,000 merged rows. It was not a small dataset.

The current recommended Phase 5 budget is:

- 15,000 balanced demonstration rows total: 150 per goal;
- 15,000 planner-rollout rows total: 150 per goal;
- 30,000 unique rows in the merged fine-tuning dataset.

The old 100,000 plus 100,000 configuration should become an optional
large-data scaling point, not the default paper run. The 150-row setting is
encoded in the latent-only preliminary campaign above; it has not replaced the
guarded paper launcher. Before changing the paper launcher, verify that the
four difficult motions can reach 150 rows and increase the collection safety
limit without changing the 500-step episode protocol.

Data preparation and hashes:
[BONES-SEED Phase-5 Data Preparation](bones-seed-phase5-data-preparation.md).

## Enc380 5B Qualification and Revised Planner Diagnostic (2026-07-30)

The root+qpos-content latent tracker reached a durable 5,000,085,504 credited
frames at
`/data/resume_store/lafan1_enc380_rootqpos_h10_z256_seed0/model_5b.pt`
(SHA-256 `d33fa146f54222848da8b9a92eb5579f5acb8b3a46c484399c906b076c219260`).
Historical qualification job `5550527` explicitly used the
`Isaac-Imitation-G1-Latent-Strict-v0` environment that trained the tracker.
Saved training and evaluation configs agree on the pelvis anchor, all strict
termination functions and thresholds, no curriculum, and the legacy reset
family. The only intervening environment-config addition is an unused expert
keypoint observation, not an actor input. The job passed the
checkpoint-completion audit, fixed four-motion selection audit, all 14
frozen-encoder tensor bindings, and its then-current protocol checks. Its 0.35
strict tracking success is no longer treated as the qualification headline:
the launcher forced 1,000 control steps from frame 0, while the training
contract is 500 control steps from a start in `[0, 200]` and therefore never
advances beyond reference cursor 700. Fall-free survival was 1.0; failures came
from the original
tracking limits (`foot_pos_xyz`: 17, `ee_body_pos`: 11, `anchor_ori`: 2, with
overlaps).

The same rollout retains 80% of motions at 500 frame-0 control steps and 65% at
reference cursor 700; the additional failures that produced 35% occurred
outside the training support. A matched rerun also removed the apparent
full-body-versus-enc380 contradiction: under the same 1,000-step disturbed
strict test, the original 670D-input latent tracker scored 35% and enc380 scored
37.5%, with both trackers fully fall-free. The strict-pass MPJPE near 39 mm is
termination-truncated and cannot be read as a full-horizon average.

The achieved-state evaluator initially assumed the full 58D `expert_motion`
term. That code defect was fixed to replace configured qpos, EE, and five-body
keypoint pose components independently; the full focused gate now passes 48
tests. The qualification launcher previously hard-coded four environments for
the full-horizon diagnostic; it now uses the same 40 environments as the
strict pass. Job `5550527` completed that corrected non-terminating pass for
1,000 steps per motion: 102.76 mm root-relative MPJPE, 0.236 rad joint RMSE,
0.590 m EE position error, and fall-free survival 1.0 over 40,000 transitions.
Its retained video is
`/home/hice1/fwu91/scratch/Research/IsaacLab/isaaclab/logs/interface_baselines/lafan1_enc380_route_capacity_5b_20260730_historical_strict_r3/qualification/full_horizon_oracle/videos/play/rl-video-step-0.mp4`
(SHA-256 `fec18dab52cde69970f3ef93a9613994c8c989713325332cee340f96acb0262e`).
The earlier four-motion job `5549977` is superseded.

Both earlier submitted planner chains remained behind `afterok` and were
canceled; no demonstrations or planner results came from them. The replacement
gate keeps the old Strict-v0 environment and strict limits but matches the
training support: 100 parallel `walk1_subject1` starts in `[0, 200]`, 500
control steps, and the original disturbances. The local gate passed at 0.89
strict success and 1.0 fall-free survival (31.06 mm termination-truncated
MPJPE). Its separate deterministic, non-terminating pass measured all 50,000
requested transitions without survivorship bias. The 1,000-step result remains
only an out-of-distribution stress diagnostic; the 0.80 threshold is not being
tuned post hoc.

The revised planner workflow removes the learned-planner rollout loop for time
and first returns to the previous `walk1_subject1` continuity motion. One
persistent Isaac session uses 100 environments and commits exactly 100
completed variable-length trajectory segments. Rows are buffered by
`(env_id, episode_id)` until reset, so partial live segments at the cutoff are
discarded and temporal histories never cross a reset. The completed local
collection has 4,864 high-level rows and paired targets from the same causal
`10 x 93` histories: 380D root+qpos packets for the explicit-planner route and
256D latent targets for the direct latent route. It is therefore usable for
both planners without another collection.

Each of the 12 logical capacity cells (four sizes x three seeds) trains both
planners once from the same paired oracle data and evaluates them. The frozen
progressive optimizer schedule is 10k/20k/30k/50k updates for
tiny/small/medium/large at effective batch 1024, with microbatches
1024/512/256/128. Evaluation uses the minimum held-out normalized-target-RMSE
checkpoint. ICE's 16-hour wall is assigned per route, so the 12 paired cells are
implemented as 24 independent array tasks.

The guarded replacement chain was submitted on 2026-07-30:
`5550598 -> 5550599 -> 5550601[0-23]%8 -> 5550602`
(qualification, one-session oracle collection, planner/evaluation route array,
aggregate). Its fresh output root is
`logs/interface_baselines/lafan1_enc380_route_capacity_5b_oracle100_progressive_b1024_20260730`.
There is no separate planner pretrain, planner-driven collection, merge,
retrain, or finetune stage. Because the motion was chosen for continuity with
prior results, this is a preliminary one-motion diagnostic rather than a
representative paper sample.

The follow-up strong-explicit screen reuses these same 4,864 rows for H30
supervision rather than collecting again. Offline reconstruction preserves the
causal state, trajectory identity, and trajectory split and reproduces every
stored H10 prefix with maximum absolute error `1.062e-6`. One medium seed-0 H30
planner is trained for 30k updates at effective batch 1024 (microbatch 256), then
the identical checkpoint is evaluated with either its first H10 executed and
H20 discarded or with exponential temporal ensembling of the three overlapping
H10 subwindows. Ensembling happens in explicit root+qpos space before the frozen
enc380 encoder, with old root poses re-expressed against the current pelvis and
per-environment histories cleared on asynchronous reset. H30 is recorded as a
higher-bandwidth diagnostic (5,700 values/s versus H10's 1,900), not as a
replacement for the fixed main explicit row.

The implementation and local gates passed on 2026-07-30: the materializer
validated all 4,864 rows; the medium H30 model completed a batch-1024,
microbatch-256 update with target width 1,140; and 20-step one-environment Isaac
rollouts passed for both first-H10 and exponential execution. The guarded
launcher is `submit_enc380_h30_temporal_ensemble_ice.sh`; it depends on source
demo job `5550599`, and its aggregate also waits for source route-array job
`5550717` so the repaired matching H10 medium/seed-0 baseline is present.

During the original route array, the explicit tasks exposed a defect in the
packet-encoder pin *audit*: it compared the command held for ten 50 Hz control
steps against a fresh target recomputed on every control step, producing a false
MSE near 0.123. The packet contents and encoder layout checks passed, and the
latent-route sibling tasks completed. Pin v2 compares the expert-packet encoder
output directly to the oracle encoder output inside the same 5 Hz publication;
the local actual-encoder gate measured exactly zero MSE and zero max error over
three publications. Recovery array `5550717[0-11]%8` waits for the original
array, reuses all existing explicit `best.pt` files, and runs only pin v2 plus
missing evaluations. Repaired main aggregation is job `5550718`.

The H30 screen is now running as ICE job `5550720`; its audited aggregate is job
`5550721`, dependent on both the H30 screen and explicit recovery array
`5550717`. Output:
`logs/interface_baselines/lafan1_enc380_h30_temporal_medium_seed0_20260730`.

## Preliminary Planner Evidence

The corrected one-motion `walk1_subject1` experiments show:

- causal planners work for both interfaces;
- at the tiny size, latent is stronger across flow, diffusion, and
  deterministic objectives in the current diagnostic;
- the three-seed flow diagnostic first reaches the fixed target at about
  `0.13M` parameters for latent and `4.19M` for explicit;
- explicit often catches up or obtains lower MPJPE at larger sizes;
- rollout fine-tuning frequently hurts tracking in the current one-motion
  setting, so demonstration-only and fine-tuned results must remain separate.

The working interpretation is that the latent interface may reduce the planner
capacity required for useful control, not that it always has a better
large-model tracking ceiling. None of these one-motion results is a paper claim
until repeated across motions.

Exact diagnostic tables and artifact paths are in
[LAFAN1 From-Scratch Interface Comparison](lafan1-from-scratch-comparison.md).

## Immediate Work Queue

1. Change the Phase 5 default paper data budget to 150 demonstration and 150
   rollout rows per goal, while preserving exact balanced counts.
2. Fix compute-local storage use and prevent repeated storage errors from
   creating gigabyte-scale logs.
3. Add an audited recovery path that either trims and reuses valid partial
   shards or deliberately starts from a fresh output root. Never silently mix
   partial seeds.
4. Run the smallest local collection/schema smoke for the revised budget.
5. Dry-run all three guarded seed launchers, then submit replacement Skynet
   chains only after the preflights pass.
6. Allow Phase 4 low-level jobs and qualification to finish; submit the Phase 4
   planner grid only after its strict gate passes.
7. Aggregate Phase 4 and Phase 5 only from complete audited seed sets, then
   build the final paper release bundle.
8. Run the bounded planner architecture/size study after the main Phase 5
   pipeline is healthy; do not multiply architecture, data, and command-style
   sweeps into one combinatorial grid.

## Execution Policy

- Use the local workstation for code debugging, inference, metrics, and video.
- Local low-level runs may reach about 10M frames for routine debugging and at
  most about 50M for a serious check. Do not run 100M locally.
- Use Skynet for long low-level convergence, large data collection, final
  verification, and paper-quality numbers.
- Preserve the frozen rewards, resets, terminations, random start range, push
  event, command cadence, and episode length unless the user explicitly
  changes the research protocol.

## Document Map

- [Project Progress Report](progress-report.md): results-facing summary in
  three fixed sections (latent encoder ablations, interface design,
  hardware), updated on every result change.
- [Experiment Navigation](../experiments/README.md): current dated campaign,
  historical launcher indexes, and staged paper-facing entrypoint.
- [Causal High-Level Interface Paper Plan](causal-interface-paper-plan.md):
  authoritative research design and phase contract.
- [Whole-Body VLA and Latent-Action Literature Review](whole-body-vla-literature-review.md):
  what current explicit-chunk and latent-action systems actually deploy, how
  they relate to our comparison, and the boundary between a native baseline
  and a literature-inspired diagnostic.
- [LAFAN1 From-Scratch Interface Comparison](lafan1-from-scratch-comparison.md):
  detailed Phase 3/4 chronology, diagnostics, checkpoints, and job history.
- [BONES-SEED Phase-5 Data Preparation](bones-seed-phase5-data-preparation.md):
  corrected data tree, hashes, caches, qualification, and Phase 5 handoff.
- [Fair Interface Baselines](fair-interface-baselines.md): operational
  two-interface runner and adapter details.
- [Context Management](context-management.md): repository ownership and where
  future context belongs.

When this page disagrees with a phase document about a frozen protocol, verify
the code and update both. When it disagrees only about current execution state,
this newer dated snapshot should be refreshed from Slurm and treated as the
status entry point.
