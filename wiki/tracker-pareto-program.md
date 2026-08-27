# Tracker Pareto Program — SR + MPJPE-L + MPJPE-G together

Started 2026-08-22. Plan of record for the effort to improve all three
canonical tracker metrics at the same time. Read this page before you extend
the program. Update the Status section when a step completes.

Goal set by the user: close the success-rate gap to released SONIC (0.9937)
without giving back our MPJPE lead, and reduce MPJPE-G, which the earlier
campaigns treated as a byproduct.

## Frontier (updated 2026-08-23; one seed everywhere)

Leaders — canonical 4,096 board, `no_push`, budget axis:

| row | SR | MPJPE-L | MPJPE-G |
|---|---:|---:|---:|
| released SONIC | 0.9937 | 28.65 | — |
| `ln_hold1_sonicreset` @30B (LEADER) | 0.9707 | 21.75 | 154.64 |
| `ln_hold1_sonicreset` @20B | 0.9558 | 22.15 | 168.15 |
| `fsq64_hold10_sonicreset` @20B (@30B finishing) | 0.9468 | 24.57 | 202.74 |
| `cont_det_ln_hold1` @10B | 0.9368 | 22.86 | 168.67 |
| 50B continuation, ratio 0.05 (jobs 5588645-48) | running | — | — |

Screen frontier — 2B, `bones_testbed4096_v1`, clean / robust(`no_push`):

| arm | clean SR/L/G | robust SR/L/G |
|---|---|---|
| `jepa_h1_ee_wide` (promotion candidate) | 0.9060 / 26.98 / 103.6 | 0.8960 / 29.08 / 156.3 |
| `ctrl_ee_wide` (promotion candidate) | 0.9082 / 24.44 / 174.2 | 0.8938 / 27.10 / 282.3 |
| `ctrl_ee1` | 0.9041 / 24.21 / 206.6 | 0.8918 / 27.09 / 352.1 |
| `jepa_nosig_h1_ee_wide` | 0.8992 / 27.07 / 107.5 | 0.8843 / 28.83 / 163.9 |
| `ctrl` (star hub, reference) | 0.9023 / 24.49 / 212.3 | 0.8828 / 27.30 / 332.4 |

Robust reading: pushes cost every arm ~0.010-0.019 SR, but the G ordering
holds and widens — `jepa_h1_ee_wide` keeps 156 mm under pushes where the ctrl
family blows out to 282-352. The JEPA+rewards stack is the robust-G frontier
by ~2x.

## Terminology (user decision, 2026-08-23)

"JEPA" means ONLY the LeWorldModel/LeJEPA-style well-posed objective:
deterministic transition modeling in token space with ONE online encoder on
both sides and SIGReg as the anti-collapse constraint — no EMA copy, no
stop-gradient (the `lejepa_*` arms). A prediction branch whose target comes
from an EMA encoder copy is NOT JEPA: that is the "EMA trick" (lagged
self-target, BYOL family), listed with the other training tricks. The
trained production recipe (`sigreg_ebm` + EMA targets, e.g.
`jepa_h1_ee_wide`) reads: DiffSR endpoint grounding + EMA token-prediction
trick + SIGReg. Arm names and W&B ids keep their historical `jepa_`
spellings. Codified in the repository-root `CONTEXT.md`.

## What moves each metric (measured, one seed each)

| lever | SR | L | G | source |
|---|---|---|---|---|
| SONIC resets + budget (10B->20B) | +0.019 | -0.7 mm | neutral | sonic-reset-20b (confounded by design) |
| hold-1 vs hold-10 | level | +1-2 mm at 2B, leads at 10B | -29% | star `use_hold1` |
| JEPA sigreg_ebm x hold-1 | level | +2.5 mm | -44% (212->119) | combos `jepa_ebm_hold1_256d` |
| online dyn finetune (cont det hold-1) | level | level | -18% (182->149) | bottleneck-10b dyn rows, scored 2026-08-22 |
| dyn on top of resetramp | level | level | none (saturates) | `cont_det_hold1_resetramp_dyn` |
| dyn on fsq64 hold-10 | -0.005 | +8% | +10% | `fsq64_hold10_dyn` (worse) |
| reconstruction objectives | level | best L | +72-115% (poison) | `recon_*`, posterior arms |
| tuned anchor weights (2026-08-04) | -0.9% survival | — | -37% | v2 reward screen, 3 seeds |

Structural readings:

- SR failures are wrist-height (`ee_body_pos`) terminations: 78-96% of all
  scoreboard failures. The wrists are the least-constrained bodies in the
  contract (Z-only termination, 2/5 of `tracking_reward_points`), and their
  matched reward `motion_ee_pos` is inert at 0.0.
- MPJPE-G is root drift. The 08-04 screen showed world-frame EE error is
  almost all root error.
- SR and G respond to different levers. No measured lever improves both.
- Failure-driven resets (ramp / SONIC selection) and dyn are substitutes for
  G, not additive: both attack the same drift mass.

## Planned campaign: `2026-08-22-pareto-stack` (2B screen, star protocol)

SUBMITTED 2026-08-22 (W&B group `pareto-stack-2b`, confirmed; user chose the
full 11-arm set): all 12 arms in flight: jobs 5587981-5588001,
`hold1_dyn_s1` 5588019-20 (gate cleared same day), and `ctrl_asymcritic`
5588028-29 (added by user decision; pure config via
`command_interface.reference.critic_components` — v2 already carried the
asymmetric-critic knob, no code change). Campaign directory:
`experiments/campaigns/2026-08-22-pareto-stack/`. Details and job map in its
README.

Q1 — is dyn real, does it stack with the best-G interface?

| arm | delta vs parent | settles |
|---|---|---|
| `hold1_dyn` s0 + s1, `use_hold1` s1 | `use_hold1` + dyn_block | dyn at 2B + a seed axis |
| `jepa_h1_dyn` s0 | `jepa_ebm_hold1_256d` + dyn_block | stack (sub-120 G) or saturate (~140, dyn dies) |

Q2 — do the unscreened reward terms move SR (wrist) and G (root)?
On `ctrl` AND on `jepa_ebm_hold1_256d`, one term each plus the pair:
`motion_ee_pos.weight=1.0`, `motion_global_anchor_pos_wide.weight=1.0`
(std 0.5), both. Optional adds from the feature menu:
`motion_body_pos_global.weight=0.5` (the literal G integrand) and the
asymmetric critic (both now arms in the campaign). Reward arms do NOT read against the star table.

Q3 — dyn vs the sonicreset chain: predicted redundant (see substitutes
reading). Decided at promotion, not at the screen.

Promotion path: Pareto winner -> 10B standard regime (row against
`cont_det_ln_hold1`) -> 20B sonicreset continuation (row against
`ln_hold1_sonicreset`). Encoders seed from parents via `encoder_arm`; no new
pretrains.

## Feature menu (graded, not yet scheduled)

Config-only: `motion_ee_pos`, `motion_global_anchor_pos_wide`,
`motion_body_pos_global`, asymmetric critic (un-prune the explicit-reference
superset for latent-mode critics, `observations.py` CriticCfg).
Cheap probe: checkpoint averaging / policy EMA at eval on existing 500M-spaced
saves.
Code investment, ranked: (1) left-right symmetry augmentation in the IPMD
buffer or a mirror-consistency loss — absent from the codebase, standard
humanoid trick, benefits every later arm; (2) teacher-student distillation
from the explicit tracker (`root_qpos_explicit`, 19.21 mm) through the
existing L2T machinery (`RLOpt/rlopt/agent/ipmd/ipmd_l2t.py`).
Held: soft pre-termination shaping, terminal fall penalty — second iteration
of the reward screen, only if the plain terms move SR.

## Push-termination attribution thread

Idea (user, 2026-08-22): during TRAINING, some terminations are caused by the
interval push, not by the policy. Evaluation is already clean — no scored row
keeps the push (`clean` = none, `robust` = no_push).

Attribution is cheap: `push_robot` is an explicit interval EventTerm
(`config/g1/common/events.py`), so the event manager knows which envs it
pushed and when. Record `last_push_step` per env; a termination within K
control steps of a push is push-attributable. Expect a bimodal
steps-since-push histogram; K sits in the valley.

Options, ranked (all are TRAINING changes, new-arm only, frozen protocol
untouched by default):

1. Exclude push-attributable failures from the SONIC failure-weight reset
   sampler. Cleans the mechanism behind the +0.019 SR without touching the
   policy's learning signal. Top pick.
2. Reclassify push-window terminations as truncations (bootstrap V(s_T)):
   the policy eats the fall but not the infinite-horizon terminal cost of an
   exogenous shove.
3. Grace window (suppress termination K steps post-push): changes frozen
   termination semantics, buffer pollution risk. Second iteration.
4. Ignore push terminations outright: rejected — removes the recovery
   pressure that pushes exist to create.

Measurement before building, both protocol-neutral:

- One diagnostic eval row WITH pushes (`--randomization all`, keeps the
  interval push) on the 10B headliner, canonical protocol otherwise. The
  pushed-vs-`no_push` SR gap bounds the size of the whole effect. Output goes
  to `logs/push_attribution/`, never to a scoreboard directory.
- Instrument the eval path with a steps-since-push-at-termination histogram
  to measure the attribution window K.

## Push-attribution measurements (2026-08-22, both complete)

- Pushed diagnostic row, `cont_det_ln_hold1` @10B, full canonical protocol
  with `--randomization all`: SR 0.9336 / L 22.87 / G 178.42 against the
  `no_push` row 0.9368 / 22.86 / 168.67. **Push-attributable SR loss at
  convergence: 0.0032** (13 of 4,096 envs). `ee_body_pos` stays 82% of the
  failures even with pushes live. Row in `logs/push_attribution/`.
- Histogram probe (1,024 envs, 2,000 steps): the K-window idea does not
  survive contact with the push cadence. Pushes fire every 1-3 s (50-150
  control steps), so nearly every step is "recently pushed" and the
  steps-since-push distribution has no bimodal valley; the 9 push-preceded
  `ee_body_pos` failures have median 60 steps since push, coincidence range.
  A/B (pushed vs `no_push`) is the honest attribution instrument, and it says
  the converged effect is small.

VERDICT: thread DEPRIORITIZED below the wrist-reward and pareto-stack work.
The sampler-exclusion arm is not worth a screen slot at a 0.003 SR ceiling.
Revisit only if trainer-side telemetry shows the push-failure share is large
EARLY in training (the converged bound says nothing about the sampler's
early-phase pollution). The instrumentation stays: `--randomization all` on
`evaluate_checkpoint` now records `aggregate.push_attribution` (overall
buckets are failure-only; completions stay in the per-term breakdown).

## Screen results (2026-08-22, clean rows, one seed)

Winner: the ee x wide reward pair. `ctrl_ee_wide` 0.9082/24.44/174.2 (campaign
SR-max); `jepa_h1_ee_wide` 0.9060/26.98/103.6 — the first arm to beat its
parent on ALL THREE metrics at matched frames, and the best MPJPE-G ever
measured in the program. Singles: wide cuts G 18-29% alone; ee is L-best and
SR-neutral alone. `ctrl_bodyglobal` mild (-15% G, SR -noise). Dyn is
null-to-negative at 2B on both seeds (seed-pair noise 0.006 SR / 6% G) — its
10B G-effect is budget-dependent and dyn does NOT ride the promotion.
`ctrl_asymcritic` still training. Full table in the campaign README;
rows in `logs/pareto_stack_eval/`.

Same day: `ln_hold1_sonicreset` @30B scored 0.9707/21.75/154.64 (from
0.9558/22.15/168.15 at 20B) — budget still buying SR, curve not flat,
gap to released SONIC now 0.023.

## Objective decomposition complete (2026-08-24)

Full table in the campaign README. Two attributions worth carrying:

- The STOP-GRADIENT is the load-bearing half of the EMA trick: adding it to
  pure JEPA moves 0.7048 -> 0.8384 SR and cuts MPJPE-G 82% (568.9 -> 104.8).
  The EMA lag adds a further +0.009 SR on top. Collapse was never the failure
  mode; co-adaptation was.
- G and (SR, L) have different sources, measured on MATCHED clips (MPJPE is
  success-only and the arms survive different sets; the 2,768-clip common
  subset is the honest comparison). DiffSR endpoint denoising owns local
  tracking: -22% and -43% in the two one-variable contrasts. The EMA-lagged
  prediction term owns global drift: -24% (`dsrsig` -> hub), versus only -8%
  for stopgrad prediction. CORRECTION: `lejepa_sg` does NOT match the hub on
  G once selection is controlled (91.87 vs 77.95, hub 18% better) — the
  earlier claim of a tie came from unmatched success sets.
- Triplet context and qvel frames both make the pretext task EASIER and the
  interface WORSE, compounding to 0.7891 together. `pred_over_capy`-style
  gate metrics predict tracker quality inversely.

## Headliner replacement (2026-08-24)

The promoted winner is intended to REPLACE `cont_det_ln_hold1` as the paper
headline row. 0 -> 20B is `2026-08-23-emastack-20b` (5590001-06, running);
20 -> 50B is `2026-08-24-emastack-50b` (planned, gated on the 20B cap, run-id
seeded). Schedule mirrors the ln leader through 20B so the rows compare, then
pins the MEASURED ratio 0.1 rather than the unscored 0.05 the ln 50B chain
uses. Rows to beat: ln@30B 0.9707/21.75/154.64, released SONIC 0.9937/28.65.

## Status

- 2026-08-23: 50B continuation submitted (`2026-08-23-sonic-reset-50b`,
  jobs 5588645-48) — +20B on the ln leader with `adaptive_uniform_ratio`
  0.1 -> 0.05 by user decision; 30B -> 50B is budget+focus confounded, the
  clean budget read stays 20B -> 30B. Frontier tables above refreshed with
  the mechanism square, robust rows, and the 30B row.
- 2026-08-22: page created. Dyn arms scored on the 4,096 board (rows in
  `logs/bottleneck_10b_4096/`, script now carries the dyn rows and
  `--skill_encoder_source`). Pareto-stack designed, awaiting user cut
  (full 10 arms vs minimum 5) and W&B group confirmation. Push-attribution
  measurements started.
