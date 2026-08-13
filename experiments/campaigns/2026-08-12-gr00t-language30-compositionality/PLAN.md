# Plan: close the gap between our planner and the SONIC-tracker planner

## The result that motivates this

Matched protocol — M3 fall-only termination, `physics=newton_mjwarp`,
150 episodes (5 per goal over the same 30 goals):

| system | tracker | hold | survival | MPJPE-L |
| --- | --- | --- | --- | ---: |
| SONIC oracle (encoder-driven) | SONIC v1.1 | 1 | 150/150 | 25.41 mm |
| SONIC-style GR00T planner | SONIC v1.1 | 1 | 150/150 | 49.76 mm |
| GR00T fsq64, rollout data | ours fsq64 | 10 | 0.933 | 58.0 mm |
| GR00T z256, rollout data | ours z256 | 10 | 0.933 | 64.7 mm |
| GR00T chunk, rollout data | ours z256 | 10 | 0.967 | 76.9 mm |

The SONIC-tracker planner beats our best arm on both axes — 14% lower MPJPE
and zero falls — while sitting on a tracker whose oracle ceiling is WORSE than
ours (25.41 mm against our z256 tracker's 18.13 mm, though that 18.13 was
measured with tracking terminations active, so it is not yet a matched cell).
The planner advantage is therefore not inherited from the low level. Something
about the interface, the data, or the training recipe is doing the work.

**Goal: make the GR00T planner on OUR trackers match or beat 49.76 mm at
150/150 survival.**

## Candidate explanations, and what separates them

1. **Hold 1 vs hold 10.** At hold 10 one latent must describe 200 ms and the
   tracker runs it open-loop; a bad latent costs ten steps. At hold 1 the
   planner re-specifies every 20 ms. Separated by Phase 0B and Phase 2B.
2. **Data quality.** SONIC's oracle collection completed 898/900 at 26.63 mm.
   Our rollout collections come from trackers that drop 6-10% of episodes, so
   their state distribution carries more near-failure and recovery. Separated
   by Phase 3.
3. **Training recipe / inference settings.** `state_dropout_prob=0.2`, four ODE
   steps, single flow sample, open-loop slot consumption. Separated by Phases
   1 and 2A.
4. **Phase channel.** Our fsq64 actor carries a 2-wide `sin_cos` channel with
   `code_period=10` encoding position within the hold window. SONIC has no
   phase channel. At hold 1 ours would sit at slot 0 forever. Measured in
   Phase 0B, addressed in Phase 2B.

## Phase 0 — baselines and the hold-1 gate (no training, ~1 h) — RUNNING

`eval_oracle_ceiling.sh <z256|fsq64>`, `HOLD` in {10, 1}, M3 fall-only,
Newton, 150 episodes. Four runs.

- **0A, HOLD=10**: the true ceiling each arm is normalized against, under the
  arm's own protocol. Missing until now.
- **0B, HOLD=1**: does our tracker tolerate a fresh latent every control step?
  Off-distribution — trained at hold 10, and for fsq64 the phase channel pins
  to slot 0.

**Decision gate.** If 0B is within roughly 15% of 0A (the known evaluation
noise), hold 1 is available to us and Phase 2B is worth building. If 0B
collapses, hold 1 needs a retrained tracker: record that, and drop Phase 2B
in favour of Phases 1, 2A and 3.

## Phase 1 — free inference knobs (no retraining, ~1 h)

Eval sweeps on the EXISTING `fsq64_rollout` head. Nothing here costs a
training run, so anything that pays off is free.

- **1A, ODE steps**: `num_inference_timesteps` 4 -> 8, 16.
- **1B, sample averaging**: draw N flow samples per publication, average
  before the FSQ snap. Cuts the stochastic component of the published latent.
- **1C, consumption**: `fresh` vs `open_loop`. Distinguishes "the prediction
  is wrong" from "the prediction goes stale".
- **1D, temporal ensembling**: blend overlapping predictions across
  publications (ACT-style).

Report each against the 58.0 mm / 0.933 baseline. Keep whatever wins.

## Phase 2 — retraining knobs (~30 min per arm)

- **2A, `state_dropout_prob` 0.2 -> 0.** The default exists to force language
  reliance; on a well-separated 30-goal set it may just be input noise.
- **2B, hold-1 planner on our tracker** (gated on Phase 0B). Needs a
  per-control-step collection with our tracker — the same change already made
  to the SONIC collector, applied to
  `eval_skill_commander_closed_loop.py`, whose rows are currently gated on the
  publication renewal mask. Then `slots: 30, hold_steps: 1` and
  `consume_slots=10` at eval.

## Phase 3 — the data-quality hypothesis

Train our-encoder targets on SONIC-DRIVEN rollouts.

- **3A**: recollect with the SONIC tracker, this time storing the 30-frame
  expert `root_qpos` lookahead (dropped from the first collection to save
  18 GB of disk).
- **3B**: re-encode that lookahead through our fsq64 encoder. The
  cross-encoder path exists (`latent.source: fsq_prequant` plus
  `layout_check_encoder`); z256 needs a plain re-encode path added.
- **3C**: train, then eval on our tracker.

**Known risk, state it with the result.** The state history in that data
carries SONIC's `last_action` distribution (29 of the 93 per-frame features),
so a head trained on it and deployed on our tracker sees a domain shift. Also
on record: re-encoding is not equivalent to fresh collection — the
`fsq64 rollout re-encoded` arm scored 94.1 mm against 58.0 mm for the natively
collected one. Expect this phase to be informative rather than decisive, and
consider **3D**: finetune the Phase-3 head briefly on our own rollout data to
recover the `last_action` distribution.

## Phase 4 — single-motion ablation (`lift_crate_walk`, rollout only, z256)

Removes language and multi-motion error so the remaining gap is purely the
state-to-latent regression. Run the surviving knobs from Phases 1-3 here
first when a full 30-motion run is too slow to iterate on.

## Phase 5 — scale the winner

Retrain the best configuration on all 30 motions, both interfaces (z256 and
fsq64), and re-run the matched table. Then EC.

## Protocol rules for every row

- M3 fall-only termination, `physics=newton_mjwarp`, 150 episodes
  (5 per goal x 30 goals), `metric_interval 10`.
- Report MPJPE-L and fall-only survival together, always.
- A single run is not a result: measured run-to-run MPJPE spread is 0.2-6.4%,
  so treat differences below about 15% as unresolved until seeds are repeated.
- Every arm is reported against its OWN oracle ceiling from Phase 0, never
  against another tracker's.

## Phase 1 results (2026-08-13)

Baseline `fsq64_rollout` at the corrected 2000-step cap: **61.60 mm / 0.933**
(the 57.99 mm previously quoted came from a 500-step cap that truncated a
third of episodes before their reference end).

| knob | fall-free | MPJPE-L |
| --- | ---: | ---: |
| baseline (`open_loop`, 4 ODE steps, 1 sample) | 0.933 | 61.60 mm |
| 1A: 16 ODE steps | 0.940 | 63.33 mm |
| 1B: average 4 flow samples | **0.980** | 67.52 mm |
| 1C: `fresh` consumption (500-step cap) | 0.967 | 68.74 mm |

**No free lunch on MPJPE.** Every knob that reduces the published latent's
variance IMPROVES survival and WORSENS tracking error. Averaging 4 samples
cuts falls from 10 to 3 in 150 episodes while adding 10% MPJPE. The consistent
reading is that averaging pulls the latent toward the conditional mean of a
multimodal target: safer, blurrier. Under the 15% noise rule, none of the
MPJPE differences is resolved; only the 1B survival change is.

**1C also refutes the staleness hypothesis.** Re-planning from the current
state at every publication is 12% WORSE than consuming the head's own
consecutive predicted slots. The multi-slot structure carries real
information, which is independent support for the hold-1 x 30-slot interface.

None of these closes the gap to the SONIC-tracker planner (49.76 mm at
150/150). Phases 2 and 3 carry the remaining hypotheses.

## Phase 0 corrected ceilings and Phase 2 results (2026-08-13)

Oracle ceilings at the honest 2000-step cap (`done_rate` 1.000):

| tracker | hold | fall-free | MPJPE-L |
| --- | --- | ---: | ---: |
| fsq64 | 10 | 0.947 | 22.06 mm |
| fsq64 | 1 | 0.960 | 24.19 mm |

Planner arms, all fsq64 tracker, fall-only, Newton, 150 episodes, 2000 steps:

| arm | fall-free | MPJPE-L |
| --- | ---: | ---: |
| 2A: hold 10, `state_dropout` 0 | 0.940 | **58.73 mm** |
| baseline: hold 10, `state_dropout` 0.2 | 0.933 | 61.60 mm |
| 2B: hold 1, 30 slots, consume 10 (12k updates) | 0.933 | 85.77 mm |
| 2B at 8k updates | 0.947 | 80.15 mm |

**2A is a small free win**: dropping `state_dropout` to 0 gives 58.73 mm, a
4.7% improvement. Below the 15% noise band, so unresolved on its own, but it
costs nothing and is directionally consistent.

**2B is refuted, and clearly.** Hold 1 on OUR tracker is 46% WORSE than hold 10
(85.77 vs 58.73 mm), and it got worse from 8k to 12k updates, so it is not
undertraining. This holds even though the ORACLE tolerates hold 1 fine
(24.19 vs 22.06 mm, and better survival).

So the interface is NOT the explanation for SONIC's planner advantage. The
tracker accepts per-step latents when they are correct; what it does not
accept is a per-step stream of PREDICTED latents. Two readings, not yet
separated:

- Our fsq64 actor's `sin_cos` phase channel (`code_period=10`) is pinned to
  slot 0 at hold 1. The oracle survives that because its latent is still the
  right one for the window; a planner's error and the wrong phase compound.
- SONIC's tracker was TRAINED at hold 1 and has no phase channel, so per-step
  latents are in-distribution for it in a way they never are for ours.

The second reading, if right, means matching SONIC's interface needs a
retrained tracker, not a retrained planner — a much larger change than this
campaign scoped.

**Standing best on our tracker: 58.73 mm / 0.940**, against the SONIC-tracker
planner's 49.76 mm / 150-150. Remaining untested: Phase 3 (data quality) and
Phase 4 (single-motion ablation).

## Phase 3 is blocked by a one-macro-state-per-env constraint (2026-08-13)

Attempting 3A — a SONIC-driven collection that also stores the 30-frame expert
`root_qpos` lookahead, so our encoders could be applied offline — fails at
config time:

```
ValueError: env.data.macro_cache_device supports the root_qpos or full_body
macro-state terms (('expert_motion_qpos', 'expert_anchor_pos_b',
'expert_anchor_ori_b'), ('expert_motion', 'expert_anchor_pos_b',
'expert_anchor_ori_b')), got ('expert_motion', 'expert_anchor_ori_b').
```

`_configure_sonic_contract` sets `expert_macro_state_terms` to SONIC's own
encoder view (`expert_motion` + `expert_anchor_ori_b`, qpos+qvel and 6D root
orientation, no anchor position). Our encoders read the 38-D `root_qpos` view.
An environment carries ONE macro-state configuration, and the contract is
applied after Hydra, so a command-line override does not survive. The two
views are mutually exclusive within a single run — this is not a flag I
missed.

**The unblocking design (not yet built).** Two passes, with the encoder input
reconstructed offline rather than captured live:

1. SONIC drives. Per control step, store the causal `10 x 93` history (already
   done), plus the robot anchor pose and the reference `local_step` and
   trajectory rank.
2. Offline, rebuild our encoder's 38-D `root_qpos` window from the reference
   arrays at that `local_step`, re-expressed against the stored robot anchor,
   then run our fsq64/z256 encoder over it.

Step 2 is exactly the anchor re-expression `_encoder_flat_input` already
performs, so the layout is pinned by an existing test; what is new is sourcing
the window from the reference arrays instead of from a stored lookahead.
Estimated at a few hours, not minutes.

**Standing conclusion for the goal.** Best on our tracker remains
**58.73 mm / 0.940** (hold 10, `state_dropout` 0), against the SONIC-tracker
planner's 49.76 mm at 150/150. Of the four candidate explanations, the
interface (Phase 2B) is refuted, the free inference knobs (Phase 1) are
exhausted and trade MPJPE for survival, one recipe knob (Phase 2A) gave a
small free gain, and the data-quality hypothesis (Phase 3) is the last one
standing but needs the two-pass rig above. Phase 4 (single-motion ablation)
remains available and is unblocked.

## Phase 4 — single-motion ablation (2026-08-13)

`lift_crate_walk_ff_start_180_R_001_A140` (manifest rank 17), 150 episodes on
that one motion, fall-only, Newton, 2000 steps.

| arm | trained on | fall-free | MPJPE-L |
| --- | --- | ---: | ---: |
| SONIC-style planner, SONIC tracker | 30 goals | 150/150 | **38.60 mm** |
| GR00T fsq64 `nodrop`, our tracker | 30 goals | 150/150 | **39.76 mm** |
| GR00T fsq64 single, our tracker | 1 goal | 150/150 | 44.55 mm |

**Two findings, both against my earlier reasoning.**

1. **Single-motion training is WORSE than the 30-goal model on the same
   motion** (44.55 vs 39.76 mm, 2,500 rows vs 141,177). The premise of the
   ablation — that language and multi-motion interference add error — is
   refuted. Multi-motion data helps, presumably as regularization; the
   state-to-latent regression is not what is limiting us.

2. **On a matched single motion our planner is level with SONIC's**: 39.76 vs
   38.60 mm, a 3% difference, well inside the 15% noise band, and both survive
   150/150. The 58.73-vs-49.76 gap over 30 goals is therefore NOT a uniform
   per-step deficit. It is concentrated in the harder motions.

That reframes the goal. There is no general "SONIC's planner construction is
better" effect to chase — on an easy motion the two are indistinguishable.
The right next question is WHICH motions carry the 30-goal gap, and whether
they share a property (long horizon, large translation, high wrist elevation
— the `ee_body_pos` failure mode already on record as dominating our tracker).
A per-motion breakdown of both 30-goal runs would answer it directly and
needs no new training.

## Per-motion comparison at n=20 (2026-08-13)

Both sides rerun at 20 episodes per motion (600 total each), fall-only,
Newton, 2000 steps. `fsq64_rollout_nodrop` on our tracker vs the SONIC-style
planner on the SONIC tracker.

Aggregate: ours **59.21 mm / 0.940**, SONIC **49.18 mm / 600-600**. Ratio 1.20.

Per-motion verdict, classifying by the 15% noise rule:

- ours better: **3** motions
- level (within 15%): **12**
- SONIC better: **15**
- median ratio **1.15**

Parity or better on **15 of 30 motions**. The deficit is concentrated: the
four worst ratios are `panic_run_away_180` (2.14), `walk_big_dog_ff_225_stop`
(1.94), `talking_with_adult_turn_walk_360` (1.87) and `rock_out_002` (1.58) —
all fast locomotion or large-translation motions. We are better on
quasi-static ones (`fishing_standing_loop` 0.71, `surrender_stop` 0.79,
`crossed_arms_idle` 0.81).

Falls are even more concentrated: **33 of our 36 falls are in two motions**
(`panic_run_away` 19/20, `walk_big_dog` 14/20). SONIC never falls.

**Correction to the Phase-4 claim.** The n=150 single-motion run read
39.76 vs 38.60 on `lift_crate_walk` (ratio 1.03, "level"). At n=20 inside the
grid the same motion reads 43.7 vs 37.4 (ratio 1.17). Both sides moved about
10%, which is inside the documented 0.2-6.4% run-to-run spread compounded
across two runs, but the RATIO is not stable at these sample sizes. Treat
single-motion parity claims as unresolved; the aggregate ratio (1.20) and the
median (1.15) are the stable quantities.

**Where the remaining work is.** Not planner construction — on half the set we
are already at parity. Two motions carry 92% of our falls and the largest
error ratios. A motion-stratified or fall-weighted retrain, or a tracker fix
for fast locomotion, is the targeted next step.
