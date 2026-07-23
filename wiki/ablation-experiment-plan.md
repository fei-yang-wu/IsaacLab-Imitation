# Ablation Experiment Plan: Interface Comparison and Language Conditioning

Drafted 2026-07-21; protocol decisions confirmed by the user the same day
(see "User-decided protocol" below). Status: **implementation started —
nothing submitted.** All cells in this plan are post-joint-order-fix (PR #24,
`900c66c`); no pre-fix checkpoint, planner sample row, or latent space may
enter any cell. Primary cluster: ICE (16h walltime cap per GPU job, resumable
segment launchers required for low-level training).

## User-decided protocol (2026-07-21)

These decisions apply to the ablation studies in this page (the frozen two-row
main comparison keeps its own stricter contract):

1. **Tracker qualification = training plateau**, not a survival-rate gate.
   A low-level tracker qualifies for planner cells when its training curves
   (episode length / return) have visibly plateaued at the 5B budget. No 0.8
   strict-success eval gate for ablation rows.
2. **Planner task input = task index.** Each motion gets an integer index fed
   through a learned embedding table — no language, no reference cursor.
3. **Evaluation protocol: start at reference frame 0, run ~700 control steps**
   (frames 0–699), the theoretical maximum frame range the low-level policy is
   exposed to in training (random start 0–200 + 500-step episode). This
   replaces the random-start 0–200 / 500-step eval for these studies.
4. **Fine-tune dataset must be sufficient for every planner size.** One fixed,
   generous data budget for all size cells so capacity is the only variable —
   no per-size data scaling.
5. **Low-level budget = 5B frames for every tracker** (vanilla, EE, FSQ,
   SONIC, latent), so convergence is not a confound.
6. **Study 1 runs on corrected LAFAN1** (40 motions).

This plan extends, and does not replace, the frozen two-row main comparison in
[causal-interface-paper-plan.md](causal-interface-paper-plan.md). The user has
explicitly widened the ablation scope to include EE-chunk and FSQ interface
rows; the two-row table remains the *main* paper result, and the extra rows
land in an ablation/appendix section.

---

## Study 1 — Interface ablation: performance vs. planner capacity

**Question.** How does the choice of planner→tracker command interface affect
(a) closed-loop performance at matched planner parameter counts, and (b) the
minimum planner size needed to reach a fixed performance target?

### Rows (4 interfaces)

| # | Interface | Command payload @ 5 Hz | Low-level consumer | Implementation status |
| --- | --- | --- | --- | --- |
| 1 | DiffSR latent (ours) | 256-d continuous latent | Frozen DiffSR latent tracker @ 50 Hz | Implemented; 5B retrain running (ICE `5525266`/`5525267`) |
| 2 | Explicit full-state packet | 670-d term-major `[580, 30, 60]`, slots 0–9 consumed once each | Frozen vanilla tracker @ 50 Hz | Implemented; vanilla tracker needs post-fix retrain |
| 3 | FSQ tokens (SONIC-style) | Discrete code chunk from `FSQSkillEncoder` | Frozen FSQ-latent tracker @ 50 Hz | Encoder exists (`RLOpt/rlopt/agent/hl_skill_encoder.py:FSQSkillEncoder`); low-level train + planner head are new work |
| 4 | EE chunks | 10-slot end-effector waypoint packet (hands + feet + root, anchor-relative) | Frozen EE tracker @ 50 Hz | **Not implemented** — new command terms, new tracker reward, new tracker training |

Row-specific design decisions:

- **FSQ planner output.** The planner predicts the continuous pre-quantizer
  embedding `z_e`; the frozen `FSQQuantizer` snaps it to the code grid before
  publication. This reuses all three continuous planner families unchanged and
  keeps the optimizer budget comparable. A token-argmax (cross-entropy) head is
  a fallback diagnostic only if the snap-through variant visibly underfits.
  Check `run_per_step_token_interface_comparison.sh` for reusable scaffolding
  before writing anything new.
- **FSQ codebook (user-decided 2026-07-21): SONIC-matched, not
  categorical-matched.** The FSQ row reproduces the released
  SONIC/GR00T-WholeBodyControl token space — tokens of shape (2, 32) at 32
  levels per dim (`gear_sonic` `all_mlp_v1.yaml`: `num_fsq_levels=32`,
  `max_num_tokens=2`), i.e. 64 discrete values × 32 levels ≈ 320 bits per
  5 Hz command, mapped onto our encoder as one 64-dim FSQ code per held
  horizon. The 64×128 categorical code is our own encoder design and gets
  its own ablation later; record bits/command for every row — bandwidth is
  a reported axis, not a controlled one.
- **EE chunk payload.** Current + 9 future frames, term-major like the explicit
  packet: per-slot {left/right hand pos, left/right foot pos, root pos} in the
  current-robot anchor frame + root ori (6d). Exact widths fixed at
  implementation time and recorded in `planner_sample_schema.py`.
- Rows 3 and 4 each require their own 5B-frame low-level training run,
  plateau qualification (user-decided: training curves plateaued, no survival
  gate), dataset-path binding, and for row 3 a
  `validate_latent_skill_checkpoint_binding.py`-style encoder binding record.
- A fifth tracker variant is trained for reference: **SONIC-style joint
  training** (encoder updated during RL with a PG term on top of the
  reconstruction loss, per the SONIC recipe) — as opposed to the frozen
  pretrain-then-RL DiffSR path. Whether it becomes a planner row or stays a
  low-level-only comparison is decided after its plateau behavior is seen.

### Fixed protocol (identical across all rows)

- Planner input: causal `10 × 93` robot history + **task index** (integer
  motion id through a learned embedding table). No language on Study 1, no
  future reference state, no reference rank/cursor.
- Publication: 5 Hz, per-environment renewal (never global-timestep modulo).
- Training episodes keep the frozen protocol (random start 0–200, 500 steps).
- **Evaluation: start at reference frame 0, run ~700 control steps
  (frames 0–699)** — the full frame range the tracker was exposed to in
  training. Same fixed horizon for every cell.
- M3 terminations: tracking-error terms disabled, `base_too_low` active;
  survival = no `base_too_low`; plus the full-horizon no-termination MPJPE
  diagnostic pass with retained video.
- One fixed, generous demonstration + rollout row budget for every size cell,
  sized so the largest planner is not data-starved; same optimizer step
  budget, same seeds, same eval starts for every cell. Demonstration-only and
  rollout-fine-tuned results reported separately, never merged.
- Push event identical across rows.

### Grid

To avoid the combinatorial sweep AGENTS.md forbids, the planner **family is
fixed to flow matching** for the interface × size grid. The 3-family
comparison (flow / diffusion / deterministic) stays a separate matched-size
study on the two main rows only, exactly as already scoped.

- Sizes: existing `tiny / small / medium / large` tiers from
  `run_one_motion_capacity_point.sh`, plus one `xlarge` point if `large` has
  not saturated. Knobs: `d_model / num_layers / num_heads` in
  `interface_planner_common.py`; **plot against actual parameter counts**
  (`parameter_counts()`), since equal tier ≠ equal params across output heads.
- Seeds: 0, 1, 2 (planner seeds; low-level trackers stay at seed 0).
- Dataset: **LAFAN1 40-motion corrected** (user-decided; the LAFAN1 latent
  low-level 5B retrain is already running). BONES-SEED-91 repetition only for
  cells that matter after LAFAN1 results are in.
- Cell count: 4 interfaces × 4 sizes × 3 seeds = 48 planner trainings + evals
  (planner training is cheap relative to low-level training; the expensive,
  serialized part is the 4 qualified trackers).

### Analyses

**A. Iso-parameter + scaling curves.** Per interface: metric vs. actual
parameter count, mean ± range over 3 seeds. Metrics: survival, success,
root-relative MPJPE (absolute and oracle-normalized), root/joint/EE errors,
action change, velocity/acceleration errors, planner latency (root forward
only, post-warmup, CUDA-synchronized), params, bandwidth.

**B. Iso-performance capacity.** Smallest tested size per interface reaching
a fixed target on the frame-0 / 700-step eval: **survival ≥ 0.8 AND
oracle-normalized MPJPE ≤ threshold** (threshold frozen from the
plateau-qualified oracles' own frame-0/700-step numbers before any planner
cell runs — proposal: ≤ 1.5× oracle MPJPE). Report as "params needed to reach
target," with the honest caveat that it is quantized to the tested grid — no
interpolation through non-monotonic curves.
`aggregate_one_motion_capacity_scaling.py` already computes both answers;
extend it to the multi-motion setting rather than writing a new aggregator.

Prior expectation to test (from the one-motion diagnostic): latent reaches the
target at ~0.13M params vs. ~4.19M explicit, but explicit can catch up or win
on MPJPE at large sizes.

Working hypothesis (user, 2026-07-21, consistent with the first 240M frames
of the 5B arms): the FB chunk tracker will be the best low-level oracle —
the command hands it the full reference — but its 670-value 5 Hz packet is
the hardest planner output; EE is a similar story (smaller 360-value packet,
but the tracker gets less information and, per the HuMI notes, EE-only
under-specifies whole-body intent). The study's job is to quantify exactly
this tracker-ceiling vs. planner-capacity trade per interface, which is why
oracle-normalized MPJPE and params-to-target are the primary axes.

### Sequencing and cost

1. **Now (running):** latent 5B low-level retrains, ICE `5525266` (BONES-91)
   and `5525267` (LAFAN1).
2. **Submit next:** post-fix vanilla (full-state) tracker at 5B on LAFAN1 via
   a resumable ICE launcher cloned from `submit_lafan1_5b_resumable_ice.sh`.
3. **FSQ track (parallelizable):** FSQ encoder pretrain (50k updates, ~35 min
   on ICE) → 5B-frame low-level → plateau qualification. If the FSQ tracker
   plateaus at a visibly broken level, report that as a result and drop the
   row's planner cells.
4. **EE track (largest new engineering):** implement command terms + env
   config → local 10M smoke → 5B-frame ICE run → plateau qualification. Same
   report-honestly rule.
5. **SONIC joint-training track:** verify/implement the encoder-updated-with-
   PG-loss path, then a 5B LAFAN1 run on the Sonic surface as the
   joint-training reference point.
6. **Planner grid:** after each tracker plateaus + streamed-equivalence /
   binding certificates regenerate, run its 12 planner cells (4 sizes × 3
   seeds). Planner cells are GPU-light and fit ICE 16h jobs without segmenting.
7. Aggregate with the extended capacity aggregator; hash-bound, no-overwrite,
   refuse mixed pre/post-fix artifacts.

---

## Study 2 — Language conditioning and motion switching (BONES-SEED-91)

**Question.** Can one language-conditioned planner drive the frozen low-level
tracker across all 91 motions, and does swapping *only* the language input
switch the executed motion?

### Setup (reuses the frozen Phase-5 pipeline)

- Dataset: BONES-SEED-91 (SONIC-exclusion-filtered manifest of the 100-motion
  tree). Language goal = 384-d MiniLM embedding of the motion's annotation,
  supplied **explicitly** at deployment — never derived from reference rank,
  expert history, or trajectory reassignment after reset.
- Planner: one shared planner per interface across all motions, language token
  appended to the causal `10 × 93` history. Both main rows (latent, explicit)
  for the paper table; the switching demo (below) can run latent-only first.
- Data budget: the revised Phase-5 default — **150 demonstration + 150
  rollout rows per goal** (91 goals → 13,650 + 13,650; 27,300 merged
  fine-tune rows), balanced exactly. Before submission, verify the four
  known-hard motions (`ab_bicycle_001_A359`, `crawl_ff_loop_180_R_001_A214`,
  `jump_sideway_135_001_A021`, `sitting_legs_bend_arms_front_loop_001_A030`)
  can reach 150 rows and raise the collection safety limit without touching
  the 500-step episode protocol. Note two of the four hard motions are among
  the SONIC exclusions candidates — confirm which of them survive in the
  91-motion manifest before budgeting.
- Rollout collection: 10 parallel envs *within* one explicit goal, all envs
  same goal + same named motion, goal/reference mismatch = immediate failure.
  Final closed-loop eval: 1 env per goal.

### Evaluations

**E1 — Per-goal command competence (paper table).** For each of the 91 goals:
success, survival, oracle-normalized MPJPE, and the full metric set, evaluated
against the matching named motion. Paired latent-minus-explicit differences by
goal within seed (per the multiseed aggregator's rules). Seeds 0/1/2.

**E2 — Language selectivity (confusion analysis).** For a fixed stratified
subset of ~20 goals (covering locomotion, dance, floor/crawl, jump
categories): command goal *g*, roll out, then score the achieved motion
against **all 20 references**; the executed motion is the argmin-error
reference. Report the 20×20 confusion matrix and top-1 selection accuracy.
This is the quantitative version of "switching the language switches the
motion" and needs no new environment code — only an offline rescoring pass
over saved rollout states.

**E3 — Mid-episode language switch (demo + metric).** Start on goal *g₁*,
swap the language embedding to *g₂* at a per-environment command-renewal
boundary at t ≈ 5 s. Metrics: post-switch tracking error vs. the *g₂*
reference (time-aligned from the switch), time-to-transition, and survival
through the switch. Retain video (print absolute path to stdout). This
needs one small, isolated env/eval change: allow the evaluation harness to
update the goal embedding at a renewal boundary mid-episode. Keep it
eval-only; do not change training. Reference alignment for scoring restarts
the g₂ reference near its start (or nearest matching phase) — exact rule
fixed at implementation time and recorded with the eval artifacts.

E1 is the paper-facing claim; E2/E3 are the demonstration/qualitative claims.
E3's mid-episode swap is out-of-distribution for the planner (trained on
single-goal episodes) — a graceful transition is a bonus finding, a stumble
then recovery is still a positive switching result, and both outcomes get
reported honestly.

### Gates (unchanged from Phase-5 rules)

1. ICE job `5525266` (BONES-91 latent low-level) completes → strict oracle
   qualification ≥ 0.8, dataset-path-bound audit, skill-encoder binding
   record.
2. Post-fix BONES-91 vanilla tracker retrain → qualification → fresh
   streamed-vanilla equivalence certificate (all 10 phases + async renewal).
3. Fresh preparation record for the data tree; `audit_bones_seed_phase5.py
   --require-body-names` passes; every explicit vanilla command passes
   `VANILLA_DATASET_PATH` explicitly.
4. Fix the compute-local-storage exhaustion that killed the first prepare
   attempt (chunked writes stay; add disk-usage preflight + bounded logging).
5. Dry-run every launcher before submission.

Stage A (latent-only E2/E3 switching demo) needs only gate 1 + 3 and can start
as soon as the running 5B job qualifies. Stage B (two-row E1 paper table)
needs all gates.

### ICE adaptation

The Phase-5 launchers are Skynet-shaped (apptainer + `/data` binds + >2-day
walltimes). Port the five-stage chain (prepare → rollout array → finetune →
final-eval array → summarize) to ICE: 16h segments where a stage can exceed
the cap (only `prepare` plausibly does at 91 goals — the failed Skynet
attempts died at ~2.3h for 98 chunks, so 150-row budgets should fit), the
`resume_store` cumulative-state pattern from the 5B launchers for anything
resumable, and `cluster_submission.json` with workspace hash + all stage job
IDs retained per run, same as the Skynet contract.

---

## Implementation status (2026-07-21)

What exploration established (no new interface code was needed for EE/FSQ/
SONIC-joint — all three are existing config surfaces):

- EE and full-body chunk command spaces already exist:
  `agent.command_space=ee_trajectory|full_body_trajectory` selects the
  `expert_window` command terms
  (`config/g1/agents/rlopt_ipmd_cfg.py:command_space_policy_input_keys`), and
  the held-chunk contract is `env.command_hold_steps=10` +
  `env.latent_patch_future_steps=10` — the same consumption the from-scratch
  redo froze (`wiki/lafan1-from-scratch-comparison.md`).
- FSQ is a pretrain flag: `train_hl_skill_pipeline.py --latent-mode fsq
  --fsq-levels ...` (`FSQSkillEncoder` in `RLOpt/rlopt/agent/hl_skill_encoder.py`);
  the encoder type round-trips through the checkpoint config, so low-level
  training is unchanged.
- SONIC-style joint training exists:
  `agent.ipmd.hl_skill_finetune_enabled=true` adds the PG loss
  (`hl_skill_pg_coeff`, default 0.05) on top of the offline diffsr
  reconstruction loss (`hl_skill_offline_diffsr_coeff`) and anchor loss.
  Segment resume restores the finetuned encoder via
  `hl_skill_command_sampler_state_dict` in `load_model`.

New code added for this plan:

- `Isaac-Imitation-G1-Strict-v0` (`ImitationG1StrictTrackEnvCfg` in
  `config/g1/imitation_g1_env_cfg.py`): the vanilla observation/agent
  contract on the same protocol surface as the strict latent default (pelvis
  anchor, strict SONIC terminations, [0, 200] reset starts). Without this,
  FB/EE trackers would have trained on the legacy torso-anchor loose-
  termination surface and the tracker rows would not be protocol-matched to
  the running latent 5B arm.
- `experiments/submit_lafan1_chunk_tracker_5b_resumable_ice.sh`: resumable
  5B ICE chains for the FB-chunk and EE-chunk trackers (plain `train.py`,
  held 10-step chunks, learned-reward terms zeroed, 12288x12/18432,
  njmax 320/nconmax 40, seed 0).
- `experiments/submit_lafan1_latent_variant_5b_resumable_ice.sh`:
  `VARIANT=fsq` and `VARIANT=sonic_joint` resumable 5B chains cloned from the
  main latent launcher (own 50k pretrain per variant).

Tracker run matrix on corrected LAFAN1 (all 5B, seed 0, ICE H100):

| Arm | Task | Launcher | Status (2026-07-22 14:20) |
| --- | --- | --- | --- |
| Full-body chunk (held 10-step) | `Isaac-Imitation-G1-Strict-v0` | `submit_lafan1_chunk_tracker_5b_resumable_ice.sh` | **DONE 5B in one segment** (`5525739`), ep_len 454.5 / r_ep 32.5 |
| EE chunk (held 10-step) | `Isaac-Imitation-G1-Strict-v0` | same launcher | **DONE 5B in one segment** (`5525740`), ep_len 423.7 / r_ep 27.6 |
| Latent (held z, deterministic) | `Isaac-Imitation-G1-Latent-v0` | `submit_lafan1_5b_resumable_ice.sh` | 4.56B/5B (`5525664` TIMEOUT = segment end), ep_len 413.8 |
| Latent + 10-step history | `Isaac-Imitation-G1-Latent-History-v0` | same launcher | 4.48B/5B (`5525687` TIMEOUT), ep_len 403.7 |
| BONES-SEED-91 latent | `Isaac-Imitation-G1-Latent-v0` | `submit_bones_seed_sonic_5b_resumable_ice.sh` | 4.46B/5B (`5525663` TIMEOUT), ep_len 199.9 |
| FSQ latent (per-step renewal) | `Isaac-Imitation-G1-Latent-v0` | `submit_lafan1_latent_variant_5b_resumable_ice.sh` `VARIANT=fsq` | **CANCELLED** at 100M — collapsed (see below) |
| SONIC joint (per-step renewal) | `Isaac-Imitation-G1-Latent-v0` | same launcher `VARIANT=sonic_joint` | **CANCELLED** at 70M — collapsed (see below) |

Episode cap is 500 control steps, so FB reached 91% of the maximum, EE 85%,
latent 83%. Both completed arms are visibly flat over their last ~1.5B
frames (FB 449 -> 456, EE 379 -> 398), satisfying the plateau qualification.

### Per-step renewal collapses low-level training (isolated 2026-07-22)

Both SONIC-flavored arms flatlined at ep_len ~2.5-2.8 with negative per-step
reward while every held-z arm climbed past 400. A controlled local isolation
(`scratchpad/hold_isolation.sh`: one shared deterministic h10 z256 encoder,
frozen, corrected LAFAN1, Newton njmax=320, 4096 envs, 30M frames per arm,
varying **only** hold/phase) reproduced it exactly:

| Arm | 10M frames | 30M frames | r_step |
| --- | ---: | ---: | ---: |
| hold=10, phase=sin_cos (control) | 10.95 | **46.38** | +0.028 |
| hold=1, phase=none (repro) | 2.72 | **2.76** | -0.039 |

Because the isolation used a frozen deterministic encoder, this rules out
FSQ, the PG-finetuning path, and encoder pretrain quality (both cancelled
jobs had healthy pretrains: FSQ code_usage_frac 1.0 / perplexity 22,
SONIC-joint recon L1 5.3). The cause is the renewal contract itself.

Mechanism: the DiffSR skill code is an offline autoencoder code over a
10-step horizon, and the sin/cos phase is the "position within the chunk"
clock the low level uses to unroll one code into a 10-step motion.
Re-encoding every control step discards that contract, and the frozen code
was never trained to be a smoothly trackable per-step signal. SONIC can do
per-step tokens because its tokenizer is co-trained with the controller;
ours is not.

**Decision: run the FSQ and SONIC-encoder rows at hold=10 (held z), like
every other row.** This also makes the ablation vary exactly one thing
(quantizer / encoder training scheme) instead of three. The divergence from
SONIC's true per-step re-tokenization is recorded as a stated limitation,
not silently dropped. `--latent-hold-steps` remains available for a future
co-trained per-step study.

Per-step renewal (user-decided 2026-07-21): the two SONIC-flavored arms use
the new pipeline knob `--latent-hold-steps 1` (SONIC-style re-encoding of the
sliding future window every control step) with `--phase-mode none` (command
dim 256); the main latent arm keeps the held-z contract
(`latent-hold-steps` defaults to the horizon), which is the planner-friendly
design and stays the paper's primary latent row.

FSQ decision (resolved 2026-07-21): the FSQ arm uses the SONIC-release token
space — `FSQ_LEVELS` defaults to 64 dims × 32 levels (~320 bits per 5 Hz
command), matching `gear_sonic`'s tokens of shape (2, 32) at
`num_fsq_levels=32`. Supporting RLOpt fix: `FSQQuantizer` now computes the
codebook size as an exact Python int and returns per-dim level indices when
the flat code index would overflow int64 (usage metrics then pool per-dim
codes, same convention as the grouped categorical encoder). Covered by the
passing `pixi run test-rlopt` suite.

## Explicit non-goals

- No combinatorial interface × family × dataset × budget sweep. The grid is
  interface × size (flow only) and the separate family study stays two-row.
- No weakening of any qualification gate to admit a struggling interface row;
  a row that fails its oracle gate is reported as failed.
- No reuse of any pre-2026-07-21 Newton checkpoint, latent space, or planner
  sample row anywhere in either study.
- The two-row main comparison and its guarded launchers stay frozen; these
  studies are additive.

## Decisions still open (need user input)

1. **Iso-performance target**: confirm survival ≥ 0.8 + MPJPE ≤ 1.5× oracle on
   the frame-0/700-step eval, or pick different thresholds once post-fix
   oracle numbers exist.

Resolved 2026-07-21 by the user: EE row is in scope now; all trackers train at
5B frames; Study 1 runs on LAFAN1; trackers qualify on training plateau; eval
is frame-0 / ~700 steps; planner task input is a task index; one
generous fixed data budget for all planner sizes.
