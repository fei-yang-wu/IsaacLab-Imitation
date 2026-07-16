# Causal High-Level Interface Paper Plan

This page is the paper-facing contract for comparing learned latent skills with
an explicit whole-body command interface. It intentionally keeps the main grid
small. The question is whether a learned latent makes high-level planning
easier, not which member of a combinatorial command-style sweep wins.

For active job IDs, checkpoint chronology, and low-level training history, read
[LAFAN1 From-Scratch Interface Comparison](lafan1-from-scratch-comparison.md).
For the older broad baseline machinery, read
[Fair Interface Baselines](fair-interface-baselines.md). For Phase-5 data
preparation, read
[BONES-SEED Phase-5 Data Preparation](bones-seed-phase5-data-preparation.md).

## Current Decision

The main paper comparison has two planner rows:

| Planner interface | Publication and consumption | Low-level controller | Role |
| --- | --- | --- | --- |
| DiffSR latent | One latent command at 5 Hz, held for ten 50 Hz control steps | Frozen DiffSR latent policy | Proposed interface |
| Explicit vanilla packet | Ten exact vanilla command frames at 5 Hz, consumed one frame per 50 Hz control step | The same frozen vanilla tracker used by the direct ceiling | Strong VLA-style explicit baseline |

A third result is reported outside the planner table:

| Ceiling | Command | Role |
| --- | --- | --- |
| Direct vanilla | Fresh expert vanilla command every 50 Hz step | Low-level tracking ceiling; not a planner row |

EE/keypoint chunks, alternative raw packet layouts, Future-CVAE, and per-step
token variants are useful diagnostics or appendix studies. They are not in the
main grid unless the paper scope is explicitly changed.

This design isolates the useful research question. Under oracle commands, the
explicit packet must reproduce the direct vanilla ceiling to numerical
tolerance because both routes feed the same sequence into the same frozen
actor. Because it exposes the complete vanilla command rather than a subset, a
competent vanilla oracle should be at least competitive with the latent oracle;
otherwise low-level tracker quality is still a confound. This is an empirical
gate, not a mathematical guarantee, because the latent and vanilla controllers
are trained differently. Under a learned planner, predicting the 670-value
explicit packet can be harder than predicting the latent. That difference is
the interface-learning question rather than a low-level-controller confound.

## Status Snapshot

### Execution placement

Use Skynet for large training and paper-scale batch evaluation. Use the local
workstation for inference, playback, targeted metric checks, and video
rendering. Isaac Lab container startup dominates small cluster inference and
render jobs, while the local GPU can reuse the installed environment and data
cache. The exception is video recorded inside a training job that is already
running on Skynet, or a workload that cannot fit locally.

As of 2026-07-16:

| Phase | Status | Required next action |
| --- | --- | --- |
| 0. Freeze protocol and provenance | Complete | Keep hashes and exact commands with every result. |
| 1. Remove planner reference leak | Complete | Keep causal-reference invariance tests mandatory. |
| 2. Share the planner pipeline | Complete as a code gate | Keep causal and demonstration labels paired with their own state frames. |
| 3. Build the streamed-vanilla baseline | Local adapter and tracker-code gate complete | Use the final Skynet run for convergence and the paper-facing oracle gate. |
| 4. No-language sample-efficiency study | Local one-motion wiring and explicit-vanilla M3 diagnostics pass; paper study pending | Run paper comparisons only after both final oracle gates pass. |
| 5. BONES-SEED language study | The fresh 100-motion data, paired low-level qualification, and staged planner code gates pass; three paper-seed pipelines are active on Skynet | Monitor the fixed seed 0/1/2 stage chains, require every per-seed audit to pass, then run the guarded multi-seed aggregate. |
| 6. Final analysis | Pending | Use Skynet only for final verification and paper-scale runs. |

The streamed path has passed a local multi-environment equivalence check over
all ten packet phases, including asynchronous per-environment republication.
On the corrected LAFAN1 data, all seven actor inputs agree within `3.28e-7`
and deterministic actions agree within `9.54e-7`. The corrected-data vanilla
tracker improved from 53.4 mean strict steps after 10.027M frames to 534.3
after 50.135M cumulative frames. At 50M, 5 of 40 motions complete 1000 steps.
This is sufficient evidence that the local code learns and that the evaluation
works; it is not a paper-quality low-level result. Further convergence belongs
to the final Skynet run.

The version-2 planner sample contract now stores separate `causal_target` and
`demonstration_target` tensors. Trainers select the target paired with their
chosen state key. This matters for explicit packets because anchor terms for a
live robot state and an offline demonstration state are expressed in different
frames. The focused planner tests (15) and Isaac causal/supervision tests (8)
pass after this migration.

The focused no-language pipeline also passed a one-motion end-to-end smoke on
`walk1_subject1`: two demonstration rows, two planner-rollout rows, one update
per training stage, and twenty requested control steps for both main planner
rows. This proves the causal sample, shared-backbone, streamed-vanilla, and
closed-loop wiring. The tiny models and diagnostic low-level checkpoints make
the scores unsuitable for a paper comparison.

A larger local explicit-vanilla diagnostic later used the 1.05B-frame
checkpoint from active Skynet job `3500993` on `walk1_subject1`. The oracle
survived all 1,000 requested steps. A medium planner trained on exactly 1,000
demonstration rows survived 64 steps; after one 1,000-row planner-rollout
fine-tuning round it survived 139 steps. Planner RMSE improved from `0.367` to
`0.201`, but both planner runs ended on `ee_body_pos`, so this remains a useful
M3 failure case rather than a successful result. This first diagnostic used an
extended 1,000-step evaluation and planner collection, so it is retained only
as history and must not be used as the final M3 protocol. M3 now keeps the
normal 500-step episode, uses random reference starts from 0-200, disables the
three tracking-error terminations, and defines survival as not triggering
`base_too_low`. The run also exposed and
fixed closed-loop manifest resolution replacing an explicit dataset cache
path. Full details and artifact paths are in
[LAFAN1 From-Scratch Interface Comparison](lafan1-from-scratch-comparison.md).

The corrected local rerun used 500-step episodes and random starts 0-200 for
both demonstration and planner-rollout rows. The demonstration-only planner
fell at step 492 with 82.0 mm MPJPE. After the matched 1,000-row rollout
fine-tune, it reached the 500-step timeout without falling and had 57.9 mm
MPJPE. This validates the revised M3 workflow but remains a one-motion,
one-seed local diagnostic.

The matching one-motion DiffSR gate now also passes on corrected
`walk1_subject1` with no reference leak. A 23.2M-parameter planner trained on
1,000 oracle-achieved-state rows survived all 500 steps for the same ten
random starts; after a 1:1 mix with 1,000 planner-rollout rows, it again had
100% fall-free survival. MPJPE changed from 62.61 to 61.66 mm and root XYZ
error from 1.090 to 0.679 m, versus the latent oracle's 33.34 mm and 0.168 m.
The end threshold pass rate changed from 0.8 to 0.6, so rollout fine-tuning is
a mixed improvement rather than a clean win. Both checkpoints pass the
counterfactual no-reference-leak audit. Exact provenance, metrics, and paths
are in
[LAFAN1 From-Scratch Interface Comparison](lafan1-from-scratch-comparison.md).

The corrected explicit-packet planner has now been rerun from causal achieved
state on those same ten starts. At the approximately 23M-parameter point, the
demonstration-only explicit planner is closer to its own oracle than the
latent planner but falls in one of ten rollouts. After interface-specific
rollout fine-tuning, both survive all ten; latent has 61.66 mm MPJPE versus
67.75 mm for explicit. Because the result changes with training stage, retain
both stages. The seed-0 capacity sweep is now complete at approximately 0.13M,
4.18M, 22.97M, and 64.36M parameters. Its curve is non-monotonic: latent is
clearly stronger at about 0.13M parameters, while explicit has better
demonstration-only physical tracking at the three larger sizes. At 64.36M,
demonstration-only MPJPE is 56.85 mm for latent and 41.07 mm for explicit;
after fine-tuning it is 64.36 and 64.41 mm. Treat this as a one-motion,
one-seed diagnostic and repeat planner seeds before claiming a capacity
advantage. Exact results and artifact paths are in the LAFAN1 page.

Add a small planner-scaling study after the interface protocol is stable. It
answers two separate questions:

1. **Same size:** at matched actual parameter counts, how does performance
   change as each planner architecture grows, and does that curve differ for
   latent commands and explicit packets?
2. **Same performance:** for a fixed tracking target, what is the smallest
   tested planner that reaches it for each architecture and interface?

Keep this study deliberately small: use the current flow-matching Transformer,
one diffusion action-chunk Transformer, and one deterministic chunk predictor.
Use the same causal history, language input when applicable, demonstrations,
rollout rows, optimizer budget, evaluation starts, and frozen low-level
controllers. Test the same four approximate size levels and report actual
parameter counts, inference latency, and output bandwidth. The primary fixed
performance target should use survival and MPJPE normalized by the matching
low-level oracle; also report absolute MPJPE. Select the smallest *tested*
model that meets the target across repeated seeds. Do not interpolate a size
requirement through a non-monotonic curve. This is an architecture-robustness
study, not permission to expand the main table into a large planner sweep.

Use actual parameters on the x-axis rather than preset names. Make the
demonstration-only curve the clean capacity comparison, then show rollout
fine-tuning as a separate curve instead of mixing the two training stages.
Parameter efficiency and runtime efficiency are different questions:
diffusion uses several planner forward passes, while deterministic prediction
uses one. Therefore every point must also report measured planner latency and
the number of command values published per second. Keep demonstrations,
optimizer updates, batch construction, evaluation motions, starts, and seeds
identical across families. Do not give one family extra updates to make it
reach the fixed target.

The first three-seed check is complete for the current flow-matching
Transformer at `tiny` and `small`. Demonstration-only latent reaches the
diagnostic survival and oracle-normalized tracking target at 129,680
parameters, while explicit first reaches it at 4,186,144 parameters. At
`tiny`, latent has 100% mean survival and 66.14 mm MPJPE versus 56.7% and
83.71 mm for explicit. At `small`, however, explicit has lower mean MPJPE:
54.26 versus 70.20 mm. This is the expected form of the scaling result: latent
may reduce the minimum useful planner size without being uniformly better
once the planner is large enough. Multiple motions are still required before
this becomes a paper claim.

The first matched-family tiny diagnostic is also complete. Flow matching,
clean-target diffusion, and deterministic chunk prediction use the exact same
Transformer parameter tensors: 129,680 parameters for latent and 131,472 for
explicit. On seed 0 and `walk1_subject1`, demonstration-only latent versus
explicit results were 100% versus 70% survival and 63.57 versus 90.03 mm MPJPE
for flow; 100% versus 70% and 66.43 versus 99.66 mm for diffusion; and 90%
versus 70% and 65.52 versus 75.04 mm for deterministic prediction. Thus the
tiny latent advantage is not unique to the flow objective. Rollout fine-tuning
again hurt tracking and often survival, so retain both stages. Repeat this
architecture check across motions before making a paper claim; do not expand
the local sweep to every size yet.

The BONES-SEED language path passed the matching local code gate on
`Neutral_kick_trash_001_A057`. Both planners received the same 930-value robot
history and the same 384-value MiniLM goal embedding. The shared tiny backbone
predicted either a 256-value DiffSR code or a 670-value vanilla packet. Each
path collected two real simulator rows, took one update, ran offline inference,
and completed a twenty-step closed-loop call with the goal supplied explicitly.
`audit_bones_seed_language_interface.py` passed. This is a wiring result only.

The shared multi-goal path also passed a two-goal simulator integration smoke.
For each main interface it collected one demonstration row per goal, trained
one shared language-conditioned planner, collected one planner-driven row per
goal, merged the balanced rows, fine-tuned the same planner, and evaluated both
goals with explicit matching goal and motion names. The audit verified 930-D
causal state, 384-D goal embeddings, 256-D versus 670-D targets, exact
goal-to-embedding matches for every saved row, compatible shared backbone
settings, no reference features in planner state, and strict frozen vanilla
tracker restore. The tiny model, one update per stage, and ten-step evaluations
are a code gate only.

On 2026-07-15 the new balanced demonstration path passed a direct two-motion
Isaac smoke for both interfaces. Each collector saved exactly two rows and
stopped at the initial publication without taking an extra simulator step.
Both samples contained `2 x 930` causal and demonstration states and `2 x 384`
language embeddings. The explicit target was `2 x 670`; the latent target was
`2 x 256`. Every saved language row exactly matched the MiniLM entry for its
saved motion name. The latent save-only path skips duplicate planner diagnostics;
all saved state, target, language, rank, and step tensors were bit-identical to
the diagnostic path under the same seed. This validates batching and row
accounting only; it is not a tracking or planner-quality result.

## Research Questions

Answer these questions in order:

1. Is the vanilla tracker competent under direct 50 Hz expert commands, and is
   the 5 Hz oracle packet provably equivalent to that ceiling?
2. Given the same causal robot history, planner architecture, data, and
   optimization budget, is predicting a DiffSR latent easier than predicting
   the exact future vanilla-command packet?
3. Does the latent advantage persist for one language-conditioned planner
   shared across BONES-SEED motions?

The no-language study answers the first two questions. BONES-SEED can test
language-conditioned motion selection and execution. It does not by itself
support a broad end-to-end VLA task-success claim because the current setup
does not provide a visual object/scene benchmark or a separate task-success
definition.

## Exact Interface Contracts

### Explicit vanilla packet

The planner publishes one packet every ten low-level steps. It contains the
current desired vanilla command plus nine future desired commands:

| Term | Per-frame width | Frames | Packet width |
| --- | ---: | ---: | ---: |
| `expert_motion` | 58 | 10 | 580 |
| `anchor_pos` | 3 | 10 | 30 |
| `anchor_ori` | 6 | 10 | 60 |
| **Total** | **67** | **10** | **670** |

Packing is term-major: `[580, 30, 60]`. Within each term, frames are ordered
from slot 0 through slot 9. The environment consumes exactly one slot at each
50 Hz step. Anchor position and orientation are re-expressed against the
current robot anchor before the current slot is sent to the actor.

The packet is an action chunk of ten consumed frames, not eleven frames. In the
current window API that means `command_future_steps=9`,
`command_hold_steps=10`, and a 10-step planner interval.

Do not train a separate "full-body chunk policy" for this main baseline. The
adapter must feed the current packet slot into the existing vanilla actor. The
actor architecture, ordered inputs, weights, and normalization behavior must
be identical to the direct ceiling.

### Direct actor and critic commands

With command-side expert noise disabled, the direct actor's
`expert_motion`, `anchor_pos`, and `anchor_ori` values are numerically the
same as the corresponding critic command entries. They live in separate
observation groups, and the critic contains additional privileged state, but
there is no intended command-value difference between them.

This distinction matters when describing the comparison: the critic remains
privileged during training, while the deployed actor receives proprioception
and the selected high-level command only.

### DiffSR latent

The proposed row publishes a DiffSR latent at 5 Hz and holds it for ten 50 Hz
steps. Its oracle label is generated by the frozen skill encoder. The planner
must not read the expert segment used to generate that label.

Record the latent width, skill-encoder checkpoint and hash, low-level
checkpoint and hash, hold interval, and normalization metadata with every
sample set and evaluation.

### Causal planner observation

Both planner rows receive the same ten-frame causal robot history: nine past
frames plus current. Each frame has 93 values:

| Feature | Width |
| --- | ---: |
| Joint positions relative to defaults | 29 |
| Joint velocities | 29 |
| IMU angular velocity | 3 |
| Projected gravity | 3 |
| Previous low-level action | 29 |
| **Total** | **93** |

The full planner input is therefore `10 x 93 = 930` values. Its feature
names, order, normalization, reset padding, and history length must be stored
in samples and checkpoints. A mismatch must fail at load time.

Planner inputs may not contain a trajectory rank, reference cursor, phase,
future expert state, expert history, or reference-relative tracking error.
Never use `current_achieved_macro_transition_batch` for deployable inference;
that retired helper preserves reference-derived information.

### Per-environment publication

Command publication and saved planner rows are scheduled per environment.
Renew a row when that environment's own hold counter expires or it resets.
Global timestep modulo logic is incorrect once environments reset
asynchronously.

Collectors must select only the renewing environment rows and must count the
actual saved rows. Paper runs use an exact positive sample-row budget; "zero
means all" and silent reuse of stale sample files are not valid protocols.

## Frozen-Tracker and Equivalence Requirements

Load the vanilla tracker through the policy-only loader:

- Load only `policy_state_dict`.
- Require a strict state-dict restore.
- Reject unsupported hidden observation-normalizer state.
- Put the tracker in evaluation mode and freeze every parameter.
- Record the tracker checkpoint path, SHA-256, ordered actor input keys,
  strict-restore flag, and frozen flag in samples, checkpoints, and results.

Before collecting explicit planner samples, generate an equivalence
certificate that:

1. Covers packet slots 0 through 9.
2. Uses at least two environments and exercises asynchronous republication.
3. Compares every actor input, not only the three command tensors.
4. Compares deterministic actions from direct and streamed paths.
5. Verifies that actor parameters and buffers do not change.
6. Records tolerances, maximum differences, tracker hash, packet shape, and
   hold schedule.

The checked actor inputs are `expert_motion`, `anchor_pos`,
`anchor_ori`, base angular velocity, relative joint position, relative joint
velocity, and previous action. A failed certificate invalidates the explicit
baseline; it is not evidence that raw full-body commands are intrinsically
worse.

## Fixed Comparison Protocol

Keep these choices fixed across the two planner rows:

- Low-level control rate: 50 Hz.
- Planner rate: 5 Hz.
- Decision interval: ten low-level steps.
- Planner input: the same `10 x 93` causal observation.
- Same motion manifest, reset distribution, 500-step episode, evaluation
  starts, M3 termination policy, and perturbations.
- Same planner backbone, capacity preset, optimizer, update counts, flow
  settings, demonstration rows, planner-rollout rows, and seed.
- Same two-stage training: demonstration pretraining, then planner-driven
  collection and retraining.
- Exact positive sample counts, measured in tensor rows.
- No command-side expert noise. Any proprioceptive noise or domain
  randomization is identical.
- Future reference access only for oracle commands, training labels, and
  evaluation metrics.

Low-level controllers differ by interface, so low-level competence is reported
separately. Always report planner performance relative to that interface's
oracle. The explicit oracle additionally must match the direct vanilla ceiling
by construction.

Use one seed for wiring and local qualification. Freeze the protocol before
running at least three independent seeds for paper results.

## Required Gates

### Data gate

Paper-facing LAFAN1 runs may not use legacy files whose `body_pos_w` contains
Isaac scene-grid offsets while `root_pos` remains clip-local. Run
`scripts/audit_g1_lafan1_body_frames.py`, use a corrected manifest, rebuild
dependent caches, and record manifest and aggregate data hashes. Never repair
the source tree in place.

### Low-level gate

Before planner training:

- Evaluate direct vanilla and DiffSR oracle tracking on every selected motion.
- Require the agreed success threshold, currently at least 0.8.
- Require streamed vanilla to match direct vanilla within the equivalence
  tolerance.
- Keep strict evaluation terminations enabled.
- Treat 10M-frame local blocks as qualification only. About 50M total frames
  is the maximum serious local check, not a target. Never add a 100M local
  block; stop as soon as the code behavior is established. Do not use local
  compute to prove final convergence.
- Keep the frozen 0-200 reset range, rewards, terminations, and other specific
  environment settings unchanged unless the user explicitly revises the
  protocol.

If the direct vanilla tracker is weak, improve or retrain it before planner
experiments. Do not compensate by training a separate chunk-specific
controller, because that would break the main baseline's interpretation.

### Planner gate

Use the shared causal sample schema and shared Transformer-flow implementation.
Only the output target and final adapter may differ. Samples, merged datasets,
planner checkpoints, and evaluations must carry compatible metadata for:

- interface and command mode;
- observation specification;
- packet or latent shape;
- publication interval;
- sample counts and collection stage;
- dataset path and manifest hash;
- low-level and skill/tracker checkpoint hashes;
- planner architecture, parameters, seed, and update counts.

Merging or evaluating incompatible artifacts must fail.

### Evaluation gate

Use closed-loop tracking as the primary result. Report:

- full-horizon success, survival, and termination cause;
- root-relative MPJPE, root height/orientation, joint, and EE errors;
- action smoothness and external-push recovery where applicable;
- planner output size and values or bits per second;
- planner parameter count, latency, and exact unique sample count;
- performance before and after planner-driven retraining;
- planner performance divided by its oracle performance.

Offline target error is diagnostic only. It cannot replace closed-loop
evaluation.

Planner latency means the synchronized forward time of the deployed
high-level planner at a command publication. It excludes Isaac stepping, the
frozen low-level controller, tracking metrics, and output writing. Evaluators
discard the first planner call as warmup and report mean, standard deviation,
p50, p95, call count, and observed batch sizes. A paper audit requires at
least one post-warmup call for a rollout that reaches a second planner
publication. If every environment terminates on the first control step, keep
the failure as a valid result and leave latency, action delta, and tracking
acceleration unavailable. Never turn an early failure into an audit failure or
fill in a made-up zero.

## Phased Execution

### Phase 0: Freeze protocol and provenance - complete

The causal observation, rates, data ownership, artifact hashes, and result
schema are fixed. Changing one of these creates a new protocol and must be
recorded explicitly.

### Phase 1: Remove planner leakage - complete

The live and offline causal builders produce the same `10 x 93` layout.
Reference-motion and cursor invariance tests pass, reset clears history, and
planner inference no longer depends on expert macro getters. Old planner
checkpoints without the observation specification are diagnostic only.

### Phase 2: Share planner infrastructure - complete as a code gate

Both main rows use the versioned causal sample schema and the same continuous
Transformer-flow planner. Both use demonstration pretraining followed by
planner-driven sample collection, exact N+N merge, retraining, and closed-loop
evaluation.

Earlier EE and other explicit-interface wiring runs helped validate this
infrastructure. They do not add rows to the focused paper comparison.

### Phase 3: Qualify streamed vanilla - active

The streamed command adapter, strict frozen loader, per-environment schedule,
provenance checks, and strengthened equivalence certificate are implemented.
Use
`experiments/interface_baselines/run_focused_causal_interface_comparison.sh`
for the focused local workflow and
`experiments/interface_baselines/audit_focused_causal_interface_comparison.py`
for the contract audit.

The local adapter and tracker-code qualification is complete. Approximately
50M frames is the local serious-check ceiling, not a target; a 100M local block
is unnecessary. Long convergence and the 0.8
paper-facing oracle gate are evaluated by the final Skynet tracker run, not by
repeated local continuation. Local planner smokes may use the diagnostic
tracker to verify code paths, but no performance comparison is valid until the
paper-facing low-level checkpoint passes the oracle gate.

### Phase 4: No-language sample efficiency - code gate complete, paper runs pending

The two corrected-LAFAN low-level prerequisites are still training on Skynet.
At the 2026-07-16 04:14 EDT health check, vanilla job `3500993` was at 2.93B
of 5.00B frames and DiffSR job `3503434` was at 1.79B of 5.00B. Both were
healthy and projected to finish within their scheduler limits. Qualification
job `3503441` is correctly waiting for both; keep the paper array blocked until
its two oracle audits and streamed-equivalence certificate pass.

For every selected LAFAN1 motion, compare only DiffSR latent and explicit
vanilla packet planners. Use a fixed model size and a small number of meaningful
sample budgets, for example 1k, 10k, and 50k. This is a sample-efficiency curve,
not a command-style sweep.

After the setup is frozen, run at least three seeds and aggregate across
motions. The direct vanilla ceiling appears beside the table but is never
counted as a planner row.

The guarded implementation is:

- `phase4_no_language_matrix.py`: maps one array index to one motion and one
  planner seed. All three fixed sample budgets run in that task and reuse its
  demonstrations, oracle evaluations, direct ceiling, and adapter check.
- `validate_phase4_no_language_submission.py`: blocks submission unless the
  corrected 40-motion manifest, both low-level checkpoint/audit pairs, the
  latent skill checkpoint, and streamed-vanilla equivalence certificate have
  the exact qualified hashes and dataset-cache identities and both oracle
  rates are at least 0.8.
- `run_phase4_no_language_sweep.sh`: runs the three budgets for one resolved
  seed/motion task and requires a passing focused audit for each budget.
- `submit_phase4_no_language_skynet.sh`: renders or submits the 120-task
  `3 seeds x 40 motions` array. It defaults to a dry run and must remain
  blocked until the low-level gates finish. It refuses an existing output root
  and writes the array job ID and snapshot hashes to
  `cluster_submission.json`.
- `aggregate_phase4_no_language_results.py`: requires the complete task grid,
  identical gate identity, passing per-budget audits, and exact paired rows;
  then writes per-motion JSON/CSV, a generated Markdown table, per-seed
  differences, and hierarchical bootstrap intervals. It refuses to overwrite
  an existing aggregate and hashes all outputs in
  `aggregation_manifest.json`.

Use only the fixed `1k 10k 50k` planner-row budgets for the paper run. These
are planner supervision samples, not low-level RL environment-frame budgets.
Do not interpret the 50k planner sample point as permission to extend local
low-level training; the separate local low-level ceiling remains about 50M
environment frames and can stop much earlier once behavior is established.

Render the array without submission:

```bash
DRY_RUN=1 \
VANILLA_TRACKER_CHECKPOINT=logs/path/to/vanilla.pt \
LATENT_LOW_LEVEL_CHECKPOINT=logs/path/to/latent.pt \
LATENT_SKILL_CHECKPOINT=logs/path/to/skill.pt \
VANILLA_QUALIFICATION_AUDIT=logs/path/to/vanilla_audit.json \
LATENT_QUALIFICATION_AUDIT=logs/path/to/latent_audit.json \
STREAMED_EQUIVALENCE_CERTIFICATE=logs/path/to/equivalence.json \
experiments/interface_baselines/submit_phase4_no_language_skynet.sh
```

After every array task passes, aggregate from the shared output root:

```bash
pixi run python \
  experiments/interface_baselines/aggregate_phase4_no_language_results.py \
  --run_root logs/interface_baselines/phase4_no_language_lafan1
```

On 2026-07-15, the final evaluator schema passed a two-environment, 22-step
Isaac check on corrected `walk1_subject1`. The direct summary contains the
dataset path, interval-push definition, strict per-environment outcomes,
termination-cause counts, all required tracking/smoothness metrics, and honest
`max_steps`/`steps_run`/`stop_reason` fields. The streamed certificate bound the
checkpoint and one-motion manifest hashes, exercised all ten packet phases and
asynchronous renewal, left the policy unchanged, and measured maximum action
difference `1.20e-6`. This is a schema and adapter check only, not a performance
result.

### Phase 5: BONES-SEED language study - paper runs active

Train one shared language-conditioned planner across BONES-SEED motions. Both
main rows receive the same causal robot history, language representation,
training examples, backbone, and optimizer budget; only the output interface
changes.

The final-candidate low-level Skynet stage uses the fresh 100-motion manifest
and trains only two controllers: direct vanilla and DiffSR latent. Render both
1B-frame jobs with:

```bash
DRY_RUN=1 experiments/interface_baselines/run_bones_seed_low_level_skynet.sh
```

The launcher verifies the three persistent data hashes before rendering or
submitting. It fixes 4096 environments, 10,173 iterations (1,000,046,592
frames), seed 0, reset steps 0-200, unchanged rewards/terminations, no BC, and
separate fresh Zarr caches. Remove `DRY_RUN=1` only to submit these low-level
candidate jobs. Their strict 100-motion oracle audits and the new vanilla
equivalence certificate must pass before any BONES planner job starts.

For the strict oracle gate, successful `reference_finished` termination is not
a tracking failure. Audit success against the evaluator's explicit strict
`tracking_failure_rate`, which excludes `reference_finished`; do not infer
success from `done_rate`.

After both final checkpoints exist, render the combined qualification job with:

```bash
DRY_RUN=1 \
VANILLA_TRACKER_CHECKPOINT=logs/path/to/vanilla.pt \
LATENT_LOW_LEVEL_CHECKPOINT=logs/path/to/latent.pt \
LATENT_SKILL_CHECKPOINT=logs/path/to/skill.pt \
experiments/interface_baselines/submit_bones_seed_low_level_qualification_skynet.sh
```

The qualification workflow evaluates all 100 motions for 1000 steps from
frame 0, requires 0.8 success for both controllers, and creates the exact
streamed-vanilla equivalence certificate. It fails immediately if either
oracle gate fails. Submit it only after replacing the three example paths with
the exact final artifacts; then pass its two audit JSON files and equivalence
JSON to the multi-goal planner launcher.

The final qualification is complete. Skynet job `3512041` finished in 11m10s
and passed both strict 100-motion gates: direct vanilla reached `0.90` success
and DiffSR latent reached `0.84`, against the fixed `0.80` threshold. The
streamed-vanilla certificate passed all ten packet phases, asynchronous
per-environment renewal, and policy immutability. Its maximum command and
action differences were `3.02e-7` and `1.31e-6`. The persistent gate root is
`logs/interface_baselines/bones_seed_100_low_level_qualification_seed0_retry_20260716`.

Before Isaac starts, qualification compares the skill encoder tensors embedded
in the latent low-level checkpoint with the selected skill checkpoint. This
binding must pass and is recorded in the latent audit. Later Phase-4 and
Phase-5 planner launchers require the binding hashes as well as the ordinary
checkpoint hashes. For the active BONES run, use `latest.pt`, which is the
exact path recorded by low-level training.

The shared Transformer-flow planner now has an optional language token. Setting
`language_dim=0` preserves the no-language model and checkpoint layout. Phase 5
uses `language_dim=384` and one language token for both rows. Training samples
store the rank-aligned goal embedding and its table hash. At deployment the
goal name must be supplied explicitly; selecting it from the expert trajectory
rank or reference cursor is forbidden. Closed-loop evaluation therefore runs
one named goal and its matching one-motion manifest per invocation, while the
planner weights are shared across the full training set.

Use the minimal local gate before any larger job:

```bash
LATENT_LOW_LEVEL_CHECKPOINT=/path/to/latent.pt \
LATENT_SKILL_CHECKPOINT=/path/to/skill_encoder.pt \
VANILLA_TRACKER_CHECKPOINT=/path/to/vanilla.pt \
experiments/interface_baselines/run_bones_seed_language_smoke.sh
```

The runner intentionally collects only two rows, takes one update, and runs
twenty control steps. It does not establish sample efficiency or convergence.
For full two-stage single-goal diagnostics, the shared latent and explicit
runners accept `LANGUAGE_EMBEDDINGS` and `LANGUAGE_GOAL_NAME`. A final
multi-goal run must collect planner-driven data separately for each explicit
goal and merge it before retraining; never use the reference cursor to choose
the language embedding. Sample compatibility checks keep the language table
path and SHA strict while allowing different saved per-row goal embeddings to
share one dataset. A different language table still fails the merge.

The full shared-planner workflow is:

```bash
LATENT_LOW_LEVEL_CHECKPOINT=/path/to/latent.pt \
LATENT_SKILL_CHECKPOINT=/path/to/skill_encoder.pt \
VANILLA_TRACKER_CHECKPOINT=/path/to/vanilla.pt \
MANIFEST=/path/to/fresh/bones_seed_manifest.json \
LANGUAGE_EMBEDDINGS=/path/to/matching_goal_embeddings.pt \
LATENT_DATASET_PATH=/path/to/new/latent_cache \
VANILLA_DATASET_PATH=/path/to/new/vanilla_cache \
PREPARATION_RECORD=/path/to/fresh/preparation/preparation.json \
VANILLA_QUALIFICATION_AUDIT=/path/to/vanilla_qualification.json \
LATENT_QUALIFICATION_AUDIT=/path/to/latent_qualification.json \
STREAMED_EQUIVALENCE_CERTIFICATE=/path/to/equivalence.json \
OUTPUT_ROOT=logs/interface_baselines/bones_seed_multigoal_seed0 \
SEED=0 \
experiments/interface_baselines/run_bones_seed_multigoal_language_comparison.sh
```

`run_bones_seed_multigoal_language_comparison.py` collects oracle
demonstrations for all named motions in two balanced multi-environment runs,
then trains one planner per interface over the merged rows. Planner-driven
collection and closed-loop evaluation remain separate for each explicit goal.
The two low-level controllers use separate content-specific caches, and every
explicit vanilla command receives `VANILLA_DATASET_PATH` directly instead of
falling back to the environment default. Qualification audits, the streamed
equivalence certificate, planner samples, and final summaries must all record
the matching dataset path.
The runner writes a per-goal CSV, an aggregate JSON, and runs
`audit_bones_seed_multigoal_language_comparison.py`. Its matching cluster mode
is `bones-seed-multigoal-language`. The cluster wrapper refuses to start unless
the fresh preparation record is complete, both selected checkpoint hashes
match passing oracle audits at the 0.8 threshold, and the streamed-vanilla
equivalence certificate covers all packet phases and asynchronous renewal.
Prepare and inspect a submission with
`DRY_RUN=1 MODE=bones-seed-multigoal-language
experiments/interface_baselines/submit_cluster_interface_baselines.sh`; do not
submit until the fresh-data and low-level oracle gates pass.

Oracle demonstration collection is now batched safely: motion rank is used
only to label and balance supervised examples, and the planner input remains
the causal robot history plus the saved language embedding. Planner-driven
collection and both closed-loop evaluation stages still start a separate Isaac
process for every goal. Planner-driven collection uses ten parallel
environments inside that one-goal process, but every environment receives the
same explicit goal and is restricted to the same named motion. It fails if the
live reference name changes instead of following reference rank, which would
leak the answer into the planner goal.

The explicit-goal cluster split is implemented. Render the complete fixed
three-seed paper study with:

```bash
DRY_RUN=1 \
VANILLA_TRACKER_CHECKPOINT=logs/path/to/qualified_vanilla.pt \
LATENT_LOW_LEVEL_CHECKPOINT=logs/path/to/qualified_latent.pt \
LATENT_SKILL_CHECKPOINT=logs/path/to/qualified_skill.pt \
experiments/interface_baselines/submit_bones_seed_multiseed_pipeline_skynet.sh
```

The wrapper fixes seeds `0 1 2` and, for a real submission, preflights all
three output roots and shared gates before submitting the first seed. The
single-seed launcher refuses to reuse an existing output root unless
`ALLOW_EXISTING_OUTPUT_ROOT=1` is set for an intentional audited resume. It
refuses to submit until the fresh preparation and language-table hashes, both
0.8 oracle audits, exact checkpoint hashes, and all streamed-vanilla
equivalence checks pass on persistent Skynet storage. One synchronized
workspace per seed then submits this dependency chain:

1. `prepare`: two balanced oracle demonstration launches and shared planner
   pretraining;
2. `rollout[0-99]`: one explicit goal per array task for both interfaces;
3. `finetune`: exact merge followed by one shared fine-tune per interface;
4. `final-eval[0-99]`: one explicit goal per array task for both interfaces;
5. `summarize`: per-goal and aggregate results plus the protocol audit.

Every later stage checks the original run configuration, workflow source
hashes, goal index and name, and upstream artifact hashes. Planner samples are
written with a 1,000-row flush threshold. A multi-environment publication can
make an individual file slightly larger (the first live 100-environment file
contained 1,002 rows), but the balanced selector and final audit still require
exactly 1,000 rows for every goal. This keeps the exact total row budget while
avoiding roughly 100,000 tiny per-step files on the shared filesystem. The
pipeline can resume from completed, hash-verified stages. Do not reduce the
100-goal set or select a goal from reference rank.

Within one `rollout` array task, sample collection uses ten parallel
environments. All ten receive the same explicit language goal and are
restricted to that same named reference motion. The collectors trim the final
publication to exactly 1,000 rows and fail if a live reference name differs
from the goal. This reduces one goal's collection horizon from 10,000 to at
most 1,000 control steps without changing its row budget, rewards, resets,
terminations, or planner cadence. Pretrained and final closed-loop evaluations
remain one environment per goal and the normal 500-step M3 episode.

Each seed output must contain `cluster_submission.json`. The Slurm submitter
writes it to persistent storage after creating the full dependency chain. It
records the seed, all five job IDs, workspace archive SHA-256, repository-sync
manifest SHA-256, array shape, and cluster snapshot path. The final protocol
audit and multi-seed aggregator require and hash this file for paper runs.

The guarded paper submission was made on 2026-07-16 after that gate passed.
All three seeds use 100 explicit goals, `0-99%4` arrays, 1,000 demonstration
and 1,000 rollout rows per goal, ten rollout environments, a medium planner,
2,000 pretraining plus 2,000 fine-tuning updates, and the normal 500-step M3
episode. The five-stage Slurm chains are:

| Seed | Prepare | Rollout array | Fine-tune | Final-eval array | Summarize |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `3512092` | `3512093` | `3512094` | `3512095` | `3512096` |
| 1 | `3512097` | `3512098` | `3512099` | `3512100` | `3512101` |
| 2 | `3512113` | `3512114` | `3512115` | `3512116` | `3512117` |

The seed snapshots are `isaaclab_20260716_040648`,
`isaaclab_20260716_040704`, and `isaaclab_20260716_040727`. They share
workspace archive SHA-256
`58b0f17a5e5f42fa9dee737b64e86a8b67d96de1bdba988a400f2e80414e728b`.
At the first runtime inspection, all three preparation jobs had passed the
persistent BONES preflight and written their seed-specific `run_config.json`;
the later stages remained dependency-blocked as intended.

After seeds 0, 1, and 2 have each passed their final protocol audit, aggregate
them with:

```bash
pixi run python \
  experiments/interface_baselines/aggregate_bones_seed_multiseed_results.py \
  --run_roots \
    logs/interface_baselines/bones_seed_100_multigoal_language_seed0 \
    logs/interface_baselines/bones_seed_100_multigoal_language_seed1 \
    logs/interface_baselines/bones_seed_100_multigoal_language_seed2 \
  --output_dir logs/interface_baselines/bones_seed_100_multigoal_language_multiseed
```

The aggregator accepts only complete paper runs and the exact seed set
`0 1 2` by default. It checks each
summarize-stage artifact hash and requires the same goals, planner settings,
workflow source hashes, data hashes, and low-level checkpoint hashes across
seeds. It writes raw per-goal rows, paired latent-minus-explicit differences,
an audited JSON summary, and `multiseed_results.md` as a directly generated
paper-table draft. Results include success, survival, root-relative MPJPE,
root position, height and orientation errors, joint and EE errors, action
change, velocity error, and acceleration error. The summary also records
planner parameter count, output values per second, and float32 bandwidth. The
main estimate first averages the paired goal results within each seed, then
reports variation across seed means. Its confidence interval resamples
training seeds and then goals within each selected seed. Tracking success is
also divided by the matching low-level oracle success recorded by the gate.
This preserves the paired comparison and does not pretend that 100 goals from
one training run are 100 independent training seeds.

Aggregation refuses an existing output directory. It writes
`aggregation_manifest.json` containing the exact aggregation command, source
hash, input-run artifact hashes, and hashes of every generated JSON, CSV, and
Markdown result. Use a new output directory for a new analysis; do not
silently overwrite a prior paper table.

The result path also preserves both closed-loop stages: the demonstration-only
planner and the planner retrained on the exact N+N demonstration/rollout
merge. It reports the paired change for every metric and verifies the exact
unique row counts used by both planners. Final and demonstration-only results
include per-environment termination terms and aggregate cause counts. Early
`reference_finished` and genuine falls are valid measured outcomes; the audit
accepts any positive rollout length up to the 500-step episode budget and
checks that an early stop is explained by `all_envs_done`.

The two M3 evaluators use the same fall-only survival definition. They disable
`anchor_pos`, `anchor_ori`, and `ee_body_pos` as termination conditions, keep
`base_too_low`, and report the tracking errors continuously. A timeout or
`reference_finished` is a successful episode end; `base_too_low` is a fall.
The M3 episode remains 500 control steps with reference starts sampled from
0-200, so the largest reference cursor is about frame 700. The low-level oracle
gate remains a separate strict evaluation with the original tracking
terminations enabled.

The final evaluators now record the existing G1 interval-push event without
changing it. A two-step Isaac runtime check on 2026-07-15 confirmed identical
metadata for both interfaces: pushes occur every 1-3 seconds using the same
configured root-velocity ranges. The final audit requires the event to remain
enabled and identical across every goal and interface. Therefore the main
tracking metrics are measured under the existing randomized-push protocol. A
separate fixed-time recovery-duration study is optional appendix work; do not
change the main environment merely to add it.

The chunk writer also passed a real Isaac runtime check on 2026-07-15. For the
same explicit BONES goal, both interfaces saved planner steps `[0, 1]` at
control steps `[0, 10]` into one two-row file. Both rows had the expected
`930`-value causal state and `384`-value language input; the targets were
`670` values for streamed vanilla and `256` for DiffSR. This checks storage and
timing only, not planner quality.

The fresh 100-motion paper tree is complete and checksum-verified on persistent
Skynet storage. Its exact paths and hashes are recorded in
[BONES-SEED Phase-5 Data Preparation](bones-seed-phase5-data-preparation.md).
The completed data workflow satisfies these requirements:

- Generate from the raw CSV and language sources into a new output tree.
- Record input hashes, output hashes, exact commands, jobs, manifest, and
  language provenance.
- Refuse existing or overlapping output directories.
- Require body names and a passing
  `scripts/audit_bones_seed_phase5.py --require-body-names` report.
- Rebuild every Zarr or derived cache after NPZ replacement.
- The fresh NPZs reproduce all 100 previously corrected files byte-for-byte,
  so final runs use the fresh manifest and preparation record rather than the
  older in-place manifest.

Use language-conditioned motion execution as the claim boundary. Add a visual
scene/object benchmark and task-success definition before calling the result an
end-to-end VLA task study.

### Phase 6: Final analysis and release - code gate complete, results pending

Once local gates pass, use Skynet only for large final verification,
multi-seed paper numbers, and essential robustness runs. Archive exact command
lines, code revisions, dirty-state patches, data/checkpoint hashes, result
audits, and cluster snapshot/job IDs.

The missing corrected-LAFAN1 DiffSR low-level prerequisite is now job
`3503434`, a latent-only matched 5B-frame run with a four-day walltime.
Corrected-LAFAN1 qualification job `3503441` depends on both that job and
direct-vanilla job `3500993`; it
requires both 40-motion oracle success rates to reach 0.8 and certifies the
exact streamed vanilla adapter. The Phase-4 planner array is still not
submitted. Current job chronology and snapshot hashes are recorded in
`wiki/lafan1-from-scratch-comparison.md`.

The Phase-5 staged launcher now writes the snapshot and job identity into each
seed output, and the aggregator binds that record and generates the main
tracking, interface-cost, and oracle-normalized Markdown tables. Final numeric
analysis remains pending until the three qualified Skynet planner runs produce
measurements.

After both aggregate directories exist, build the final reproducibility index
with:

```bash
pixi run python \
  experiments/interface_baselines/build_paper_release_bundle.py \
  --phase4_aggregate logs/interface_baselines/phase4_no_language_lafan1/aggregate \
  --phase5_aggregate logs/interface_baselines/bones_seed_100_multigoal_language_multiseed \
  --output_dir logs/interface_baselines/paper_release_v1
```

The builder refuses an existing output directory. It rehashes every generated
aggregate output, all 360 Phase-4 task-budget audits, the Phase-4 cluster
submission record, and the six required source artifacts for each of the three
Phase-5 seeds. It also requires the exact seed, motion, goal, budget, planner
rate, and target-dimension contracts. Only then does it write
`paper_release_manifest.json` and `paper_release_index.md`. Do not run it with
diagnostic or incomplete aggregate directories.

## Diagnostic and Appendix Interfaces

The repository contains useful scaffolding for EE/keypoint chunks,
Future-CVAE, and per-step latent tokens. Local 2026-07-14 qualifications showed
that the Future-CVAE and per-step-token policies failed their strict oracle
gates after about four steps, even after the corrected-data rerun. These
results are diagnostic and must not be scaled or used to enlarge the main
table.

Keep these variants when they help debug temporal consumption, test a
literature-inspired adapter, or provide a focused appendix ablation. Label
them as style-inspired unless the named method's released tokenizer, action
head, training details, and native timing are actually reproduced.

## Paper Claim Boundaries

A defensible no-language claim is:

> Under the same causal robot history, planner backbone, training data, and
> closed-loop evaluation, we compare a learned DiffSR skill interface with the
> exact future command sequence consumed by a strong frozen vanilla whole-body
> tracker. The latent is a better planning interface only if it reaches a
> larger fraction of its oracle performance with less data, bandwidth, or
> planner compute.

The explicit oracle is expected to equal the direct vanilla ceiling. A lower
learned-planner score can therefore be attributed to generating the
high-dimensional command packet only after provenance, equivalence, and
low-level gates pass.

Do not claim reproduction of GR00T, SONIC, HuMI, LeVERB, or another named
system from command shape alone. Those systems motivate the explicit-command
and latent-action interface question. Native reproductions require their
released components and important training settings.

The primary-source method comparison and the mapping from published systems
to our main and appendix experiments are maintained in
[Whole-Body VLA and Latent-Action Literature Review](whole-body-vla-literature-review.md).

## Reproducibility Record

Every paper result must record:

- exact command line;
- top-level, RLOpt, and ImitationLearningTools revisions and dirty status;
- manifest path and content hash;
- checkpoint paths and SHA-256 hashes;
- planner observation specification and normalization;
- command rates, packet/latent dimensions, and hold behavior;
- random seed and reset schedule;
- exact sample rows, updates, model parameters, and evaluation steps;
- equivalence certificate and focused comparison audit;
- cluster snapshot and job ID for final Skynet runs.

Keep durable rules in `AGENTS.md`, this research contract here, and changing
job chronology in the from-scratch comparison page.
