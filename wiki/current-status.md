# Project Live Status

Last verified: 2026-07-20, after making the SONIC surface + release policy
contract the confirmed default (H100/H200 single-GPU access removes the
compute-scale objection) and preparing the first ICE H100 VRAM ablation and
BONES-SEED SONIC-latent submissions at the new ~10B / 2B-cap scales.

This is the living memory for the active research project. Read it first when
returning to the project or starting a new agent session. It answers **where we
are now**. The detailed protocol and experiment history remain in the linked
phase documents.

Update this page after a meaningful code decision, qualification result,
cluster submission, job failure, or paper result. Verify changing external
state such as Slurm jobs before treating a status below as current. Keep old
chronology in the phase-specific pages instead of allowing this page to grow
without bound.

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

Data-budget interpretation is important: one saved row is one 5 Hz planner
decision containing a ten-frame 50 Hz command chunk. The failed configuration
requested 100,000 demonstration rows plus 100,000 rollout rows per interface,
then fine-tuned on 200,000 merged rows. It was not a small dataset.

The current recommended Phase 5 budget is:

- 15,000 balanced demonstration rows total: 150 per goal;
- 15,000 planner-rollout rows total: 150 per goal;
- 30,000 unique rows in the merged fine-tuning dataset.

The old 100,000 plus 100,000 configuration should become an optional
large-data scaling point, not the default paper run. This budget change has
not yet been encoded into a replacement launcher or submitted. Before doing
so, verify that the four difficult motions can reach 150 rows and increase the
collection safety limit without changing the 500-step episode protocol.

Data preparation and hashes:
[BONES-SEED Phase-5 Data Preparation](bones-seed-phase5-data-preparation.md).

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
