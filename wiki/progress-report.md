# Project Progress Report

A regularly-updated results-facing summary in three fixed sections:

1. latent encoder experiments and ablations;
2. interface design experiments;
3. hardware.

This page records **results and their interpretation**. Live scheduler state,
job chronology, and incident history stay in
[`current-status.md`](current-status.md); frozen protocols stay in the phase
documents. Update this page whenever a campaign produces or invalidates a
result, and stamp the section with the verification date.

Last updated: 2026-07-30.

---

## 1. Latent encoder experiments and ablations

Verified: 2026-07-29 (Stable-vs-Strict 500M inference diagnostic complete).

### Question

Which latent-learning objective produces the best low-level command interface:
a successor-representation objective (DiffSR, with various bottlenecks) or a
future-window reconstruction objective (AE / VQ-VAE / FSQ / CVAE families)?

Design, capacity caveats, and launchers:
[`latent-learning-ablation-plan.md`](latent-learning-ablation-plan.md);
campaign front door:
`experiments/campaigns/2026-07-22-latent-learning-ablation/`.

### Result: all twelve local 10M qualification arms passed

Shared protocol: corrected 40-motion LAFAN1, strict pelvis-anchored surface,
h10 encoder window, command held for ten 50 Hz steps, identical policy,
optimizer, seed, and resets. Each arm trained ~10M frames locally. Numbers are
medians over the first/final 20% metric windows; the gate detects wiring and
an early learning signal, **not convergence**.

Qualification root:
`logs/latent_ablation/local_10m_h200_gate_20260722/<arm>/qualification.json`.

| Arm | Objective family | Final ep_len | Final r_ep | Initial ep_len | Note |
| --- | --- | ---: | ---: | ---: | --- |
| `gumbel` | DiffSR, single K=512 Gumbel codebook | 28.05 | 0.832 | 4.36 | best at 10M, but a 9-bit lower-capacity diagnostic |
| `vqvae` | reconstruction, EMA VQ K=512 | 26.39 | 0.780 | 4.09 | best reconstruction arm |
| `deterministic` | DiffSR, continuous z + L2 | 25.47 | 0.642 | 4.34 | current default |
| `continuous_ae` | reconstruction, identity bottleneck | 25.34 | 0.672 | 4.10 | |
| `fsq_recon` | reconstruction, FSQ 5x4 | 24.63 | 0.699 | 4.12 | |
| `gaussian` | DiffSR, Gaussian posterior + KL | 23.85 | 0.600 | 4.11 | |
| `gumbel_multicat` | DiffSR, grouped Gumbel 64x128 | 22.43 | 0.560 | 4.15 | 448-bit core arm |
| `sonic_fsq_pg` | reconstruction + policy gradient into encoder | 21.73 | 0.545 | 4.21 | SONIC-style objective isolation |
| `categorical` | DiffSR, grouped hard categorical 64x128 | 21.25 | 0.526 | 4.09 | 448-bit core arm |
| `cvae` | reconstruction, conditional VAE + KL | 10.07 | 0.122 | 4.08 | learns, clearly weaker |
| `fsq` | DiffSR, FSQ 64x128 | 4.00 | 0.006 | 3.51 | near-flat; passed on reward sign only |
| `vq` | DiffSR, single K=512 EMA VQ | 3.29 | -0.035 | 2.73 | near-flat lower-capacity diagnostic |

### Interpretation (early-signal only)

- Both objective families produce a learnable command interface: the top
  reconstruction arms (`vqvae`, `fsq_recon`, `continuous_ae`) are
  interleaved with the top DiffSR arms at 10M frames. SR vs reconstruction is
  not separable at this budget; the H200 convergence runs decide it.
- The two near-flat arms are both quantized DiffSR bottlenecks: `fsq`
  (despite 448-bit capacity) and the low-capacity `vq` diagnostic. DiffSR-FSQ
  should be watched as a possible bottleneck/objective mismatch rather than
  dismissed — it passed the gate formally but shows no episode-length growth.
- `gumbel` leading at 10M is not a capacity conclusion; it has only 9 nominal
  bits and is tagged as a diagnostic. Early speed may simply reflect an easier
  optimization landscape.
- Adding policy gradient into the encoder (`sonic_fsq_pg` vs `fsq_recon`)
  did not help at this budget (21.73 vs 24.63 ep_len).

### Stable-vs-Strict low-level inference at ~500M

ICE retained both checkpoints needed for this diagnostic. Completed job
`5542378` supplied the new Stable/SONIC
`Isaac-Imitation-G1-Latent-v0` checkpoint at 500,072,448 frames; the earlier
Strict run supplied `Isaac-Imitation-G1-Latent-Strict-v0` at 500,170,752
frames. Both use the same tensor-identical h10 DiffSR skill encoder
(`5c84ff72...264ea`).

Matched full-horizon model inference covered all 40 corrected LAFAN1 motions
for 1,000 steps each, with deterministic tracking and every early termination
disabled:

| Recipe | Root-relative MPJPE | Joint RMSE | Velocity | Acceleration | Action change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stable/SONIC, `4096 x 24` | 111.17 mm | 0.303 rad | 0.654 m/s | 11.884 m/s2 | 1.692 |
| Stable/SONIC, `16384 x 12` | 112.00 mm | 0.295 rad | 0.627 m/s | 10.787 m/s2 | 1.428 |
| Strict | 129.84 mm | 0.275 rad | 0.576 m/s | 8.009 m/s2 | 1.085 |

The same-geometry Stable row is only 0.74% above the original Stable result,
so training geometry was not driving the observed MPJPE difference. It remains
13.74% below Strict, while joint and temporal metrics are worse. This is still
a diagnostic trend rather than a claim because it is below the roughly 15%
resolution threshold suggested by earlier repeated inference. The new
strict-termination pass measured 33.26 mm and 0.23 success; as before, that
MPJPE is computed only over valid transitions before/through unequal
terminations, reinforcing why the 40,000-sample full-horizon comparison is
primary.

### Status and next step

The guarded H200 submission (`submit_all_h200_after_local_qualification.sh`,
profile `training_profile.h200.approved.env`: one H200, 16,384 envs x 12
steps, minibatch 24,576) is validated but **deliberately not submitted**.
Convergence-based comparison, seeds 1-2 for surviving core rows, and the
grouped-quantizer extension for a capacity-fair reconstruction comparison
remain per the plan.

### MPJPE plan for this ablation (to do)

Episode length and return are proxies; the concrete tracking metric is
root-relative MPJPE, already implemented as `tracking_mpjpe_m[m]` in the
shared oracle evaluator
(`experiments/campaigns/2026-07-23-bones-phase5-language-local10/command_space_ablation/evaluate_checkpoint.py`).

1. **Offline, now (local):** evaluate each of the twelve 10M qualification
   checkpoints with the shared evaluator on the matched task
   (`Isaac-Imitation-G1-Latent-Ablation-v0` for reconstruction arms, the
   latent surface for DiffSR arms), fixed 100 envs / 1000 steps / seed 0, and
   additionally the mandatory full-horizon no-early-termination diagnostic
   pass. Write `eval_mpjpe.json` beside each `qualification.json` and
   aggregate one ranked table into the campaign directory. Verify first that
   the evaluator can rebuild each arm's encoder/quantizer wiring; it was
   built for the oracle command-space paths.
2. **Online, before H200 submission:** the environment already maintains
   MPJPE accounting (episode accounting across async resets fixed in
   `af73212`); confirm the term reaches the RLOpt training logger/W&B for the
   ablation tasks so every H200 arm logs MPJPE continuously, and extend
   `analyze_local_qualification.py` to record it (parse it from the metrics
   stream rather than adding a second stdout regex if a structured source
   exists).
3. **Reporting:** final comparison keeps the full frozen metric set (survival,
   MPJPE, root/joint/EE error, action change, velocity/acceleration error,
   code usage/perplexity, effective rank) at the converged checkpoint, per
   the ablation plan.

---

## 2. Interface design experiments

Verified: 2026-07-30.

### Question

Is a learned latent skill command a better high-level planner interface than
the explicit action/state chunks used by current humanoid VLA systems?
Authoritative protocol:
[`causal-interface-paper-plan.md`](causal-interface-paper-plan.md).

### Frozen main comparison

Two rows only, both consuming a frozen 50 Hz tracker from a causal `10 x 93`
robot-history planner input at 5 Hz: DiffSR 256-value latent vs an explicit
670-value packet of ten vanilla full-body commands. The direct 50 Hz vanilla
tracker is the low-level ceiling, not a row.

### Results so far

- **Streamed-vanilla equivalence certificate (BONES-SEED): passed.** All ten
  packet phases, asynchronous renewal, and policy immutability; max command /
  action differences `3.02e-7` / `1.31e-6`. The explicit-packet row is
  therefore a faithful re-timing of the qualified vanilla tracker.
- **Phase-5 low-level qualification (BONES-SEED-100, job `3512041`): passed.**
  Direct vanilla strict success 0.90, DiffSR latent 0.84 (gate 0.80), with
  the skill encoder tensor-bound to the latent checkpoint.
- **One-motion planner diagnostics (`walk1_subject1`):** causal planners work
  for both interfaces; at tiny scale the latent interface is stronger across
  flow, diffusion, and deterministic objectives; the three-seed flow
  diagnostic reaches the fixed target at ~`0.13M` parameters (latent) vs
  ~`4.19M` (explicit); explicit often catches up or wins on MPJPE at larger
  sizes; rollout fine-tuning frequently hurts one-motion tracking. Working
  interpretation: the latent interface may reduce required planner capacity,
  not necessarily raise the tracking ceiling. Not paper claims until repeated
  across motions.
- **Phase 4 (LAFAN1, no language):** planner grid still blocked on the paired
  low-level oracle audits; low-level prerequisites were re-launched after the
  joint-order fix forced retraining.
- **Enc380 content-controlled diagnostic:** the durable 5B root+qpos-content
  latent tracker is underqualified on the same historical Strict-v0 environment
  used for training. Its all-40 strict oracle success is 0.35 against the fixed
  0.80 gate despite 1.0 fall-free survival; the corrected all-40,
  non-terminating pass measures 102.76 mm MPJPE over 40,000 transitions. No
  planner stage ran. This rules out comparative planner claims from this
  checkpoint unless a new low-level training budget or recipe is explicitly
  approved.
- **Phase 5 (BONES-SEED, language):** first three-seed planner chains failed
  in preparation (compute-local disk exhaustion + four motions below the
  1,000-row target); the revised default budget is 150 demo + 150 rollout
  rows per goal. A latent-only, `preliminary_unqualified=true` H200 pilot
  chain was submitted 2026-07-23 (jobs `3560697`-`3560701`); it cannot enter
  the paper aggregate.

---

## 3. Hardware

Verified: 2026-07-22.

### Compute hardware (training-throughput findings)

- **H200 is the approved production profile:** 16,384 envs x 12 rollout steps
  sustained ~90.4k FPS, ~21% above H100 at 12,288 x 12; minibatch 24,576,
  actor/critic LR `1e-3`. Encoded in
  `experiments/campaigns/2026-07-22-latent-learning-ablation/latent_ablation/training_profile.h200.approved.env`.
- **Blackwell (RTX Pro 6000, `ice-bw-gpu`) works but is 48 GB:** the software
  stack (kernels, Newton, pretrain) is clean, yet the 12,288-env scaled
  config OOMs; H100/H200 80/141 GB remain the usable partitions for the
  scaled config. 16,384 envs does not fit the SONIC release-size network on
  one H100.
- **ICE walltime is hard-capped at 16h GPU-minutes per job** regardless of
  partition/QoS; long runs must be resumable multi-segment chains with
  iteration counts capped to exit before the wall.
- **Checkpoint durability is fixed cluster-wide:** Slurm TIMEOUT previously
  destroyed node-local output (~48 GPU-hours lost on 2026-07-21); every
  container profile now binds a persistent central log root, verified by
  smoke job `5526584`. Checkpoints no longer depend on exit-handler syncs.
- **Newton contact capacity is a per-step complexity budget, not an env-count
  scaling:** njmax=320/nconmax=40 is the validated setting; proportional
  scaling by env count was wrong (7.4M overflow events in one "successful"
  arm before the fix).

### Robot hardware readiness (sim2sim, transfer)

No real-robot deployment yet; current work is the sim2sim gate that precedes
it.

- **Newton joint-order leak found and fixed (2026-07-21):** expert-command
  observations and the action offset used the live articulation order (27/29
  slots differ between PhysX and Newton). All pre-fix Newton checkpoints are
  invalidated; the index contract is now pinned and regression-tested on both
  backends.
- **Two additional cross-backend bugs fixed:** stale derived body-frame state
  after reset (first post-reset observation was stale/zero on all prior
  training) and a PhysX solver-iteration override (8/4 instead of the
  asset's 32/1).
- **A genuine dynamics gap remains after all ordering fixes:** a matched
  checkpoint survives fully on Newton (0.126 rad joint error) but falls at
  5.36 s on PhysX (0.242 rad). Decision: re-freeze the protocol on the
  randomized event config for **every** experiment, sequenced with the
  retraining already forced by the joint-order fix. See
  [`sim2sim-dynamics-gap-and-randomization.md`](sim2sim-dynamics-gap-and-randomization.md).

---

## Update cadence

- After every campaign result, qualification pass/fail, or invalidation:
  update the affected section and its "Verified" date.
- Keep tables backed by machine-readable artifacts (`qualification.json`,
  audit JSON, aggregation manifests); never transcribe unaudited numbers.
- When a section's content becomes historical, move the detail into the
  matching phase document and keep only the current conclusion here.
