# LAFAN1 Latent-Learning Ablation Plan

Status: all twelve local qualification arms passed on 2026-07-22, and the H200
jobs subsequently ran. As verified on 2026-07-26, eleven of the twelve arms
reached 4,525,129,728 frames on ICE; the twelfth, `continuous_ae`, has a run
directory with no checkpoints and needs a separate look before it can be
reported. The approved H200 profile is based on the independent ICE screen
(16,384 environments x 12 steps sustained about 90.4k FPS) and the launch
wrapper still requires all twelve qualification records.

Checkpoint density note: on 2026-07-26, ICE scratch filled and the eleven
finished runs were thinned from 181 to 84 checkpoints each, keeping every
~100M-frame checkpoint plus each run's final one. Selecting the converged
checkpoint therefore resolves to 100M, not the 25M save interval those jobs
used. Metric CSVs and final checkpoints are intact.

Local qualification is now an additional mandatory gate. Every arm must train
for at least 10M frames on the corrected local LAFAN1 cache, produce a loadable
checkpoint, retain finite metrics, and show either a 10% episode-length or 2%
per-step-reward improvement between its first and final 20% windows without a
catastrophic regression in the other signal. These thresholds detect broken
wiring and an early learning signal; they are not convergence criteria.
For DiffSR arms, this local wiring gate uses 5k encoder-pretraining updates so
all modes can be exercised promptly. Production H200 jobs still use the full
50k-update encoder pretraining budget specified below.

## Dataset and fixed protocol

“LaForge” does not exist in this workspace. This plan interprets it as the
corrected 40-motion LAFAN1 dataset already used by the FB- and EE-chunk command
ablations:

```text
/data/lafan1_corrected_8e95d557/manifests/g1_lafan1_manifest.json
/data/lafan1_corrected_8e95d557/g1_hl_diffsr
manifest sha256: d972c37c41dadbb68c30fc456a9dc9c1bd6d30ed0b7aa9d34b1797472c945db8
```

Every row uses the strict pelvis-anchored LAFAN1 protocol, h10 (current plus
nine future reference frames), a command held for ten 50 Hz control steps, the
same policy/critic architecture, optimizer profile, reset distribution, seed,
and evaluation starts. The earlier FSQ/SONIC jobs are not evidence against
those objectives: both changed to per-step latent renewal, and the controlled
hold experiment identified that renewal change as the cause of collapse.

## Study A: reconstruction-trained families

These rows use the new ablation-only task
`Isaac-Imitation-G1-Latent-Ablation-v0`. VQ-VAE and FSQ explicitly optimize
future-window reconstruction, as requested.

| Row | Bottleneck/objective | Policy gradient into encoder? |
| --- | --- | --- |
| Continuous AE | identity quantizer, 64-D code, reconstruction | no |
| VQ-VAE | EMA VQ, K=512, 64-D embedding, reconstruction + commitment | no |
| FSQ reconstruction | 5 scalar dimensions x 4 levels, projected to 64-D, reconstruction | no |
| SONIC-style objective | same FSQ and reconstruction as preceding row | yes |
| Conditional VAE | 64-D Gaussian posterior, current-frame conditional prior, reconstruction + KL | no |

The “SONIC-style” label is deliberately narrow: it tests the user-specified
FSQ + reconstruction + PG objective while holding our environment and policy
fixed. It is not a claim to reproduce SONIC's complete architecture, data,
reset sampling, or 320-bit released token space.

SONIC's collector and optimizer cadence are deliberately separated. The
collector holds one FSQ code for ten control steps and changes only its
sin/cos phase inside that chunk. During PPO, the policy forward pass uses the
exact command saved in the rollout. A straight-through surrogate sends policy
gradient into the posterior encoder only on code-renewal transitions; it does
not re-encode or replace every stored transition command. This preserves the
5 Hz command contract and avoids ten redundant encoder PG passes per chunk.

The established multi-group budget is 64 groups x 128 choices, or
`64 * log2(128) = 448` nominal bits per command. The current online
reconstruction quantizers cannot represent that comparison fairly: online VQ
has one codebook, and its FSQ implementation collapses all scalar symbols into
one joint integer index whose cardinality would overflow at this scale.
Therefore the VQ-VAE/FSQ/SONIC rows above are an objective-isolation pilot
only. FSQ reconstruction versus SONIC is still controlled because both use
the exact same bottleneck, but neither should be compared to the 448-bit rows
as a capacity-matched result.

The paper-facing reconstruction comparison requires a grouped quantizer
extension before launch: 64 independent groups, 128 entries per group, and a
4-D embedding per group concatenated to the common 256-D command. Both grouped
VQ-VAE and grouped FSQ/SONIC must expose per-group usage and perplexity without
packing the 448-bit tuple into a single integer. The launcher remains a hard-
gated pilot until that implementation and checkpoint round-trip test exist.

## Study B: DiffSR bottlenecks

All rows use the same DiffSR loss, h10/z256 trunk, 50k pretraining updates,
frozen encoder during controller learning, and held z + sin/cos phase command.
The implemented choices are:

| Mode | Meaning | Primary status |
| --- | --- | --- |
| `deterministic` | continuous z with L2 regularization; current default | core |
| `gaussian` | continuous stochastic posterior with KL regularization | core |
| `categorical` | hard straight-through grouped categorical | core |
| `gumbel_multicat` | grouped categorical with annealed Gumbel-softmax | core |
| `gumbel` | one K=512 Gumbel codebook | lower-capacity diagnostic |
| `fsq` | finite scalar quantization | core |
| `vq` | one K=512 EMA vector quantizer + commitment | lower-capacity diagnostic |

The primary implemented discrete settings now match the established capacity:
grouped categorical and grouped Gumbel use 64 groups x 128 categories, and FSQ
uses 64 scalar symbols x 128 levels. Each has 448 nominal bits. With z256, the
grouped categorical variants assign four embedding dimensions to each group;
FSQ is projected to the same 256-D controller command. The common phase clock
is appended afterward.

The existing single-code `gumbel` and `vq` modes have K=512, only 9 nominal
bits, and are retained only as explicitly tagged lower-capacity diagnostics.
They cannot support a capacity-matched conclusion. An exact VQ row requires a
new grouped EMA-VQ mode with 64 independent 128-entry codebooks and 4-D
embeddings per group. A flat VQ codebook with `2^448` entries is neither the
same factorization nor computationally possible, so the plan will not pretend
that K=512 is a fair substitute.

Continuous deterministic/Gaussian rows have no directly comparable nominal
bit count. Report their 256 floating-point values and measured entropy or
effective-rank diagnostics instead of assigning them a fictitious 448-bit
capacity.

## Study C: grouped-VQ capacity ablation (added 2026-07-26)

Study B asks which bottleneck *family* to use at one fixed capacity. Study C
holds the family fixed at the grouped product codebook that previously tracked
the continuous deterministic latent and moves only its two capacity axes: the
number of groups `G` and the per-group codebook size `C`.

Bottleneck is `gumbel_multicat` for every row — `G` independent per-group
codebooks of `C` entries, per-group Gumbel-softmax with hard straight-through,
per-group code dim `256 / G`. Everything else is the Study B protocol: DiffSR
spectral objective, h10, z256 trunk, 1024/512/512 encoder, 50k pretraining
updates, frozen encoder during controller learning, held z + sin/cos phase,
corrected LAFAN1, seed 0, and the approved H200 geometry.

| Arm | G | C | code dim | nominal bits/command | encoder params |
| --- | --- | --- | --- | --- | --- |
| `g16_c128` | 16 | 128 | 16 | 112 | 2.92M |
| `g32_c128` | 32 | 128 | 8 | 224 | 3.97M |
| `g64_c128` (anchor) | 64 | 128 | 4 | 448 | 6.08M |
| `g128_c128` | 128 | 128 | 2 | 896 | 10.28M |
| `g64_c16` | 64 | 16 | 4 | 256 | 2.37M |
| `g64_c64` | 64 | 64 | 4 | 384 | 3.96M |
| `g64_c512` | 64 | 512 | 4 | 576 | 18.78M |

`g64_c128` is protocol-identical to the Study B `gumbel_multicat` arm and is
the reference point for "close to continuous deterministic". If that Study B
job is ever run, its checkpoint may be reused instead of re-running the anchor;
they may not both enter one table as independent rows.

Three quantities move together in this grid and none may be reported alone as
"capacity": nominal bandwidth `G * log2(C)`, per-group code dim `256 / G`, and
encoder head width `G * C` (hence total encoder parameters, which span 2.4M to
18.8M here). Report all three per arm alongside per-group code usage and
perplexity. The current grouped encoder pools its usage/perplexity diagnostics
over groups rather than reporting them per group; a per-group breakdown is
still owed before any paper-facing claim about which groups collapse.

The grid is not a substitute for the missing grouped EMA-VQ mode noted in
Study B. `gumbel_multicat` selects codes from per-group logits, not by
nearest-neighbour assignment, so Study C ablates the capacity of our grouped
product codebook, not of a classical VQ quantizer.

Launchers and gate:

```text
experiments/campaigns/2026-07-26-groupvq-capacity-ablation/run.sh check
experiments/campaigns/2026-07-26-groupvq-capacity-ablation/run.sh local
experiments/campaigns/2026-07-26-groupvq-capacity-ablation/run.sh ice
```

`check` is a seconds-long CPU pre-flight that builds, quantizes, and
round-trips every grid point. `local` is the same 10M-frame wiring gate the
Study A/B arms passed, with 5k pretraining updates. `ice` defaults to
`MODE=print`; `MODE=validate` and `MODE=submit` both require the approved H200
profile plus a complete passing local qualification root, and `MODE=submit`
additionally requires `CONFIRM_SUBMIT=lafan1-groupvq-capacity`. At the approved
geometry one 15:59:00 segment covers about 4.54B of the 5B cap, so each arm
needs one continuation segment via `TRAIN_CHECKPOINT`, `PRETRAINED_CHECKPOINT`,
and `COMPLETED_FRAMES`.

## Convergence and reporting

An arbitrary 500M-frame snapshot is a qualification check, not a comparison.
For each row:

1. Smoke one rollout iteration and validate checkpoint round-trip, finite
   reconstruction/latent losses, and non-collapsed code diagnostics.
2. Train seed 0 with checkpoints every 25M frames until both episodic return
   and episode length plateau. The default cap is 5B frames; reaching the cap
   without plateau is reported as non-converged, not treated as a final score.
3. Select the converged checkpoint rather than automatically selecting the
   last checkpoint. Run the same strict oracle evaluation and the required
   no-early-termination full-horizon diagnostic/video.
4. Repeat the surviving core rows for seeds 1 and 2. Report wall-clock time and
   frames to plateau, final return/length, survival, MPJPE/root/joint/EE error,
   action smoothness, velocity/acceleration error, effective rank, posterior
   statistics, reconstruction error, code perplexity/usage, and bandwidth.

No row may borrow another row's checkpoint, encoder, dataset cache, or planner
samples. A collapsed row remains a failed result; qualification thresholds are
not weakened to retain it.

## Launchers and gate

```text
experiments/campaigns/2026-07-22-latent-learning-ablation/latent_ablation/submit_lafan1_reconstruction_ablation_ice.sh
experiments/campaigns/2026-07-22-latent-learning-ablation/latent_ablation/submit_lafan1_diffsr_bottleneck_ablation_ice.sh
experiments/campaigns/2026-07-22-latent-learning-ablation/latent_ablation/training_profile.example.env
```

Both launchers default to `MODE=print`. Real submission requires a copied
profile containing `PROFILE_APPROVED=1`, plus:

```bash
MODE=submit \
CONFIRM_SUBMIT=lafan1-latent-ablation \
TRAINING_PROFILE=/absolute/path/to/approved.env \
experiments/campaigns/2026-07-22-latent-learning-ablation/latent_ablation/submit_lafan1_reconstruction_ablation_ice.sh
```

The approved profile was selected from the independent H100/H200 and
learning-rate wall-clock study. The H200 profile is
`training_profile.h200.approved.env`; the example and pending profiles remain
non-submittable by design.

The prepared H200 profile uses one H200, 16,384 environments x 12 rollout
steps, minibatches of 24,576, actor/critic LR `1e-3`, and centralized normal
RLOpt logs. Preview all twelve commands with:

```bash
MODE=print \
experiments/campaigns/2026-07-22-latent-learning-ablation/latent_ablation/submit_all_h200_after_local_qualification.sh
```

Actual submission additionally requires all twelve passing
`qualification.json` records under one `LOCAL_QUALIFICATION_ROOT`; the wrapper
refuses missing or failed arms. Generate those records with:

```bash
MODE=run \
OUTPUT_ROOT=/absolute/path/to/local_10m_gate \
experiments/campaigns/2026-07-22-latent-learning-ablation/latent_ablation/run_lafan1_local_10m_qualification.sh
```

Before submission, exercise the complete gate and print the exact twelve
commands without changing scheduler state:

```bash
MODE=validate \
LOCAL_QUALIFICATION_ROOT=/absolute/path/to/local_10m_gate \
TRAINING_PROFILE=experiments/campaigns/2026-07-22-latent-learning-ablation/latent_ablation/training_profile.h200.approved.env \
experiments/campaigns/2026-07-22-latent-learning-ablation/latent_ablation/submit_all_h200_after_local_qualification.sh
```

If an arm has not plateaued before the 16-hour boundary, both launchers accept
`TRAIN_CHECKPOINT` and `COMPLETED_FRAMES` for a single-arm continuation.
DiffSR continuation additionally requires `PRETRAINED_CHECKPOINT`, ensuring the
controller is rebuilt against the exact original encoder before its full
checkpoint state is restored.
