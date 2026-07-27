# 2026-07-26 DiffSR grouped-VQ capacity ablation

Extends the 2026-07-22 latent-learning study along the two capacity axes of the
grouped codebook. The spectral (DiffSR) objective, h10 horizon, z256 trunk,
50k-update pretrain, frozen encoder, held command, dataset, optimizer profile,
seed, and evaluation protocol are all fixed. Only the group count `G` and the
per-group codebook size `C` move.

The bottleneck is `gumbel_multicat`: `G` independent per-group codebooks of `C`
entries each, per-group Gumbel-softmax with hard straight-through, per-group
code dim `256 / G`. `g64_c128` is the anchor — the configuration that
previously tracked the continuous deterministic latent — and is
protocol-identical to the `gumbel_multicat` arm of the 2026-07-22 DiffSR
bottleneck study.

## Grid (7 arms, seed 0)

| Arm | G | C | code dim | nominal bits/command | encoder params |
| --- | --- | --- | --- | --- | --- |
| `g16_c128` | 16 | 128 | 16 | 112 | 2.92M |
| `g32_c128` | 32 | 128 | 8 | 224 | 3.97M |
| `g64_c128` (anchor) | 64 | 128 | 4 | 448 | 6.08M |
| `g128_c128` | 128 | 128 | 2 | 896 | 10.28M |
| `g64_c16` | 64 | 16 | 4 | 256 | 2.37M |
| `g64_c64` | 64 | 64 | 4 | 384 | 3.96M |
| `g64_c512` | 64 | 512 | 4 | 576 | 18.78M |

Known confound, to be reported rather than hidden: the encoder head is `G * C`
wide, so encoder parameter count is not constant across the grid. Bandwidth,
code dim, and head size all move with `G` and `C`; report all three per arm and
do not attribute a difference to "capacity" alone.

Authoritative design and caveats:
[`wiki/latent-learning-ablation-plan.md`](../../../wiki/latent-learning-ablation-plan.md)
("Study C").

## Entry points

CPU pre-flight over every grid point (build, quantize, checkpoint round-trip):

```bash
experiments/campaigns/2026-07-26-groupvq-capacity-ablation/run.sh check
```

Print the local 10M wiring gate:

```bash
experiments/campaigns/2026-07-26-groupvq-capacity-ablation/run.sh local
```

Run the gate into a fresh output root (sequential, one arm at a time):

```bash
MODE=run \
OUTPUT_ROOT=/absolute/path/to/groupvq_local_gate \
experiments/campaigns/2026-07-26-groupvq-capacity-ablation/run.sh local
```

Preview the seven ICE H200 commands without touching the scheduler:

```bash
experiments/campaigns/2026-07-26-groupvq-capacity-ablation/run.sh ice
```

Exercise the complete gate (approved profile + every passing local record) and
still submit nothing:

```bash
MODE=validate \
LOCAL_QUALIFICATION_ROOT=/absolute/path/to/groupvq_local_gate \
experiments/campaigns/2026-07-26-groupvq-capacity-ablation/run.sh ice
```

Real submission additionally requires the confirmation token:

```bash
MODE=submit \
CONFIRM_SUBMIT=lafan1-groupvq-capacity \
LOCAL_QUALIFICATION_ROOT=/absolute/path/to/groupvq_local_gate \
experiments/campaigns/2026-07-26-groupvq-capacity-ablation/run.sh ice
```

## Execution geometry (2026-07-26)

This grid runs on **coe-gpu H100** at 12,288 envs x 12 steps, minibatch 18,432
(`groupvq_ablation/training_profile.h100.coe.env`), not the H200 profile. Every
H200 GPU on both ice-gpu and coe-gpu was allocated at submission time (one free
cluster-wide) while coe-gpu had about 40 free H100s. An 80 GB H100 cannot hold
the H200 profile's 16,384-environment point, so the geometry follows the
2026-07-22 `h100_e12288_lr1e3` arm instead.

Because of that, **all seven arms including `g64_c128` are re-run here**. The
finished 4.53B H200 `gumbel_multicat` run is at a different env count and
minibatch, so it is not a row of this grid; the 2026-07-22 study stays a
separate table.

One 14h segment covers roughly 4.0B of the 5B cap at the assumed 80k FPS, so
each arm needs one continuation segment.

## Reused, not copied

- Alternate geometry, if H200s free up: `training_profile.h200.approved.env`
  from the 2026-07-22 campaign (one H200, 16,384 envs x 12 steps, minibatch
  24,576, LR 1e-3, 5B frame cap, 15:59:00 segments). Pass it via
  `TRAINING_PROFILE=`; the launcher defaults to it.
- Qualification analyzer: `analyze_local_qualification.py` from the same
  campaign, so both studies share one pass/fail contract.
- Cluster submitter: `submit_hl_skill_pipeline_pace_2b.sh` from the 2026-07-23
  H200 campaign.
