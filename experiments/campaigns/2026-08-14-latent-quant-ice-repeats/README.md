# 2026-08-14 — latent-quant ICE seed repeats / scale-ups

Seed repeats and frame scale-ups of the latent-bottleneck ablation arms
(`../2026-08-13-bones129k-latent-quant-ablation/`) on PACE ICE. One (arm, seed)
submission is a pretrain -> lowlevel `afterok` chain. Data: full BONES-SEED
(129,785 motions) from the shared allocation
`/storage/ice-shared/vip-vwt/g1-imitation`; outputs to
`/data/quant_repeats/<arm>_seed<seed>` on scratch.

## How to run

```bash
./experiments/campaigns/2026-08-14-latent-quant-ice-repeats/submit.sh fsq64 1
# inspect the printed preflight table + plan dir, then run the printed:
pixi run python -m imitation_experiments.pipeline.cluster submit --plan <dir> --confirm <PLAN_SHA>
pixi run python -m imitation_experiments.pipeline.cluster status
pixi run python -m imitation_experiments.pipeline.cluster logs --submission <dir> --stage pretrain
```

Scale-up finals: `./submit.sh fsq64 1 --set vars.frame_cap=2000000000`.
Lowlevel-only resubmission against an encoder already on disk:
`./submit.sh fsq64 1 --only-stage lowlevel`.

`campaign.yaml` is the single declaration of arms, resources, dataset paths,
and preflight requirements. `plan` fails before sbatch when the dataset is not
visible under the job binds, the output root is not writable, quota headroom
is low, or the Slurm log dir is bad — each of these killed jobs on 2026-08-15
(5577430/35/84/98/507).

## Cutover complete (2026-08-15)

`run.sh` is retired (now a deprecation shim pointing at `submit.sh`); the real
ICE smoke chain (jobs 5577564/5577565) validated the control plane
end-to-end, including a live mid-chain rollback (5577560 auto-cancelled when
a bad walltime rejected the lowlevel sbatch — that run found the ICE 16h
walltime cap, since fixed in `campaign.yaml`). `submit.sh` is the only path
now; `docker/cluster/cluster_interface.sh` and everything it used to invoke
are deprecation shims across the whole repo.

## LayerNorm arm validity window (2026-08-15)

Between 2026-08-14 and 2026-08-15, `run.sh` applied `--encoder_layer_norm`
unconditionally, so every arm submitted in that window pretrained WITH
LayerNorm regardless of arm name, and the `_ln` arms were no-op duplicates of
their siblings. Do not aggregate seeds from that window with post-fix seeds of
the non-LN arms; see the note in the 08-13 campaign README.

## Why the ep_len gap opened, and the arms that close it (2026-08-15)

Every arm of the 08-13 100M ablation stalled near ep_len 45 while the 08-09
reference (W&B `2zxhc8su`, `IPMD_bones129k_old_z256_hold1_seed0`) reached
120.2 at the same 100M frames and 199.85 at 3.7B. Four single-variable local
runs at matched 100M frames eliminated the obvious suspects: frame budget
(the reference is 3x ahead at equal frames), reset scheme (`fsq64_oldreset`,
44.46), the tuned reward/curriculum recipe (`fsq64_tuned`, 47.15), and the
macro-window anchor frame (`fsq64_robotanchor`, 45.93). A git diff of every
change since `381b5af` found no regression: the changes are additive.

The two replica arms then reproduced the reference config on current code and
both recovered — `z256_replica` reached ep_len 53.3 at 40M and
`z256_replica_heading` 32.5 at 30M, against reference milestones of 44.2 and
26.9, so both are on or slightly ahead of the reference curve while the 64-D
arms sat near 22 at 40M. That confirms there is no code regression and that
the anchor frame is not the cause, because the `robot_heading` replica
recovers too.

What remains is the latent interface itself, which the ablation changed as a
block: a 258-D `sin_cos` command from a z256 deterministic encoder on the old
trunk (mish, 1024/512/512, feature 128, embed 512) versus a 64-D command with
no phase from the new trunk (silu, 2048/1024/512/512, feature 256, embed
1024). `fsq64_oldtrunk` and `z256_newtrunk` split those two factors: the
first keeps the narrow command and restores the old trunk, the second keeps
the new trunk and restores the wide phased command. Whichever recovers names
the cause.

Until that resolves, treat the 08-13 12-arm oracle table as measured on a
handicapped interface. It ranks bottleneck methods against each other, but it
does not establish what any of them reach on a healthy interface, and the
narrow 41-47 spread across very different bottlenecks is consistent with the
interface, not the bottleneck, being the binding constraint.

### Split result: the trunk is refuted, the code width is the constraint (2026-08-15)

Both split arms are in at 100M frames, and they answer the question cleanly.
`fsq64_oldtrunk` — the narrow 64-D command rebuilt on the OLD trunk — reached
ep_len 51.25 / MPJPE-L 52.70, statistically the same place as `fsq64` on the new
trunk (51.72 at 984M frames). `z256_newtrunk` — the wide command on the NEW
trunk — reached ep_len 91.55 at 68M and is tracking the reference curve, next to
`z256_replica` 121.49 and `z256_replica_heading` 118.92 at the full 100M against
the reference's 120.2. The encoder trunk therefore explains nothing, and the
command the tracker receives per publication explains everything.

A four-way config diff of the lowlevel runs (`rt33shvd`, `2zxhc8su`,
`fsq64_oldtrunk`, `z256_newtrunk`) leaves only three substantive differences:
`actor.dim` / `latent_dim`, `code_latent_dim`, `code_period`, and
`command_phase_mode`. Rewards, terminations, optimizer, selection, and batch
shape are identical across all four.

Quantization is not the factor. `rt33shvd` is discrete FSQ at 64 code dims and
reached ep_len 195.96 / MPJPE-L 30.52 at 5B, and in the other direction
`cont_det` is continuous at 64 dims and stalled at ep_len 53.81 by 350M. Both
64-D arms fail regardless of whether the bottleneck is discrete, and a 64-D
discrete arm succeeds once its hold is 10.

Note also that `command_phase_mode=sin_cos` carries no information at
`code_period=1`. The phase is `(phase_period - latent_steps).clamp(min=0) /
phase_period` with `phase_period = code_period`, so a period of 1 pins it at a
constant and the two appended dims are dead. `2zxhc8su` and both replicas are
effectively 256 informative dims plus two constants, not 258.

So among hold-1 arms, 256 code dims work and 64 do not, and separately a 64-D
code works when its hold is 10. Two arms close the remaining ambiguity:
`fsq64_hold10` (jobs 5577679/5577680) rebuilds `rt33shvd`'s exact interface on
current code, and `fsq256_hold1` (5577687/5577688) widens FSQ to 256 at hold 1.
If both recover, the rule is that the tracker needs a minimum code capacity per
publication, reachable either by widening the code or by lengthening the hold,
and the 08-13 ablation must be rerun at 256 dims before any of its bottleneck
comparisons mean anything.

### Full grid, 100M frames, one seed each (2026-08-15)

All nine arms finished at exactly 100,270,080 frames on ICE, same reward set,
same terminations, same optimizer, same 129k dataset, `robot_heading` anchor,
LayerNorm off except where noted. `ep_len` is the discriminator.

| arm | width | quantizer | command boundary | hold | ep_len |
|---|---:|---|---|---:|---:|
| `z256_newtrunk` | 256 | none | continuous | 1 | 123.76 |
| `z256_replica` | 256 | none | continuous | 1 | 121.49 |
| `z256_replica_heading` | 256 | none | continuous | 1 | 118.92 |
| `fsq256_hold1` | 256 | FSQ 32 levels | lattice, Identity | 1 | 110.71 |
| `fsq64_hold10` | 64 | FSQ 32 levels | lattice, Identity | 10 | 102.99 |
| `fsq64_oldtrunk` (LN on) | 64 | FSQ 32 levels | lattice, Identity | 1 | 51.25 |
| `fsq64` | 64 | FSQ 32 levels | lattice, Identity | 1 | 50.53 |
| `cont_det` | 64 | none | continuous | 1 | 38.49 |
| `fsq64_proj` | 64 | FSQ 32 levels | `nn.Linear(64, 64)` | 1 | 26.52 |

Reference for scale: `2zxhc8su` reached 120.2 at the same 100M frames.

Four things fall out, and one of them reverses a working assumption.

**Command width at hold 1 is the dominant factor.** Every 256-wide arm sits at
110-124; every 64-wide hold-1 arm sits at 27-51. Nothing else in the grid moves
an arm across that gap.

**Quantization is not the factor, and at 64 dims it helps.** At 256, FSQ 110.71
against continuous 118.92-123.76 — the same regime. At 64, FSQ 50.53 beats
continuous `cont_det` 38.49. Discreteness was never the handicap.

**A longer hold rescues a 64-D code, and this is why `rt33shvd` worked.**
`fsq64_hold10` reaches 102.99 against `fsq64`'s 50.53 from a byte-identical
encoder pretrain — the two differ only in `code_period`, republication cadence,
and whether the `sin_cos` phase is live. So `rt33shvd`'s 195.96 at 5B came from
its hold of 10, not from its trunk, not from LayerNorm, and not from a
different codebook: its `sonic_fsq_levels` is the same 64 entries of 32.

**LayerNorm is irrelevant here.** `fsq64_oldtrunk` with LN on lands at 51.25
against `fsq64` with LN off at 50.53, so the pre-quantization-saturation worry
is refuted for this bottleneck.

**The learned projection at the command boundary hurts, badly.** `fsq64_proj`
is the worst arm in the grid at 26.52, half of `fsq64`'s 50.53, differing only
in `code_to_latent` being `nn.Linear(64, 64)` instead of `nn.Identity()`.
Publishing the raw lattice beats publishing a projection of it. This is the
opposite of what the projection was expected to do, and it is one seed, so
treat it as a lead rather than a result.

Read the MPJPE columns with care and do not rank arms by them. An arm that
terminates at ep_len 27 only ever reports error over the easy early frames of
an episode, so MPJPE across rows with different ep_len is not a like-for-like
comparison; the 45.8-54.7 spread across this grid is an artifact of that, not a
finding. These are training-time metrics at one seed, not the frozen oracle
protocol.

The practical rule: the tracker needs either a wide command or a long hold. A
64-D code republished every step is below what it can use, and the 08-13 12-arm
ablation compared bottleneck methods entirely inside that dead zone.
