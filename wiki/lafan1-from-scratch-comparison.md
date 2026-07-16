# LAFAN1 From-Scratch Interface Comparison

Redo of the PR#19 LAFAN1 motion-tracking comparison with every low-level
policy trained from scratch, so the comparison measures interface quality
instead of low-level checkpoint provenance. A fourth, deliberately basic
`single_frame_full_body` policy supplies the direct 50 Hz reference-tracking
ceiling against which the 5 Hz latent skill oracle can be interpreted.

The earlier three-interface redo is retained below as experiment history. It
is not the current paper scope. The active main planner comparison is defined
in [Causal Interface Paper Plan](causal-interface-paper-plan.md): DiffSR latent
at 5 Hz versus a ten-frame packet of exact vanilla full-body commands at 5 Hz,
both driven by the same causal planner inputs. EE chunks and other latent
variants are diagnostics or appendix rows. Direct vanilla at 50 Hz is the
low-level ceiling, not a planner row.

## Corrected data state after the 2026-07-14 invalidation

The defect described below has been fixed in the current 40-motion local NPZ
tree. The earlier converter saved rigid-body positions in Isaac's shared world
frame, including each environment's scene-grid origin, while saving root
positions in the motion frame. It also mixed Unitree SDK joint order with
Isaac articulation order. All policies and caches made from that older tree
remain invalid even though the paths have since been corrected.

This defect removes meaningful body-position reward and inflated local MPJPE
from about 134 mm on repaired motions to about 4,782 mm. The four Skynet jobs
listed below used the affected data path and must be treated as invalid for
the paper regardless of how far they trained. Do not submit follow-up
evaluation jobs from those checkpoints.

The corrected exporter now removes scene origins, calls `sim.forward()` before
reading rigid-body transforms, saves joints in Isaac articulation order, and
records both joint and body names. The current local dataset identity is:

| Item | Value |
| --- | --- |
| Manifest | `data/lafan1/manifests/g1_lafan1_manifest.json` |
| Motions | 40 |
| Manifest SHA-256 | `d972c37c41dadbb68c30fc456a9dc9c1bd6d30ed0b7aa9d34b1797472c945db8` |
| Aggregate NPZ SHA-256 | `8029acbce33ae49f0847cee8b894bcfd3f77afafa649ce1e89f5eafbf39916e2` |

The body-frame audit passes 40/40 motions. A fixed-start replay of
`walk1_subject1` also matched all 32 stored body positions against independent
Isaac FK within `2.3e-7 m`. The stale `data/lafan1/g1_hl_diffsr` cache was not
reused; local qualification uses the fresh content-specific cache
`/tmp/iltools_g1_lafan1_tracking_corrected_8029acbce33a`.

The code provides two guards:

```bash
pixi run python scripts/audit_g1_lafan1_body_frames.py \
  --manifest /path/to/manifest.json \
  --report /path/to/body_frame_audit.json

pixi run python scripts/repair_g1_lafan1_body_offsets.py \
  --input_dir data/lafan1/npz/g1 \
  --output_dir /path/to/separate_corrected_npz
```

The repair tool never edits in place. Prefer regeneration with the fixed
`batch_csv_to_npz.py`. Always rebuild derived Zarr data after an NPZ change.

### Corrected-data Skynet vanilla ceiling

The corrected 40-motion dataset was published to
`GeorgiaTech/g1_lafan1_50hz` at Hugging Face commit
`8e95d557dbf6720fa49eb45a8726e53515f72d61`. The upload contains the 40
corrected NPZs, relative manifest, body-frame audit, and the previously
generated language commands in exact manifest order. The vanilla run is not
language-conditioned.

Skynet job `3500993` is the replacement direct-reference ceiling run. It uses
the same 5B-frame budget and direct `single_frame_full_body` setup as invalid
job `3496079`, but downloads into the new persistent root
`/nethome/fwu91/scratch/Research/IsaacLab/data/lafan1_corrected_8e95d557` and
builds a new `g1_hl_diffsr` cache there. It was running on `synapse` from the
fresh snapshot `isaaclab_20260714_221425` when last checked on 2026-07-14.
Job `3500636` was an erroneous BONES-SEED submission and was cancelled while
still pending. Job `3500648` used the correct LAFAN1 arguments but failed before
data setup because a reused snapshot collided on its writable overlay. Neither
failed attempt produced a training result.

### Corrected-data local vanilla qualification

The direct 50 Hz vanilla tracker was trained in 10M-frame blocks on the
corrected data, with no command-side expert noise and no BC. The agreed local
serious-check budget is approximately 50M total frames:

| Cumulative frames | Mean strict survival (of 1000) | Done rate | Joint RMSE | Status |
| ---: | ---: | ---: | ---: | --- |
| 10.027M | 53.4 | 1.0 | 0.2095 rad | Fails low-level gate |
| 20.054M | 240.2 | 1.0 | 0.1943 rad | Improving, still fails low-level gate |
| 30.081M | 359.7 | 0.975 | 0.1911 rad | Improving, still fails low-level gate |
| 40.108M | 471.0 | 0.900 | 0.1927 rad | 4/40 complete; still fails low-level gate |
| 50.135M | 534.3 | 0.875 | 0.1953 rad | 5/40 complete; sufficient local code check |

The 50M checkpoint is under
`logs/rlopt/ipmd/Isaac-Imitation-G1-v0/2026-07-14_21-16-41/`.
The strict evaluation and run provenance are under
`logs/interface_baselines/phase3_vanilla_corrected_50m_20260714/`.
The streamed-vanilla certificate from the 10M checkpoint covers all ten packet
slots and asynchronous renewal, with maximum command difference `3.28e-7` and
maximum deterministic-action difference `9.54e-7`. This proves the adapter.

Additional fixed-protocol continuations reached 110M before the local-budget
clarification and peaked at 40% strict success. They are diagnostic only and
are not a precedent for further local scaling. A 120M continuation was stopped
at 2.56M frames into its block and produced no checkpoint or evaluation. A
separate 10M adaptive-reset diagnostic was rejected; the protocol remains the
original fixed 0-200 reset range with unchanged rewards and terminations.
Long convergence and the 0.8 paper-facing oracle gate belong to the final
Skynet run. Local planner wiring may proceed with a diagnostic checkpoint, but
its performance must not be interpreted as an interface result.

### Local explicit-vanilla M3 diagnostic from the active Skynet run

On 2026-07-15, the latest complete checkpoint from corrected-LAFAN1 vanilla
job `3500993` was copied locally for a one-motion planner check. The source was
`model_step_1050083328.pt` (about 1.05B frames), with SHA-256
`3b1bb1f1e196e3f7b37eebde4185587a19b68be26cd381476e34c0074bea50ac`.
The local copy is under
`logs/downloaded_checkpoints/lafan1_corrected_vanilla_job3500993/`.

The diagnostic used corrected `walk1_subject1`, one environment, the normal
push event, the exact 10-frame vanilla packet at 5 Hz, and the frozen direct
vanilla tracker at 50 Hz. The medium shared planner received only the 930-value
causal robot history and predicted the 670-value packet. It used exactly 1,000
demonstration rows and 2,000 pretraining updates, then exactly 1,000
planner-rollout rows mixed 1:1 with the demonstration rows for 2,000
fine-tuning updates. The requested closed-loop horizon was 1,000 control
steps.

This was the first diagnostic protocol. It extended the normal 500-step task
episode and collected planner rows without the intended 0-200 random start
range. Keep its numbers only as debugging history. The corrected M3 protocol
does not extend the episode: each rollout lasts at most 500 control steps,
starts from a reference frame sampled in 0-200, and therefore reaches at most
about reference frame 700. It disables `anchor_pos`, `anchor_ori`, and
`ee_body_pos` terminations, keeps `base_too_low`, reports tracking errors as
continuous metrics, and defines survival as not falling.

The corrected diagnostic was then rerun from scratch with the same 1,000-row
budget per stage and 2,000 updates per training stage. Both demonstration and
planner-rollout rows use the corrected 500-step, random-start 0-200 M3
protocol, and their strict 2,000-row merge passes.

| Corrected M3 row | Episode steps | Fall-free survival | Planner RMSE | MPJPE | Root XY error | Joint RMSE | EE position error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| demonstration-only | 492 | 0.0 | 0.683 | 82.0 mm | 2.24 m | 0.210 rad | 2.22 m |
| rollout-finetuned | 500 | 1.0 | 0.315 | 57.9 mm | 0.429 m | 0.183 rad | 0.449 m |

The demonstration-only planner fell through `base_too_low` at step 492. The
fine-tuned planner reached the normal 500-step timeout without falling. This is
still a one-motion, one-seed local diagnostic, not an interface comparison.
Its main value is that it confirms the intended M3 code and shows that rollout
fine-tuning improves both fall-free survival and continuous tracking metrics.

| Row | Survival steps | Tracking success | Planner target RMSE | MPJPE | Joint RMSE | Failure |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| streamed-vanilla oracle | 1,000 | 1.0 | n/a | 32.5 mm | 0.147 rad | none |
| demonstration-only planner M3 | 64 | 0.0 | 0.367 | 69.7 mm | 0.201 rad | `ee_body_pos` |
| rollout-finetuned planner M3 | 139 | 0.0 | 0.201 | 37.6 mm | 0.127 rad | `ee_body_pos` |

The mixed-data offline RMSE was `0.0374`, so the remaining failure is a
closed-loop problem rather than an inability to fit the saved rows. One round
of rollout fine-tuning improved survival by about 2.2 times and improved all
listed tracking errors, but it did not solve the motion. Do not treat this
single seed, motion, checkpoint, or sample budget as a paper comparison.

Artifacts are under
`logs/interface_baselines/local_vanilla_m3_job3500993_step1050083328_walk1/`.
The corrected paths use the suffix `m3_500step_random0_200`; the final summary
is under
`full_body_trajectory_streamed_vanilla/chunked_transformer_medium_1000/eval_finetuned_m3_500step_random0_200/summary.json`.
Rendered rollouts are under the sibling directories
`video_pretrained_m3_500step_random0_200/videos/play/rl-video-step-0.mp4` and
`video_finetuned_m3_500step_random0_200/videos/play/rl-video-step-0.mp4`.
Rendering preserves the protocol but produces a separate stochastic push
realization: the rendered demonstration-only planner fell at step 468, while
the rendered fine-tuned planner completed all 500 steps without falling.
The strict merge manifest records 1,000 rows from each stage and 2,000 rows in
total. During this run, the closed-loop evaluator was found to overwrite an
explicit `--dataset_path` when it re-resolved the manifest. This caused the
demonstration/planner metadata merge to fail, as intended. The evaluator now
preserves an explicit dataset path, the planner-rollout rows were recollected,
and the strict merge passed without rewriting saved metadata.

### One-motion causal DiffSR planner gate

On 2026-07-15, the corrected-LAFAN1 DiffSR path passed the missing one-motion
planner check on `walk1_subject1`. This check answers a narrow question: can a
planner command a frozen latent low-level policy for one motion using achieved
robot state only, without reading expert motion history? It is not an
interface comparison or a paper result.

The frozen low-level checkpoint is
`model_step_1250033664.pt` from corrected-LAFAN1 job `3503434`, with SHA-256
`b442a16d12eb183e07d45116c6e7f44eb2f8b1b955fe0445c8cde68b895f9869`.
The matching skill checkpoint has SHA-256
`c9cf9691823043d63cafababfb3b3ce2182215f905b50540801ffc210e040728`.
The binding certificate passed with all 14 encoder tensors identical. The
motion NPZ has SHA-256
`c4ec70e04aebecf9735bab0016752df2c14377d613e1cc42e4045db7c3ce0103`.

The planner is the medium 23,164,448-parameter chunked Transformer. Its only
input is ten causal robot frames, each containing 93 values: relative joint
position, relative joint velocity, base angular velocity, projected gravity,
and the previous action. The flattened input is 930 values. There is no
language input and `reference_features` is empty. It predicts one 256-value
DiffSR code every ten control steps.

Training used exactly 1,000 oracle-achieved-state rows for 2,000 updates. A
second stage collected exactly 1,000 rows from that planner, merged the two
sources 1:1, and fine-tuned for another 2,000 updates. Both trainers used
`state_key=planner_state`; `expert_planner_state` was retained only for
diagnostics and was numerically different from the causal state. The
fine-tuned checkpoint has SHA-256
`4100a200e7c8493de558287012b09e611f652ee673d50fd7a37a93be9a31c442`. The
no-reference-leak certificates also perturb the forbidden expert fields and
require the planner output to remain unchanged. Both the pretrained and
fine-tuned certificates passed with maximum prediction change zero.

All three rows below use the same ten deterministic reference starts
`[197, 113, 172, 41, 152, 177, 10, 136, 137, 129]`, the normal 500-step
episode, the existing interval push event, and the same frozen low-level
policy. M3 disables only `anchor_pos`, `anchor_ori`, and `ee_body_pos`
terminations and counts `base_too_low` as a fall.

| Row | Fall-free survival | End threshold pass | MPJPE | Root XYZ error | Joint RMSE | EE position error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| latent oracle | 1.0 | 1.0 | 33.34 mm | 0.168 m | 0.135 rad | 0.187 m |
| causal planner, demonstrations only | 1.0 | 0.8 | 62.61 mm | 1.090 m | 0.178 rad | 1.099 m |
| causal planner, rollout-fine-tuned | 1.0 | 0.6 | 61.66 mm | 0.679 m | 0.168 rad | 0.697 m |

This proves that a no-leak latent planner can produce a stable 10-second walk
for all ten starts. Rollout fine-tuning improved average root drift, joint
error, EE error, and MPJPE, but it did not improve every measure: root
orientation, velocity, acceleration, action change, and the coarse final
threshold score worsened. The planner is therefore working, but still has a
clear gap to its own latent oracle. Do not hide that mixed result.

Artifacts are under
`logs/interface_baselines/lafan1_one_motion_latent_gate_20260715/`. The two
causal certificates are
`protocol_checks/causal_no_reference_leak.json` and
`protocol_checks/causal_no_reference_leak_finetuned.json`; the encoder binding
record is `protocol_checks/latent_skill_binding.json`. Quantitative summaries
are in `oracle_walk1_subject1_10starts/`,
`planner_pretrained_walk1_subject1_10starts/`, and
`planner_finetuned_walk1_subject1_10starts/`. Frame-zero videos are in the
corresponding `*_frame0/videos/play/` directories.

### Matched one-motion latent versus explicit packet

The corrected explicit-packet row was rerun on 2026-07-16 after finding that
the older demonstration-only diagnostic had trained from
`expert_planner_state`. The replacement trains from the same deployable
`planner_state` used by the latent row. Both interfaces now use the exact same
ten starts `[197, 113, 172, 41, 152, 177, 10, 136, 137, 129]`, 1,000
demonstration rows, 2,000 pretraining updates, 1,000 interface-specific
planner-rollout rows, a strict 1:1 merge, and 2,000 fine-tuning updates.

The explicit tracker is the 1.05B-frame checkpoint with SHA-256
`3b1bb1f1e196e3f7b37eebde4185587a19b68be26cd381476e34c0074bea50ac`.
Its fresh streamed-equivalence certificate has maximum command difference
`4.45e-7` and action difference `1.91e-6`. Both fine-tuned planners pass the
same counterfactual no-reference-leak audit.

| Interface and stage | Parameters | Fall-free survival | End threshold pass | MPJPE | MPJPE / own oracle | Root XYZ error | Joint RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| explicit oracle | n/a | 1.0 | 1.0 | 30.77 mm | 1.00 | 0.341 m | 0.148 rad |
| explicit demonstrations only | 22,973,472 | 0.9 | 0.7 | 45.20 mm | 1.47 | 0.419 m | 0.171 rad |
| explicit rollout-fine-tuned | 22,973,472 | 1.0 | 0.6 | 67.75 mm | 2.20 | 0.798 m | 0.167 rad |
| latent oracle | n/a | 1.0 | 1.0 | 33.34 mm | 1.00 | 0.168 m | 0.135 rad |
| latent demonstrations only | 22,965,792 | 1.0 | 0.8 | 62.61 mm | 1.88 | 1.090 m | 0.178 rad |
| latent rollout-fine-tuned | 22,965,792 | 1.0 | 0.6 | 61.66 mm | 1.85 | 0.679 m | 0.168 rad |

At this medium size, the demonstration-only explicit planner preserves more
of its oracle tracking accuracy, although one of ten robots falls. Rollout
fine-tuning restores explicit survival but worsens its average physical
tracking. The fine-tuned latent planner then has lower MPJPE and root error.
This is a mixed one-motion result, not evidence that either interface wins in
general. It shows that demonstration-only and rollout-fine-tuned results must
both remain in the study.

Explicit artifacts are under
`logs/interface_baselines/lafan1_one_motion_explicit_gate_20260716/`. The
pretrained and fine-tuned checkpoint SHA-256 values are
`878d4fc5711978c93beda7c2ee581d7cac9eddec61c07b12d3d07332baf95052` and
`129ca1a86d5ca7cea6888005e8a41e96a52b0c6b32c0ca59413179a268021fe7`.
The fine-tuned local video is
`planner_finetuned_walk1_subject1_video/videos/play/rl-video-step-0.mp4`.

### Planner capacity scaling diagnostic

This local diagnostic holds data, updates, seed, starts, low-level
policies, and planner family fixed while varying only planner capacity. Use
the existing `tiny`, `small`, `medium`, and `large` Transformer presets. Their
actual parameter counts are:

| Preset | Latent parameters | Explicit parameters |
| --- | ---: | ---: |
| tiny | 129,680 | 131,472 |
| small | 4,182,304 | 4,186,144 |
| medium | 22,965,792 | 22,973,472 |
| large | 64,353,056 | 64,364,576 |

These pairs are close enough for the same-size comparison, but always report
the actual count. For every size and interface, retain the demonstration-only
and interface-specific rollout-fine-tuned checkpoints. Plot survival and
oracle-normalized physical errors against parameter count. Report the smallest
tested size reaching a fixed threshold. Do not interpolate through this curve
because the observed results are non-monotonic, and do not claim a precise
threshold from model-size labels. A separate
planner-family comparison may be run at matched parameter counts, but it is a
secondary study and must not replace the shared-architecture interface table.

The secondary planner-family study has two views and must report both:

- **Same size:** plot performance against actual parameter count for every
  planner family and interface. Compare architectures at each matched size,
  and compare latent against explicit within each architecture. Keep the
  approximate 0.13M, 4.2M, 23M, and 64M levels when the architecture supports
  them.
- **Same performance:** for each family and interface, report the smallest
  tested parameter count that reaches the predeclared survival and
  oracle-normalized MPJPE target across repeated seeds. Also retain absolute
  MPJPE so normalization cannot hide poor physical tracking.

Limit the planner families to the current flow-matching Transformer, one
diffusion action-chunk Transformer, and one deterministic chunk predictor.
Everything except the planner family and size remains fixed: causal input,
language token, output interface, training samples, update count, batch size,
evaluation starts, frozen low-level controller, and command rate. Record
actual parameters, latency, and output bandwidth. Use at least three planner
seeds for a claimed curve. Never choose the best seed, interpolate through a
non-monotonic curve, or use different training data to help one architecture.
Treat demonstration-only and rollout-fine-tuned planners as separate curves.
The former is the primary model-capacity result. Because diffusion makes
multiple forward calls per command, report inference latency next to parameter
count; a low-parameter but slow planner is not automatically more efficient.

The demonstration-only tiny point is complete for seed 0 on the same ten
`walk1_subject1` starts:

| Tiny interface and stage | Parameters | Fall-free survival | End threshold pass | MPJPE | Root XYZ error | Planner latency, batch 10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| latent, demonstrations only | 129,680 | 1.0 | 0.8 | 63.57 mm | 1.593 m | 5.85 ms |
| latent, rollout-fine-tuned | 129,680 | 1.0 | 0.9 | 95.21 mm | 1.859 m | 5.85 ms class |
| explicit packet, demonstrations only | 131,472 | 0.7 | 0.4 | 90.03 mm | 1.812 m | 5.93 ms |
| explicit packet, rollout-fine-tuned | 131,472 | 0.5 | 0.0 | 147.34 mm | 1.846 m | 5.93 ms class |

At approximately 130k parameters, the latent interface is clearly easier for
this demonstration-only one-motion planner. The tiny latent offline RMSE was
`0.1128`, while explicit plateaued at `0.4251`; these cross-interface RMSE
values are diagnostic only because the target spaces differ. Do not promote
this single-motion, single-seed point to a general scaling conclusion until
the remaining capacity points are complete. Size-specific rollout fine-tuning
is now complete and is harmful for both tiny planners: latent retains
fall-free survival but loses tracking accuracy, while explicit loses both
survival and tracking accuracy. Both fine-tuned checkpoints pass the causal
no-reference-leak audit. This suggests that the tiny planners cannot absorb
the wider planner-visited state distribution under the fixed training recipe;
do not discard or silently replace this negative result. Artifacts are under
`logs/interface_baselines/lafan1_one_motion_capacity_scaling_20260716/tiny/`.

The matched small point is also complete and passes both causal no-leak
audits:

| Small interface and stage | Parameters | Fall-free survival | End threshold pass | MPJPE | Root XYZ error | Joint RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| latent, demonstrations only | 4,182,304 | 1.0 | 0.7 | 71.11 mm | 0.937 m | 0.181 rad |
| latent, rollout-fine-tuned | 4,182,304 | 1.0 | 0.2 | 87.09 mm | 1.208 m | 0.165 rad |
| explicit packet, demonstrations only | 4,186,144 | 0.9 | 0.7 | 50.29 mm | 0.637 m | 0.173 rad |
| explicit packet, rollout-fine-tuned | 4,186,144 | 1.0 | 0.6 | 66.57 mm | 0.947 m | 0.173 rad |

At approximately 4.18M parameters, explicit has lower MPJPE and root error in
both stages. Rollout fine-tuning restores explicit survival but worsens its
average tracking; latent remains fall-free but also loses MPJPE and root
accuracy while improving joint error. The tiny, small, and medium results are
therefore not monotonic enough to infer a clean crossover without the large
point and repeated seeds. Preserve every stage and metric in the aggregate;
do not select whichever stage makes one interface look better. Artifacts are
under
`logs/interface_baselines/lafan1_one_motion_capacity_scaling_20260716/small/`.

The large point completes the four-size, seed-0 diagnostic and both fine-tuned
checkpoints pass the causal no-reference-leak audit:

| Large interface and stage | Parameters | Fall-free survival | End threshold pass | MPJPE | MPJPE / own oracle | Root XYZ error | Joint RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| latent, demonstrations only | 64,353,056 | 1.0 | 0.9 | 56.85 mm | 1.71 | 1.078 m | 0.171 rad |
| latent, rollout-fine-tuned | 64,353,056 | 1.0 | 0.7 | 64.36 mm | 1.93 | 1.066 m | 0.154 rad |
| explicit packet, demonstrations only | 64,364,576 | 1.0 | 0.8 | 41.07 mm | 1.34 | 0.506 m | 0.162 rad |
| explicit packet, rollout-fine-tuned | 64,364,576 | 1.0 | 0.5 | 64.41 mm | 2.09 | 0.784 m | 0.160 rad |

At this size, the demonstration-only explicit planner is stronger on MPJPE,
root error, and joint error. After rollout fine-tuning, MPJPE is essentially
tied, latent has slightly lower joint error, and explicit has lower root error.
Fine-tuning again worsens MPJPE and the end threshold score for both rows even
though all ten robots remain fall-free.

Across the complete seed-0 curve, larger is not uniformly better and the
interface ordering depends on both size and training stage. Under the
diagnostic criterion of 100% fall-free survival and MPJPE at most 75 mm, the
smallest observed demonstration-only points are 129,680 parameters for latent
and 64,364,576 for explicit. After rollout fine-tuning they are 22,965,792 and
4,186,144 parameters, respectively. These are observed tested points, not
stable size requirements. Repeated planner seeds are required before making a
capacity-efficiency claim.

All four sizes, both stages, oracle-normalized metrics, exact starts, hashes,
and training-contract checks are generated from artifacts by
`experiments/interface_baselines/aggregate_one_motion_capacity_scaling.py`.
The current outputs are under
`logs/interface_baselines/lafan1_one_motion_capacity_scaling_20260716/capacity_summary/`.
The medium compatibility entries are symbolic links to the original matched
gate artifacts, so checkpoints are not duplicated.

The seed-1 and seed-2 repeats for `tiny` and `small` are complete. Both
interfaces and both training stages pass the causal no-reference-leak audit.
At `tiny`, the three-seed demonstration-only means are 100% survival and
66.14 ± 3.30 mm MPJPE for latent versus 56.7% survival and 83.71 ± 5.96 mm for
explicit. At `small`, they are 100% and 70.20 ± 5.49 mm for latent versus
93.3% and 54.26 ± 3.81 mm for explicit. The rollout-fine-tuned small means are
100% survival for both, with 76.59 ± 9.19 mm for latent and 59.61 ± 7.05 mm
for explicit. This supports a real low-capacity advantage for the latent
interface, but not a general advantage at larger capacity.

Using the predeclared three-seed diagnostic target of mean survival at least 0.9
and mean MPJPE no worse than 2.0 times each interface's own oracle, the
smallest tested demonstration-only model is `tiny` (129,680 parameters) for
latent and `small` (4,186,144 parameters) for explicit. After rollout
fine-tuning, explicit reaches the target at `small`; latent does not reach it
at either repeated size. These are still one-motion diagnostic results. The
artifact-driven repeated-seed summary is under
`logs/interface_baselines/lafan1_one_motion_capacity_scaling_multiseed_20260716/three_seed_summary/`.

The matched planner-family code gate and first tiny diagnostic are complete.
The training and both closed-loop evaluators now load flow-matching,
clean-target diffusion, and deterministic chunk planners through one
checkpoint contract. At each interface, all three families have identical
parameter counts and receive the same causal history, samples, update count,
starts, and frozen low-level controller. Seed-0 demonstration-only results are:

| Planner family | Latent survival | Latent MPJPE | Explicit survival | Explicit MPJPE |
| --- | ---: | ---: | ---: | ---: |
| flow matching | 1.0 | 63.57 mm | 0.7 | 90.03 mm |
| diffusion | 1.0 | 66.43 mm | 0.7 | 99.66 mm |
| deterministic | 0.9 | 65.52 mm | 0.7 | 75.04 mm |

Both new families pass real latent and explicit Isaac wiring smokes and both
fine-tuned no-reference-leak audits. The result supports an architecture-robust
latent advantage at approximately 130k parameters, but it is still a
one-motion, one-seed result. Rollout fine-tuning worsened physical tracking for
all three tiny families and substantially reduced explicit survival for flow
and diffusion. The artifact-driven family summary is under
`logs/interface_baselines/lafan1_one_motion_planner_families_20260716/seed0_summary/`.

### Corrected-data local DiffSR qualification

The matching corrected-data DiffSR path completed a 10.027M-frame local code
qualification. Its fixed evaluator ran all 40 motions and correctly stopped
each environment independently. Strict success was 0%, mean strict survival
was 40.025 control steps, MPJPE was 88.45 mm, and joint RMSE was 0.2316 rad.
The representation and evaluator wiring pass, but the low-level oracle gate
does not. Do not scale this locally to chase convergence; the final DiffSR
oracle run belongs on Skynet.

### Corrected-data Skynet DiffSR and qualification

On 2026-07-15, corrected-LAFAN1 DiffSR job `3503380` was canceled while
resource-pending and produced no output. Its two-day walltime was too short for
the measured latent throughput at a 5B-frame target. Authoritative replacement
job `3503434` uses a four-day walltime and snapshot
`isaaclab_20260715_042305` (workspace archive SHA-256
`1216a364b68756c58c30da94e258a759db51e33903e4416ec192fea2ad26f722`).
It is the latent-only counterpart to vanilla job `3500993`: seed 0, 4096
environments, 50,865 iterations (`5,000,232,960` frames), the exact corrected
40-motion manifest and `g1_hl_diffsr` cache, DiffSR horizon 10 and code size
256, fixed reset range 0-200, unchanged rewards and terminations, no reward
estimator, and no BC. It submits no EE/full-body controller and runs no
paper-facing planner evaluation. The dedicated launcher is
`experiments/interface_baselines/run_lafan1_diffsr_low_level_skynet.sh`.

Qualification jobs `3503401` and `3503425` were canceled while
dependency-pending and produced no output. The first predated the exact parsed
run-name resolver; the second depended on the canceled two-day latent job.
Authoritative qualification job `3503441` is dependency-blocked on
`afterok:3500993:3503434`. It uses snapshot
`isaaclab_20260715_042326` and the same verified workspace archive SHA-256
`1216a364b68756c58c30da94e258a759db51e33903e4416ec192fea2ad26f722`.
At runtime it resolves the unique latent training directory from the frozen
run ID using the exact parsed `agent.logger.exp_name`, requires the exact final
`model_step_5000232960.pt`, verifies that the
selected DiffSR encoder is tensor-for-tensor identical to the encoder embedded
in the low-level checkpoint, and then runs the two 40-motion oracle audits plus
the streamed-vanilla equivalence certificate. Both oracle success rates must
be at least 0.8. Phase 4 remains unsubmitted unless this job succeeds and all
three artifacts pass inspection.

At the 2026-07-16 04:14 EDT health check, vanilla job `3500993` had reached
`2,930,049,024 / 5,000,232,960` frames at about 30.3k frames/s and latent job
`3503434` had reached `1,790,017,536 / 5,000,232,960` at about 26.1k frames/s.
Both were healthy. Their six-day and four-day walltimes, respectively, leave
enough time at those observed rates. Qualification job `3503441` remains
dependency-blocked as intended; do not submit Phase 4 from intermediate
checkpoints.

### BONES-SEED Phase-5 low-level submission status

The final two-controller launcher passed its dry run on 2026-07-14 with the
fresh 100-motion manifest, preparation record, and MiniLM table hashes. It
renders only the DiffSR latent controller and the direct 50 Hz vanilla
controller, each at the final 1B-frame Skynet budget. This does not change the
local rule: about 50M total frames is the maximum serious local check, not a
target. There is no reason to run a 100M local block; small local runs only
need to show that the code follows the intended protocol.

The 2026-07-14 attempts all stopped before `sbatch` while the server-side
`rsync` worker was blocked in an NFS RPC wait. The timestamped directories
`isaaclab_20260714_234125`,
`isaaclab_20260714_234653`, `isaaclab_20260714_234908`, and
`isaaclab_20260714_235332` are incomplete pre-submission snapshots, not job
provenance.

On 2026-07-15 the launcher switched to a single verified workspace archive,
which is extracted on compute-local storage instead of checking out thousands
of files on Skynet NFS. Both final low-level candidates were then submitted:

| Controller | Job | Snapshot | State at 00:30 EDT |
| --- | ---: | --- | --- |
| DiffSR latent | `3501873` | `isaaclab_20260715_000935` | running on `synapse`; skill encoder and diagnostic planner complete, low-level Isaac process active |
| direct vanilla 50 Hz | `3501960` | `isaaclab_20260715_002253` | running on `synapse`; unique archive bootstrap succeeded and the Isaac training process is active |

The shared low-level run directories are:

- latent: `logs/rlopt/ipmd/Isaac-Imitation-G1-Latent-v0/2026-07-15_00-31-18`;
- vanilla: `logs/rlopt/ipmd/Isaac-Imitation-G1-v0/2026-07-15_00-32-35`.

At 00:46 both assigned A40s were actively computing with 26.6 GB (latent) and
13.9 GB (vanilla) allocated. Both reached iteration 102 and 10,027,008 frames:
latent reported 21.8k frames/s and vanilla 24.6k frames/s. Neither had reached
the first 50M checkpoint interval yet.

At 01:31 EDT both jobs were still healthy on `synapse`. Latent had reached
iteration 713 and 70,090,752 frames at 23.6k frames/s; vanilla had reached
iteration 814 and 80,019,456 frames at 27.9k frames/s. Both wrote the expected
first checkpoint, `model_step_50036736.pt`. Qualification job `3502183`
remained pending on the two training dependencies.

At 01:52 EDT both jobs remained healthy. The latest latent report was iteration
916 and 90,046,464 frames at 23.5k frames/s; vanilla had reached iteration 1221
and 120,029,184 frames at 27.6k frames/s. Qualification job `3502183` was still
`PENDING (Dependency)`, so no planner submission was allowed.

At 02:10 EDT the latest latent report was iteration 1119 and 110,002,176
frames; vanilla had reached iteration 1526 and 150,011,904 frames. Both jobs
remained `RUNNING` on `synapse`, and qualification remained dependency-blocked.

At 02:22 EDT both jobs were still healthy on `synapse`. Latent had reached
iteration 1323 and 130,056,192 frames; vanilla had reached iteration 1730 and
170,065,920 frames. Qualification job `3502183` remained `PENDING
(Dependency)`.

The guarded Phase-4 no-language array and complete-grid aggregator are now
implemented, but no planner array has been submitted. The launcher fixes three
planner seeds, all 40 LAFAN1 motions, and the three planner-row budgets
`1k/10k/50k`. It validates the exact corrected LAFAN1 manifest, checkpoint,
audit, skill-encoder, and streamed-equivalence hashes before submission. One
array task handles all three budgets for one motion and one seed, so
demonstrations and Isaac startup work are reused without mixing planner-rollout
samples between budgets. Phase 4 remains blocked on separate corrected-LAFAN1
0.8-or-better oracle qualifications for both controllers. BONES qualification
job `3502183` is only a Phase-5 gate and cannot satisfy the manifest-bound
LAFAN1 validator.

The Phase-4 gate now also binds the content-specific latent and vanilla cache
paths in both oracle audits and the streamed-equivalence certificate. Its
specialized array submitter records the job ID, array shape, and workspace
hashes in the persistent result root. Aggregation requires that record,
refuses output overwrite, and generates both a Markdown sample-efficiency
table and a hash manifest for its JSON/CSV/Markdown outputs.

At 02:46 EDT, corrected-LAFAN1 direct-vanilla job `3500993` was still running
on `synapse` and had reached 410,025,984 of 5,000,232,960 frames. The current
BONES-SEED candidates were also healthy: latent job `3501873` had reached
160,038,912 of 1,000,046,592 frames and vanilla job `3501960` had reached
200,048,640 of 1,000,046,592 frames. BONES qualification job `3502183`
remained pending on those two dependencies.

At 03:15 EDT the BONES-SEED training jobs remained healthy. Latent job
`3501873` had written through `model_step_150011904.pt`; vanilla job `3501960`
had written through `model_step_250085376.pt`. Authoritative qualification job
`3503120` was dependency-blocked, as intended. No additional local training
was started; local work remained limited to code, audit, and command-routing
checks.

At 03:28 EDT all long-running prerequisites were still healthy: corrected
LAFAN1 direct-vanilla job `3500993` and BONES-SEED jobs `3501873` and
`3501960` remained `RUNNING` on `synapse`; qualification `3503120` remained
`PENDING (Dependency)`. The BONES latent and vanilla directories had written
through `model_step_200048640.pt` and `model_step_250085376.pt`, respectively.
No Phase-4 or Phase-5 planner job had been submitted.

At 04:05 EDT, BONES latent job `3501873` had reached 250,085,376 frames and
vanilla job `3501960` had reached 320,077,824 frames. Their recorded commands
match the fresh 100-motion manifest, separate `latent_seed0` and
`vanilla_seed0` caches, reset range 0-200, fixed command interfaces, disabled
reward estimator and BC, and the 1,000,046,592-frame target. No local training
or planner job was started.

The Phase-4 output schema also passed a fresh local Isaac check at
`.tmp/phase4_schema_smoke/` on 2026-07-15. It used two environments and 22
steps on corrected `walk1_subject1`; no training was run. The streamed adapter
again covered all ten phases and asynchronous renewal, with maximum command
difference `3.87e-7` and deterministic-action difference `1.19e-6`. The direct
summary recorded strict outcomes, termination causes, push metadata, and all
paper metrics. An initial attempt pointed at the generic unified cache and
failed before simulation because that cache did not match the selected
one-motion manifest. The passing rerun used the content-specific corrected
cache recorded by the 50M run. This reinforces the rule that the manifest and
dataset cache must be routed together.

Seed-0 latent job `3501873` was launched with an empty online logger backend,
while vanilla writes W&B metrics. This does not change the learner or the saved
50M checkpoints, so do not cancel the healthy latent run. Build its paper
learning curve by running the fixed oracle evaluation on each saved checkpoint.
The launcher now explicitly enables the same W&B project for latent on future
paired seeds.

The latent snapshot contains a 48,728,513-byte archive with SHA-256
`d4bd4564e712f4223c4675afed986fedeb90f3ae09689b32b7ca0fef123ffd88`.
The vanilla replacement contains a 48,734,002-byte archive with SHA-256
`aaf046ab0cde041834ed81a0c57cb46a971dbf68f9eee4f3b54e50ff060d4feb`.
The only experiment-relevant difference is the fixed archive bootstrap and
new qualification tooling; low-level training code and arguments are
unchanged. Neither job had emitted its first RL iteration at the 00:30 process
check, so both still require log and checkpoint monitoring. The recorded
repositories are top-level `9d12c03` with the intended dirty overlay, RLOpt
`7349731` with its intended dirty overlay, and clean
ImitationLearningTools `d3d5532`. The latent log confirms archive extraction,
the fresh BONES manifest, and all requested low-level arguments reached the
compute node.

Vanilla job `3501874` failed during bootstrap before training. Both archive
jobs originally extracted to a directory named `workspace`, so the second job
collided with the first job's `workspace.img` overlay. The replacement uses a
Slurm-job-specific extracted directory, skips the Isaac Sim cache copy, uses
the shared SIF, and requests an 8192 MB job-specific overlay. Job `3501874`
must never be used as a checkpoint source.

Read the current scheduler state before treating the table above as current:

```bash
ssh skynet "bash -lc 'squeue -j 3501873,3501960'"
```

Do not submit planner jobs when these training jobs merely finish. First run
the strict 100-motion vanilla and DiffSR oracle audits and regenerate the
streamed-vanilla equivalence certificate from the exact vanilla checkpoint.
The DiffSR audit uses the explicit strict tracking-failure rate rather than
`done_rate`, because a successful motion may end through `reference_finished`.
The fixed launcher is
`experiments/interface_baselines/submit_bones_seed_low_level_qualification_skynet.sh`;
it is implemented and tested.

The original qualification job `3502183` was canceled before it ran. Its
snapshot predated strict dataset-path provenance; code review also found that
its vanilla audit command compared against the latent cache path while the
latent audit omitted an expected cache. The evaluations themselves routed the
right caches, but that old job could not produce a trustworthy gate artifact.

Replacement job `3503087` was also canceled before it ran after the audit was
further tightened to reject a shortened debug evaluation that merely stops
within the 1,000-step budget. Qualification job `3503120` was then canceled
while still dependency-pending because its command named `best.pt`, while the
latent low-level training command records `latest.pt`. The two files differ,
although inspection showed their 14 runtime skill-encoder tensors are
identical. No evaluation ran and no qualification output was created.

The authoritative replacement is job `3503363`, queued with Slurm dependency
`afterok:3501873:3501960`. Its snapshot is
`isaaclab_20260715_040451`; the 48,816,581-byte workspace archive has SHA-256
`c2bb326c9c237b8763f6decf62c40e9a3c93569ed46f6e97bc83bf62ef63cb1a`.
It targets the guaranteed final filenames `model_step_1000046592.pt` in the
latent and vanilla run directories listed above and the exact skill path used
by latent training,
`base_pipeline/skill_encoder_h10_z256/checkpoints/latest.pt`. Before launching
Isaac it compares that checkpoint's encoder weights tensor-for-tensor against
the encoder embedded in the latent low-level checkpoint. The latent audit
repeats and records this binding, and both Phase-4 and Phase-5 planner gates
require it. Submission-time checkpoint checks remain deferred because the
final files do not exist yet; the dependent job still checks them before
evaluation and fails closed. Its vanilla and latent audits bind the matching
`vanilla_seed0` and `latent_seed0` caches, require the full fixed evaluation
request, and the streamed-equivalence certificate binds the vanilla cache,
manifest, and exact tracker checkpoint. No planner job has been submitted.

Job `3503363` later failed before either Isaac evaluation because the
qualification shell passed `--expected_dataset_path` to the raw BONES data
preflight, which does not accept or need a Zarr-cache argument. Cache identity
is checked by the separate vanilla and latent oracle audits. The qualification
shell now keeps the preflight manifest-only and still passes the exact
`vanilla_seed0` and `latent_seed0` paths to the controller-specific audits.
The focused pure-Python suite passes 108 tests after this correction.

Replacement qualification job `3512041` was submitted on 2026-07-16 from
snapshot `isaaclab_20260716_035410`, with workspace archive SHA-256
`f3636850bdd04eb7f3ee5042f1cd5b34f835c4844c13b8fd6928eb3e0f943418`.
It uses a fresh result root,
`logs/interface_baselines/bones_seed_100_low_level_qualification_seed0_retry_20260716`.
It completed in 11m10s. Direct vanilla passed its strict oracle gate with
`0.8999999762` success and DiffSR latent passed with `0.8399999738`, both above
the fixed `0.8` threshold. All 14 skill-encoder tensor bindings pass. The
streamed-vanilla certificate covers packet phases 0 through 9 and asynchronous
per-environment renewal, leaves the policy unchanged, and reports maximum
command and action differences `3.02e-7` and `1.31e-6`. The checkpoint hashes
are `e5b8da6736844d34bbed6c549f7939cc3b6397b69502c61a105bce9f63e13782`
for vanilla and
`904229f5737256843e8046ac77b96a39d8e5dd441274e91813b7efc63d00b202`
for latent. This cleared the Phase-5 low-level gate.

Oracle demonstration collection now
uses 100 environments and exact per-motion row accounting, reducing that stage
from 200 Isaac launches to two: one latent and one explicit. The shared smoke
suite checks the generated commands and the collectors fail if any motion ends
below its row budget. Planner-driven collection and closed-loop evaluation
still launch one process per goal. Planner-driven collection uses ten
same-goal, same-motion environments within that process and fails on any
goal/reference mismatch. Do not choose a language embedding from a reference
rank. The stages run as resumable explicit-goal arrays and merge their audited
outputs afterward.

The balanced collector itself passed a two-motion local Isaac smoke on
2026-07-15 for both main interfaces. Each path saved one row per motion with
the required `930`-value causal input and `384`-value language embedding; the
targets were `670` values for explicit vanilla and `256` for DiffSR. Both
collectors stopped before the first environment step once the exact budgets
were complete. Treat this as a code gate only.

The resumable planner split is implemented, locally tested, and submitted.
`submit_bones_seed_multigoal_pipeline_skynet.sh` validates the
completed qualification artifacts before it can call `sbatch`, then uses one
workspace snapshot for `prepare -> rollout[0-99] -> finetune ->
final-eval[0-99] -> summarize`. Each array task receives its goal only from
`SLURM_ARRAY_TASK_ID`; reference rank is never used to choose language. Stage
records bind the original configuration, workflow source hashes, explicit goal
identity, and file or directory hashes. The collectors now buffer 1,000 rows
before flushing a sample file; a multi-environment publication can make a file
slightly larger. The balanced selector and final audit still require the exact
per-goal and total sample counts while avoiding about 100,000 small files on
Skynet NFS. Pure command-routing, dependency, resume, goal,
hash, and sample-chunk tests pass. Job `3512041` and all three gate artifacts
passed before the paper chains were submitted.

At 04:30 EDT on 2026-07-15, BONES latent job `3501873` had reached
290,095,104 frames at about 23.9k frames/s, with recent per-step reward 0.0498.
BONES vanilla job `3501960` had reached 380,043,264 frames at about 28.8k
frames/s, with recent per-step reward 0.0553. Corrected-LAFAN vanilla job
`3500993` had reached 590,020,608 of 5,000,232,960 frames; recent episode
length remained around 437 steps and per-step reward was 0.0601. These are
health checks only, not oracle or planner results. Corrected-LAFAN latent job
`3503434` was still pending for resources, and qualification jobs `3503363`
and `3503441` remained dependency-blocked. No planner job was submitted.

### Preliminary unqualified Phase-5 DiffSR planner

Later on 2026-07-15, BONES low-level jobs `3501873` and `3501960` completed
their fixed 1B-frame budgets. Qualification job `3503363` then failed before
either oracle evaluation because its snapshot passed
`--expected_dataset_path` to `audit_bones_seed_phase5.py`, while that audit
snapshot did not define the option. The already completed latent skill-binding
check passed all 14 tensors. This is a qualification-tooling failure, not an
oracle pass or failure.

The user explicitly requested early Phase-5 DiffSR planner results without
waiting for the repaired qualification. A separate preliminary, latent-only
ten-goal seed-0 chain was therefore submitted from snapshot
`isaaclab_20260715_150805`:

| Stage | Job |
| --- | ---: |
| balanced demonstrations and planner pretraining | `3506446` |
| per-goal pretrained evaluation and rollout collection array | `3506447` |
| merged rollout fine-tuning | `3506448` |
| per-goal final evaluation array | `3506449` |
| preliminary summary | `3506450` |

It uses the final 1B-frame DiffSR checkpoint, its tensor-matched skill encoder,
the fresh 100-motion data tree, the first ten explicit goals, 1,000
demonstration rows and 1,000 planner-rollout rows per goal, the medium planner,
seed 0, and 500-step M3 evaluations. The output root is
`logs/interface_baselines/bones_seed_100_diffsr_preliminary_10goals_seed0_20260715`.
The runner records `preliminary_unqualified=true`, omits the paper audit, and
cannot enter the paired or multi-seed paper aggregate. The guarded 100-goal
latent-versus-explicit paper workflow remains blocked on repaired
qualification.

The preliminary prepare job failed after collecting 9,149 of the required
10,000 balanced latent demonstration rows: nine motions reached 1,000 rows,
while `ab_bicycle_001_A359` reached 851 before the four-times safety horizon.
The dependent jobs were canceled without running. Balanced collection already
stops immediately at the exact per-motion budget, so the safety cap was raised
from four to eight times the nominal control horizon without changing the
saved row count or optimizer protocol. The cap and its recorded run-config
value are covered by the focused tests. Do not resume the preliminary chain;
use the guarded paired paper pipeline after qualification passes.

The chunk writer then passed a real Isaac check for both interfaces. One goal
ran for eleven low-level steps, producing exactly two planner rows at control
steps 0 and 10 in one file. The explicit file had shapes `2 x 930`, `2 x 670`,
and `2 x 384` for state, target, and language; the latent file had `2 x 930`,
`2 x 256`, and `2 x 384`. This verifies that chunking preserves the two 5 Hz
rows and their goal labels. It is not a performance result.

Multi-seed result aggregation is also implemented but has no paper inputs yet.
`aggregate_bones_seed_multiseed_results.py` requires at least three unique,
fully audited seeds by default and rejects mismatched goals, code hashes, data,
checkpoints, or changed summarize-stage artifacts. It records paired
latent-minus-explicit differences within goal and seed, seed-level variation,
and a hierarchical bootstrap confidence interval. The per-seed summary now
retains the full tracking and smoothness metric set instead of discarding it,
records planner size and output bandwidth, and the multi-seed result reports
tracking success relative to each interface's qualified oracle. Do not run it
on the local smoke output to create a performance claim.

The Phase-5 paper handoff now has a fixed three-seed wrapper,
`submit_bones_seed_multiseed_pipeline_skynet.sh`. It renders or submits exactly
planner seeds 0, 1, and 2, performs all three real preflights before submitting
the first chain, and refuses an existing seed output by default. The staged
Slurm submitter stores `cluster_submission.json` under each persistent run
root with all five job IDs and the workspace/repository-manifest hashes. Final
audits and aggregation require that record, and aggregation rejects a
substituted three-seed set. The aggregate also generates
`multiseed_results.md` directly from the audited JSON so paper tables do not
depend on manual transcription. It also refuses output overwrite and hashes
all aggregate artifacts in `aggregation_manifest.json`.

After qualification passed on 2026-07-16, the fixed three paper chains were
submitted. Seed 0 uses jobs `3512092 -> 3512093 -> 3512094 -> 3512095 ->
3512096`; seed 1 uses `3512097 -> 3512098 -> 3512099 -> 3512100 -> 3512101`;
seed 2 uses `3512113 -> 3512114 -> 3512115 -> 3512116 -> 3512117`. In each
chain the two array jobs are `0-99%4`. All three snapshots share workspace
archive SHA-256
`58b0f17a5e5f42fa9dee737b64e86a8b67d96de1bdba988a400f2e80414e728b`.
At first inspection, all three prepare jobs had written a passing BONES
preflight and seed-specific run configuration; downstream jobs were waiting
on their intended dependencies. These are active runs, not completed paper
results.

Both final evaluators also passed a two-step Isaac metadata check for the
unchanged G1 push event. They record the same 1-3 second interval and the same
root-velocity ranges, and the paper audit now rejects missing or unequal push
protocols. This is provenance and code-path evidence, not a performance
result.

The same runtime check exposed and fixed two result-schema problems before any
paper planner submission. The explicit planner had used threshold success
while latent used strict termination success; both now use strict failure
terms for the main success number and retain threshold success separately.
Also, the audit no longer rejects a valid result merely because the single
goal ended before step 1000. It requires a positive length within the budget,
an honest stop reason, per-environment termination terms, and aggregate cause
counts. Before/after planner-retraining results and exact unique training-row
counts are retained in both single-seed and multi-seed summaries.

Planner-only inference timing is now part of both final evaluator schemas and
the workflow source hash. It synchronizes CUDA around only the deployed
high-level planner forward, drops one warmup call, and reports milliseconds,
call counts, and observed batch sizes. A 22-step, one-environment Isaac check
observed post-warmup calls on both paths. The roughly 1 ms values came from
tiny diagnostic planners and are instrumentation evidence only, not a planner
speed result.

The focused no-language planner then passed a tiny one-motion wiring smoke on
`walk1_subject1` using sample schema v2, the same `10 x 93` causal input, and
the same Transformer-flow backbone for the 256-D latent and 670-D explicit
packet outputs. It used two demonstration rows, two rollout rows, and one
update per stage. The scores are intentionally not reported as results.

## Why the redo

The PR#19 result table is internally inconsistent: the latent oracle
(ground-truth skill codes from the encoder) scored 0.100 success / 216 mm
MPJPE while the EE and full-body chunk *planners* (generated commands) scored
0.675-0.875 / ~50 mm. A planner cannot outperform its own interface's oracle,
so the gap can only come from unmatched low-level policies. This protocol
fixes that by:

- training all compared low-level policies from scratch on the same full
  40-motion LAFAN1 manifest, same seed, same ~5B-frame budget;
- using plain `IPMD` for all three (no bilinear representation learning);
- matching the h10 horizon everywhere: skill encoder horizon, latent phase
  clock, EE/full-body `command_future_steps`, and baseline chunk length;
- matching the planner pretrain budget (5000 updates for the latent planner
  and for the baseline chunk planners) and the per-motion finetune budget
  (2000 updates), with the chunk planners allowed to use all collected
  samples like the latent finetune does;
- matching the high-level cadence: the original protocol let the chunk
  planners replan every control step (50 Hz receding horizon with achieved-
  state feedback) while the latent z is held for 10 steps (5 Hz). The redo
  uses the held-chunk contract everywhere: chunk planners are queried once
  per horizon (`--planner_update_interval 10`) and the EE/full-body
  low-levels are **trained** under the same held-window consumption
  (`env.command_hold_steps=10`) so there is no train/deploy mismatch. The
  EE/FB oracle rows are also measured under the held contract, mirroring how
  the latent oracle z is encoded once per horizon and held;
- following standard VLA-WBC middleware semantics for chunk consumption
  (the "re-expression" variant): chunk *content* is frozen at the planner
  rate — lookahead shrinks toward the hold boundary with tail padding, no
  post-renewal information enters the command — but coordinates are
  re-expressed in the robot's current anchor frame every control step, the
  way real humanoid stacks (H2O/OmniH2O/HOVER-style) re-express world-frame
  targets with odometry. Freezing the chunk's coordinate frame for the whole
  hold (the naive reading of "held chunk") would impose a drift-compensation
  burden no real stack has and unfairly handicap the chunk baselines;
- reporting oracle rows for **all planner-bearing** interfaces and gating planner rows
  on oracle competence (success >= 0.8), following the
  [fair-interface-baselines](fair-interface-baselines.md) protocol.

## Stack

- top-level: `feat/lafan1-fromscratch-eval` (branched from PR#19)
- RLOpt: `feat/baseline` (includes the concat-representation and hl-skill
  encoder fixes; PR#19's pinned RLOpt commit predates them)
- ImitationLearningTools: `main` (zarr short-trajectory sharding fix)

The submitted snapshot uses top-level commit `9d12c03` with the intended
dirty top-level overlay, RLOpt `7349731` on `feat/baseline`, and
ImitationLearningTools `d3d5532` on `main`. The dependency overlays were
clean when staged.

## Invalidated Skynet training snapshot

As of 2026-07-14, the active low-level training jobs are:

| Interface | Job | Task | Contract |
| --- | ---: | --- | --- |
| latent skill oracle | 3493194 | `Isaac-Imitation-G1-Latent-v0` | h10 z, 5 Hz |
| EE trajectory oracle | 3495690 | `Isaac-Imitation-G1-v0` | held h10 chunk, 5 Hz |
| full-body trajectory oracle | 3495708 | `Isaac-Imitation-G1-v0` | held h10 chunk, 5 Hz |
| single-frame full-body oracle | 3496079 | `Isaac-Imitation-G1-v0` | direct frame, 50 Hz |

All four jobs were running at the last read-only check on 2026-07-14. Query
Skynet before treating their scheduler state as current, but do not use their
checkpoints for paper results because the input data failed the body-frame
audit.

### Local Phase 2 wiring gate

The shared continuous planner pipeline passed its representative LAFAN1 code
gate on `walk1_subject1` on 2026-07-14. The run used the exact
750,059,520-frame checkpoints from jobs 3493194, 3495690, and 3495708, plus the
horizon-10 latent skill encoder. All three interfaces used seed 0, a ten-frame
causal robot history, 5 Hz planner decisions, two demonstration rows, two
planner-rollout rows, one pretrain update, one retrain update, and a requested
20-step evaluation budget.

The local audit passed at:

```text
.tmp/phase2_lafan_walk1_smoke/phase2_protocol_audit.json
```

This is a wiring result only. The full-body and EE checkpoints terminated
after two actual evaluation steps. They do not pass the oracle quality gate,
and no planner comparison from this smoke run should appear in the paper. The
active 5B jobs remain the source of candidate low-level checkpoints.

The original EE/full-body jobs 3493615 and 3493792 used a frozen anchor for
the entire held chunk. They were stopped at approximately 560M and 510M
frames after that semantic mismatch was found. Jobs 3495690 and 3495708 are
the replacements: the held chunk's content remains frozen, but its coordinates
are re-expressed in the current robot anchor each control step. A smoke test
covered 125 re-expression checks with maximum error below `5e-6`.

The first direct-reference submission, 3496069, was cancelled because it used
a relative manifest path, while `data/` is intentionally excluded from the
staged repository. The final job, 3496079, uses:

```text
/nethome/fwu91/scratch/Research/IsaacLab/data/lafan1/manifests/g1_lafan1_manifest.json
```

Its staged snapshot is:

```text
/nethome/fwu91/scratch/Research/IsaacLab/isaaclab_20260714_091028
```

The snapshot contains `repo_sync_manifest.tsv`. It was built by excluding
`data/`, `.tmp/`, and the three submodule directories from the top-level
sync; hard-linking the prior IsaacLab payload; and syncing explicit RLOpt and
ImitationLearningTools overlays. The submission used the shared SIF, skipped
the cache copy, requested an 8192 MB overlay, and used the long QOS with a
six-day walltime.

## Basic direct-reference oracle

`single_frame_full_body` is the most basic low-level training setup in this
comparison. At every 50 Hz control step, the actor receives the current
reference frame through `expert_motion`, `expert_anchor_pos_b`, and
`expert_anchor_ori_b`. These are the same semantic expert entries included
in the critic observation, but actor and critic are separate observation
groups. With command-side expert noise disabled, the shared command entries
have the same numerical values; the critic additionally receives privileged
robot state.

This baseline asks a narrower question than the matched 5 Hz planner table:
**what performance is lost when a direct 50 Hz full-body reference is
compressed into a 5 Hz temporally extended latent skill?**

| Oracle | Command size/rate | Intended interpretation |
| --- | --- | --- |
| latent skill | 256-D z at 5 Hz, held for h10 | compressed temporally extended representation |
| single-frame full body | 67-D expert frame at 50 Hz | direct-reference low-level ceiling |

The direct oracle is intentionally the stronger, simpler information
interface. It is not a fourth row in the cadence-matched planner comparison.
Use it as a low-level oracle ceiling and label the 10x command-rate advantage.

The submission matches the latent oracle everywhere that is meaningful:

- full 40-motion LAFAN1 manifest, seed 0, plain `IPMD`, 4096 environments,
  and 50,865 iterations (approximately 5B environment frames);
- `agent.command_space=single_frame_full_body` and
  `agent.ipmd.use_latent_command=false`;
- no future command window and no command hold;
- `env.command_observation_source=reference`;
- reset steps 0-200 with `env.random_reset_full_trajectory=false`, matching
  the active latent job's starting-state distribution;
- all reward-estimator loss coefficients set to zero;
- critic cells `[768,512,256]` to match latent critic capacity; the policy
  already uses `[512,256,128]`.

Do not replace this setup with a proprio-only policy: that would remove the
reference command and test a different problem. Do not change the low-level
algorithm when the purpose is to isolate representation/interface cost.

For a reproducible render-only submission check from the repository root:

```bash
DRY_RUN=1 \
CLUSTER_GIT_SYNC_FIRST=0 \
CLUSTER_EXTRA_RSYNC_EXCLUDES="data/ .tmp/ IsaacLab/ RLOpt/ ImitationLearningTools/" \
CLUSTER_LINK_ISAACLAB_FROM_PREVIOUS=1 \
CLUSTER_SKIP_CACHE_COPY=1 \
CLUSTER_USE_SHARED_SIF=1 \
CLUSTER_OVERLAY_SIZE_MB=8192 \
CLUSTER_SLURM_QOS=long \
CLUSTER_SLURM_TIME_LIMIT=6-00:00:00 \
COMMAND_SPACES=single_frame_full_body \
SEEDS=0 \
NUM_ENVS=4096 \
MAX_ITERATIONS=50865 \
COMMAND_FUTURE_STEPS=0 \
MANIFEST=/nethome/fwu91/scratch/Research/IsaacLab/data/lafan1/manifests/g1_lafan1_manifest.json \
PROJECT_NAME=G1-Imitation-LAFAN1-FromScratch \
GROUP_NAME=lafan1_fromscratch_h10_ipmd_5b_seed0 \
RUN_PREFIX=lafan1_fromscratch_h10_ipmd_5b_seed0 \
EXTRA_OVERRIDES="env.random_reset_step_min=0 env.random_reset_step_max=200 env.random_reset_full_trajectory=false env.command_hold_steps=0 agent.ipmd.reward_loss_coeff=0.0 agent.ipmd.reward_l2_coeff=0.0 agent.ipmd.reward_grad_penalty_coeff=0.0 agent.ipmd.reward_logit_reg_coeff=0.0 agent.ipmd.reward_param_weight_decay_coeff=0.0 agent.value_function.num_cells=[768,512,256] agent.logger.backend=wandb" \
experiments/command_space_ablation/submit_cluster_oracle_ablation.sh
```

Remove `DRY_RUN=1` only after checking that the rendered batch script retains
the absolute persistent manifest path.

## Runner

```bash
experiments/interface_baselines/run_lafan1_from_scratch_comparison.sh
```

Defaults: `SEED=0`, `HORIZON_STEPS=10`, `Z_DIM=256`, `STATE_HISTORY_STEPS=9`,
`LOW_LEVEL_ALGO=IPMD`, `TRAIN_NUM_ENVS=4096`, `TRAIN_MAX_ITERATIONS=50865`
(~5B frames at 24 steps/iteration), `TRAIN_SAVE_INTERVAL=250000000`,
`TRAIN_WALLTIME=7-00:00:00`, `EVAL_NUM_ENVS=4`.

Those `TRAIN_*` values apply only to the historical cluster stages. The active
paper protocol now uses the focused BONES-SEED Skynet launcher documented
above. Local stages use `LOCAL_TRAIN_MAX_ITERATIONS`, defaulting to the largest
whole iteration count at or below 50M frames (508 iterations with 4096
environments), and reject larger local requests.

## Local sequential workflow (single GPU)

Local stages are diagnostic only: they show that data, training, checkpointing,
and evaluation behave as intended. They are not a substitute for a long
cluster run and are capped at about 50M total environment frames per
controller. The older EE and full-body rows are appendix diagnostics, not main
paper rows.

```bash
# full chain: latent stack -> EE -> full-body -> per-motion evals -> summary
STAGE=local-all \
experiments/interface_baselines/run_lafan1_from_scratch_comparison.sh

# or stage by stage
STAGE=local-train experiments/interface_baselines/run_lafan1_from_scratch_comparison.sh
STAGE=local-eval  experiments/interface_baselines/run_lafan1_from_scratch_comparison.sh
STAGE=summarize   experiments/interface_baselines/run_lafan1_from_scratch_comparison.sh
```

At the default 4096 environments, local training uses 508 iterations, or
49,938,432 frames per controller. Do not continue those checkpoints past 50M
locally to chase convergence. Use Skynet for longer convergence checks and all
paper-facing results.

Resume behavior: each sub-stage is skipped automatically when its final
artifact exists — the latent stack via
`<root>/latent/base_pipeline/low_level_checkpoint.txt` (with encoder/planner
checkpoints reused individually), EE/full-body via
`<root>/<interface>_checkpoint.txt`. An interrupted EE/full-body RL run can be
resumed with `EE_RESUME_CHECKPOINT=` / `FB_RESUME_CHECKPOINT=` pointing at the
newest `model_step_*.pt`. Rerunning `STAGE=local-all` after an interruption
therefore continues where it left off (finished per-motion eval outputs are
also reused by the underlying runners where they support it).

The local stages set `TMPDIR` and the Unitree USD cache to workstation-safe
paths (`/tmp/isaaclab_pipeline_$USER`, `~/.cache/isaaclab_imitation/`),
because the latent pipeline's `/data/tmp` default does not exist locally.

## Historical three-interface cluster workflow

This section preserves the earlier EE/full-body experiment path for debugging
and appendix work. It is not the active main comparison and should not be run
as a three-style sweep. The active main rows are DiffSR latent and the exact
10-frame vanilla packet through the frozen direct tracker. Use the dedicated
latent-only launcher and guarded qualification described above, followed by
`submit_phase4_no_language_skynet.sh` only after the gate passes.

The historical runner now fails closed whenever EE or full-body stages are
enabled. `ALLOW_LEGACY_THREE_INTERFACE=1` is required to run those paths and
should be used only when the user explicitly requests an appendix diagnostic;
it is not part of the paper protocol.

Note the SLURM submit script pins `--qos=short`; if that QOS caps walltime
below 7 days, resume the RL jobs from the latest `model_step_*.pt` with
`scripts/rlopt/train.py --checkpoint ...` (checkpoints are saved every
`TRAIN_SAVE_INTERVAL` frames).

### 1. Train (3 cluster jobs, ~5B frames each)

```bash
DRY_RUN=1 STAGE=submit-train \
experiments/interface_baselines/run_lafan1_from_scratch_comparison.sh
# then rerun without DRY_RUN=1
```

- latent: `MODE=lafan1-motion-tracking` with `RUN_BASE_PIPELINE=1` runs
  `run_lafan1_no_language_pipeline.sh` (skill encoder 5000 updates -> base
  planner 5000 updates -> IPMD low-level on `Isaac-Imitation-G1-Latent-v0`
  with `command_source=hl_skill`).
- EE / full-body: `submit_cluster_oracle_ablation.sh` trains IPMD on
  `Isaac-Imitation-G1-v0` with `agent.command_space=ee_trajectory` /
  `full_body_trajectory`, `command_future_steps=10`, reference commands.

### 2. Evaluate (3 cluster jobs, per-motion, all 40 motions)

Locate the trained EE/full-body checkpoints first:

```bash
RUN_PREFIX=lafan1_fromscratch_h10_ipmd_5b_seed0 \
experiments/command_space_ablation/list_cluster_checkpoints.sh
```

```bash
STAGE=submit-eval \
EE_TRAJECTORY_CHECKPOINT=logs/rlopt/ipmd/Isaac-Imitation-G1-v0/.../models/model_step_....pt \
FULL_BODY_TRAJECTORY_CHECKPOINT=logs/rlopt/ipmd/Isaac-Imitation-G1-v0/.../models/model_step_....pt \
experiments/interface_baselines/run_lafan1_from_scratch_comparison.sh
```

The latent job reads its checkpoints from
`<RUN_ROOT>/latent/base_pipeline/`; override the low-level with
`LATENT_LOW_LEVEL_CHECKPOINT=...` if the recorded absolute path is stale.

Per motion this produces: latent oracle closed-loop, latent base-planner
closed-loop, per-motion planner finetune + finetuned closed-loop, EE/FB
low-level oracle, EE/FB chunk planner pretrain + pretrained closed-loop,
achieved-state finetune + finetuned closed-loop.

### 3. Summarize

```bash
STAGE=summarize \
experiments/interface_baselines/run_lafan1_from_scratch_comparison.sh
```

Writes `summary/per_motion.csv`, `summary/method_summary.csv`, and
`summary/method_summary.md` under the experiment root
(`logs/lafan1_motion_tracking_evaluation/lafan1_fromscratch_h10_ipmd_5b_seed0/`).

The table contains oracle, planner-base, and planner-finetuned rows for all
three interfaces. `success_gated` / `*_gated` columns average only over
motions where that interface's oracle passes the 0.8 success gate. If an
interface's oracle fails the gate on most motions, its planner rows measure
low-level incompetence, not interface quality, and must not be used for
comparison claims — this is the failure mode that invalidated the PR#19
table.

Known metric caveat: the EE/full-body *oracle* rows use `1 - done_rate` as
the success fallback because `evaluate_checkpoint.py` does not emit the
threshold-based `tracking_success_rate`; the `success_source` column records
this. Planner rows use the threshold-based metric on all interfaces.

## High-level cadence contract

All interfaces share the same System-1/System-2 contract: the high level
emits one command packet per 10-step macro step (5 Hz at 50 Hz control), the
low level executes it for the full hold.

| Interface | HL query rate | packet | packet dim |
| --- | --- | --- | --- |
| latent_skill | 5 Hz | z (held, sin/cos phase) | 256 |
| ee_trajectory | 5 Hz | 11-frame EE pose chunk (held, shifted view) | ~936 |
| full_body_trajectory | 5 Hz | 11-frame full-body chunk (held, shifted view) | ~1742 |

Set `BASELINE_PLANNER_UPDATE_INTERVAL=1 BASELINE_COMMAND_HOLD_STEPS=0` to
reproduce the legacy per-step receding-horizon baselines as a diagnostic
(that variant gives the chunk planners 10x the replanning frequency and
should be labeled as such, not presented as the matched comparison).

## Remaining protocol asymmetries (accepted, favor the baselines)

- The chunk planners are trained per motion (single-motion specialists),
  while the latent planner is pretrained once on all 40 motions and only
  finetuned per motion. If the latent interface still wins, the claim is
  stronger.
- The chunk planners are chunked Transformers (~20M params); the latent
  planner is the existing flow-matching SkillCommander (~2M params). Report
  parameter counts alongside tracking metrics.
