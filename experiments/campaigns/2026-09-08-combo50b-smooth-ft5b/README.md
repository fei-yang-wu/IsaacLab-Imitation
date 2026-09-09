# Combo50B smoothness fine-tuning: completed +5B campaign

Completion record, 2026-09-09. All four seed-0 arms completed the 5B fine-tune
target, finishing at **55,000,301,568 cumulative frames** from the
50,000,166,912-frame combo50b base. All four evaluation boards are complete:
**52 cells** (16 latest, 4 nominal 3B, 16 4B, 16 5B).
Completion, metric values and crash history below are user-provided campaign
evidence; this documentation update did not access the cluster or independently
re-read remote result JSONs. Job and artifact provenance is consolidated from
the existing submission record. The evaluator call chain was checked locally.

## Training contract and provenance

Each arm continued the exact `model_step_50000166912.pt`; no new encoder
training. The inherited contract retains p5_affine z64+phase, ten-step actor
history, fullbatch/3epochs, EE/wide rewards, Newton physics, 16,384 environments,
24-step rollouts and the final reset mix (20% uniform).

| Arm (seed 0) | Change from base | Training job | afterany resume fallback |
| --- | --- | --- | --- |
| action01 | `action_rate_l2`: -0.03 → -0.1 | 5738566 | 5738568 |
| antishake4 | `anti_shake_ang_vel`: -0.005 → -0.02; action-rate stays -0.03 | 5738572 | 5738573 |
| ema08 | `EMAJointPositionAction`, alpha=0.8; rewards unchanged | 5738575 | 5738576 |
| normfix | `agent.ppo.update_normalizers_after_rollout=true`; rewards/actions unchanged | 5739801 | 5739807 |

The requested cap was 55,000,166,912; rollout rounding produced
5,000,134,656 additional frames per arm. Each allocation used one H200,
160G RAM and 16 CPUs, with up to two 15:59 segments. Fallbacks resume the
arm's output tree with the same total cap and add no budget; their IDs record
submission provenance, not an assertion that every fallback performed training.
Checkpoints were saved every 0.5B. W&B project/group:
`g1-bs-pareto` / `combo50b-smooth-ft5b`.

Loaded optimizer state was retained: actor/log_std LR
1.7341529915832615e-5, critic LR 1e-5. `critic_lr_schedule=constant` prevents
the extended cap from reheating the completed 50B linear critic schedule;
other inherited behavior includes actor adaptation. No reset curriculum restart.
EMA filters scaled joint-position targets, with alpha weighting the new target;
its state resets between episodes and alpha=0.8 is retained at evaluation/runtime.
Normfix freezes actor/critic input statistics through collection, optimization
and KL diagnostics, then updates once from collected current observations before
checkpointing and collector synchronization. Loaded statistics are retained;
the other arms keep legacy forward-driven updates.

Training root: `/data/combo50b_smooth_ft5b/<arm>_seed0/tracker/`.
Within each run below, checkpoints are `models/model_step_<pinned_frames>.pt`.

| Arm | Run directory |
| --- | --- |
| action01 | `2026-09-08_15-57-43_wandb-c50bsft-action01-s0-3637ed` |
| antishake4 | `2026-09-08_16-03-53_wandb-c50bsft-antishake4-s0-b3a47d` |
| ema08 | `2026-09-08_16-05-07_wandb-c50bsft-ema08-s0-9ae0e3` |
| normfix | `2026-09-08_17-59-04_wandb-c50bsft-normfix-s0-e38be5` |

Historical qualification: the first three plans passed dataset/checkpoint/storage/
container preflight; normfix also passed 15 focused normalization, IPMD DAgger
and L2T tests and a real-checkpoint Newton smoke with three PPO updates
(1,152 additional frames, exit 0). These are prior checks, not rerun here.

## Evaluation protocol, boards and jobs

Evaluation uses `bones_testbed4096_v1` and frozen `sonic_capability124_v1`,
clean and no-push robust, seed 0, deterministic actions, and each arm's original
environment contract. Runtime and macro caches use CPU for the large board.
The capability124 subset is SONIC-calibrated and in-distribution, not an unbiased
holdout. Milestone plans retain the `eval_latest.yaml` evaluation arguments,
changing checkpoint trees and output roots only. The frozen latest configuration
and checkpoint manifest remain the provenance for that snapshot.

| Board | Completed cells | Result root |
| --- | ---: | --- |
| latest | 16 | `/data/eval/combo50b_smooth_latest_20260908` |
| milestone_3b | 4 (normfix only) | `/data/eval/combo50b_smooth_milestone_3b_20260908` |
| milestone_4b | 16 | `/data/eval/combo50b_smooth_milestone_4b_20260908` |
| milestone_5b | 16 | `/data/eval/combo50b_smooth_milestone_5b_20260908` |

Each result root has `testbed4096` and `capability124` subdirectories.
Latest pin trees: `/data/combo50b_smooth_eval_pinned_20260908/<arm>_seed0`.
Milestone pin trees:
`/data/combo50b_smooth_milestone_{3,4,5}b_pinned_20260908/<arm>_seed0`,
containing `tracker/f<pinned_frames>/models/model_step_<pinned_frames>.pt`
linked to the corresponding arm's training checkpoint. Historical milestone
preflights, exact cell/checkpoint guards and ZIP CRC checks passed.

The following IDs are the recorded submissions; arrows retain known startup
failure/resubmission chains. Completion of all cells is supplied, but this record
does not invent IDs for additional recovered attempts absent from the old notes.

| Board | Arm | testbed4096_clean | testbed4096_robust | capability124_clean | capability124_robust |
| --- | --- | --- | --- | --- | --- |
| latest | action01 | 5740411 → 5740453 | 5740412 | 5740413 | 5740414 |
| latest | antishake4 | 5740415 | 5740416 | 5740417 | 5740418 |
| latest | ema08 | 5740419 | 5740420 | 5740421 | 5740422 |
| latest | normfix | 5740423 | 5740424 | 5740425 | 5740426 |
| 3b | normfix | 5744504 | 5744505 | 5744506 | 5744507 |
| 4b | action01 | 5744443 | 5744444 | 5744445 | 5744446 → 5744481 |
| 4b | antishake4 | 5744482 → 5745038 → 5745800 | 5744483 | 5744484 | 5744497 |
| 4b | ema08 | 5744499 | 5744500 | 5744501 | 5744502 |
| 4b | normfix | 5747055 | 5747056 | 5747057 | 5747058 |
| 5b | action01 | 5745952 | 5745953 | 5745955 | 5745956 |
| 5b | antishake4 | 5746588 | 5746589 | 5746590 | 5746592 |
| 5b | ema08 | 5746606 | 5746607 | 5746608 | 5746609 |
| 5b | normfix | 5747178 | 5747179 | 5747180 | 5747181 |

## Results and interpretation limits

Metric keys are `aggregate.tracking_success_rate` (success rate),
`successful_metrics.tracking_mpjpe_mm.mean` (MPJPE in mm), and
`aggregate.num_evaluated_envs` (`n_motions`). Success rate covers **all evaluated
motions**; MPJPE covers **successful motions only**. The `n_motions` column is
not the successful-motion count. Values retain the supplied precision.

- **Single seed per arm, no repeats, so between-arm gaps are not resolvable.**
- **The latest board is frame-unmatched:** normfix is pinned at 50,500,337,664
  versus 51,500,285,952 for the other three arms.
- **milestone_3b contains normfix only.** The other arms' 3B evaluations live
  in the `latest` run, per the supplied run history; 3B is not a four-arm
  comparison. The supplied `latest` snapshot below contains +1.5B pins for
  those arms, not their 3B rows. No other-arm 3B metric values are supplied here.
- **Normfix's 3B pin (53,000,011,776) is 155,136 frames BELOW the exact 3B
  boundary (53,000,166,912), while every other pin is above its boundary.**
  Other latest pins exceed their +1.5B boundary by 119,040 frames; normfix's
  latest pin exceeds +0.5B by 170,752. The 4B and 5B pins exceed their exact
  boundaries by 186,368 and 134,656 frames respectively.
- A comparison to the original base includes extra training; without a matched
  unchanged continuation, it cannot isolate a causal benefit of normfix or
  another intervention. These SR/MPJPE observations do not establish smoothness
  improvement: jerk/action-delta measurements and matched completed-clip
  intersections are not included in the supplied results.

### latest

| Arm | Row | Success rate | MPJPE (mm, successful motions) | n_motions | Pinned cumulative frames |
| --- | --- | ---: | ---: | ---: | ---: |
| action01 | capability124_clean | 1.0 | 18.743 | 124 | 51500285952 |
| action01 | capability124_robust | 0.99194 | 21.413 | 124 | 51500285952 |
| antishake4 | capability124_clean | 0.99194 | 18.295 | 124 | 51500285952 |
| antishake4 | capability124_robust | 0.97581 | 19.380 | 124 | 51500285952 |
| ema08 | capability124_clean | 1.0 | 18.906 | 124 | 51500285952 |
| ema08 | capability124_robust | 0.98387 | 19.896 | 124 | 51500285952 |
| normfix | capability124_clean | 0.99194 | 18.419 | 124 | 50500337664 |
| normfix | capability124_robust | 0.98387 | 20.782 | 124 | 50500337664 |
| action01 | testbed4096_clean | 0.98413 | 21.256 | 4096 | 51500285952 |
| action01 | testbed4096_robust | 0.97656 | 23.330 | 4096 | 51500285952 |
| antishake4 | testbed4096_clean | 0.98364 | 20.688 | 4096 | 51500285952 |
| antishake4 | testbed4096_robust | 0.97607 | 22.280 | 4096 | 51500285952 |
| ema08 | testbed4096_clean | 0.98413 | 20.809 | 4096 | 51500285952 |
| ema08 | testbed4096_robust | 0.97607 | 22.511 | 4096 | 51500285952 |
| normfix | testbed4096_clean | 0.98364 | 20.732 | 4096 | 50500337664 |
| normfix | testbed4096_robust | 0.97461 | 22.354 | 4096 | 50500337664 |

### 3b

| Arm | Row | Success rate | MPJPE (mm, successful motions) | n_motions | Pinned cumulative frames |
| --- | --- | ---: | ---: | ---: | ---: |
| normfix | capability124_clean | 1.0 | 18.277 | 124 | 53000011776 |
| normfix | capability124_robust | 0.99194 | 20.820 | 124 | 53000011776 |
| normfix | testbed4096_clean | 0.98462 | 20.353 | 4096 | 53000011776 |
| normfix | testbed4096_robust | 0.97583 | 22.342 | 4096 | 53000011776 |

### 4b

| Arm | Row | Success rate | MPJPE (mm, successful motions) | n_motions | Pinned cumulative frames |
| --- | --- | ---: | ---: | ---: | ---: |
| action01 | capability124_clean | 1.0 | 18.547 | 124 | 54000353280 |
| action01 | capability124_robust | 0.96774 | 20.313 | 124 | 54000353280 |
| antishake4 | capability124_clean | 1.0 | 18.013 | 124 | 54000353280 |
| antishake4 | capability124_robust | 0.99194 | 20.787 | 124 | 54000353280 |
| ema08 | capability124_clean | 1.0 | 18.191 | 124 | 54000353280 |
| ema08 | capability124_robust | 0.99194 | 20.598 | 124 | 54000353280 |
| normfix | capability124_clean | 1.0 | 18.376 | 124 | 54000353280 |
| normfix | capability124_robust | 1.0 | 20.710 | 124 | 54000353280 |
| action01 | testbed4096_clean | 0.98340 | 21.161 | 4096 | 54000353280 |
| action01 | testbed4096_robust | 0.97778 | 23.421 | 4096 | 54000353280 |
| antishake4 | testbed4096_clean | 0.98486 | 20.627 | 4096 | 54000353280 |
| antishake4 | testbed4096_robust | 0.97534 | 22.518 | 4096 | 54000353280 |
| ema08 | testbed4096_clean | 0.98413 | 20.656 | 4096 | 54000353280 |
| ema08 | testbed4096_robust | 0.97729 | 22.749 | 4096 | 54000353280 |
| normfix | testbed4096_clean | 0.98462 | 20.514 | 4096 | 54000353280 |
| normfix | testbed4096_robust | 0.97534 | 22.255 | 4096 | 54000353280 |

### 5b

| Arm | Row | Success rate | MPJPE (mm, successful motions) | n_motions | Pinned cumulative frames |
| --- | --- | ---: | ---: | ---: | ---: |
| action01 | capability124_clean | 0.99194 | 19.227 | 124 | 55000301568 |
| action01 | capability124_robust | 0.97581 | 20.082 | 124 | 55000301568 |
| antishake4 | capability124_clean | 1.0 | 17.955 | 124 | 55000301568 |
| antishake4 | capability124_robust | 0.99194 | 21.575 | 124 | 55000301568 |
| ema08 | capability124_clean | 1.0 | 17.728 | 124 | 55000301568 |
| ema08 | capability124_robust | 0.98387 | 19.682 | 124 | 55000301568 |
| normfix | capability124_clean | 1.0 | 18.498 | 124 | 55000301568 |
| normfix | capability124_robust | 0.99194 | 20.437 | 124 | 55000301568 |
| action01 | testbed4096_clean | 0.98291 | 21.326 | 4096 | 55000301568 |
| action01 | testbed4096_robust | 0.97900 | 23.561 | 4096 | 55000301568 |
| antishake4 | testbed4096_clean | 0.98486 | 20.657 | 4096 | 55000301568 |
| antishake4 | testbed4096_robust | 0.97827 | 22.736 | 4096 | 55000301568 |
| ema08 | testbed4096_clean | 0.98486 | 20.753 | 4096 | 55000301568 |
| ema08 | testbed4096_robust | 0.97852 | 22.669 | 4096 | 55000301568 |
| normfix | testbed4096_clean | 0.98462 | 20.754 | 4096 | 55000301568 |
| normfix | testbed4096_robust | 0.97754 | 22.565 | 4096 | 55000301568 |

## Evaluation infrastructure finding

Approximately 8 of approximately 50 evaluation jobs died during Kit startup
with a libc `getenv` / `XOpenDisplay` segmentation fault. All recovered by
resubmission. This is **not a regression**: the same crash appears in job
`5593235` on 2026-08-27. The approximate job count describes the reported crash
history; the completed result inventory is exactly 52 cells across four boards.

The startup exposure comes from the evaluator entrypoint:
`scripts/rlopt/eval_checkpoint_tree.py` delegates through `runpy.run_module` to
`imitation_experiments.lowlevel.evaluate_checkpoint`, which calls
`AppLauncher(args_cli)` unconditionally. Thus this evaluation path starts Kit
even with `--headless` and `physics=newton_mjwarp`. This call chain explains why
Kit startup is encountered; it does not independently identify the low-level
cause of the libc/XOpenDisplay fault. A kitless path exists in
`scripts/rlopt/eval_skill_commander_closed_loop.py` behind `--assert-kitless`,
but it is a different evaluator. Switching is entrypoint work, not a flag-only
change to this checkpoint-tree evaluator.

## Consolidated submission provenance

Training plans and records: `logs/cluster_control/combo50b-smooth-ft5b/`.
Initial plan SHA prefixes: action01 `8e144b27`, antishake4 `d7d2dbdb`, ema08
`edab19c1`; normfix `72b7754a`. Normfix submission record:
`submission-20260908-212117.json`.
Milestone plans and submission records are under
`logs/cluster_control/combo50b-smooth-milestone-{3,4,5}b-eval/`.
Recorded plan identifiers (including retries):

- `combo50b-smooth-mileston-action01-s0-20260909-033324-2be560ca`
- `combo50b-smooth-mileston-action01-s0-20260909-035740-3856b3df`
- `combo50b-smooth-mileston-antishake4-s0-20260909-035751-76f2f248`
- `combo50b-smooth-mileston-ema08-s0-20260909-035759-36b80e76`
- `combo50b-smooth-mileston-normfix-s0-20260909-035815-f36290c2`
- `combo50b-smooth-mileston-antishake4-s0-20260909-042909-f9c7bb3f`
- `combo50b-smooth-mileston-antishake4-s0-20260909-050200-274e860b`
- `combo50b-smooth-mileston-action01-s0-20260909-052501-4866ade8`
- `combo50b-smooth-mileston-antishake4-s0-20260909-054810-384872c4`
- `combo50b-smooth-mileston-ema08-s0-20260909-054812-077385dd`
- `combo50b-smooth-mileston-normfix-s0-20260909-061202-a820068a`
- `combo50b-smooth-mileston-normfix-s0-20260909-075302-f4e2778f`

Recorded submission filenames: `submission-20260908-212117.json`, `submission-20260909-042954.json`, `submission-20260909-050244.json`, `submission-20260909-052557.json`, `submission-20260909-054905.json`, `submission-20260909-054930.json`, `submission-20260909-061304.json`, `submission-20260909-075355.json`.

| Provenance | SHA-256 |
| --- | --- |
| Initial three-arm training source archive | `8d0eb0ba4b073877e50d55b2e8f8b80d2dce20a2bbcbc1c191a94644c01f854d` |
| Normfix training plan | `72b7754a79f55e202f3249ce3e94236e2d21b687f034d379175552e9410655dd` |
| Normfix training source archive | `b26ba999813a861c9f5071840b0044e88cf71759e219856635c2b5dec90e03d7` |
| Latest evaluation source archive | `41d682dbaaedf5e1904d065ddbe44668aeff7850cec1340fb749421d8ebfef9f` |
| Latest action01 retry source archive | `8260353ad14ef6590fb1499b25100f5f955532122a13cd09820c02a3da969991` |
| Initial action01 4B plan | `2be560ca258166010e0d4d7c7a967441379e171d3e80495069c763911e7aee0e` |
| Milestone multi-arm source archive | `4c828510335e7c4c1a31528fff30562935af8ef84b3dbe8de2b4c4d47fc42930` |

## Action01 55B EC release (2026-09-09)

The final `model_step_55000301568.pt` action01 checkpoint was exported with
`combo64_history10_v1`, original frozen encoder, hold 1, and no output EMA.
Checkpoint SHA-256: `55d9c9608b078cff3bab422f5d86ce376f33afe880b1c69893c39244328e7c9f`.
RLOpt/TorchScript export parity was exact on 512 observations; policy/encoder
ONNX maximum errors were 4.7684e-6 / 1.3113e-6. EC independently verified the
bundle; all 10 downloaded runtime files matched the tested export byte-for-byte.

Published under `controller/action01_55b` and `checkpoints/action01_55b` in
[`fei-yang-wu/ec-g1-gr00t-eval-kit`](https://huggingface.co/fei-yang-wu/ec-g1-gr00t-eval-kit/tree/832f85896c08494d9a229dca0bcc828182a188ac/controller/action01_55b),
immutable revision `832f85896c08494d9a229dca0bcc828182a188ac`.
EC contains a matching model pin and `examples/lifecycle_sim_action01_55b.yaml`.
The separate 10B continuation is not part of this release.

Native asynchronous MuJoCo rehearsal covered every one of the 13 pinned SONIC
deployment examples for its full horizon: **9/13 SONIC successes**, successful
frame-weighted MPJPE-L 27.85 mm; all-motion MPJPE-L/G 62.08/1949.72 mm.
Failures: both Macarena variants, mirrored kick, mirrored one-leg jump.
There were zero runtime faults and scheduler deadline misses, but 3800
zero-lead response-slot misses across 7604 control ticks. The height-only
survival field incorrectly flags the successful squat. This single-pass EC
measurement differs from the randomized Isaac evaluation.

Artifacts: `logs/action01_55b_release_20260909/async_results.json`,
`hf_verification.json`, `async_artifacts/`, and `videos/` (all 13 full-horizon
50-fps reference-left/policy-right clips, including failures). `index.html`
provides a gallery. Follow cameras independently center the two robots;
consult global-error metrics for translation drift. EC validation:
198 tests passed, 6 skipped, and lifecycle YAML parsed successfully.
