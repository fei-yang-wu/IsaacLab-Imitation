# Direct affine-phi LSTM tracker (2026-09-04) -- PARKED 2026-09-06

This campaign starts from the `lstm_affine_std` tracker recipe. The policy
receives a trained 64-D affine feature `phi(s_history, z)` plus two phase values.
The feature head is pretrained at width 64, and the tracker has 20,480
environments as requested. Both changes are recorded for the comparison.

The fixed recipe is:

- 64-D continuous latent and a LayerNorm past-5 affine encoder;
- `optim.weight_decay=1e-2` and a linear critic learning-rate schedule to
  `1e-5`;
- a 256-unit LSTM actor and a feed-forward critic;
- `random80_adaptive20` resets and the 5M-to-30M termination curriculum;
- hold 1, sine/cosine phase, 12,288 environments, and 10B frames;
- W&B project `g1-bs-pareto`.

There is no explicit actor observation history. The LSTM carries the policy's
temporal state. The frozen encoder uses its past five macro frames only to
compute the 64-D affine feature. The policy command is
`[phi(s_history, z); sin(phase); cos(phase)]`, with width 66. At hold 1 the
phase is constant `[0, 1]`; it stays enabled by user decision. There is no
learned projection after phi. The recurrent actor itself remains nonlinear.

This is a head-to-head comparison with the existing HeadLinear result. It is
not a matched one-variable ablation against HeadLinear.

The RLOpt runtime now restores the trained `jepa_ntp` DiffSR head from the
merged past-5 checkpoint and requires the six-frame source window. It refuses
a merged checkpoint that has no trained head. The source contains six 38-D
root_qpos frames (228 values), anchored on the current frame at stride 1.
The five-tensor command-sampler return contract stays compatible with planner
adapters, and z-only commands do not gather unused history.

## Budget and stages

The campaign contains one arm, `phi_lstm`, at seed 0. The first H200 job
pretrains the encoder for 50,000 updates with batch size 8192. It uses the
past-5 affine merged-head recipe with `diffsr_feature_dim=64`. Two tracker
jobs follow, each carrying the full 10B frame target (20,346 iterations of
20,480 x 24 frames). The first tracker requires successful pretraining; its
successor uses `afterany` so a wall-time checkpoint resumes the global budget.

Data: BONES-SEED 129,785 motions, persist id
`bones_seed_sonic_full_129785@e714bbff`, reference arrays
`/storage/ice-shared/vip-vwt/g1-imitation/datasets/bones_seed_full/ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1`.
Outputs: `/data/direct_affine_phi64/phi_lstm_seed0/`.

W&B project: `g1-bs-pareto`; group: `direct-affine-phi`.

## Plan

From the repository root:

```bash
./experiments/campaigns/2026-09-04-direct-affine-phi/submit.sh phi_lstm 0
```

The wrapper validates, preflights, and freezes a plan. It does not submit.
Review the printed plan and use its exact `submit --confirm <PLAN_SHA>` command
only after explicit approval.

## Status

- 2026-09-04: corrected the campaign after the actor-history clarification.
  All earlier plans for the 256-D feature are obsolete and were not submitted.
- 2026-09-04: user authorized ICE submission with a 64-D feature, phase on,
  20,480 environments, and 10B frames. Local qualification precedes submission.
- Qualification: 165 RLOpt tests and 18 control-plane tests pass. A simulator
  smoke completed two encoder updates at batch 8192, then two tracker
  iterations with 32 environments and the production 24-step rollout. Saved
  configuration confirms `latent_dim=66`, `code_latent_dim=64`, command mode
  `phi`, LSTM size 256, and zero actor observation-history lengths. Logs:
  `logs/direct_affine_phi64_smoke/`. This checks wiring, not H200 capacity at
  20,480 environments or convergence.
- RLOpt commit: `e0d8a88`, published on `feat/direct-phi64-conditioning`.
- Submitted 2026-09-04 after eight live ICE preflight checks passed. Workspace
  commit `ed20123`; unrelated existing changes were included in the archived
  workspace but were not committed. The control plane recorded no drift
  between plan and submission. Archive SHA:
  `fb11624344847732fdc0bf16b308e8874cf4c92866b11a8086a2a4108ae71f51`.

| Stage | ICE job | Dependency | State at submission check |
| --- | --- | --- | --- |
| Encoder pretrain | 5697130 | none | RUNNING |
| Tracker segment 1 | 5697131 | afterok:5697130 | PENDING |
| Tracker segment 2 | 5697132 | afterany:5697131 | PENDING |

Submission record:
`logs/cluster_control/direct-affine-phi/direct-affine-phi-phi_lstm-s0-20260904-234437-d86a4691/submission-20260904-234515.json`.

Exact submit command, run from the repository root using the installed Pixi
environment (the `pixi` launcher was unavailable on PATH):

```bash
PYTHONPATH=source/imitation_experiments .pixi/envs/default/bin/python -m imitation_experiments.pipeline.cluster submit \
  --plan logs/cluster_control/direct-affine-phi/direct-affine-phi-phi_lstm-s0-20260904-234437-d86a4691 \
  --confirm d86a4691e29fece70bec94fa707c56a2c7220585032105f25f955985ef90760a
```

## First submission failed on GPU memory (2026-09-04)

Jobs 5697130-5697132. The pretrain COMPLETED in 42:50 and its encoder is on
disk at `/data/direct_affine_phi64/phi_lstm_seed0/encoder/checkpoints/latest.pt`
(50,000 updates). The tracker died after 7:06:

```
RuntimeError: Graph launch error: Warp CUDA error 2: out of memory
    (in function wp_cuda_graph_launch, warp.cu:3710)
```

raised from `newton_manager.step` -> `wp.capture_launch`, i.e. physics could
not allocate its CUDA graph. At 20,480 environments the recurrent actor's BPTT
activations plus the 491,520-frame rollout buffer take the room. `lowlevel2`
then failed on `--checkpoint points at a tree with no model_step_<N>.pt file`,
a cascade of the first failure, not a second fault.

First fix, 16,384 environments, FAILED the same way (job 5698695, 10:25,
OOM after iteration 77). The phi command path costs more GPU memory than a
z-only command: the frozen encoder gathers a six-frame 228-value source
window per environment per step and the restored DiffSR head runs on it.
Throughput at 16,384 was 135k fps against 149k for the z-only LSTM arms, the
same signal. Second fix: 12,288 environments, the demotion `lstm_nophase`
needed for the identical failure. The 10B frame target is unchanged;
`max_iterations` recomputes to 33,908. Batch shape now differs from
`lstm_affine_std` (16,384), so that comparison carries a batch-size
difference as well as the command representation.

Every resubmission plans only the tracker stages
(`--only-stage lowlevel1,lowlevel2`) so the completed encoder is reused
rather than pretrained again.

## PARKED -- recurrent actor stopped (2026-09-06)

The 10B budget finished on 2026-09-06: job 5699181 COMPLETED at exactly
10,000,171,008 frames in 6:34:50, after 5699180 timed out at 7.32B and resumed
from the 7B checkpoint.

Final row, board `bones_testbed4096_v1`, `--randomization none`, seed 0, one
pass: SR 0.8474, MPJPE-L 22.92 mm, MPJPE-G 65.97 mm, acc 4.789 m/s^2, jerk
205.65 m/s^3, action_delta_l2 0.8813. Survival 313.76 steps; terminations 3,472
`reference_finished`, 418 `ee_body_pos`, 180 `anchor_ori`, 50 `anchor_pos`, 0
`time_out`. That MPJPE-G is the lowest of any arm in this comparison set and
the SR is the lowest; three variables separate this arm from `combo` (phi
conditioning, affine encoder, recurrent actor), so it attributes nothing on its
own.

User decision, 2026-09-06: the recurrent-actor axis is parked. No new LSTM arm
is submitted. See `wiki/current-status.md`, section "Recurrent (LSTM) actor:
PARKED (2026-09-06)", for the matched pairs and the confounds.

## Phi-conditioning diagnostic (2026-09-07)

Recomputed all available clean-board checkpoint rows using the canonical
per-episode, survival-step-weighted reduction. Raw JSON mirrors, executable
analysis script and complete output are under `logs/phi_conditioning_eval/`.
All rows passed ordered canonical rank, full completion and zero-timeout
checks. This is one seed per recipe, with unmatched architecture/encoder/
critic inputs and training batch sizes; it cannot identify a phi causal effect.
The recurrent evaluator qualification caveat remains.

Full-board final rows (SR / success-only MPJPE-L / MPJPE-G): phi_lstm
0.8474 / 22.92 / 65.99 mm; nolatent 0.9043 / 23.20 / 79.16 mm;
z-conditioned lstm_affine_std 0.9180 / 23.14 / 112.43 mm;
z-conditioned combo MLP 0.9214 / 22.65 / 88.52 mm. Minor differences
from older rows are canonical reduction differences.

To test success-set selection, restrict errors to the intersection of each
pair's successful trajectory ranks. This is a diagnostic subset, not a new
headline board or a causal control. Errors retain survival-step weighting.

| Pair, A / B | Shared successes | Only A / only B succeed | Shared MPJPE-L A / B (mm) | Shared MPJPE-G A / B (mm) |
| --- | ---: | ---: | ---: | ---: |
| phi_lstm / nolatent | 3434 | 37 / 270 | 22.44 / 21.44 | 63.35 / 61.52 |
| phi_lstm / z lstm_affine_std | 3454 | 17 / 306 | 22.77 / 20.89 | 65.62 / 75.28 |
| nolatent / z combo MLP | 3660 | 44 / 114 | 23.03 / 21.71 | 78.14 / 72.89 |

The phi_lstm-vs-nolatent global-error advantage disappears on shared
successes. The nolatent-vs-combo advantage reverses. Phi_lstm retains
12.8% lower global error than the z LSTM on their shared successes, but
has 9.0% higher local error and succeeds on 289 fewer motions overall.
That retained global difference accompanies lower root XY error
(56.1 vs 66.2 mm), consistent with less translation drift on this subset.
It does not establish phi as the cause.

Nolatent retains lower body jerk than combo on shared successes:
179.27 vs 192.42 m/s³ (6.8% lower), while action delta differs by 1.1%
(0.825 vs 0.834). Phi_lstm is not smoother than z LSTM on their shared
successes: jerk 205.47 vs 193.80 m/s³; action delta 0.880 vs 0.837.

Learning trajectory: phi_lstm at 3.5B, 8B, 9B, 10B gives
0.8101/25.96/85.67, 0.8459/23.71/68.64, 0.8496/23.33/68.14,
0.8474/22.92/65.99 (SR/L/G). Late training improves pose errors without
consistent success gains. Nolatent at 8.5B, 9B, 10B gives
0.9014/23.86/83.74, 0.8979/23.85/85.73, 0.9043/23.20/79.16;
this sparse one-pass tail does not establish convergence.

Failure flags at 10B: phi_lstm has 418 EE-height, 180 root-orientation,
and 50 root-height failures; z LSTM has 267, 65, 20. Nolatent has
305, 82, 27; combo has 265, 54, 22. Flags can overlap and must not
be summed as disjoint failures.

A causal phi test needs separate trackers trained with z versus phi from
the same frozen encoder, identical MLP/history, critic inputs, rewards,
reset schedule and batch size, with repeated seeds/evaluations. Switching
an existing actor's command at evaluation time would break its trained
interface. No new training was submitted for this analysis.
