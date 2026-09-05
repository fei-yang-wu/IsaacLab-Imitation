# Direct affine-phi LSTM tracker (2026-09-04)

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
- hold 1, sine/cosine phase, 16,384 environments, and 10B frames;
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

Fix: 16,384 environments, which both LSTM arms of `2026-09-02-lstm-hub64-10b`
run at. The 10B frame target is unchanged; `max_iterations` recomputes to
25,432. The resubmission plans only the tracker stages
(`--only-stage lowlevel1,lowlevel2`) so the completed encoder is reused rather
than pretrained again.
