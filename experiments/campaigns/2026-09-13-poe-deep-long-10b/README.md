# Clean PoE: depth 7, larger batch, longer pretraining, then 10B tracker

One combined scaling arm, `depth7_b32768_u200k`, seed 0. Fresh pretraining
followed by a frozen-encoder tracker. This is a combined experiment, not an
isolated estimate of depth, batch size, or training duration.

## Protocol

- Clean direct PoE denoiser: `eps = sum_k z_k E_k(s, y_t, t)`;
  `phi_parameterization=identity`, `mu_conditioning=pair`, z and feature width 64.
  No affine bias or extra command coordinates are introduced by the decoder.
- Expert-network hidden widths: `[1024,1024,2048,2048,2048,2048,2048]` (depth 7).
- Pretrain: 200,000 updates, batch 32,768, encoder/NTP learning rate 0.0003.
  This is 6,553,600,000 sampled windows, 16 times the original 50k x 8192
  control and four times each previous equal-example optimization arm.
- Preserve capacity-sweep SIGReg coefficient 1, latent L2 coefficient 0.001,
  deterministic LayerNorm encoder, past-five plus current source,
  intermediate window, horizon 10, boundary_next diffusion target,
  endpoint coefficient zero, root_qpos inputs and stride 1.
- Same BONES-SEED 129785 arrays, persist ID `bones_seed_sonic_full_129785@e714bbff`.
  Trajectory split seed 0, evaluation fraction 0.1. Ordinary evaluation uses
  batch 8192 and four batches; final paired probe uses 1024 examples per
  split per noise level, seed 1729. Logging every 250 updates.
- Low level: existing PoE tracker protocol, 20,480 environments, 24-step
  rollouts, 20,346 total iterations (approximately 10B frames), hold 1,
  64-D z plus sin/cos phase (66 actor-command coordinates), encoder frozen.
  Reward, network, reference selection and curriculum settings match
  `2026-09-12-poe-base-z64-10b` apart from run identity and encoder path.
- Persistent output:
  `/storage/ice-shared/vip-vwt/scratch-fwu91/poe_deep_long_10b/depth7_b32768_u200k_seed0/`.
  Low level binds `encoder/checkpoints/latest.pt` produced by this pretrain.
- W&B project `g1-bs-pareto`, group `poe-deep-long-10b`.

All three allocations use one H200, 16 CPUs, 160G RAM, 15:59 walltime.
`pretrain -> lowlevel1` uses afterok. `lowlevel1 -> lowlevel2` uses afterany
and resumes the persistent tracker checkpoint with the full cumulative
frame target; the second allocation does not request another 10B frames.
Each stage has a separate compiler cache. No planner training is included.

The exact linearity is in the diffusion prediction field. These experiments
do not establish conservative scalar energies, exact clean-density PoE
sampling, or semantic composability; low-level performance remains to be measured.

## Qualification and launch

Frozen plan:
`logs/cluster_control/poe-deep-long-10b/poe-deep-long-10b-depth7_b32768_u200k-s0-20260913-035015-8e1a34bf`.

Plan SHA: `8e1a34bf1d4e9da10e92b1c832626f02d92a1d83dc0d868520202d00e5db104c`.
Cluster dataset, writable storage, free space, image, log directory and
credential-file preflights passed. Resolved-command checks verified the
clean z64 parameterization and unchanged low-level protocol.
Local qualification artifacts: `logs/poe_deep_long_smoke/`.

Qualification passed: two pretrain updates at full batch 32768; checkpoint
metadata and NTP-head presence verified; one low-level update at 32
environments (768 frames), command width 66, clean exit. Three existing
capacity/rescue campaign tests passed. Documentation was added after the
initial plan, so submission uses a freshly sealed plan recorded below.

## Submitted 2026-09-13 03:53 UTC

- Pretrain: **5770203**.
- Low level: **5770204**, afterok:5770203.
- Low-level continuation: **5770205**, afterany:5770204.
- Submitted plan: `poe-deep-long-10b-depth7_b32768_u200k-s0-20260913-035317-b364692a`.
- SHA: `b364692a6021a759f1d2c0dc1cbe15b0140aec09d6df4a945834915b868c74f3`.
- Verified source archive: `605923bff711657f9fa24196c2a2919a6077cfde0eba775b389916068118ca82` (70.1 MB).
- The earlier plan was superseded after documentation changed; no drift
  override was used. No algorithm changes or submodule commits were made
  for this combined arm. Existing uncommitted sources are captured by the archive.

## Combo recipe audit, 2026-09-13

The submitted PoE tracker follows the earlier PoE/base tracker, not the
combo tracker. Submitted combo plans explicitly add ten-step actor history
on projected gravity, angular velocity, joint position, joint velocity and
last action; weight decay 0.01; and linear critic decay to 1e-5. These were
omitted from this run. Its runtime agent config confirms weight decay 0,
constant critic LR 0.001, and latent_command excluded from input normalization.
Combo uses 16384 environments versus this run's 20480. Combo-50B additionally
uses SONIC reset sampling with staged uniform-share 0.8 -> 0.2 and a 50B
budget; the original combo-10B retains random80_adaptive20 like this run.
Therefore current tracking differences do not isolate the skill parameterization.

Live snapshot: pretrain 5770203 COMPLETED, exit 0:0, runtime 09:05:52;
tracker 5770204 RUNNING; continuation pending. Final pretrain update 200000:
paired held-out dynamics loss 4.77668; ordinary held-out NTP 5.06857;
mean latent coordinate std 0.0151615, RMS 0.0253817, effective rank 57.616.
Tracker log at 11:01:32 reports 1,140,326,400 frames. No matched tracking
evaluation was performed by this audit.

Suggested next comparison: same-budget combo-10B tracker recipe with only
the encoder binding changed, then separately test fixed invertible latent
scaling (no centering). Such scaling preserves linear code composition;
small command scale is a conditioning hypothesis, not proven causality.
No running jobs or training settings were changed by this audit.
