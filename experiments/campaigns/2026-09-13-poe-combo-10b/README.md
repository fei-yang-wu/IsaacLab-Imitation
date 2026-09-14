# PoE encoder with exact combo-10B tracker

Fresh tracker, seed0, using the completed depth7/batch32768/200k PoE encoder.
The reference is `2026-09-01-latent64-probe-10b`, arm `combo`. Resolved
training arguments match exactly apart from encoder binding and run identity.

- Actor proprioceptive history: ten frames on gravity, angular velocity,
  joint positions, joint velocities, and last action; critic single frame.
- Optimizer weight decay 0.01; linear critic LR from 0.001 to 0.00001.
- 16384 environments x 24 steps, approximately 10B frames total.
- Frozen 64-D z plus sin/cos phase, hold1; no latent rescaling.
- Preserve combo-10B random80_adaptive20 resets and 5M-30M termination
  curriculum, rewards, networks, dataset and macro input contract.
- Two H200/16CPU/160G/15:59 allocations; second resumes with afterany,
  retaining the full cumulative budget. Independent compiler caches.
- W&B: `g1-bs-pareto` / `poe-combo-10b`.
- Output: `/storage/ice-shared/vip-vwt/scratch-fwu91/poe_combo_10b/poe_combo_seed0/`.

Encoder source job 5770203 completed 200000 updates, exit0:0. Binding:
`/storage/ice-shared/vip-vwt/scratch-fwu91/poe_deep_long_10b/depth7_b32768_u200k_seed0/encoder/checkpoints/latest.pt`.
SHA256: `65ec86f01deab9955598ac8ea68406c49572184baa17f519579e1fc3cbe07839`.
This is the final checkpoint, not a result-selected best checkpoint.
Existing PoE tracker jobs 5770204/5770205 are unchanged.

This isolates encoder binding against the historical combo-10B recipe,
not pretraining budget or parameterization alone. Against the existing PoE
tracker it changes the combo settings together. Single seed; no results yet.
Local qualification artifacts: `logs/poe_combo_smoke/`.

## Submitted 2026-09-13 15:18 UTC

Jobs: **5771844 -> 5771845** (afterany continuation).
Plan: `poe-combo-10b-poe_combo-s0-20260913-151719-67826e66`.
Plan SHA: `67826e668e5570a5b57077337cf16f1a0b25e51db5fc8770398b30aecf94b00a`.
Source archive: `3510ae7989a293fb8cc6a7d21d94e9edc8ead8030529128180d6c34fc959cfe2`.

Qualification passed: exact resolved combo-10B training-argument parity
apart from run identity and encoder binding; local/remote encoder SHA256
match; identity/pair z64 checkpoint metadata; one tracker update using
that completed checkpoint at 32 environments (768 frames), clean exit.
Runtime smoke configuration confirms weight_decay=0.01 and linear critic
schedule. All nine preflight checks passed. No algorithm changes were made;
existing uncommitted runtime sources are captured in the submitted archive.

## Canceled by user request

2026-09-13T16:51:39.785477+00:00: canceled jobs 5771844 and 5771845 through the cluster control plane. User judged performance unsatisfactory. Existing artifacts and the source encoder were retained; other campaign jobs were not changed. This is not a completed 10B result.
