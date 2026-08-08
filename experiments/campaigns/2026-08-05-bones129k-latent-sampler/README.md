# BONES-129k reset/scheme comparison (ICE H200)

Five 10B-frame-cap jobs share the tuned v2 controller environment, BONES-SEED
129,785 reference arrays, Newton, 16,384 environments x 24 rollout steps,
minibatch 294,912, gamma 0.97, seed 0, and the new
`random80_adaptive20` reset sampler. This matches the production rollout-24
run supplied for the comparison; the jobs differ in how the policy command is
constructed.

ICE caps each allocation at 16 hours. A slower online-latent arm may time out
before reaching 10B; checkpoints are saved every 50M frames under persistent
`/data`, so reaching the cap can require a continuation allocation.

| arm | command contract |
|---|---|
| `reset80_diffsr` | existing frozen root-qpos DiffSR z256 + phase, h10, hold10 |
| `sonic_fsq32` | SONIC-style online FSQ, consecutive future10 (0.18 s), joint qpos/qvel + anchor-pose input, hold1 |
| `sonic_fsq32_v2` | SONIC-release-aligned online FSQ: 10 future frames spaced 0.1 s (0.9 s span), 14-body keypoint positions + 6D root-ori input, hold1 |
| `vqvae_k32` | reconstruction EMA VQ-VAE, future10, hold10, one K=32/64-D codebook + phase |
| `autoencoder` | continuous reconstruction autoencoder, future10, hold10, 64-D code + phase |
| `root_qpos_explicit` | direct 38-D qpos + root pose, single frame, renewed every step |

The official SONIC config says `max_num_tokens=2`, `fsq_level_list=32`, so its
token is 64 independent scalar coordinates with 32 levels each. The VQ arm's
literal `K=32` matches that per-codebook cardinality, but **not** SONIC's
`32^64` product capacity: this repo currently has one flat EMA-VQ codebook and
cannot instantiate an equivalent product table. Results must retain that
qualification.

The 2026-08-07 SONIC-fidelity audit (arXiv 2511.07820 Table 3 plus the
released `gear_sonic` configs, especially
`exp/manager/universal_token/all_modes/sonic_bones_seed.yaml`) confirmed that
`sonic_fsq32` already matches SONIC on the training objective — hold=1
re-encode each control step, policy gradient into the encoder, reconstruction
MSE at SONIC's released coefficient 0.01, FSQ 64x32, encoder MLP
[2048, 1024, 512, 512] SiLU, no phase channel — but not on the encoder input:
SONIC's window is `num_future_frames=10` at `dt_future_ref_frames=0.1`
(offsets 0, 5, ..., 45 frames; a 0.9 s span), and its input features are the
14-body reference keypoint positions in the robot's local frame plus the 6D
root-orientation difference (`command_multi_future_nonflat` +
`motion_anchor_ori_b_mf_nonflat`, "noz"). `sonic_fsq32_v2` closes both gaps
via the `future10_stride5` encoder view and the `[keypoint_pos, root_ori]`
component selection over SONIC's own 14-body list. Remaining known
structural deltas, retained deliberately: the surrounding controller recipe
(rewards, resets, PPO geometry and learning rates) is this campaign's common
contract rather than SONIC's, and the reconstruction term steps a dedicated
Adam(2e-5) while SONIC sums one weighted loss into its single actor
optimizer.

Run a one-update smoke for all arms before any remote validation:

```bash
MODE=smoke experiments/campaigns/2026-08-05-bones129k-latent-sampler/run.sh
```

Then use the printed smoke root for the guarded remote steps:

```bash
MODE=print experiments/campaigns/2026-08-05-bones129k-latent-sampler/run.sh

MODE=validate LOCAL_SMOKE_ROOT=/absolute/path/from-smoke \
  experiments/campaigns/2026-08-05-bones129k-latent-sampler/run.sh

MODE=submit CONFIRM_SUBMIT=bones129k-latent-sampler \
  LOCAL_SMOKE_ROOT=/absolute/path/from-smoke \
  experiments/campaigns/2026-08-05-bones129k-latent-sampler/run.sh
```

The W&B destination is the `g1-bones-seed` project and
`bones129k-ablation` group, matching the supplied rollout-24 run.

The first five-arm submission is recorded in `cluster_submission.json`: ICE
jobs `5567801`, `5567802`, `5567803`, `5567804`, and `5567809`, respectively
in the table order above (without `sonic_fsq32_v2`). All jobs use the same
verified workspace archive and retain their arm-specific feature tags in W&B.
The SONIC-release-aligned `sonic_fsq32_v2` arm was submitted separately on
2026-08-07 as ICE job `5571455` after the fidelity audit above; its own
source-contract hash and smoke root are appended in
`cluster_submission.json`.
