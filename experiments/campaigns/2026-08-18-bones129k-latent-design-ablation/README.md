# 2026-08-18 — BONES-SEED latent-design ablation

This campaign implements the low-level method ablation that we fixed in the
paper discussion: a 2 x 2 comparison of the skill objective and the latent
bottleneck. The primary table is the result. Training curves are secondary.

Status, last verified 2026-08-18: prepared and locally qualified. Both new
reconstruction arms completed one offline Isaac/Newton update and one
128-frame frozen-encoder IPMD iteration. All four seed-0 control-plane plans
resolve offline. No ICE job has been submitted, and there is no experiment
result yet.

## Question

Does the spectral DiffSR objective help because it learns transition-relevant
features, and does a discrete finite scalar quantization (FSQ) bottleneck help
or hurt after the command cadence is fixed?

The old autoencoder runs cannot answer this question. They trained the encoder
online with the tracker, used a full-body future input, used another bottleneck,
and stopped before the paper budget. This campaign adds an offline
`reconstruction` objective to the same skill-encoder trainer that DiffSR uses.
Both objectives now use the same pretrain-then-freeze path.

## Matrix

| arm | objective | bottleneck | command |
|---|---|---|---|
| `spectral_cont256` | endpoint DiffSR | continuous 256-D | 256 values + 2 phase values |
| `spectral_fsq64` | endpoint DiffSR | FSQ 64 x 32 | 64 values + 2 phase values |
| `recon_cont256` | input-window reconstruction | continuous 256-D | 256 values + 2 phase values |
| `recon_fsq64` | input-window reconstruction | FSQ 64 x 32 | 64 values + 2 phase values |

“Reconstruction” means mean squared error on the exact 380-value encoder input:
the current 38-value `root_qpos` frame plus nine intermediate frames. The held
endpoint stays hidden from every encoder. The reconstruction decoder is used
only during offline pretraining. The low-level tracker receives the frozen
encoder output, as it does for DiffSR.

The pairwise comparisons are:

- `spectral_cont256` against `recon_cont256`: objective effect with a continuous
  bottleneck.
- `spectral_fsq64` against `recon_fsq64`: objective effect with an FSQ
  bottleneck.
- The two continuous-to-FSQ differences: bottleneck effect and objective x
  bottleneck interaction.

## Fixed contract

All four arms use:

- BONES-SEED 129,785 reference motions with persist ID
  `bones_seed_sonic_full_129785@e714bbff`.
- `Isaac-Imitation-G1-v2` and the 38-value `root_qpos` macro frame.
- Macro stride 1 and `robot_heading` anchors.
- Horizon 10, intermediate encoder window, command hold 10, and `sin_cos`
  phase.
- Encoder trunk `2048, 1024, 512, 512`, SiLU, without LayerNorm.
- 50,000 offline encoder updates with batch size 8,192.
- A frozen encoder during low-level training.
- 16,384 environments, rollout length 24, gamma 0.97, and the tuned low-level
  reward, reset, and termination contract.
- `reference_prefetch_mode=next`. Do not replace it with `next_and_reset`; that
  changes the exact reset distribution.
- A 10B global environment-frame target. Each of four ICE walltime segments
  receives the full target and resumes through `cumulative_env_frames`.

The proposed Weights & Biases group is `latent-design-ablation`. Confirm this
name before any submit action. The campaign does not use retired W&B shared
mode.

## Qualification and run order

1. Run the pure-Python reconstruction tests and resolve all four campaign arms.
   **Passed 2026-08-18.**
2. Run a short local Isaac smoke for both reconstruction arms. Check that the
   reconstruction loss is finite, the FSQ encoder uses more than one level, and
   the low-level runner loads the frozen checkpoint. **Passed 2026-08-18:**
   both pretrain losses were finite, FSQ used multiple levels, and both command
   widths completed one 128-frame low-level iteration.
3. Plan seed 0 for all four arms. Inspect the frozen command and `PLAN_SHA`.
   **Offline plan validation passed 2026-08-18; no plan was submitted.**
4. Submit seed 0 only after the W&B group is confirmed.
5. Run the frozen 4,096-motion SONIC board on complete 10B checkpoints.
6. Run seeds 1 and 2 with the same contract. A one-seed row is preliminary.

From the repository root, these commands validate and freeze plans. They do not
submit jobs:

```bash
for arm in spectral_cont256 spectral_fsq64 recon_cont256 recon_fsq64; do
    ./experiments/campaigns/2026-08-18-bones129k-latent-design-ablation/submit.sh "${arm}" 0
done
```

The control plane prints a separate `submit --plan ... --confirm <PLAN_SHA>`
command for each plan. Do not run those commands until the frozen commands and
the W&B group are approved.

## Evaluation and reporting

Use `eval_scoreboard4096.sh` after the ICE checkpoints are mirrored to
`logs/bones_latent_ablation_mirror/<arm>_seed<seed>/`. The script keeps the
frozen paper board: ranks 12288-16383, frame-0 starts, mode actions, no push,
released SONIC termination thresholds, and Newton/MJWarp.

Primary metrics:

- SONIC success rate.
- Success-only root-relative mean per-joint position error, MPJPE-L, in mm.

Also report the true environment frames for every row. The explicit
`root_qpos` row and released SONIC are reference rows, not ablation arms. The
explicit row remains a 7.60B-frame ceiling and is not frame-matched to these
10B arms. Do not describe a comparison with it as a win.

Use this table shape:

| objective | bottleneck | seed | frames | success rate | success-only MPJPE-L | status |
|---|---|---:|---:|---:|---:|---|
| endpoint DiffSR | continuous 256-D | 0 | — | — | — | pending |
| endpoint DiffSR | FSQ 64 x 32 | 0 | — | — | — | pending |
| reconstruction | continuous 256-D | 0 | — | — | — | pending |
| reconstruction | FSQ 64 x 32 | 0 | — | — | — | pending |
| explicit `root_qpos` reference | none | 0 | 7.60B | reference | reference | frames not matched |
| released SONIC reference | released tokenizer | — | — | reference | reference | external recipe |

Treat a relative difference below about 15% in the high-error regime as
unresolved unless repeated seeds support it. Do not select one metric after the
result is known. Report success rate and success-only MPJPE-L together, and
inspect per-termination counts when they disagree.
