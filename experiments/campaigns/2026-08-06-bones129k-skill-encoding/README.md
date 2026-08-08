# BONES-129k skill-transition factorization ablation

This campaign changes what the frozen 256-D DiffSR skill code must explain.
Everything downstream stays matched to the current BONES-129k controller:
`Isaac-Imitation-G1-v2`, root-qpos encoder input, 10-frame skill window,
10-step held code plus sin/cos phase, 16,384 environments x 24 rollout steps,
minibatch 294,912, gamma 0.97, Newton/MJWarp, tuned rewards, and
`random80_adaptive20` reset sampling.

## Status (2026-08-06)

All three one-update Isaac smokes and the ICE data/output gates passed against
source contract `f8db5faa403aa4f7`. The H200 jobs are submitted under W&B
project `g1-bones-seed`, group `skill-encoding-ablation`:

| arm | encoder pretrain | dependent 5B controller |
|---|---:|---:|
| `state_occupancy` | `5570344` | `5570359` |
| `semimarkov_chain` | `5570351` | `5570368` |
| `endpoint_delta` | `5570358` | `5570370` |

Each controller has an arm-specific `afterok` dependency and cannot begin
until its 50,000-update encoder pretrain succeeds. Exact provenance is in
`cluster_submission.json`.

The existing `reset80_diffsr` run from ICE job `5567801` is the endpoint
control:

```text
z = E(s_t, s_{t+1:t+9})
p(s_{t+10} | s_t, z)
```

The three new arms are:

| arm | training factorization | intended pressure on z |
|---|---|---|
| `state_occupancy` | sample `h_k` uniformly from `{2,4,6,8,10}` and learn `p(s_{t+h_k} | s_t,z)` without giving `h_k` to the decoder | represent the option's visited-state distribution, not only its endpoint |
| `semimarkov_chain` | sample one factor from `p(s_{t+h_k} | s_{t+h_{k-1}},z)` over checkpoints `{0,2,4,6,8,10}` | make one held code explain intra-option dynamics throughout the chunk |
| `endpoint_delta` | learn `p(s_{t+10}-s_t | s_t,z)` | represent relative effect while discarding absolute-pose reconstruction pressure |

For the two multi-checkpoint objectives, one factor is sampled independently
per training row. This is an unbiased estimator of the uniform sum and keeps
the encoder batch at 8,192 instead of multiplying DiffSR memory by five.
Evaluation scores every checkpoint and reports both per-offset and uniform-mean
losses.

These are fixed-duration option models, not full Option-Critic. They do not yet
learn an initiation set or termination function. The next credible extension,
if one factorization wins, is a duration-conditioned kernel
`p(s_{t+tau}, tau | s_t,z)` plus a learned termination hazard. Adding that now
would confound representation and controller execution because the qualified
tracker consumes exactly ten slots per code.

Run local one-update smokes first:

```bash
MODE=smoke experiments/campaigns/2026-08-06-bones129k-skill-encoding/run.sh
```

Then validate ICE inputs and output paths:

```bash
MODE=validate LOCAL_SMOKE_ROOT=/absolute/path/from-smoke \
  experiments/campaigns/2026-08-06-bones129k-skill-encoding/run.sh
```

Submission first creates all three encoder-pretrain jobs, then one dependent
5B-frame low-level job per arm. W&B project is `g1-bones-seed`; group defaults
to `skill-encoding-ablation` and requires the matching confirmation token:

```bash
MODE=submit CONFIRM_SUBMIT=skill-encoding-ablation \
  LOCAL_SMOKE_ROOT=/absolute/path/from-smoke \
  experiments/campaigns/2026-08-06-bones129k-skill-encoding/run.sh
```

ICE scratch was at 90.3% before this campaign. Low-level checkpoints therefore
save every 250M frames under persistent `/data`, bounding timeout loss while
avoiding the 50M cadence that made the preceding five-arm campaign consume
33 GB. Do not delete old checkpoints without explicit approval.
