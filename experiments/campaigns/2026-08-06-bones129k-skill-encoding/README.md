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

## Result (2026-08-07): no factorization beat plain endpoint prediction

All three controllers reached 5.00B frames and COMPLETED (`5570359`, `5570368`,
`5570370`, ~11 h each). Scored on the workstation with
[`eval_mpjpe.sh`](eval_mpjpe.sh) at 512 environments x 1000 steps, seed 0,
frame-0 starts, termination curriculum off, both mandated passes. Raw summaries
in `logs/skill_encoding_mpjpe_eval/`.

Strict pass (the four SONIC window terminations active):

| arm | success | survival | MPJPE mm | MPJPE succ mm | joint RMSE | EE local m |
|---|---|---|---|---|---|---|
| endpoint (control) | 0.697 | 276.3 | 21.29 | 20.13 | 0.1478 | 0.0347 |
| `state_occupancy` | 0.650 | 264.0 | 21.83 | 20.48 | 0.1453 | 0.0358 |
| `semimarkov_chain` | 0.688 | 275.0 | 22.23 | 21.06 | 0.1483 | 0.0367 |
| `endpoint_delta` | 0.619 | 258.2 | 30.17 | 28.93 | 0.1862 | 0.0502 |

Full-horizon diagnostic (every early termination off):

| arm | MPJPE mm | MPJPE succ mm | joint RMSE | EE local m |
|---|---|---|---|---|
| endpoint (control) | 27.92 | 28.86 | 0.1599 | 0.0444 |
| `state_occupancy` | 30.83 | 31.64 | 0.1630 | 0.0499 |
| `semimarkov_chain` | 33.04 | 34.30 | 0.1665 | 0.0536 |
| `endpoint_delta` | 41.03 | 42.04 | 0.2085 | 0.0676 |

The ordering is monotone and identical across both passes and every metric:
control < `state_occupancy` < `semimarkov_chain` < `endpoint_delta`, i.e. MPJPE
degrades monotonically with how much intra-chunk structure the objective forces
`z` to carry. Full-horizon MPJPE-G is +10.4% / +18.3% / +47.0% over the control.

**Pretrain quality did not predict controller quality, and reward did not
predict MPJPE.** `endpoint_delta` has the best reconstruction (0.0087) and a
214x real-vs-shuffled separation, and the worst controller.
`semimarkov_chain` has 13x better reconstruction than `state_occupancy` and a
nominally higher training `r_step` than the control (0.2131 vs 0.2117), yet is
18% worse on full-horizon MPJPE. Score this axis on the evaluation metric, not
on pretrain diagnostics or training reward.

Two limits on the strength of this result. The control's only surviving
checkpoint is 7.55B against the arms' 5.00B (its 5B save was removed in the
2026-08-07 ICE quota prune), so its margin is an upper bound; the three arms are
frame-matched to each other and their ordering is unaffected. Each arm is one
seed, on a stack with ~12% evaluation spread, so the 10.4% control-vs-occupancy
gap is inside noise while the 18.3% and 47.0% gaps are not.

Full-horizon survival is identical across arms (347.7 steps, 0.973) because with
terminations disabled an episode ends only on `reference_finished` or
`time_out`, which the reference clip lengths decide rather than the policy. That
is a protocol match check, not a tie between arms.

Still untested: every arm here conditions the transition on the single frame
`s[t]`. Conditioning on a past chunk is a separate axis this campaign does not
touch.
