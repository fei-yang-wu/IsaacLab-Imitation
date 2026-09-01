# 2026-09-01 — does a 64-D merged latent train under the 50B regime? 10B

One arm, one job, seed 0. W&B project `g1-bs-pareto`, group
`latent64-probe-10b`, run id `l64p-z64m-s0`.

| arm | encoder | head | width | command | budget |
|---|---|---|---|---:|---:|
| `z64_merged` | `/data/latent_star_v2/hub_seed0/encoder` (reused) | merged | 64 | 66 | 10B |

## The reference run

`diffntp_chunk_50b` in
[`../2026-08-31-diffntp-chunk-50b-fullbatch/`](../2026-08-31-diffntp-chunk-50b-fullbatch/README.md),
W&B run `diffntp-chunk-50b-s0-2026-09-01_08-48-21`, jobs 5601421-5601431. Its
`std1`-`std3` stages completed 10B under `random80_adaptive20` with the
5M-30M termination curriculum. This arm copies that stage field for field:
`rlopt_ipmd_tuned_fullbatch_cfg_entry_point`, 20,480 environments x 24 rollout
steps, `motion_ee_pos` 1.0, `motion_global_anchor_pos_wide` 1.0,
`action_rate_l2` -0.03, `tracking_reward_points` 4.0,
`reference_prefetch_mode=next`, no proprio history, no weight decay, no critic
schedule, checkpoints every 500M.

## What moves, and what that costs

Two fields, by user decision on 2026-09-01: the latent width 256 -> 64, and
the pretrain head. The reference binds `diffntp_chunk_h1_ee_wide`, the
two-head form; this arm binds the star-v2 hub, the merged single head
(`--jepa_ntp_chunk_span boundary_next --jepa_endpoint_coeff 0`), which is the
formulation being carried forward.

So a difference against the reference **cannot be attributed to the width**.
The clean width comparison at the merged head is already running and needs no
job here: `hub` (64-D) against `g3_cont256` (256-D) in
`2026-08-30-latent-star-v2`, same head, same pretrain recipe, 5B each. What
this arm adds is whether the merged 64-D interface trains under the 50B
chain's own tracker regime and budget — star-v2 runs the sonic 0.8 -> 0.2
failure ramp at 5B instead.

Everything else in the hub encoder's pretrain matches the reference's:
`--jepa_loss sigreg_ebm`, deterministic latent, `--encoder_layer_norm`,
`--encoder_window_mode intermediate`, `--horizon_steps 10`, 50,000 updates at
batch 8,192, the 380-value `root_qpos` macro state, stride 1, `robot_heading`
anchor. `latest.pt.json` reads `{"update": 50000}`, so that pretrain finished.

## The phase channel is inert at hold 1

`command_phase_mode=sin_cos` with `phase_source='hold'` and `code_period=1`
publishes a constant: `phase = (phase_period - latent_steps) / phase_period`
evaluates to 0 on every step (`RLOpt/rlopt/agent/hl_skill_diffsr.py`; the
config docstring calls it "Informationally dead at `code_period=1`"). It is
kept only because the reference keeps it, so the command is 66 wide against
the reference's 258 with two constant values in both.

This also means `leader64_h1_nophase`'s missing phase channel was never a
second variable against its 256-D control: that arm's stall at 0.84B — episode
length 50-62, MPJPE-L flat near 51 mm, against 166 / 42.6 mm by 0.17B — was a
width difference at the two-head form.

## Budget

One segment carrying the full 10B target (20,346 iterations); the 15:59
walltime ends it. The reference took three segments to reach 10B, so read this
arm against the reference at the **same cumulative frame count**, never against
its 10B endpoint.

## Running

```bash
python -m imitation_experiments.pipeline.cluster plan \
    --campaign experiments/campaigns/2026-09-01-latent64-probe-10b/campaign.yaml \
    --arm z64_merged --seed 0
python -m imitation_experiments.pipeline.cluster submit --plan-sha <PLAN_SHA>
```

Per AGENTS.md every result table carries the `sonic_v1_1` row for the same
board.

## `nophase` result (2026-09-01)

`nophase` (5606899) diverges from `z64_merged` from about 60M frames and sits
at ep_len 37-41 from 100M on, where the control is at 100-151; its trajectory
matches the cancelled `nophase_wd_clin` on every logged series, so the
optimizer extras were inert. The failure signature is the KL-adaptive
learning rate collapsing to 2.6e-5 (control 2.0e-4) under `train/kl_approx`
0.033-0.036. The code says the phase pair is the constant `(0, 1)` at hold 1
(confirmed on the 500M checkpoint normalizer statistics), so the mechanism is
open; see `wiki/current-status.md` (2026-09-01, "The 64-D hold-1 phase pair
decides training") for the three candidate routes and the seed experiment
that separates them.

## `nophase_linlr` (2026-09-01)

`nophase` with the KL-adaptive actor learning rate replaced by a linear decay
(`--agent rlopt_ipmd_tuned_fullbatch_linearlr_cfg_entry_point`, new in
`config/g1/agents/rlopt_ipmd_cfg.py`): actor lr 2e-4 down to 1e-5 over the
10B budget (61,038 scheduler steps = 3 updates x 20,346 iterations), critic
pinned at 1e-3, everything else the `nophase` batch script. The adaptive
rule is what turned the high-KL regime of `nophase` into an 8x lr cut; this
arm asks whether the no-phase interface trains once that feedback loop is
gone. Two things move against `nophase` (schedule shape, and the start value
2e-4 instead of 1e-3), so a recovery is attributable to the schedule only if
the control's own adaptive lr is accepted as the reference for the start
value.
