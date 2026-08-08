# 2026-08-07 — BONES-129k latent mode at SONIC's stride-5 window

Three latent bottlenecks, one scaled recipe, on the 129,785-clip BONES-SEED set.
Each arm is a DiffSR encoder pretrain plus an `afterok`-dependent 5B-frame
low-level controller on ICE H200.

## What is being compared

The only difference between arms is the skill encoder's latent bottleneck
(`--latent_mode`). Everything else — macro interface, window, DiffSR geometry,
tracker/critic capacity, rewards, resets, PPO geometry, frame budget, seed — is
identical by construction.

| arm | `--latent_mode` | bottleneck | actor command |
|---|---|---|---|
| `det64` | `deterministic` | continuous 64-D code, no quantizer | 64 + 2 phase = 66 |
| `fsq64` | `sonic_fsq` | FSQ, 64 coordinates x 32 levels; the quantizer output **is** the command (no learned projection at the boundary) | 66 |
| `gumbel64` | `gumbel_multicat` | product codebook, 8 groups x 32 categories, per-group Gumbel-softmax, temperature annealed 2.0 -> 0.5 over 10,000 of 50,000 updates | 66 |

`gumbel64`'s anneal length is a deliberate departure from the
`gumbel_tau_anneal_iters=2000` default, which would finish annealing in the
first 4% of the pretrain budget.

## The stride-5 delta

This campaign's other change against
`2026-08-06-bones129k-sonic-fsq-scale` is `env.expert_macro_frame_stride=5`.

The DiffSR macro window keeps its 10 slots, but they are now spaced 5 reference
frames apart instead of 1. At 50 Hz that is SONIC's released tokenizer cadence
(`dt_future_ref_frames=0.1`): the encoder compresses **0.9 s** of future
reference motion instead of 0.18 s, and the endpoint objective's target moves
from `s[t+10]` to `s[t+50]` -- one stride past the window the encoder reads.
The command is still held 10 control steps (0.2 s),
so the window is long relative to the hold — the same shape SONIC uses.

**The width does not change.** A 380-wide root_qpos macro state is 380 wide at
every stride, so a stride-1 encoder paired with a stride-5 environment produces
no shape error and no warning, only a silently off-distribution command. Two
things prevent that:

1. `env.expert_macro_frame_stride` is recorded into the skill checkpoint's
   `config` at pretrain time (there is deliberately no separate CLI flag — two
   sources for one value is how they drift apart).
2. The low level compares the checkpoint's stride against the live environment's
   and refuses the pairing, the same way it already refuses a `horizon_steps`
   mismatch.

The local smoke asserts (1) explicitly and exercises (2) end to end.

## Fixed contract

- Task `Isaac-Imitation-G1-v2`, Newton MJWarp, ICE `ice-gpu` / `coe-ice`, H200.
- Macro state: `root_qpos` (`expert_motion_qpos` + anchor pos + anchor ori),
  38/frame, 380 over the 10-slot window.
- Encoder MLP `[2048, 1024, 512, 512]` SiLU, no LayerNorm; DiffSR feature 256,
  embed 1024, `g`/`mu` `[1024, 1024, 512]`; `intermediate` window mode,
  `endpoint` objective; 50,000 updates at batch 8192.
- Tracker and critic `[2048, 2048, 1024, 1024, 512, 512]` SiLU.
- 16,384 envs, rollout 24, gamma 0.97, 5B frames, checkpoint every 250M,
  `random80_adaptive20` resets, termination curriculum 5M -> 30M.
- Reference arrays `g1_bones_seed_sonic_full_129785_e714bbff_v1`
  (129,785 trajectories, 47,491,234 frames), resident.
- Seed 0.

## Running it

Dry run:

```bash
MODE=print ./experiments/campaigns/2026-08-07-bones129k-latent-mode-stride5/run.sh
```

Local smoke — one pretrain update plus one low-level iteration per arm, on the
workstation. Required before `validate` or `submit`; the marker records the
source-contract hash and the arms covered, and a later stage refuses a stale or
partial one:

```bash
MODE=smoke ./experiments/campaigns/2026-08-07-bones129k-latent-mode-stride5/run.sh
```

Gate check without touching the scheduler (remote reference identity plus fresh
output paths):

```bash
MODE=validate LOCAL_SMOKE_ROOT=<smoke-dir> ./experiments/campaigns/2026-08-07-bones129k-latent-mode-stride5/run.sh
```

Submit all six jobs (3 pretrain, 3 dependent controllers):

```bash
MODE=submit LOCAL_SMOKE_ROOT=<smoke-dir> CONFIRM_SUBMIT=latent-mode-stride5 ./experiments/campaigns/2026-08-07-bones129k-latent-mode-stride5/run.sh
```

Submission writes `cluster_submission.json` with the source-contract hash, all
six Slurm job IDs, and the full shared contract. It refuses to overwrite an
existing record, and `validate`/`submit` refuse an output path that already
exists on ICE.

`LATENT_ARMS` selects a subset of arms for any mode.

## Status

Submitted 2026-08-07 to ICE. W&B: project `g1-bones-seed`, group
`latent-mode-stride5`. Full record in `cluster_submission.json`.

| arm | pretrain | controller |
|---|---|---|
| `det64` | 5571878 | 5571879 (`afterok:5571878`) |
| `fsq64` | 5571880 | 5571881 (`afterok:5571880`) |
| `gumbel64` | 5571882 | 5571883 (`afterok:5571882`) |

Gates cleared before submission: local smoke over all three arms at contract
`58ec1874dd9d0ca5…`, the recorded stride asserted as 5 in each pretrain
checkpoint, remote reference identity (129,785 / 47,491,234 /
`bones_seed_sonic_full_129785@e714bbff`), and six fresh output paths.

The stride guard was also checked negatively on the workstation: a low level
pointed at a stride-5 encoder with `env.expert_macro_frame_stride=1` refuses to
start with `Skill encoder macro-window stride does not match the environment`.
