# 2026-08-02 — v2 command interface, LAFAN1 det-SR 5B

First cluster run on the declared command interface (repo commit `874ef0c`).
The recipe is unchanged from the established latent protocol; what is new is
the environment surface it runs on.

## What this run is

| | |
|---|---|
| task | `Isaac-Imitation-G1-v2` |
| dataset | corrected LAFAN1, 40 motions, manifest `d972c37c…` |
| latent | deterministic SR, continuous, z=256, encoder horizon 10 → published width 258 |
| geometry | 12,288 envs × 12 steps = 147,456 frames/iteration |
| budget | 5B frames |
| W&B | project `g1-lafan1`, group `v2`, tags `sr,det,v2,lafan1,<stage>` |

The command surface is declared once, on
`env.command_interface`: an always-present dataset-backed `reference` channel
plus a single agent-published `actor` channel (`LatentCommandCfg`, dim 258).
The training entry point binds the agent config to that interface, so the
actor/critic/encoder input keys are derived rather than restated — the
`env.latent_command_dim` / `env.command_mode` knobs the older launchers pass no
longer exist.

## Stages

Submitted separately: the ICE wall is ~16 h, and a combined job that dies
mid-pretrain loses both stages.

```bash
# 1. fresh deterministic-SR encoder -> /data/pretrain_store/<tag>/checkpoints/latest.pt
DRY_RUN=0 STAGE=pretrain ./submit_v2_det_sr_lafan1_5b_ice.sh

# 2. low-level IPMD on the frozen encoder (gated on it existing)
DRY_RUN=0 STAGE=lowlevel ./submit_v2_det_sr_lafan1_5b_ice.sh

# 3. second segment, to top up to the 5B cap
DRY_RUN=0 STAGE=lowlevel \
  COMPLETED_FRAMES=4258971648 \
  TRAIN_CHECKPOINT=/data/v2_command_interface/<tag>/rlopt_train/.../latest.pt \
  ./submit_v2_det_sr_lafan1_5b_ice.sh
```

`DRY_RUN=1` (the default) prints the plan and the exact command without
contacting the cluster.

## Why two low-level segments

At the measured latent-arm rate for this geometry (~76k fps) a 15:59 wall fits
28,883 iterations ≈ 4.26B frames, so 5B needs one full segment plus a 5,026-
iteration top-up. The submitter computes this from `COMPLETED_FRAMES` and caps
each segment itself.

Checkpoints are written to the `/data` bind, never the per-submission
workspace: an ICE `TIMEOUT` is a hard SIGKILL that wipes node-local output
before any log sync runs (this is how the 5525664 policy checkpoints were
lost).

## Status

| stage | job | state |
|---|---|---|
| pretrain | `5558033` | submitted 2026-08-02 |
| lowlevel seg 1 | — | blocked on the encoder |
| lowlevel seg 2 | — | blocked on seg 1 |

## Gates the submitter enforces

- corrected-LAFAN1 manifest sha256 and NPZ count on the cluster (refuses to
  submit on a mismatch),
- for `STAGE=lowlevel`, the encoder checkpoint exists and is >1 MB,
- `env.refresh_zarr_dataset=false` — the `/data` cache is shared with other
  LAFAN1 arms and a refresh rebuilds it underneath them.
