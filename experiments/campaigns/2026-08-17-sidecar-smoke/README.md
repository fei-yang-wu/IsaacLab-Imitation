# 2026-08-17 — Sidecar smoke (plumbing, not an experiment)

One short ICE run whose only purpose is to prove three pieces of plumbing work
together on real hardware. **Nothing here is a result.** 512 environments and a
~30M-frame budget are far below any regime that produces a tracking number, and
the encoder gets 400 pretrain updates. Do not cite anything this run produces.

## What it proves

1. **The workspace archive fix.** `submit` packs ~21 MB (was 3.2 GB) and
   publishes it to the content-addressed store
   `<control_root>/workspaces/<sha>.tar.gz`; the plan directory holds a symlink
   into it, and a second submit of an unchanged tree reuses the stored copy.
2. **Checkpoint cadence and provenance.** `agent.save_interval=5000000` writes a
   tracker checkpoint every 5M frames, each carrying `cumulative_env_frames` —
   the x-axis every sidecar point is plotted against.
3. **The eval sidecar publishing to W&B during training.** Each new checkpoint
   becomes one EC/MuJoCo rehearsal point (`ec_latent_rehearsal_v1`, board
   `selected10_repeats5_v1`, sensor noise on, sync lockstep) in a companion W&B
   run while the training job is still going.

## Running it

```bash
./submit.sh 0                 # plan: preflight + freeze, prints PLAN_SHA
# then the printed: ... submit --plan <dir> --confirm <PLAN_SHA>
./watch_sidecar.sh 0          # workstation: mirror checkpoints + score + log
```

## Why the sidecar runs on the workstation

The sidecar needs the `onnx-export` Pixi environment (bundle export) and
Embodied-Control's `lowlevel-sim` environment (MuJoCo). Neither exists inside
the cluster container image, which carries only the `container-runtime`
environment for Isaac Sim training. The sidecar is CPU-only and one board takes
about a minute, so running it on the workstation against a mirrored checkpoint
tree costs the training job nothing. `watch_sidecar.sh` rsyncs
`model_step_*.pt` from the ICE run directory on an interval and points
`ec_tracker_sidecar scan --watch` at the mirror; the mirror keeps the arm tree's
shape so the sidecar finds `encoder/checkpoints/latest.pt` by itself.

Running the sidecar on the compute node would need those two environments built
on the cluster — worth doing only if this pattern proves useful enough to keep.

## W&B

Project `g1-bones-seed`, group `sidecar-smoke`:

- `fsq64-hold10-pretrain-s0` — encoder pretrain
- `fsq64-hold10-s0` — tracker training
- `fsq64_hold10-s0-ec-sidecar` — the sidecar's points, `sidecar/mpjpe_l_mm`,
  `sidecar/fall_free_rate`, and friends against `cumulative_env_frames`

The sidecar writes a **separate** run rather than logging into the trainer's:
two processes writing one run id race, and the sidecar outlives any single
chained segment. Same project and group, so the curves overlay in one
workspace.
