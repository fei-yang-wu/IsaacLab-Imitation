# 2026-08-03 — Tuned 5B LAFAN1 low level on the PhysX backend

One run: the tuned v2 det-SR latent recipe, 5B frames on corrected LAFAN1, with
`physics=physx` instead of `newton_mjwarp`. Everything else is held to the
Newton tuned 5B run so the backend is the only free variable.

Launcher: `submit_tuned_5b_physx_ice.sh` (DRY_RUN=1 by default).

## Why

`wiki/sim2sim-backend-verification.md` (2026-08-03) closed the cross-backend
investigation:

- The joint order is clean. Two independent tools agree there is no residual
  index leak, so no remapping refactor would move any number.
- On a policy-free oracle probe — no checkpoint involved — **PhysX tracks the
  reference 3x better than Newton**: joint MAE 0.0327 vs 0.0975 rad, applied
  torque 1.49 vs 7.15 Nm, zero vs -1.605/s of `joint_limit` penalty.
- Stock MuJoCo, brought in as the referee precisely because `newton_mjwarp`
  *is* MuJoCo Warp, lands on PhysX's numbers rather than Newton's on exactly the
  joints where Newton is 14-31x off.
- The MJWarp model Isaac hands the solver is parameter-exact, and both the
  contact budget and the contact pipeline reproduce every metric bit-identically.

So Isaac Lab's Newton path is the outlier, and every 5B checkpoint to date is
overfit to it — the tuned 5B latent checkpoint goes from 19.9 mm MPJPE on Newton
to 334.5 mm on PhysX over 300 steps. This run retrains the recipe on the backend
that agrees with the referee.

## The configuration

| | |
|---|---|
| task | `Isaac-Imitation-G1-v2` |
| optimizer recipe | `--agent rlopt_ipmd_tuned_cfg_entry_point` (`G1ImitationTunedRLOptIPMDConfig`) |
| environment recipe | `action_rate_l2=0`, `tracking_reward_points=4.0`, termination curriculum 5M→30M |
| geometry | 12288 envs × 6 rollout steps = 73,728 frames/iter, minibatch 18,432 |
| data | `/data/lafan1_corrected_8e95d557`, manifest sha `d972c37c…`, 40 NPZ |
| encoder | `/data/pretrain_store/lafan1_v2_det_sr_h10_z256_seed0`, frozen, reused |
| budget | 5B frames, seed 0, segmented under a 15:59 wall |
| W&B | `g1-lafan1`, group `physx-5b` |

Two things the launcher does differently from
`2026-08-02-rlopt-hp-search/submit_tuned_5b_ice.sh`, both deliberate:

**It selects the recipe by entry point, not by a copied override list.** The
tuned config is registered, so a launcher that copies its fields can drift from
it silently. The environment half genuinely does not live on the agent config
and is still passed explicitly.

**Rollout 6, not 12.** That is what the registered recipe sets, and it was
measured at +7.3% return and +6.8% episode length at unchanged MPJPE against a
byte-identical config (arms `s1` vs `r0`). The class docstring still carries a
stale "geometry unchanged" bullet; the code below it sets 6.

The encoder is reused rather than re-pretrained. `train_hl_skill_diffsr.py` is
an offline entrypoint that fits DiffSR on the dataset cache and cannot start Kit
at all, and the post-2026-07-21 planner frame is byte-identical across backends,
so the encoder carries no backend dependence. Reusing it is also what keeps this
a single-variable comparison.

## The GPU, and the throughput it turned out to have

`runtime_bootstrap.validate_gpu_policy` rejects PhysX/Kit on compute-only parts
(A100/H100/H200); ICE's PhysX-qualified GPUs are L40S, A40 and RTX6000. The
launcher defaults to H100 plus the documented headless escape hatch
`--experimental-compute-only-physx`, which had never been run on this cluster.

**It works.** Job 5560022 logged `PhysX GPU policy accepted: NVIDIA H100 80GB
HBM3` and `Experimental headless compute-only PhysX override enabled`, started
Kit in ~5 min, built the 12288-env PhysX scene and trained. The qualified
fallback is still one knob if a later node refuses:

```bash
GPU_GRES=gpu:l40s:1 DRY_RUN=0 ./submit_tuned_5b_physx_ice.sh
```

**Throughput: 33,421 fps** at iteration 136, against Newton's 62,406 fps at the
same task and data — a ratio of 0.54, close to the 0.63x seen in the 2026-07
matched 4096 × 24 pair. PhysX costs roughly 1.9x the wall-clock per frame here.

`SEGMENT_FPS` shipped at a deliberately low 17,000, because an ICE TIMEOUT is a
hard SIGKILL that runs no final save: undersizing a segment costs one extra
submission, oversizing costs everything since the last save. Segment 1 is
therefore about half the size the wall would allow and exits cleanly at ~7.8 h.
Use `SEGMENT_FPS=30000` from segment 2 on (~1.66B frames per segment), which
still keeps margin under the measured rate. `SAVE_INTERVAL` is halved to 50M
against the Newton launcher and can stay there.

At 30,000 fps, 5B takes four segments in total.

## Resuming

```bash
DRY_RUN=0 COMPLETED_FRAMES=<frames so far> \
    TRAIN_CHECKPOINT=/data/physx_5b/<run_tag>/rlopt_train/.../latest.pt \
    ./submit_tuned_5b_physx_ice.sh
```

## Reading the results

`episode/return` is **not** comparable to any run that does not double
`tracking_reward_points`. Judge on MPJPE and episode length, which no reward
weight can inflate.

## Status

| stage | state |
|---|---|
| launcher + dry run | done |
| segment 1 | ICE job **5560022**, running, 12,783 iters / 942M frames |
| segments 2-4 | pending segment 1; resubmit with `SEGMENT_FPS=30000` |
