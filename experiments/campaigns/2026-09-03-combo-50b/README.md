# 2026-09-03 -- `combo` at 50B

The stacked 64-D arm promoted from 10B to the headline frame budget, on the
user's direction. Recipe unchanged from `2026-09-01-latent64-probe-10b` arm
`combo`; only the budget and the segment count move.

| | |
|---|---|
| encoder | past-5 **affine** phi (`p5_affine`), 64-D merged hub, hold 1, sin_cos phase |
| actor | MLP, ten-step history on the five policy terms (critic single frame) |
| optimizer | full-batch / 3-epoch entry point, `weight_decay=1e-2`, linear critic decay to 1e-5 |
| resets | `sonic`, uniform share ramped 0.8 -> 0.2 (see below), 5M-30M termination curriculum |
| rewards | motion_ee_pos 1.0, motion_global_anchor_pos_wide 1.0, tracking_reward_points 4.0, action_rate_l2 -0.03 |
| scale | 16,384 environments x 24 = 393,216 frames per batch, 127,157 iterations |

What it is promoted on, one seed, `bones_testbed4096_v1`: at 10B `combo`
read 0.9214 SR / 22.64 MPJPE-L / 88.52 MPJPE-G clean and 0.9146 / 24.84 /
124.52 robust, the best tracking rows of the 10B group (control 0.9292 /
23.25 / 93.09 clean at 9.5B, `lstm` 0.9121 / 21.24 / 103.11, `lstm_affine`
0.9062 / 22.27 / 110.63). It stacks four levers at once, so no row here
attributes to one of them.

Seven chained 15:59 segments, `afterany`, each carrying the FULL 50B cap:
`_apply_critic_lr_schedule` reads `collector.total_frames`, which is the
segment's own `--max_iterations`, so a smaller cap on a later segment would
push the critic learning rate back up. At the ~150k fps this arm held on
`coe-gpu`, one segment buys about 8.6B, so six reach the cap and the seventh
is slack.

Disk: `agent.save_interval=1000000000` (1B, against the 500M of the 10B
campaigns) -- 50 checkpoints x 244 MB = 12.2 GB. The ICE home quota was at
265 GB of 300 GB at submission, so 500M would not have fit.

```bash
pixi run python -m imitation_experiments.pipeline.cluster plan \
    --campaign experiments/campaigns/2026-09-03-combo-50b/campaign.yaml --arm combo --seed 0
# then the printed submit line
```

## Reset schedule (user, 2026-09-03)

The reset mix ramps from 80% uniform / 20% failure-drawn to 20% / 80%, so
late training concentrates on the frames where the tracker terminates.

| segment | selection | uniform share | curriculum |
|---|---|---|---|
| 1 | `sonic` | 0.8 -> 0.5 over its first 4B | 5M-30M |
| 2 | `sonic` | 0.5 -> 0.2 over its first 4B | off |
| 3-7 | `sonic` | pinned 0.2 | off |

Two mechanics force this shape:

* The ramp drives `SonicAdaptiveResetSampler.uniform_sampling_rate` and is
  legal only under `selection=sonic`. `random80_adaptive20` puts the
  adaptive sampler inside a random-trajectory wrapper that takes 80% of
  resets, so a ramp there rescales the remaining 20% only; the config
  refuses the combination (the `cont_det_hold1_resetramp` near-no-op of
  2026-08-28).
* `_advance_adaptive_ratio_curriculum` reads `common_step_counter`, which
  restarts every segment, so a ramp must finish inside the segment that
  starts it. A segment buys about 8.6B at ~150k fps, hence two 4B ramp
  stages rather than one 10B ramp. The mix reaches 20/80 at roughly 13B and
  holds it for the remaining 37B.

Second change against the 10B `combo` row, worth stating with any
comparison: `sonic` also drops the random first-half start frames that
`random80_adaptive20` supplied, because SONIC picks rank and frame jointly
from its bin distribution.

## Local 46B progress evaluation and studio videos (2026-09-07)

Pulled the latest saved checkpoint at retrieval, `model_step_46000373760.pt`,
from ICE run `2026-09-06_23-17-33_wandb-c50b-combo-s0-a4644b` while segment 8
(job 5716204) was running. This is a single-seed, in-progress snapshot, not a
completed 50B result. Remote/local checkpoint SHA-256 matched
`f44702fe3cfcc83306a91af5e0268d258f81182d5fc281bc0cd7a225a822f1d9`; all 18
encoder tensors matched the downloaded `p5_affine` encoder exactly.

Local Newton/MJWarp, `bones_testbed4096_v1`, clean (`none`), seed 0, mode actions,
frame-0 starts: **SR 0.9805 (4016/4096), success-only micro MPJPE-L 20.86 mm,
MPJPE-G 67.98 mm**. All environments finished in 1457 steps; zero timeouts.
The exact ordered board ranks were verified. Explicit ranks resolve to the
evaluator's fixed `custom` rank-table schedule. Runtime and macro caches were
placed on CPU to fit the A4500's 20 GB GPU memory; 4096 environments retained.

Artifacts in `logs/combo_50b_local_20260907/`: original tracker and encoder,
saved training configs, `commands.json`, `eval_clean.json`, `audit.json`, and
`video_validation.json`. Five full-motion 1920x1080 videos use `studio_light`
with `hero_low`: ranks 606 (walk), 467 (run), 118 (jump), 2365 (dance), and 51
(clap). The renderer disables early terminations and randomization. Videos use
PhysX/RTX, so these presentation diagnostics are distinct from the Newton
metric pass. `videos/render_summary.json` records paths, steps, and settings;
all five MP4s were verified with ffprobe and the walking frame inspected.

## EC asynchronous integration and HF release (2026-09-07)

The same 46,000,373,760-frame snapshot was exported with the new
`combo64_history10_v1` preset: 996 actor inputs, ten-frame proprioceptive
history, heading-anchored root_qpos, one-tick hold, and encoding every control
tick. All 18 encoder tensors bind exactly. RLOpt/TorchScript parity is exact;
ONNX max error is 4.292e-6 policy / 1.312e-6 encoder. EC gained native
heading-only root_qpos packing (XY-only origin, preserved height and tilt),
and the oracle sweep now derives lead time from hold duration.

`logs/combo_50b_local_20260907/ec_async_50hz.json` records a local asynchronous
MuJoCo pass on all ten `bones_seed_language10_v1` motions. All full horizons
completed: 10/10 no-fall and post-hoc SONIC success; frame-weighted all-frame
MPJPE-L **12.39 mm**, MPJPE-G **134.11 mm**. Feeding-birds had 749.56 mm global
error despite passing the criterion. This single pass is a different population
and backend from the canonical Isaac board, not a cross-backend certification.
All 5,137 control ticks encoded; 20,548 independent physics steps; no faults,
control scheduler misses, or physics deadline misses. Worst motion compute
p99: 1.88 ms. Zero-lead asynchronous replies missed 2,567 request deadlines;
those ticks used the buffered current reference and still encoded normally.
The earlier `ec_async.json` is excluded because its on-acceptance encoder
ran at the wrong cadence. Raw telemetry and a feeding-birds comparison video
are retained beside the final report.

Validation: 19 exporter tests, 90 EC native tests, 198 EC default tests passed
(6 default skips). Release inventory: `controller/combo_46b`,
`checkpoints/combo_46b`, `evaluation/combo_46b`, and `runtime/combo_46b` in
`fei-yang-wu/ec-g1-gr00t-eval-kit`; EC's `model.pin.json` records the immutable
HF revision and per-file hashes. The existing FSQ GR00T planner is not bound
to this encoder and is not claimed compatible.

Verified HF revision: `461f4c1e33a5d1f63499846fc865a0c8ef595081` (27 release
files, every remote LFS SHA-256 matched; original checkpoint plus selected
metadata downloaded and hashed). EC fetched and pinned the ten bundle files
at `assets/models/controller/combo_46b/model.pin.json`. Full verification is
in `logs/combo_50b_local_20260907/hf_verification.json`.
