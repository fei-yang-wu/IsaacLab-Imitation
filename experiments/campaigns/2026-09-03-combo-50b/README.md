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

## Completion check (2026-09-07 19:38 UTC)

The 50B chain is not yet complete. Segment 8 `5716204` timed out at
48,550,379,520 logged frames; segment 9 `5716205` is running from the
retained 48,000,270,336-frame checkpoint. Its latest sampled log is
48,130,424,832 / 50,000,166,912 frames, about 150.6k frames/s, implying
roughly 3.5 hours remaining if throughput holds. The replayed 0.55B-frame
interval is below W&B's existing high-water mark, so W&B warns that those
points are ignored; live Slurm logs show training advancing. No final 50B
checkpoint exists at this check. User authorized final eval jobs conditional
on training completion; none submitted at this check because it is unfinished.

## Final eval retry (2026-09-08 00:30 UTC)

Training `5716205` COMPLETED at 50,000,166,912 frames. Final checkpoint
`/data/combo_50b/combo_seed0/tracker/2026-09-07_15-16-57_wandb-c50b-combo-s0-a4644b/models/model_step_50000166912.pt`.
Evaluation `5725616` failed in Kit startup (XOpenDisplay crash, exit 139),
without a 50B result. Retry `5725971` submitted after all preflights passed,
using a final-checkpoint-only tree at `/data/combo_50b_final_eval` and the
existing combo50b_live clean protocol. Plan SHA
`9f7bccb88f6c61ea6362cb409f911a1928d3f8c293770dc0e5a0c2291ec8ce02`.
Latest completed row remains 49,000,218,624 frames: SR 0.9846,
success-only micro MPJPE-L 20.53 mm / MPJPE-G 69.59 mm, seed 0,
canonical clean 4096 board. Exact ranks, full completion, zero timeouts and
clean mode settings validated. Local JSON in `logs/combo_final_eval/`.

## 2B checkpoint evaluation (2026-09-08 00:33 UTC)

User requested the 2B checkpoint of this same combo-50B lineage for comparison
with the explicit 2B rows. Submitted ICE `5725997`, seed 0, canonical clean
4096 board, existing combo50b_live protocol. Checkpoint:
`/data/combo_50b/combo_seed0/tracker/2026-09-03_11-49-00_wandb-c50b-combo-s0-a4644b/models/model_step_2000289792.pt`.
Exactly 2,000,289,792 frames, matching the explicit checkpoints. Pinned tree:
`/data/combo_50b_2b_eval/combo50b_seed0/tracker/f2000289792/models/`.
All preflight checks passed. Plan SHA:
`7e93f741b454ef7d4bd8563fddb9a24d37bf419ae9b7b5d74b1c1cbe7e6b88c4`.
Expected result `/data/eval/latest_eval/combo50b_seed0_clean_f2000289792.json`.
Frame/board matching does not by itself establish a one-variable comparison;
training schedule and recipe differences still need accounting at reporting.

## Verified 2B and final 50B results (2026-09-08 UTC)

Jobs `5725997` (2B) and `5725971` (50B) completed successfully in 8m00s
and 8m01s. Canonical clean 4096 board, seed 0, success-only micro errors:

| Frames | SR | MPJPE-L mm | MPJPE-G mm | Body jerk m/s³ | Action delta L2 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2,000,289,792 | 0.8782 (3597/4096) | 27.61 | 121.37 | 210.2 | 0.852 |
| 50,000,166,912 | 0.9827 (4026/4096) | 20.54 | 70.51 | 193.1 | 0.855 |

Verified exact checkpoint paths, ordered canonical ranks, full completion,
zero timeouts, deterministic mode and disabled startup/reset/push randomization.
Raw artifacts mirrored in `logs/combo_final_eval/`; remote originals in
`/data/eval/latest_eval/`. Single-seed measurements. At matched 2B frames,
root-qpos explicit has higher SR and lower local error than combo, while combo
has lower global error; EE explicit has almost equal SR and larger local/global
errors. Different success sets and training schedules preclude a causal interface
claim from these aggregate rows alone.

## Final 50B EC release and SONIC deployment examples (2026-09-08)

Published 50,000,166,912-frame controller under `controller/combo_50b`
in `fei-yang-wu/ec-g1-gr00t-eval-kit`, revision
`0678275bb8265e726e3dc7386f5fb3c8e42a8ec0`. Original tracker and encoder,
canonical Isaac result and export verification are included. Native and
TorchScript parity are exact; ONNX policy max error 4.7684e-6 and encoder
1.3113e-6; encoder binding is frozen_identical with zero divergence.
All 15 uploaded files passed inventory/size checks and LFS SHA checks where
applicable; manifest, policy ONNX and encoder ONNX were downloaded and hashed.

Private dataset `fei-yang-wu/sonic-deployment-motions`, revision
`afb8a5fadc691ca8eb0072188ad65521f7958d0b`, contains all 13 exact deployment
examples shipped in upstream NVlabs/GR00T-WholeBodyControl commit
`087f9ac01d46f6d8e4d0b73c01ae64799f292a38`. Discovered through
https://nvlabs.github.io/GEAR-SONIC/ and its linked source. These are NOT
verified identities of the website video trajectories or an official benchmark.
Includes upstream CSVs/license, 78 verified LFS object hashes, EC reference
conversion, pose audit, full asynchronous telemetry and 13 comparison videos.
All 164 published files were verified by size and LFS SHA where present;
selected downloaded reference manifest/qpos/summary bytes were hashed.
Dataset privacy was verified. Existing model-repo visibility was preserved.

EC fetched and pinned the controller (10 files) and reference tree (9 files)
under `assets/models/controller/combo_50b` and
`assets/models/reference/sonic_deployment`; every installed hash was checked.
New EC example `examples/lifecycle_sim_combo_50b.yaml` selects the walking
motion and matching frame horizon. EC has four new uncommitted pin/example
files; no submodule commit or parent-pointer update was made.

Async full-motion single pass: 7/13 post-hoc SONIC successes, successful-frame
MPJPE-L/G 27.90/266.00 mm. All-frame MPJPE-L/G 99.29/2351.37 mm includes
continued rollout after failures and is not comparable to canonical success-only
Isaac results. All 13 horizons completed (7604 ticks), all ticks encoded at
50 Hz, zero faults and missed control deadlines, worst motion tick p99 1.78ms.
3800 zero-lead oracle response deadlines were missed; those ticks used buffered
current references. The `no_fall` metric is only a >0.4m base-height check and
flags the successful squat; do not interpret 10/13 as a reliable fall statistic.
Walking, dance, lunges and squat pass; Macarena, kicks and jumps expose failures.
Full failed motions are retained in videos. Cameras independently center robots
(reference left, policy right); global drift must be read from metrics/telemetry.
All 13 videos decode; walking and Macarena frames visually inspected after
fixing the initial floor-only replay camera. No new simulation code was required.

Local evidence: `logs/combo_50b_release_20260908/` (export_verified.log,
async_eval.json, videos/, video_validation.json, hf_*_verification.json).
Reproduction command: EC `examples/combo_50b_sonic_deployment.md`.

## Frozen capability124 evaluation (2026-09-08)

User selected `sonic_capability124_v1` after the requested 119-motion set
could not be identified. Submitted ICE `5735277` for final 50,000,166,912
frames, clean deterministic mode, seed 0, frame-0 sequential starts,
10,000-step generous full-motion cap. Frozen 124-rank JSON SHA verified:
`19b83597f0e7bf86fb462ae691b1dad455bb6b8cc130a9a4c702062aa75de147`.
The new combo50b_capability124 arm in latest-eval uses explicit frozen ranks
and 124 environments; output is isolated at `/data/eval/combo50b_capability124`.
All preflights passed. Plan SHA:
`98343eede4300c442ed5842447bd376b33709969a819313948402b8aaf792606`.
This is a SONIC-calibrated in-distribution subset, not held out or unbiased;
keep its SR and success-only MPJPE-L/G separate from the canonical board.

### Capability124 result verified (2026-09-08)

Job `5735277` COMPLETED 0:0 in 11m32s. Final 50,000,166,912-frame combo,
seed 0, clean frozen SONIC-calibrated 124-motion subset: **SR 1.0000
(124/124), success-only micro MPJPE-L 18.22 mm / MPJPE-G 65.06 mm**.
Tracking acceleration distance 3.69 m/s², body jerk 162.2 m/s³,
action delta L2 0.715, using the canonical summarizer reduction.
Verified final checkpoint identity, exact ordered frozen ranks, all_envs_done,
done_rate=1, no timeouts, mode actions and no startup/reset/push randomization.
Single seed and calibrated in-distribution population; not an unbiased or
held-out benchmark. Local raw artifact:
`logs/combo50b_capability124/combo50b_seed0_clean_f50000166912.json`.

## Capability124 paper rendering (2026-09-08)

Submitted ICE job `5735360` using `render_capability124.yaml`; all preflights
passed. Final checkpoint `model_step_50000166912.pt`, p5_affine encoder,
z64 plus phase, ten-step actor history, seed 0, exact frozen 124 ranks.
One H200, 160G RAM, 16 CPUs, 15:59 walltime maximum. Plan SHA:
`5c3b3ebc624ffe3666810644121b10fb39aa040cc40cd1711a341156a1cede12`.

Produces full-horizon 3840x2160 MP4 videos and eight-pose opaque PNG sequence
figures with studio_light background and left-to-right time. Individual pose
frames, background plates, actual achieved pose takes, and render_summary.json
are retained for paper layout and provenance. Sequence figures horizontally
space poses for readability; they are not spatial displacement measurements.
Videos preserve actual trajectories; root straightening is off. Early
terminations are disabled so failures remain visible. This PhysX diagnostic
render is separate from Newton numeric evaluation; do not claim identical
trajectories or attach Newton statistics to individual rendered frames.

Persistent ICE output:
`/storage/ice-shared/vip-vwt/scratch-fwu91/renders/combo50b_capability124_20260908/`.
Submission is verified; rendered artifacts still require completion and visual
inspection before being called paper-ready.

### Render cancellation and local preview

User reported cluster videos blurry/noisy; cancelled render job 5735360 and
verified Slurm CANCELLED. Completed numerical evaluation is unchanged.
Rendered rank 1432 locally on RTX A4500 with the same final checkpoint and
interface: studio_light, hero_low, 1920x1080, explicit DLAA, full motion.
Output: logs/combo50b_local_render_20260908/videos/rank-001432-dance_hiphop_point_sequence_R_fast_001_A319_M.mp4.
Process exited 0 and full video decoded without errors. Frame at 1s inspected:
robot and background appear clean. Camera, resolution, AA and hardware changed,
so this does not isolate the cause of the cluster noise. Exact command saved
under logs/combo50b_local_render_20260908/command.json. No local batch launched.

### Local lighting comparison and transparent walking assets

Replayed the same rank1432 achieved take under light, dark, studio_dark and
photoreal, with fixed hero_low camera, 1080p and DLAA. All four processes
completed; FFmpeg full decode checks passed, frame-at-1s comparison and temporal
sheets inspected. Evidence: logs/combo50b_lighting_comparison/. studio_dark
provides stronger separation; studio_light is better suited to a white page.

Rendered clean full-motion rank45540 walk_ff_loop_180_R_002_A268 locally,
then replayed the achieved take at 4K. An isolated prototype under
logs/combo50b_walking_alpha/render_alpha.py adds instance-ID masks for /Robot
paths, excluding the floor and shadows. Eight frames composed at equal image
scale, chronological left-to-right, with editorial spacing/common baseline.
Transparent shaded PNG and dark silhouette PNG are 1887x488; recolorable SVG
traces the same masks. Inspected the white-background preview and verified
alpha spans 0..255. Actual shaded PNGs retain alpha. These are PhysX visual
assets, separate from Newton metrics. Reproduction commands and compose.py
are saved beside assets; prototype has not been promoted to shared code.

### Official SONIC v1.1 studio-dark comparison

Rendered official nvidia/GEAR-SONIC sonic_v1_1/last.pt, pinned revision
9c0ff22; local SHA256 matched NVIDIA Hub LFS OID. Existing release actor and
observation adapter, no finetuning. Six clean PhysX rollouts at frame0/seed0,
ranks 1070,4091,4696,21539,45540,2953; full horizon with terminations disabled.
Joint/reference order assertions passed. Actual pose takes replayed in matching
studio_dark/hero_low/1080p/DLAA settings, root straightening off. Each robot
has an independent follow camera; this comparison does not visualize global
drift. SONIC keeps its own encoder/action/actuator contract; this is our native
PhysX reproduction, not upstream deployment runtime.

All six SONIC/Combo pairs have identical frame counts and FPS; all paired videos
passed FFmpeg full decoding, sampled comparison frames inspected. Outputs:
logs/sonic_v11_studio_dark_comparison/paired/ (combo left, official SONIC right).
Checkpoint verification, prototype collection/replay/pairing scripts, provenance
and validation.json retained there. No aggregate performance conclusion from
this selected qualitative batch. The collector's inherited printed deviation
string says Newton even though the explicit physics override is PhysX; no
numeric result from that diagnostic output is reported.
