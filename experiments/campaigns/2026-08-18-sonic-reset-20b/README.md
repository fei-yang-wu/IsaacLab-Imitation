# 2026-08-18 — SONIC resets, 10B to 20B, on both surviving interfaces

Continue the two leading trackers from `2026-08-15-latent-bottleneck-10b` for a
second 10B, changing exactly one thing about training: **how a reset picks the
trajectory and the start frame**. Both arms resume from their base arm's final
checkpoint at cumulative 10,000,269,312 frames and run to a global 20B.

| arm | base arm | interface | 10B starting point (4,096 board) |
|---|---|---|---|
| `ln_hold1_sonicreset` | `cont_det_ln_hold1` | continuous 256-D deterministic latent, hold 1 | 0.9368 SR / 22.86 mm |
| `fsq64_hold10_sonicreset` | `fsq64_hold10` | discrete 64 x 32 FSQ lattice, hold 10 | 0.9197 SR / 24.93 mm |

Both are kept alive on purpose. The continuous arm leads both boards today; the
discrete arm is SONIC's own token space and the interface a language planner
emits into, so it is the one a downstream planner row depends on.

## The one variable

`env.command_interface.reference.selection`: **`random80_adaptive20` ->
`sonic`**.

The only field that differs between those two presets is
`random_trajectory_sampling_ratio`, **0.8 -> 0.0**. Everything else already
matched: both are `full_trajectory=true` with
`adaptive_failure_rate_max_over_mean=200.0`, `adaptive_bin_size=50`,
`adaptive_pre_failure_window=200`. So the change is precisely: *delete the
explicit uniform-trajectory branch and let every reset come from the SONIC
joint rank+frame failure sampler.*

On top of that, a landing ramp **inside** the sampler:

| segment | `adaptive_uniform_ratio` | failure-weighted share of resets |
|---|---|---|
| 5 | 0.5 -> 0.1, linear over its own 2.5B frames | 50% -> 90% |
| 6, 7, 8 | pinned 0.1 (SONIC's release value) | 90% |

The ramp is a landing, not the experiment. The resume point is a converged 10B
tracker whose whole training distribution was 80% uniform-trajectory; stepping
straight to 10% uniform is a large one-shot shift in the state distribution the
critic was fitted on.

### Why this is not a repeat of `cont_det_hold1_resetramp`

That arm (10B, inconclusive: 4,096 SR 0.9307 / 23.84 mm against its control's
0.9343 / 22.60 mm) ramped `adaptive_uniform_ratio` 0.8 -> 0.2 while leaving
`random_trajectory_sampling_ratio` at 0.8. The adaptive branch was therefore
still only 20% of resets, and the failure-weighted share moved **4% -> 16%**,
not 20% -> 80%. Its `start_mode=adaptive` override was dead code as well: with
`full_trajectory=true` the generic start-frame sampler is never constructed
([reference.py:702](../../../source/isaaclab_imitation/isaaclab_imitation/tasks/manager_based/imitation/mdp/commands/reference.py:702)).
The branch ratio itself was never touched by any arm before this campaign.

## What is held fixed

Byte-identical to the base campaign's `tuned_lowlevel` and `lowlevel_tail`
apart from the selection line: rewards, optimizer, network shape, gamma,
minibatch shape, `expert_batch_size`, dataset, `robot_heading` anchor, hold,
latent width, and seed. `hl_skill_finetune_enabled=false` on both arms, so the
DiffSR encoder stays the frozen pretrained one and the reset distribution is
the only moving part.

The termination curriculum is disabled in **every** segment: it completed
inside the base arm's segment 1 (5M -> 30M frames), so terminations sit at
their strict SONIC values throughout.

Each arm loads its base arm's encoder file directly
(`/data/bottleneck_10b/<base>_seed0/encoder/checkpoints/latest.pt`), which is
the same encoder already embedded in the checkpoint being resumed — the
skill-checkpoint binding is identical by construction, not by luck.

## Segmentation and the frame budget

`frame_cap` is the **global** 20B, handed to every segment. The resumed
checkpoint carries `cumulative_env_frames = 10,000,269,312`; `load_model`
restores it as the frame offset and the loop continues from there, so the chain
stops at exactly 20B no matter how the walltime divides it. `max_iterations` is
50,863 for all four segments. Never shrink it to fit a walltime.

The base arm's final checkpoint predates the `cumulative_env_frames` key (its
campaign tarball froze pre-2026-08-16 RLOpt; verified on the local mirror of
the exact file), so segment 5 passes
`agent.collector.initial_frame_offset=10000269312` — a new `CollectorConfig`
field that seeds the resume offset only when the loaded checkpoint carries no
cumulative count. A checkpoint value always wins over it, so segments 6-8,
which resume this arm's own checkpoints, are unaffected by the flag's absence.

`--checkpoint` receives a checkpoint **tree**, not a file: `train_impl`
resolves it to the newest `model_step_<N>.pt` by mtime, because step numbers
restart per segment. Segment 5 points at the base arm's tree; segments 6-8
point at this arm's own tree.

Four segments of `15:59:00` on `gpu:h200:1`, chained `afterany`. The base
campaign averaged ~2.5B frames per segment, so ~2.7 days per arm.

### The ramp must not re-sweep

Ramp progress is `common_step_counter * num_envs`, and nothing restores that
counter across a resume. Segments 6-8 therefore pin
`adaptive_uniform_ratio=0.1` with `adaptive_uniform_ratio_final=null` rather
than inheriting the ramp block. `cont_det_hold1_resetramp_dyn` shipped without
this on 2026-08-17 and re-swept its ramp from the start of every segment; that
is the failure this campaign's `reset_hold_args` exists to prevent.

## W&B

Group `sonic-reset-20b`. Run ids `ln-hold1-sonicreset-s0-r1` and
`fsq64-hold10-sonicreset-s0-r1`, one run per arm across all four segments. The
`-r1` is a generation suffix: the first submission's runs were deleted during
cleanup, and W&B **permanently refuses a deleted run id** (410 "previously
created and deleted; try a new run id" — the error that killed the second
submission in six minutes). Bump the suffix if a run must ever be deleted
again.

`WANDB_MODE=shared` stays on so the EC/MuJoCo sidecar can write `Eval/*` into a
live run. Shared mode **discards `wandb.log(step=...)`** ("In shared mode, the
use of `wandb.log` with the step argument is not supported and will be
ignored", `wandb/sdk/wandb_run.py`), which is why the two shared-mode runs
submitted on 2026-08-17 (`cont-det-hold1-dyn-s0`,
`cont-det-hold1-resetramp-dyn-s0`) plot against a bare log-call index instead
of frames. RLOpt now publishes the frame count as the ordinary metric
`env_frames` and declares it the wandb x-axis, so these runs keep a frame axis
in either mode and stay comparable with the pre-08-17 runs.

**Run-id reuse is a real hazard here.** `WANDB_RUN_ID` is `<arm>-s<seed>` with
`WANDB_RESUME=allow`, so a cancelled segment and its replacement append into
one history with no marker except the `logdir:` tag. If a segment must be
resubmitted, expect the overlap and read the `env_frames` axis, not row order.

## Pipeline

```bash
./submit.sh ln_hold1_sonicreset 0
./submit.sh fsq64_hold10_sonicreset 0
```

`submit.sh` only plans: it validates, preflights (including that the base arm's
tracker tree and encoder exist), freezes the argv, and prints the `submit
--plan <dir> --confirm <PLAN_SHA>` line. Nothing reaches Slurm until that
second command runs.

Score both arms with the existing runners once checkpoints land — the 4,096
Isaac board (`2026-08-15-latent-bottleneck-10b/eval_scoreboard4096.sh`, ~4 min
per arm locally) is the deciding one, and the strat-64 EC/MuJoCo sidecar is the
CPU screen. Read `wiki/canonical-paper-metrics.md` before quoting any number.

## Result (2026-08-21)

Both chains COMPLETE at exactly 20,000,145,408 frames (ln: segment 5 TIMEOUT +
segment 6 finished 2026-08-19 14:00; fsq: segment 6 finished 12:20; segments
7/8 found the budget met and exited in ~6 min). Scored locally on the frozen
4,096 scoreboard (`2026-08-15-latent-bottleneck-10b/eval_scoreboard4096.sh`,
rows in `logs/bottleneck_10b_4096/*_f20000145408.json`):

| arm | frames | SR | succ MPJPE-L | succ MPJPE-G | `ee_body_pos` |
|---|---:|---:|---:|---:|---:|
| `ln_hold1_sonicreset` | 20.00B | **0.9558** | **22.15 mm** | 168.1 mm | 136 |
| base `cont_det_ln_hold1` | 10.00B | 0.9368 | 22.86 mm | — | 217 |
| `fsq64_hold10_sonicreset` | 20.00B | **0.9468** | 24.57 mm | 202.7 mm | 189 |
| base `fsq64_hold10` | 10.00B | 0.9197 | 24.93 mm | — | 260 |

One seed. The gain over the base row confounds the reset scheme with the
second 10B of frames BY DESIGN — this is a continuation, and no 20B
`random80_adaptive20` control exists. What the rows support: continuing the
leaders under SONIC's failure sampler improves both interfaces on both axes,
and `ee_body_pos` failures drop by ~30-40%. They do not support attributing
the gain to the sampler alone. Released SONIC still leads SR (0.9937); both
arms lead it on MPJPE-L.

## Status

Running 2026-08-18, both arms, seed 0: `ln_hold1_sonicreset` jobs
5580282-5580285 (attempt 4), `fsq64_hold10_sonicreset` jobs 5580273-5580276
(attempt 3; `env_frames` verified starting at 10,000,662,528).

- Attempt 1 (5580042/46 + chains): cancelled after 9.2 h (~3.0B frames) — the
  base checkpoint's missing `cumulative_env_frames` key made the frame axis and
  budget restart at zero. Outputs wiped, W&B runs deleted.
- Attempt 2 (5580139-46): failed at wandb init in 6 minutes — deleting a W&B
  run burns its id forever (HTTP 410). Hence the `-r1` run-id suffix.
- Attempt 3, ln arm only (5580269-72): `KeyError: 0` in the terminal-obs
  reader at the first reset. Its tarball froze mid-edit of a concurrent
  session: `imitation_rl_env_v2` published the new batched `_env_ids`
  final_obs while `envs/rlopt.py` was still the old per-env reader. The fsq
  tarball, packed 34 seconds later, caught both sides new and runs fine.
  Do not submit while another session is editing this tree.
