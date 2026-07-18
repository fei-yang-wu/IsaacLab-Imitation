# Frozen-skill latent training scale benchmark

This workflow measures the shortest single-GPU wall-clock time to a sustained
episodic return of 7.5 on corrected LAFAN1. It varies only the number of Isaac
environments, PPO rollout steps per environment, and PPO minibatch size.

The frozen task profile is `Isaac-Imitation-G1-Latent-v0` with
`agent.ipmd.command_source=hl_skill`, the qualified H10/Z256 DiffSR skill
checkpoint, sinusoidal phase features, environment rewards, five PPO epochs,
and the frozen reset range 0-200. Skill pretraining and video rendering are not
part of the timed runs.

The three batch quantities are distinct:

```text
global rollout batch = num_envs * rollout_steps
minibatches per epoch = ceil(global rollout batch / minibatch_size)
optimizer updates per rollout = 5 * minibatches per epoch
```

## Local qualification

Dry-run and validate all pinned paths and hashes:

```bash
experiments/training_scale/run_local_latent_scale_benchmark.sh
```

Run the five-point constant-global-batch ridge. The default is approximately
10M frames per configuration and 50.14M effective frames in aggregate:

```bash
DRY_RUN=0 experiments/training_scale/run_local_latent_scale_benchmark.sh
```

Local runs qualify correctness, throughput, and memory behavior. The local RTX
PRO 6000 is not a substitute for the A40 ranking.

## Skynet A40 screen

Inspect the exact submission without contacting Skynet:

```bash
experiments/training_scale/submit_latent_scale_benchmark_skynet.sh
```

Submit one sequential A40 allocation after the local screen is reviewed:

```bash
DRY_RUN=0 experiments/training_scale/submit_latent_scale_benchmark_skynet.sh
```

The default A40 matrix has ten 75M-frame screens (about 750M effective frames),
a 42,000 MiB VRAM guard, and a hard 2B aggregate-frame cap. It includes four
constant-98,304-sample ridge points, two balanced larger-batch points, two
rollout-batch probes, and two minibatch probes. The locally dominated 16,384 x
6 endpoint is excluded. Every child process starts from scratch with the same
seed and frozen inputs. The output root contains `benchmark_plan.json`, one
directory per configuration, and an incrementally updated
`benchmark_results.json`.

The primary metric is the confirmation time for three consecutive logged
last-100-episode returns at or above 7.5. First-hit time, frames to threshold,
median logged FPS, peak VRAM, total training time, and process time are retained
as secondary diagnostics.

The cluster runtime binds compute-local job storage to the container's `/tmp`.
This is required for sequential Isaac processes: URDF-to-USD conversion uses a
hard-coded `/tmp/IsaacLab` path, which otherwise consumes the writable overlay.
The `a40-remaining` preset omits the baseline and is intended only for an
audited continuation when a completed baseline is already retained.

After the seed-0 screen, use `PRESET=a40-confirm` with independent `SEED`
values to confirm only the two leaders. This preset contains 12,288 x 8 with a
24,576 minibatch and 8,192 x 12 with a 12,288 minibatch; at 75M frames each,
one confirmation seed consumes about 150M effective frames.

For a single long convergence run, use the validated winner through the
`a40-production` preset. The largest exact 98,304-frame rollout multiple below
the 2B cap is 1,999,994,880 frames:

```bash
DRY_RUN=0 \
PRESET=a40-production \
TARGET_FRAMES_PER_RUN=1999994880 \
AGGREGATE_FRAME_CAP=2000000000 \
experiments/training_scale/submit_latent_scale_benchmark_skynet.sh
```

The completed A40 investigation and production recommendation are recorded in
[`2026-07-17-a40-results.md`](2026-07-17-a40-results.md).
