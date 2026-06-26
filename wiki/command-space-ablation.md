# Command-Space Ablation

This plan compares how much whole-body imitation performance depends on the
high-level command representation, separately from how good a planner is at
generating that command.

## Level 1: Oracle Command Tracking

Train the same low-level policy and reward setup while feeding command tensors
from the expert reference trajectory. This isolates command expressiveness.

Command spaces:

- `single_frame_full_body`: existing single-step full-body reference command.
  The policy receives joint position/velocity plus anchor pose error.
- `full_body_trajectory`: BeyondMimic-style low-level interface. The command is
  a future whole-body trajectory window, exposed as `expert_window` motion plus
  anchor pose terms.
- `ee_trajectory`: VLA-style task-space interface. The command is a future
  end-effector pose trajectory for both feet and both wrists, exposed as
  `expert_window.expert_ee_pos_b` and `expert_window.expert_ee_ori_b`.

All three keep the current proprioceptive inputs. The critic remains
privileged with robot state so the ablation primarily changes the policy command
surface, not the value function's state-estimation problem.

Local smoke:

```bash
DRY_RUN=1 experiments/command_space_ablation/run_local_oracle_smoke.sh
experiments/command_space_ablation/run_local_oracle_smoke.sh
```

The local smoke launcher sets `SAVE_INTERVAL=1` by default so a one-iteration
debug run emits a checkpoint that can be reused by the interface-baseline smoke
tests. Override it if you only need training logs.

Full cluster dry run:

```bash
DRY_RUN=1 experiments/command_space_ablation/submit_cluster_oracle_ablation.sh
```

The default cluster launcher runs three seeds (`2024 2025 2026`) for the three
oracle command spaces. It uses `4096` envs and `10173` iterations, matching
about 1B frames at 24 frames per rollout iteration. The default command horizon
is 25 future steps, so each trajectory command contains the current frame plus
25 future frames. `docker/cluster/.env.cluster` is set to
`CLUSTER_G1_MANIFEST_PATH=${CLUSTER_DATA_DIR}/unitree/manifests/g1_unitree_dance102_manifest.json`
for the first Dance102 pass; override `MANIFEST` or the cluster env if you want
the full G1 LAFAN manifest.

The command observation source defaults to `reference`, which reads commands
directly from the synchronized expert trajectory. Set
`COMMAND_OBSERVATION_SOURCE=planner_oracle` to route those same reference
commands through the env's planner command buffers. That smoke path exercises
the Level 2 command bridge without changing the command contents.

Useful overrides:

```bash
COMMAND_SPACES="full_body_trajectory ee_trajectory" \
COMMAND_FUTURE_STEPS=50 \
SEEDS="2024 2025 2026" \
DRY_RUN=1 experiments/command_space_ablation/submit_cluster_oracle_ablation.sh
```

## Evaluation Table

After training, evaluate the best or final checkpoint for each command-space /
seed pair with the deterministic checkpoint evaluator. The wrapper expects
`CHECKPOINTS` in command-space-major order, then seed order, matching the submit
script's nested loops.

For the default cluster sweep, first discover checkpoints from the shared
cluster log root:

```bash
ALLOW_MISSING=1 experiments/command_space_ablation/list_cluster_checkpoints.sh
```

For matched intermediate evaluations, request a target frame count. This avoids
mixing older 50M checkpoints with newer 100M checkpoints while waiting for slower
seeds:

```bash
TARGET_STEP=50000000 \
ALLOW_MISSING=1 experiments/command_space_ablation/list_cluster_checkpoints.sh
```

Once every row reports `ok`, use `CHECKPOINTS_HOST` for local evaluation after
copying or mounting the remote files, or `CHECKPOINTS_CONTAINER` when submitting
evaluation jobs through `docker/cluster/cluster_interface.sh`, where the shared
cluster log root is visible as `/workspace/isaaclab/project/logs`.

To submit deterministic evaluation jobs on the cluster, either let the launcher
discover the latest ordered checkpoints:

```bash
TARGET_STEP=50000000 \
OUTPUT_TAG=oracle_50m \
experiments/command_space_ablation/submit_cluster_evaluations.sh
```

or paste the helper's `CHECKPOINTS_CONTAINER` output explicitly:

```bash
CHECKPOINTS="logs/rlopt/ipmd/Isaac-Imitation-G1-v0/.../models/model_step_50000000.pt ..." \
OUTPUT_TAG=oracle_50m \
experiments/command_space_ablation/submit_cluster_evaluations.sh
```

The cluster launcher submits one evaluation job per checkpoint and writes one
CSV / JSON pair per run under
`logs/command_space_ablation/eval_results/$OUTPUT_TAG` in the shared cluster
log mount. It defaults to `CLUSTER_SKIP_CACHE_COPY=1` because evaluation results
are already on the shared log bind mount and Isaac Sim cache rsync failures can
otherwise mark completed evaluations as failed. Merge those per-run CSVs before
rendering the final comparison table:

```bash
python experiments/command_space_ablation/merge_eval_csvs.py \
    --input_dir logs/command_space_ablation/eval_results/oracle_50m
```

For planner-burden evaluations, set `PLANNER_MODE` and restrict
`COMMAND_SPACES` to trajectory command spaces. The launcher defaults
`COMMAND_OBSERVATION_SOURCE` to `planner` whenever `PLANNER_MODE != none`; set
`COMMAND_OBSERVATION_SOURCE` explicitly only when testing a specific override.

```bash
TARGET_STEP=100000000 \
TARGET_STEP_TOLERANCE=3000000 \
COMMAND_SPACES="full_body_trajectory ee_trajectory" \
PLANNER_MODE=hold_current \
OUTPUT_TAG=planner_hold_current_100m \
experiments/command_space_ablation/submit_cluster_evaluations.sh
```

```bash
SEEDS="2024 2025 2026" \
COMMAND_SPACES="single_frame_full_body full_body_trajectory ee_trajectory" \
CHECKPOINTS="/path/single_seed2024.pt /path/single_seed2025.pt /path/single_seed2026.pt \
/path/full_body_seed2024.pt /path/full_body_seed2025.pt /path/full_body_seed2026.pt \
/path/ee_seed2024.pt /path/ee_seed2025.pt /path/ee_seed2026.pt" \
experiments/command_space_ablation/evaluate_oracle_checkpoints.sh
```

The evaluator writes one JSON file per checkpoint plus a shared CSV and Markdown
summary table under `experiments/command_space_ablation/eval_results/` unless
`OUTPUT_DIR` or `OUTPUT_CSV` is set. It also writes `aggregate.md`, grouped by
`command_space` and `planner_mode`, for quick multi-seed comparison. Core
reported metrics include episode return, survival steps, done / terminated /
truncated rates, root tracking, joint position and velocity RMSE, tracked-body
pose error, end-effector pose error, action smoothness, and planner-command
error when the planner bridge is used.

For a single checkpoint:

```bash
pixi run -e isaaclab python experiments/command_space_ablation/evaluate_checkpoint.py \
    --task Isaac-Imitation-G1-v0 \
    --algo IPMD \
    --headless \
    --checkpoint /path/to/model_step_1000000000.pt \
    --command_space full_body_trajectory \
    --command_future_steps 25 \
    --motion_manifest ./data/unitree/manifests/g1_unitree_dance102_manifest.json \
    --output_json /tmp/full_body_eval.json \
    --output_csv /tmp/command_space_eval.csv
```

Use the same `COMMAND_FUTURE_STEPS`, manifest, and command-space value as the
training run. Set `COMMAND_OBSERVATION_SOURCE=planner_oracle` in the wrapper to
verify that the planner buffer path still reproduces oracle commands before
plugging in a real planner.

## Current 100M Evidence

As of 2026-06-23, the Dance102 100M checkpoint comparison has three completed
evidence slices:

- `oracle_100m`: direct reference commands for all three command spaces.
- `planner_reference_100m_device_fix`: external planner buffer publishes the
  exact reference command for `full_body_trajectory` and `ee_trajectory`.
- `planner_hold_current_i5_100m`: external planner publishes every 5 policy
  steps and the command is held between updates.

The reference-vs-oracle bridge sanity check passed: `planner_reference` matches
oracle for `ee_trajectory`, and is within seed noise for `full_body_trajectory`.
The command RMSE metrics in the planner buffer path are exactly zero for both
trajectory command spaces.

Completed aggregate table:

| tag | command_space | planner_mode | n | return | return vs reference | survival | survival vs reference | done_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| oracle | single_frame_full_body | none | 3 | 72.61 +/- 1.2615 | n/a | 930.16 +/- 9.7074 | n/a | 0.0833 +/- 0.0074 |
| reference | ee_trajectory | reference | 3 | 49.27 +/- 1.8194 | 100.00% | 685.49 +/- 23.27 | 100.00% | 0.5052 +/- 0.0993 |
| noisy005 | ee_trajectory | noisy_reference | 3 | 48.70 +/- 1.5432 | 98.83% | 676.39 +/- 21.77 | 98.67% | 0.5182 +/- 0.0808 |
| hold5 | ee_trajectory | hold_current | 3 | 23.47 +/- 2.4796 | 47.63% | 386.35 +/- 51.65 | 56.36% | 1.0000 +/- 0.0000 |
| hold25 | ee_trajectory | hold_current | 3 | 2.6588 +/- 0.1380 | 5.3961% | 65.55 +/- 4.0627 | 9.5620% | 1.0000 +/- 0.0000 |
| reference | full_body_trajectory | reference | 3 | 26.23 +/- 1.2589 | 100.00% | 371.78 +/- 19.02 | 100.00% | 0.8828 +/- 0.0338 |
| noisy005 | full_body_trajectory | noisy_reference | 3 | 23.78 +/- 0.2167 | 90.66% | 336.78 +/- 1.7235 | 90.59% | 0.9036 +/- 0.0133 |
| hold5 | full_body_trajectory | hold_current | 3 | 11.97 +/- 0.1995 | 45.62% | 177.74 +/- 6.4771 | 47.81% | 1.0000 +/- 0.0000 |
| hold25 | full_body_trajectory | hold_current | 3 | 2.7588 +/- 0.1505 | 10.52% | 62.66 +/- 2.0354 | 16.86% | 1.0000 +/- 0.0000 |

Interim read: the full-body trajectory command is already harder to track under
perfect commands at 100M, while the EE trajectory command keeps roughly twice
the absolute return and survival. Under a slow `hold_current` planner, both
trajectory interfaces degrade sharply. At interval 5, EE keeps more absolute
return and survival and slightly more survival relative to its own reference
baseline. At interval 25, both interfaces are nearly collapsed; full-body keeps
slightly more relative return, but both have about 60-66 survival steps and a
100% done rate. Under small command noise (`PLANNER_NOISE_STD=0.05`), both
interfaces are robust, but EE retains more of its reference performance
(`98.83%` return, `98.67%` survival) than full-body trajectory (`90.66%`
return, `90.59%` survival).

Use the comparison helper to regenerate the current table from merged result
directories:

```bash
python experiments/command_space_ablation/compare_eval_tags.py \
    --eval_root logs/command_space_ablation/eval_results \
    --tag oracle=oracle_100m \
    --tag reference=planner_reference_100m_device_fix \
    --tag hold5=planner_hold_current_i5_100m \
    --tag hold25=planner_hold_current_i25_100m \
    --tag noisy005=planner_noisy_reference_std005_100m \
    --baseline_tag reference \
    --output_md logs/command_space_ablation/eval_results/available_comparison.md
```

## Level 2: Closed-Loop Planner Comparison

After Level 1 establishes that each command space can train a competent
low-level policy, evaluate planner burden with a closed-loop high-level planner
for each command space:

- Whole-body planner: generates full-body trajectory commands, analogous to a
  BeyondMimic-style motion-generation planner followed by the same low-level
  tracker.
- EE planner: generates full end-effector pose trajectories, analogous to
  current VLA pipelines that plan in task space.
- Learned latent planner: generates the compact learned command used by the
  current latent-command stack.

For the closed-loop comparison, keep the low-level policy checkpoint fixed per
command space and report task success, imitation/tracking quality, planner
latency, invalid-command rate, and recovery after planner errors.

The env-side bridge is available through `env.command_observation_source`:

- `reference`: default Level 1 oracle path.
- `planner_oracle`: fills the planner command buffers from the current reference
  command before each observation. Use this to smoke-test the bridge.
- `planner`: reads only agent/planner-published command buffers. External
  planners should call `set_agent_full_body_trajectory_command(...)`,
  `set_agent_ee_trajectory_command(...)`, or the generic
  `set_agent_trajectory_command(...)` on the Isaac env or RLOpt wrapper before
  policy inference.

The checkpoint evaluator includes simple planner publishers so the command
bridge can be compared before integrating a learned or optimization-based
planner:

- `PLANNER_MODE=reference`: publishes the exact expert trajectory through the
  external planner API. This should match `planner_oracle` up to observation
  timing and is the bridge sanity baseline.
- `PLANNER_MODE=hold_current`: repeats the current command frame over the whole
  future horizon. This measures how much the low-level policy depends on real
  future planning rather than current-pose tracking.
- `PLANNER_MODE=noisy_reference`: adds Gaussian noise to the oracle command
  before publishing. Use `PLANNER_NOISE_STD` to sweep command robustness.
- `PLANNER_MODE=zero`: publishes all-zero trajectory commands as a lower-bound
  sanity check.

Planner modes are only meaningful for `full_body_trajectory` and
`ee_trajectory`, so leave `single_frame_full_body` out of these sweeps.

Examples:

```bash
COMMAND_SPACES="full_body_trajectory ee_trajectory" \
CHECKPOINTS="/path/full_body_seed2024.pt /path/ee_seed2024.pt" \
PLANNER_MODE=reference \
experiments/command_space_ablation/evaluate_oracle_checkpoints.sh

COMMAND_SPACES="full_body_trajectory ee_trajectory" \
CHECKPOINTS="/path/full_body_seed2024.pt /path/ee_seed2024.pt" \
PLANNER_MODE=hold_current \
experiments/command_space_ablation/evaluate_oracle_checkpoints.sh

COMMAND_SPACES="full_body_trajectory ee_trajectory" \
CHECKPOINTS="/path/full_body_seed2024.pt /path/ee_seed2024.pt" \
PLANNER_MODE=noisy_reference \
PLANNER_NOISE_STD=0.03 \
experiments/command_space_ablation/evaluate_oracle_checkpoints.sh
```

Set `PLANNER_UPDATE_INTERVAL` above `1` to simulate slower planner cadence; the
last published command is held between updates. The real whole-body and EE
planners should ultimately replace these simple publishers but keep the same
planner-buffer API and metric table.
