# Dance102 No-Language Planner Debug

Date: 2026-06-18 to 2026-06-19

This note summarizes the single-trajectory closed-loop debugging run for the
hierarchical G1 planner stack. The goal was to remove language embedding as a
possible compounding error and test whether a state-history-conditioned planner
can drive a low-level latent policy on one expert trajectory.

## Goal

We split the system into:

- System 0: low-level latent-conditioned IPMD bilinear policy.
- System 1: no-language high-level planner that maps state history to latent
  skill commands.
- System 2: absent for this experiment.

The intended diagnostic was:

1. Pretrain a fresh Dance102 skill encoder.
2. Pretrain a fresh Dance102 planner with no language, state-history input, and
   latent skill `z` as the action target.
3. Train the low-level policy with the oracle skill encoder frozen.
4. Record inference videos with oracle latent commands and with the trained
   planner.

This isolates single-trajectory planning quality before adding language
embeddings or a future vision-language System 2.

## Configuration

Run root:

```text
logs/dance102_single_trajectory_debug/20260618_150520_dance102_h10_hist10_no_language_flow
```

Main metadata:

- Trajectory: `dance102`
- Source NPZ: `data/unitree/npz/g1/G1_Take_102.bvh_60hz.npz`
- Manifest: `data/unitree/manifests/g1_unitree_dance102_manifest.json`
- Dataset cache: `data/unitree/g1_dance102_hl_diffsr`
- Task: `Isaac-Imitation-G1-Latent-v0`
- Low-level algorithm: `IPMD_BILINEAR`
- Environments: `4096`
- Horizon: `10`
- Planner history: `9` past states plus current state, so `10` state frames
- Skill latent width: `256`; low-level latent command width is `258` because
  the command includes phase sin/cos
- Planner type: flow matching
- Planner inference steps: `16`
- Language condition: none

The pipeline script that reproduces this setup is:

```bash
scripts/rlopt/run_dance102_no_language_debug_pipeline.sh
```

For a posthoc achieved-state playback using an already completed run:

```bash
RUN_ROOT=logs/dance102_single_trajectory_debug/20260618_150520_dance102_h10_hist10_no_language_flow \
scripts/rlopt/run_dance102_no_language_posthoc_eval.sh
```

## Code Changes

Top-level repo changes:

- `scripts/rlopt/train_skill_commander.py`
  - Adds `--no_language`.
  - Adds `--state_history_steps`.
- `scripts/rlopt/eval_skill_commander_m1.py`
  - Evaluates planner input using the configured state history.
  - Skips wrong-language metrics when the planner is no-language.
- `scripts/rlopt/play.py`
  - Adds `--output_dir` so eval videos can be written outside the checkpoint
    log directory.
- `source/isaaclab_imitation/isaaclab_imitation/envs/imitation_rl_env.py`
  - Adds state-history support to expert, current-reference, and
    achieved-state macro transition samplers.
  - Updates achieved-state batches so the current state-history frame uses the
    robot achieved state.
- `source/isaaclab_imitation/isaaclab_imitation/envs/rlopt.py`
  - Forwards `state_history_steps` through the RLOpt wrapper.
- `.gitignore`
  - Ignores generated high-level DiffSR dataset caches under
    `data/unitree/*_hl_diffsr/`.

RLOpt submodule changes:

- `rlopt/agent/skill_commander.py`
  - Supports no-language planners with zero-width language conditioning.
  - Supports flattened state-history planner input.
  - Saves checkpoint metadata for macro-state width, planner-state width,
    language conditioning, and history length.
  - Replays no-language planners without requiring a language embedding table.
- `rlopt/agent/ipmd/ipmd.py`
  - Allows `command_source=skill_commander` without
    `skill_commander_embeddings_path` when the checkpoint is no-language.
- `tests/test_skill_commander.py`
  - Adds coverage for no-language flow-matching planner training, checkpoint
    roundtrip, and sampler replay.

## Results

Skill encoder:

```text
logs/dance102_single_trajectory_debug/20260618_150520_dance102_h10_hist10_no_language_flow/skill_encoder_h10_z256/checkpoints/latest.pt
```

Planner:

```text
logs/dance102_single_trajectory_debug/20260618_150520_dance102_h10_hist10_no_language_flow/planner_flow_matching_no_language_hist10/checkpoints/latest.pt
```

M1 expert-state planner eval:

- Aggregate `z_cosine`: `0.9864834100008011`
- Aggregate `z_mse`: `0.011249552015215158`
- Per-trajectory Dance102 `z_cosine`: `0.9863936752080917`
- Per-trajectory Dance102 `z_mse`: `0.011133248684927821`

Summary file:

```text
logs/dance102_single_trajectory_debug/20260618_150520_dance102_h10_hist10_no_language_flow/m1_eval_planner_no_language/summary.json
```

Low-level training:

- Completed `10000/10000` iterations.
- Completed `983040000/983040000` frames.
- Final logged reward step was about `0.0773`.
- Final logged episode return was about `38.6234`.
- Pipeline-selected checkpoint:

```text
logs/rlopt/ipmd_bilinear/Isaac-Imitation-G1-Latent-v0/2026-06-18_15-10-57/models/model_step_980090880.pt
```

Training videos:

- `96` training videos were generated under:

```text
logs/rlopt/ipmd_bilinear/Isaac-Imitation-G1-Latent-v0/2026-06-18_15-10-57/videos/train
```

## Evaluation Videos

The main pipeline generated two videos. Both were verified with `ffprobe` as
`499` frames at `50` fps, duration `9.98` seconds.

Oracle high-level skill command video:

```text
logs/dance102_single_trajectory_debug/20260618_150520_dance102_h10_hist10_no_language_flow/video_eval_oracle_hl_skill/videos/play/rl-video-step-0.mp4
```

Trained planner video using reference-state high-level conditioning:

```text
logs/dance102_single_trajectory_debug/20260618_150520_dance102_h10_hist10_no_language_flow/video_eval_trained_planner_no_language/videos/play/rl-video-step-0.mp4
```

The first trained-planner playback used:

```text
agent.ipmd.skill_commander_use_achieved_state=false
```

That means the high-level planner was conditioned on the reference macro state,
not the robot's achieved macro state.

To get the stricter check, the posthoc script was rerun and produced:

Oracle posthoc video:

```text
logs/dance102_single_trajectory_debug/20260618_150520_dance102_h10_hist10_no_language_flow/video_eval_oracle_hl_skill_posthoc/videos/play/rl-video-step-0.mp4
```

Trained planner video using achieved-state high-level conditioning:

```text
logs/dance102_single_trajectory_debug/20260618_150520_dance102_h10_hist10_no_language_flow/video_eval_trained_planner_no_language_achieved_state/videos/play/rl-video-step-0.mp4
```

Both posthoc videos were also verified with `ffprobe` as `499` frames at `50`
fps, duration `9.98` seconds.

## How To Read This Experiment

The M1 result says the planner can reconstruct oracle `z` well on expert macro
states after language is removed. That is useful but not sufficient for
closed-loop success, because M1 does not roll out the low-level policy and does
not expose planner inputs to achieved-state drift.

The useful comparison is therefore:

- Oracle video: tests whether the low-level policy can execute the oracle
  skill encoder commands.
- Reference-state planner video: tests whether the learned planner reproduces
  oracle commands when conditioned on the expert trajectory state.
- Achieved-state planner video: tests whether the planner remains usable when
  its current-state input comes from the robot's achieved rollout state.

If the oracle video works but the achieved-state planner video fails, the
remaining problem is likely System-1 robustness to achieved-state distribution
shift, not the low-level controller or language embedding. If both planner
videos fail similarly, inspect the planner target geometry and checkpoint
loading path. If only the reference-state planner works, train with more
achieved-state/noisy-state augmentation before reintroducing language.

## Caveats

- Generated logs, videos, checkpoints, and dataset caches are intentionally not
  tracked by git.
- The artifact paths above are local to the run machine/worktree. Use the run
  root and relative layout to locate equivalent artifacts after copying logs.
- The no-language planner is a diagnostic System-1, not the final
  language-conditioned planner.
- The achieved-state posthoc helper initially failed in tmux because plain
  `python` was not on `PATH`; it now falls back to `pixi run -e isaaclab python`
  when needed.
