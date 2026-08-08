# Closed-Loop Skill Commander Eval

Last updated: 2026-08-05.

This page documents the local evaluation workflow used to debug the
language/state-conditioned high-level planner. The core question is whether
System 1 can drive System 0 in closed loop:

- System 0: low-level latent-conditioned policy.
- System 1: SkillCommander planner, conditioned on language plus state, or on
  state only for no-language diagnostics.
- System 2: absent in the current stack.

## Current selected-ten collection protocol

For the 2026-08-05 BONES-SEED language10 experiment, the unit of collection is
a complete trajectory, not a planner-row target. Launch all ten named motions
in one process with 100 environments per motion (1,000 total). Every environment
starts its assigned reference at frame 0, receives oracle latent commands, and
executes the frozen low-level policy until an official SONIC tracking failure
or `reference_finished`. Disable foot-position XYZ and `base_too_low`, disable
only the push event, retain startup/reset randomization, and use deterministic
policy actions.

Use `--balanced_trajectories_per_motion 100`, `--trajectory_ranks`,
`--sonic_success_terminations`, `--disable_push_event`,
`--sample_future_window_frames 30`, and `--require_root_qpos_samples` with
`--save_rollout_training_samples`. Each completed-trajectory record includes
termination reasons and tracking success. Each planner publication includes
causal state history, oracle latent target, identity/step keys, current expert
and achieved 38-D root-qpos, and masked `[30,38]` expert root-qpos lookahead.

Pretrain and evaluate the oracle-only planner before collecting planner-driven
data. The medium default is 10,000 updates with checkpoints and closed-loop
evaluation every 2,000 updates. The reusable front door is
`experiments/campaigns/2026-08-05-bones-language10-oracle-pretrain/run.sh`.
After milestone evaluation, `MODE=video` runs the required per-motion
full-horizon pass with every early termination disabled. It uses deterministic
policy inference and `--randomized_no_push`, so startup/reset randomization is
retained while only the interval push is removed; every retained absolute
video path is printed. The unperturbed visualizer default is not the selected-
ten protocol.
The row-budgeted Dance102 and frozen paper Phase-5 recipes below remain useful
historical protocols, but they are not the selected-ten default.

## Selected-ten latent receding-horizon comparison

The 2026-08-06 follow-up reuses the complete-trajectory sample keys and stored
30-frame root-qpos windows; it does not recollect oracle rollouts. At each 5 Hz
publication an H3 planner predicts three ordered 256-D H10 tokens. The command
for the current interval is formed from the aligned candidates predicted at the
current, previous, and second-previous publications. Only `z` is fused; the
low-level ten-step sin/cos phase is regenerated after fusion.

The fixed seven rows are H1, then two H3 supervision frames crossed with three
execution rules. `future_publication` selects the stored oracle latent at
`planner_step + k`, so each token is expressed in the achieved pelvis frame at
its intended future publication. `current_publication` encodes all three
stored H10 root-qpos windows in the current publication frame and is retained
as an explicit stale-frame diagnostic. Execution is `first`, exponential with
decay 0.5, or clipped/gated exponential (one target-standard-deviation residual
clip, normalized RMS gate 2.0, cosine gate 0.5).

All rows keep 5 Hz publication, H10 execution, the frozen root-qpos encoder and
tracker, 10k optimizer updates, explicit matching language goals, deterministic
policy actions, startup/reset randomization, no push event, and 100 trajectories
for each of the same ten motions. The campaign front door is
`experiments/campaigns/2026-08-06-bones-language10-latent-receding/run.sh`.
Do not add the proposed execute-5/10 Hz row using this tracker: its code hold and
phase were trained for ten steps, so that row requires a separately qualified
matched tracker contract.

The completed SR-first ranking is future/fresh-only 0.401,
future/clipped-gated 0.396, future/exponential 0.394, current/fresh-only 0.359,
H1 0.329, current/clipped-gated 0.317, and current/exponential 0.283.
Success-only MPJPE-L for the two leading future rows is 49.82 mm fresh-only and
39.72 mm clipped/gated. Therefore use future-publication supervision. Report
fresh-only as the strict SR winner and clipped/gated as the near-tied quality
Pareto point; do not average latents trained in the stale current-publication
frame.

## Eval Modes

Use these names consistently when reading logs:

| Mode | What runs the robot? | What planner input is scored? | Purpose |
| --- | --- | --- | --- |
| M1 expert-state eval | no simulator rollout | expert/reference macro state | Checks whether the planner learned the oracle `z` target on dataset states. |
| Oracle drive | `agent.ipmd.command_source=hl_skill` | achieved macro state can be scored post-policy | Checks whether System 0 can execute oracle high-level commands and collects achieved-state planner samples. |
| M3 planner drive | `agent.ipmd.command_source=skill_commander` | achieved macro state | True closed-loop System 1 -> System 0 eval. |

M1 is necessary but not sufficient. A planner can score well on expert states
and still fail when its input is the robot's achieved state.

## Closed-Loop Eval Script Semantics

The main diagnostic entrypoint is:

```bash
pixi run -e isaaclab python scripts/rlopt/eval_skill_commander_closed_loop.py
```

Important defaults:

- `--output_dir` writes a separate log entry; do not write eval videos under the
  training checkpoint's log directory.
- `--motion_name` restricts the env to one trajectory/motion name.
- `--max_steps <= 0` runs until the active reference trajectory ends.
- `--video_length 500` records about 10 seconds at 50 Hz while the rollout can
  continue longer for metrics.
- `--keep_time_out` is off by default, so time-limit termination is disabled.
- `--keep_early_terminations` is off by default, so non-reference failure
  terminations are disabled. The default eval stop should be reference end,
  unless `--max_steps` is set.
- `--save_rollout_training_samples` writes achieved-state planner inputs and
  oracle `z` targets under `rollout_training_samples/`.

For true planner-driven M3 eval, the Hydra args must include:

```bash
agent.ipmd.command_source=skill_commander
agent.ipmd.skill_commander_use_achieved_state=true
```

For oracle-drive sample collection, use:

```bash
agent.ipmd.command_source=hl_skill
```

That drives the robot with oracle `z`, while the eval script records the
planner's achieved-state input and target pairs for rollout finetuning.

## Dance102 Artifacts

The Dance102 no-language debug run used:

```bash
RUN_ROOT=logs/dance102_single_trajectory_debug/20260618_150520_dance102_h10_hist10_no_language_flow
TASK=Isaac-Imitation-G1-Latent-v0
ALGO=IPMD_BILINEAR
MANIFEST=data/unitree/manifests/g1_unitree_dance102_manifest.json
DATASET=data/unitree/g1_dance102_hl_diffsr
SKILL_CKPT=$RUN_ROOT/skill_encoder_h10_z256/checkpoints/latest.pt
PLANNER_CKPT=$RUN_ROOT/planner_flow_matching_no_language_hist10/checkpoints/latest.pt
LOW_LEVEL_CKPT=logs/rlopt/ipmd_bilinear/Isaac-Imitation-G1-Latent-v0/2026-06-18_15-10-57/models/model_step_980090880.pt
EVAL_ROOT=logs/planner_robustness/20260621_dance102_eval_recipe
```

Common Dance102 latent overrides:

```bash
COMMON_OVERRIDES=(
  "env.lafan1_manifest_path=$MANIFEST"
  "env.dataset_path=$DATASET"
  "env.refresh_zarr_dataset=false"
  "env.latent_command_dim=258"
  "agent.ipmd.latent_dim=258"
  "agent.ipmd.latent_steps_min=10"
  "agent.ipmd.latent_steps_max=10"
  "agent.ipmd.hl_skill_horizon_steps=10"
  "agent.ipmd.hl_skill_command_mode=z"
  "agent.ipmd.latent_learning.command_phase_mode=sin_cos"
  "agent.ipmd.latent_learning.code_latent_dim=256"
  "agent.ipmd.latent_learning.code_period=10"
)
```

## Step 1: M1 Expert-State Check

Run this before interpreting closed-loop failures:

```bash
pixi run -e isaaclab python scripts/rlopt/eval_skill_commander_m1.py \
  --headless \
  --task "$TASK" \
  --num_envs 256 \
  --checkpoint "$PLANNER_CKPT" \
  --output_dir "$EVAL_ROOT/m1" \
  --batch_size 1024 \
  --eval_batches 4 \
  --splits all \
  --per_trajectory \
  --trajectory_ranks 0 \
  --per_trajectory_batch_size 1024 \
  --per_trajectory_batches 4 \
  --flow_num_inference_steps 16 \
  --flow_inference_noise_std 0.0 \
  "${COMMON_OVERRIDES[@]}"
```

For the Dance102 run, M1 was high (`z_cosine` around `0.986`), so the planner
had learned the expert-state mapping. The later failure was an achieved-state
closed-loop issue, not a basic checkpoint-loading issue.

## Step 2: Oracle-Drive Sample Collection

This checks System 0 with oracle commands and saves achieved-state samples for
finetuning:

```bash
pixi run -e isaaclab python scripts/rlopt/eval_skill_commander_closed_loop.py \
  --headless \
  --task "$TASK" \
  --algorithm "$ALGO" \
  --checkpoint "$LOW_LEVEL_CKPT" \
  --planner_checkpoint "$PLANNER_CKPT" \
  --skill_checkpoint "$SKILL_CKPT" \
  --output_dir "$EVAL_ROOT/oracle_drive_samples" \
  --motion_name dance102 \
  --metric_interval 1 \
  --save_rollout_training_samples \
  --flow_num_inference_steps 16 \
  --flow_inference_noise_std 0.0 \
  "agent.ipmd.command_source=hl_skill" \
  "agent.ipmd.hl_skill_checkpoint_path=$SKILL_CKPT" \
  "${COMMON_OVERRIDES[@]}"
```

Expected useful signals:

- `published_z_vs_target/z_cosine` near 1.0 means oracle command publication is
  aligned with the target `z`.
- `m3/z_cosine` measures what the planner would have predicted from achieved
  state while the robot was oracle-driven.
- Samples land in:

```text
$EVAL_ROOT/oracle_drive_samples/rollout_training_samples/
```

## Step 3: Rollout Finetune The Planner

Finetune the planner on the achieved-state samples collected above:

```bash
pixi run python scripts/rlopt/finetune_skill_commander_rollout.py \
  --checkpoint "$PLANNER_CKPT" \
  --samples_dir "$EVAL_ROOT/oracle_drive_samples/rollout_training_samples" \
  --output_dir "$EVAL_ROOT/planner_rollout_ft_oracle" \
  --num_updates 2000 \
  --batch_size 256 \
  --lr 1.0e-4 \
  --flow_loss_coeff 1.0 \
  --endpoint_loss_coeff 1.0 \
  --flow_num_inference_steps 16 \
  --flow_inference_noise_std 0.0
```

The finetuned checkpoint is:

```text
$EVAL_ROOT/planner_rollout_ft_oracle/checkpoints/latest.pt
```

In the Dance102 debug run, this rollout finetune changed the achieved-state
planner-driven eval from early failure to full-reference survival.

## Step 4: True M3 Planner-Driven Eval

Run the finetuned planner in closed loop, with achieved-state conditioning:

```bash
FT_PLANNER_CKPT=$EVAL_ROOT/planner_rollout_ft_oracle/checkpoints/latest.pt

pixi run -e isaaclab python scripts/rlopt/eval_skill_commander_closed_loop.py \
  --headless \
  --video \
  --video_length 500 \
  --task "$TASK" \
  --algorithm "$ALGO" \
  --checkpoint "$LOW_LEVEL_CKPT" \
  --planner_checkpoint "$FT_PLANNER_CKPT" \
  --skill_checkpoint "$SKILL_CKPT" \
  --output_dir "$EVAL_ROOT/m3_rollout_ft_oracle" \
  --motion_name dance102 \
  --metric_interval 1 \
  --flow_num_inference_steps 16 \
  --flow_inference_noise_std 0.0 \
  "agent.ipmd.command_source=skill_commander" \
  "agent.ipmd.skill_commander_checkpoint_path=$FT_PLANNER_CKPT" \
  "agent.ipmd.skill_commander_embeddings_path=" \
  "agent.ipmd.skill_commander_use_achieved_state=true" \
  "agent.ipmd.skill_commander_flow_num_inference_steps=16" \
  "agent.ipmd.skill_commander_flow_inference_noise_std=0.0" \
  "${COMMON_OVERRIDES[@]}"
```

Video output:

```text
$EVAL_ROOT/m3_rollout_ft_oracle/videos/play/rl-video-step-0.mp4
```

For a 10-second-only smoke, add:

```bash
--max_steps 500
```

Without `--max_steps`, the eval runs until reference end and records the first
`--video_length` frames.

## Optional Baseline M3 Before Finetune

To measure the original planner's achieved-state gap, use the same command as
Step 4 but set:

```bash
--planner_checkpoint "$PLANNER_CKPT"
--output_dir "$EVAL_ROOT/m3_baseline"
"agent.ipmd.skill_commander_checkpoint_path=$PLANNER_CKPT"
```

In the Dance102 debug run, the baseline planner-driven rollout died early and
had low achieved-state `z` cosine. The rollout-finetuned planner survived to
reference end with high M3 cosine.

## Interpreting Outcomes

- Oracle drive works, M1 works, M3 fails: planner is not robust to
  achieved-state distribution shift. Use rollout/DAgger-style finetuning or
  better achieved-state augmentation.
- Oracle drive fails: System 0 or the skill encoder/latent-command wiring is the
  first thing to debug.
- M1 fails: planner training, checkpoint selection, language embedding lookup,
  or target `z` geometry is wrong before closed-loop rollout matters.
- M3 works only with `skill_commander_use_achieved_state=false`: the eval is not
  yet the real closed-loop planner test; it is still using reference-state
  conditioning.

## Batch LAFAN1 Variant

For the full LAFAN1 manifest, use the all-trajectory runner:

```bash
scripts/rlopt/run_lafan1_rollout_ft_all_tmux.sh
```

It performs the same sequence per trajectory: oracle-drive sample collection,
rollout finetune, and achieved-state planner-driven video eval. Override
`RANKS`, `LIMIT`, `OUTPUT_ROOT`, checkpoint paths, or `VIDEO_LENGTH` through
environment variables.
