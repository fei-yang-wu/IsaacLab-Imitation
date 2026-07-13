# BONES Seed Language Planner Memory

Last updated: 2026-07-07.

This page is a long-term memory snapshot for the BONES Seed language-conditioned
planner experiment. It records the current artifacts, code changes, eval results,
and rerun commands so future work can restart from the same state.

## Goal

Demonstrate a language-goal conditioned planner on G1 imitation data:

- System 0: one frozen low-level IPMD bilinear policy.
- System 1: one merged SkillCommander planner.
- Goal selection: language embedding for the intended trajectory/motion.
- Evaluation expectation: choose an exact reference trajectory and matching
  language goal, because the first robot joint/root frame matters a lot.

The current working setup uses the BONES Seed demo8 subset. The same low-level
policy and same merged planner can run all 8 selected motions when the trajectory
and language goal are paired.

## Data And Language Artifacts

BONES Seed demo8 manifest:

```text
data/bones_seed/manifests/g1_bones_seed_language_demo_8_manifest.json
```

Cached G1 trajectory dataset:

```text
data/bones_seed/g1_hl_diffsr
```

Language embedding table:

```text
data/bones_seed/language/g1_bones_seed_language_demo_8_minilm_goal_embeddings.pt
```

The 8 motion ranks are:

| Rank | Motion |
| ---: | --- |
| 0 | `Neutral_stoop_down_001_A057` |
| 1 | `big_heavy_one_hand_front_high_to_front_low_R_001_A524` |
| 2 | `big_heavy_one_hand_front_low_to_front_high_R_001_A524` |
| 3 | `big_light_two_hands_pick_up_front_medium_R_001_A509` |
| 4 | `drinking_standing_mug_R_001_A282` |
| 5 | `inside_door_handle_left_side_open_walk_close_behind_R_001_A513` |
| 6 | `inside_door_handle_right_side_open_walk_turn_close_R_001_A514` |
| 7 | `read_book_both_hands_sitting_R_001_A456` |

## Checkpoints

Low-level 1B IPMD bilinear policy:

```text
logs/rlopt/ipmd_bilinear/Isaac-Imitation-G1-Latent-v0/2026-07-06_22-30-42/models/model_step_1000046592.pt
```

The 1B run completed with the final log line approximately:

```text
iter=10173/10173 | frames=1000046592/1000046592 | r_step=0.0562 | ep_len=217.51 | r_ep=12.4794 | pi_loss=-0.0065 | fps=40610.7065
```

Fresh skill encoder:

```text
logs/bones_seed_language/from_scratch_1b_demo8_20260706_220758/skill_encoder_h25_z256/checkpoints/latest.pt
```

Base language commander:

```text
logs/bones_seed_language/from_scratch_1b_demo8_20260706_220758/commander_contrastive_5000/checkpoints/latest.pt
```

Merged rollout-finetuned planner:

```text
logs/bones_seed_language/from_scratch_1b_demo8_20260706_220758/m3_rollout_ft_merged_20260707_080013/planner_rollout_ft_merged/checkpoints/latest.pt
```

Merged planner finetune used 24 oracle rollouts, 8 motions x 3 seeds, merged
into 11,559 achieved-state planner samples. Final finetune metrics at update
10,000 were:

| Metric | Value |
| --- | ---: |
| `eval/z_cosine` | 0.9770 |
| `eval/z_mse` | 0.0208 |
| `eval/z_hat_rms` | 0.6213 |
| `eval/z_target_rms` | 0.6402 |

## Code Changes Needed For This State

### RLOpt submodule

`RLOpt/rlopt/agent/skill_commander.py` needed runtime fixes for
`FrozenSkillCommanderSampler`. The class intentionally does not call
`super().__init__`, so it must call shared helper functions directly:

- `_resolve_device(device, env)`
- `_build_diffsr(config, state_dim, device)`
- `_validate_macro_batch(...)` as `_validate_hl_macro_batch`

It also supports forced language-goal overrides:

- `goal_name`
- `goal_rank`

`RLOpt/rlopt/agent/ipmd/ipmd.py` exposes those overrides in Hydra config:

```text
agent.ipmd.skill_commander_goal_name=<motion_name>
agent.ipmd.skill_commander_goal_rank=<rank>
```

Set only one of `goal_name` or `goal_rank`. Leave both unset for the default
behavior, where language comes from the active trajectory rank.

### Top-level eval script

`scripts/rlopt/eval_skill_commander_closed_loop.py` now prints a traceback
around planner/agent initialization before re-raising. This was useful because
Isaac shutdown could otherwise hide the real Python exception behind Hydra's
generic error message.

## Full M3 Eval Results

Metric-only full M3 eval completed for all 8 motions:

```text
logs/bones_seed_language/from_scratch_1b_demo8_20260706_220758/m3_rollout_ft_merged_20260707_080013/eval_finetuned_per_motion
```

Aggregate files:

```text
logs/bones_seed_language/from_scratch_1b_demo8_20260706_220758/m3_rollout_ft_merged_20260707_080013/eval_finetuned_per_motion/aggregate_metrics.json
logs/bones_seed_language/from_scratch_1b_demo8_20260706_220758/m3_rollout_ft_merged_20260707_080013/eval_finetuned_per_motion/metrics_table.csv
```

Summary:

| Metric | Unweighted mean | Step-weighted mean |
| --- | ---: | ---: |
| `m3_z_cosine` | 0.7707 | 0.8150 |
| `m3_z_mse` | 0.2361 | 0.1884 |
| `published_z_vs_target_z_cosine` | 0.6686 | 0.7476 |
| `published_z_vs_target_z_mse` | 0.3253 | 0.2486 |
| `return_sum_mean` | 34.9934 | 48.4300 |
| `done_rate` | 1.0 | 1.0 |

Weak cases to inspect:

- Rank 5, `inside_door_handle_left_side_open_walk_close_behind_R_001_A513`:
  `m3_z_cosine` around 0.17.
- Rank 6, `inside_door_handle_right_side_open_walk_turn_close_R_001_A514`:
  `m3_z_cosine` around 0.60 to 0.65 depending on eval run.

Strong cases:

- Ranks 1, 3, 4, 7 generally have high M3 cosine, about 0.92 to 0.96.

## Paired Video Eval

Paired videos were generated with the exact reference motion and matching
language goal forced by name. This is the recommended visual quality check.

Video output root:

```text
logs/bones_seed_language/from_scratch_1b_demo8_20260706_220758/m3_rollout_ft_merged_20260707_080013/eval_videos_paired
```

Index:

```text
logs/bones_seed_language/from_scratch_1b_demo8_20260706_220758/m3_rollout_ft_merged_20260707_080013/eval_videos_paired/video_index.md
logs/bones_seed_language/from_scratch_1b_demo8_20260706_220758/m3_rollout_ft_merged_20260707_080013/eval_videos_paired/video_index.json
logs/bones_seed_language/from_scratch_1b_demo8_20260706_220758/m3_rollout_ft_merged_20260707_080013/eval_videos_paired/video_runs.json
```

All 8 paired video runs returned code 0 and each has:

```text
rank_XXXX_<motion>/videos/play/rl-video-step-0.mp4
```

Visual eval metrics from `video_index.json`:

| Rank | Motion | Steps | Return | M3 Cosine |
| ---: | --- | ---: | ---: | ---: |
| 0 | `Neutral_stoop_down_001_A057` | 340 | 19.170 | 0.671 |
| 1 | `big_heavy_one_hand_front_high_to_front_low_R_001_A524` | 419 | 31.521 | 0.957 |
| 2 | `big_heavy_one_hand_front_low_to_front_high_R_001_A524` | 510 | 38.392 | 0.877 |
| 3 | `big_light_two_hands_pick_up_front_medium_R_001_A509` | 168 | 11.771 | 0.919 |
| 4 | `drinking_standing_mug_R_001_A282` | 551 | 41.505 | 0.959 |
| 5 | `inside_door_handle_left_side_open_walk_close_behind_R_001_A513` | 393 | 26.228 | 0.175 |
| 6 | `inside_door_handle_right_side_open_walk_turn_close_R_001_A514` | 316 | 19.484 | 0.603 |
| 7 | `read_book_both_hands_sitting_R_001_A456` | 1156 | 91.656 | 0.958 |

## Forced Language Intervention Test

The forced-language hook was checked with the same reference motion and same
checkpoints, changing only the language input.

Output:

```text
logs/bones_seed_language/from_scratch_1b_demo8_20260706_220758/m3_rollout_ft_merged_20260707_080013/eval_language_override/comparison.json
```

Reference fixed to `Neutral_stoop_down_001_A057`:

| Forced goal | Return | `published_z_vs_m3/z_cosine` | `m3/z_cosine` |
| --- | ---: | ---: | ---: |
| `Neutral_stoop_down_001_A057` | 6.488 | 0.543 | 0.749 |
| `read_book_both_hands_sitting_R_001_A456` | 5.517 | 0.173 | 0.448 |

This confirms that changing only the language embedding changes the planner's
published command stream. For actual quality demos, use paired trajectory and
language goals, because the first frame is part of the control problem.

## Rerun Command Template

Use this template for one paired video eval. Replace `MOTION` with one of the
motion names above.

```bash
MOTION=Neutral_stoop_down_001_A057
RUN_ROOT=logs/bones_seed_language/from_scratch_1b_demo8_20260706_220758/m3_rollout_ft_merged_20260707_080013

OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y TORCHDYNAMO_DISABLE=1 \
pixi run -e isaaclab python scripts/rlopt/eval_skill_commander_closed_loop.py \
  --headless \
  --device cuda:0 \
  --num_envs 1 \
  --task Isaac-Imitation-G1-Latent-v0 \
  --algorithm IPMD_BILINEAR \
  --seed 0 \
  --checkpoint logs/rlopt/ipmd_bilinear/Isaac-Imitation-G1-Latent-v0/2026-07-06_22-30-42/models/model_step_1000046592.pt \
  --skill_checkpoint logs/bones_seed_language/from_scratch_1b_demo8_20260706_220758/skill_encoder_h25_z256/checkpoints/latest.pt \
  --planner_checkpoint "$RUN_ROOT/planner_rollout_ft_merged/checkpoints/latest.pt" \
  --output_dir "$RUN_ROOT/eval_videos_paired/${MOTION}" \
  --motion_name "$MOTION" \
  --metric_interval 1 \
  --flow_num_inference_steps 16 \
  --flow_inference_noise_std 0.0 \
  --video \
  --video_length 1300 \
  "agent.ipmd.skill_commander_goal_name=$MOTION" \
  "agent.ipmd.command_source=skill_commander" \
  "agent.ipmd.skill_commander_checkpoint_path=$RUN_ROOT/planner_rollout_ft_merged/checkpoints/latest.pt" \
  "agent.ipmd.skill_commander_embeddings_path=" \
  "agent.ipmd.skill_commander_flow_num_inference_steps=16" \
  "agent.ipmd.skill_commander_flow_inference_noise_std=0.0" \
  "agent.ipmd.skill_commander_use_achieved_state=true" \
  "agent.ipmd.hl_skill_finetune_enabled=false" \
  "env.lafan1_manifest_path=data/bones_seed/manifests/g1_bones_seed_language_demo_8_manifest.json" \
  "env.dataset_path=data/bones_seed/g1_hl_diffsr" \
  "env.refresh_zarr_dataset=false" \
  "env.latent_command_dim=258" \
  "agent.ipmd.latent_dim=258" \
  "agent.ipmd.hl_skill_horizon_steps=25" \
  "agent.ipmd.hl_skill_command_mode=z" \
  "agent.ipmd.latent_steps_min=25" \
  "agent.ipmd.latent_steps_max=25" \
  "agent.ipmd.latent_learning.command_phase_mode=sin_cos" \
  "agent.ipmd.latent_learning.code_latent_dim=256" \
  "agent.ipmd.latent_learning.code_period=25" \
  "agent.ipmd.reward_loss_coeff=0.0" \
  "agent.ipmd.reward_l2_coeff=0.0" \
  "agent.ipmd.reward_grad_penalty_coeff=0.0" \
  "agent.ipmd.reward_logit_reg_coeff=0.0" \
  "agent.ipmd.reward_param_weight_decay_coeff=0.0"
```

To test language sensitivity while holding the reference fixed, replace the
goal override with either:

```text
agent.ipmd.skill_commander_goal_rank=7
```

or:

```text
agent.ipmd.skill_commander_goal_name=read_book_both_hands_sitting_R_001_A456
```

## Validation Already Run

Syntax check:

```bash
pixi run python -m py_compile \
  RLOpt/rlopt/agent/ipmd/ipmd.py \
  RLOpt/rlopt/agent/skill_commander.py \
  scripts/rlopt/eval_skill_commander_closed_loop.py
```

Runtime checks:

- 8/8 metric-only M3 evals returned code 0.
- 8/8 paired video evals returned code 0.
- Forced-language intervention eval returned code 0 for both matching and
  mismatched language goals.

## Next Work

- Visually inspect the paired videos, especially ranks 5 and 6.
- If ranks 5 and 6 look poor, collect more door-like data or oversample them in
  merged planner finetuning.
- Consider a small script wrapper for paired video generation so the current
  one-off tmux/Python loop becomes a reusable command.
- When the RLOpt changes are ready, commit/push the submodule change and update
  the top-level submodule pointer.
