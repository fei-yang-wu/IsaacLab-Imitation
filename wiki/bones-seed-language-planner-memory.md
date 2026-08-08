# BONES Seed Language Planner Memory

Last updated: 2026-08-06.

This page is a long-term memory snapshot for the BONES Seed language-conditioned
planner experiment. It records the current artifacts, code changes, eval results,
and rerun commands so future work can restart from the same state.

## Next task-active selected ten (2026-08-06)

The next development selection is
`experiments/campaigns/2026-08-05-bones-language10-screen/selected10_taskactive_v2.json`.
It does not overwrite the completed v1 experiment. It retains stoop/pickup,
lift-crate, feeding-birds, slow arc walking, and mosquito-drive-away, then adds
slow straight walking, one-hand carrying while walking backward, a full
open/traverse/close door sequence, injured-torso diagonal walking, and a
one-hand heavy-object transfer from high to low.

The selection requires 1.0 low-level SONIC candidate SR and clear task or
language directability. It also rejects motions with more than 20% hold frames
or an uninterrupted hold longer than 1.50 seconds. Here a hold frame means root
linear speed below 0.03 m/s, root angular speed below 0.10 rad/s, and joint RMS
speed below 0.15 rad/s; only segments at least 0.20 seconds long count. These
are local selection definitions, not SONIC metrics. Selection preparation
enforces the declared limits.

The v2 set averages 2.65% held frames, has a maximum of 8.71%, and no hold
longer than 1.18 seconds. Drinking, fishing, phone typing, greeting, and
surrender are removed. Fishing is deliberately excluded despite robust planner
tracking because it contains a 4.44-second hold and is 38.9% held overall. The
five replacements are low-level qualified but not planner-qualified; build a
fresh manifest, language table, reference arrays, oracle trajectory collection,
and planner before reporting their robustness.

## Current selected-ten baseline (2026-08-05)

The active local language experiment is no longer the historical demo8 setup
below. The selected ten, exact imperative descriptions, source ranks, and
screen metrics are frozen in
`experiments/campaigns/2026-08-05-bones-language10-screen/selected10.json`.
Their ordered v2 manifest, canonical language sidecar, 384-D
`all-MiniLM-L6-v2` embedding table, and compact root-qpos reference arrays live
under `data/bones_seed_language10_v1/`.

Collection is complete-trajectory based. One process uses 1,000 environments:
ten motions times 100 trajectories. References start at frame 0; the oracle
latent plus frozen low-level policy runs until SONIC tracking failure or
reference completion. Foot XYZ and base-height termination are disabled,
pushes are disabled, other domain randomization remains enabled, and policy
actions are deterministic. Saved publications contain causal robot history,
the oracle 256-D latent, current expert/achieved 38-D root-qpos, and a masked
30-frame expert root-qpos lookahead. Termination causes and success are stored
per completed trajectory.

The first shared planner is trained only on the oracle-policy trajectories.
The medium default is 10,000 updates, trajectory-wise 80/20 train/validation,
and deterministic closed-loop evaluation at 2k, 4k, 6k, 8k, and 10k. Report
official SONIC completion SR and MPJPE-L only over successful trajectories;
inspect the milestone curve before deciding whether to add planner-driven
data. The end-to-end 20-update smoke passed on all ten explicit goal bindings;
its SR of 0.4 is a wiring result, not a trained result. Run the current workflow
through `experiments/campaigns/2026-08-05-bones-language10-oracle-pretrain/run.sh`.

The full seed-0 run completed on 2026-08-05 and was optimizer-preservingly
extended to 20k on 2026-08-06. The 2k-through-20k SONIC SR curve is
0.295/0.293/0.307/0.306/0.339/0.305/0.307/0.312/0.322/0.344; success-only
MPJPE-L is 47.79/53.39/52.60/49.08/46.68/44.09/48.98/41.96/45.22/45.48 mm.
The heuristic does not declare a plateau because the curve oscillates, but the
20k gain over 10k is only 0.5 SR points while held-out normalized RMSE improves
from 0.1628 to a best 0.1485. Do not extend again based on offline fit alone.
Fishing, lift-crate, and slow-arc walking remain the dominant successes; four
goals are still at zero at 20k. This is a closed-loop precision/covariate-shift
problem. DAgger was intentionally not started.

The first controlled follow-up is the seven-row H3 latent receding-horizon
study in
`experiments/campaigns/2026-08-06-bones-language10-latent-receding/`.
It predicts three ordered H10 latents at the unchanged 5 Hz rate. It compares
fresh-only, raw exponential, and clipped/gated overlap execution for both
future-publication-frame targets and deliberately stale current-publication-
frame targets, with the 10k H1 checkpoint as control. This isolates temporal
overlap from planner cadence: execute-5/10 Hz is deferred until a matched
low-level tracker is trained or independently qualified.

That comparison is now complete. The strict SR-first order is
future/fresh-only 0.401, future/clipped-gated 0.396, future/exponential 0.394,
current/fresh-only 0.359, H1 0.329, current/clipped-gated 0.317, and
current/exponential 0.283. Future/clipped-gated is the quality Pareto point at
39.72 mm successful MPJPE-L versus 49.82 mm for the 0.401-SR winner. The
five-success total gap is small and the effect is motion-dependent:
clipped/gated harms feeding birds (77 to 2 successes) while helping stoop (9
to 38) and mosquito drive-away (15 to 51). The conclusion is to keep
future-publication supervision, reject stale-frame overlap, and retain
fresh-only plus clipped/gated as the two justified operating points until a
multi-seed or explicitly goal-conditioned gate resolves the tradeoff.

## Latent compositionality analysis (2026-08-06)

The 30-motion latent analysis is complete under
`experiments/campaigns/2026-08-06-bones-latent-compositionality/`. t-SNE is
visualization only; it neither performs clustering nor supplies coordinates to
the metrics. The actual test retrieves latent-nearest windows after excluding
the query's entire source motion, then measures how close those neighbors are
in a translation/yaw-normalized 30-step reference-kinematic space.

A unique publication means one `(motion, reference step)` command. The 100
randomized replicas are averaged before retrieval so they do not count as 100
independent examples. The kinematic distance ratio divides retrieved-neighbor
distance by all-cross-motion random distance; 1.0 is random and lower is
better. Confidence intervals resample whole motions.

The operational 30-motion collection contains 137,801 rows, 1,364 complete
30-step publications, and 3,000 completed trajectories. Its cross-motion
latent/kinematic distance Spearman rho is 0.792. At k=10, latent neighbors have
0.605 times random kinematic distance (95% CI 0.562–0.685) and 27.9x random
kinematic-neighbor recall. Removing panic, big-dog walking, and rock-out—the
three badly tracked motions—leaves the result unchanged at 0.609 (95% CI
0.565–0.691).

The reference-only scale control samples one non-mirrored actor/take from each
of 500 distinct normalized BONES action families and five windows per motion.
Across those 2,500 windows, rho is 0.827 and the k=10 distance ratio is 0.542
(95% CI 0.530–0.553). This is an intrinsic frozen-encoder test; it complements
but does not replace rollout-conditioned geometry.

Semantic evidence is selective. Locomotion, forward motion, and slow
locomotion transfer across clips, while manipulation and object-loaded labels
do not. HDBSCAN clusters on the rollout data align much more with motion
identity (AMI 0.981) than activity (AMI 0.502), and the 500-family test has no
stable HDBSCAN clusters at all. The supported conclusion is a continuous
movement manifold with cross-motion local geometry, not discrete semantic
skill clusters. The robust27 gallery uses median, not best-case, queries and
five distinct neighbor motions so both matches and failures remain visible.

The follow-up phase-traversal test uses the existing 85 exact temporal phases,
not newly inferred t-SNE segments. It maps them into ten frozen shared semantic
regions so differently worded phases can be compared across motions. A phase
centroid is the mean PCA-50 latent for one annotated segment. A return ratio for
`A -> other -> A` is the distance between the two `A` centroids divided by the
mean outward/inward excursion distance; below 1 indicates a latent return.

On the robust 27 motions, all 1,317 unique publications and 72 phases are used.
Cross-motion k=10 semantic-region agreement is 25.0% versus 13.5% matched
random, an improvement of 11.5 points (motion-bootstrap 95% CI 5.6–19.4).
Leave-one-motion-out nearest-region classification reaches 43.4% balanced
accuracy across ten regions. However, only 36.1% of phase centroids have a
same-region phase as their nearest cross-motion semantic competitor. K-means
at ten clusters has semantic AMI 0.232 and purity 0.486; HDBSCAN assigns every
phase centroid to noise. These are local semantic neighborhoods, not clean
global semantic clusters.

The time-ordered trajectory test finds four robust `A -> other -> A` sequences;
three return below ratio 1. Neutral stoop executes forward locomotion -> loaded
locomotion -> forward locomotion with ratio 0.633. Drinking returns to
stationary at 0.250, and one cellphone manipulation return scores 0.319; its
stationary return is a retained failure at 1.839. Including the three
failure-heavy motions weakens held-out balanced accuracy to 31.6% and yields
four of seven returns, while the cross-motion neighbor improvement remains
positive. The campaign's interactive map can switch between t-SNE for local
display and PCA-2 for a linear display; neither projection is used for metrics.

## Historical demo8 baseline (2026-07-07)

## Goal

Demonstrate a language-goal conditioned planner on G1 imitation data:

- System 0: one frozen low-level IPMD bilinear policy.
- System 1: one merged SkillCommander planner.
- Goal selection: language embedding for the intended trajectory/motion.
- Evaluation expectation: choose an exact reference trajectory and matching
  language goal, because the first robot joint/root frame matters a lot.

The historical setup used the BONES Seed demo8 subset. The same low-level
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
