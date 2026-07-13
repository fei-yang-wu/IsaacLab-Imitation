# LAFAN1 Motion Tracking Evaluation

This file records how to rerun the LAFAN1 motion tracking experiment.

## Goal

Run 40 LAFAN1 motions and compare:

- Oracle latent tracking
- Base planner tracking
- Finetuned planner tracking
- EE chunk planner baseline
- Whole-body chunk planner baseline

Finetune setup:
    1. Collect oracle rollout
    2. Finetune Planner
    3. Eval on this motion

So each motion has its own finetuned planner. They all start from the same base
planner checkpoint.

## Main Script

```bash
experiments/interface_baselines/run_lafan1_motion_tracking_evaluation.sh
```

Default setup:


| config        | value                                           |
| ------------- | ----------------------------------------------- |
| task          | `Isaac-Imitation-G1-Latent-v0`                  |
| algo          | `IPMD`                                          |
| horizon       | `10`                                            |
| state history | `9`                                             |
| z dim         | `256`                                           |
| planner       | `flow_matching`                                 |
| manifest      | `data/lafan1/manifests/g1_lafan1_manifest.json` |




## Data

Expected paths:

```bash
MANIFEST_PATH=data/lafan1/manifests/g1_lafan1_manifest.json
DATASET_PATH=data/lafan1/g1_hl_diffsr
```



## Run With Existing Checkpoints

```bash
RUN_BASE_PIPELINE=0 \
SKILL_CHECKPOINT=/path/to/skill_encoder/checkpoints/latest.pt \
PLANNER_CHECKPOINT=/path/to/planner/checkpoints/latest.pt \
LOW_LEVEL_CHECKPOINT=/path/to/low_level/model_step_x.pt \
RANKS=all \
experiments/interface_baselines/run_lafan1_motion_tracking_evaluation.sh
```

Quick dry run:

```bash
DRY_RUN=1 \
RUN_BASE_PIPELINE=0 \
RUN_ORACLE_RECON_EVAL=0 \
RUN_BASE_PLANNER_PREDICT_EVAL=0 \
RUN_ORACLE_LL_EVAL=0 \
RUN_PLANNER_ROLLOUT_FINETUNE=0 \
RUN_FINETUNED_PLANNER_PREDICT_EVAL=0 \
RUN_FINETUNED_PLANNER_LL_EVAL=0 \
RUN_HAND_DESIGNED_BASELINES=0 \
RANKS=0 \
LIMIT=1 \
SKILL_CHECKPOINT=/tmp/fake \
PLANNER_CHECKPOINT=/tmp/fake \
LOW_LEVEL_CHECKPOINT=/tmp/fake \
experiments/interface_baselines/run_lafan1_motion_tracking_evaluation.sh
```



## Run Base Pipeline First

Leave this on:

```bash
RUN_BASE_PIPELINE=1
```

The script will call:

```bash
scripts/rlopt/run_lafan1_no_language_pipeline.sh
```



## Per-Motion Outputs

Each motion gets one folder:

```text
RUN_ROOT/per_trajectory/rank_<rank>_<motion_name>/
```

Important outputs:


| path                      | meaning                        |
| ------------------------- | ------------------------------ |
| manifest_single.json      | one-motion manifest            |
| oracle_recon_eval         | encoder reconstruction eval    |
| planner_predict_base      | base planner offline eval      |
| oracle_ll_eval            | oracle latent tracking         |
| oracle_ll_collect         | rollout samples for finetune   |
| planner_rollout_ft_single | finetuned planner checkpoint   |
| planner_predict_finetuned | finetuned planner offline eval |
| planner_ll_finetuned      | finetuned planner tracking     |




## Baselines

Baselines are off by default:

```bash
RUN_HAND_DESIGNED_BASELINES=0
```

Run EE and whole-body separately.

EE:

```bash
RUN_HAND_DESIGNED_BASELINES=1 \
BASELINE_INTERFACES=ee_trajectory \
EE_TRAJECTORY_CHECKPOINT=/path/to/ee_chunk_ipmd.pt \
experiments/interface_baselines/run_lafan1_motion_tracking_evaluation.sh
```

Whole-body:

```bash
RUN_HAND_DESIGNED_BASELINES=1 \
BASELINE_INTERFACES=full_body_trajectory \
FULL_BODY_TRAJECTORY_CHECKPOINT=/path/to/full_body_chunk_ipmd.pt \
experiments/interface_baselines/run_lafan1_motion_tracking_evaluation.sh
```

The baseline horizon defaults to `HORIZON_STEPS`.

## Cluster Mode

```bash
MODE=lafan1-motion-tracking \
experiments/interface_baselines/submit_cluster_interface_baselines.sh
```

Useful variables:

```bash
RUN_BASE_PIPELINE=0
SKILL_CHECKPOINT=/path/to/skill.pt
PLANNER_CHECKPOINT=/path/to/planner.pt
LOW_LEVEL_CHECKPOINT=/path/to/low_level.pt
RANKS=all
LIMIT=0
RUN_HAND_DESIGNED_BASELINES=0
```



## Metrics

Main metrics:


| metric                                | meaning                     |
| ------------------------------------- | --------------------------- |
| tracking_success_rate               | success rate                |
| tracking_mpjpe_mm                   | root-relative body MPJPE    |
| root_height_error_m                 | root height error           |
| root_ori_error_rad                  | root orientation error      |
| ee_pos_error_m                      | end-effector position error |
| joint_pos_rmse_rad                  | joint position RMSE         |
| joint_vel_rmse_radps                | joint velocity RMSE         |
| tracking_velocity_distance_mps      | velocity distance           |
| tracking_acceleration_distance_mps2 | acceleration distance       |


Success threshold:
For all steps:
    1. root height error <= 0.25 m
    2. root orientation error <= 1.0 rad



## Current Results


| Method                             | N   | Success | MPJPE-L (mm) | Root H (m) | Root Ori (rad) | EE Pos (m) | Joint Pos (rad) | Joint Vel (rad/s) | Vel Dist (m/s) | Acc Dist (m/s^2) |
| ---------------------------------- | --- | ------- | ------------ | ---------- | -------------- | ---------- | --------------- | ----------------- | -------------- | ---------------- |
| Oracle recon                       | 40  | 0.100   | 216.33       | 0.215      | 0.994          | 3.095      | 0.369           | 1.681             | 0.802          | 8.781            |
| Planner base closed-loop           | 40  | 0.000   | 459.45       | 0.588      | 2.122          | 3.462      | 0.735           | 3.008             | 1.179          | 17.271           |
| Planner finetuned closed-loop      | 40  | 0.025   | 342.52       | 0.412      | 1.574          | 2.710      | 0.491           | 1.961             | 0.888          | 10.116           |
| EE chunk planner base              | 40  | 0.675   | 60.80        | 0.028      | 0.217          | 0.744      | 0.222           | 1.286             | 0.545          | 6.487            |
| EE chunk planner finetuned         | 40  | 0.825   | 57.20        | 0.026      | 0.219          | 0.651      | 0.221           | 1.200             | 0.515          | 6.148            |
| Whole-body chunk planner base      | 40  | 0.825   | 54.91        | 0.020      | 0.182          | 0.473      | 0.196           | 1.346             | 0.520          | 6.575            |
| Whole-body chunk planner finetuned | 40  | 0.875   | 46.79        | 0.022      | 0.181          | 0.512      | 0.182           | 1.107             | 0.477          | 6.301            |


