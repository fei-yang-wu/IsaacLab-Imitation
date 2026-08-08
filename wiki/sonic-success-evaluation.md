# SONIC-Compatible Success Evaluation

This page defines the checkpoint-evaluation pass used when reporting SONIC's
motion-level success rate and success-only MPJPE-L. Status: 2026-08-05.

This is an **external metric-compatibility pass**, not the task's strict oracle
qualification protocol. It also does not replace the required full-horizon,
non-terminating diagnostic. Keep all three artifacts separately labeled.

## Upstream definition

Pinned upstream source: NVlabs `GR00T-WholeBodyControl` commit
`aa263a8a4a71ab30e93c3289988479fc114b0c97`.

- [SONIC paper](https://nvlabs.github.io/GEAR-SONIC/static/pdf/sonic_paper.pdf)
- [released evaluation termination config](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/aa263a8a4a71ab30e93c3289988479fc114b0c97/gear_sonic/config/manager_env/terminations/tracking/eval.yaml)
- [released evaluation callback](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/aa263a8a4a71ab30e93c3289988479fc114b0c97/gear_sonic/trl/callbacks/im_eval_callback.py)

SONIC's evaluation config has only three failure terms plus motion completion:

| failure term | released threshold | local implementation |
|---|---:|---|
| pelvis/root Z error | 0.25 m | `anchor_pos`, with both adaptive thresholds set to 0.25 |
| either ankle or wrist Z error | 0.25 m | `ee_body_pos`, with both adaptive thresholds set to 0.25 |
| full pelvis/root orientation error | 1 rad | `anchor_ori` threshold 1.0 |

There is no foot-position-XYZ termination in the released evaluation config.
Set `foot_pos_xyz=null`. Also keep `base_too_low` disabled: it is not one of the
released evaluation failure terms.

A motion is successful only if it completes its reference without firing a
failure term. The released callback computes `1 - mean(terminated)` over the
motions and micro-averages tracking metrics over successful trajectories: sum
the successful frame-level metric values and divide by their total frame count.

## Keep randomization but disable the push

This pass retains startup and reset randomization, evaluates the policy's mode
action deterministically, and removes only the interval velocity push. Use both
push safeguards:

```text
--randomization no_push
--action_sampling mode
env.events.push_robot=null
```

The `no_push` profile preserves asset-property randomization at startup and the
randomized reference-state reset, but removes `push_robot`. The explicit Hydra
override makes the push decision visible in the resolved launch config and
protects the method if randomization-profile wiring changes later. Before
accepting a result, require:

```text
metadata.action_sampling == "mode"
metadata.randomization_profile == "no_push"
metadata.randomization_kept.startup == true
metadata.randomization_kept.reset == true
metadata.randomization_kept.push == false
metadata.push_perturbation.enabled == false
```

Do not compare this result with an evaluation that disables all randomization
or retains pushes.

## Relationship to this repo's other passes

| pass | early termination | push | purpose | primary outputs |
|---|---|---|---|---|
| task-strict qualification | checkpoint task's frozen terms | frozen internal protocol | internal oracle gate | strict success and termination causes |
| SONIC-compatible | released 0.25 / 1.0 / 0.25 terms; no foot XYZ | startup + reset randomization, no push | external SR definition | completed-motion SR and success-only MPJPE-L |
| full-horizon diagnostic | all early terms disabled | match the protocol being diagnosed | failure-inclusive tracking and visual diagnosis | full-horizon MPJPE plus retained video |

Do not substitute one for another. In particular, the SONIC-compatible pass is
looser than the current v2 training task and therefore cannot satisfy an
internal strict-qualification gate.

## Reconstruct the launch contract first

Before launching, read the checkpoint's actual training command and saved task
and agent configs. Preserve its:

- task id and physics backend;
- algorithm and agent entry point;
- actor command width;
- encoder checkpoint and encoder interface;
- latent horizon, code period, command mode, and phase mode;
- dataset/reference-array path and content id;
- tracked body names and order.

Do not pair an old checkpoint with the latest task defaults. For G1 latent
checkpoints, invoke the `g1-encoder-interface` skill if the encoder contract is
not already proven. The intended evaluation domain retains startup and reset
randomization, removes the push, and uses deterministic/mode policy actions.

## Launch template

Run from the repository root. This example is the `root_qpos`, 258-wide,
horizon-10 contract used by the 2026-08-05 rollout-24 evaluation. Replace paths
and adjust agent overrides only when the checkpoint's recorded config requires
it.

```bash
CHECKPOINT=/absolute/path/to/model_step_N.pt
ENCODER=/absolute/path/to/encoder.pt
REFERENCE_ARRAYS=/absolute/path/to/reference_arrays
PERSIST_ID=dataset-content-id
OUTPUT_JSON=logs/eval/model_step_N_sonic_eval.json

pixi run -e isaaclab python -u \
  -m imitation_experiments.lowlevel.evaluate_checkpoint \
  --task Isaac-Imitation-G1-v2 --algo IPMD \
  --checkpoint "$CHECKPOINT" \
  --agent_entry_point rlopt_ipmd_tuned_cfg_entry_point \
  --randomization no_push --action_sampling mode \
  --num_envs 4096 --steps 10000 --seed 0 \
  --reference_start_frame 0 --reset_schedule sequential \
  --output_json "$OUTPUT_JSON" --headless \
  physics=newton_mjwarp \
  env.events.push_robot=null \
  env.data.manifest=null \
  env.data.reference_arrays_dir="$REFERENCE_ARRAYS" \
  env.data.persist_id="$PERSIST_ID" \
  env.data.reference_arrays_warm_workers=8 \
  env.data.reference_prefetch_mode=next \
  env.data.macro_cache_device=cuda:0 \
  env.data.runtime_cache_body_names=[pelvis,left_hip_roll_link,left_knee_link,left_ankle_roll_link,right_hip_roll_link,right_knee_link,right_ankle_roll_link,torso_link,left_shoulder_roll_link,left_elbow_link,left_wrist_yaw_link,right_shoulder_roll_link,right_elbow_link,right_wrist_yaw_link] \
  env.command_interface.actor.dim=258 \
  env.expert_macro_state_terms=[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b] \
  env.terminations.anchor_pos.params.threshold=0.25 \
  env.terminations.anchor_pos.params.down_threshold=0.25 \
  env.terminations.anchor_ori.params.threshold=1.0 \
  env.terminations.ee_body_pos.params.threshold=0.25 \
  env.terminations.ee_body_pos.params.down_threshold=0.25 \
  env.terminations.foot_pos_xyz=null \
  env.terminations.base_too_low=null \
  agent.ipmd.latent_dim=258 \
  agent.ipmd.command_source=hl_skill \
  agent.ipmd.hl_skill_checkpoint_path="$ENCODER" \
  agent.ipmd.hl_skill_horizon_steps=10 \
  agent.ipmd.hl_skill_command_mode=z \
  agent.ipmd.latent_steps_min=10 \
  agent.ipmd.latent_steps_max=10 \
  agent.ipmd.latent_learning.code_period=10 \
  agent.ipmd.latent_learning.command_phase_mode=sin_cos \
  agent.ipmd.latent_learning.code_latent_dim=256 \
  agent.ipmd.hl_skill_finetune_enabled=false
```

`--steps` is a safety ceiling, not the evaluation horizon. Set it above the
longest assigned clip. The evaluator stops early when every environment is
done. A run that reaches `max_steps` has not proven motion-level success and
must not be reported as SONIC SR.

For a manifest plus Zarr dataset, replace the reference-array block with the
matching `--motion_manifest` and `--dataset_path` flags. Keep the content hashes
and ordered motion assignment fixed across checkpoints.

## Read and validate the result

Use these JSON fields:

| quantity | field |
|---|---|
| SONIC SR | `aggregate.completed_tracking_success_rate` |
| successful motions | `successful_metrics.tracking_mpjpe_mm.num_successful_envs` |
| total motions | `aggregate.num_evaluated_envs` |
| success-only MPJPE-L, mm | `successful_metrics.tracking_mpjpe_mm.mean` |

The evaluator defines completed success as
`reference_finished AND NOT(any tracking failure)`. This deliberately differs
from `aggregate.tracking_success_rate`, which represents no observed failure
and can include unfinished survivors under a short cap.

Before reporting, require:

```text
aggregate.done_rate == 1.0
aggregate.time_out_rate == 0.0
metadata.action_sampling == "mode"
metadata.randomization_profile == "no_push"
metadata.randomization_kept.startup == true
metadata.randomization_kept.reset == true
metadata.randomization_kept.push == false
metadata.push_perturbation.enabled == false
stop_reason == "all_envs_done"
steps_run < max_steps
```

Check `aggregate.termination_cause_env_counts` and confirm that only
`anchor_pos`, `anchor_ori`, `ee_body_pos`, and `reference_finished` fired. A
motion that finishes on the same step as a failure is still a failure.

For checkpoint comparisons, hash the ordered trajectory ranks and require the
same value for every output:

```bash
jq -c '[.per_environment[].trajectory_rank]' "$OUTPUT_JSON" | sha256sum
```

## Validated rollout-24 example

The gamma-0.97, rollout-24 run was evaluated on 2026-08-05 with
deterministic/mode actions, Newton, startup and reset randomization, no push,
4096 sequentially assigned BONES-SEED motions, and the exact frozen encoder.
Every pass ended with `all_envs_done` at step 5610 or 5896. The ordered-rank
hash was
`786ef6775930c34179b774cb215e233c3f7b2bb32ef46bb6fc660206324e8285`.

| training frames | successful motions | SONIC SR | success-only MPJPE-L |
|---:|---:|---:|---:|
| 2.00B | 3686 / 4096 | 89.99% | 26.16 mm |
| 2.50B | 3716 / 4096 | 90.72% | 25.49 mm |
| 3.00B | **3737 / 4096** | **91.24%** | 25.53 mm |
| **3.50B** | **3737 / 4096** | **91.24%** | **24.90 mm** |

The 3.00B and 3.50B checkpoints tie on SR. Select 3.50B because it lowers
MPJPE-L by 0.64 mm at the same successful-motion count. These are
criterion-compatible numbers on this fixed BONES-SEED subset, not a direct
comparison with SONIC's paper dataset or split. The older 91.77--92.85% table
used `--randomization none` and is a separate nominal-tracking diagnostic, not
this protocol.

## Artifact naming

Use an output root or label containing `sonic_eval`. Retain:

- result JSON and run log;
- checkpoint and encoder SHA-256 hashes;
- dataset path/content id;
- task and agent entry point;
- ordered-rank hash;
- exact launch command or resolved config.
