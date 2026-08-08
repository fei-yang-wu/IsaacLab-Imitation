---
name: sonic-success-eval
description: Evaluate IsaacLab-Imitation low-level G1 checkpoints with the official SONIC motion-success criterion and success-only MPJPE-L. Use when the user asks for SONIC success rate, paper-compatible SR, push-disabled or foot-position-disabled evaluation, checkpoint comparison by completed motions, or MPJPE-L computed only on successful trajectories.
---

# SONIC success evaluation

Run a deterministic-policy, randomized-environment, push-disabled, full-clip
checkpoint evaluation with SONIC's released evaluation thresholds. Keep this
pass distinct from task-strict qualification and the non-terminating
diagnostic.

Read `wiki/sonic-success-evaluation.md` before launching. Treat its pinned
upstream links and result-field definitions as the detailed source of truth.

## Reconstruct the checkpoint contract

Inspect the training launch, saved environment/agent config, and checkpoint
metadata. Preserve all of the following:

- task id and physics backend;
- algorithm and `--agent_entry_point`;
- actor command width and encoder interface;
- exact encoder checkpoint, horizon, latent mode, phase mode, and
  frozen/finetuned state;
- dataset or reference-array path plus content identity;
- tracked body order.

Invoke `g1-encoder-interface` before pairing an encoder when the interface is
not already proven. Never rebuild an old checkpoint from today's task defaults.
Change only the evaluation settings below.

## Apply the SONIC criterion at launch

Use `imitation_experiments.lowlevel.evaluate_checkpoint` with deterministic
policy actions (`--action_sampling mode`), `--randomization no_push`, start
frame 0, and a fixed sequential assignment. The `no_push` profile retains
startup and reset randomization while removing the interval push. Pass
`env.events.push_robot=null` explicitly as an auditable second guard. Apply
these overrides exactly:

```text
env.events.push_robot=null
env.terminations.anchor_pos.params.threshold=0.25
env.terminations.anchor_pos.params.down_threshold=0.25
env.terminations.anchor_ori.params.threshold=1.0
env.terminations.ee_body_pos.params.threshold=0.25
env.terminations.ee_body_pos.params.down_threshold=0.25
env.terminations.foot_pos_xyz=null
env.terminations.base_too_low=null
```

These mean:

- fail when pelvis Z error exceeds 0.25 m;
- fail when any ankle or wrist Z error exceeds 0.25 m;
- fail when full pelvis orientation error exceeds 1 rad;
- do not fail on foot XYZ error;
- retain startup and reset randomization;
- apply no interval velocity push;
- use the deterministic/mode policy action.

Setting both adaptive position thresholds to 0.25 disables the crouching
relaxation without swapping the task's predicate. The orientation predicate
compares squared angular error, so a threshold of 1.0 is an angle limit of
1 rad.

## Run the complete motions

Use enough `--steps` for the longest evaluated clip. Prefer the maximum clip
length plus one; a deliberately generous cap is fine because the evaluator
stops when all environments are done. Never report an SR from a run whose
`stop_reason` is `max_steps` or whose `done_rate` is below 1.

For the standard large local pass, use 4096 environments. When comparing
checkpoints, keep `num_envs`, seed, reset schedule, start frame, data identity,
and trajectory-rank assignment identical.

Use the command template in `wiki/sonic-success-evaluation.md`; add the exact
checkpoint-specific agent and data overrides after the fixed CLI arguments.

## Validate before reporting

Require all of these:

- `aggregate.num_evaluated_envs` equals the requested environment count;
- `aggregate.done_rate == 1.0`;
- `aggregate.time_out_rate == 0.0`;
- `metadata.action_sampling == "mode"`;
- `metadata.randomization_profile == "no_push"`;
- `metadata.randomization_kept.startup == true`;
- `metadata.randomization_kept.reset == true`;
- `metadata.randomization_kept.push == false`;
- `metadata.push_perturbation.enabled == false`;
- `stop_reason == "all_envs_done"`;
- no non-SONIC failure term is active;
- checkpoint comparisons have the same ordered trajectory-rank hash.

Report:

- SR from `aggregate.completed_tracking_success_rate`;
- numerator from `successful_metrics.tracking_mpjpe_mm.num_successful_envs`;
- denominator from `aggregate.num_evaluated_envs`;
- MPJPE-L from `successful_metrics.tracking_mpjpe_mm.mean`.

Do not report `aggregate.tracking_success_rate` as SR unless every evaluated
environment finished. It intentionally exposes survivor status and can count
unfinished clips under a short cap. A clip that reaches `reference_finished`
on the same step as a failure remains a failure.

State that the result is criterion-compatible but not directly comparable to
SONIC's paper number when the motion dataset or split differs.

## Preserve the other evaluation passes

This pass does not replace:

- task-strict oracle qualification with the checkpoint's frozen task terms;
- the full-horizon diagnostic with all early terminations disabled and its
  required retained video.

Label artifacts explicitly as `sonic_eval` to prevent accidental aggregation
with either protocol.
