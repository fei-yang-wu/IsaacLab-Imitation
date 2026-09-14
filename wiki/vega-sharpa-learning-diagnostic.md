# Vega U + Sharpa learning diagnostic

Parked on 2026-09-14 on branch `dev-dex`. No training job is
running for this task. Generated data, checkpoints and videos remain in the
local worktree; source, tests, asset definitions and this report are in Git.

The September 14 checks fixed PPO consistency defects, but none of the four
new 1,000-iteration variants achieved unassisted box success (0/32 each).
These local checks address the failed box endpoint results in
[the object-task evaluation](vega-sharpa-object-task.md). The primary outcome
remains box COM error at the scheduled Reference end, with success at 5 cm.
All local comparisons use seed 42. They are qualification results, not
multi-seed convergence claims.

## Physical cause of the assisted position bias

For the saved 3,000-iteration policy at assistance 0.75, the final 40 control
steps have approximately these mean world-frame forces on the box:

| Contribution | X (N) | Y (N) | Z (N) |
| --- | ---: | ---: | ---: |
| All raw Newton contacts | -1.779 | -0.155 | 3.996 |
| Contacts inferred from momentum balance | -1.933 | -0.069 | 3.884 |
| Virtual controller | 1.934 | 0.066 | -0.940 |
| Gravity | 0 | 0 | -2.943 |

The hands push the box above the Reference and load the virtual spring. The
17-link contact summary misses collisions on virtual knuckle links such as
`*_MCP_VL`. Those excluded links contribute about [-1.212, 0.698, 1.630] N;
adding them recovers the full raw contact sum. The diagnostic does not show
a contact force frame or sign error. No collision geometry was removed.
The raw contact snapshot is from the last physics substep, while momentum
balance uses the whole control step, so small differences are expected.

Records: `outputs/vega_sharpa/contact_balance_diagnostic.{json,npz,log}`.

## PPO consistency defects and fixes

At fixed policy weights, the old KL meter updated running normalization
statistics and sampled actions. In the cold-normalizer diagnostic, 20 calls
increased the sample count from 1 to 15,361 and drove apparent KL as high as
230.68 without an optimizer step. The evaluation-mode control kept KL at
zero. With the mature saved checkpoint, the effect was much smaller; this
does not by itself explain the late learning-rate floor.

RLOpt now measures KL without buffer or RNG changes. Its optional
`ppo.normalizer_update_mode=rollout` freezes statistics through collection
and learning, and updates them once per full rollout. Synthetic startup
shape probes no longer update the statistics. The legacy minibatch schedule
remains the default for other tasks.

The collector and trainer share the normalizer and agree on stored action
distributions before learning. A loss forward with statistics frozen also
leaves the policy unchanged. However, the original initial learning rate
of 0.001 produces a genuinely large first gradient update. The normalization
fix is not a claim that this optimizer overshoot is solved.

A separate fresh-agent diagnostic with initial learning rate 0.00001 and
startup normalizer count 1 produced first-update KL 0.01494 and KL 0.00543
after 20 updates. A loss-only forward gave KL 3.7e-8. This establishes that
the smaller initial rate controls the measured jump; it does not establish
box-task improvement. This probe uses the evaluator's noise-free environment
and reward settings, so it is not a matched training-performance comparison.
Record: `outputs/vega_sharpa/ppo_first_update_low_lr.{json,log}`.

The corrected 50-iteration smoke completed with finite tensors. Actor and
critic counts were both 153,601 = 1 + 50 × 128 × 24. Its first iteration KL
was 206.8; subsequent logged KL values were near 0.01. The logged learning
rate is the last minibatch rate, not every rate used within the iteration.

## Controlled local comparison

The normalization comparison uses 1,000 iterations (3.072M environment
frames), 128 environments, 24-step rollouts, five epochs, four minibatches,
and seed 42. Reference, rewards, curriculum, action settings and optimizer
configuration match the original local run. Only the normalization and KL
consistency changes above differ. The baseline checkpoint is
`logs/vega_sharpa_local_ppo/2026-09-13_16-14-13_run-f3f2c225/models/model_step_3072000.pt`.

The new run writes `logs/vega_sharpa_norm_compare/` and
`outputs/vega_sharpa/norm_compare1000.log`. Evaluation uses all 32
first-start trials, the full 503-step horizon, and separate assistance 0.75
and 0 conditions. No reward or assistance variant is folded into this row.

Two further local rows use the same 1,000-iteration budget and seed. The
reward row changes only the object-keypoint tracking reward schedule to a
constant 1.0, from the original initial zero. This term scores both box
position and orientation through keypoints. The gate row adds only
`ObjectSuccessAssistanceCurriculum` to that reward row. Other reward schedules
retain their original timing. All three rows start from scratch.

The assistance gate starts at 1.0 and follows the original ordered assistance
levels. A reduction requires at least 256 completed episodes and 80% endpoint
position success at the current stage, plus a 2,400-control-step minimum
stage duration (100 PPO iterations). Early terminations are failures. Startup
resets and episodes spanning a stage change do not contribute. Evidence is
cleared after each decision window. The same 5 cm endpoint metric is used.
These gate thresholds are declared local-test choices, not tuned outcomes.
The checkpoint hook stores the gate stage, completed-episode evidence and
stage clock. Restore validates gate settings and assistance levels, then
starts fresh physical episodes; in-flight episodes are not counted again.
Local checkpoint tests and a real 50-iteration continuation pass. Restoring
the 100-iteration gate checkpoint resumed at assistance 0.75, ran exactly
50 additional iterations, and saved 460,800 total frames, control step 3,600,
and actor/critic normalizer counts 460,801. ICE continuation is still a
separate runtime qualification step. Record:
`outputs/vega_sharpa/gate_resume_verification.json`.

The common training invocation is:

```bash
VEGA_SHARPA_REFERENCE_MANIFEST=data/dexmanip/vega_sharpa_gravcomp/local100_grounded/manifest.json \
OMNI_KIT_ACCEPT_EULA=YES pixi run --no-install -e isaaclab python scripts/rlopt/train.py \
  --task Isaac-Imitation-Vega-Sharpa-v0 --algo ppo --num_envs 128 \
  --max_iterations 1000 --seed 42 --headless --assert-kitless \
  physics=newton_mjwarp agent.logger.backend=csv \
  agent.logger.log_dir=logs/vega_sharpa_norm_compare \
  agent.ppo.normalizer_update_mode=rollout
```

For the reward row, set the output directory to
`logs/vega_sharpa_object_reward_compare` and add
`env.curriculum.fixed_timestep.params.reward_weight_schedules.object_keypoints_tracking_exp=1.0`.
For the gate row, keep that reward override, use
`logs/vega_sharpa_object_gate_compare`, and add
`env.curriculum.fixed_timestep.func=isaaclab_imitation.tasks.manager_based.dexmanip.vega_sharpa_curriculum:ObjectSuccessAssistanceCurriculum`.

## Endpoint results

### Additional position-only comparison

Inspection of `object_keypoints_tracking_exp` found six unit-length axis
keypoints and `exp(-squared_error / 0.1)`. With orientation held correct, a
5 cm translation therefore retains `exp(-0.0025 / 0.1) = 0.9753` reward.
The proxy is much less sensitive to the primary position tolerance than its
name suggests, and it also scores orientation.

A fourth, separately recorded 1,000-iteration row adds only a direct current
Reference COM-position reward to the gate row. Its weight is 1.0 and its
position scale is 0.05 m: `exp(-COM_error_squared / 0.05**2)`. It uses the
same COM offset and frame convention as endpoint evaluation. Orientation
remains a separate outcome. The term is disabled by default, and the
comparison enables it with `env.rewards.object_center_tracking_exp.weight=1.0`.
All other gate-row settings stay fixed. This additional row was declared
after inspecting the broad keypoint reward, before observing its outcomes.
Its output directory is `logs/vega_sharpa_object_center_compare`.

### Fixed-horizon scores

Every row uses 32 first-start trials and the full 503-step Reference horizon.
Errors are mean final box COM distance. These are one-seed qualification
results; the 32 worlds are not 32 independent training seeds.

| 1,000-iteration policy | Assistance | Final error (cm) | Success within 5 cm | Angular error (deg) |
| --- | ---: | ---: | ---: | ---: |
| Original baseline | 1 | 8.20 | 0/32 | 6.8 |
| Normalization fixes | 1 | 5.58 | 6/32 | 9.1 |
| Keypoint reward weight 1 | 1 | 5.86 | 0/32 | 5.0 |
| Keypoint reward + success gate | 1 | 6.10 | 0/32 | 6.9 |
| Gate + direct COM reward | 1 | 5.11 | 9/32 | 4.4 |
| Original baseline | 0.75 | 7.78 | 0/32 | 9.2 |
| Normalization fixes | 0.75 | 6.10 | 2/32 | 7.1 |
| Keypoint reward weight 1 | 0.75 | 6.00 | 0/32 | 5.4 |
| Keypoint reward + success gate | 0.75 | 6.62 | 0/32 | 7.3 |
| Gate + direct COM reward | 0.75 | 5.69 | 0/32 | 5.7 |
| Original baseline | 0 | 48.23 | 0/32 | 113.4 |
| Normalization fixes | 0 | 102.81 | 0/32 | 157.0 |
| Keypoint reward weight 1 | 0 | 109.50 | 0/32 | 127.2 |
| Keypoint reward + success gate | 0 | 120.75 | 0/32 | 142.7 |
| Gate + direct COM reward | 0 | 103.31 | 0/32 | 153.4 |

Normalization fixes modestly improve assisted endpoints but worsen the
unassisted mean endpoint error in this local comparison. They do not solve
the manipulation task. All three evaluations reached the fixed deadline,
and policy weights and normalizer buffers remained unchanged.

Record: `outputs/vega_sharpa/object_task_norm_compare.{json,npz,log}`.
Video: `outputs/vega_sharpa/object_task_norm_unassisted.mp4`, fixed trial 0.

Increasing the keypoint reward to 1.0 did not improve primary success:
all 32 trials failed at each assistance level. Record:
`outputs/vega_sharpa/object_task_reward_compare.{json,npz,log}`.

The gate row reduced assistance to 0.75 at iteration 100 and to 0.5 at
iteration 201, then stayed at 0.5 through iteration 1,000. Its last completed
training evidence window had 14.8% endpoint success. Its full-horizon
evaluation also failed all 32 trials at each tested assistance level.
Record: `outputs/vega_sharpa/object_task_gate_compare.{json,npz,log}`.

Adding direct COM tracking improves the assisted mean error relative to the
gate row, but does not achieve unassisted success. It ends at assistance 0.5;
its last training evidence window has 24.9% endpoint success. Full-assistance
evaluation reaches 9/32 successes, while assistance 0.75 and 0 both reach
0/32. Record: `outputs/vega_sharpa/object_task_center_compare.{json,npz,log}`.

No tested variant improves unassisted mean endpoint error over the saved
baseline, and every variant has zero primary unassisted successes. This does
not qualify the policy for the proposed ICE campaign. No ICE training was
submitted. The four local budgets are complete; they were not extended to
search for convergence. Open diagnostic issues include the incomplete hand
contact coverage in the wrench reward and the large initial optimizer step.
The lower-learning-rate probe is not an endpoint-performance comparison.

### Comparison artifacts and videos

`outputs/vega_sharpa/learning_comparison.{json,csv,png,pdf}` contains the
combined result and input-report hashes. The aggregator rejects incomplete
runs, different Reference identities, changed thresholds, unequal starts or
horizons, and unfrozen policies. Each `+` column adds one change to the
preceding variant.

The following videos show the predetermined first world, including the full
unassisted failure. Orange is the recorded actual box, cyan the current
Reference box, and green the final center goal. Each video has 503 frames at
20 Hz. They render recorded Newton states, not a new controller rollout.

- `outputs/vega_sharpa/object_task_norm_unassisted.mp4`
- `outputs/vega_sharpa/object_task_center_unassisted.mp4`
- `outputs/vega_sharpa/object_task_center_assisted.mp4`

Rebuild the combined artifacts from the repository root:

```bash
pixi run --no-install python -m imitation_experiments.evaluation.vega_sharpa_comparison \
  --report Original=outputs/vega_sharpa/object_task_assisted.json \
  --report Original=outputs/vega_sharpa/object_task_unassisted.json \
  --report 'Norm fix=outputs/vega_sharpa/object_task_norm_compare.json' \
  --report +Keypoints=outputs/vega_sharpa/object_task_reward_compare.json \
  --report +Gate=outputs/vega_sharpa/object_task_gate_compare.json \
  --report +COM=outputs/vega_sharpa/object_task_center_compare.json \
  --output outputs/vega_sharpa/learning_comparison
```

## Validation results

The repository's standard RLOpt test set passed 153 tests. Six new regression
tests passed for KL buffer/RNG preservation, mixed module modes, rollout
statistic updates and synthetic collector probes. The unrestricted RLOpt
suite could not collect all tests because of dependency deprecation warnings
treated as errors and a missing external imitation input. That attempt is
recorded in `outputs/vega_sharpa/rlopt_regression.log`; it is not a passing
full-suite result.

The final experiment-library run passed 492 tests with 11 skips. The script
set passed 121 tests with two skips. The gate and direct box-center reward
passed eight tests together. Logs are
`outputs/vega_sharpa/{experiment_tests_learning_final,script_tests_learning,object_curriculum_reward_tests_final}.log`.
Six additional comparison-aggregation tests passed, including rejection of
incomplete and mismatched reports. All four final checkpoints were finite
and had the expected actor/critic normalization count of 3,072,001.
