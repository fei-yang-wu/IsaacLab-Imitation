# Vega U + Sharpa local PPO learning check

Date: 2026-09-13. Status: the local run is complete; unassisted regression confirmed. This is a one-seed
training-readiness check on the corrected, cropped/slowed Reference.

The user deferred ICE planning until the local curve can be assessed.
The proposed later ICE settings were accepted: baseline PPO, seed 42,
10 billion environment frames, W&B group `vega-sharpa-ppo`. No cluster
submission is authorized or active from this task. Direct SSH was unavailable during this run. The user subsequently configured
the `ice` alias through a jump host and login now succeeds; remote job preflight
remains a separate step.


The primary evaluation has since been changed, at the user’s request, to
[final object-task success](vega-sharpa-object-task.md). The tracking-termination
results below are retained as diagnostics and are not the primary task score.

## Frozen local protocol

- Task: `Isaac-Imitation-Vega-Sharpa-v0`, Newton/MJWarp, Kit-free.
- Reference: `data/dexmanip/vega_sharpa_gravcomp/local100_grounded/manifest.json`.
- Fixed base z: 0.18694717 m, derived from the mesh bottom.
- Source frames: 603–764 inclusive, retimed 3x; 484 frames at 20 Hz.
- Seed: 42. Parallel environments: 128.
- PPO budget: 3,000 rollout iterations = 9,216,000 environment frames.
- Rollouts: 24 steps; five optimizer epochs and four minibatches per rollout.
- Checkpoints: every 250 iterations. CSV metrics: every iteration.
- Original reward and assistance schedule retained. First assistance
  reduction is at 2,000 iterations, from 1.0 to 0.75; object tracking weight
  changes from 0.0 to 0.1. Hand tracking weights are 0.25 throughout.

The physical fixtures passed at 128 environments before this run, including
base-floor alignment, contacts, PD, wrench frames, support and gravity.
Record: `outputs/vega_sharpa/local_ppo_runtime128.json`.

```bash
VEGA_SHARPA_REFERENCE_MANIFEST=data/dexmanip/vega_sharpa_gravcomp/local100_grounded/manifest.json \
OMNI_KIT_ACCEPT_EULA=YES pixi run --no-install -e isaaclab python scripts/rlopt/train.py \
  --task Isaac-Imitation-Vega-Sharpa-v0 --algo ppo \
  --num_envs 128 --max_iterations 3000 --seed 42 --headless --assert-kitless \
  physics=newton_mjwarp agent.logger.backend=csv \
  agent.logger.log_dir=logs/vega_sharpa_local_ppo \
  agent.save_interval_iterations=250 agent.save_interval=24576000
```

Run directory:
`logs/vega_sharpa_local_ppo/2026-09-13_16-14-13_run-f3f2c225/`.
The run's `params/agent.yaml`, `params/env.yaml`, and `command.txt` record
the resolved settings. The frame-sized save interval above is stated at
the recipe's reference batch size and is restated to 768,000 frames locally.

## Assessment

Assess the curve using reward per control step, episode return/duration,
contact reward, reference-end share, tracking errors and PPO stability.
Reward values before and after a curriculum change have different weights
and assistance. Do not interpret their difference as a matched comparison.
Logged tracking metrics are reset summaries; fixed-start evaluation supplies
proper episode means measured before reset and Reference advance. Wrist
errors use the desired hand pose relative to the live object. They therefore
include object/grasp displacement and must not be read as fixed-world arm
forward-kinematics error.

The zero-residual baseline follows Reference joint targets without learned
joint corrections. Evaluate it and frozen PPO checkpoints on the same
32-world first-frame and evenly spaced feasible-start pools, with exact
assistance scales 0.75 and 0.0. Disable observation corruption and keep the
same startup material seed. Hold the evaluation reward weights fixed.

Measure each world's first episode only, including its terminal transition.
A completion reaches the end of the prepared Reference without a simultaneous
wrist/object tracking termination. The object termination limits remain
0.2 m and 0.7 rad; either wrist is limited to 0.2 m. Assistance is set to the
declared value before every step, including the initial reset hold.

Policy state is restored strictly, including its observation-normalizer
buffers. Both policy parameters and buffers must remain unchanged during
evaluation. Retained trajectories are actual Newton rollout states; their
MuJoCo-rendered videos must be labelled with the assistance level.

```bash
VEGA_SHARPA_REFERENCE_MANIFEST=data/dexmanip/vega_sharpa_gravcomp/local100_grounded/manifest.json \
OMNI_KIT_ACCEPT_EULA=YES pixi run --no-install -e isaaclab python scripts/rlopt/evaluate_vega_sharpa.py \
  --protocol tracking --num-envs 32 --assistance 0.75 0 --starts first grid \
  --checkpoints logs/vega_sharpa_local_ppo/2026-09-13_16-14-13_run-f3f2c225/models/model_step_9216000.pt \
  --output outputs/vega_sharpa/local_ppo_eval_final.json physics=newton_mjwarp

pixi run --no-install python -m imitation_experiments.audit.vega_sharpa_training \
  --run-dir logs/vega_sharpa_local_ppo/2026-09-13_16-14-13_run-f3f2c225 \
  --num-envs 128 --expected-iterations 3000 --require-complete \
  --output outputs/vega_sharpa/local_ppo_curves.png
```

## Results

This one-seed local run completed all 3,000 iterations (9,216,000 frames,
60,000 optimizer updates). All 12 saved checkpoints and all optimization
records are finite. Training took 2,785.63 s; evaluation work ran concurrently
for part of that interval, so this is not an isolated throughput benchmark.

The curve shows early contact learning followed by a plateau. Under unchanged
full assistance, reward per control step rose from 0.105 (first 100 iterations)
to 0.246 (iterations 1,800–1,999). Its final 100-iteration mean was 0.257 after
assistance fell to 0.75 and object tracking acquired weight 0.1; this later
number uses a different reward/assistance regime. Contact reward rose, while
logged object and object-relative wrist errors also increased. The adaptive
learning rate was at its minimum, 1e-5, in every logged iteration; the cause needs a
separate diagnostic rather than an assumption.

Fixed-start results below use one training seed and 32 simulated trials per
case. “First” is the first frame of the prepared crop, not the original full
capture. “Grid” spans the fixed feasible-start pool and has varying remaining
horizons; its completions are not full-capture successes.

| Checkpoint | Unassisted first-start mean duration | Full crop completed from first start | Remaining crop completed from grid starts |
| --- | ---: | ---: | ---: |
| Zero residuals | 0.25 s | 0/32 | 1/32 |
| 250 iterations | 2.44 s | 0/32 | 3/32 |
| 1,000 iterations | 3.40 s | 0/32 | 6/32 |
| 2,000 iterations | 3.31 s | 0/32 | 5/32 |
| 3,000 iterations | 0.61 s | 0/32 | 3/32 |

A fresh same-process recheck reproduced the regression: the 1,000-iteration
checkpoint averaged 3.70 s from the first frame without assistance, versus
0.57 s for the 3,000-iteration checkpoint (one seed, 32 trials each; 0/32 full
crop completions at both). All first-start failures were object-pose tracking
terminations. With 0.75 assistance, both PPO and zero residuals completed
32/32 first-start and grid trials. That assisted outcome is not evidence of
an unassisted expert.

The after-transition restart also passed: the 2,000-iteration checkpoint
continued for two updates of the rollout loop to 6,150,144 frames and 40,040
optimizer updates, with curriculum control step 48,048 and scale 0.75.

## Retained evidence

- `outputs/vega_sharpa/local_ppo_curves.png` and `.pdf`: completed training curves.
- `outputs/vega_sharpa/local_ppo_curves.json`: all scalar checks and window statistics.
- `outputs/vega_sharpa/local_ppo_training_audit.json`: resolved-config hashes, all checkpoint hashes/counters and restart audit.
- `outputs/vega_sharpa/local_ppo_assessment.json`: machine-readable comparison, protocol qualifications and source records.
- `outputs/vega_sharpa/local_ppo_eval_early.json`: baseline and 250/1,000-iteration evaluation.
- `outputs/vega_sharpa/local_ppo_eval_2000.json`: the transition checkpoint.
- `outputs/vega_sharpa/local_ppo_eval_final.json`: baseline and final checkpoint.
- `outputs/vega_sharpa/local_ppo_eval_regression_check.json`: fresh 1,000 versus 3,000 unassisted recheck.
- Each evaluation has a matching `.npz` with recorded first-world trajectories.
- `outputs/vega_sharpa/local_ppo1000_unassisted.mp4`: early unassisted policy, ending at failure.
- `outputs/vega_sharpa/local_ppo3000_unassisted.mp4`: final unassisted policy, ending at failure.
- `outputs/vega_sharpa/local_ppo3000_assisted075.mp4`: final policy with 0.75 object assistance.

Final checkpoint:
`logs/vega_sharpa_local_ppo/2026-09-13_16-14-13_run-f3f2c225/models/model_step_9216000.pt`.
The earlier checkpoints are retained for diagnosis, including the stronger
early unassisted checkpoint at 1,000 iterations.

## Assessment and next local work

The training implementation is functioning, but the current result does not
justify a large ICE run. Higher assisted reward did not preserve the earlier
unassisted behavior. Inspect the reward/assistance balance and the initial
grasp failure, and separately check KL/normalization behavior behind the
minimum learning rate. Keep any resulting change in a new recorded run so
this baseline remains interpretable. ICE planning remains deferred.

Validation: experiment-library tests 485 passed / 11 skipped; script tests
121 passed / 2 skipped; targeted evaluator/audit tests and Ruff checks passed.
