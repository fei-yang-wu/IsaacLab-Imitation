# Vega U + Sharpa: object-centered task success

This is the primary task-outcome evaluation requested by the user. Reward,
robot tracking and episode duration remain diagnostics; they do not define
object success. The tests below use one training seed and 32 fixed-start
simulated trials per condition on the current cropped/slowed Reference.

## Primary metric

Let c(T) be the actual box center of mass at the Reference's scheduled end,
and c_ref(T) the center computed from the final Reference box pose using the
same local COM offset. The endpoint position error is

`e_position = ||c(T) - c_ref(T)||_2`.

Position-only success is `e_position <= 0.05 m` at that end time. Every trial
is in the denominator. The reported 2 cm and 10 cm success rates show how the
answer depends on tolerance. The 5 cm default is a starting evaluation
criterion, not a measured hardware tolerance.

The box root-origin position error is also retained, so the choice of center
is explicit. Quaternion angular error and pose success at 5 cm / 0.35 rad
(about 20 degrees) are secondary. Orientation is not a hidden gate on the
primary position-only rate. There is no additional lift-height requirement.

The current goal center is approximately [-0.13193, -0.01732, 1.08176] m.
It is 14.9 cm from the starting center. This scores the end of our prepared
segment, not the complete original ARCTIC manipulation task.

## Fixed-horizon evaluation

`--protocol object-task` is the evaluator default. It disables the intermediate wrist and object tracking
terminations during evaluation and runs to the end of the Reference. A policy
can deviate and recover; a dropped box is still measured at the same deadline.
The existing training rewards, actions and termination settings are unchanged.

For this Reference, the rollout is 503 control steps at 20 Hz, including the
initial hold. Metrics are captured after physics and before the terminal
reset. The evaluator checks that the command's terminal success log agrees
with the actual pre-reset endpoint score. A separate runtime check verifies
the logged mean position error as well. Policy parameters and normalizer
buffers remain frozen. Observation corruption is off; startup material seed
is 42. Assisted and unassisted results are reported separately.

Future Vega training logs now expose:

- `Metrics/sharpa_reference/object_task_success`: position-only success at the Reference end.
- `Metrics/sharpa_reference/final_box_position_error_m`: final distance to the final box-center target, refreshed before reset.
- `Metrics/sharpa_reference/final_box_orientation_error_rad`: terminal box orientation error.
- `Metrics/sharpa_reference/object_pose_success`: secondary position-and-orientation score.
- `Metrics/sharpa_reference/object_task_reached_end`: distinguishes an endpoint from an early training termination.

The Vega `task_success` alias now uses this position-only endpoint definition.
Its older inherited lift condition was incompatible with this lowering/rotation
crop. Existing CSV files have not been rewritten and their old `task_success`
column must not be interpreted as the new endpoint score. Training episodes
still stopped early by optimization guards count as failures in this log;
checkpoint evaluation supplies the full-horizon object-only test.

## Results

One training seed, 32 trials per condition, first frame of the same prepared
Reference. Errors are means across all trials, measured at the fixed end.

| Policy | Assistance | Final center error | Success at 5 cm | Success at 10 cm | Angular error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Zero residuals | 0 | 109.25 cm | 0/32 | 0/32 | 53.6 deg |
| PPO 250 | 0 | 99.95 cm | 0/32 | 3/32 | 52.2 deg |
| PPO 1,000 | 0 | 48.23 cm | 0/32 | 0/32 | 113.4 deg |
| PPO 2,000 | 0 | 84.22 cm | 0/32 | 0/32 | 116.7 deg |
| PPO 3,000 | 0 | 115.27 cm | 0/32 | 0/32 | 72.4 deg |
| Zero residuals | 1 | 1.16 cm | 32/32 | 32/32 | 0.2 deg |
| Zero residuals | 0.75 | 1.96 cm | 32/32 | 32/32 | 0.2 deg |
| PPO 1,000 | 1 | 8.20 cm | 0/32 | 32/32 | 6.8 deg |
| PPO 1,000 | 0.75 | 7.78 cm | 0/32 | 32/32 | 9.2 deg |
| PPO 3,000 | 1 | 11.89 cm | 0/32 | 3/32 | 2.7 deg |
| PPO 3,000 | 0.75 | 12.20 cm | 0/32 | 1/32 | 3.1 deg |

The tested PPO checkpoints do not meet the 5 cm endpoint criterion. At their
training assistance levels, PPO 1,000 finishes about 8.2 cm from the goal and
PPO 3,000 about 12.2 cm away. Without assistance, the PPO 3,000 box ends about
115 cm from the goal on average. The full-horizon videos show the assisted
position bias and the unassisted drop directly.

## Videos and records

The overlays use orange for the actual box, cyan for the current Reference
box, and green for the final center goal region. They display both current
box-center tracking error and distance to the final goal. These are recorded
Newton checkpoint rollouts rendered with the same robot/object geometry;
they are not reconstructed training rewards or kinematic Reference replay.
One fixed trial (world 0) is shown in each video.

- `outputs/vega_sharpa/object_task_1000_training_assistance.mp4`: PPO 1,000, assistance 1.0.
- `outputs/vega_sharpa/object_task_3000_training_assistance.mp4`: PPO 3,000, assistance 0.75.
- `outputs/vega_sharpa/object_task_3000_unassisted.mp4`: PPO 3,000, assistance 0.0, full horizon.
- `outputs/vega_sharpa/object_task_unassisted.json` and `.npz`: all five unassisted cases and their trajectories.
- `outputs/vega_sharpa/object_task_assisted.json` and `.npz`: six assisted cases and their trajectories.
- `outputs/vega_sharpa/object_task_summary.json`: combined endpoint table, tolerance sensitivity and input hashes.
- `outputs/vega_sharpa/object_task_scorecard.png` and `.pdf`: object success rates and final error distributions.
- `outputs/vega_sharpa/object_task_logging_check.json`: terminal log versus pre-reset physical-state verification.

```bash
export VEGA_SHARPA_REFERENCE_MANIFEST="$PWD/data/dexmanip/vega_sharpa_gravcomp/local100_grounded/manifest.json"
OMNI_KIT_ACCEPT_EULA=YES pixi run --no-install -e isaaclab python scripts/rlopt/evaluate_vega_sharpa.py \
  --protocol object-task --num-envs 32 --starts first --assistance 0 \
  --goal-position-tolerance-m 0.05 --goal-orientation-tolerance-rad 0.35 \
  --checkpoints logs/vega_sharpa_local_ppo/2026-09-13_16-14-13_run-f3f2c225/models/model_step_9216000.pt \
  --output outputs/vega_sharpa/object_task_check.json physics=newton_mjwarp

MUJOCO_GL=egl pixi run --no-install -e isaaclab python scripts/viz/render_vega_sharpa_reference.py \
  --reference "$VEGA_SHARPA_REFERENCE_MANIFEST" \
  --rollout outputs/vega_sharpa/object_task_check.npz \
  --trajectory-key model_step_9216000__first__assistance_0 \
  --reference-overlay --label 'PPO 3000 | assistance 0' \
  --output outputs/vega_sharpa/object_task_check.mp4
```

`--protocol tracking` retains the earlier termination-based diagnostic for
comparison. It is not the primary object-task success definition.

## ICE

The configured `ice` SSH alias now works through its jump host. A read-only
login returned `login-ice-gnr-1.pace.gatech.edu`. No job was submitted by this
metric/evaluation task. Asset staging, remote preflight and a reviewed
submission plan remain separate steps if cluster training is resumed.
