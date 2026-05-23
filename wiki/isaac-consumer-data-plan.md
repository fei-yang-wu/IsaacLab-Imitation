# Isaac Consumer Data Plan

Last refreshed: 2026-05-13.

This branch should focus on the consumer and robot-integration side of the
imitation pipeline. Action labeling with teacher or teleoperation policies can
run on another machine; this repo should make IsaacLab-Imitation reliably load,
inspect, replay, and train from the resulting small action-labeled G1 datasets.

## Branch Decision

Split the work into two lanes.

Producer lane, owned outside this repo/computer:

- Run SONIC, GR00T, BeyondMimic, or another teacher/teleoperation policy stack.
- Label a small motion subset before attempting a full dataset.
- Export robot observations, explicit action labels, joint order, FPS, robot
  profile, teacher checkpoint, and source-motion provenance.
- For BONES-SEED, use `gear_sonic` preprocessing to produce filtered
  `motion_lib` data instead of reimplementing raw CSV/BVH conversion in
  `iltools`.

Consumer lane, owned by this repo/computer:

- Keep G1 Isaac integration, replay, timing, and reset behavior reliable.
- Make `iltools` consume one small agreed action-labeled G1 format first.
- Treat offline pretraining as flat transition/window sampling.
- Use multi-trajectory cursors only for online IsaacSim reference tracking and
  finetuning.
- Prefer MuJoCo CPU playback for quick motion visualization. Isaac replay is
  still useful for integration, FK/body-state export, and final policy
  finetuning, but it should not block basic dataset visual inspection.

## Data Products

There are two different products. Keep them separate.

Reference motion data:

- BONES-SEED/GR00T `motion_lib`, LAFAN1 NPZ, or other mocap-derived motion.
- Contains root, joints, velocities, and sometimes body states.
- Feeds Isaac online reference tracking through `ParallelTrajectoryManager`.
- Does not automatically provide supervised policy actions.

Action-labeled robot data:

- Teleoperation datasets or teacher-labeled rollouts.
- Contains observations and action targets.
- Feeds offline behavior cloning or representation pretraining through a flat
  transition/window sampler.
- Should not require online multi-trajectory cursors.

The common future direction is that a teacher-labeled mocap rollout and a real
teleop dataset both export into the same action-labeled robot schema.

## Minimal Action-Labeled Contract

Start with a narrow G1 contract rather than a universal data abstraction:

```text
schema: g1_action_labeled_v0
robot_profile: g1_29dof
joint_order: isaaclab_g1_29dof
fps: 30 or 50
action_type: target_joint_position
```

Required tensors:

```text
observation.state.joint_position      (T, 29)
observation.state.joint_velocity      (T, 29)
action.joint_position                 (T, 29)
episode_index                         (T,)
frame_index                           (T,)
```

Optional but useful tensors:

```text
observation.state.root_pos            (T, 3)
observation.state.root_quat_wxyz      (T, 4)
observation.state.body_rotation_6d    (T, 6)
action.root_pos                       (T, 3)
action.root_quat_wxyz                 (T, 4)
action.body_rotation_6d               (T, 6)
```

Metadata must say whether an action is an absolute joint target, default-relative
joint target, PD target, normalized policy action, residual command, or something
else. Do not infer this after export.

## Immediate Milestones

1. Clean up dataset replay and debug tools.
2. Add or reuse a MuJoCo CPU viewer for quick Unitree/BONES-SEED motion
   visualization.
3. Verify Unitree WBT LeRobot desired joint replay at native 30 Hz.
4. Convert a tiny desired-reference segment into NPZ only after replay looks
   right.
5. Load that NPZ through the current LAFAN1-compatible manifest path.
6. Add the first small action-labeled dataset loader in `iltools`.
7. Overfit a tiny BC or reconstruction objective before scaling data.

## Replay Rules

For pure dataset visualization, do not use physics stepping.

The trusted LAFAN1-style conversion path writes root and joint state directly,
then renders:

```text
robot.write_root_state_to_sim(...)
robot.write_joint_state_to_sim(...)
sim.render()
scene.update(sim.get_physics_dt())
```

`scripts/replay_reference.py` also forces `env.replay_only=True`, disables
terminations/rewards by default, and uses the env replay-only path that writes
reference state directly instead of calling Isaac physics stepping.

For Unitree WBT LeRobot, use the desired command field when the goal is to
visualize the label/reference sequence:

```text
action.robot_q_desired
```

Use the measured state only when debugging real robot tracking quality:

```text
observation.state.robot_q_current
```

The Unitree dataset card defines both 36-D fields as:

```text
root position (x, y, z)
root quaternion (w, x, y, z)
29 robot joint positions
```

The dataset preview shows timestamp increments of about `0.033333`, so native
playback/export should use `30 Hz`. Do not resample to 50 Hz while debugging the
visual semantics of the source motion.

## Timing Rules

For direct replay/export:

```bash
--fps 30 --output_fps 30
```

For an NPZ manifest consumed by the G1 env:

- Store `fps=30` in the NPZ.
- Put `input_fps=30` in the manifest entry.
- Let `ImitationG1LafanTrackEnvCfg.sync_control_rate_to_manifest=True` set the
  env control rate from the manifest.
- With the default preferred physics rate, 30 Hz control becomes 240 Hz physics
  with `decimation=8`.

Do not root-height-align the Unitree LeRobot trajectory when the target is exact
dataset visualization. Root z alignment is a debugging convenience, not a data
preserving transform.

## Current Debug Hypothesis

If the generated Unitree WBT video looks like the robot jumps or falls while the
intended label is “squat, pick up a pillow, walk forward, squat, place it on a
sofa,” first check these before changing the loader:

- The replay field should be `action.robot_q_desired`, not necessarily
  `observation.state.robot_q_current`.
- Root z alignment should be `none`.
- FPS should stay at the native 30 Hz.
- Quaternion order should remain `wxyz`.
- The zarr/NPZ cache should be rebuilt after changing the replay field or root
  alignment.

Only after those checks fail should we suspect joint order or Isaac articulation
state writing.

## Source Links

- Unitree WBT pillow dataset:
  <https://huggingface.co/datasets/unitreerobotics/G1_WBT_Brainco_Pickup_Pillow>
- GR00T Whole-Body Control:
  <https://github.com/NVlabs/GR00T-WholeBodyControl>
- GR00T training guide:
  <https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/docs/source/user_guide/training.md>
