# Local Data Layout

This directory is for local and generated motion assets. Most of it remains git-ignored on purpose.

Tracked manifests and templates live under `source/isaaclab_imitation/isaaclab_imitation/manifests/`:

- `g1_lafan1_manifest.template.json`: tracked template for a full local G1 LAFAN1 manifest

Expected local paths:

- `data/lafan1/raw/g1/`: downloaded or source CSV motions
- `data/lafan1/npz/g1/`: converted G1 NPZ motions
- `data/lafan1/manifests/g1_lafan1_manifest.json`: full local manifest
- `data/lafan1/manifests/g1_debug_manifest.json`: smaller local subset manifest

Common flows:

1. Download the Hugging Face G1 dataset and prepare NPZ plus a full manifest:

```bash
./scripts/data/download_g1_lafan1_data.sh
```

Equivalent lower-level Python command:

```bash
pixi run -e isaaclab python scripts/data/setup_lafan1_dataset.py \
    --prepare-npz --headless
```

For the G1 CSV set, you can auto-trim the common arms-up alignment pose while
building HF-ready NPZ files with:

```bash
pixi run -e isaaclab python scripts/data/setup_lafan1_dataset.py \
    --prepare-npz --headless \
    --auto_trim_mode g1_shoulder_roll
```

2. If NPZ files already exist, regenerate the full manifest:

```bash
pixi run python scripts/data/write_lafan1_npz_manifest.py \
    --npz_dir data/lafan1/npz/g1 \
    --manifest_path data/lafan1/manifests/g1_lafan1_manifest.json
```

3. If you want to hand-edit a manifest instead of generating one, copy the tracked template:

```bash
mkdir -p data/lafan1/manifests
cp source/isaaclab_imitation/isaaclab_imitation/manifests/g1_lafan1_manifest.template.json \
   data/lafan1/manifests/g1_lafan1_manifest.json
```

After copying, replace the placeholder motion names and paths with your local NPZ files.

If you already have raw CSV files plus existing NPZ files and only want to add
per-motion trim ranges to the manifest, use:

```bash
pixi run -e isaaclab python scripts/data/prepare_lafan1_from_csv.py \
    --csv_dir data/lafan1/raw/g1 \
    --npz_dir data/lafan1/npz/g1 \
    --manifest_path data/lafan1/manifests/g1_lafan1_manifest.json \
    --recursive \
    --assume_npz_exists \
    --auto_trim_mode g1_shoulder_roll
```

For the generic `Isaac-Imitation-G1-LafanTrack-v0` task, pass your manifest explicitly with:

```bash
env.lafan1_manifest_path=./data/lafan1/manifests/g1_lafan1_manifest.json
```

## Vega-Wuji Dexterous References

`Isaac-Imitation-Vega-Wuji-v0` accepts only the ILTools dexterous Reference
schema. A legacy joint-only NPZ is not a task input. The task loads one NPZ or
one hash-bound JSON Manifest with `iltools.core.load_dexterous_reference_set`.
The environment has no bundled Reference default. Set `env.reference_path` or
`DEXMANIP_VEGA_WUJI_REFERENCE_PATH` to an input that you prepared.

Each NPZ contains:

- `qpos` and `qvel`: `[T, 59]`, with named Vega-Wuji joints.
- `fixed_root_pose_w`: `[7]`.
- Left and right wrist poses: `[T, 7]`, plus the exact MuJoCo site name for
  each pose field.
- Object names, rigid asset paths, positive radii, poses `[T, O, 7]`, and
  world-frame linear-then-angular twists `[T, O, 6]`.
- Object and support-surface scales, plus a SHA-256 digest for each local
  scene asset.
- Hash-bound collision dependencies for wrapper assets. In particular, a URDF
  hash alone is insufficient: every referenced collision mesh is declared and
  verified separately.
- The five named distal hand-frame poses for each hand: `[T, 5, 7]`.
- Optional static support-surface names, USD paths, and poses.
- Padded contacts with fixed hand sides and slots. Positions and normals use
  `[T, S, C, 3]`; object indices and active masks use `[T, S, C]`.
- Typed `ScenePhysics`: object mass, local COM, COM-frame diagonal inertia,
  object/support friction, and restitution.
- Typed `TrainingQualification`: runtime flags, real contact-geometry
  provenance, and a full-frame signed-distance audit. The training boundary
  caps the declared penetration tolerance at 1 mm.

All file poses are world-frame XYZ plus WXYZ. Relative asset paths resolve
from the NPZ directory. The environment converts quaternions to Isaac Lab
XYZW only when it writes live asset state.

All motions in one Manifest must use the same fixed root, object names and
assets, object radii, support scene, hand-frame names, and contact-link layout.
A JSON Manifest must declare `model` and `model_sha256`. The environment checks
that hash against the MJCF that Newton imports for both replay and training.
Training requires this Manifest plus a passing typed qualification on every
motion. A direct NPZ is accepted only by the explicit inspection/state-lock
replay path; it does not bind a robot model and is never a training input.
The scene importer accepts rigid USD assets and rigid URDFs only. A URDF can
have fixed joints, but it must merge to one body. Articulated object motion is
not in this schema.

For Vega/Wuji, `right_wrist_pose_w` is the world pose of `right_palm` and
`left_wrist_pose_w` is the world pose of `left_palm`. A Reference and its
Manifest must declare those two frame names. In the vendored MJCF, each palm
site is an identity frame on `r_mount` or `l_mount`. The standalone asset
audit and the environment startup gate check this invariant. Newton control
uses the two mount bodies. `R_ee` and `L_ee` are parent bodies with a fixed
pi rotation relative to the palm frames and are not wrist tracking frames.

### Retarget and export

Use `iltools.retarget.MujocoDualHandRetargeter` for the two wrist SE(3) tasks
and the bounded Vega/Wuji joint solution. Finger inputs to this stage must
already be robot-space targets. Set its wrist sites to `left_palm` and
`right_palm`; the retargeter records those names in its output. Then use:

- `iltools.retarget.dexterous_reference_from_trajectory`
- `iltools.core.save_dexterous_reference_npz`
- `iltools.core.create_dexterous_reference_manifest`

Manifest creation verifies every local object and support-surface asset against
the SHA-256 stored in its Reference. It fails before writing if an asset is
missing, unhashed, or has changed. Strict verification parses URDF collision
mesh URIs and exact-matches their typed dependency hashes. Without OpenUSD/pxr,
support USD must be self-contained ASCII USDA; external-reference or binary
USD is rejected rather than incompletely certified. Dependency-aware training
qualification uses schema v2; v1 remains loadable for inspection but cannot
pass the training gate. The task-specific writer also compiles the
MJCF, checks all 59 actuator-order joint limits, rejects non-finite/out-of-limit
positions, and checks stored velocities against the 20 Hz finite difference.
It verifies typed training qualification by default. The explicit
`--allow-unqualified-inspection` escape writes an inspection-only Manifest that
the environment will reject for training. The task-specific CLI is a thin
wrapper over those ILTools APIs:

```bash
pixi run -e isaaclab python scripts/data/write_vega_wuji_reference_manifest.py \
    --npz data/dexmanip/motions/first.npz \
    --npz data/dexmanip/motions/second.npz \
    --output data/dexmanip/manifests/vega_wuji_manifest.json \
    --model source/isaaclab_imitation/isaaclab_imitation/assets/vega_wuji/vega_u_wuji_v2_beta1_with_mount.xml
```

Each input must be a full ILTools dexterous Reference NPZ, not a joint-only
archive. Inputs for this task must already be sampled at 20 Hz; resample during
retargeting before saving them.

The exporter accepts source data in the `robot` or `world` coordinate frame.
It applies `fixed_root_pose_w` when the source frame is `robot`, so every saved
`*_w` array is in the world frame. Contact producers must pad contacts to one
fixed slot count.

Object dynamics use `object_twists_w[T, O, 6]` for the object pose/root origin,
in world-frame linear XYZ then angular XYZ order. Retargeters may provide this
array explicitly. If it is
absent, ILTools derives it deterministically from `object_poses_w` at the
Reference FPS with quaternion-sign-safe shortest rotations and stable
one-sided endpoint velocities. The policy observes both the live and desired
13-value pose-plus-twist state for each object; older pose-only NPZ files are
therefore upgraded deterministically when loaded.
Isaac's rigid-object velocity writer accepts COM velocity, so reset converts
the stored root-origin linear velocity with the qualified object COM offset;
live policy state converts back through `root_link_vel_w` and stays in the
stored convention.

The older `retarget_ego_exo4d.py` and `retarget_egodex.py` scripts can still
produce intermediate robot joint targets. Their joint-only output must be
augmented with wrist, object, hand-frame, scene, and contact data through
ILTools before this task can load it.

### Policy and scene contract

The policy action has 59 values in live actuator order, one for each Vega/Wuji
joint. The adapter applies `tanh`, uses scale 0.05 for `Lift` and 0.15 for each
other joint, filters the residual with an exponential moving average, adds it
to the action-aligned Reference joint position, and clips the result to the
live soft limits. The policy also observes each hand's processed 27-value
target: seven arm-joint targets followed by 20 finger-joint targets. These two
terms keep the former per-hand feedback width but do not add policy action
dimensions. The older 52-value wrist-Jacobian adapter remains available as a
non-default compatibility class.

Transition conditioning includes current robot `[q, qdot]`, desired robot
`[q_ref, qdot_ref]`, current object pose-plus-twist, and desired object
pose-plus-twist. Each object state is 13 values in
`[XYZ, WXYZ, linear-XYZ, angular-XYZ]` order. Adding live object dynamics and
the six target twist values increases the single-object policy vector from 590
to 609 values (`+19` per object relative to the pose-only transition contract).

The positive imitation objective combines whole-robot position and velocity,
object position/orientation/twist, wrist/fingertip, and contact/wrench
tracking. Regularizers cover action rate and magnitude, joint limits, velocity,
and torque. Terminations cover whole-robot/wrist/object deviation, non-finite
state, reference horizon, and excessive robot-support normal force. The support
force term is a Newton force proxy rather than a penetration-depth estimate;
its default 5/50/75 N shaping/saturation/termination thresholds must be
calibrated against qualified scene traces before a production run.

One zero-size virtual object action term is added for each rigid object. It
applies a Newton-compatible pose-PD wrench. The released fixed curriculum
(from the upstream CHORD recipe) reduces this virtual control and increases
the object tracking reward.
The contact-wrench support calculation uses the released command friction
coefficient of 0.1.

Reset writes the robot and every object from one synchronized Reference frame.
Finger random scaling is off by default. Object root-origin twist is converted
to Isaac's COM-velocity convention using the qualified COM. A smooth warmup
decays virtual object control from one to the curriculum value; policy
evaluation forces it to exactly zero from frame zero.

At startup, the task applies the released 64-bucket friction range `[2.0,
2.01]` and restitution 0.0 to every colliding Wuji hand shape through Newton's
public per-shape model arrays. Each selected shape gets one bucket with the run
seed. Full Newton body paths must be below the `Robot` articulation, and the
leaf must be one exact Wuji MJCF body name. Vega arm materials keep their
authored values. If a Newton backend supplies only leaf labels, the
exact-name check remains active, but articulation-path scoping is not
available. Newton MJWarp has one sliding-friction value per shape, so the port
uses the released static-friction draw for that value and consumes, but cannot
apply, the separate dynamic-friction draw. The imported damped MuJoCo contact
`solref` supplies the zero-bounce behavior; the Newton restitution field is
also set to 0.0.

USD/URDF material binding can fail on instanced scene meshes. A second startup
event therefore matches every object and support collision subtree in every
replicated environment and reapplies typed `ScenePhysics` through Newton's
public arrays, then verifies the write. Newton again has one coefficient, so
the scene contract's dynamic friction is authoritative at runtime; static
friction remains recorded for backends that distinguish the two.

Isaac Lab's Newton contact sensor does not publish contact points. The task
uses the public Newton 1.2.1 contact buffer after each solver step. It averages
the object-side manifold points and sums the force for each object and hand
link. Live wrench arms and contact observations therefore use solver contact
points. The sensor still enables the optional Newton force buffer.

### Validate, replay, and train

Validate the vendored 59-actuator robot-only MJCF before Isaac starts:

```bash
pixi run python scripts/data/validate_vega_wuji_asset.py \
    --model source/isaaclab_imitation/isaaclab_imitation/assets/vega_wuji/vega_u_wuji_v2_beta1_with_mount.xml
```

Replay one Reference with Newton:

```bash
TERM=xterm OMNI_KIT_ACCEPT_EULA=YES \
pixi run -e isaaclab python scripts/viz/replay_vega_wuji_reference.py \
    --reference data/dexmanip/manifests/vega_wuji_manifest.json \
    --motion-index 0 \
    --video \
    --physics newton_mjwarp \
    --viz none
```

Replay uses `--motion-index` and starts at frame zero. It keeps the JSON
Manifest intact so the environment verifies its robot-model hash. Training
samples motions and valid start frames. After the global virtual object control
scale falls below 0.1, 10% of training resets start at frame zero by default.
This first-frame mix closes the gap to evaluation, which always starts at frame
zero.

Run RLOpt PPO. This task has no RSL-RL or PhysX registration:

```bash
TERM=xterm OMNI_KIT_ACCEPT_EULA=YES \
pixi run -e isaaclab python scripts/rlopt/train_newton.py \
    --task Isaac-Imitation-Vega-Wuji-v0 \
    --algo PPO \
    --num_envs 4096 \
    env.reference_path=data/dexmanip/manifests/vega_wuji_manifest.json \
    physics=newton_mjwarp
```

The RLOpt config translates the released CHORD PPO recipe: 24 steps per
environment, 20,000 updates, four mini-batches over five epochs, ELU networks
with widths `[1024, 512, 256, 128]`, running input normalization, initial
policy standard deviation 0.1, clip 0.1, entropy weight 0.001, learning rate
0.001, and target KL 0.005. Checkpoints use the released 200-update cadence,
converted to RLOpt's environment-frame unit.

The released recipe states its rollout split and checkpoint cadence as counts
that hold at every environment count, while RLOpt sizes both in frames. The
config therefore also declares `reference_num_envs`, `mini_batches_per_rollout`,
and `save_interval_iterations`, and the launcher restates the frame sizes at the
environment count actually used. A run at any `--num_envs` keeps four
mini-batches per rollout and the 200-update checkpoint cadence; a run at
`reference_num_envs` is unchanged. Passing `agent.loss.mini_batch_size=` or
`agent.save_interval=` explicitly opts out of the restatement for that field.
The local pre-release config disables remote experiment logging by default, so
the smoke/training command does not require service credentials; campaigns may
opt into their logger explicitly.

Set `DEXMANIP_VEGA_WUJI_REFERENCE_PATH` to use another NPZ or Manifest. The
vendored MJCF and meshes use Git LFS. Run `git lfs pull` after a fresh clone.
