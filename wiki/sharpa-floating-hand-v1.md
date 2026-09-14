# Sharpa floating-hand imitation v1

This baseline reproduces the video-to-data method with two independent
free-floating Sharpa hands, one rigid object, PhysX, and RSL-RL PPO. It does
not use a combined environment USD. Isaac Lab assembles the two Apache-2.0
Sharpa URDFs, their meshes, and the selected rigid object at run time.

## Data conversion

The `isaaclab` Pixi environment includes the ILTools `sharpa` extra with
PyArrow, Pinocchio 3.7, Pink 4.2, DAQP, and the required CMEEL ABI pins.

Convert a processed `ManoSharpaData` Parquet dataset:

```bash
pixi run -e isaaclab python scripts/data/convert_mano_sharpa_to_iltools.py \
  --input /path/to/processed_parquet \
  --object-asset /path/to/one_rigid_object.usd \
  --output /path/to/iltools_sharpa \
  --dataset-name my_sharpa_set
```

Add `--retarget`, `--left-mjcf`, and `--right-mjcf` for MANO-only input. The
retargeter uses the source eleven Pink tasks, sequential warm starts, a 200 Hz
solve rate, DAQP, and 100 iterations per frame (the source dataset-script
default). Conversion resamples to 20 Hz by default. The output has validated
NPZ references, `trajectories.zarr`, and `manifest.json`.

Each reference records retarget provenance in its metadata: `retarget_source`
(`iltools_pink` when ILTools solved the row, `parquet_robot_columns` when the
stored source columns were trusted), the solver settings and MJCF hashes,
per-hand IK task-error and iteration statistics, and `retarget_solution_kind`.
A `placeholder` kind means every frame reports zero task error after one
iteration; that is a schema fixture such as the source `synthbox_processed`,
not an IK result, and must not be used as a reference.

Gate any solver, MJCF, or resampling change with a frame-level comparison
against a trusted reference:

```bash
pixi run -e isaaclab python scripts/audit/compare_sharpa_references.py \
  --reference /path/to/source_ik/manifest.json \
  --reference /path/to/iltools_pink/manifest.json \
  --max-wrist-position-m 0.001 --max-wrist-rotation-rad 0.01 \
  --max-joint-rad 0.001 --max-object-position-m 1e-6 \
  --min-contact-agreement 1.0
```

The ILTools test `tests/retarget/test_sharpa_source_parity.py` runs the
adjacent video-to-data solver and the ILTools port on the same frames. Run it
through the `isaaclab` environment when the `video_to_data` checkout is
present next to this repository.

For the easy source `synthbox_box_open_000` fixture, select the box bottom as
the documented rigid proxy (the source contains a bottom and an articulated
lid):

```bash
pixi run -e isaaclab python scripts/data/convert_mano_sharpa_to_iltools.py \
  --input ../video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/human_motion_data/synthbox/synthbox_processed \
  --object-asset source/isaaclab_imitation/isaaclab_imitation/assets/sharpa_wave/rigid_box_proxy.usda \
  --rigid-object-index 0 \
  --output /tmp/sharpa_synthbox_box_open \
  --dataset-name synthbox_box_open_rigid
```

Validate the converted set before simulation:

```bash
pixi run -e isaaclab python scripts/data/inspect_sharpa_reference.py \
  --manifest /tmp/sharpa_synthbox_box_open/manifest.json
```

Sharpa v1 rejects articulated objects and more than one object. The included
source synthbox sample has an articulated scene, so the simulation smoke test
uses a documented rigid box proxy.

## Smoke fixture and replay

```bash
pixi run -e isaaclab python scripts/data/make_sharpa_rigid_smoke_fixture.py \
  --output /tmp/sharpa_fixture

pixi run -e isaaclab python scripts/viz/replay_sharpa_reference.py \
  --reference_manifest /tmp/sharpa_fixture/manifest.json \
  --num_envs 1 --steps 100 --headless
```

Add `--video --video_length 102 --video_dir /path/to/videos` to save an MP4
for visual inspection. The replay script records the rendered zero-residual
reference, rather than a trained policy rollout.

Pass `--source-parquet /path/to/synthbox_processed` to keep the source sample's
retargeted hands, object base motion, and contacts. The script maps the
sample's articulated object parts to the single rigid proxy. Without this
option, it creates a small synthetic contract fixture.

The task is `Isaac-Sharpa-V2D-PhysX-v0`. It uses a 0.01 second PhysX step,
five physics steps per policy action, 22 finger joints per hand, and 56 policy
actions in total. Reference arrays are loaded to the simulation device before
stepping. Per-environment motion and start-frame indices are sampled in memory.

The same task exposes the G1-style Newton backend preset. Select it with the
Hydra override below (the historical task name is retained for compatibility):

```bash
export SHARPA_REFERENCE_MANIFEST=/path/to/iltools_sharpa/manifest.json

pixi run -e isaaclab python scripts/rsl_rl/train.py \
  --task Isaac-Sharpa-V2D-PhysX-v0 --num_envs 4096 \
  --logger tensorboard --headless physics=newton_mjwarp
```

Newton uses its MJWarp solver and Newton contact sensors, while PhysX remains
the default. The Sharpa action term keeps the native PhysX gravity-compensation
query when available and derives the equivalent wrist-body wrench from Newton
mass/gravity metadata otherwise. Playback uses the same override:

```bash
pixi run -e isaaclab python scripts/rsl_rl/play.py \
  --task Isaac-Sharpa-V2D-PhysX-v0 --checkpoint /path/to/model.pt \
  --num_envs 1 --max_steps 100 --headless physics=newton_mjwarp
```

## Source-parity task and RLOpt training

`Isaac-Sharpa-V2D-Source-v0` reproduces the released recipe and needs a
reference converted at the released half-speed playback:

```bash
pixi run -e isaaclab python scripts/data/convert_mano_sharpa_to_iltools.py \
  --input /path/to/<ds>_loaded --object-asset /path/to/rigid_object.usda \
  --rigid-object-index 0 --retarget --left-mjcf ... --right-mjcf ... \
  --motion-speed 0.5 --output /path/to/iltools_sharpa_speed05 \
  --dataset-name my_sharpa_set_speed05
```

Train it with RLOpt (PhysX is the parity backend; the Kit-first bootstrap is
required because the hands are URDF assets):

```bash
export SHARPA_REFERENCE_MANIFEST=/path/to/iltools_sharpa_speed05/manifest.json

pixi run -e isaaclab python scripts/rlopt/train_physx.py \
  --task Isaac-Sharpa-V2D-Source-v0 --algo PPO --num_envs 4096 --headless
```

Add `agent.logger.backend=` for a local smoke without W&B. The same RLOpt
config is registered for `Isaac-Sharpa-V2D-PhysX-v0`, and `scripts/rsl_rl/train.py`
still works for both tasks as the A/B baseline.

## Training and playback

The environment reads `env.reference.manifest`. Set this environment variable
before the Isaac Lab entry point constructs the Hydra config:

```bash
export SHARPA_REFERENCE_MANIFEST=/path/to/iltools_sharpa/manifest.json

pixi run -e isaaclab python scripts/rsl_rl/train.py \
  --task Isaac-Sharpa-V2D-PhysX-v0 \
  --num_envs 4096 --logger tensorboard --headless
```

For a quick check, use `--num_envs 4 --max_iterations 3`. Playback must use
the same manifest and checkpoint:

```bash
export SHARPA_REFERENCE_MANIFEST=/path/to/iltools_sharpa/manifest.json

pixi run -e isaaclab python scripts/rsl_rl/play.py \
  --task Isaac-Sharpa-V2D-PhysX-v0 \
  --checkpoint /path/to/model.pt --num_envs 1 --max_steps 100 --headless
```

The default command advances on the first control step after reset. Set
`env.legacy_reset_hold=true` only to reproduce the released command-freeze
behavior during source parity studies.

The reward manager exposes separate trajectory and task-objective terms:
`object_pose_tracking_exp` tracks the live object pose against the current
reference frame, while `object_goal_tracking_exp`, `object_lift_progress`, and
`object_task_success` shape the final manipulation objective. Their weights are
introduced by the fixed curriculum after the initial imitation stages. Command
metrics also report final position/orientation error, lift height, task success,
and expected-versus-observed contact agreement.

## Scope

This first version covers one rigid manipulated object. Camera perception,
articulated objects, multi-object scenes, policy-quality claims, and
Vega-Wuji policy transfer remain out of scope. Existing Vega-Wuji references
continue to load through the fixed-base schema path.
