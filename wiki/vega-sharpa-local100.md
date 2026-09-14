# Vega U + Sharpa local training check

Updated: 2026-09-13. The mounted task runs with Newton/MJWarp and RLOpt PPO.
The local milestone is a 100-iteration integration test, with a retained
checkpoint and a tested restart. This is preliminary runtime evidence, not
an unassisted manipulation success result.

The implementation is on branch `dev-dex`, with its local worktree at
`.codex/worktrees/vega-sharpa-local`. Run the commands below from that
worktree. It has its own locked Pixi environments. The original checkout's
uncommitted work and input datasets were preserved.

## Robot and data contract

| Item | Local configuration |
| --- | --- |
| Task | `Isaac-Imitation-Vega-Sharpa-v0` |
| Robot | Fixed Vega U with two Sharpa Wave hands |
| Controlled joints | 14 arm joints and 44 finger joints; 58 actions |
| Setup joints | Lift, torso and head fixed at zero |
| Base placement | XYZ `[-0.04, 0.55, 0.18694717]` m, yaw -90°; base mesh bottom aligned with the z=0 floor |
| Mounts | Identity at `L_arm_l8` and `R_arm_l8`, following the collaborator's Sharpa composition convention; physical adapters are not measured |
| Robot controller | Arm PD 400/40, velocity feedforward, vendor Sharpa finger actuators, ideal robot gravity compensation |
| Simulation | Newton/MJWarp, 0.01 s physics step, five substeps per 20 Hz control step |
| Parallel worlds | 64 separate Newton physics worlds with the same coordinate origin |
| Data | Real ARCTIC `s07_box_grab_01`, from the existing ILTools/Pink Sharpa Reference |
| Local section | Source Reference frames 603–764 inclusive; the full 965-frame input and its arm audit remain available |
| Playback | The selected section is slowed by another factor of three: 484 frames at 20 Hz, about 24 s |
| Object | True-size rigid bottom proxy; seven explicit convex pieces preserve the box cavity |
| Object physics | 0.3 kg; mesh-derived COM and diagonal inertia approximation; cross terms omitted |
| Stand | Static cylinder, radius 0.055 m, height about 0.959 m; radius is a prototype assumption, position/top come from the first source object pose |
| Collisions | Robot self-collision and robot/stand collision enabled; vendor adjacent-link exclusions plus the two connected torso/shoulder housing pairs retained |

The local section exercises mounted manipulation. Full capture coverage,
approach/grasp/set-down phase coverage, physical adapter measurements, and
longer unassisted policy training remain subsequent work. This crop is not
a qualification of the rejected portions of the capture.

The original station placed the base frame at z=0, but the mesh extends
0.18694717 m below that frame. Preparation now derives the root height from
the base collision mesh. An explicit height below the floor is rejected during
preparation, publication and task loading. The runtime audit measures the
converted base mesh in every world. The corrected Reference re-solves arm IK;
object trajectories and stand poses are unchanged from the previous section.

The data pipeline first solves the arms with ILTools. It then repairs hand
self-collision, projects finger/object contact constraints, allows at most
0.05 rad of additional correction per arm joint, retimes the motion, and
rechecks the interpolated poses. It recomputes robot contact witnesses,
body poses and velocities. Human MANO contact labels are not presented as
measured robot contacts.

The prepared section passes its declared native checks: at most 1 mm
penetration, interval joint speeds below 2.4 rad/s for arms and 11.62 rad/s
for fingers, and source wrist deviation below 50 mm and 0.2 rad. The final
audits record the actual maxima and all frame values. These bounds qualify
the local test input, not hardware deployment.

Robot gravity compensation uses `gravcomp=1` in MJCF and `mjc:gravcomp=1`
on all robot rigid bodies in USD. The pinned Newton backend ignores the
Isaac per-body `disable_gravity` flag. World gravity stays enabled and the
object has no gravity compensation. Conversion also expands body collision
exclusions into explicit shape pairs because Newton 1.2.1 only reads USD
`physics:filteredPairs` on shapes. Runtime checks verify active backbone
colliders, compensation and the connected shoulder exclusions in all worlds.

## Rebuild assets and data

The existing floating-hand Reference is an input. New files go under
`data/dexmanip/vega_sharpa_gravcomp/`.

```bash
OMNI_KIT_ACCEPT_EULA=YES pixi run -e isaaclab python scripts/data/prepare_vega_sharpa_assets.py \
  --source-manifest data/dexmanip/sharpa/arctic_box_grab_speed05_usd/manifest.json \
  --object-mesh data/dexmanip/sharpa/assets/arctic_box_bottom/bottom_watertight_tiny.obj \
  --output data/dexmanip/vega_sharpa_gravcomp

pixi run -e isaaclab python scripts/data/prepare_vega_sharpa_reference.py \
  --source-manifest data/dexmanip/sharpa/arctic_box_grab_speed05_usd/manifest.json \
  --model data/dexmanip/vega_sharpa_gravcomp/model/vega_u_sharpa.xml \
  --robot-asset data/dexmanip/vega_sharpa_gravcomp/model/usd/asset.json \
  --scene data/dexmanip/vega_sharpa_gravcomp/scene/scene.json \
  --start-frame 603 --stop-frame 765 --time-stretch 3 \
  --output data/dexmanip/vega_sharpa_gravcomp/local100_grounded
```

The converter records the actual generated USD path and layer hashes in
`model/usd/asset.json`. Data publication and runtime loading check the robot,
object and support identities. The preparation directory retains the full
arm solve, source-fidelity audit, collision/speed audit, crop and recipe.

## Qualify the runtime

```bash
export VEGA_SHARPA_REFERENCE_MANIFEST="$PWD/data/dexmanip/vega_sharpa_gravcomp/local100_grounded/manifest.json"
OMNI_KIT_ACCEPT_EULA=YES pixi run -e isaaclab python scripts/audit/check_vega_sharpa_runtime.py \
  --num-envs 64 --steps 120 \
  --output outputs/vega_sharpa/grounded_runtime64.json physics=newton_mjwarp
```

The check exercises converted base/floor alignment, body/joint mapping, arm PD, rotated body forces and
torques, positive contacts on both hands, zero separated contacts, support
and free fall, and finite rollout state. The positive-contact fixture uses
a deliberate solid overlap; its forces are not normal grasp/reset forces.
Add `--unassisted` to record a zero-residual rollout with object assistance
disabled. Terminations in that rollout measure work left for the policy.

Keep `env_spacing=0` for this Newton recipe. Each environment has a distinct
physics-world ID. A shared coordinate origin keeps static geometry correct
in the pinned batched MuJoCo model. The runtime check verifies the world
count and support behavior across the entire batch. PhysX has not been
qualified for this mounted task.

## Train and restart

```bash
OMNI_KIT_ACCEPT_EULA=YES pixi run -e isaaclab python scripts/rlopt/train.py \
  --task Isaac-Imitation-Vega-Sharpa-v0 --algo ppo \
  --num_envs 64 --max_iterations 100 --seed 42 --headless --assert-kitless \
  physics=newton_mjwarp agent.logger.backend=csv \
  agent.logger.log_dir=logs/vega_sharpa_grounded100
```

The mounted recipe uses 24 rollout steps, four minibatches, five optimizer
epochs, CSV metrics each iteration, and checkpoints every 50 iterations.
At 64 environments, 100 iterations mean 153,600 environment frames and
2,000 optimizer updates. The source assistance curriculum is active during
this short test. Longer training must evaluate reduced and zero assistance.

The verified clean run is under
`logs/vega_sharpa_grounded100/2026-09-13_16-04-06_run-33187b3b/`.
It completed 100 iterations with checkpoints at 76,800 and 153,600 frames.
All 100 optimization records, including gradient norms and losses, are finite;
all saved policy/value, normalizer and Adam tensors are finite. Episode metrics
begin after the first completed episode. These are integration checks only.

```bash
OMNI_KIT_ACCEPT_EULA=YES pixi run -e isaaclab python scripts/rlopt/train.py \
  --task Isaac-Imitation-Vega-Sharpa-v0 --algo ppo \
  --num_envs 64 --max_iterations 102 --seed 42 --headless --assert-kitless \
  --checkpoint logs/vega_sharpa_grounded100/2026-09-13_16-04-06_run-33187b3b/models/model_step_153600.pt \
  physics=newton_mjwarp agent.logger.backend=csv agent.save_interval=1536 \
  agent.logger.log_dir=logs/vega_sharpa_grounded_resume
```

`max_iterations` is the total budget on restart. The checkpoint restores
policy/value parameters, observation normalization, Adam state, cumulative
frames, optimizer-update count, and the Isaac curriculum clock. The collector
starts new episodes under that restored curriculum. A different task or
Reference fingerprint is rejected.

The current mechanical record is `outputs/vega_sharpa/grounded_runtime64.json`.
The revised kinematic video is `outputs/vega_sharpa/grounded_reference_replay.mp4`.
The correction record `outputs/vega_sharpa/grounded_correction_record.json`
includes data identities, native audit maxima, runtime results and test records.
`outputs/vega_sharpa/grounded_training_audit.json` records checkpoint hashes,
finite CSV/tensor checks and the exact training/restart commands.

The corrected one-seed run completed 100 iterations in 77.98 seconds of
training (startup excluded). The tested restart is
`logs/vega_sharpa_grounded_resume/2026-09-13_16-06-20_run-ba271509`;
it continued for exactly two iterations to 156,672 frames, 2,040 optimizer
updates and curriculum step 2,448. All optimization metrics and checkpoint
tensors are finite, and the resumed policy weights changed.

The corrected unassisted zero-residual diagnostic is
`outputs/vega_sharpa/grounded_unassisted64.json`: 120 control steps across
64 environments, 1,536 failure terminations, longest episode four steps.
The assisted diagnostic had no terminations in its 120 control steps.
These are local integration/baseline results, not trained-policy success.
Earlier runs with `qualified_` names used the below-floor base placement and
are retained as diagnostic history. Their checkpoints belong to a different
Reference fingerprint and cannot be used for the corrected training input.

The changes include reusable ILTools mounted-hand/scene processing and a
small restart hook in its existing dual-hand solver. RLOpt also changed:
its generic checkpoint now stores cumulative frames and PPO update count.
The Isaac-specific curriculum and Reference checks live in the workspace,
not in RLOpt. Existing uncommitted source work was preserved.

## Inspect the prepared motion

```bash
MUJOCO_GL=egl pixi run -e isaaclab python scripts/viz/render_vega_sharpa_reference.py \
  --reference "$VEGA_SHARPA_REFERENCE_MANIFEST" \
  --output outputs/vega_sharpa/grounded_reference_replay.mp4
```

This video is a kinematic replay of the prepared Reference. The training
and runtime logs are separate evidence. It is not a trained-policy video.
