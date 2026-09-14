# Vega U + Sharpa implementation plan

Updated: 2026-09-13. Status: the mounted task and local 100-iteration path are
implemented; full capture coverage and expert convergence remain open.
See the [local training runbook](vega-sharpa-local100.md) for the tested
scope, configuration, artifacts and restart commands. The current station
height is derived from the base mesh so its bottom meets the floor; the
previous zero-height base frame was below the floor and has been corrected.

The target is one Vega U with two Sharpa Wave hands, trained with RLOpt PPO
inside IsaacLab-Imitation. ILTools owns retargeting and processed data.
Newton or PhysX is acceptable. Select the backend by measured correctness
and usable training throughput on the assembled robot.

The first task is the bimanual ARCTIC `s07_box_grab_01` box-on-stand sequence.
It supplies a common task with the collaborator's work and the existing
Sharpa data. This is the proposed first milestone, followed by more motions.
Real-robot deployment is a later milestone.

## What we can reuse

| Existing work | Use in this plan | Limit of the evidence |
| --- | --- | --- |
| [ILTools Sharpa Pink retargeter](../ImitationLearningTools/iltools/retarget/sharpa_pink.py) and [MANO loader](../ImitationLearningTools/iltools/datasets/mano_sharpa/loader.py) | Human hand/object input, Sharpa finger solve, resampling, provenance | The writer currently produces floating-hand References; mounted arm state must be added. |
| [Collaborator's Sharpa-on-Vega environment](../../DexManip/isaac_bench/v2d_wuji/sharpa_vega_env.py) and [robot config](../../DexManip/isaac_bench/v2d_wuji/sharpa_vega_robot.py) | Assembly, joint/body mappings, actuator configuration, mounted command/action pattern | Code exists. The Wuji training result does not establish successful Sharpa training. Referenced Sharpa assembly assets must be generated or located. |
| [Collaborator's arm IK](../../DexManip/isaac_bench/v2d_wuji/wuji_arm_ik.py) | Mount transforms, sequential arm solve, branch handling, velocity limits, feasibility labels | Port reusable logic into ILTools and validate it against the selected assembly. |
| [Sharpa source task](../source/isaaclab_imitation/isaaclab_imitation/tasks/manager_based/dexmanip/config/sharpa/sharpa_source_env_cfg.py) | Object/contact rewards, curriculum, training recipe, backend adapters | This is a floating-hand task. Its local training smoke is preliminary runtime evidence. |
| [Newton wrench correction](../source/isaaclab_imitation/isaaclab_imitation/newton_wrench_frame.py) | Correct force and torque coordinates for the pinned affected Isaac Lab version | Re-run the force test on the mounted assembly and after a dependency upgrade. |
| [Vega/Wuji task](../source/isaaclab_imitation/isaaclab_imitation/envs/vega_wuji_imitation_env.py) | Fixed-base scene, Reference loading, joint-order validation, replay tooling | Wuji joint counts, limits, frames, and promotion rules are specific to that task. |

The August SOMA/Wuji work remains useful reference material. The September
floating-Sharpa work supplies the immediate integration baseline. This plan
sets the direction for the new mounted-Sharpa task; the older task protocols
remain recorded under their existing task IDs.

## Ownership and data flow

```text
ARCTIC + MANO
  -> isolated LOAD stage
  -> ILTools Sharpa hand retarget
  -> ILTools Vega arm IK and contact/trajectory checks
  -> full-robot DexterousReference + assets + Manifest
  -> IsaacLab-Imitation mounted-Sharpa task
  -> RLOpt PPO
  -> unassisted evaluation and retained videos
```

| Owner | Deliverable |
| --- | --- |
| `ImitationLearningTools/` | Reusable hand/arm retargeting, support processing, contact geometry, audits, Reference export and resampling. |
| `source/isaaclab_imitation/` | Robot assets/config, backend adapters, live observations/actions, rewards, resets and terminations. |
| `source/imitation_experiments/` | Shared replay, evaluation, provenance and benchmark logic. |
| `scripts/` | Thin conversion, asset, replay and training entrypoints. |
| `RLOpt/` | Existing PPO; change it only for a demonstrated algorithm/runtime gap. |
| `experiments/campaigns/` | Frozen training/evaluation configuration and launchers, created after local qualification. |

Runtime and data preparation should consume repo-owned packages and explicit
inputs. NVIDIA and the collaborator's repos supply implementation references.
Preserve required attribution when porting their code. The isolated MANO LOAD
environment can remain separate; ILTools owns subsequent data processing.

## 1. Freeze the robot and first-task contract

Record the current working-tree state, dependency revisions and input hashes
before implementation. Preserve the substantial uncommitted work in the main
repo and its submodules. Keep existing References as immutable inputs and
write new candidates to new paths.

Assemble a robot-only Vega U + Sharpa asset using the actual Vega variant,
the existing Sharpa descriptions, and explicit left/right adapter transforms.
The collaborator's [composition tool](../../DexManip/isaac_bench/tools/compose_urdf.py)
supports Sharpa, but its identity mounts still assume adapter thickness and
clocking. Record those as simulation assumptions until hardware geometry is
available. Include adapter mass and collision geometry when available.

For the first implementation, fix the base, head, lift and torso at declared
station settings and control both arms and both hands. This follows the
collaborator's station model. Verify whether the selected hardware actually
supports active lift/torso control before expanding the action space.
Derive joint sets and action dimensions from the assembled model.

Use the existing ARCTIC box data at true size. Retain the rigid-proxy decision
explicitly: check that omitted lid motion does not define the selected task.
Freeze the source crop, playback speed, object physics, support dimensions,
station pose and gravity-compensation setting in one config. Keep robot,
object and support assets separate.

**Exit:** the assembly loads, both hands have the correct orientation, named
wrist frames agree across offline and live forward kinematics, and the
actuator/joint sets and physical parameters have an auditable source.

## 2. Prove contact and control behavior on both backends

Build small deterministic fixtures around the assembled robot before a PPO
run. Use the existing locked environments first. Pre-convert assets where
needed for Newton's Kit-free startup and inspect the resulting collision
geometry, including the box cavity and retained hand links.

Test Newton first because the current workspace has a corrected floating-hand
runtime. Run the same fixtures on PhysX. The mounted Sharpa model has its own
solver and contact requirements; neither floating-hand results nor Wuji
results qualify it.

Required fixtures:

- Robot PD tracking in free space; check position, velocity and torque limits.
- Known world/local forces and torques on rotated bodies, including object
  assistance and gravity compensation.
- Separated hand/object geometry gives no contact; a deliberate loaded
  fingertip/object contact gives a nonzero signal on the expected hand link.
  Check contact point, force direction and object motion together.
- An object rests on the stand and falls when unsupported with assistance off.
- Robot self-collision and robot/stand collision follow the declared filters.
- Repeated resets and replication preserve those behaviors.

The existing [backend smoke script](../scripts/audit/check_sharpa_backend_smoke.py)
checks an upper contact-force bound but does not require a positive signal
in a known-contact fixture. Add that check to the new qualification path:
a broken sensor returning zero must fail.

For PhysX, investigate the current per-shape/per-body filtering failure and
the collaborator's solver-partition setting as separate hypotheses. Any
collision aggregation or simplification must retain a traceable mapping to
hand links and pass the same geometry/contact checks. A zero contact signal
cannot be replaced by an assumed contact. Stop expanding the PhysX repair
scope once Newton meets the task's needs; the reverse also applies.

**Exit:** at least one backend passes the physical fixtures. Record the
other backend's concrete failure and reproduction, if any. Use plain MuJoCo
for fast geometry/IK iteration, then validate the exported asset in Isaac;
plain MuJoCo does not certify Newton/MJWarp behavior.

## 3. Produce a full mounted-Sharpa Reference in ILTools

Start from the real ARCTIC loaded data and the existing Sharpa hand recipe.
Keep source wrist, fingertip and object targets available independently of
the robot solution. Preserve the object size and contact timing.

Port the collaborator's arm-IK structure into ILTools: derive flange targets
from the mount transform, solve each arm sequentially, control branch changes,
enforce joint velocity limits, and label feasibility after the final solve.
Choose one station placement for the sequence. Transform the hands, object,
contact geometry and support consistently when changing the scene frame.

Start with a wrist/hand solve followed by arm IK. If reachable wrist targets
remain the limiting issue, compare the collaborator's modified Sharpa recipe
against the existing source recipe on the same capture. Its wrist offsets
are candidate calibration values, not universal constants. A later coupled
refinement can trade wrist, fingertip and contact error within measured bounds.

Export one `fixed_base` DexterousReference with ordered full-robot `qpos` and
`qvel`, actual wrist and hand-link poses, object poses/twists, support assets,
contact points/normals/link identities, feasibility labels and provenance.
Resample onto one declared control timeline and recompute velocities from
that timeline. Bind the output to the robot, object and collision dependencies.

Audit anatomical/joint limits, continuity, arm feasibility, fingertip fit,
self-collision, forbidden contact, intended contact and support clearance.
Use the actual collision geometry; a convex hull can fill a box cavity.
Inspect approach, grasp, rotation/lift and set-down phases in a synchronized
human/robot video. Report human-target error and object-contact error separately.

Correct bad contact locally when needed. Through-wall fingers need a repair
that recovers the intended surface side and preserves time continuity;
pressing a deeply penetrating hand toward its old targets is not an accepted
repair. Recompute every audit after the final change. Freeze acceptance
thresholds for Sharpa before selecting the final candidate.

**Exit:** a reproducible, physically interpretable Reference and a start-frame
pool that preserves a usable complete manipulation sequence. Report rejected
frames explicitly; hiding difficult manipulation frames does not pass the gate.

## 4. Add the mounted-Sharpa task

The new task is registered as `Isaac-Imitation-Vega-Sharpa-v0`. Its current
launch commands and local qualification scope are in the linked runbook.

Reuse the fixed-base task structure and the Sharpa reward/contact code through
small explicit configuration boundaries. Add the Sharpa embodiment definition
under `assets/` and a task config under `tasks/manager_based/dexmanip/config/`.
Extract shared command/action helpers only where both implementations need them.

Use arm and finger joint-position residuals around the Reference, with
separate scales, smoothing and live joint limits. Include arm reference
velocity feedforward if the free-space control test supports it. Observe
current arm/finger state, desired robot state, wrist/object errors, contact
features and previous actions. Resolve joint/body order by names.

Reset the fixed robot and objects to the same Reference frame with consistent
velocities. Start qualification from a feasible pre-grasp frame. Add random
feasible starts only after those resets work. Keep the Reference clock,
assistance schedule and checkpoint-resume counters explicit and tested.

Keep the real stand. The target task includes robot/stand collision. If a
collision-filtered reproduction is useful to isolate a regression, give it
an explicit diagnostic config and report it separately. Adjust station pose
or repair the trajectory when the hand intersects the real stand.

Define training readiness for this new task separately from learned success.
Data checks and controlled assisted replay must establish coherent targets,
valid resets, finite physics and working contacts. An unassisted zero-residual
replay measures what the policy must correct; dropping the object is not by
itself proof that learning is impossible. Successful unassisted manipulation
is the trained-policy evaluation gate. This contract must be explicit rather
than inherited accidentally from the Vega/Wuji promotion tool.

**Exit:** the new task runs through the RLOpt interfaces, data/action timing
is correct, and assisted and unassisted replay produce interpretable records.

## 5. Select the training backend at useful scale

Repeat the known-contact fixtures and mounted task checks at increasing
environment counts: initially 1 and 16, then 64, 256, and the intended training
count as memory permits. Include resets, active contact and optimizer state
in the measurement. The count ladder is a proposed test protocol.

| Decision order | Required evidence |
| --- | --- |
| Physics correctness | Expected contact signals, bounded state, valid force frames and no silent collision loss. |
| Training correctness | Finite observations/rewards/gradients, advancing References, successful save/resume and meaningful contact reward. |
| Capacity | Measured peak GPU memory with the full robot and active training; adequate headroom. |
| Speed | Steady-state environment frames/second and PPO update time at a passing count. |

Choose one backend for the first convergence run. Exact numerical agreement
between Newton and PhysX is not required. Shared task semantics and physical
correctness are required. Keep their physical-material/contact settings
recorded with every comparison. RSL-RL is available for a targeted trainer
diagnostic only if an RLOpt issue remains after the environment checks.

**Exit:** one measured backend/configuration, a passing local PPO smoke, and
a resumable checkpoint whose optimizer, normalization and curriculum state
are restored correctly. This is qualification, not a success-rate result.

## 6. Train and evaluate the first expert

Use the existing CHORD-derived PPO recipe as the starting point. Prioritize
object pose/contact behavior, with hand/arm tracking as shaping and action
regularization. Use virtual object assistance during training and reduce it
on a recorded schedule. Preserve the schedule's units when batch size or GPU
count changes. Transfer the collaborator's tighter late-stage object tracking
curriculum only after the mounted Sharpa baseline is stable.

Use local runs for qualification, within the existing roughly 50M-frame local
ceiling. Prepare a frozen cluster campaign for convergence. The repository's
default long-run target is about 10B environment frames unless another budget
is selected; that budget does not stretch the assistance schedule. Configure
the actual cluster walltime, persistent checkpoints and resume chain explicitly.
The first run uses one seed; the completion study repeats with three seeds.
Partial runs and local smoke metrics remain preliminary.

Before training, freeze evaluation starts, physical settings, success/error
thresholds and checkpoint selection. Evaluate with virtual object assistance
exactly zero from the first frame and after every reset:

- Full-sequence first-frame trials, counting every trial and failure.
- Random feasible-start trials, with remaining-horizon eligibility reported.
- Object position/orientation tracking, completion, drops, contact behavior,
  robot/stand collisions, joint limits and action saturation.
- A zero-residual replay baseline under the same evaluation settings.
- Representative retained videos including failures and all set-down phases.

Report both the collaborator's CHORD success criterion and full-sequence
completion. A partial completion score or an object caught by the stand must
not be presented as a complete manipulation. Record termination thresholds
beside tracking results.

**First demonstration milestone:** a complete unassisted mounted-Sharpa
rollout of the box task. **Completion of this plan:** repeatable evaluated
policies across seeds, a retained checkpoint, versioned assets/data/config,
and commands that rebuild the Reference, train, resume and evaluate.

## Immediate work queue

1. Record the working-tree/input snapshot and assemble the Vega U + Sharpa
   model with explicit mounts, setup joints and support geometry.
2. Extend the contact/force qualification fixtures and test the assembled
   model on Newton and PhysX.
3. Move the arm-IK/data composition into ILTools and generate the ARCTIC
   mounted-Sharpa Reference with a synchronized inspection video.
4. Implement the new task and qualify RLOpt at a useful environment count.
5. Freeze the convergence campaign and unassisted evaluation protocol.

The remaining hardware inputs are the exact Vega model revision, adapter
clocking/thickness and whether lift/torso are intended to move during a task.
They do not prevent a simulation prototype with explicitly recorded assumptions.
They must be resolved before claiming that prototype matches the physical robot.
