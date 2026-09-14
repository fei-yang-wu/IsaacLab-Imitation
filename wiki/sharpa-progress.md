# Sharpa floating-hand progress snapshot

Status date: 2026-09-06

This page records the current state of the Sharpa floating-hand imitation
milestone. The goal is an object-centric video-to-data pipeline: source MANO
and object motion is converted to a dual free-floating Sharpa reference, then
tracked in Isaac Lab with residual actions and trained with RSL-RL PPO.

## What is implemented

### Data and retargeting

- `DexterousReference` supports the v2 NPZ schema and distinguishes
  `fixed_base` from `dual_floating_hand` layouts. Legacy v1 references load as
  fixed-base references.
- Dual-hand references store left-then-right 22-joint `qpos`/`qvel`, wrist
  poses and twists, hand-link poses, object poses and twists, contacts, scene
  assets, hashes, and physics metadata. The JSON manifest remains the v1
  hash-bound manifest wrapper around v2 reference files.
- `ManoSharpaLoader` reads source `ManoSharpaData` Parquet. It preserves metric
  object motion and MANO contact geometry, rejects unsupported articulated or
  multi-object inputs at the Sharpa boundary, and writes canonical NPZ/Zarr
  trajectories plus a manifest.
- `SharpaPinkRetargeter` reproduces the source Pink/Pinocchio approach:
  eleven task groups per hand, sequential warm starts, limits and costs,
  200 Hz solves, and DAQP.
- Conversion and inspection entry points are available in
  `scripts/data/convert_mano_sharpa_to_iltools.py` and
  `scripts/data/inspect_sharpa_reference.py`.

### Isaac Lab environment

- The task `Isaac-Sharpa-V2D-PhysX-v0` assembles two independent floating
  Sharpa Wave articulations, one rigid manipulated object, and optional static
  support geometry at runtime. No combined environment USD is required.
- Physics runs at 100 Hz and policy control at 20 Hz. The two hand residual
  action terms provide 56 policy actions total: wrist position, wrist
  orientation residual, and 22 finger residuals per hand.
- The command manager tracks live object pose and recomputes object-centric
  targets. Rewards and metrics cover hand/joint tracking, object tracking,
  contact agreement, lift progress, goal error, and task success.
- The reset command-freeze defect is corrected by default. Set
  `env.legacy_reset_hold=true` only for source-parity experiments.

### PPO and physics backends

- The source RSL-RL PPO configuration is ported: 24 rollout steps, 20,000
  intended iterations, ELU MLP `[1024, 512, 256, 128]`, five epochs, four
  minibatches, adaptive `1e-3` learning rate, `gamma=0.99`, `lambda=0.95`,
  clip `0.1`, entropy `0.001`, and desired KL `0.005`.
- PhysX remains the default backend for backward compatibility.
- Newton is selectable using the G1-style Hydra override
  `physics=newton_mjwarp`. Newton uses MJWarp solver settings and Newton
  contact sensors. PhysX-only mass/gravity accessors have backend-neutral
  fallbacks in the Sharpa wrist and virtual-object controllers.

## Memory cost against the G1 task (2026-09-06)

Measured with `scripts/bench/measure_task_memory.py`: peak GPU memory
attributed to the training process tree, sampled once a second, Newton
backend, PPO, same launcher and iteration count for both tasks. One unrelated
job held 5.5 GB on the same GPU throughout and is excluded by the per-process
attribution.

| Environments | G1 vanilla GPU | Sharpa manipulation GPU | Ratio |
| --- | --- | --- | --- |
| 128 | 1126 MiB | 1668 MiB | 1.48 |
| 512 | 2742 MiB | 4972 MiB | 1.81 |
| 2048 | 8798 MiB | 18468 MiB | 2.10 |

Both scale linearly across that range (largest fit error 49 MiB):

| Task | Fixed | Per environment |
| --- | --- | --- |
| `Isaac-Imitation-G1-v0` | 655 MiB | 4.0 MiB |
| `Isaac-Sharpa-V2D-Source-v0` | 520 MiB | 8.8 MiB |

The fixed cost is the same for both; manipulation costs **2.2 times more per
environment**. That is the scene, not the policy: two 23-body hand
articulations, a mesh object, and 44 contact-filter shapes per environment
against G1's single articulation.

Host memory is not the constraint. Peak resident set stayed between 3.7 and
5.6 GB for both tasks at every environment count.

Extrapolating the fit to the released 4096-environment recipe gives about
17 GB for G1 and about 36 GB for Sharpa. On this 48 GB card the source
environment count fits for manipulation, with little headroom left for a
second job. That extrapolation is an estimate from three measured points, not
a measurement; each cell is one short run.

## Real capture: ARCTIC box grab (2026-09-06)

`s07/box_grab_01` of ARCTIC is loaded, retargeted, and rendered. It replaces
synthbox as the sequence quality can be judged on. The synthbox fixture stays
only as a schema smoke.

- Source: ARCTIC `raw_seqs`, downloaded with the project's own credentialed
  scripts and verified against the official checksums. Objects come from the
  public ArtiGrasp release.
- LOAD ran natively (Docker is absent here) in the sibling Pixi project
  `/home/fwu91/Documents/DexManip/v2d_loader`, which is the only place
  manotorch and chumpy are installed. Output: 725 frames at 30 Hz, hand span
  17.7 cm, minimum fingertip-to-object distance 0.6 mm, contact on 548 of 725
  right-hand frames.
- The ARCTIC box is hinged. Its lid moves at most 0.012 rad in this sequence,
  so the bottom is a documented rigid proxy
  (`scripts/data/make_rigid_object_urdf.py`, 24 by 22 by 34 cm).
- Retarget: `data/dexmanip/sharpa/arctic_box_grab_speed05`, 965 frames at the
  released half-speed playback, contact geometry on 764 frames, IK mean task
  error 1.9 cm and maximum 6.6 cm.
- Videos: `logs/rsl_rl/sharpa_v2d/retarget_videos/arctic_box_grab_*`.

### Parity against the upstream solver on real data

The upstream retarget was run natively on the same loaded Parquet
(`scripts/data/run_source_sharpa_retarget.py`) and compared frame by frame.
The gate that synthbox passed **fails on this sequence**, on the finger
channel only:

| Channel | Result against a 965-frame sequence |
| --- | --- |
| Wrist position, left / right | max 0.27 mm / 0.58 mm |
| Wrist rotation, left / right | max 0.84 mrad / 0.45 mrad |
| Object pose, contact agreement | exact, 1.0 |
| Finger joints | mean 0.00015 rad, 99th percentile 0.0034 rad, max 0.1104 rad |

The finger differences are transient, not a systematic offset: 5 of 965
frames exceed 0.05 rad and the neighbouring frames return to about 0.0003 rad.
That is the signature of a redundant inverse-kinematics problem whose two
implementations occasionally settle in different but equally valid finger
configurations, then re-converge. It is not evidence of an implementation
difference, and it is not yet evidence against one. **The gate threshold for
real, redundant targets is an open decision; do not restate the port as
source-parity on real captures until it is set.**

Report: `logs/rsl_rl/sharpa_v2d/retarget_parity/parity_arctic_source_vs_iltools.json`.

## Source-parity task and RLOpt training (2026-09-05)

The plan is in [sharpa-recipe-port-plan.md](sharpa-recipe-port-plan.md).
Two tasks are registered:

- `Isaac-Sharpa-V2D-PhysX-v0`, the **task variant** with this repository's
  object-goal and success rewards.
- `Isaac-Sharpa-V2D-Source-v0`, the **source-parity variant**: the released
  twelve-term observation (504 values), the 65-value command, friction-cone
  contact-wrench rewards on 17 sensed hand links per hand, released
  terminations, hand material randomization, the `step` virtual-object
  decay with its settling hold, and the six-term curriculum. It refuses
  references not converted with `--motion-speed 0.5`. Contract test:
  `tests/test_sharpa_source_parity_contract.py`.

Both tasks carry `rlopt_ppo_cfg_entry_point` (`SharpaRLOptPPOConfig`),
asserted equal to the RSL-RL recipe field by field. RLOpt gained
`ppo.truncation_bootstrap = "rsl_rl"` (time-outs add `gamma * V(s_t)` and
count as terminal before GAE; unit test against the RSL-RL return loop).

Two defects of the task variant were found and fixed during the port:

- The finger observation scaled reference-ordered joint positions with
  Isaac-ordered joint limits (the two orders differ).
- The PhysX contact sensor pattern `{ENV}/<side>_robot/.*` resolved only the
  palm body, so the boolean contact reward never saw finger contact. The
  parity task senses the object and filters every collision shape of the 17
  links (22 shapes per hand with the fingertip elastomers) and reduces
  shapes to links in the command.

Smokes on the half-speed synthbox reference, all exit 0 with finite rewards:

| Task | Trainer and backend | Shape |
| --- | --- | --- |
| task variant | RLOpt PhysX | 4 envs x 2 it, 8 envs x 6 it |
| source-parity | RLOpt PhysX | 4 envs x 2 it, 16 envs x 12 it |
| source-parity | RLOpt Newton (`train_newton.py`, `physics=newton_mjwarp`) | 4 envs x 2 it, after keeping `replicate_physics=True` (the adapter needs it) |

## Validation completed

| Gate | Result |
| --- | --- |
| ILTools loader, schema, retarget, and core tests | Passed in the focused suite |
| Sharpa task contract and curriculum tests | Passed; 7 contract tests in the latest focused run |
| Ruff on Sharpa task and tests | Passed |
| PhysX PPO smoke, 1 environment, 1 iteration | Passed |
| Newton PPO smoke, 1 environment, 1 iteration | Passed |
| Newton PPO, 4 environments, 20 iterations | Passed; finite rollouts and checkpoint saved |
| Newton checkpoint playback, 20 control steps | Passed |
| Existing/default PhysX smoke after Newton changes | Passed |

The Newton 4-environment run is stored under
`logs/rsl_rl/sharpa_v2d/2026-09-02_09-35-02/`, with checkpoint
`model_19.pt`. It ran at approximately 159 environment steps/second. Mean
returns stayed finite around `-4.8` to `-5.0`, and mean episode lengths were
around three control steps. These numbers establish backend stability only;
they are not a manipulation-success result.

## Retarget parity against the source method

Status: **resolved on 2026-09-02.** The earlier "source versus Pink" mismatch
was an artifact of the comparison target, not a defect in the port.

- The source fixture `synthbox_processed` is **fabricated**. The source
  generator `scripts/make_synthbox_fixtures.py` fills its `robot_*` columns
  with an IK-like placeholder: the wrist equals the raw MANO wrist with an
  identity rotation, the finger joints are a sine ramp, every task error is
  exactly zero, and every frame reports one iteration. The
  "Source/reference retarget" pane of the earlier videos rendered that
  placeholder, so the visible difference (about `0.26 m` wrist and `0.74 rad`
  finger) measured the placeholder, not the source solver.
- Solver-level parity holds. The ILTools test
  `tests/retarget/test_sharpa_source_parity.py` runs the source
  `SharpaHandKinematics` and the ILTools `_SharpaHandSolver` on the same MANO
  frames with the same seed and passes at `1e-3 m`, `1e-2 rad`, and `1e-3 rad`.
- End-to-end parity holds. Running the source script
  `scripts/retarget/synthbox_to_sharpa.py` on `synthbox_loaded` (CPU, viser
  stubbed) and converting its output with the ILTools converter, then comparing
  it with the ILTools `--retarget` conversion through the same rigid proxy and
  20 Hz resampling, gives the following maxima over 102 frames:

  | Quantity | Max difference |
  | --- | --- |
  | Wrist position, left / right | `9.2e-6 m` / `7.8e-6 m` |
  | Wrist rotation, left / right | `1.2e-4 rad` / `1.0e-4 rad` |
  | Finger joint (worst: `left_middle_PIP`) | `3.1e-4 rad` |
  | Object pose | `0` |
  | Contact activity agreement | `1.0` |

  The result is identical for 100 and 200 solver iterations; the synthbox
  frames converge before the source default of 100.
- The Sharpa MJCFs in
  `source/isaaclab_imitation/isaaclab_imitation/assets/sharpa_wave/xmls/` are
  byte-identical to the source `assets/xmls/sharpawave/` files.

Tooling added for this gate:

- `scripts/audit/compare_sharpa_references.py` compares two references frame
  by frame (wrist position and rotation per hand, finger joints, object pose,
  contact activity) and fails on thresholds.
- `ManoSharpaLoader` now records retarget provenance in the reference
  metadata: `retarget_source` (`iltools_pink` or `parquet_robot_columns`),
  solver settings and MJCF hashes when ILTools solved the row, per-hand IK
  task-error and iteration statistics, and `retarget_solution_kind`
  (`ik`, `placeholder`, or `unknown`). `inspect_sharpa_reference.py` prints a
  warning for `placeholder` and `unknown` solutions.
- `SharpaPinkRetargeter.max_iterations` now defaults to 100 to match the
  source dataset scripts (`setup_sharpa_kinematics`), not the 200 of the
  source class default.

`data/dexmanip/sharpa/synthbox_mano_pink_rigid_v2` was regenerated with the
provenance metadata; it passes the parity gate against the source IK output.
The source IK Parquet, its ILTools conversion, the driver that ran the source
script, and the JSON parity report are archived under
`logs/rsl_rl/sharpa_v2d/retarget_parity/`.

What the synthbox result does not show: the fixture hand is a crude planar
skeleton about `9 cm` long, while the Sharpa Wave fingertips sit about `20 cm`
from `hand_C_MC`. With `mano_to_robot_scale = 1.0` the solver trades the
low-cost wrist task against the tip tasks, so the wrist sits about `0.26 m`
from the MANO wrist and the fingers curl. That is the correct source behavior
on an unrealistic input; it says nothing about retarget quality on real MANO
captures. No real MANO capture (TACO, ARCTIC, HOT3D, OakInk2) is present on
this workstation; those `human_motion_data` directories are empty.

The first frame of the Isaac videos can be black because of camera warm-up;
frame samples after startup are the meaningful visual checks.

## Current limitations

- The synthbox source scene contains an articulated box. Sharpa v1 intentionally
  simulates only one rigid body, so the current smoke uses the box bottom as a
  rigid proxy and cannot claim lid-opening fidelity.
- The source fixture is deliberately simple and mostly static in the selected
  rigid body. Its `synthbox_processed` `robot_*` columns are a fabricated
  placeholder. It is useful for contract and parity checks, not for judging
  manipulation or retarget quality.
- Short PPO runs have objective reward terms disabled by the curriculum and
  are dominated by early wrist-away terminations. No task-success or policy
  quality claim has been established.
- The broad repository test suite still has unrelated collection/runtime
  failures when optional `loco_mujoco` or MuJoCo OpenGL dependencies are not
  available.

## Newton runs the task (2026-09-07)

The 2026-09-06 failure had one cause and it was not in the recipe or the data.
Isaac Lab `3.0.0b2.post1` composes external wrenches into the body frame and
its Newton assets copy that body-frame wrench straight into Newton's
world-frame `state.body_f`. Every Newton wrench landed rotated by the inverse
body orientation: the floating hands drifted 0.10 to 0.16 m per control step
and the virtual object left at 1.9 m/s with nothing touching it. Proof, on a
contact-free pinned reset: a world +Z force equal to the object's weight,
passed as global, gave (0.321, -0.371, -0.488) m/s; the same numbers passed as
local gave exactly zero. `isaaclab_imitation/newton_wrench_frame.py` installs
a version-gated correction at `isaaclab_imitation.tasks` import; with it the
proof inverts. Details and the ruled-out explanations are in the campaign
README, Defect 2. G1 Newton results from before this date used misframed push
events.

Two more divergences from the release were found and matched on the way: the
object collider (importer default convex hull, release convex decomposition;
against the hull the penetration audit reads 35.17 mm, against the
decomposition 15.84 mm) and the object's rigid-body properties
(`max_contact_impulse`, velocity caps, zero offsets).

Gate, 8 environments, 40 steps, Newton, default contact stiffness:
**PASSED**, zero terminations, 12.1 N peak contact. Local training smoke, 512
environments, 60 updates: `ep_len=455` (was 1.0), `r_step=-0.0141` (was NaN),
`contact_wrench_support_reward` 0.153 (was 0), time-out terminations 99 %.
Qualification only.

What did not fix it, each measured: soft contact (solref 0.05) cut the reset
impulse from 54 N to 9 N in MuJoCo but the gate still failed; contact settling
makes grasp-frame penetration worse because the decomposed walls are 8 to
10 mm thick and the retarget goes up to 15.8 mm in; running the upstream
retargeter natively reproduces the same penetration to 0.002 mm. PhysX remains
unable to filter these contacts on the GPU (per-collider entries, two
colliders on several links; campaign README, Defect 5).

## Neither backend runs the task yet (2026-09-06)

The Skynet calibration job 3771925 completed cleanly at 4096 environments,
kit-less, 3673 frames per second, and produced **no learning signal**:
`r_step=nan`, `ep_len=1.0000`, every `Episode_Reward/*` term zero. The local
512-environment Newton qualification run had the identical signature from
iteration 1 to iteration 650 before it was killed. Neither run says anything
about the recipe. Three defects were found instead.

**Defect 1, fixed.** `object_body_prim_path` assumed a USD proxy carries its
rigid body on the asset root. A USD written by Isaac Lab's own URDF converter
keeps the importer's `Geometry/<link>` nesting, so the scene pointed its rigid
object view and its contact sensor at a prim with no rigid body. PhysX logged
`Failed to find rigid body` and the contact sensor failed to initialize. The
path is now read from the stage, in a child interpreter, because opening a USD
stage before Isaac Sim starts crashes Kit inside `SimulationApp._start_app`.
Four tests cover it.

**Defect 2, open: Newton injects a penetration impulse at reset.** The
retargeted reset pose puts the fingers slightly inside the object. On the
identical scene with zero actions, first control step:

| Quantity | PhysX | Newton |
| --- | --- | --- |
| peak contact force | 7.5 N | 51.1 N |
| summed contact force | 10.9 N | 102.3 N |
| object angular velocity | 2.2 rad/s | 30.0 rad/s |
| object orientation error | 0.060 rad | 0.431 rad |
| wrist error after one step | 0.043 m | 0.157 m |

The object weighs 0.3 kg, so Newton applies about seventeen times its weight.
Both termination thresholds are crossed within one or two control steps.

Ruled out, each by measurement: a port error in the tracking gains (the
released values were read from the upstream checkout and match exactly:
linear stiffness 50.0, angular stiffness 12.0, angular damping 0.1,
`sim.dt` 0.01, `decimation` 5); a quaternion-convention mismatch (Newton
reports `root_link_quat_w` in XYZW, as assumed, and the controller's own
torque stays under 1 N m); the external-wrench frame (a world +Z force of
exactly `m * g` with `is_global=True` holds the hand at 0.000 m/s and
0.000 rad/s under Newton); and an explicit-integration timestep limit
(halving the step helps, quartering it makes the spin worse, which is the
opposite of a timestep instability). The fix belongs in the reset pose or in
Newton's contact settings.

**Defect 3, open: PhysX GPU cannot filter these contacts.** At 512
environments PhysX rejects all 44 hand collision shapes used as contact
filters, `GPU contact filter for collider ... is not supported`, once per
shape per environment, then faults with `CUDA error: misaligned address`
inside `GpuRigidContactView`. This is the fault previously recorded only as a
CUDA illegal memory access. Filtering on the links' rigid-body prims instead
removes the warning but matches no rigid contact entry at all, so the sensor
cannot initialize; the shape-level filters are kept. PhysX's *net* contact
forces work, only the filtered per-link matrix is empty.

**The object collider was wrong (fixed 2026-09-07).** Our port spawned the
object with Isaac Lab's default `collision_type="Convex Hull"`. The release
asks for `collider_type="convex_decomposition"`. For the ARCTIC box, a
container, the hull fills the cavity, so a hand reaching inside reads as deep
penetration that is not there. The port also omitted the release's
`max_contact_impulse=1e3`, `retain_accelerations`, `enable_gyroscopic_forces`,
`max_linear_velocity`, `max_angular_velocity`, and its zero contact and rest
offsets. All are now matched.

The importer's `collision_type` does not survive this repository's
asset-transformer and multi-physics conversion steps, which rewrite
`physics:approximation` back to `convexHull`, so
`convert_sharpa_assets_to_usd.py` now sets the token on the written payload
layers and prints how many meshes it changed. The result is greppable in the
USD.

**The penetration is real, but smaller than first measured.** The earlier
35.17 mm figure was taken against the convex hull and was inflated by the
cavity. Measured against the convex decomposition, which is what Isaac
simulates, on 965 frames:

| Quantity | convex hull | convex decomposition |
| --- | --- | --- |
| deepest penetration | 35.17 mm | **15.84 mm** |
| mean penetration | 6.47 mm | **6.19 mm** |
| frames over 2 mm | 688 of 965 | 688 of 965 |

Every penetrating link is a finger, never the palm; the pinky distal phalanx
is worst.

Fixing the collider moved the Newton gate from 45.37 N to 31.75 N peak contact
but it still fails: 6 of 8 steps terminate, mean episode length 0.50.

**Why pressing the fingers onto the surface does not fix it.** The settle
implemented in `iltools/retarget/sharpa_settle.py` makes penetration worse,
identically under every solver setting tried (timestep 0.002 down to 0.0002,
60 to 500 steps, actuator stiffness 1x to 10x): on the grasp frames it goes
from 13.36 mm to 35.03 mm with joint shifts up to 1.4 rad. On the reach frames,
where nothing touches, it is a clean no-op with 0.003 rad of shift, so the
machinery is correct.

The reason is geometric. The decomposed box walls are **8 to 10 mm thick** and
the retarget penetrates by up to 15.8 mm, nearly twice the wall thickness. Those
fingers are not inside a wall, they are through it. Depenetration cannot
recover the intended side, and pressing the finger toward the same target
drives it further out the far side. A settle that presses can only fix a pose
that is still on the right side of the surface.

**How the release handles this, read from the upstream checkout 2026-09-06.**
It does not repair penetration. Its retargeter has no collision awareness at
all; `collision` appears only in its viser visualizer. Instead
`scripts/filter_penetrations.py`, exposed as the `hand_penetration` data
quality check, *rejects* a whole sequence when any frame exceeds 2 cm. It
measures capsule and sphere primitives from `*_sharpa_wave_primitive.urdf`
against the object's **convex hull**, and it deliberately skips any object
whose hull volume exceeds three times the mesh volume, to avoid false
positives on hollow objects such as mugs and bowls.

That check cannot see our penetration. Run against this reference with the
upstream code and the upstream threshold, it returns
`{'pass': True, 'score': 0.0, 'reason': 'ok (max=0.00cm)'}` at both stride 3
and stride 1, because the ARCTIC box bottom is a container whose hull-to-mesh
volume ratio is **7.03**, above the 3.0 cutoff. The object is skipped and the
sequence passes vacuously. A convex hull also cannot express the real problem:
a finger inside the box cavity is inside the hull but not through a wall.

`dataset_s07_box_grab_01` is the release's own documented example sequence, so
the release trains on this exact data with this exact penetration. The
decisive difference is therefore **the backend, not the data**: on the
identical pose PhysX answers with 7.5 N and Newton with 51.1 N. The released
recipe is a PhysX recipe and PhysX absorbs it.

**Tested directly, 2026-09-06: running the upstream pipeline does not help.**
The upstream retargeter was run natively on the same ARCTIC sequence
(`scripts/data/run_source_sharpa_retarget.py`), converted with no further
retargeting, and measured with the same tool. It is a genuinely different
reference (different SHA, `qpos` differs by up to 0.1104 rad, matching the
earlier parity finding) and it penetrates by the same amount:

| Reference | deepest | mean | frames over 2 mm |
| --- | --- | --- | --- |
| upstream retarget, native | 35.1692 mm | 6.4676 mm | 688 of 965 |
| ILTools Pink port | 35.1692 mm | 6.4692 mm | 688 of 965 |

Its own filter passes it vacuously as well: both box parts have hull-to-mesh
volume ratios of 7.03 and 4.71, over the 3.0 cutoff, so the check is skipped
and returns `{'pass': True, 'score': 0.0}` on the upstream pipeline's own
output. And it fails the Newton backend gate the same way as our reference:
peak contact 45.37 N, a termination on all 6 of 6 steps, mean episode length
0.50 control steps.

This is a useful parity result in its own right: the ILTools port reproduces
the upstream retarget's contact geometry to within 0.002 mm of mean
penetration. It also closes the option of adopting the upstream pipeline to
fix this. The release's answer is PhysX, not better data.

Contact settling (`iltools/retarget/contact_settling.py`, never called by
`convert_mano_sharpa_to_iltools.py`) remains a genuine improvement and is what
Newton would need, since it uses real collision geometry and handles concave
objects. It is a deliberate deviation from the release, not a parity fix.

**PhysX is not a fallback.** Even at 4 environments, where it passes the
backend smoke gate, its filtered per-link contact is 0.00 N on every step, so
`contact_wrench_support_reward` and `unintended_contact_penalty` are
permanently zero while `missed_contact_penalty` accumulates. The recipe's
central reward cannot fire. Newton's filtered contact matrix works.

**Gate.** `scripts/audit/check_sharpa_backend_smoke.py` reproduces all of this
in about a minute and must pass before any cluster submission:

| Backend | Verdict | Peak contact | Mean episode length |
| --- | --- | --- | --- |
| PhysX | PASSED | 2.4 N | 3.5 control steps |
| Newton | FAILED | 24.6 N | 0.5 control steps |

## Next work queue (2026-09-06)

Everything below is blocked until one backend passes
`scripts/audit/check_sharpa_backend_smoke.py`. No cluster submission until
then.

1. Measure the actual finger-object penetration depth at reset on
   `arctic_box_grab_speed05`, per frame. The retargeter already exposes a
   contact settle depth; decide whether the reset pose is retargeted out of
   penetration or Newton's contact response is softened.
2. Re-run the backend gate on Newton after each change. It is the acceptance
   test, not a diagnostic.
3. Fix the Skynet batch script to sample GPU memory *during* a job. The
   calibration job sampled only before and after, so both reads were 0 MiB and
   the 4096-environment fit on a 48 GB card is still an extrapolation.
4. Resolve PhysX GPU contact filtering (Defect 3) if PhysX is to be the
   backend. That is the release's own backend, so it is the parity-correct
   choice if it can be made to scale.

## Earlier work queue (2026-09-05)

1. RSL-RL versus RLOpt matched local blocks on the source-parity task (same
   seed, same frame count), then the D3 and D4 rungs of the plan.
2. Real capture: native LOAD stage (no Docker here), support-surface and
   penetration ports, promotion gate. See the plan page.

## Earlier work queue (2026-09-02)

1. Done 2026-09-02: source parity is proven at the solver and reference level
   (see above). Keep `compare_sharpa_references.py` as the gate for any
   solver, MJCF, or resampling change.
2. Obtain one real MANO capture (a TACO or ARCTIC sequence with a single rigid
   object) and run the same gate on it. That is the first input on which
   retarget quality, not only parity, can be judged. Choose
   `mano_to_robot_scale` from the source dataset script for that capture.
3. Promote the real contact-rich snack-box sequence to a fully checked rigid
   proxy/reference and verify contact timing in both PhysX and Newton.
4. Run longer Newton PPO with the objective curriculum activated, then compare
   trajectory return, episode length, object-goal error, lift, and success—not
   just backend throughput.
5. Scale environment count only after the 4-environment Newton run remains
   stable on the target machine; the earlier PhysX 64-environment attempt hit
   a CUDA illegal-memory failure.
