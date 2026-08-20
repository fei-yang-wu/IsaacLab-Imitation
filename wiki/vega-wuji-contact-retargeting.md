# Vega-Wuji Contact Retargeting

Status as of 2026-08-20. This page owns the retargeting half of
`Isaac-Imitation-Vega-Wuji-v0`: how a SOMA human-object demonstration becomes a
Reference whose hands actually touch the object, and why that did not work
before.

Read this page before changing
`scripts/data/convert_soma_g1_parquet_to_vega_wuji.py`, the ILTools
`iltools/retarget/` contact modules, or any clearance gate.

All numbers below come from one sequence
(`2026-03-23_18-10-01_corn_can_right_left_handover_01`) and one configuration
each. They are **preliminary**: no repeated seeds, no frozen protocol, one
object. Treat them as signs that tell you where to look next.

## The metric

Every result on this page uses the same measurement, so the arms are
comparable:

> **Contact recovery rate** = the fraction of the 28 source-active hand frames,
> in the emitted 32-frame Reference, for which `recover_contact_sequence`
> measures real contact geometry.

A hand frame is source-active when the Parquet's `hand_contact_active` flag is
set for that hand at that frame. There are 19 left and 9 right.

## Why there was no contact

The converter guaranteed non-contact by construction. Two mechanisms did it.

1. `_project_palm_positions_nonpenetrating` pushed each palm radially outward
   to at least `DEFAULT_PALM_OBJECT_STANDOFF_M = 0.28` m from the object
   **centre**. The function name states the intent: "project palm origins to
   conservative non-contact scene standoffs".
2. `_collision_safe_finger_projection` then shrank finger closure until a 5 mm
   clearance gate passed. At 0.28 m that gate is trivially satisfied, so both
   hands kept closure scale 1.0 and nothing was detected as wrong.

Measured consequence on the original Reference: during **all 28** source-active
hand frames, no fingertip came within 5 cm of the can. The closest any
fingertip ever got, over the whole trajectory, was **+9.5 mm**, and that
happened at a frame where neither hand is source-active. Contact recovery was
**0 of 28**.

The missing contact-geometry columns in the Parquet are therefore **not** the
binding constraint. The kinematics are. Even perfect source contact data would
have had nothing to attach itself to.

## What the source data actually says

The decisive measurement was a frame audit of the Parquet, not a new algorithm.

`ee_pose_w` holds the G1 palm-link poses and shares the object's
`robot_base_z_up` frame. During source-active frames:

| Palm | Minimum to can centre | Median to can centre |
| --- | ---: | ---: |
| `left_hand_palm_link` | 0.091 m | **0.112 m** |
| `right_hand_palm_link` | 0.075 m | **0.109 m** |

The can radius is 0.067 m, so the palm sits about 4 cm off the surface while
grasping. That is physically sensible, and it means the source is coherent.

The converter was placing the palm at 0.28 m: **2.5 times further away than the
source data says**. It was also applying `global_object_motion_scale = 0.4` and
`local_hand_object_geometry_scale = 0.75`, which shrink the object's travel and
the scene geometry while the hands follow a differently scaled path. That
decoupling is what kept the hands off the can.

## What was built

Two ILTools modules on `dev-dex`, both fail-closed and unit-tested against
analytically known sphere geometry.

- **`iltools/retarget/contact_recovery.py`** (10 tests). Measures contact
  geometry from a MuJoCo witness-segment query: the contact point on the link,
  the point on the object, and unit normals along the separation axis. A
  source-active frame whose fingertips do not reach stays **inactive** and is
  reported as unreached, because a binary flag is not evidence of contact.
  Penetration deeper than the limit raises. The output is a strict lower bound
  on the true contact set.
- **`iltools/retarget/contact_settling.py`** (7 tests). DexMachina's functional
  retargeting: replay the retarget as soft position targets against a fixed
  scene and let contact push the hand onto the surface. Gravity is off, because
  the correction is geometric. Every frame starts at its own retarget, never at
  the previous settled pose, so actuator lag is never recorded as a contact
  correction.

In the converter, `--contact-seeking` makes contact reachable:

- The scene audit gained a gate-scene selection. A clearance gate on the can
  forbids the very thing the task needs, so contact seeking **measures and
  reports every scene but gates none** during projection. The default path is
  unchanged and still gates both scenes.
- The projected trajectory is then settled, and the settling report is written
  into the Reference metadata under `geometry_clearance_audit.contact_settling`.
- `--contact-settle-steps` exposes the settle depth. Penetration converges by
  60 steps; 200 and 600 change nothing.

## Result: unscaling the motion beats soft replay, six times over

| Configuration | Min distance | Frames | Contacts | Rate |
| --- | ---: | ---: | ---: | ---: |
| Baseline, 0.28 m standoff | +9.5 mm | 0/28 | 0 | 0% |
| A: contact-seeking gates only | +1.3 mm | 1/28 | 1 | 3.6% |
| A + settling, 0.02 m standoff | +11.6 mm | 0/28 | 0 | 0% |
| **B: `--global-motion-scale 1.0 --local-geometry-scale 1.0`** | −11.7 mm | **6/28** | **16** | **21.4%** |
| B + settling, 60 steps | −2.2 mm | 6/28 | 16 | 21.4% |

Both scales were already command-line flags. **The win needed no new code**; it
needed the frame audit that showed the scales were wrong.

**Settling is a complement, not a competitor.** Once the retarget reaches the
object, soft replay removes 81% of the penetration (−11.7 mm to −2.2 mm) for a
mean joint shift of only 0.013 rad, and costs no contacts. It appeared harmful
in arm A only because the retarget was not reaching: it had no penetration to
resolve and merely nudged self-collisions apart.

## The blocker now: the arm cannot reach

Measured on the best Reference:

- Wrist IK position error, left arm: **max 0.223 m, mean 0.072 m, p95 0.222 m**.
  Right arm: max 0.093 m, mean 0.027 m.
- `Lift` is saturated in **19 of 32** frames and `torso_flip` in **18 of 32**.

The left-arm error matches the palm-gap inflation exactly. Against the source's
own palm-to-can distance, the emitted wrist is **2.24 times further on the left
and 1.54 times further on the right** during contact frames. The arm is pinned
at its travel limits and simply cannot get closer.

Moving the object anchor changes which arm succeeds but not how many frames
succeed:

| `--object-anchor-target` | Left IK max | Right IK max | Left frames | Right frames | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.80 0.15 1.14 (default) | 0.223 m | 0.093 m | 3/19 | 3/9 | 6/28 |
| 0.65 0.15 1.14 | 0.109 m | 0.046 m | 6/19 | 0/9 | 6/28 |
| 0.80 0.15 1.25 | 0.253 m | 0.145 m | — | — | 6/28 |

Halving the IK error did not raise the total. **One rigid object anchor cannot
put both handover phases inside the workspace**: the left phase and the right
phase happen at different object positions, and the Vega 1U's two arms have
different reachable regions. This is an embodiment limit, not a parameter to
tune.

A falsified hypothesis worth recording: the 0.20 m palm-support floor is **not**
the cause. Lowering it to 0.10 m or 0.05 m made recovery worse, 6/28 down to
2/28, because the can sits about 0.25 m above the support top and a lower palm
moves away from it.

## Plan forward

Ordered by expected value. Each step names the gate that decides whether it
worked.

### 1. Phase-aware object placement

The measured trade-off says a single anchor serves one arm at a time. Replace
the one rigid `object_anchor_target` with a placement that respects the
handover phases, either by solving one anchor per contact phase and blending
across the handover, or by solving the anchor that maximises joint headroom
over the whole trajectory instead of being hand-set.

*Gate*: total recovery rises above 6/28 with left and right both non-zero, and
`Lift` / `torso_flip` saturation falls below 19 and 18 frames.

### 2. Reachability-first retargeting

Today the pipeline places the scene, then discovers the arm cannot reach.
Invert it: measure the Vega 1U reachable workspace once, then choose the scene
transform so the demonstration's contact phases land inside it. This is what
"workspace projection" was supposed to mean.

*Gate*: left-arm wrist IK p95 error below 0.05 m without raising
`--max-wrist-position-error` above its 0.05 m default.

### 3. Finish the CHORD contact targets

Contact targets tell the fingers where to touch. Without them the settle has to
guess.

`py-soma-x` 0.2.1 is public on PyPI (Apache-2.0) and
`soma.assets.get_assets_dir()` pulls the HuggingFace bundle, including
`SOMA_neutral.npz` and `correctives_model.pt`. It reconstructs an
18,056-vertex human mesh with 2,925 vertices for each hand. Build the layer with
`enable_procedural_transforms=False, load_correctives_model=False` and call it
with `apply_correctives=False`.

**Unresolved**: driving the layer from the Parquet payload needs a 77-to-78
joint index mapping. The payload uses the 77-name `SOMA_JOINTS_ORDER` from
`robotic_grounding/retarget/params.py`, the layer's `joint_parent_ids` has 78
entries, and `public_joint_names` has 105. Every attempt so far leaves a
0.78 to 0.85 m joint-fit residual, which means the pose, not just the frame, is
wrong.

With the mesh posed correctly, apply CHORD's own rule: mark contact where an
object surface vertex is within **1 cm** of a hand vertex
(`approximate_contact_with_id`), then assign each contact to the nearest robot
link.

*Gate*: the joint-fit residual falls below 0.05 m, and the recovered human
contact frames agree with `hand_contact_active` on most source-active frames.

### 4. Contact-guided finger inverse kinematics

With targets from step 3, solve the finger joints so the fingertips reach the
contact points, instead of searching one uniform closure scale for each hand.
The present search only ever *reduces* closure to satisfy a clearance gate; it
can never close a gap.

*Gate*: recovery rate above 50% with penetration after settling better than
−2 mm.

### 5. GRAIL as the fallback

If steps 1 to 4 do not close the gap, adopt GRAIL's shape: reconstruct metric
4D human-object trajectories with interaction-aware optimisation and retarget
the human **and the object together**, rather than treating object motion as a
fixed input. This is the agreed fallback and it directly addresses the failure
mode above, where the object placement and the arm workspace are solved
separately.

### 6. Wire the qualification gate

`verify_training_qualification` needs a `ContactSequence` with at least one
active slot carrying finite non-zero positions and unit normals.
`recover_contact_sequence` already emits exactly that shape. Once recovery is
high enough to be useful, feed it into the converter output so a Reference can
stop being `inspection_only`.

*Gate*: a Reference passes `verify_training_qualification` without any manual
edit to its metadata.

## Open defects

- The in-converter settle also fights the support cylinder and only reaches
  −13.3 mm residual penetration, while a can-only settle on the same trajectory
  reaches −2.2 mm. The settle needs either more authority or a staged pass.
- The retargeted hand pose is **self-colliding**. On the emitted 32 frames the
  settle moves only 6 of 59 joints, all thumb or finger, `l_thumb_cmc_abd` by
  0.82 rad, with no object penetration present. The finger map produces invalid
  poses independent of the object.

## Traps that have already cost wrong conclusions

- **`fixed_root_pose_w` is `[0, 0, 0.19]`, not identity.** The robot MJCF base
  sits at its own origin while the Reference stores world poses about that root.
  Any MuJoCo query must express object poses in the robot base frame first;
  `audit/vega_wuji_scene_clearance.py` does this with `relative_pose_wxyz`.
  Skipping it puts the can 19 cm too high and fabricates a penetration that is
  not there.
- **`soma_joints` in the Parquet payload is root-normalised in the human
  frame**, not the object frame. Comparing it against object poses directly is
  meaningless. Use `ee_pose_w`, which does share the object frame.
- **MuJoCo can return a false zero** for separated mesh pairs queried with a
  large `distmax`. Cap the query at the threshold you care about. A stable value
  across several cutoffs is trustworthy; a value that only appears at a large
  cutoff is not.

## Rerun commands

From the repository root. The conversion needs the Isaac Lab environment for
`pyarrow`.

```bash
# Best current configuration, 6/28 frames and 16 contacts.
pixi run -e isaaclab python scripts/data/convert_soma_g1_parquet_to_vega_wuji.py \
    --output logs/reference_replay/vega_wuji/contact_seeking/reference.npz \
    --contact-seeking --inspection-only \
    --palm-object-standoff-m 0.02 \
    --global-motion-scale 1.0 --local-geometry-scale 1.0 \
    --geometry-clearance-m 0.001 --max-wrist-position-error 0.30
```

```bash
# Independent clearance audit; never trust a stored audit block.
pixi run python -m imitation_experiments.audit.vega_wuji_scene_clearance \
    <reference.npz> --output audit.json
```

The contact modules are exercised by the ILTools suite:

```bash
pixi run python -m pytest -q ImitationLearningTools/tests/retarget
```

## Related pages

- [Project Live Status](current-status.md)
- [Context Management](context-management.md): which repository owns an edit.
  The contact modules are ILTools; the converter and gates are this repo.
