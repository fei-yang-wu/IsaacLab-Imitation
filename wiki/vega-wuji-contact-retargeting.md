# Vega-Wuji SOMA/CHORD Contact Retargeting

Status as of 2026-08-24: the corn-can handover was **rejected as a source
motion** after visual review, and the converter now also carries the SOMA
`snack_box_pick_and_place_01` bimanual grasp+lift. Switching source fixed the
arm-side defects outright: wrist residuals `0.6-2.3 mm`, robot self-collision
from `-14.2 mm`/412 violating pair-frames down to `-2.8 mm` or `0.00 mm`, and
support penetration (`-13.3 mm` on the handover) gone entirely. Arm posture is
natural in review: bent elbows at waist height, no overhead branch.

Finger accuracy was then refined twice. Adopting the notes' finger-direction
palm alignment (definition B, `--align-finger-directions`, default on) gave
fingertip p95 `20.9/17.9 -> 14.3/12.2 mm`. Authorizing the shared `Lift` and
`torso_flip` axes in the dual-wrist solve (`--shared-positioning-axes`,
default on) then gave **`0.36/0.37 mm`**. Both hands now pass all four finger
gates.

The remaining blockers are geometric, not kinematic: hand-into-box
penetration and robot self-collision. Every artifact is inspection-only.
**Do not train from or promote any of them.**

The earlier bent-elbow corn-can candidate remains useful arm-placement and
visual evidence, but it predates the adopted anatomical wrist and five-tip
recipe. Its geometry pass must not be presented as a pass of the colleague's
four gates.

This page owns the boundary between human motion reconstruction, robot
retargeting, and RL policy training for `Isaac-Imitation-Vega-Wuji-v0`.

## The overall imitation-learning picture

A reconstructed human motion is not yet an executable robot rollout. The full
pipeline has four distinct jobs:

1. **Human reconstruction:** recover metric wrist/hand motion, object motion,
   and source contact evidence from SOMA/CHORD.
2. **Kinematic robot Reference:** map the important task-space motion to the
   robot and solve a reachable, visually plausible joint trajectory.
3. **Contact augmentation:** fit robot fingers to contact evidence without
   invalidating the accepted arm trajectory, then verify collision and contact
   geometry.
4. **RL control:** train a residual policy around the Reference so the simulated
   robot tracks it under gravity and contact while virtual object assistance is
   reduced to zero.

The environment's action adapter implements
`q_target = clamp(q_reference + filtered_residual, soft_limits intersected with
the Wuji anatomical envelope)`. The Reference is therefore a nominal motion
and reward target. It does not need to be an unassisted open-loop controller,
but it must be physically coherent enough for RL to learn local corrections. A
visually implausible or discontinuous Reference is a poor curriculum even if
isolated geometry checks pass.

## Adopted Wuji recipe

The archive at
`/home/fwu91/Documents/DexManip/wuji_retargeting_notes.tgz` has SHA-256
`b9da2d008ccfa6a881bb67bdfa1475c60879471d4d210a9e4d03d68d63e8c0b3`.
Its seven extracted files match the existing
`/home/fwu91/Documents/DexManip/wuji_retargeting_notes/` directory byte for
byte. The archive reports a successful MANO-to-Wuji corpus result, but it does
not contain the source sequences, commands, or machine-readable audit output.
Treat that corpus result as upstream evidence, not as reproduced evidence in
this repository.

The successful mapping is named `v1` in the notes, although it targets the
Wuji Hand 2 V2 asset. The mapping named `v2` is an unsuccessful intermediate
link experiment. The adopted recipe is:

- solve the wrist and five fingertip positions; do not imitate human joint
  angles, knuckles, intermediate links, fingertip directions, or coupling;
- use wrist position/orientation costs `1.0/0.03`;
- use fingertip costs `1, 1, 1, 1, 0.5` from thumb through pinky;
- tighten PIP to `0..100 deg`, DIP to `0..80 deg`, thumb MCP to a `0 deg`
  lower bound, thumb IP to `0..100 deg`, and non-thumb MCP abduction to
  `+/-40 deg` before IK;
- use a `0.002` slight-curl posture prior;
- keep source-to-robot fingertip scale exactly `1.0`;
- solve sequentially with the previous frame as the warm start and enforce a
  trajectory velocity bound;
- obtain palm yaw and palm-normal offset from identity-specific, zero-pose MCP
  geometry, not posed motion behaviour;
- recompute all evidence from final FK after every later contact or collision
  projection.

The repository implementation uses bounded MuJoCo damped least squares with a
descent line search. It solves all 318 source samples in order, then samples the
result at the 20 Hz Reference times. The `34.99 deg` emitted-frame limit becomes
`3.499 deg` per 200 Hz source step, so ten source steps cannot add up to an
invalid Reference jump. It preserves the task costs and constraints above, but
it is not numerically identical to the notes' Pink, Pinocchio, and DAQP solver.
That difference is recorded in conversion metadata.

Each hand must pass all four raw-value gates:

| Gate | Limit |
| --- | ---: |
| Mean wrist position residual | `<= 2.5 mm` |
| p95 of the per-frame mean of five tip residuals | `<= 5 mm` |
| Maximum adjacent-frame joint jump | `<= 35 deg` |
| Frames with more than `5 deg` summed IP hyperextension | `0%` |

Missing measurements fail. Rounded values are presentation only. Offline
conversion emits an inspection NPZ Reference without `TrainingQualification`.
Full-horizon Newton state-lock and zero-virtual-controller unassisted replay
define the required hash-bound evidence for a future promotion tool; no such
attestation writer exists yet, and the task rejects training until one exists.
Both Vega-Wuji residual action adapters also intersect the live soft limits
with the same anatomical finger envelope, so a policy residual cannot
reintroduce IP hyperextension at runtime.

### Current pinned-clip diagnostic

The 32-frame, 20 Hz command below uses the adopted method and records complete
final-FK measurements. It is a rejected diagnostic, not a successful retarget:
Reference SHA-256:
`9c0a5df6075444f453cb9e826d0a318b23a894a9e0ce0d11042190bb591d6ca1`.

| Metric | Left | Right |
| --- | ---: | ---: |
| Mean wrist residual | 0.375 mm | 0.152 mm |
| Tip residual p95 | 8.301 mm | 1.933 mm |
| Maximum joint jump | 34.99 deg | 19.92 deg |
| Hyperextension frames | 0% | 0% |

The left hand fails only its fingertip gate. The right hand passes all four
hand gates. The emitted motion also fails geometry: its worst robot
self-collision is `-14.195 mm` with 412 violating pair-frame contacts across
27 frames, and the left fingers penetrate the support by `-13.283 mm` at
emitted frame 10. Can clearance and can/support clearance pass.

The SOMA chain audit found an earlier off-by-one error: non-thumb `*1` is the
metacarpal base and `*2` is the MCP. Calibration now evaluates the payload's
identity and scale at zero local pose and uses the four `*2` points. The
resulting fit is about `0.871/0.872` scale, `2.276/2.278 mm` RMS, and `-16.3
deg` yaw for left/right. This fitted scale is a geometry diagnostic only;
fingertip targets remain at the required scale `1.0`.

A read-only, single-clip Pink/Pinocchio/DAQP probe used the same 318 targets,
costs, anatomical limits, posture prior, arm lock, and emitted-cadence bound.
It produced `7.712/1.934 mm` left/right tip p95 and `34.98/19.89 deg` maximum
finger jumps. It therefore did not clear the left-tip gate. Pinocchio also
needed a semantics-preserving MJCF element-order workaround to load this
asset. The probe shows that Pink is feasible, but it does not support changing
the production solver before the target/geometry failure is resolved.

Reproduce this fail-closed diagnostic from the repository root:

```bash
pixi run -e isaaclab python \
  scripts/data/convert_soma_g1_parquet_to_vega_wuji.py \
  --source /home/fwu91/Documents/DexManip/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/human_motion_data/whole_body/soma/sequence_id=2026-03-23_18-10-01_corn_can_right_left_handover_01/robot_name=g1/data.parquet \
  --support /home/fwu91/Documents/DexManip/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/human_motion_data/whole_body/reconstructed_stage/2026-03-23_18-10-01_corn_can_right_left_handover_01_support.usda \
  --output /tmp/vega_wuji_adopted_method_diagnostic.npz \
  --phase-aware-object-placement \
  --global-motion-scale 0.6 \
  --bent-elbow-inactive-rest \
  --filter-cutoff-hz 10 \
  --inspection-only

pixi run python scripts/audit/audit_wuji_finger_joints.py \
  --reference /tmp/vega_wuji_adopted_method_diagnostic.npz \
  --anatomical --require-gates
```

The conversion completes as inspection evidence; the audit intentionally exits
nonzero with the left fingertip failure listed. Adding
`--require-finger-gates` to conversion instead rejects before writing the
Reference.

## Source selection: why the handover was replaced

Visual review of the corn-can adopted-method render rejected it: the right
elbow cranked to head height with a contorted wrist, the can floated beside
the hand, and the left hand splayed open through the tabletop. The measured
cause is the source motion, not the solver. A **handover** forces cross-body
reaches: the two source wrists close to `0.16 m` of each other, which the
fixed-base dual-arm Vega cannot express, and it needed heavy scene remapping
(`0.4-0.6` global motion scale plus per-phase anchors) on top.

The converter therefore carries a **sequence registry** (`PINNED_SEQUENCES`,
selected with `--sequence`). Each entry owns its source Parquet and support
USD with their SHA-256 digests, the crop, the support cylinder copied verbatim
from the support USDA `Cylinder` prim, the object asset digests, and the
runtime object name. `corn_can_handover` reproduces the historical pins
exactly; the legacy `PINNED_*` module names are aliases of that entry.

`snack_box_pick` was selected on measured source geometry, not preference:

| Property | corn-can handover | snack-box grasp+lift |
| --- | ---: | ---: |
| Minimum L-R source wrist gap | 0.16 m | 0.41-0.44 m |
| Vega neutral palm spacing | 0.458 m | 0.458 m |
| Object size (mesh extents) | 0.07 x 0.07 x 0.11 m | 0.43 x 0.48 x 0.25 m |
| Contact frames in crop (L/R) | 19 / 9 | 56 / 55 |
| Object XY travel in crop | - | 0.287 m |

The frozen crop is `[400, 468)` at 200 Hz: pre-grasp reach, both contacts
closing from about f440, and the lift (`+0.13 m`). It stops before the carrier
turns and walks, which is what tilted the box and dragged the arms toward
their limits in the first `[400, 480)` attempt.

Rejected alternatives: `blue_trash_can_drag` is a floor-level task
incompatible with the tabletop scene, and `synthbox_box_open` needs an
articulated lid the converter's single-rigid-object scene does not model.

## Finger accuracy: what was actually binding

With the arms fixed, the fingers became the binding constraint. This section
records the investigation in the order it happened, because two of the
intermediate conclusions were wrong and the wrong ones are instructive. The
answer is in "The shared positioning axes were the binding constraint" below;
read that first if you only want the result.

**The Wuji hand is longer than this human's hand.** Per-finger wrist-to-tip
excess (Wuji neutral minus source p95) is `23 / 46-48 / 47-48 / 56-57 /
60-63 mm` for thumb through pinky. Pinning the robot palm at the human's
anatomical wrist therefore asks longer robot fingers to reach nearer targets.

**The grasp posture is at the abduction hardware limit.** All four fingers on
both hands sit at `+34` to `+37 deg` mean MCP abduction, pinned at the `+40
deg` cap for 75% of frames, and never negative. A uniform same-sign bias is
the notes' Pitfall 2 signature, so the palm yaw was re-verified directly: the
fitted correction lands the source knuckles on the Wuji MCP origins to within
`+/-1.2 deg` per knuckle at `2.32 mm` RMS. The palm frame is correct, so the
`+35 deg` demand is the real posture of this grasp, and `+/-40 deg` is the
Wuji URDF *mechanical* limit, not an anatomical choice. There is no headroom.

Two rigid corrections were measured. `--palm-object-standoff-m` translates the
palm **and its fingertip targets together**, radially away from the object, and
is the correct hand-size correction. A finger-axis retreat
(`--hand-morphology-standoff`) moves the palm **without** its targets and is
measured negative; it is retained only as a documented diagnostic, default off.

Standoff sweep on the frozen crop (tip p95 left/right, worst signed distance;
`+5.00 mm` is the audit threshold-query floor, meaning "no contact"):

| `--palm-object-standoff-m` | tip p95 L/R | intended hand | forbidden robot |
| ---: | ---: | ---: | ---: |
| 0.18 (no-op) | 20 / 23 mm | -89.3 mm | -35.9 mm |
| 0.31 | 20.9 / 17.9 mm | -39.1 mm | +5.00 mm |
| 0.33 | 100.7 / 51.6 mm | +5.00 mm | +5.00 mm |
| 0.35 | 84.3 / 18.3 mm | +5.00 mm | +5.00 mm |
| 0.37 | 22.0 / 18.3 mm | +5.00 mm | +5.00 mm |

The response is **not monotonic**, which is itself the finding: mean achieved
wrist-orientation error is `19-22 deg`, so pushing the palm out changes which
arm-IK branch realizes the palm frame, and the finger stage then solves against
a differently-rotated palm. Fingertip residual is hostage to wrist orientation,
not to the standoff.

### Finger-direction palm alignment (definition B)

The notes distinguish two palm alignments, and the converter previously used
only the first. **Definition A** overlays knuckle *positions*; it fixes where
the fingers start. **Definition B** overlays finger *directions*; it fixes
which way they point, and only the second controls how much MCP abduction the
retarget must spend against the `+/-40 deg` mechanical limit.

`--align-finger-directions` (default on since 2026-08-24) composes the
definition-B yaw onto the definition-A correction. It maps each posed SOMA
proximal phalanx (`*2` MCP to `*3` PIP) into the knuckle-calibrated palm
frame, measures the per-finger abduction demand against the Wuji zero-pose
fan, and rotates the palm by the mean of the per-finger medians. Frames whose
in-plane projection is under 40% of the segment are dropped, since a finger
curled through the palm normal has no meaningful in-plane angle.

Measured on the frozen crop, both hands: yaw `+7.3 deg`, per-finger demand
`[11.5, 8.6, 5.2, 4.0]` deg left and `[13.9, 9.2, 3.7, 2.2]` deg right.

| Metric | definition A only | A + finger-direction yaw |
| --- | ---: | ---: |
| Fingertip p95, left / right | 20.93 / 17.89 mm | **14.33 / 12.19 mm** |
| Hyperextension | 0% | 0% |
| Hand-can penetration | -39.1 mm | -36.7 mm |

Cross-checked on the rejected corn-can source, which shares no scene
parameters with the snack box: left fingertip p95 improves `8.30 -> 6.64 mm`
and the right hand keeps passing all four gates (`0.16 mm` wrist,
`3.55 mm` tip p95, `20.8 deg` jump, 0% hyperextension). The yaw is therefore
a property of the SOMA-to-Wuji hand mapping, not a fit to one clip.

### What does not work, measured

- **Raising `--wrist-orientation-weight`.** At `0.3` and `1.0` the mean
  orientation error only falls `18.9 -> 13.2 -> 12.2 deg` while wrist
  *position* collapses from `0.6-2.3 mm` to `41-43 mm` and then `77-237 mm`;
  fingertip p95 does not improve. The demanded palm orientation is near the
  edge of the arm's dexterous workspace, so orientation cannot be bought.
  `--position-priority-wrist-ik` gives the best wrist position
  (`0.46-0.59 mm`) at the same orientation error and the same tip residual.
- **Retreating the palm along its finger axis** at any magnitude. Even `15 mm`
  degrades tip p95 (`14.3 -> 19.7 mm` left). See the standoff table above.
- **More solver effort.** `mean_iterations` is 97-99 of 100, but that is
  expected: the tip-only early exit is disabled whenever a posture prior is
  active, so the loop runs until the line search stops improving. The four
  abduction joints are pinned at a hard bound; more iterations cannot move
  them.

### The shared positioning axes were the binding constraint

`Lift` (a 0-0.4 m vertical slide) and `torso_flip` are shared positioning axes
for both arms. The converter locked them out of the dual-wrist solve unless
phase-aware placement or contact seeking was requested, purely to keep the
legacy fixed-base transform reproducible. Locking them costs the solve two
degrees of freedom, and the fingers pay for the palm orientation the arms then
cannot reach.

Controlled comparison on the frozen crop; the object trajectory is identical
to the bit (`0.0 mm` position, `0.0` quaternion), so the only variable is
which joints the arm IK may use:

| Metric | Lift/torso locked | authorized |
| --- | ---: | ---: |
| Fingertip p95, left / right | 14.33 / 12.19 mm | **0.36 / 0.37 mm** |
| Wrist mean, left / right | 1.07 / 0.74 mm | 0.42 / 0.43 mm |
| Mean palm orientation error | 23.79 deg | **1.15 deg** |
| Maximum joint jump | 34.99 deg (at the cap) | 13.87 deg |
| MCP abduction, mean / at cap | +35 deg / 70% | **-14 deg / 0-7%** |
| Four finger gates | fail | **pass, both hands** |

`Lift` moves to `0.26-0.40 m` and `torso_flip` to `0-0.12`: the robot raises
its column to meet the table instead of reaching for it with the arms alone.
The abduction saturation disappears, which confirms the mechanism - the
fingers had been spending their lateral range compensating a mis-rotated palm.

This **supersedes** the earlier reading of this page that an embodiment gap
was binding. That conclusion came from measuring the tip constellation at the
under-actuated solution and mistaking a constrained optimum for a geometric
floor. The `7.0 mm` RMS figure below is retained only as a record of that
measurement; it is not a floor, and the achieved `0.36 mm` disproves it.

### Adjacent-finger self-collision and the abduction envelope

With the palm placed correctly the fingers stop spending abduction on
compensation and swing negative (adduction) instead, and the left ring and
pinky then overlap: `-6.4 mm` worst penetration, `l_ring_finger_proximal_abd`
against `l_pinky_middle`, with the pinky pinned at `-39.6 deg`.

The `+/-40 deg` MCP abduction range is the Wuji **mechanical** limit; adjacent
fingers foul each other well before it, so that range is not a usable
anatomical envelope. `--mcp-abduction-limit-deg LOWER UPPER` overrides it.
Measured on the frozen crop, with everything else fixed:

| Lower bound | Fingertip p95 L/R | Worst self-collision | Self-collision |
| ---: | ---: | ---: | --- |
| -40 (mechanical) | 0.36 / 0.37 mm | -6.40 mm | fails |
| **-25** | **1.12 / 1.03 mm** | **-0.90 mm** | **qualified** |
| -15 | 2.13 / 2.11 mm | -0.84 mm | qualified; jump rises to 26.8 deg |

`-25 deg` is the adopted value: it clears self-collision while leaving the
fingertip residual four times inside its gate. It is currently a documented
per-run override rather than a library default, because it was fitted on one
clip; measuring the collision-free envelope directly from the hand geometry
(sweeping abduction until adjacent finger geoms touch) would justify making it
the default in `WUJI_ANATOMICAL_LIMITS_DEG`.

### Accepted configuration

```bash
pixi run -e isaaclab python \
  scripts/data/convert_soma_g1_parquet_to_vega_wuji.py \
  --sequence snack_box_pick --end-frame-exclusive 468 \
  --output logs/reference_replay/vega_wuji/snack_box_pick_2026-08-24/reference/snack_box_pick_refined_20hz.npz \
  --object-anchor-target 0.65 0.0 1.05 \
  --global-motion-scale 1.0 --local-geometry-scale 1.0 --motion-time-scale 4.0 \
  --palm-object-standoff-m 0.34 --palm-support-standoff-m 0.02 \
  --mcp-abduction-limit-deg -25 40 \
  --bent-elbow-inactive-rest --filter-cutoff-hz 10 --inspection-only
```

| Gate | Left | Right |
| --- | ---: | ---: |
| Mean wrist residual (<= 2.5 mm) | 0.45 mm | 0.46 mm |
| Fingertip p95 (<= 5 mm) | 1.12 mm | 1.03 mm |
| Maximum joint jump (<= 35 deg) | 13.87 deg | 10.45 deg |
| Hyperextension frames (0%) | 0% | 0% |

All four gates pass on both hands, robot self-collision is qualified, and
forbidden robot-object clearance is qualified. The one remaining
non-qualified geometry group is `intended_hand` at `-7.10 mm`: that is the
hands pressing into the box, which is the point of a grasp. Gate it with
`--contact-seeking` rather than by pushing the hands away, and note that
`--palm-object-standoff-m 0.37` moves the hands clear of the box with the same
fingertip accuracy if a contact-free inspection Reference is wanted.

The standoff now behaves monotonically and does not disturb fingertip
accuracy (`0.36/0.37 mm` at every value from `0.31` to `0.37`), because it
translates the palm and its targets together and the palm orientation is
tracked to about a degree. Before the shared axes were authorized the same
sweep was erratic (`21 -> 101 -> 84 -> 22 mm`), which was the arm-IK branch
flipping under a 19-24 deg orientation error.

### Superseded: the constellation measurement

Comparing the achieved robot fingertip constellation with the human's over the
crop, by inter-tip distance:

| Pair group | Mean difference |
| --- | ---: |
| Thumb to each finger | `+9.0` to `+12.9 mm` |
| Finger to finger | `-3.5` to `+3.7 mm` |
| RMS over all ten pairs | `7.0 mm` |

The four-finger constellation is reproducible; the thumb sits about a
centimetre too far from the fingers. No rigid palm placement can remove an
intra-hand shape difference, so `7.0 mm` RMS is a floor for exact fingertip
imitation on this pairing, against a `5 mm` p95 gate. This is consistent with
the per-tip residuals, where the thumb lands within `0.2 mm` while the four
fingers absorb `10-17 mm`.

Conclusion: exact fingertip imitation is geometrically unavailable for this
pairing, and the wrist stage is not the cause. The next step is not solver
tuning. Finger targets must come from **object-surface contact points** (the
CHORD `--soma-chord-contacts` / `--contact-target-ik` path) instead of human
fingertip positions, so a larger hand can contact the same faces at its own
scale, with the thumb allowed to oppose rather than to match a human thumb
position.

### Current snack-box artifacts

```text
logs/reference_replay/vega_wuji/snack_box_pick_2026-08-24/
  reference/snack_box_pick_refined_20hz.npz      accepted; all four gates pass
  rendered/snack_box_pick_refined_3q.mp4
  reference/snack_box_pick_contact_free_20hz.npz same recipe at standoff 0.37,
                                                 hands clear of the box
  reference/snack_radial031.npz                  historical: Lift/torso locked
  rendered/snack_radial031_3q.mp4                the locked-axes comparison
```

Accepted Reference SHA-256:
`45b4be9cbe34f118f65085b4b75dcc79426ecc398d119b44d2c25d9e85b8570e`

Reproduce the contact-preserving candidate from the repository root:

```bash
pixi run -e isaaclab python \
  scripts/data/convert_soma_g1_parquet_to_vega_wuji.py \
  --sequence snack_box_pick --end-frame-exclusive 468 \
  --output logs/reference_replay/vega_wuji/snack_box_pick_2026-08-24/reference/snack_box_pick_refined_20hz.npz \
  --object-anchor-target 0.65 0.0 1.05 \
  --global-motion-scale 1.0 --local-geometry-scale 1.0 --motion-time-scale 4.0 \
  --palm-object-standoff-m 0.31 --palm-support-standoff-m 0.02 \
  --bent-elbow-inactive-rest --filter-cutoff-hz 10 --inspection-only

pixi run -e isaaclab python \
  scripts/audit/summarize_vega_wuji_reference.py \
  --reference logs/reference_replay/vega_wuji/snack_box_pick_2026-08-24/reference/snack_box_pick_refined_20hz.npz
```

The default `--palm-object-standoff-m 0.28` and `--palm-support-standoff-m
0.20` are corn-can values and are **wrong for this clip**: the source palms sit
`0.21-0.27 m` from the box centre and dip to table level, so the defaults would
shove both hands off the box and lift them 20 cm above the table.

## The road to RL training

Three gates stand between a retargeted Reference and a training run. They are
different in kind, and only the third is plumbing.

### Gate 1 — measured contact geometry (data)

The environment consumes contacts, it does not merely check them
(`vega_wuji_imitation_env.py:398-425` reads `object_positions_w`,
`link_normals_w`, `active`). The pinned Parquet carries only the binary
`hand_contact_active` flag, so the converter refuses to invent a
`ContactSequence` (`convert_..._vega_wuji.py:3495`) and
`--contact-seeking --soma-chord-contacts` is mandatory without
`--inspection-only`.

**Status: passing.** On the `[400, 480)` crop at `--motion-time-scale 8.0`,
per-side recovery is `1.00` left and `0.867` right against a `0.50` gate, and
the whole geometry audit qualifies: robot self-collision `-1.00 mm`, intended
hand-object contact `-0.51 mm`, forbidden robot-object clearance clear.

Two audit defects were fixed to get there, both of which had been rejecting
physically fine motion:

- The object-vs-support audit measured every mesh vertex against an
  **infinite** plane. The snack box (`0.43 x 0.48 m`) overhangs its support
  cylinder (`0.45 m` diameter), so a tilting corner hanging past the table
  edge read as penetration. It is now restricted to vertices actually over
  the support disc, and still reports the infinite-plane minimum alongside.
  Effect on this clip: `-0.534 mm` -> `-0.203 mm`.
- That audit then allowed `1 um` of penetration, while every other
  penetration gate in the converter allows `1 mm` (robot self-collision,
  contact recovery, and the declared `CollisionClearanceQualification`). A
  micron is a numerical epsilon, not a physical criterion, on a mesh
  reconstructed from video and then re-anchored. It now allows `1 mm` like
  its siblings.

### Gate 2 — runtime-promotion attestation (was a hard stop; now implemented)

Offline MuJoCo fitting is not evidence that Newton can execute a Reference.
The converter therefore hardcodes `training_qualification = None` and
`runtime_qualified = False`, and the environment used to reject **every**
Reference from a function whose entire body was a `raise`. No writer existed.

The two evidence producers already existed in
`scripts/viz/replay_vega_wuji_reference.py`: a full-horizon **state-lock**
replay (teleport onto every Reference frame, measure simulator drift) and a
full-horizon **zero-assistance** replay (`--unassisted-dynamics`: zero policy
residual, virtual object controller forced to exactly zero). What was missing
was the writer and verifier that bind those records to a motion and a model.

Now implemented:

- `source/imitation_experiments/imitation_experiments/audit/vega_wuji_promotion.py`
  — the typed attestation, its builder, and its verifier.
- `scripts/data/promote_vega_wuji_reference.py` — consumes both replay
  records, refuses unless each one completed the full horizon and passed,
  attaches a typed `TrainingQualification`, and writes the sidecar.
- `vega_wuji_imitation_env.py` now verifies that sidecar instead of raising.

The binding is on Reference **content**, not file bytes: promotion re-saves the
NPZ to attach the qualification, which changes the file hash but not the
motion, so the canonical float32 qpos digest is the identity that ties a replay
record to the Reference the environment loads. The verifier re-checks the
evidence itself, so hand-editing `"passed": true` into the sidecar does not
promote a Reference whose replay actually failed.

Promotion runs **before** manifest creation, because the manifest writer
requires an already-qualified NPZ:

```text
convert (--contact-seeking --soma-chord-contacts)
  -> replay --reference <npz>                       (state lock)
  -> replay --reference <npz> --unassisted-dynamics
  -> promote_vega_wuji_reference.py
  -> write_vega_wuji_reference_manifest.py
  -> train_newton.py
```

### Gate 3 — manifest and launch (plumbing)

Training rejects a raw NPZ; it needs the hash-bound JSON Manifest, and it must
be launched through `scripts/rlopt/train_newton.py` with
`physics=newton_mjwarp` (the plain `train.py` cannot run the MJCF-to-USD
conversion; recorded runs under `outputs/vega_wuji_manifest_newton_smoke*`
show `train.py` producing no checkpoint and `train_newton.py` producing one).

### Fingertips on the box, object at true size

Decision of 2026-08-24: keep the object at its true size and aim the
fingertips at the **contact points on the box**, not at the human's fingertip
positions. On a true-size object those are different points, because the Wuji
hand is larger than the source hand, and only the contact location is a
physical requirement of the manipulation.

Implemented as `--object-anchored-fingertips` (default on): during contact a
fingertip target becomes the measured CHORD contact point carried on the
object; away from contact it stays the wrist-anchored human fingertip. The two
are cross-faded so a target path is continuous through a contact edge.

One trap is pinned by a regression test. CHORD parks an **inactive** contact
slot at the world origin rather than at a meaningful point, so the raw target
array must never be averaged over time. The cross-fade smooths the
*displacement* from the wrist-anchored target, which is exactly zero without
contact. Smoothing the mask instead dragged fingertips toward the origin and
produced a `518 mm` fingertip residual on the real clip.

A second pass (`--reconcile-fingertips-after-contact`, default on) re-solves
the fingertips onto those targets after the contact and collision repair
stages, because those stages shape the whole hand and bulk-shift the tips:
`0.1-4.2 mm` at the IK stage became `9-25 mm`, uniform across all five
fingers. It runs before every geometry audit, so its result is re-verified.

Result on the `[400, 480)` crop at `--motion-time-scale 8.0`, 64 frames,
3.15 s:

| Gate | Left | Right | Limit |
| --- | ---: | ---: | ---: |
| Mean wrist residual | 0.40 mm | 0.41 mm | 2.5 mm |
| Fingertip p95 | **1.24 mm** | **2.03 mm** | 5 mm |
| Maximum joint jump | 10.39 deg | 5.79 deg | 35 deg |
| Hyperextension | 0% | 0% | 0% |

**All four finger gates pass on both hands, with measured contacts enabled.**

### The cost, measured

Putting the fingertips on the box is not free on a true-size object. The same
run records what the pose looked like immediately before the reconciliation
(`strict_final_finger_tuck`) and after it (`semantic_can_20hz`):

| | hand into object | robot self-collision | fingertips off contact |
| --- | ---: | ---: | ---: |
| Before reconciliation | -0.51 mm (qualified) | -1.00 mm (qualified) | 9-25 mm |
| After reconciliation | **-10.25 mm** | **-5.21 mm** | 1.2-2.0 mm |

That is the embodiment gap stated geometrically: a larger hand can put its
fingertips on the human's contact points, or keep the rest of itself out of a
true-size box, but not both. Ordering does not escape it either. Running the
reconciliation *before* the penetration tuck instead leaves the hand deep
enough in the object that the tuck's multistart search reports **no feasible
left-finger sample at all**. Both orderings were measured.

Standing the fingertip target off the surface by its own `9 mm` collision
radius along the contact normal does not help and is off by default
(`--fingertip-contact-standoff-m 0`): it rotates the finger and drives the
middle phalanges in, so whole-hand penetration worsened from `-10.25` to
`-15.73 mm`.

### The penetration barrier

Implemented (option 2 below): `--reconciliation-clearance-m` turns the
clearance into a task inside the reconciliation solve
(`FingertipAvoidanceConfig`) rather than a repair afterwards, so the solver
trades fingertip accuracy for staying outside the object instead of a later
stage undoing its work. It carries both the hand-versus-object pairs and the
same-hand self-contact pairs, which only exist after the tips are pulled back
and so are discovered from a first solve and resolved in a second.

Three bugs were found and fixed while getting it to work, each measured:

- **Escape sign.** MuJoCo's witness pair means opposite things either side of
  contact: apart, it points across the gap, so escaping means moving against
  it; overlapping, it points from the deepest robot point out through the
  surface, so escaping means moving along it. One sign for both drove
  penetrating geoms deeper, `-10.25 -> -20.36 mm`.
- **Unnamed geoms.** The audited collision geoms carry no names, so a
  name-keyed pair list silently matched nothing and the self-contact task was
  inert. Pairs are now keyed by geom id.
- **Relative Jacobian.** For a robot-on-robot pair both witnesses move with
  the joints being solved, so separation responds to relative motion. A
  one-sided Jacobian pushed in directions that closed the gap:
  `-7.48 -> -10.67 mm`. With the relative form, `-2.60 mm`.

The barrier is first-order, so it stalls where the escape direction lies in
the contacting point's Jacobian null space: a straight finger pressed face-on
into a slab cannot bend itself off that slab. It is a cost, not a constraint,
and the fail-closed geometry audit still gates the result.

Effect on the accepted clip, against the pre-barrier pose:

| Metric | Before barrier | With barrier |
| --- | ---: | ---: |
| Hand into object | -10.25 mm | **-0.52 mm (qualified)** |
| Robot self-collision | -5.21 mm | -2.60 mm |
| Fingertip p95, left | 1.24 mm (pass) | 3.95 mm (pass) |
| Fingertip p95, right | 2.03 mm (pass) | **7.76 mm (fail)** |

It did what it was for: hand-object penetration is qualified, and
self-collision more than halved. It cost more fingertip accuracy than the
`3-4 mm` of headroom allowed on the right hand, which now misses its gate by
`2.76 mm`, and self-collision is still `1.6 mm` outside its `1 mm` tolerance.
The weight (`25.0`) is the knob that sets this trade and is not yet exposed on
the command line.

Three ways forward, in increasing order of work:

1. Decide that `~10 mm` of *intended* hand-object overlap is acceptable for a
   Reference whose object is virtually controlled during warmup, and gate
   intended contact separately from forbidden contact. This is a threshold
   decision, not an implementation.
2. Give the reconciliation solver a penetration barrier, so it trades a
   millimetre or two of fingertip accuracy (there is `3-4 mm` of headroom
   under the gate) for staying outside the object.
3. Scale the object with the hand, which makes the two goals identical again
   at the cost of a larger box.

### The one remaining blocker

With contacts enabled the **final finger acceptance** fails:

| Gate | Left | Right | Limit |
| --- | ---: | ---: | ---: |
| Mean wrist residual | 0.40 mm | 0.41 mm | 2.5 mm |
| Fingertip p95 | **59.76 mm** | **10.20 mm** | 5 mm |
| Maximum joint jump | **59.79 deg** | **45.04 deg** | 35 deg |
| Hyperextension | 0% | 0% | 0% |

The cause is measured, not mysterious. Two contact-repair stages apply large
per-frame corrections with no temporal coupling and no bound relative to the
previous frame:

- `contact_constraint_projection` — attraction onto contact witnesses, max
  joint shift `0.797 rad` (`45.7 deg`), correcting 42 of 64 side-frames;
- `contact_scene_projection` `finger_can_closure` — pushing fingers back out
  of the object, max joint shift `1.077 rad` (`61.7 deg`) over 29 frames.

They also fight each other within a frame: attraction pulls fingertips to
`0.5 mm` off the surface while closure allows `0.5 mm` of penetration, and the
Wuji hand is larger than the source hand, so closure has to open the fingers
hard. That is the "finger popping" this page has listed as the next gate.

Fixing it is a design task, not a parameter change, and it has two parts that
should be decided explicitly:

1. **Temporal regularization.** The tips-only solver already carries a
   per-frame change bound (`max_frame_change_rad`, derived so that ten source
   steps cannot exceed the emitted-frame limit). The contact projectors carry
   `max_step` per *iteration* but nothing bounding total per-frame change, and
   they expose no such parameter. Adding one is an ILTools change.
2. **What the acceptance gate should measure.** The gate compares the final
   pose against the **human fingertip targets**, while the contact stages
   deliberately optimize toward **object-surface witnesses**. When the two
   disagree the gate must fail, by construction. Deciding whether a
   contact-fitted Reference should be scored against contact-consistent
   targets instead is a change to the training contract that
   `_verify_adopted_wuji_training_gates` enforces, so it should not be made
   silently.

Do not paper over this by appending a smoothing pass after the audits: every
geometry gate above is measured before that point, and a post-audit edit would
emit an unaudited trajectory.

## Why the previous result was rejected

The superseded Reference is:

```text
logs/reference_replay/vega_wuji/soma_chord_qualified_2026-08-20/
  reference/corn_can_handover_soma_chord_20hz.npz
  rendered_motion_wide/vega_wuji_corn_can_handover_retargeted.mp4
```

It should be treated as a diagnostic artifact only. Joint arm/finger keypoint
projection and later collision/contact projectors were all allowed to change
the arms. Those stages could preserve numerical wrist/contact objectives while
selecting visually poor elbow branches. The resulting overhead/chicken-wing
posture invalidates its earlier promotion and training recommendation.

The source SOMA sequence itself is natural: its hands remain around waist/chest
height. The distortion was introduced during robot retargeting, not by the
source motion.

## Current architecture: strict wrist/hand split

The enforced dependency order is:

```text
human wrist + object motion
            |
            v
object-relative workspace projection
            |
            v
weighted SE(3) dual-wrist IK -- sequential warm start --> arm trajectory
            |                                         |
            |                                         +-- lock exactly
            v
SOMA-to-Wuji finger mapping
            |
            v
optional finger-only keypoint/contact/collision refinement
```

The concrete split invariants are:

- anatomical SOMA wrists are mapped with the geometry-derived palm correction;
- wrist position/orientation use Pink-compatible cost amplitudes `1.0/0.03`
  (MuJoCo objective coefficients `1.0/0.0009`), with a sequential arm-IK warm
  start;
- phase-aware object placement and a bent-elbow inactive rest are enabled in
  the pinned diagnostic above; position-priority wrist IK and arm posture
  costs remain off;
- the fourteen arm joints are captured immediately after wrist IK; shared Lift
  and torso axes enter that solve only for explicit phase-aware/contact modes;
- five-tip IK and every later keypoint, CHORD, support, can, and self-collision
  stage receive finger joints only;
- physics settling and arm radial-escape projection are disabled in strict
  split mode;
- a final assertion fails if any captured arm joint changes by more than
  `1e-12`.

This matches the intended simpler idea: treat each robot hand as a floating
target and let IK explain that target with the arms. Contact fitting is not
allowed to reinterpret the arm posture afterward.

## Source mapping

### SOMA reconstruction

The solved SOMA layer mapping remains valid. The layer's extra entry is a
synthetic `Root` at index zero; the payload's 77 names match after removing that
entry. On the pinned 318-frame crop, reconstruction error is approximately
0.96 mm mean, 1.55 mm p95, and 1.76 mm maximum.

### Wrist conventions

Two wrist signals exist and must not be conflated. `ee_pose_w` belongs to the
already-retargeted source robot. The bridged SOMA anatomical hand root is the
root of the human fingertip geometry. They differ by roughly 5--6 cm and have
different frame conventions. The adopted method always uses the anatomical
root. It composes the source orientation with the knuckle-derived
source-to-Wuji correction and applies the measured palm-normal origin offset.
The older `ee_pose_w` artifacts remain historical visual baselines only.

### Finger mapping

The SOMA hinge-angle mapping supplies only the initial finger seed; wrist IK
generates the arm trajectory. For non-thumb fingers, SOMA `*2`, `*3`, `*4`,
and `*End` are MCP, PIP, DIP, and tip. The metacarpal `*1` stays in the palm.
After wrist IK, the adopted solver changes only the 20 Wuji finger joints per
hand to fit five geometry-calibrated SOMA fingertip targets. The arm trajectory
is then immutable. A uniform closure or later contact projection may change
fingers only, and the four acceptance gates are measured again from the final
trajectory.

## Historical floating-wrist baseline

The historical baseline artifact is:

```text
logs/reference_replay/vega_wuji/wrist_hand_split_2026-08-20/
  reference/corn_can_handover_floating_wrist_hand_ik_baseline_22cm_20hz.npz
  rendered/corn_can_handover_floating_wrist_hand_ik_baseline_22cm.mp4
  rendered/corn_can_handover_floating_wrist_hand_ik_baseline_22cm_close.mp4
```

Reference SHA-256:
`bf1bd111f0a71879f5f66848a8aca73aecadc05649f9f125b9b2f3448f621bce`

| Metric | Result |
| --- | ---: |
| Frames / rate / duration | 127 / 20 Hz / 6.3 s |
| Wrist source | source `ee_pose_w` |
| Object-center wrist standoff | 0.22 m |
| Wrist p95 error, left / right | 23.85 / 19.85 mm |
| Wrist maximum error, left / right | 31.42 / 38.44 mm |
| Post-IK locked-joint drift | 0 |
| Authored can/support geometry | qualified at 1 mm |
| Dense 200 Hz diagnostic path | qualified at 1 mm |
| Robot self-collision | qualified |
| Runtime/training qualified | no |

The 22 cm distance is a center-to-wrist morphology standoff, not a human palm
gap. Wuji wrist-to-tip lengths are roughly 153--209 mm, and the G1 wrist frame
orientation put rigid/finger geometry through the can at 18 cm. At 22 cm the
emitted motion is clear. A 1 mm inspection clearance is used because the 5 mm
gate rejected an otherwise separated right fingertip by about 3.9 mm; the
baseline is meant to approach the object, not prove a contact-free production
trajectory.

This artifact was the first visual checkpoint for the split architecture, not
an accepted final pose solution. It predates the anatomical wrist/five-tip
gates. It is not a contact Reference: contact arrays are inactive and no
measured robot contact recovery was attempted.

### Scene-placement diagnosis after visual rejection

The baseline can reaches approximately 0.99 m forward and -0.39 m laterally,
against neutral palms around `(0.79, +/-0.23)` m. That initially suggested the
human scene had simply been placed too far away. Controlled diagnostics refuted
that as the sole cause:

- manually moving the object/support center to x=0.65 m made the wrist targets
  unreachable; per-side p95 position error rose as high as 243 mm;
- reachability-first XY anchors around x=0.88 m reduced left elbow saturation
  from 61 to 26 frames but did not remove the strange hand orientation;
- raising the can and support by 0.14 m retained 21--23 mm wrist p95 error and
  reduced torso/elbow saturation to 25/11 frames;
- increasing wrist-orientation weight from `1e-6` to `1e-3` reduced saturation
  further, but hand flips remained and wrist p95 error rose to 34/39 mm.

Those tests were confounded by the model's all-zero, fully straight arm seed
and by parking the inactive hand over the table. After replacing both, the
user's distance hypothesis became useful: explicit handover anchors at x=0.65 m
near the front edge reduced the remaining right-forearm/table penetration from
41 frames to five shallow frames at 14 cm active wrist clearance. Raising only
the active wrist clearance to 18 cm removed those final violations. The table
and inactive rest pose remain at their original heights.

## Historical best visual candidate

The best historical visual/geometry inspection artifact is:

```text
logs/reference_replay/vega_wuji/wrist_hand_split_2026-08-20/
  reference/corn_can_handover_bent_rest_near_edge_active18cm_20hz.npz
  rendered/corn_can_handover_bent_rest_near_edge_active18cm_3q.mp4
  rendered/corn_can_handover_bent_rest_near_edge_active18cm_front.mp4
  rendered/corn_can_handover_bent_rest_near_edge_active18cm_two_view.mp4
```

Reference SHA-256:
`3c569693c271b1e051a421d0c4ebf7fb7f35458828d85005bf783b166016aa31`

| Metric | Result |
| --- | ---: |
| Frames / rate / duration | 127 / 20 Hz / 6.3 s |
| Inactive palm rest | about `(0.30, +/-0.35, 0.75)` m |
| Contact-phase object anchors | `(0.65, +0.20, 1.14)` / `(0.65, -0.20, 1.14)` m |
| Active wrist/support clearance | 0.18 m |
| Wrist p95 error, left / right | 19.04 / 17.08 mm |
| Wrist maximum error, left / right | 30.06 / 30.26 mm |
| Mean wrist orientation error | 8.91 deg |
| Left / right finger closure scale | 0.9696 / 1.0 |
| Post-IK locked-joint drift | 0 |
| Authored can/support/self geometry | qualified at 5 mm |
| Runtime/training qualified | no; superseded by the anatomical recipe |

The important correction was not a more elaborate finger solver. The old
inactive target was the model's all-zero straight-arm palm, raised above the
support. That forced both arms to hover over the table. The new rest target was
constructed from a symmetric, limit-margined bent-elbow posture and is also the
initial IK seed. During source-active phases, only the corresponding wrist is
projected above the near-edge object trajectory.

Both source-view MP4s were decoded end-to-end with FFmpeg 7.0.2 and inspected
as 16-frame FFmpeg contact sheets. They are H.264, 960x540, 20 fps, and 6.35 s.
The synchronized two-view delivery render is H.264, 1920x540, 20 fps, 6.35 s,
with SHA-256
`ab6fda60c309e32cce19aad9bb798315fb6c9032d0dfe843fedb8ec0c040d7e5`.

## Strict SOMA/CHORD diagnostic

The nearest contact-aware diagnostic is:

```text
logs/reference_replay/vega_wuji/wrist_hand_split_2026-08-20/
  reference/corn_can_handover_wrist_hand_split_standoff_18cm_20hz.npz
  rendered/corn_can_handover_wrist_hand_split_standoff_18cm.mp4
```

Its legacy geometry audit showed that the split itself worked:

- post-wrist arm drift is exactly zero;
- wrist p95 error is 18.68 / 18.99 mm;
- support, forbidden wrist/arm-can geometry, and robot self-collision pass;
- its only stored geometry blocker was eleven left-finger penetration frames.

The converter now has a deterministic finger-only multistart fallback. On the
stored trajectory it finds collision-safe poses for all eleven frames in about
three seconds while preserving zero arm drift. However, the largest required
finger correction is 3.26 rad in joint-vector norm and per-frame correction can
create visible finger popping. That fallback is fail-closed geometry repair,
not the trajectory-level contact treatment needed for promotion. The new
tips-only solver already uses sequential warm starts and a `34.99 deg`
temporal bound, but this historical artifact has not been regenerated or
accepted under the new four-gate schema.

## Implementation

The strict split is implemented in:

- `scripts/data/convert_soma_g1_parquet_to_vega_wuji.py`
  - `--split-wrist-hand-ik`;
  - `--wrist-orientation-weight`;
  - `--arm-previous-posture-weight`;
  - `--arm-neutral-posture-weight`;
  - `--position-priority-wrist-ik` and
    `--position-priority-slack-m`;
  - `--bent-elbow-inactive-rest`;
  - locked-joint verification;
  - finger-only post-IK projectors;
  - deterministic locked-wrist finger feasibility repair.
- `ImitationLearningTools/iltools/retarget/wuji_finger.py`
  - anatomical limits, slight-curl posture, geometry-only palm fit, bounded
    tips-only trajectory IK, final-FK measurements, and raw gates.
- `scripts/audit/audit_wuji_finger_joints.py`
  - fail-closed stored-Reference audit; required mode rejects missing wrist or
    fingertip measurements.
- `ImitationLearningTools/iltools/retarget/soma_chord.py`
  - identity-specific SOMA zero-pose reconstruction;
  - corrected non-thumb CHORD link anatomy, with the metacarpal retained in
    the palm and `*2..*End` mapped to proximal, middle, and distal links.
- `source/isaaclab_imitation/isaaclab_imitation/tasks/manager_based/dexmanip/actions.py`
  - runtime intersection of soft limits with the Wuji anatomical envelope.
- `scripts/viz/replay_vega_wuji_reference.py`
  - manifest/motion/model hash binding and exact zero-assistance checks for
    full-horizon Newton replay.
- `ImitationLearningTools/iltools/retarget/dual_hand.py`
  - sequential wrist IK;
  - optional strict position/orientation/posture task hierarchy;
  - nonlinear primary-task line search;
  - previous-frame and neutral posture costs.
- `scripts/viz/render_vega_wuji_reference_mujoco.py`
  - headless kinematic rendering of qualified or inspection-only References;
  - configurable wide and hand/object close-up cameras.

## Historical candidate provenance

The following command records how the historical visual candidate was made.
It intentionally uses the superseded `0.001` wrist-orientation weight. The
current converter always uses the anatomical SOMA wrist and tips-only hand
stage, so rerunning it is not expected to reproduce the stored artifact hash
or qualify the result:

```bash
pixi run -e isaaclab python \
  scripts/data/convert_soma_g1_parquet_to_vega_wuji.py \
  --output /tmp/corn_can_handover_historical_flags.npz \
  --object-anchor-target 0.80 0.15 1.14 \
  --phase-aware-object-placement \
  --phase-anchor-max-shift-m 0.4 \
  --phase-object-anchor-targets 0.65 0.20 1.14 0.65 -0.20 1.14 \
  --global-motion-scale 0.4 --local-geometry-scale 1.0 \
  --motion-time-scale 4.0 \
  --palm-object-standoff-m 0.22 --palm-support-standoff-m 0.18 \
  --max-wrist-position-error 0.08 \
  --wrist-orientation-weight 0.001 \
  --arm-previous-posture-weight 0.1 \
  --arm-neutral-posture-weight 0.0001 \
  --position-priority-wrist-ik --position-priority-slack-m 0.002 \
  --bent-elbow-inactive-rest --split-wrist-hand-ik --inspection-only

pixi run -e isaaclab python \
  scripts/viz/render_vega_wuji_reference_mujoco.py \
  --reference /tmp/corn_can_handover_historical_flags.npz \
  --output /tmp/corn_can_handover_historical_flags.mp4 \
  --width 960 --height 540 --azimuth 135 --distance 2.25 \
  --elevation -8 --lookat 0.5 0.0 0.88
```

This remains useful only as an inspection-only, no-contact visual baseline.

## Reproduce the rejected baseline

From the repository root:

```bash
pixi run -e isaaclab python \
  scripts/data/convert_soma_g1_parquet_to_vega_wuji.py \
  --output /tmp/corn_can_handover_historical_baseline_flags.npz \
  --phase-aware-object-placement \
  --phase-object-anchor-targets 0.80 0.20 1.14 0.95 -0.20 1.14 \
  --global-motion-scale 0.4 --local-geometry-scale 1.0 \
  --motion-time-scale 4.0 \
  --palm-object-standoff-m 0.22 --palm-support-standoff-m 0.22 \
  --geometry-clearance-m 0.001 \
  --split-wrist-hand-ik --arm-neutral-posture-weight 0.0001 \
  --dense-geometry-audit-fps 200 --inspection-only
```

No `--contact-seeking` or `--soma-chord-contacts` flag is present: this is the
intentional no-contact baseline.

```bash
# Wide view.
pixi run -e isaaclab python \
  scripts/viz/render_vega_wuji_reference_mujoco.py \
  --reference /tmp/corn_can_handover_historical_baseline_flags.npz \
  --output /tmp/corn_can_handover_historical_baseline_flags.mp4 \
  --azimuth 135 --distance 3.0 --elevation -12

# Hand/object close-up.
pixi run -e isaaclab python \
  scripts/viz/render_vega_wuji_reference_mujoco.py \
  --reference /tmp/corn_can_handover_historical_baseline_flags.npz \
  --output /tmp/corn_can_handover_historical_baseline_flags_close.mp4 \
  --azimuth 135 --distance 1.55 --elevation -18 --lookat 0.72 0.0 1.25
```

## Qualification boundary

Geometry-clean does not mean runtime-qualified. A training Reference still
needs all of the following:

- visually accepted arm and hand motion;
- finite, measured robot contact witnesses with unit normals;
- useful contact recovery on at least 50% of each required hand's source-active
  frames;
- typed intended-hand versus forbidden-arm/wrist can checks;
- robot-support, can-support, and self-collision checks;
- all four final-FK Wuji gates with no missing measurement;
- a hash-bound, full-horizon Newton state-lock record;
- a hash-bound, full-horizon unassisted record with virtual-object-controller
  scale exactly zero at reset and every step.

Only after those gates pass should residual-policy training begin. The policy
then has to replace virtual object assistance under unassisted evaluation; a
zero-residual replay is only a diagnostic baseline.

## Next gate

Keep the passing wrist trajectory fixed. Resolve the left fingertip branch
without weakening scale `1.0`, the anatomical limits, the slight-curl prior, or
the temporal gate. Then remove left-hand self-collision and support penetration
with trajectory-level, finger-only SOMA/CHORD contact fitting. The contact
stage must preserve exact arm lock, avoid finger popping, recover finite robot
contact witnesses on at least 50% of each active hand's frames, and retain all
final-FK and geometry gates. Only then run the hash-bound Newton state-lock and
zero-assistance unassisted protocols before a future training-promotion tool is
implemented.

## Traps

- `fixed_root_pose_w` translates the saved Reference by 0.19 m. Convert world
  object poses back to the robot-base frame before standalone MuJoCo audits.
- `soma_joints` is root-normalized in the human frame. Do not compare it
  directly with object poses.
- `ee_pose_w` and the anatomical SOMA wrist are not interchangeable.
- CHORD crop resampling must include the crop's source start time; otherwise
  contact labels silently shift.
- A can-clearance gate that treats all robot-can pairs alike forbids the task.
  Intended hand contact and forbidden arm/wrist contact must remain typed.
- Never allow a late contact or collision projector to move an accepted arm
  trajectory.
- Do not smooth a contact-qualified trajectory without re-running every
  geometry and contact gate.

## Related pages

- [Project Live Status](current-status.md)
- [Context Management](context-management.md)
