# Running NVIDIA's released SONIC tracker inside our G1 environment

Goal: load the released SONIC G1 controller (`nvidia/GEAR-SONIC`,
`sonic_release/last.pt`) into `Isaac-Imitation-G1-v2`'s physics and evaluate it
on our BONES-SEED references under our own protocol, next to our FSQ64 tracker.

This is deliberately **not** a "SONIC baseline" claim. It is a native
reproduction attempt, and every deviation we cannot reproduce is recorded here.

## Artifacts

Downloaded 2026-08-07 to `/mnt/hsstorage/fwu91/sonic_release/` (public,
ungated): `last.pt` (469 MB), `config.yaml`, `observation_config.yaml`,
`model_encoder.onnx` (50 MB), `model_decoder.onnx` (41 MB). The ONNX pair is
used only as ground truth for verification, never in the loop.

Adapter: `imitation_experiments/lowlevel/sonic_release_actor.py` (+ tests). It
reads the tensors directly; no `gear_sonic` code is imported or vendored.

## What the released actor is

All SiLU, no observation normalizer, 25.87M parameters.

| module | shape |
| --- | --- |
| `encoders.g1` | 640 -> [2048, 1024, 512, 512] -> 64 |
| FSQ | 2 tokens x 32 coordinates, 32 levels, output `round(bound(z))/16` |
| `decoders.g1_dyn` | 994 -> [2048, 2048, 1024, 1024, 512, 512] -> 29 |

`994 = 64 token + 930 proprioception`.

## Verified equivalences

Both checked against the released ONNX export, which is an independent
implementation of the same weights:

| check | result |
| --- | --- |
| decoder on random 994 input | max abs diff **2.9e-06** (float32 noise) |
| encoder at zero input | **bitwise identical** |
| encoder on random input, correct layout | **exact, 0.0** |

The zero-input match is what pins the FSQ convention: the `eps = 1e-3` bound and
the `/16` normalization are confirmed, not assumed.

The v1.1 export matches the same torch adapter pattern after selecting the g1
path with scalar `encoder_mode_4 = 0`: encoder max abs diff **0.0**, decoder max
abs diff **4.77e-07**.

## The encoder input layout, and why it is a trap

The 640 is **not** ten frames of `[qpos, qvel, ori]`. SONIC's flat
`command_multi_future` is term-major `[qpos(290) | qvel(290)]`, reshaped to
`(10, 58)` before the 6-wide anchor orientation is appended per row. Each
64-wide block therefore holds **two consecutive frames of one term**:

```
block 0..4 : [qpos[2b], qpos[2b+1], anchor_ori[b]]
block 5..9 : [qvel[2b], qvel[2b+1], anchor_ori[5 + b]]
```

Recovered by probing all 1762 ONNX encoder input slots one at a time and
matching each to a torch input column: 640/640 unique, zero ambiguous. The tidy
per-frame layout produces plausible but wrong tokens, which would have silently
poisoned every downstream number. The same probe showed ONNX mode index 0 is
**teleop** (267 inputs, exactly `encoders.teleop`'s width); g1 is mode 1.

## Three cadences, easily conflated

| path | naming | spacing | span |
| --- | --- | --- | --- |
| robot / motion reference (the g1 encoder) | `..._10frame_step5` | `dt_future_ref_frames 0.1` | 0.9 s lookahead |
| human / SMPL | `..._10frame_step1` | `smpl_dt_future_ref_frames 0.02` | 0.18 s lookahead |
| proprioception history | `his_*_10frame_step1` | one control step | 0.18 s of past |

Our reference window must use the **robot** cadence: offsets
{0, 5, ..., 45} frames at 50 fps.

## The 930 proprioception layout

IsaacLab concatenates a group by the **declaration order of the config class's
fields**, not by the order keys appear in a YAML override. SONIC's `PolicyCfg`
declares `base_ang_vel` (107), `joint_pos` (108), `joint_vel` (109), `actions`
(110), and `gravity_dir` far later (128). So:

```
930 = ang_vel(10x3) || joint_pos_rel(10x29) || joint_vel_rel(10x29)
      || last_action(10x29) || gravity_dir(10x3)
```

Each term is flattened oldest-frame-first, newest-last
(`CircularBuffer.buffer`). This resolves a real conflict: the training YAML
lists `gravity_dir` first, the deploy YAML lists it last, and the deploy YAML is
the one that matches the class field order. (The deploy header's "436" total is
stale - it describes a 4-frame-history variant, not this checkpoint.)

## The 64 values per reference frame

`command_multi_future = cat([joint_pos_multi_future, joint_vel_multi_future])`
(term-major, matching what the ONNX probe recovered), where both are raw
reference values straight from the motion library - **not** relative to the
default pose. Per frame: 29 reference joint positions, 29 reference joint
velocities, and the 6D anchor orientation.

`future_time_steps_init = arange(10) * frame_skips` = **[0, 5, ..., 45]**, so
the window starts at the *current* reference frame and reaches 45 frames
(0.9 s) ahead.

The anchor-orientation encoding is identical to ours, verified line by line:

| | SONIC | ours |
| --- | --- | --- |
| difference | `quat_inv(robot_anchor_quat_w) * ref_root_quat` | `subtract_frame_transforms(robot_anchor, ref_anchor)` = same direction |
| 6D | `matrix_from_quat(q)[..., :2].reshape(N, -1)` | `quat_to_rot6d_flat`: `matrix_from_quat(q)[..., :2].reshape(N, -1)` |

So the encoder view is
`components=("joint_qpos_qvel", "root_ori"), past_steps=0, future_steps=10,
frame_stride=5`, fed through `pack_encoder_window`.

## What already matches our environment

Tier 2 turned out to be observation plumbing only - no physics reconstruction:

| aspect | status |
| --- | --- |
| G1 29-DoF joint order | SONIC's `G1_ISAACLAB_JOINTS` equals our `G1_29DOF_ISAACLAB_JOINT_NAMES` — but see below, the *live* order depends on the source |
| actuator gains, effort limits, armature | already ported, reached through `G1SonicRobotCfg` |
| action scale | same formula, `0.25 * effort_limit / stiffness` |
| initial joint pose | identical |
| control rate | both `sim_dt 0.005`, `decimation 4`, 50 Hz |
| proprioception terms and histories | our v2 policy group applies 10-step histories to exactly those five terms |
| anchor body | both `pelvis`, and v2 sets it without an override |

`projected_gravity` versus SONIC's `gravity_dir`
(`quat_apply(quat_inv(robot_anchor_quat_w), (0,0,-1))`) is left as a runtime
assertion rather than a source claim. For the G1 the pelvis is the root link, so
they should agree exactly; measured live, max abs diff 4.8e-07.

### The joint order is not one order

"Identity" was too simple, and cost a day. Three orders coexist:

- the **articulation buffer** under Newton is the grouped Unitree SDK order
  (whole left leg, whole right leg, waist, left arm, right arm);
- the **action term** is `preserve_order=True` over
  `G1_29DOF_ISAACLAB_JOINT_NAMES`, the interleaved order, which is SONIC's;
- the **observation terms** `joint_pos_rel` / `joint_vel_rel` are *also*
  declared with `preserve_order=True` over that list, so the observation manager
  has already gathered them into the interleaved order — and so is
  `expert_motion`, which the reference channel indexes through `find_joints(...,
  preserve_order=True)`.

So a harness reading `robot.data` must permute, and a harness reading the
observation group must not. Both mistakes are silent: the robot falls and the
metrics read like a weak policy. The evaluator now decides it empirically at
startup by comparing each candidate alignment against the live reset pose
(measured: `joint_pos_rel` error 0.0 canonical versus 0.395 raw;
`expert_motion` 0.050 versus 0.420).

### The URDF spawn trap

Assigning the bare `UNITREE_G1_29DOF_SONIC_CFG` to `scene.robot` spawns the
robot from URDF. The URDF-to-MJC import logs `Stiffness and damping not
available joint ...` and the articulation reaches Newton **with no actuator
gains**, so every joint position target is a no-op and the robot only falls
under gravity. `robot.data.joint_stiffness` still reports the configured gains,
so it does not catch this. Use `G1SonicRobotCfg` — the same actuators on the
preconverted USD. The test that found it: drive the env with zeros and compare;
the joint state after one step matched the policy rollout to 1e-7.

## Results (2026-08-07)

Ten-motion `bones_seed_language10_v1` references, frame-0 starts, `no_push`
randomization (startup and reset randomization kept), deterministic actions,
Newton/MJWarp, seed 0.

SONIC-compatible pass — released thresholds only (`anchor_pos` and
`ee_body_pos` 0.25 m, `anchor_ori` 1.0 rad, no `foot_pos_xyz`), 100
environments, ran 1055 steps until every motion ended, seeds 0/1/2:

| quantity | Newton/MJWarp | PhysX |
| --- | ---: | ---: |
| SONIC SR (completed without a failure term) | **1.000** (100/100, every seed) | **1.000** |
| success-only MPJPE-L, micro-averaged by frame | **23.39 mm** (23.28 / 23.65 / 23.23) | 25.46 mm (25.54 / 25.15 / 25.70) |
| success-only MPJPE-G | 100.25 mm | 104.41 mm |

Randomization fully off (`--randomization none`, Newton, seed 0): MPJPE-L
**19.04 mm**, MPJPE-G 34.55 mm. So startup plus reset randomization costs about
4 mm of MPJPE-L, and most of the MPJPE-G gap — the reset pose offset persists as
global root error while the pose is still tracked root-relatively.

Full-horizon diagnostic — every early termination disabled, `no_push`, 10
environments, 1055 steps, no falls: MPJPE-L 23.74 mm, MPJPE-G 121.22 mm. Video:
`logs/sonic_release_eval/videos/sonic_release_full_horizon_seed0/sonic_release_full_horizon_seed0-step-0.mp4`.

This reproduces the paper's sub-25 mm MPJPE claim. Ten motions, our references,
our backends — a native reproduction on our protocol, not SONIC's dataset or
split.

**Isaac Sim's PhysX does not help.** It is consistently ~2 mm worse than
Newton/MJWarp, three-seed ranges disjoint (25.15-25.70 versus 23.23-23.65).

### SONIC v1.1 checkpoint

The public `sonic_v1_1/last.pt` checkpoint has a larger decoder and uses
`motion_anchor_ori_heading_mf_nonflat` for the encoder root orientation instead
of `motion_anchor_ori_b_mf_nonflat`. It cannot use the original release adapter
as is, even though the input width is still 640. The v1.1 adapter reconstructs
the heading-relative orientation from our full `expert_anchor_ori_b` and the
live robot root heading.

The table below uses the same selected-ten references, assignment, backend,
randomization profile, deterministic actions, and SONIC-compatible success
criterion for both public checkpoints. v1.1 keeps perfect completion and lowers
root-relative tracking error on this subset.

| checkpoint | SONIC SR | success-only MPJPE-L | success-only MPJPE-G | anchor position |
| --- | ---: | ---: | ---: | ---: |
| `sonic_release/last.pt`, matched selected-ten | 1.000 (100/100) | 23.53 mm | 99.94 mm | 0.0968 m |
| `sonic_v1_1/last.pt`, seed-0 selected-ten | **1.000** (100/100) | **21.17 mm** | 100.99 mm | 0.0988 m |

On the matched 4096-motion block, v1.1 also improves local tracking slightly
and completes two more motions, but its global/root drift is larger:

| checkpoint | SONIC SR | success-only MPJPE-L | success-only MPJPE-G | anchor position |
| --- | ---: | ---: | ---: | ---: |
| `sonic_release/last.pt` | 0.98999 (4055/4096) | 28.49 mm | 196.99 mm | 0.193 m |
| `sonic_v1_1/last.pt` | **0.99048** (4057/4096) | **26.93 mm** | 228.82 mm | 0.226 m |

The rank assignment is identical for the two public checkpoints. This update
does not change the later comparison against our tracker, which uses the
original release checkpoint.

### Matched comparison against our own tracker

Identical references, protocol, thresholds, backend, seeds, and MPJPE definition
(both root-relative over the same 14 `G1_TRACKED_BODY_NAMES`):

| tracker | MPJPE-L, seeds 0/1/2 | mean | SR |
| --- | --- | ---: | ---: |
| **ours, rollout-24 gamma-0.97 3.5B latent** | 15.60 / 15.68 / 15.87 | **15.72 mm** | 1.000 |
| released SONIC | 23.28 / 23.65 / 23.23 | 23.39 mm | 1.000 |

Both complete every motion, so this is not a success-rate trade. Results in
`logs/language10_ours_eval/`. But this is the set our checkpoint trained on and
where both controllers are perfect — see the 4096-motion comparison below, which
is the one that actually ranks them.

### The 4096-motion matched comparison

Identical motions (ranks 4096-8191, verified equal and unique on both sides),
`no_push`, SONIC thresholds, Newton, seed 0:

| | released SONIC | ours, rollout-24 3.5B |
| --- | ---: | ---: |
| SONIC SR | **0.9897** (4054/4096) | 0.9124 (3736/4096) |
| success-only MPJPE-L | 28.48 mm | **24.55 mm** |
| MPJPE-L on the 3733 motions **both** completed | 27.53 mm | **24.53 mm** |
| success-only MPJPE-G | 194.87 mm | - |
| anchor position error | 0.282 m | - |

Success-only MPJPE across different success rates is not comparable, so the
intersection row is the honest one; the biased reading overstated our advantage
by a quarter (3.95 vs 3.00 mm). The released checkpoint completes 321 motions
ours fails; ours completes 4 that SONIC fails.

They trade rather than rank: **the released checkpoint is markedly more robust —
better SR than every one of our trackers on this protocol — while ours tracks
~3 mm tighter on what it completes.** SONIC's 195 mm global against 28 mm local,
with 0.282 m of anchor error, is pose held accurately while drifting globally.

**Trap: `sequential` does not mean "ranks 0..N".** It advances a global cursor,
so the block a run scores depends on how many resets happened during setup. The
first attempt scored SONIC on ranks 4096-8191 and our tracker on 12288-16383 —
disjoint, silently. Pin with `--trajectory_ranks` and compare the recorded
`trajectory_ranks_sha256` before believing a cross-checkpoint number.

### Withdrawn: the first SR 1.000 / 36.64 mm reading

The SR stands; every tracking metric in it does not. Isaac Lab resets a finished
environment *inside* `step`, and the command manager recomputes its metrics on
the fresh post-reset state — the robot sitting on its new reference. The
evaluator snapshotted metrics after `step` returned, so every episode that ended
reported that placement noise as its tracking error. The tell was a
`--randomization none` run scoring MPJPE 0.004 mm and anchor error 1.5e-06 m:
impossible for a policy, exactly right for a robot just placed on its reference.
The evaluator now carries each environment's last in-episode reading and commits
that when the episode ends. `evaluate_checkpoint.py` never had this bug — it
accumulates its own per-step metrics under an active mask — so the 24.90 mm
rollout-24 number is unaffected.

### Calibration, closed empirically

With the faults above fixed, the layout sweep separated cleanly (32
environments, 200 steps, terminations off, so MPJPE discriminates):

| proprioception order | history order | MPJPE-L |
| --- | --- | ---: |
| **gravity_last** | **oldest_first** | **25.06 mm** |
| gravity_last | newest_first | 388.08 mm |
| gravity_first | oldest_first | 389.87 mm |
| gravity_first | newest_first | 396.23 mm |

The 930 vector is therefore SONIC's `PolicyCfg` field-declaration order with
gravity last, each term flattened oldest frame first. The stride-5 window starts
at the current frame: measured live, the slot-0-to-slot-1 gap is about 4.4x the
gap one control step opens between successive slot 0s.

An earlier sweep of the same 2x2 returned four **bit-identical** rollouts at
SR 0.000. That is withdrawn: the actions were not reaching the articulation, so
it measured nothing. Four different decoder inputs producing one trajectory is
the signature of that fault, not evidence about layout.
