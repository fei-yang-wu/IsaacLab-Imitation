# Running NVIDIA's released SONIC G1 tracker in our env (Tier 2)

Goal: load NVIDIA's public GEAR-SONIC G1 controller and run it closed-loop
inside our `Isaac-Imitation-G1` env on our BONES-SEED references, to get a
SONIC number on our protocol and to align our pipeline conventions to theirs.

"Tier 2" = run their checkpoint in **our** IsaacLab env (both are IsaacLab, so
this is observation/action plumbing, not a physics rebuild). Distinct from
Tier 1 (pure-torch forward pass, no sim) and Tier 3 (their MuJoCo harness).

Artifacts (public, ungated, HF `nvidia/GEAR-SONIC`), stored at
`/mnt/hsstorage/fwu91/sonic_release/`:

- `last.pt` (469 MB) — PyTorch PPO snapshot. SHA-256
  `e6bdab3f64a39336b3d41877d4f497d05f58af275f288ec0e6746c283ded8909`.
- `config.yaml`, `observation_config.yaml` — the release env/obs contract.
- `model_encoder.onnx` (50 MB), `model_decoder.onnx` (41 MB) — deployment
  graphs, used here as bitwise ground truth for the rebuilt torch actor.

The full `gear_sonic` source is checked out at `/tmp/sonic_main_check` (not
ours; a reference read only).

## Result (2026-08-07, `evaluate_sonic_release.py`)

The library evaluator
`imitation_experiments/lowlevel/evaluate_sonic_release.py` runs the released
checkpoint on `Isaac-Imitation-G1-v2` (Newton/MJWarp) over the ten-motion
`bones_seed_language10_v1` reference arrays, frame-0 starts, `no_push`
randomization, deterministic actions.

SONIC-compatible pass (released thresholds `anchor_pos`/`ee_body_pos` 0.25 m,
`anchor_ori` 1.0 rad, no `foot_pos_xyz`), 100 environments, 1055 steps until
every motion ended, three seeds:

| quantity | Newton/MJWarp | PhysX |
| --- | ---: | ---: |
| SONIC SR (motion completed, no failure term) | **1.000** (100/100, all seeds) | **1.000** |
| success-only MPJPE-L, micro-averaged by frame | **23.39 mm** (23.28 / 23.65 / 23.23) | 25.46 mm (25.54 / 25.15 / 25.70) |
| success-only MPJPE-G | 100.25 mm | 104.41 mm |

With randomization fully off (`--randomization none`, Newton, seed 0):
MPJPE-L **19.04 mm**, MPJPE-G 34.55 mm, anchor 0.0351 m / 0.0390 rad. Startup
plus reset randomization therefore costs about 4 mm of MPJPE-L, and most of the
MPJPE-G difference: the reset pose offset persists as global root error while
the policy still tracks the pose root-relatively.

Full-horizon diagnostic (every early termination disabled, `no_push`), 10
environments, 1055 steps, no falls: MPJPE-L 23.74 mm, MPJPE-G 121.22 mm, anchor
0.154 m / 0.041 rad. Video:
`logs/sonic_release_eval/videos/sonic_release_full_horizon_seed0/sonic_release_full_horizon_seed0-step-0.mp4`.

This reproduces the paper's sub-25 mm MPJPE claim. Scope: ten motions, our
references, our backends — a native reproduction on our protocol, not SONIC's
paper dataset or split.

**The physics backend is not the lever.** PhysX is consistently ~2 mm *worse*
than Newton/MJWarp here, with disjoint three-seed ranges (25.15-25.70 versus
23.23-23.65). Whatever separates a run from the paper number, it is not the
choice of Isaac Sim's PhysX over MJWarp.

### Our own tracker beats it on the same ten motions

Matched comparison, 2026-08-07: identical references, protocol, thresholds,
backend, seeds, and MPJPE definition (both root-relative over the same 14
`G1_TRACKED_BODY_NAMES`, each side's own root position subtracted). Our
rollout-24 gamma-0.97 3.5B latent tracker
(`logs/rollout24_gamma097_foot_disabled_eval/checkpoints/model_step_3500015616.pt`
with its bound encoder) versus the released SONIC checkpoint:

| tracker | MPJPE-L, seeds 0/1/2 | mean | SR |
| --- | --- | ---: | ---: |
| **ours, rollout-24 3.5B** | 15.60 / 15.68 / 15.87 | **15.72 mm** | 1.000 |
| released SONIC | 23.28 / 23.65 / 23.23 | 23.39 mm | 1.000 |

Both complete every motion, so this is not a success-rate trade: ours is 33%
lower error at the same perfect completion. Results in
`logs/language10_ours_eval/`.

### On the canonical rank block: comparable to every 4096-env eval we have

Every existing 4096-env evaluation in this repo scores trajectory ranks
**12288-16383** (`jq -c '[.per_environment[].trajectory_rank]' | sha256sum` =
`786ef6775930c34179b774cb215e233c3f7b2bb32ef46bb6fc660206324e8285`). Scoring the
released checkpoint there, with the same hash verified, makes it comparable to
the whole table at once:

| checkpoint | frames | SONIC SR | MPJPE-L |
| --- | --- | ---: | ---: |
| **released SONIC** | - | **0.9937** | 28.65 mm |
| ablation rollout24 | 6.25 B | 0.9238 | 24.28 mm |
| ablation gamma099_rollout24 | 6.55 B | 0.9229 | 31.78 mm |
| rollout24 gamma-0.97 | 3.5 B | 0.9124 | **24.90 mm** |
| fsq_scale old_z256 | 5.0 B | 0.9060 | 24.52 mm |
| ablation control | 5.90 B | 0.9021 | 28.77 mm |
| ablation gamma099 | 5.65 B | 0.8967 | 35.99 mm |
| fsq64_sonic | 5.0 B | 0.8940 | 25.74 mm |
| fsq64_tuned | 5.0 B | 0.8710 | 27.81 mm |

The released checkpoint completes 4070/4096 - **about 7 points above our best** -
while sitting mid-pack on MPJPE. Its MPJPE-G 172.1 mm against 28.6 mm local
repeats the drift signature. (The 0.240 m anchor figure from these runs is the
value at the **final step**, not an episode mean: the anchor metrics on the
command term held the instantaneous error, which `CommandTerm.reset` logs at the
step the episode ends. Fixed 2026-08-07 - they now accumulate like MPJPE, with
the terminal value kept under `anchor_*_final_*`. Anchor numbers recorded on
this page predate that fix and are terminal-step samples; the MPJPE numbers were
already episode means and are unaffected.) **The gap is
robustness, not tracking precision.** Result:
`logs/sonic_release_4096/sonic_release_ranks12288_16383.json`.

The two rank blocks agree, which is a useful check that the block choice does
not drive the conclusion: ranks 4096-8191 gave SR 0.9897 / 28.48 mm, ranks
12288-16383 gave SR 0.9937 / 28.65 mm.

### The 4096-motion matched comparison (the one that ranks them)

Both controllers on the **identical** 4096 BONES-SEED motions - trajectory ranks
4096-8191, verified equal and unique on both sides - `no_push`, SONIC-compatible
thresholds, Newton, seed 0, frame-0 starts:

| | released SONIC | ours, rollout-24 3.5B |
| --- | ---: | ---: |
| SONIC SR | **0.9897** (4054/4096) | 0.9124 (3736/4096) |
| success-only MPJPE-L | 28.48 mm | **24.55 mm** |
| success-only MPJPE-G | 194.87 mm | - |
| anchor position / orientation error | 0.282 m / 0.051 rad | - |
| failures | 42 `ee_body_pos` | 286 `ee_body_pos`, 61 `anchor_ori`, 24 `anchor_pos` |

Success-only MPJPE across *different* success rates is not comparable - the
controller that fails the hard motions is scored on an easier subset. Restricted
to the **3733 motions both completed**: released SONIC **27.53 mm**, ours
**24.53 mm**. The 3 mm tracking advantage survives the bias correction; the
biased reading (3.95 mm) overstated it by a quarter. SONIC completes 321 motions
ours fails; ours completes 4 that SONIC fails.

So the two trade rather than rank: **the released checkpoint is far more robust
(8 points more motions completed, and better than every one of our trackers on
this protocol), ours tracks ~3 mm tighter on what it completes.** Its MPJPE-G of
195 mm against 28 mm local, with 0.282 m of anchor position error, is the
signature the data-plane docstring warns about - holding pose accurately while
drifting globally.

This supersedes the earlier reading of the ten-motion result. Our 15.72 mm there
was on a motion set our checkpoint trained on and where both controllers are
perfect; it is not a general ranking, and the worry that the released checkpoint
would not generalize to our retarget is refuted outright.

**Trap: `sequential` does not mean "ranks 0..N".** The schedule advances a global
cursor, so which block of motions a run scores depends on how many resets
happened during setup. The first attempt at this comparison scored the released
checkpoint on ranks 4096-8191 and our tracker on 12288-16383 - disjoint sets,
silently. Pin ranks explicitly (`--trajectory_ranks`) and compare the recorded
`trajectory_ranks_sha256` before believing any cross-checkpoint number.

### Superseded readings

**Withdrawn: the first SR 1.000 / 36.64 mm reading (2026-08-07 evening).** The
SR is unchanged, but every tracking metric in it was the **reset placement
noise**, not the episode's tracking error. Isaac Lab resets a finished
environment *inside* `step`, and the command manager then recomputes its metrics
on the fresh post-reset state — where the robot sits on its new reference. The
evaluator snapshotted metrics after `step` returned, so every episode that ended
reported that post-reset reading. The tell was a `--randomization none` run
scoring MPJPE 0.004 mm and anchor error 1.5e-06 m: physically impossible for a
policy, exactly right for "robot was just placed on the reference." Any
evaluator that reads `reference_command.metrics` at termination has this bug;
`evaluate_checkpoint.py` does **not** (it accumulates its own per-step metrics
under an active mask, so the 24.90 mm rollout-24 number is unaffected).

Two earlier readings are also withdrawn, not merely refined:

- `scripts/rlopt/eval_sonic_release_closed_loop.py`'s "64 envs, survival 457.9,
  fall-free 0.859" was a *fall-free survival* number under a different task and
  reset protocol, not SONIC SR.
- The first `evaluate_sonic_release.py` sweep (SR 0.000, survival 9.56, four
  proprioception layouts bit-identical) was an **artifact**. All three faults
  below (1-3) were live at once; the bit-identical rollouts were the tell,
  because a changed decoder input that changes nothing means the actions never
  landed.
  No conclusion about proprioception layout may be drawn from that sweep.

## Four faults that each read as "the policy cannot track"

Found and fixed 2026-08-07 while turning the sweep into a real number. Each is
silent: the robot falls, the metrics look like a weak policy, and nothing in the
summary says the input was wrong.

0. **Metrics read after `step` are post-reset placement noise** — see the
   withdrawn reading above. The evaluator now carries each environment's last
   in-episode reading and commits *that* when the episode ends.
1. **URDF spawn kills the actuators under Newton.** Setting
   `scene.robot = UNITREE_G1_29DOF_SONIC_CFG` spawns from URDF; the URDF-to-MJC
   import logs `Stiffness and damping not available joint ...` and the
   articulation reaches Newton with no gains, so **every joint position target
   is a no-op** and the robot only falls under gravity. `robot.data.joint_stiffness`
   does *not* reveal this - that buffer echoes the config whether or not the
   simulator honours it. Use `G1SonicRobotCfg`, the same actuators on the
   preconverted USD. Diagnosed by driving the env with zeros: the joint state
   after one step matched the policy rollout to 1e-7.
2. **The reference is already in SONIC's joint order.** The reference channel
   indexes the articulation through
   `find_joints(target_joint_names, preserve_order=True)`, so `expert_motion`
   comes out interleaved. Permuting it into SONIC order a second time scatters
   the reference across the wrong joints.
3. **So is `joint_pos_rel` / `joint_vel_rel`.** Those terms are declared with
   `SceneEntityCfg(..., joint_names=G1_29DOF_ISAACLAB_JOINT_NAMES,
   preserve_order=True)`, i.e. the observation manager already gathered them
   into the interleaved order. Only a tensor read straight off `robot.data`
   arrives in the grouped Newton SDK order. The 2026-08-07 morning correction
   below is right about the articulation buffer and wrong about this task's
   observation terms.

The evaluator now checks 2 and 3 against the live reset state instead of
reasoning about them, refuses a non-USD spawn, and carries `--action_source
zeros` for the "do the actions land at all" test.

## What is verified

### Loading (done)

`last.pt` pickles HuggingFace `trl`/`accelerate` objects this workspace does
not install. They carry no tensors. `SonicReleaseActor` loads it with a custom
unpickler that stubs any unresolvable class. Actor tensors:

- `encoders.g1`: 640 -> [2048, 1024, 512, 512] -> 64, SiLU.
- FSQ: 64 = 2 tokens x 32 levels, 32 levels each (~320 bits), no parameters.
- `decoders.g1_dyn`: 994 -> [2048, 2048, 1024, 1024, 512, 512] -> 29, SiLU.
- `994 = 64 token + 930 proprioception`; `640 = 10 frames x 64`.
- `std` (29,) is a PPO exploration scale; evaluation uses the mean action.

The `teleop` (267) and `smpl` (840) encoders and the `g1_kin` reconstruction
decoder are unused for tracking.

### Torch reimplementation (done, bitwise-exact)

`source/imitation_experiments/imitation_experiments/lowlevel/sonic_release_actor.py`.
Verified against the released ONNX on random input, max abs diff `0.0`:

- Decoder: exact (2.9e-6 float noise before an exact-int recheck).
- Encoder + FSQ: **bitwise 0.0**. Zero-input tokens match too, which isolates
  weights/FSQ from input layout.
- Regression test `tests/test_sonic_release_actor.py` (7 tests) runs against a
  saved fixture; the checkpoint-dependent test skips when `last.pt` is absent.

FSQ convention (`vector_quantize_pytorch.FSQ`, 32 levels): bound with
`eps=1e-3`, half-level shift so `z=0` maps to a level center, round with a
straight-through estimator, normalize by `L // 2 = 16`. Output lattice is
`{-1.0, -0.9375, ..., 0.9375}`, step `1/16`.

### Encoder input layout (done, recovered by probing ONNX)

The 640 encoder input is **not** ten tidy `[qpos, qvel, ori]` frames. SONIC's
flat `command_multi_future = cat([joint_pos_multi_future, joint_vel_multi_future])`
(commands.py:897 — the observation-function docstring saying "body positions"
is stale) is term-major `[jpos(290) | jvel(290)]`, reshaped to `(10, 58)`
before the 6-wide anchor orientation is appended. Each 64-wide block therefore
holds **two consecutive frames of one term**:

    block 0..4 : [ref_joint_pos[2b], ref_joint_pos[2b+1], anchor_ori[b]]
    block 5..9 : [ref_joint_vel[2b], ref_joint_vel[2b+1], anchor_ori[5 + b]]

`pack_encoder_window(joint_pos, joint_vel, anchor_ori)` reproduces this exactly
from three `[B, 10, {29,29,6}]` tensors. Recovered by a slot-by-slot ONNX probe
and confirmed bitwise.

Semantics of the 58+6:

- `ref_joint_pos` (29): **absolute** reference joint angles at stride-5 future
  frames (`dt_future_ref_frames=0.1`, 0.9 s span). Absolute, not `-default`.
- `ref_joint_vel` (29): reference joint velocities at the same frames.
- `anchor_ori` (6): 6D of `matrix_from_quat(quat_inv(robot_anchor_quat_w) @
  ref_root_quat)[..., :2]` — first two rotation-matrix columns of the reference
  root orientation **relative to the live robot anchor**, using the full robot
  orientation (pitch+roll+yaw), not heading-only. This couples the current
  robot state into the encoder input every control step.

This matches our `expert_motion` term (58 = 29 qpos + 29 qvel) plus
`expert_anchor_ori_b` (6) at `frame_stride=5`, **provided** our
`expert_anchor_ori_b` uses the same relative-to-live-robot, full-orientation,
first-two-columns, column-flatten convention (see open items).

Correction to the record: our 2026-08-07 audit and the `sonic-latent-learning-
ground-truth` memory state SONIC's BONES encoder reads 480 values of 14-body
keypoint positions + root-ori. That is the `sonic_bones_seed.yaml` experiment
config. The **released** checkpoint's g1 encoder reads joint_pos + joint_vel +
anchor_ori (58+6), i.e. our original `sonic_fsq32` feature set, not the
keypoint `sonic_fsq32_v2` (ICE 5571455) rebuild. Both are "SONIC"; they are
different configs.

### 930 proprioception layout (done)

SONIC's policy obs is five IsaacLab terms, each `history_length=10`,
concatenated **term-major**, each term's 10 frames contiguous oldest->newest
(our installed IsaacLab `CircularBuffer.buffer` returns "most recent at the
end"; `flatten_history_dim` reshapes per term):

    [gravity_dir(30) | base_ang_vel(30) | joint_pos_rel(290) | joint_vel_rel(290) | last_action(290)]

`assemble_proprioception(...)` builds this from five `[B, 10, w]` tensors. This
is **not** our `planner_state`'s `10 x 93` frame-major layout; both are 930 and
feeding one for the other is a silent, plausible error.

Per-term semantics: `gravity_dir` is gravity in the pelvis-anchor frame (SONIC
`gravity_dir`), not base-frame `projected_gravity`; `joint_pos_rel` /
`joint_vel_rel` are current robot `joint - default`; `last_action` is the
previous 29-joint action (no hand). Note the asymmetry: the **encoder** eats
absolute reference joint angles, the **decoder proprioception** eats
relative current joint angles.

### Joint order (LIVE-VERIFIED; subtler than "identity")

SONIC's 29-vectors are in the **interleaved** IsaacLab order: its
`G1_ISAACLAB_JOINTS` list equals our `G1_29DOF_ISAACLAB_JOINT_NAMES`
one-to-one (entry 9 is our `waist_pitch_joint` = SONIC's `torso_link`, i.e. the
joint whose child link is torso). So SONIC's declared order and our configured
target order are the same.

**But under Newton the live articulation buffer is a different order.** The
`--assert-kitless` Newton backend enumerates joints in the *grouped* SDK order
(full left leg, full right leg, waist, left arm, right arm) — printed live as
`robot.joint_names`. Our env logs "Articulation joint order differs from
configured target_joint_names ... rebuilding the reference->target remap" and
remaps reference/action to the interleaved target. Consequences, all confirmed
in a live rollout:

- **Reference (`expert_motion`) and the action term**: already interleaved (the
  env's remap), so **no manual permutation**.
- **A tensor read straight from `robot.data.joint_pos` / `joint_vel`**: grouped
  Newton order -> must be permuted into the interleaved order. The live
  permutation is
  `[0,6,12,1,7,13,2,8,14,3,9,15,22,4,10,16,23,5,11,17,24,18,25,19,26,20,27,21,28]`,
  built with `robot.find_joints(G1_29DOF_ISAACLAB_JOINT_NAMES, preserve_order=True)`.
- **The `joint_pos_rel` / `joint_vel_rel` observation terms are NOT such a
  tensor.** They are declared over `G1_29DOF_ISAACLAB_JOINT_NAMES` with
  `preserve_order=True`, so the observation manager already gathered them into
  the interleaved order. A harness that reads the observation group must not
  permute them; one that reads `robot.data` itself must. Measured: mean abs
  error 0.0 against the canonically ordered live state, 0.395 against the raw
  articulation buffer.

This is the [[newton-joint-order-invalidation]] trap, and it cuts both ways:
reading `robot.data.joint_pos` into SONIC's decoder without a permutation is
wrong, and permuting an already-canonical observation term is equally wrong.
Both are plausible and silent. Decide it per source, empirically, at startup.

### Physical layer (already aligned — no build)

- Actuators / gains / action scale: our repo already carries
  `UNITREE_G1_29DOF_SONIC_CFG` and `UNITREE_G1_29DOF_SONIC_ACTION_SCALE`, byte-
  identical to SONIC's per-group tables. Use them through `G1SonicRobotCfg`
  (same actuators on the preconverted USD), never by assigning the bare
  `UNITREE_G1_29DOF_SONIC_CFG` to `scene.robot`: that spawns from URDF and the
  gains do not survive the import into Newton. Tables: arms 25/25/25/25/25/5/5 N.m,
  5020x5+4010x2; feet/waist 2x5020; waist_yaw 7520-14; legs hip_yaw 88/7520-14,
  hip_roll+knee 139/7520-22; hip_pitch overridden to 139/7520-22). Action scale
  `0.25 * effort_limit / stiffness`, same formula.
- Timing: `sim_dt=0.005`, `decimation=4`, 50 Hz control on both sides.
- Their extra: `action_clip_value=20.0` (clip action before scaling); mirror it.
- Init pose matches (hip_pitch -0.312, knee 0.669, ankle_pitch -0.363, elbow
  0.6, shoulder roll +-0.2).

## Convention checks (resolved by reading both sources)

1. **Anchor-ori 6D matches.** Our `expert_anchor_ori_b`
   (`body_pose_in_anchor_frame` -> `quat_to_rot6d_flat`) computes
   `matrix_from_quat(q_robot_anchor^-1 @ q_ref)[..., :2].reshape(N, -1)`,
   algebraically identical to SONIC's `root_rot_dif_l` (commands.py:1924):
   relative to the live robot anchor, full orientation, first two rotation-
   matrix columns, column-major flatten. `_compiled.py:40` already documents
   aligning to SONIC's scalar-first WXYZ convention and the roll-quaternion
   trap. The observation-function default is `anchor_body_name="torso_link"`,
   but `Isaac-Imitation-G1-v2` sets the reference channel's anchor to `pelvis`
   and propagates it to the observation and reward terms, so it already matches
   SONIC's `anchor_body: pelvis` (config:377) with no override. A harness on any
   other surface must set it explicitly.
2. **`gravity_dir` == `projected_gravity`.** SONIC's `gravity_dir` is gravity in
   the pelvis-anchor frame (full orientation). The G1 pelvis is the base/root
   link, so this equals our base-frame `projected_gravity`. No heading-only
   canonicalization is applied.
3. **`expert_motion` is absolute reference joint angles**, matching SONIC's
   `joint_pos_multi_future` (both raw from the motion library, not `-default`).

## Calibration items (all closed 2026-08-07 against live rollouts)

Once the three faults above were fixed, the 2x2 layout sweep separated cleanly.
32 environments, 200 steps, all early terminations disabled, so MPJPE rather
than survival does the discriminating:

| proprioception order | history order | MPJPE-L |
| --- | --- | ---: |
| **gravity_last** | **oldest_first** | **25.06 mm** |
| gravity_last | newest_first | 388.08 mm |
| gravity_first | oldest_first | 389.87 mm |
| gravity_first | newest_first | 396.23 mm |

1. **History flatten direction: oldest first**, as IsaacLab's `CircularBuffer`
   documents. (It also warms the whole buffer with the first pushed value on
   reset, so the first ten steps are not zero-padded.)
2. **Proprioception order: gravity last**, i.e. SONIC's `PolicyCfg` field
   declaration order, not the training YAML's key order.
3. **Stride-5 window origin: slot 0 is the current frame.** Measured live: the
   slot-0-to-slot-1 gap is ~4.4x the gap one control step opens between
   successive slot 0s, and slot 0 matches the live pose at reset (mean abs
   error 0.05 rad, against 0.42 for the permuted alternative).
4. **`last_action`** is the raw 29-joint action buffer in the action term's
   order, which for this task is already SONIC's order.

## How to run it

`imitation_experiments/lowlevel/evaluate_sonic_release.py` boots
`Isaac-Imitation-G1-v2`, forces the released checkpoint's contract
(`G1SonicRobotCfg` actuators, SONIC action scale, the stride-5 encoder view,
10-step histories on the five proprioception terms, observation corruption off),
and drives joint targets directly. It does **not** use the IPMD command
interface: SONIC's actor is a separate encoder -> FSQ -> decoder module.

SONIC-compatible success pass:

```bash
REFDIR=data/bones_seed_language10_v1/reference_arrays/root_qpos_v1
pixi run -e isaaclab python -u -m imitation_experiments.lowlevel.evaluate_sonic_release \
  --sonic_checkpoint /mnt/hsstorage/fwu91/sonic_release/last.pt \
  --num_envs 100 --steps 1500 --seed 0 --headless \
  --randomization no_push --reference_start_frame 0 \
  --proprioception_order gravity_last --history_order oldest_first \
  --label sonic_release_compatible_seed0 \
  --output_json logs/sonic_release_eval/sonic_compatible_seed0.json \
  physics=newton_mjwarp env.data.manifest=null \
  env.data.reference_arrays_dir="$REFDIR" \
  env.data.persist_id='bones_seed_language10_v1@60a5b7a5' \
  env.episode_length_s=40.0 env.events.push_robot=null \
  env.terminations.anchor_pos.params.threshold=0.25 \
  env.terminations.anchor_pos.params.down_threshold=0.25 \
  env.terminations.anchor_ori.params.threshold=1.0 \
  env.terminations.ee_body_pos.params.threshold=0.25 \
  env.terminations.ee_body_pos.params.down_threshold=0.25 \
  env.terminations.foot_pos_xyz=null
```

`env.episode_length_s` must exceed the longest clip, or `time_out` truncates a
motion before `reference_finished` and SONIC SR is understated. Require
`done_rate == 1.0`, `steps_run < steps`, and no `time_out` in the termination
causes before reporting.

Full-horizon diagnostic plus video: add `--disable_early_terminations --video
--video_length <steps>` and drop the termination overrides. The retained video's
absolute path is printed as `[VIDEO] ...`.

Diagnostics built into the evaluator, in the order they earn their keep:

- `--action_source zeros` holds the default pose. If the rollout is unchanged,
  the actions are not reaching the articulation and no number is real.
- `--diagnose_steps N` prints per-step checksums of the encoder window, the
  proprioception vector, the action, the processed joint target, and the
  resulting joint state, plus the window cadence ratio.
- Startup: refuses a non-USD robot spawn, and checks both `expert_motion` and
  `joint_pos_rel` against the live reset pose in each candidate joint order.

`scripts/rlopt/eval_sonic_release_closed_loop.py` is the earlier standalone
harness on `Isaac-Imitation-G1-Sonic-v0`. It reads `robot.data` directly (so it
*does* need the articulation->interleaved permutation) and reports fall-free
survival, not SONIC SR. Keep it for the kitless Newton path; use the library
evaluator for numbers.
