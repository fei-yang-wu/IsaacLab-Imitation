# Sharpa source-parity training on ARCTIC box grab

Status: **unblocked 2026-09-07, calibration to be re-planned.** The first
calibration (job 3771925) proved the profile, container, staging and kit-less
runtime, and also proved the task did not learn. The cause was a defect in the
installed Isaac Lab's Newton backend, not in the recipe or the data; it is
corrected in this repository, the local gate and a local training smoke now
pass on Newton, and the data has been re-staged. The `source_parity` chain
waits on a fresh calibration with the corrected stack.

## Research question

Does this workspace's port of the video-to-data (CHORD) Sharpa recipe train a
dual floating-hand manipulation policy on a real hand-object capture, at the
released recipe scale?

One arm changes nothing about the recipe. It is a reproduction, not an
ablation. `calibration` is the same arm truncated to 40 PPO updates and exists
only to prove the profile, the container, the data staging, and the memory
fit before a long job is queued.

## Protocol

| Item | Value |
| --- | --- |
| Task | `Isaac-Sharpa-V2D-Source-v0` (source-parity variant) |
| Reference | `arctic_box_grab_speed05`, ARCTIC `s07/box_grab_01`, 965 frames at the released half-speed playback |
| Trainer | RLOpt PPO, `scripts/rlopt/train_newton.py`, `physics=newton_mjwarp` |
| Environments | 4096 (the released count) |
| Budget | 30000 PPO updates = 2.95B environment frames |
| Seeds | 0 (add seeds only after the first arm completes) |
| W&B | project `v2d_hands`, group `sharpa-arctic-source-parity` |

The curriculum is indexed by PPO updates, not frames, so `max_iterations` is
what reproduces the released schedule. The last stage begins at update 15500
and the released checkpoints are `model_30000`.

Newton was chosen as the backend because PhysX faults on this scene above a
few environments. Both of those statements are now qualified: see Defect 2 and
Defect 3 below. Neither backend currently runs the task correctly.

The hands and the object are pre-converted to USD by
`scripts/data/convert_sharpa_assets_to_usd.py`, so nothing needs Kit's URDF
importer and Newton runs with `--assert-kitless`, exactly as the G1 imitation
runs do on ICE. No display is involved.

## Slurm facts, verified on the login node 2026-09-06

| | Value |
| --- | --- |
| `wu-lab` max walltime | **4:00:00** |
| `overcap` max walltime | 2-00:00:00, same L40S nodes, preemptible |
| QOS `short` / `long` max wall | 2-00:00:00 / 7-00:00:00 |
| L40S availability | `gpu:l40s:8` on two nodes, `gpu:l40s:4` on one |

The partition cap binds regardless of QOS, so a `wu-lab` segment cannot
exceed four hours. `partition` and `qos` are profile-level fields; a stage may
set only `gres`, `mem`, `cpus_per_task`, and `time_limit`. Moving to the
two-day `overcap` window means editing `profile_skynet.yaml`, and accepting
preemption.

Segments therefore run 3:59:00 and chain with `afterany`, so a TIMEOUT
predecessor still releases its successor. Every segment carries the **full**
frame target; the walltime ends it and the next one continues from the
checkpoint's `cumulative_env_frames`. A segment that finds the budget already
complete runs zero iterations and exits cleanly, so re-submitting the same
four-segment chain until the budget completes is the intended operation.

## Prerequisites

1. **Data staging: done 2026-09-06.** 4.5 MB under the Skynet `data_dir`,
   which binds to `/data`. Preflight confirms both paths.

   ```
   data/dexmanip/sharpa/arctic_box_grab_speed05/   -> /data/dexmanip/sharpa/arctic_box_grab_speed05/
   data/dexmanip/sharpa/assets/arctic_box_bottom/  -> /data/dexmanip/sharpa/assets/arctic_box_bottom/
   ```

2. **Portable asset paths: done 2026-09-06.** A `DexterousReference` records
   absolute asset paths, so a staged copy pointed at paths that do not exist
   on the cluster and the environment would have failed before training.
   ILTools now resolves a moved asset by searching the Reference's own
   directory and up to six parents for the tail of the recorded path, longest
   tail first, and accepts a candidate **only** when its SHA-256 equals the
   recorded digest. Relocation therefore changes where a file may be found,
   never whether its content is the declared one, and a Reference with no
   declared digest is never relocated. Verified against this dataset with an
   unreachable recorded path, plus four unit tests in
   `ImitationLearningTools/tests/core/test_dexterous_reference.py`.

3. **Skynet Slurm binaries are off the non-login PATH: fixed 2026-09-06.** The
   first submission uploaded its workspace and then failed with
   `sbatch: command not found` (exit 127). A non-login ssh shell does not read
   the profile scripts that add Slurm to PATH. `ClusterProfile` now takes an
   optional `slurm_bin_dir`, prepended to PATH for `sbatch`, `squeue`,
   `sacct`, and `scancel`; the Skynet profile declares
   `/opt/slurm/Ubuntu-20.04/current/bin`. The value is recorded in the
   submission record so `status` and `cancel` reach the same binaries. Three
   tests cover it in `test_cluster_config.py`.

   This was the first real exercise of the Skynet profile, which had never
   been submitted against. Its `gres` default is `gpu:a40:1`; this campaign
   overrides it per stage to `gpu:l40s:1`.

## Memory budget

Measured locally with `scripts/bench/measure_task_memory.py` (Newton, PPO):
520 MiB fixed plus 8.8 MiB per environment, linear from 128 to 2048
environments with under 50 MiB of fit error. 4096 environments extrapolates to
**about 36.4 GB against a 40 GB card**. That is roughly 3.6 GB of headroom and
it is an extrapolation, not a measurement. The `calibration` arm exists to
turn it into a measurement. If it fails, drop to 3072 environments
(`--set vars.train_num_envs=3072`, about 27.6 GB).

## Commands

Local qualification (already run, 512 environments, Newton, 27M frames):

```bash
export SHARPA_REFERENCE_MANIFEST=$PWD/data/dexmanip/sharpa/arctic_box_grab_speed05/manifest.json
pixi run -e isaaclab python scripts/rlopt/train_newton.py \
  --task Isaac-Sharpa-V2D-Source-v0 --algo PPO --num_envs 512 \
  --max_iterations 2200 --seed 0 --headless \
  agent.logger.backend=csv physics=newton_mjwarp
```

Plan (never submits; read the preflight table it prints):

```bash
./submit.sh calibration 0
./submit.sh source_parity 0
```

Submit only after reading the plan output and confirming the printed hash:

```bash
pixi run python -m imitation_experiments.pipeline.cluster submit \
    --plan <plan_dir> --confirm <PLAN_SHA>
```

## Data identity

| Artifact | Identity |
| --- | --- |
| Source capture | ARCTIC `raw_seqs` `s07/box_grab_01`, checksum-verified against the official manifest |
| Object | ArtiGrasp `box`, bottom part, rigid proxy written by `scripts/data/make_rigid_object_urdf.py` |
| Reference set | `data/dexmanip/sharpa/arctic_box_grab_speed05`, 965 frames, contact geometry on 764 |
| Retarget | ILTools Pink port, `--mano-to-robot-scale 1.0`, `--motion-speed 0.5` |

Retarget parity against the upstream solver on this sequence: wrist position
within 0.58 mm and wrist rotation within 0.84 mrad, object pose exact, but the
finger channel reaches 0.1104 rad on 5 of 965 frames and therefore **fails**
the threshold the synthetic fixture passed. See `wiki/sharpa-progress.md`. The
gate for redundant inverse kinematics on real data is an open decision; no
source-parity claim rests on this sequence yet.

## Results

| Arm | Job | Result |
| --- | --- | --- |
| `calibration` seed 0 | 3771925 | COMPLETED, 20:14, exit 0:0, on an A40; no learning signal (Defect 2) |

The job did what a capacity check is for and then some. It ran all 40 PPO
updates at 4096 environments, kit-less, at 3673 environment frames per second,
and exited cleanly. Five earlier submissions (3771894 through 3771923) failed
on profile, container and staging problems that are now fixed.

It did **not** measure memory: the batch script samples `nvidia-smi` before and
after the job, never during, so both samples read 0 MiB. The 4096-environment
fit on a 48 GB card remains an extrapolation, not a measurement. Fix the batch
script to sample during the run before treating 4096 as proven.

It also showed the task does not train. Every logged quantity was empty:

| Quantity | Value |
| --- | --- |
| mean step reward | NaN |
| mean episode length | 1.0 control step |
| policy loss | NaN |
| every `Episode_Reward/*` term | 0 |

The local 512-environment qualification run had the same signature from its
first iteration through iteration 650, about 90 minutes, before it was killed.
Neither run is evidence about the recipe. Both are evidence about two defects.

## Defect 1: the object had no rigid body (fixed 2026-09-06)

`object_body_prim_path` assumed that a USD proxy carries its rigid body on the
asset root, and returned `{ENV_REGEX_NS}/object`. That is wrong for the USD
this campaign uses, because it was written by Isaac Lab's own URDF converter,
which keeps the importer's `Geometry/<link>` nesting. The real body is at
`{ENV_REGEX_NS}/object/Geometry/object`.

The consequence was silent. PhysX logged `Failed to find rigid body at
'/World/envs/env_0/object'` and `Pattern '/World/envs/env_*/(object)' did not
match any rigid contact for filters`, then the contact sensor failed to
initialize. The object was not simulated and no hand-object contact could be
sensed.

The path is now read from the stage instead of assumed, and the read happens
in a child interpreter because opening a USD stage before Isaac Sim starts
makes Kit crash during `SimulationApp._start_app`. Four tests cover it in
`source/isaaclab_imitation/tests/test_sharpa_source_parity_contract.py`,
including one that requires the URDF and its converted USD to agree.

With the fix, a PhysX reset places both wrists exactly on the reference
(0.0000 m error) and rewards are finite.

## Defect 2: Isaac Lab wrote body-frame wrenches into Newton's world-frame body_f (fixed 2026-09-07)

This was the cause of the empty runs, and it is not a property of the recipe.

Isaac Lab composes every external wrench into the **body** frame
(`WrenchComposer.compose_to_body_frame`). Its Newton assets in
`isaaclab==3.0.0b2.post1` copy that body-frame wrench straight into Newton's
`state.body_f`, which Newton documents as an external wrench **in world frame**
at the center of mass. Nothing rotates in between. Under Newton every external
wrench therefore lands rotated by the inverse body orientation. The floating
hands and the virtual object are both driven by such wrenches, so under Newton
the hands drifted 0.10 to 0.16 m per control step and the object left at
1.9 m/s with nothing touching it. Every episode ended on its first step and the
statistics over zero completed episodes were the NaNs the run reported.

Proof, on a contact-free pinned reset (reach frame 10), a world +Z force equal
to the object's weight:

| Passed as | Object velocity after one step |
| --- | --- |
| global (`is_global=True`) | (0.321, -0.371, -0.488) m/s: rotated, still falling |
| local, same numbers (`is_global=False`) | (0.000, 0.000, 0.000) m/s: perfect hold |

Three explanations were tested and ruled out first, each by measurement: the
tracking gains and timestep are reproduced exactly from the upstream checkout;
Newton reports quaternions in the XYZW order the controllers assume; and a
world-frame force applied through the center of mass induces no spurious spin.
Reset penetration is real (see Defect 4) but was not the cause.

`isaaclab_imitation/newton_wrench_frame.py` installs a correction at
`isaaclab_imitation.tasks` import: after composition it rotates the two output
buffers body to world, for composers attached to Newton assets only. It is
gated on the exact affected release so a fixed Isaac Lab is not double
rotated, and `ISAACLAB_IMITATION_NEWTON_WRENCH_FIX=0` disables it. Four unit
tests cover the rotation and the gate. With it installed the proof above
inverts: global holds the object, local is rotated. The installed changelog
has no entry for this; it should be reported upstream.

The correction applies to every Newton task in this repository that uses an
external wrench, including G1 push events. G1 Newton results from before
2026-09-07 were produced with misframed pushes.

## Defect 3: the object collider was a convex hull (fixed 2026-09-07)

Our port spawned the object with Isaac Lab's importer default,
`collision_type="Convex Hull"`. The release asks for `convex_decomposition`.
The ARCTIC box bottom is a container, so its hull fills the cavity and a hand
reaching inside reads as deep penetration that is not there. The port also
omitted the release's `max_contact_impulse=1e3`, `retain_accelerations`,
`enable_gyroscopic_forces`, both velocity caps, and its zero contact and rest
offsets. All are matched now.

The importer's `collision_type` does not survive this repository's
asset-transformer and multi-physics conversion, which rewrite
`physics:approximation` back to `convexHull`, so
`convert_sharpa_assets_to_usd.py` sets the token on the written payload layers
and prints how many meshes it changed. The value is greppable in the USD. Any
reference converted before this carries the old object hash and is refused by
the contract; both ARCTIC references were rebuilt against the new asset.

## Defect 4: the retarget penetrates the object (real, tolerated by design)

Measured with `scripts/audit/measure_sharpa_reference_penetration.py` against
the decomposed collider, 965 frames: **15.84 mm** at worst, **6.19 mm** on
average, 688 frames over 2 mm. Every penetrating link is a finger; the pinky
distal phalanx is worst. Against the convex hull the same data reads 35.17 mm,
which is the cavity artefact, not penetration.

This is not a port error. The upstream retargeter was run natively on the same
sequence and produces the same 15.84 mm and 6.19 mm; its own penetration filter
skips this object entirely (hull-to-mesh volume ratio 7.03, above its 3.0
cutoff) and `dataset_s07_box_grab_01` is the release's own documented example.
The release does not repair penetration. It relies on PhysX to absorb it. Once
Defect 2 was corrected, Newton absorbs it too at its default contact stiffness:
a MuJoCo probe of the reset impulse gives 54 N at the default (solref 0.02)
and 9 N at solref 0.05, and the gate passes at the default, so the softening
is kept only as an opt-in converter flag (`--contact-stiffness`).

Pressing the fingers onto the surface in MuJoCo (DexMachina style,
`iltools/retarget/sharpa_settle.py`) does not help on the grasp frames: the
decomposed walls are 8 to 10 mm thick and the retarget goes up to 15.8 mm in,
so those fingers are through the wall and a press drives them out the far
side. The machinery is kept, it is a no-op on non-penetrating frames, and it
is not used for this campaign.

## Defect 5: PhysX GPU cannot filter these contacts (open, precisely diagnosed)

The `omni.physics.tensors` contract is explicit: `filter_patterns` select
filter **rigid bodies**. With body-level filters PhysX reports, for four
environments, `Filter pattern ... did not match the correct number of entries
(expected 4, found 8)`: PhysX contact entries are per collider, and several
hand links carry two colliders (a phalanx mesh plus the fingertip elastomer),
so a body filter is ambiguous. Shape-level filters are unique but the GPU
pipeline rejects every one of the 44: `GPU contact filter for collider ... is
not supported`, after which it faults in `GpuRigidContactView` at 512
environments. Even where PhysX runs, its filtered per-link contact matrix is
0.00 N on every step, so `contact_wrench_support_reward` can never fire there.
PhysX therefore cannot run this recipe on the GPU with this hand asset unless
each link is reduced to one collider. Newton reads exact contact pairs and is
unaffected. Not pursued further while Newton works.

## Gate before any further submission

`scripts/audit/check_sharpa_backend_smoke.py` resets the task, steps it with
zero actions, and fails on a NaN reward, an excessive contact force, a
termination while tracking the Reference, or one-step episodes.

```bash
export SHARPA_REFERENCE_MANIFEST=$PWD/data/dexmanip/sharpa/arctic_box_grab_speed05_usd/manifest.json
pixi run -e isaaclab python scripts/audit/check_sharpa_backend_smoke.py \
    --num-envs 8 --steps 40 physics=newton_mjwarp
```

| Date | Backend | Stack | Verdict | Peak contact | Mean episode length |
| --- | --- | --- | --- | --- | --- |
| 09-06 | Newton | as submitted (job 3771925) | FAILED | 24.6 N | 0.5 steps |
| 09-06 | PhysX | same | PASSED, but filtered contact 0.00 N | 2.4 N | 3.5 |
| 09-07 | Newton | + convex decomposition | FAILED | 31.8 N | 0.5 |
| 09-07 | Newton | + soft contact (solref 0.05) | FAILED | 45.4 N later | 0.5 |
| 09-07 | Newton | + wrench-frame correction, default contact, 8 env x 40 steps | **PASSED** | 12.1 N | 20.5 (maximum) |

Local training smoke with the corrected stack, Newton, 512 environments,
60 PPO updates, kit-less: `r_step=-0.0141`, `ep_len=455` control steps,
10,094 frames per second. Last-iteration reward terms:
`contact_wrench_support_reward` 0.153, `hand_keypoints_tracking_exp` 0.109,
`missed_contact_penalty` -0.365; terminations: time-out 99.0 %, wrist-away
1.0 %, object-away 0.0 %. Before the correction every one of these was NaN or
zero. This is a qualification, not a result.

## Memory measurement

The batch template now samples `nvidia-smi` every 30 s for the whole job and
prints `[INFO] Peak GPU memory during job: N MiB`. Job 3771925 sampled only
before and after and read 0 MiB both times, so the 4096-environment fit is
still an extrapolation until the next calibration reports the peak.
