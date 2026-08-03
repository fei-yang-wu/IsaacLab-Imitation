# Sim2Sim Backend Verification: Newton vs PhysX

Status as of 2026-08-03. This page records why Newton-trained G1 low-level
checkpoints collapse under the PhysX/Isaac Sim backend.

Read this before running any cross-backend evaluation or interpreting a
Newton-vs-PhysX comparison in `wiki/isaaclab3-cu130-runtime-migration.md`.

**Start with [the 2026-08-03 section](#2026-08-03-the-order-is-clean-the-gap-is-dynamics).**
Everything above it describes the 2026-07-21 ordering leak, which is fixed and
now measured to be fixed. The remaining gap is not an indexing problem.

## Summary

Isaac Lab backends enumerate the G1 articulation differently: PhysX is
breadth-first, Newton/MJWarp is depth-first per limb. **27 of 29 joint slots
differ.** The repo pins a canonical order, `G1_29DOF_ISAACLAB_JOINT_NAMES`,
which is *the PhysX order*.

The pin was applied to proprioception, body observations, and the action
targets. It was **not** applied to the expert command path or to the action
offset. Both leaks are no-ops under PhysX and active under Newton, so a
Newton-trained checkpoint encodes a Newton-specific joint permutation.

## The two leaks

### 1. Expert command terms are emitted in live joint order

`_g1_expert_motion_obs_params` in
`source/isaaclab_imitation/isaaclab_imitation/tasks/manager_based/imitation/config/g1/imitation_g1_env_cfg.py:245`
builds `SceneEntityCfg("robot", joint_names=G1_29DOF_JOINT_NAMES)` with **no
`preserve_order=True`**, unlike the adjacent `_g1_canonical_joint_obs_params`
at :291.

`SceneEntityCfg` with `preserve_order=False` returns indices ascending in the
live enumeration, so the selection is the identity and the values stay in
*live* order. The expert frame itself is already live-ordered:
`_align_reference_target_joints_to_articulation`
(`envs/imitation_rl_env.py:530`) explicitly sets
`tm.target_joint_names = robot.joint_names` and rebuilds the remap.

Affected terms (8): `expert_motion` in `policy`, `critic`, `expert_state`,
`expert_goal`, `expert_window`, and `reward_input`, plus `joint_pos` and
`joint_vel` in `expert_state`. The same defect exists in
`_g1_expert_window_motion_obs_params` at :261.

Because `reward_input.expert_motion` is affected, the IPMD reward and
discriminator signal was also computed on permuted data during Newton training.

### 2. The action offset is written in live order into pinned slots

`randomize_joint_default_pos`
(`tasks/manager_based/imitation/mdp/events.py:80`) ends with

```python
offset[env_ids_for_slice, joint_ids] = selected_pos
```

`selected_pos` comes from `asset.data.default_joint_pos` (live order) and
`joint_ids` is `slice(None)`, but the action term's `_offset` is in the pinned
order. Measured under Newton: slot 3 holds `0.675`, the *knee* default, while
pinned slot 3 is `left_hip_roll_joint`. Under PhysX the same slot holds
`-0.0034`, which is correct.

## Reproducing the audit

`scripts/audit/dump_backend_index_contract.py` dumps every resolved joint/body index,
the action offset/scale, and all observation term widths, then diffs two
backends and classifies each term as a correct remap, a wildcard selection, or
a live-order leak.

```bash
pixi run -e isaaclab python scripts/audit/dump_backend_index_contract.py \
    --task Isaac-Imitation-G1-Latent-Strict-v0 --num_envs 2 --headless \
    --output logs/index_contract/newton.json physics=newton_mjwarp \
    env.lafan1_manifest_path=data/lafan1/manifests/g1_lafan1_walk1_subject1_manifest.json

OMNI_KIT_ACCEPT_EULA=YES pixi run -e isaaclab python scripts/audit/dump_backend_index_contract.py \
    --task Isaac-Imitation-G1-Latent-Strict-v0 --num_envs 2 --headless \
    --output logs/index_contract/physx.json physics=physx \
    env.lafan1_manifest_path=data/lafan1/manifests/g1_lafan1_walk1_subject1_manifest.json

pixi run -e isaaclab python scripts/audit/dump_backend_index_contract.py --compare \
    logs/index_contract/newton.json logs/index_contract/physx.json
```

This reports 9 leaks and cleanly separates them from the ~20 terms whose ids
differ because `preserve_order=True` is doing its job.

## Behavioral confirmation

Checkpoint `L1_strict/model_step_992870400.pt` (Newton-trained, ~993M frames),
`walk1_subject1`, seed 0, 500 control steps, terminations disabled, identical
otherwise. `--emulate_joint_order_from` permutes `expert_motion` and the action
offset into the other backend's ordering.

| Arm | Backend | Command/offset order | Order | Survived | Fall | Joint MAE (first 67 steps) |
| --- | --- | --- | --- | ---: | ---: | ---: |
| A | PhysX | native (canonical) | mismatched | 67/500 | 1.34 s | 0.517 rad |
| B | PhysX | Newton-emulated | matched | 323/500 | 6.46 s | 0.240 rad |
| C | Newton | native (Newton) | matched | 500/500 | never | 0.110 rad |
| D | Newton | PhysX-emulated | mismatched | 111/500 | 2.22 s | 0.431 rad |

Removing the mismatch on PhysX (A to B) raises survival 4.8x. Injecting it on
Newton (C to D) cuts survival 4.5x. Joint tracking error is ~0.43-0.52 rad
whenever the ordering is mismatched and ~0.11-0.24 rad when it is matched, in
both directions. The ordering leak is therefore the dominant cause.

**A residual gap remains.** Arm B still falls at 6.46 s with 0.240 rad error
against Newton's 0.110 rad. That residual is the genuine solver difference and
is the part that domain randomization would need to address.

## Genuine backend differences (next to quantify)

With the ordering fixed, one already shows up in the deterministic contract
dump: at reset, on the same reference pose with randomization disabled, the
live planner frame differs between backends in 320 of 930 values, and the
blocks are **not** permutations of each other. Newton reports
`base_ang_vel = [0, 0, 0]` where PhysX reports `[0.633, 0.832, -0.432]`, and
`joint_pos_rel` differs by up to 0.137 rad while `joint_vel_rel` and
`last_action` are exactly identical. That is a reset-state fidelity difference,
not an index problem, and it is the first concrete piece of the residual gap.

The gap, the configuration asymmetries, and a tiered randomization plan are
worked through in
[Sim2Sim Dynamics Gap and Randomization](sim2sim-dynamics-gap-and-randomization.md).

The remaining suspects, still unquantified:

- PhysX runs TGS with `solver_position_iteration_count=8` /
  `velocity=4`; Newton runs MJWarp `implicitfast`, `num_substeps=1`,
  pyramidal cone, `impratio=1`.
- The packaged USD asks for `solverPositionIterationCount = 32` /
  `velocity = 1`, but `assets/robots/unitree.py:100` overrides it to 8/4.
- The USD is PhysX-authored: convex-hull and primitive colliders,
  `enabledSelfCollisions = 1`, no authored contact/rest offsets and no filtered
  collision pairs. MJWarp does not consume the PhysX solver schema.
- The checked-in Newton preset uses `njmax=95, nconmax=10`
  (`imitation_g1_env_cfg.py:171`), below the `288/32` recorded as retained in
  `wiki/current-status.md` after the BONES-SEED NaN investigation.

## The fix (2026-07-21)

1. `_g1_expert_motion_obs_params` and `_g1_expert_window_motion_obs_params`
   (`imitation_g1_env_cfg.py:245`, `:270`) now select
   `G1_29DOF_ISAACLAB_JOINT_NAMES` with `preserve_order=True`. This covers all
   eight leaked command terms, since every call site shares these two helpers.
2. `randomize_joint_default_pos` (`mdp/events.py:80`) now gathers the offset
   through the action term's own `_joint_ids` mapping instead of copying
   live-order values into pinned slots.
3. `_current_causal_planner_frame` and
   `causal_planner_observation_from_expert_frame`
   (`envs/imitation_rl_env.py`) reorder their joint blocks through the new
   `_pinned_joint_ids()` helper, so the recorded planner frame no longer mixes
   live-order joint state with a pinned `last_action`.
4. `scripts/data/batch_csv_to_npz.py` applied the SDK-to-articulation scatter
   **twice** (4f054db added it, e3ebd2b re-added an identical copy). It is a
   permutation, not an involution, so the second application moved 27 of 29
   joints while `joint_names` still claimed the correct order. Live from
   2026-07-16; removed. **Every data tree on disk is dated 2026-07-14 and is
   unaffected** — verified both by date and by checking that
   `left_knee_joint` stays strictly positive (0.156 to 1.747 rad) across 13,065
   frames of `walk1_subject1`.

Verification after the fix:

- `dump_backend_index_contract.py --compare` reports **no leaks**, and the two
  backends' action offsets agree within randomization noise.
- The offline planner frame is now **byte-identical** across backends.
- `test_g1_backend_joint_contract.py` grew from 2 to 5 tests, covering every
  command term plus a catch-all that fails on any unpinned multi-joint
  selection. Reintroducing the original bug fails 3 of the 5.
- Behavioral, same checkpoint and protocol as the table above:

| Arm | Backend | Env | Shim | Survived | Fall |
| --- | --- | --- | --- | ---: | ---: |
| C | Newton | pre-fix | none | 500/500 | never |
| E | Newton | post-fix | none | 113/500 | 2.26 s |
| F | Newton | post-fix | Newton order | 500/500 | never |

Arm F reproduces arm C to three decimal places (joint MAE 0.12504 vs 0.12518),
so the fix is exactly the inverse of the bug. Arm E confirms that legacy
checkpoints are invalidated: they now fail on Newton too.

`compare_policy_reference.py --emulate_joint_order_from <contract.json>` is
retained as a legacy-checkpoint shim. It is diagnostic only and must never
produce a paper number.

## Recorded data status

Audited separately, since a permutation can propagate into datasets:

| Artifact | Status |
| --- | --- |
| Source LAFAN1 / BONES-SEED NPZ and Zarr | **Safe.** `batch_csv_to_npz.py` stores `joint_names`/`body_names`, and every consumer remaps by name via `_map_reference_to_target`. |
| `record_policy_rollout.py` state arrays | **Contaminated and mislabeled.** Live-order `joint_pos`/`joint_vel`/`body_*_w`, but the file writes no `joint_names` key, so the loader falls back to the pinned list and silently labels Newton data as PhysX. Actions are pinned, so a single file mixes two orderings. |
| Planner sample rows (`.pt`) | **Contaminated.** `causal_state_history` and `demonstration_state_history` carry 58 live-ordered values plus 29 pinned ones per frame; no joint names and no backend recorded. Explicit-interface rows are permutable; latent rows contain encoder outputs and must be regenerated. |
| Skill encoder / DiffSR latent space | **Contaminated but self-consistent.** Train-time and runtime both fed live order, so nothing visibly broke; the weights are baked to the Newton permutation and every latent checkpoint's encoder now receives a permuted input. |

Two follow-ups, not yet done: `record_policy_rollout.py` should write
`joint_names`/`body_names` like the other two NPZ writers, and planner sample
metadata should record the physics backend.

## Implications

- Every Newton-trained checkpoint encodes a Newton-specific joint permutation.
  It cannot be deployed to PhysX or to hardware without either retraining or a
  recorded compensating permutation.
- The Newton-vs-PhysX training comparison in
  `wiki/isaaclab3-cu130-runtime-migration.md:266-283` (Newton reward 0.0324 vs
  PhysX 0.0543) was very likely measuring this leak, not the solver.
- `source/isaaclab_imitation/tests/test_g1_backend_joint_contract.py` asserts
  `preserve_order=True` for the action term and `joint_pos_rel`/`joint_vel_rel`
  only, which is why this survived. Extend it to every command term.
- Fixing the leak changes the semantic layout of `expert_motion` and the action
  offset, so it invalidates existing Newton checkpoints.

## Artifacts

- Index contracts: `logs/index_contract/{newton,physx}.json`
- Validation arms and per-step metrics:
  `logs/sim2sim_validation/{A_physx,B_physx_newton_order,C_newton,D_newton_physx_order}/`
  each with `metrics.json` and a 10 s video under
  `videos/compare_policy_reference/`.

## 2026-08-03: the order is clean, the gap is dynamics

Re-measured on the v2 surface (`Isaac-Imitation-G1-v2`) with the tuned 5B
latent checkpoint `logs/tuned_5b_eval/ckpt/tuned5b_latest.pt`. Three findings,
in the order they have to be read.

### 1. Every cross-backend probe had silently stopped controlling its inputs

`dump_backend_index_contract.py`, `diagnose_g1_dynamics.py`, and
`sim2sim_backend_eval.py` pinned the reference start frame by writing:

```python
env_cfg.random_reset_step_min = 0
env_cfg.random_reset_step_max = 0
env_cfg.random_reset_full_trajectory = False
```

Those fields live on `ImitationG1BaseTrackingEnvCfg` (the frozen v0/v1
lineage). `ImitationG1V2EnvCfg` derives from `ImitationLearningEnvCfg` and does
not have them; reset sampling moved onto
`command_interface.reference.selection`. A configclass accepts the unknown
attributes without complaint, so on **every v2 probe** the write did nothing
and the default selection stood: `schedule="random"`, start frame uniform over
0-200.

Two backends therefore scored *different motions from different frames*, with
startup randomization, reset perturbations, and interval pushes all live and
consuming the RNG at different points. `diagnose_g1_dynamics.py` additionally
could not run at all — it computed `SCRIPT_DIR / "rlopt"` from `scripts/bench/`
and died on `ModuleNotFoundError: runtime_bootstrap`.

Fixed by `imitation_experiments.audit.backend_determinism`, which writes
whichever surface the config actually has and **raises** when it recognizes
neither. All three probes now record the pinned settings in their output, and
`sim2sim_backend_eval.py --compare` refuses two runs whose protocols differ.

### 2. There is no residual joint-order leak

Controlled protocol: 32 envs, 300 steps, seed 0, frame 0, `round_robin`
trajectories, no randomization, actor mode (not sampled), tracking
terminations disabled.

`sim2sim_backend_eval.py` now snapshots the **post-reset observation**, before
any physics has run. Both backends have just been teleported onto the same
reference frame, so every term is a pure function of the reference and the
index contract; a solver cannot explain a difference there.

| check | result |
| --- | --- |
| live joint enumeration | 27/29 slots differ (as expected) |
| every `policy` + `critic` observation term | **identical**, max abs diff < 1e-4 |
| robot joint pos / vel at reset (by name) | **identical**, max abs diff 0.000000 |
| root pos / quat at reset | **identical**, max abs diff 0.000000 |
| `dump_backend_index_contract.py --compare` | **no backend-dependent index leak** |

Two independent tools agree. The canonical order (`G1_29DOF_ISAACLAB_JOINT_NAMES`,
`preserve_order=True`) is applied consistently across the command path, the
encoder window, proprioception, actions, rewards, and terminations. **A further
"pin one canonical order everywhere" refactor would not move any number here.**

### 3. The residual gap is dynamics, and Newton is the outlier

Same controlled run, the checkpoint's own rollout:

| metric | Newton | PhysX | ratio |
| --- | ---: | ---: | ---: |
| MPJPE after 1 control step | 3.55 mm | 6.77 mm | 1.90x |
| MPJPE mean over 300 steps | 19.9 mm | 334.5 mm | 16.8x |
| MPJPE final | 40.2 mm | 471.0 mm | 11.7x |

The ratio is already 1.9x after a *single* control step from a bit-identical
state, holds near 2.2x for ~7 steps, then compounds into a fall. That is a
per-step actuator/solver response difference, not an accumulating contact
artifact.

> **Superseded in part.** The paragraph below this table originally attributed
> the gap to an under-damped implicit PD / armature regime. The 2026-08-03
> MuJoCo baseline (next section) **disproves** that. The measurements here
> stand; the mechanism claim does not.

The policy-free control settles the direction. `diagnose_g1_dynamics.py
--action-mode reference` sends the oracle next-pose action and involves no
checkpoint at all (128 envs, 500 steps, frame 0, no randomization):

| metric | Newton | PhysX |
| --- | ---: | ---: |
| joint MAE | 0.0975 rad | **0.0327 rad** |
| applied torque | 7.15 Nm | **1.49 Nm** |
| mean \|action\| | 1.690 | **1.032** |
| `joint_limit` reward/s | -1.605 | **0.000** |
| `action_rate_l2` reward/s | -9.682 | **-0.747** |
| mean episode length | 4.0 steps | **9.7 steps** |

**PhysX tracks the oracle 3x better than Newton.** PhysX is not the broken
backend; the 5B checkpoint is overfit to Newton-specific dynamics. Newton's
error concentrates in the light, weakly-geared joints — `left_wrist_roll`
0.550 rad against PhysX's 0.039 (14x), then `wrist_pitch`, `elbow`,
`right_ankle_roll` (0.253 vs 0.008, 31x) — while the heavy hip/knee joints are
close. Excess torque, joint-limit violations, and 13x the action-rate penalty
on exactly the low-inertia joints is the signature of an under-damped implicit
PD, which points at the armature / stiffness-to-inertia regime under MJWarp's
`implicitfast` integrator. Not yet attributed further.

Ruled out along the way: the MJWarp contact-budget overflow. Newton logs
`Number of Newton contacts (2405) exceeded MJWarp limit (2304)` with the
shipped `nconmax=10`, but raising it to `nconmax=200, njmax=288` reproduces
every metric above **bit-identically** — the overflow hits ~2 of 64,000 steps
and changes nothing. The preset is still worth raising, but it is not this.

### Artifacts (2026-08-03)

- `logs/sim2sim_controlled/{newton_mjwarp,physx}.{json,log}` — policy rollout,
  with the full post-reset observation snapshot and per-step MPJPE trace.
- `logs/dynamics_controlled/{newton_mjwarp,physx,newton_bigcon}.{json,log}` —
  policy-free oracle probe, plus the contact-budget control.
- `logs/index_contract_v2/{newton_mjwarp,physx}.{json,log}` — v2 index contract.

### What this means for the checkpoint

The tuned 5B checkpoint cannot be deployed or evaluated on PhysX as-is, and the
reason is not recoverable by any remapping. Either evaluate it on Newton only,
or close the actuator gap and retrain. Deciding that needs a call on which
backend is the deployment target.

## 2026-08-03 (later): MuJoCo as the third party

Two Isaac backends disagreeing cannot say which is wrong, because both are
under test. Stock MuJoCo breaks the tie, and it is the right referee precisely
because `newton_mjwarp` *is* MuJoCo Warp: anything stock MuJoCo reproduces is a
property of the model, not of Isaac Lab's wrapping.

`scripts/bench/mujoco_reference_tracking_baseline.py` runs the same oracle
next-pose law over the same NPZ reference in plain MuJoCo, with the repo's own
`ImplicitActuatorCfg` numbers injected as position-servo gains
(`kp`/`kv`/`forcerange`/`armature`). It has a `--base fixed` mode, which is what
makes the measurement mean anything: an open-loop humanoid on a floating base
falls within ~1 s in *any* simulator, and past that the joint error measures the
fall rather than the actuator.

**Note on what the MJCF is.** Training spawns the G1 from
`g1_description/g1_29dof_rev_1_0.usd`; the vendored MJCF is read only by
`unitree_joint_order.py` for its actuator name order. The MJCF is not the
simulated model. What makes the comparison valid is that the runtime physical
parameters come from `ImplicitActuatorCfg` identically on both backends, and
this probe injects those same numbers.

### The Unitree parameter deltas are real but are NOT the cause

Unitree's own MuJoCo simulation model (`unitree_mujoco`,
`unitree_robots/g1/g1_29dof.xml`, commit ae6a840) declares for every joint
`armature=0.01`, `damping=0.05`, `frictionloss=0.2` (`0.1` for wrists), and
runs at MuJoCo's default `timestep=0.002`. The repo's runtime has frictionloss
0, no passive joint damping, wrist armature 0.0036-0.0043, and `sim.dt=0.005`.
Four real differences against the vendor's own validated setup.

Sweeping each one in stock MuJoCo (`--sweep`, fixed base, 300 steps,
`dance1_subject1`) moves essentially nothing:

| arm | joint MAE | chatter | torque Nm | left_wrist_roll |
| --- | ---: | ---: | ---: | ---: |
| repo runtime | 0.0620 | 0.0049 | 1.79 | 0.0207 |
| + timestep 0.002 | 0.0621 | 0.0049 | 1.78 | 0.0213 |
| + frictionloss 0.2/0.1 | 0.0627 | 0.0045 | 1.78 | 0.0240 |
| + joint damping 0.05 | 0.0622 | 0.0048 | 1.78 | 0.0221 |
| + armature 0.01 | 0.0621 | 0.0049 | 1.79 | 0.0211 |
| Unitree, all four | 0.0632 | 0.0045 | 1.78 | 0.0263 |

Every arm is stable, and none is meaningfully better. **The armature /
stiffness-to-inertia hypothesis from earlier today is wrong**, and so is the
timestep one. Adopting Unitree's joint parameters is defensible on fidelity
grounds but will not close this gap.

### MuJoCo agrees with PhysX; Isaac's MJWarp is the outlier

Per-joint MAE, on the joints where Newton is worst. `mjc-free` is the pre-fall
window of the floating-base run; `mjc-fixed` is the welded-base run:

| joint | mjc-free | mjc-fixed | IL-newton | IL-physx |
| --- | ---: | ---: | ---: | ---: |
| left_wrist_roll | 0.0364 | 0.0207 | **0.5500** | 0.0394 |
| left_wrist_pitch | 0.0184 | 0.0169 | **0.2625** | 0.0486 |
| right_ankle_roll | 0.0764 | 0.0041 | **0.2532** | 0.0082 |
| left_elbow | 0.0406 | 0.0481 | **0.2256** | 0.0449 |
| right_elbow | 0.0571 | 0.0357 | **0.2071** | 0.0475 |

Stock MuJoCo lands on PhysX's number, not Newton's. Caveat: the MuJoCo probe
runs one motion on one model, the Isaac probes run 128 environments round-robin
over all 40 LAFAN1 motions, so absolute values are not directly comparable --
`shoulder_roll` is actually *higher* in MuJoCo than in either backend. The
signal is the pattern: exactly the joints where Newton is 14-31x off are joints
where stock MuJoCo and PhysX agree closely.

### The model Isaac hands MuJoCo Warp is parameter-exact

`scripts/audit/dump_mjwarp_model_contract.py` reaches into
`NewtonManager._solver.mjw_model` after the environment is built and compares
what MuJoCo Warp holds against what the config asked for. All 29 joints:

| quantity | result |
| --- | --- |
| armature | **exact match** to `ImplicitActuatorCfg` |
| actuator `kp` (`gainprm[0]`) | **exact match** |
| actuator `kv` (`-biasprm[2]`) | **exact match** |
| actuator force range | **exact match** |
| frictionloss | 0 everywhere (as configured -- no friction is set) |
| body mass, body inertia | **exact match** for all 30 bodies |
| integrator | `implicitfast`, as configured |
| timestep | 0.005 after stepping, matching `sim.dt / num_substeps` |

The timestep needs the "after stepping" qualifier: a freshly built model still
holds MuJoCo's 0.002 default, and the solver only writes the real dt inside
`step()`. Reading it before stepping shows a mismatch that is not real -- the
audit now reports both and judges on the post-step value.

### Also ruled out

- **Contact budget.** `nconmax=10` overflows ~2 steps in 64,000; raising it to
  `200` with `njmax=288` reproduces every metric bit-identically.
- **Contact pipeline.** The preset runs `use_mujoco_contacts=False` (Newton's
  own collision pipeline). Switching to MuJoCo-native contacts also reproduces
  every metric bit-identically. Consistent with episodes ending after ~4 control
  steps, before ground contact dominates.

### Where this leaves it

The MuJoCo Warp model is correct, MuJoCo the solver is correct, and the joint
order is correct. The fault is therefore in Isaac Lab's Newton backend *around*
the solver -- the per-step data flow: how joint position targets are written,
how state is read back, or the reset/teleport path -- not in any parameter this
repo configures.

The next probe should step Newton and PhysX one control step at a time from the
identical post-reset state and diff the intermediate quantities (`joint_pos`,
`joint_vel`, `applied_torque`, the written position target) rather than the
end-of-rollout metrics. The divergence is already measurable at step 1, so it
does not need a long rollout -- it needs per-substep instrumentation.

Not yet attributed. Do not adopt a mechanism claim for this gap without that
measurement; two plausible ones have already been wrong.

### Artifacts (2026-08-03, later)

- `logs/mujoco_baseline/sweep_{fixed,free}_dance1.json` -- MuJoCo sweeps.
- `logs/mjwarp_contract/newton.json` -- the MJWarp model contract.
- `logs/dynamics_controlled/newton_{bigcon,mjcontacts}.json` -- the two controls.

## 2026-08-03 (asset): the USD is correct; mass DR is a measurement trap

Checked because the tracking gap concentrates in the wrists, and a wrong link
mass there would explain it.

**The asset is clean.** `scripts/audit/audit_g1_link_mass_contract.py` derives
each link's expected mass from the URDF -- folding every fixed-joint child into
its nearest movable ancestor, since that is what the importer does -- and
compares it against the spawned articulation. With randomization off, **both
backends pass**: every link within 1e-3 kg, total 33.3411 kg, matching the URDF
exactly. The USD's own authored `physics:mass` values match too.

**It is also genuinely Unitree's.** `g1_29dof_rev_1_0.usd` and all three layers
under `configuration/` are byte-identical (sha256) to
`unitreerobotics/unitree_model`, `G1/29dof/usd/g1_29dof_rev_1_0`. Unitree
publishes USD there -- not in `unitree_mujoco` or `unitree_rl_gym`, which is why
an earlier pass through only those two repos wrongly concluded they ship none.

**The trap.** `G1SonicEventCfg.randomize_rigid_body_mass` scales
`.*wrist_yaw.*|torso_link` by `(0.8, 2.5)` at startup. Auditing the asset with
that event live reports:

| link | URDF | "PhysX" | "Newton" |
| --- | ---: | ---: | ---: |
| torso_link | 7.8170 | 8.4485 | 12.2379 |
| left_wrist_yaw_link | 0.2546 | 0.4731 | 0.5687 |
| right_wrist_yaw_link | 0.2546 | 0.5529 | 0.5984 |
| total | 33.3411 | 34.4895 | 38.4201 |

which reads as a 15% asset defect *and* a backend disagreement, and is neither:
it is one dice roll per link, drawn from different RNG positions on the two
backends. The three affected links are exactly the event's selector. Any asset
or backend comparison must disable randomization first; the audit script now
forces it rather than offering a flag.

The Newton-vs-PhysX dynamics runs recorded above were already run with
`randomization_kept: {startup: false, reset: false, push: false}`, so they are
unaffected -- but note that the DR range itself is wide (up to 2.5x on torso and
both wrists) and lands on the joints where Newton tracks worst.

### Our USD vs the official rev_1_0 MJCF: the same robot

Compared directly (`physics:mass` / `physics:diagonalInertia` /
`physics:lowerLimit` / `upperLimit` / `drive:angular:physics:maxForce` from the
USD against a compiled `g1_29dof_rev_1_0.xml`):

| quantity | result |
| --- | --- |
| link mass, all 30 bodies | **match**, total 33.3411 kg both |
| principal inertia, all 30 bodies | **match** |
| joint range, all 29 joints | **match** |
| joint effort limit, all 29 joints | **match** |

The only nonzero delta is `torso_link`: 7.818 kg in the MJCF against 7.8170 in
the USD, with 0.05% on inertia. The URDF sums to 7.817 (torso 6.78, head 1.036,
logo 0.001), so the USD is exact and the MJCF is rounded. One gram.

Two comparison traps to avoid when repeating this. MuJoCo's `body_inertia` is
the principal inertia in its own principal frame, and the axis **order** need
not match USD's `diagonalInertia` -- compare sorted, or 23 of 30 bodies look
wrong. And `actuatorfrcrange` in these MJCFs is authored on the *joint*
(`jnt_actfrcrange`), not on the actuator; reading `actuator_forcerange` returns
zeros and makes all 29 joints look mismatched.

Our vendored `g1_29dof_rev_1_0.xml` is also byte-identical (sha256
`165fa7a5...`) to `unitree_rl_gym`'s copy. So all three of our artifacts -- USD,
MJCF, URDF -- are the official rev_1_0 and agree with each other.

### rev_1_0 is a different robot from `g1_29dof`

`third_party/unitree_mujoco/g1/g1_29dof.xml` is the **non-rev_1_0** revision.
Diffing Unitree's own two files in one repo, 39 lines differ:

| | `g1_29dof` | `g1_29dof_rev_1_0` |
| --- | ---: | ---: |
| waist_yaw / waist_roll / torso meshes | `*.STL` | `*_rev_1_0.STL` |
| `waist_support_link` | present | **removed** |
| waist_yaw_link mass | 0.2440 | 0.2140 |
| waist_roll_link mass | 0.0470 | 0.0860 |
| torso_link mass | 9.5980 | 7.8180 |
| hip_roll `actuatorfrcrange` | ±88 | **±139** |

So they are not interchangeable: the waist assembly was redesigned and the hip
roll actuator was re-rated. Use the non-rev file for the passive parameters it
declares (`armature` / `damping` / `frictionloss`), never for its inertials or
effort limits.

### One real parameter mismatch: hip_pitch effort

Against the **rev_1_0 USD and MJCF, which agree on all 29 joints**, exactly one
config entry is out of spec:

| joint | asset (USD = MJCF) | SONIC cfg |
| --- | ---: | ---: |
| hip_pitch | 88 | **139** |

`UNITREE_G1_29DOF_SONIC_CFG` allows 58% more hip-pitch torque than the robot
has. Note the rev_1_0 URDF separately disagrees with both the USD and the MJCF
on ankle and waist (35 vs 50 N·m); the config follows the USD/MJCF there, which
is the defensible side. Correct hip_pitch on fidelity grounds independently of
the backend gap -- but it changes the action scale, so it invalidates existing
checkpoints.

### Cross-checked against TWIST2 (a deployed sim2real G1 system)

[TWIST2](https://github.com/amazon-far/TWIST2) teleoperates a real G1 and ships
the MuJoCo model its RL low-level controller is deployed against. It is
**rev_1_0** -- `waist_yaw_link_rev_1_0.STL` / `torso_link_rev_1_0.STL` meshes,
hip_roll `actuatorfrcrange="-139 139"`, no `waist_support_link`. Its
`g1_29dof_rev_1_0.xml` differs from ours in 24 lines, all of them IMU sensor
declarations and skybox/ground textures -- no physics.

Its `assets/g1/g1_sim2sim_29dof.xml` carries per-motor armature, and comparing
it to `UNITREE_G1_29DOF_SONIC_CFG`:

| joint | TWIST2 armature / effort | SONIC armature / effort | |
| --- | ---: | ---: | --- |
| hip_pitch | 0.0103 / **88** | 0.0251 / **139** | **disagrees** |
| hip_roll | 0.0251 / 139 | 0.0251 / 139 | agrees |
| hip_yaw | 0.0103 / 88 | 0.0102 / 88 | agrees |
| knee | 0.0251 / 139 | 0.0251 / 139 | agrees |
| shoulder / elbow | 0.003597 / 25 | 0.003610 / 25 | agrees |
| ankle_pitch / ankle_roll | 0.003597 / 50 | 0.007219 / 50 | armature 2x apart |
| waist_roll | 0.0103 / 50 | 0.007219 / 50 | armature differs |

**This settles hip_pitch.** TWIST2 puts it on the N7520-14.3 motor -- armature
0.0103, 88 N·m -- exactly like the rev_1_0 URDF, the rev_1_0 MJCF, the official
USD, and this repo's own base `UNITREE_G1_29DOF_CFG`. Five independent sources
say 88; only the SONIC override says 139, and it moved effort, stiffness,
damping *and* armature to the `*_7520_22` values together, so it is a coherent
mistake rather than a typo.

The ankle and waist_roll armature disagreements are open -- TWIST2 is one
system's tuning, not a vendor spec, so treat those as a prompt to check rather
than a correction. Do not take TWIST2's wrist effort limits at all: its own file
is left/right asymmetric there (`left_wrist_pitch` ±5 against
`right_wrist_pitch` ±25), which is a defect on their side.

TWIST2 also runs MuJoCo at `timestep=0.001, iterations=50, solver=PGS` against
this repo's `sim.dt=0.005` -- worth noting when reading any cross-system
dynamics comparison.

## 2026-08-03 (per-step): the actuator path is correct; it is contact

`scripts/audit/sim2sim_step_divergence.py` drives all three engines with an
identical, **state-independent** joint-position target sequence from an
identical initial state, and records the full state after every control step.
Reference point: **CPU MuJoCo + rev_1_0 + the SONIC actuator parameters** (the
2026-08-03 decision), with both Isaac backends measured against it.

Three design points, each of which the first run of the probe got wrong and the
built-in checks caught:

- The command is a joint-position **target**, not an action. Isaac applies
  `target = offset + scale * action`; MuJoCo's servo takes the target. Sending
  the same action sends different physical commands, so the probe specifies
  targets and inverts Isaac's affine map.
- The per-joint sinusoid's phase and rate are derived from the joint's rank in
  the **name-sorted** order, never its index in the engine's array. Index-derived
  parameters send a different command to the same joint on each backend.
- The reference pose is a declared constant, not each engine's default. Isaac's
  `default_joint_pos` for this task is all zeros while SONIC's `init_state` is a
  crouched stance -- centering on "the default" makes two different experiments.

A `targets_sha` computed over the name-sorted command array is recorded in every
trace and `--compare` refuses to proceed unless the two match. All three engines
now report `c6ee23c2...`.

Robot released at pelvis z = 1.30 m (feet ~0.5 m clear), 20 control steps:

| phase | steps | Newton vs MuJoCo | PhysX vs MuJoCo |
| --- | --- | ---: | ---: |
| free flight | 1-13 | **≤ 1.7e-6 rad** | ~1e-3 rad, growing |
| ground contact | 14-20 | 2e-2 → 1.3e-1 rad | 1e-1 → 2.9e-1 rad |

**In free flight Newton MJWarp reproduces CPU MuJoCo to float noise.** Root
height agrees to five decimals every step (1.29756 / 1.29756, ... 0.96134 /
0.96134). Articulation dynamics, the actuator PD, armature, effort saturation
and the integrator are therefore *positively verified correct* on the Newton
backend against an external reference -- not merely "no defect found".

PhysX differs from MuJoCo throughout free flight, but the signature is a
constant ~2-4 mm root offset that is already present at step 0 (1.29926 against
1.30000) rather than a growing dynamics error, and all three land at the same
height (0.7899 / 0.7898 / 0.7909). That is consistent with a sub-step phase
offset in when state is read back, not with a different gravity or mass.

Both backends diverge sharply at step 14-16, which is exactly when the feet
reach the floor (free fall of 0.5 m = 0.319 s = 16 control steps at 20 ms).

**So the residual Newton-vs-PhysX gap is in contact, not in the actuator or
integration path.** That contradicts the earlier `use_mujoco_contacts=True`
control, which reproduced every metric bit-identically -- and bit-identical
results across two genuinely different contact pipelines are themselves
implausible. Treat that control as unverified: it was never confirmed that the
override reached the solver (unlike the `nconmax` change, which was confirmed by
the overflow warning disappearing).

Next: repeat this probe from a **settled stance on the ground** rather than in
free flight, so the whole trace is contact-dominated, and re-test
`use_mujoco_contacts` with an assertion that the flag actually took effect.

## 2026-08-03 (final): ranked against CPU MuJoCo in both regimes

Both regimes, identical initial state, identical commands (enforced by
`targets_sha`). Max joint deviation from CPU MuJoCo:

| regime | Newton | PhysX | ratio |
| --- | ---: | ---: | ---: |
| free flight, step 5 | 4.10e-7 | 1.75e-3 | **4279x** |
| free flight, step 13 | 1.70e-6 | 2.30e-3 | 1354x |
| contact-loaded, settled (pos) | 6.68e-4 | 5.04e-3 | 7.5x |
| contact-loaded, settled (vel) | 1.54e-3 | 1.51e-1 | **98x** |
| contact-loaded, +5 steps | 9.25e-3 | 1.50e-2 | 1.6x |
| contact-loaded, +20 steps | 4.49e-2 | 3.64e-2 | 0.8x |

**Newton is closer to CPU MuJoCo in every regime except deep into contact
loading, where the two backends become comparable.** In free flight the margin
is ~1000-4000x, which is near-tautological (Newton *is* MuJoCo Warp) but does
positively confirm the MJWarp configuration is faithful. Under contact the
margin collapses to 7.5x and then to parity by 20 loaded steps.

So the divergence is contact-specific and **degrades with time under load** --
consistent with the `mjDSBL_MULTICCD` lead, since MuJoCo's docs describe that
flag as existing for flat surfaces where a single contact point causes "sliding
or wobbling".

Two corrections this round, both of which had produced a wrong conclusion:

- The earlier ground-tracking result that appeared to favor PhysX (`wrist_roll`
  0.039 against Newton's 0.550) is **invalid**. It used the state-dependent
  oracle action, so once the engines differed at all they received *different
  commands* (`action_abs_mean` 1.690 vs 1.032). With identical commands Newton
  is closer. Never rank engines with a closed-loop command.
- The first contact-loaded run reported CPU MuJoCo frozen at its initial state.
  That was a missing settle loop on the MuJoCo side only; the probe's step-0
  gate caught it. Both sides now settle, and all three agree on the settled
  stance (root height within 0.13 mm).

Residual caveat: the CPU MuJoCo reference still loads the **MJCF's** foot
geometry (8 explicit contact spheres) while both Isaac backends load the USD's
convex hulls, so the contact-loaded rows compare slightly different feet and
likely understate agreement.

### Newton solver config: what changed against the 5B runs

`G1ImitationPhysicsCfg.newton_mjwarp`, four fields:

| field | 5B runs | now |
| --- | ---: | ---: |
| `nconmax` | 10 | 200 |
| `njmax` | 95 | 288 |
| `use_mujoco_contacts` | False | True |
| `tolerance` | 1e-6 (default) | 1e-8 |

Everything else was already identical to stock MuJoCo and is unchanged.

**The measured behavioural effect is nil.** The ground oracle probe is
bit-identical before and after (joint MAE 0.0975, torque 7.15, `wrist_roll`
0.5500), and touchdown divergence is unchanged at 2.099e-2. Both changes
demonstrably reach the solver -- `nconmax` silences the overflow warning and
`use_mujoco_contacts` flips `mjw_model.opt.run_collision_detection` -- they
simply do not move these scenarios. The free-flight MuJoCo agreement was
measured with the **old** config and reproduces with the new one, so it was not
gained by this change.
