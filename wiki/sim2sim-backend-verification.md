# Sim2Sim Backend Verification: Newton vs PhysX

Status as of 2026-07-21. This page records why Newton-trained G1 low-level
checkpoints collapse under the PhysX/Isaac Sim backend, and the evidence that
the cause is a joint-ordering leak rather than a solver-realism gap.

Read this before running any cross-backend evaluation or interpreting a
Newton-vs-PhysX comparison in `wiki/isaaclab3-cu130-runtime-migration.md`.

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

`scripts/dump_backend_index_contract.py` dumps every resolved joint/body index,
the action offset/scale, and all observation term widths, then diffs two
backends and classifies each term as a correct remap, a wildcard selection, or
a live-order leak.

```bash
pixi run -e isaaclab python scripts/dump_backend_index_contract.py \
    --task Isaac-Imitation-G1-Latent-Strict-v0 --num_envs 2 --headless \
    --output logs/index_contract/newton.json physics=newton_mjwarp \
    env.lafan1_manifest_path=data/lafan1/manifests/g1_lafan1_walk1_subject1_manifest.json

OMNI_KIT_ACCEPT_EULA=YES pixi run -e isaaclab python scripts/dump_backend_index_contract.py \
    --task Isaac-Imitation-G1-Latent-Strict-v0 --num_envs 2 --headless \
    --output logs/index_contract/physx.json physics=physx \
    env.lafan1_manifest_path=data/lafan1/manifests/g1_lafan1_walk1_subject1_manifest.json

pixi run -e isaaclab python scripts/dump_backend_index_contract.py --compare \
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
4. `scripts/batch_csv_to_npz.py` applied the SDK-to-articulation scatter
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
| Source LAFAN1 / BONES-SEED NPZ and Zarr | **Safe.** `csv_to_npz.py` and `batch_csv_to_npz.py` both store `joint_names`/`body_names`, and every consumer remaps by name via `_map_reference_to_target`. |
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
- `source/isaaclab_imitation/test_g1_backend_joint_contract.py` asserts
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
