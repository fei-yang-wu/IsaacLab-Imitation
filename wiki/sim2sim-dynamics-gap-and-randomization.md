# Residual Sim2Sim Dynamics Gap and Randomization Plan

Status as of 2026-07-21. Companion to
[Sim2Sim Backend Verification](sim2sim-backend-verification.md), which covers
the joint-ordering leak. **Read that page first**: the ordering bug accounted
for most of the Newton/PhysX gap, and nothing on this page is measurable until
that fix is in.

This page records what is left of the gap after the ordering fix, and proposes
how to close it.

## Why sim2sim is worth investing in

The goal is hardware. PhysX is not the real G1, but it is a genuinely
independent dynamics implementation of the same robot — a different solver, a
different contact model, and a different integrator on the same URDF, masses,
and actuator contract.

That makes it a **held-out test domain with a ground-truth reference**, which
is something hardware cannot give cheaply. Randomization ranges are normally
guessed and validated only by an expensive robot experiment. Here they can be
tuned against a measurable target:

> Train under Newton with randomization. Evaluate zero-shot under PhysX.
> Never train on PhysX.

Zero-shot PhysX survival then becomes an honest generalization metric and a
proxy for hardware transfer, available on every training run at the cost of one
evaluation. If a randomization scheme does not close the Newton-to-PhysX gap,
it is unlikely to survive contact with the real robot.

This only works if PhysX stays uncontaminated. Do not tune Newton's solver to
match PhysX, and do not add PhysX rollouts to training.

## What the residual gap actually is

Measured on `L1_strict/model_step_992870400.pt`, `walk1_subject1`, seed 0, 500
control steps, with the joint ordering matched so the comparison is honest:

| Backend | Survived | Fall | Joint MAE (first 67 steps) |
| --- | ---: | ---: | ---: |
| Newton (training domain) | 500/500 | never | 0.110 rad |
| PhysX (matched ordering) | 323/500 | 6.46 s | 0.240 rad |

So a controller trained to 993M frames on one solver, with today's
randomization, degrades from perfect tracking to a fall in 6.5 s on another
solver of the same robot. That is the size of the problem.

These are the pre-fix numbers. After the reset and solver fixes below the gap
is materially the same (Newton 0.126 rad and full survival, PhysX 0.242 rad and
a fall at 5.36 s), which is the point: those were correctness fixes, not gap
closers.

All numbers so far are **one motion, one seed**. Treat the fall step as
indicative, not precise, until the matrix is repeated across motions and seeds.

### Resolved: the reset-state discrepancy was a real bug, now fixed

An earlier version of this page reported a suspected reset-state fidelity
difference. Investigated on 2026-07-21: it was **a genuine bug in this repo,
affecting both backends**, and it is fixed. It was not a solver difference.

Part of the originally reported difference was measurement error — the
"deterministic" contract dump disabled only the joint-default randomization,
leaving the reference reset's pose and velocity randomization active, so the
two backends started from genuinely different states. With all reset
randomization disabled the difference collapsed from 320/930 values to 50/930,
and the remainder isolated cleanly:

| Quantity | Newton | PhysX |
| --- | --- | --- |
| `joint_pos_rel`, `joint_vel_rel`, `last_action` | identical | identical |
| `root_ang_vel_w` | `[-0.1056, 0.1919, 0.0008]` | `[-0.1056, 0.1919, 0.0008]` |
| `root_ang_vel_b` | `[0, 0, 0]` | `[0.633, 0.832, -0.432]` |

World frame was correct and identical on both. Body frame was wrong on both:
‖`root_ang_vel_w`‖ = 0.219 but ‖PhysX `root_ang_vel_b`‖ = 1.131, and a rotation
preserves magnitude, so the PhysX value was **stale**, not rotated. Newton
returned zeros because its buffer is lazily allocated and had never been
filled.

**Root cause.** Isaac Lab caches derived quantities such as `root_lin_vel_b`
and `root_ang_vel_b` in `TimestampedBuffer`s guarded by
`timestamp < _sim_timestamp`, and `update(dt)` only advances that timestamp by
`dt`. Both reset events in `mdp/events.py` called `asset.update(dt=0.0)` —
whose own docstring says `dt` "must be a positive value" — so the timestamp
never advanced, every derived buffer was still considered fresh, and the
body-frame values were served from the **pre-reset** state.

`base_lin_vel` and `base_ang_vel` are policy observations, so the first
observation after **every** reset was corrupted, on both backends, throughout
all training to date.

**Fix.** `_force_refresh_derived_state` in `mdp/events.py` sets the derived
buffer timestamps to `-1.0` after a reset write, the same idiom Isaac Lab uses
internally in its own `write_*_to_sim` paths. Advancing `_sim_timestamp`
instead would have driven the `joint_acc` finite difference and invented an
acceleration from the reset discontinuity. After the fix both backends report
identical body-frame velocities and magnitude is preserved, and the
deterministic contract dump reports no state differences at all.

Reproduce with:

```bash
pixi run -e isaaclab python scripts/dump_backend_index_contract.py --compare \
    logs/index_contract/fix_newton.json logs/index_contract/fix_physx.json
```

### The gap did not move

Both fixes above are correctness fixes; neither closed the transfer gap.
Re-measured with the reset fix and the USD solver settings in place, ordering
matched on both sides via the legacy shim:

| Backend | Survived | Fall | Joint MAE |
| --- | ---: | ---: | ---: |
| Newton | 500/500 | never | 0.126 rad |
| PhysX | 268/500 | 5.36 s | 0.242 rad |

Against the pre-fix numbers (Newton 500/500 / 0.125; PhysX 323/500 at 6.46 s /
0.239) the residual gap is essentially unchanged: still about **1.9x** the
joint tracking error and a fall in the 5 to 6.5 s range. Do not read the
268-vs-323 movement as a regression — the solver configuration changed
underneath it, and single-seed single-motion fall steps are not that precise.

The conclusion stands: the remaining gap is a genuine dynamics difference and
is what randomization has to address.

### Known configuration asymmetries

These are real differences in how the two backends are set up. Some are
deliberate, some look accidental.

| Item | PhysX | Newton / MJWarp |
| --- | --- | --- |
| Solver | TGS | `implicitfast`, `num_substeps=1` |
| Iterations | 8 position / 4 velocity | not applicable |
| Contact | rigid, convex hulls + primitives | soft, pyramidal cone, `impratio=1` |
| Capacity | n/a | `njmax=95`, `nconmax=10` |
| Self-collision | `enabledSelfCollisions=1`, honored | not consumed from the PhysX schema |
| Contact/rest offset | scene defaults (unauthored) | MJWarp margins |

Two of these looked like defects rather than choices:

1. **Resolved 2026-07-21.** The packaged USD requests
   `solverPositionIterationCount = 32` / `velocity = 1`, but
   `unitree_g1_29dof_usd_articulation_cfg` copied the URDF importer's
   `articulation_props` onto the USD spawn and overrode it to 8 / 4 — a generic
   Unitree converter default, not a choice made for the G1. That override is
   removed; the spawn now leaves `articulation_props` unset so the asset's own
   schema governs. Verified on the live stage: the spawned prim reports
   `solverPositionIterationCount = 32`, `solverVelocityIterationCount = 1`,
   `enabledSelfCollisions = True`. `rigid_props` is still applied, because the
   USD authors only mass, inertia, and centre of mass on the links. Newton does
   not consume the PhysX solver schema, so this changes the PhysX arm only.
2. **Still open.** The checked-in Newton preset uses `njmax=95, nconmax=10`
   (`imitation_g1_env_cfg.py:171`), below the **288 / 32** recorded in
   `wiki/current-status.md` as retained after the BONES-SEED NaN investigation.
   At 95/18 that investigation saw 951 constraint overflows in one rollout.
   Raise this before trusting Newton on contact-heavy motions.

## Current randomization

`G1EventCfg` (`imitation_g1_env_cfg.py`), all `startup` mode unless noted:

| Term | Range |
| --- | --- |
| `physics_material` | static friction 0.3–1.6, dynamic 0.3–1.2, restitution 0.0–0.5, 64 buckets |
| `add_joint_default_pos` | ±0.01 rad |
| `base_com` | `torso_link` only, x ±0.025, y/z ±0.05 m |
| `reset_reference_state` (reset) | pose x/y ±0.05, z ±0.01, rpy ±0.1/0.1/0.2; joints ±0.1 rad |
| `push_robot` (interval 1–3 s) | lin ±0.5/0.5/0.2, ang ±0.52/0.52/0.78 |

`G1SonicEventCfg` adds `randomize_rigid_body_mass`, scale 0.8–2.5, but **only
on `.*wrist_yaw.*|torso_link`**, and relaxes the push interval to 4–6 s.

### What is missing

Everything that governs how a joint *responds to a command* is deterministic:

- **No actuator gain randomization.** Stiffness and damping are fixed constants
  derived from `armature * ω²` with `ω = 10·2π` and `ζ = 2.0`. The policy has
  seen exactly one PD contract in 993M frames.
- **No armature randomization.** Armature is reflected rotor inertia — the
  parameter MJWarp and PhysX handle most differently, and one of the least
  accurately known on real hardware.
- **No leg or torso link-mass randomization.** The SONIC variant randomizes
  wrists and torso only; the leg links that dominate walking are fixed, and the
  base `G1EventCfg` randomizes no masses at all.
- **No actuation latency.** Sim applies the command in the same control step.
  The real G1 has roughly 10–20 ms of comms and driver delay. For a high-gain
  whole-body tracker this is usually the single largest sim2real term.
- **No effort-limit, joint-friction, or passive-damping randomization.**
- **No control-rate jitter.**
- **Flat ground only.** Friction is randomized; geometry is not.

## Proposed randomization scheme

Tiered by expected value per unit of implementation and compute risk. Ranges
are starting points to be tuned against the zero-shot PhysX metric, not
final values.

### Tier 0 — parity first (not randomization)

Do not randomize over a bug.

1. ~~Resolve the reset-state discrepancy.~~ **Done 2026-07-21** — it was a
   stale derived-buffer bug in this repo, corrupting the first observation
   after every reset on both backends. See above.
2. ~~Take the PhysX solver iteration count from the asset.~~ **Done
   2026-07-21** — now 32/1 as the USD requests, verified on the live stage.
3. **Open:** raise Newton to `njmax=288, nconmax=32`.
4. **Open:** re-measure across several motions and seeds, not just
   `walk1_subject1` at seed 0.

Items 1 and 2 did not move the transfer gap; they were correctness debts worth
paying before attributing anything to the solver.

### Tier 1 — actuation realism (highest value)

Sampled per environment at `startup`, so each env is a fixed but different
robot; the policy must be robust across envs rather than reactive within an
episode.

| Parameter | Proposed range | Rationale |
| --- | --- | --- |
| Joint stiffness | ×[0.75, 1.3], log-uniform, per actuator group | The policy currently knows one PD law exactly. |
| Joint damping | ×[0.7, 1.4], log-uniform, sampled independently of stiffness | Decoupling stiffness and damping stops it from learning a fixed ζ. |
| Armature | ×[0.5, 2.0], log-uniform | Poorly known; the main integrator-sensitivity term. |
| Link mass, all bodies | ×[0.9, 1.1] | Extends SONIC's wrist/torso-only scaling to the legs. |
| COM offset, all major links | ±0.02 m | Currently torso only. |
| Actuation latency | 0–2 control steps (0–40 ms), resampled per episode | The most common sim2real failure and currently exactly zero. |

Use log-uniform for multiplicative gains so ×0.75 and ×1.33 are equally likely.

### Tier 2 — command and sensing path

| Parameter | Proposed range |
| --- | --- |
| Effort limit | ×[0.8, 1.0] (never above nominal) |
| Passive joint friction | 0–0.05 N·m |
| Passive joint damping | ×[1.0, 1.5] |
| Control-rate jitter | ±1 sim step on the decimation boundary |
| Proprioception noise | joint pos ±0.005 rad, joint vel ±0.05 rad/s |
| IMU bias | gravity vector ±0.02, resampled per episode |

Verify what `observations.policy.enable_corruption` already injects before
adding noise terms, so they are not double-counted.

### Tier 3 — contact, aimed at the measured asymmetry

The two backends differ most in contact, so this is where sim2sim transfer
should be most informative.

| Parameter | Proposed range |
| --- | --- |
| Foot friction | widen to 0.2–1.8 static |
| Restitution | keep 0.0–0.5 |
| Contact/rest offset | ±0.005 m (PhysX side) |
| Ground slope | ±3° |
| Ground roughness | 0–0.02 m |

### Protocol

- **Sample per environment, not per step**, for anything physical. Per-step
  resampling teaches noise rejection, not robustness to a different robot.
- **Keep one nominal env config** with randomization disabled for
  comparability against historical numbers.
- **Start narrow and widen**, gated on Newton performance not regressing. If
  nominal-Newton tracking degrades materially, the ranges are too wide for the
  current model capacity. Automatic domain randomization is a later option;
  do not start there.
- **Report both** nominal-Newton and zero-shot-PhysX for every run. The gap
  between them is the number this work is trying to reduce.

## MPJPE is measured under perturbation

`eval_skill_commander_closed_loop.py` never touched randomization -- it only
disabled terminations -- so every MPJPE reported to date was measured with the
full training perturbation live: reset pose +/-0.1/0.1/0.2 rad rpy plus
+/-0.1 rad joint noise, interval pushes every 1-3 s, and mass, friction,
restitution and COM randomization.

Root-relative MPJPE subtracts root *position* but not root *orientation*, so a
perturbed root rigidly rotates every body inside the root-relative frame and
each body picks up roughly its distance from the root times the rotation.
Measured at reset on the G1's 14-body set, before any physics step:

| Config | MPJPE at reset |
| --- | ---: |
| Shipped evaluation config | 45.22 mm |
| Joint noise removed | 38.79 mm |
| All reset randomization removed | 0.00 mm |

The rigid-rotation signature is unambiguous: the pelvis, being the root body,
reads exactly 0.00 mm, and the correlation between a body's radius from the
root and its error is 0.974. Translation cancels by construction, so
essentially all of the 38.79 mm is the orientation perturbation.

The exact zero is a useful negative result in its own right: there is no
systematic reference-versus-URDF body-frame offset sitting underneath every
MPJPE this project reports.

### Which protocol to use

Two different numbers, do not conflate them:

- **Paired interface comparison.** Keep the perturbations. Both rows see the
  same ones, and `AGENTS.md` requires the push event be identical across
  interfaces and not altered solely for evaluation. Absolute values are
  inflated; the comparison is unaffected.
- **Absolute tracking claims, and any external comparison.** Use
  `--deterministic_tracking`, which starts exactly on the reference and removes
  pushes and randomization. Our perturbed numbers are **not** comparable to
  SONIC's, which reports root-relative MPJPE in mm with domain randomization
  listed under training only.

The flag is additive, not a replacement for either existing pass, matching the
precedent that `AGENTS.md` already sets by requiring an extra full-horizon
diagnostic pass. It records what it changed in the result file, and prefixes
its metrics with `deterministic_tracking/` so the paper aggregators -- which
look up bare names such as `tracking_mpjpe_mm` -- fail loudly rather than
silently pooling two protocols.

One definitional caveat for any external comparison: SONIC's paper does not
state whether it aligns root orientation before computing MPJPE. Ours does not.
If theirs does, the two metrics differ in sensitivity even under identical
randomization.

## Interaction with the frozen paper protocol

**Decision (2026-07-21, user): randomization applies to every experiment.**

`AGENTS.md` freezes the environment protocol for the causal-interface
comparison and requires the interval-push event to be identical for both
interface rows. Randomization changes that protocol, so the protocol is
**re-frozen on the randomized event config** rather than randomizing a subset
of runs. Consequences, all of which must hold:

- Both main planner rows, both low-level controllers, the direct vanilla
  ceiling, and every scaling-study arm train and evaluate under the **identical
  randomized event config**. No arm keeps the old config.
- Existing qualification artifacts, equivalence certificates, and oracle audits
  were produced under the old protocol and **do not carry over**. They must be
  regenerated once the randomized protocol is fixed.
- Randomization ranges are frozen at the same time as the protocol. Tuning
  ranges between arms would silently make the comparison unfair, which is
  exactly what this decision exists to prevent.
- The per-environment sampling is seeded per run, so two interfaces at the same
  seed see the same sequence of sampled robots. Verify this explicitly rather
  than assuming it; it is the property that makes the comparison paired.
- **Never randomize the evaluation domain differently between rows.** Report
  nominal-Newton and zero-shot-PhysX for every arm.

Because this invalidates the existing artifacts anyway, sequence it with the
retraining already forced by the joint-order fix rather than as a second
retraining wave.

## Open questions

- Does the residual gap shrink with training length alone, or is it structural?
  Worth one control: evaluate an earlier checkpoint of the same run under PhysX
  and see whether the gap is widening with training, which would indicate the
  policy is progressively overfitting to MJWarp contact.
- Is the gap concentrated in contact-rich motions? All measurements so far are
  `walk1_subject1`. Repeat across `jumps1`, `fallAndGetUp1`, and `dance1`
  before concluding it is a general contact-model issue.
- How much of the gap is the reset-state discrepancy alone? Measurable by
  starting both backends from a settled state instead of a reference write.
