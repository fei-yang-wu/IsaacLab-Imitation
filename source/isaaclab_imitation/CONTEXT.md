# CONTEXT.md — `isaaclab_imitation` (environment extension)

Bounded context: the installable Isaac Lab extension that defines the
imitation environments, their command interface, and their data plane. Read
the repository root [`CONTEXT.md`](../../CONTEXT.md) first.

## Ubiquitous language

- **Reference channel** — the always-present, dataset-backed command channel.
  Rewards, terminations, MPJPE, and `reset_reference_state` are measured
  against it in every mode, including planner evaluation where it only
  scores. Only *selection* is pluggable (`ReferenceSelectionCfg`: which
  motion, which start frame).
- **Actor channel** — the single command the actor consumes. Exactly one of:
  - `ExplicitCommandCfg` — vanilla full-body command, source `reference`.
  - `LatentCommandCfg` — DiffSR / SONIC latent, source `agent`.
  - `ChunkCommandCfg` — planner packet, source `external`.
  The env config carries one of these and is the single authority on what
  the actor, the critic, and the encoder read. Agent configs consume the
  derivation (`actor_command_keys`); they never restate it.
- **Encoder view** — `EncoderViewCfg`, the windowed reference terms a latent
  recipe's encoder (posterior / prior) reads. It is a view, not a channel.
- **Contracts** (`contracts/`) — env-free schemas the live env and offline
  tooling must agree on byte-for-byte (causal planner observation, command
  channels, publisher, publish schedule). They import only torch, so
  contract tests run without a simulator.
- **Causal planner observation** — nine past frames plus current, 93 values
  per frame (`10 x 93`). The only deployable planner input. Never use
  `current_achieved_macro_transition_batch` as a planner input.
- **ExpertDataPlane** (`envs/expert_data_plane.py`) — the owned component of
  the v2 env that holds dataset load, reference caches, frame refresh,
  expert window and macro-transition sampling, and the MPJPE metric.
  Two-phase construction: `__init__` before managers, `finalize` after the
  scene exists.
- **v2 env** — `envs/imitation_rl_env_v2.py` (`ImitationRLEnv`), the current
  CommandManager-based environment. The **legacy env**
  (`envs/imitation_rl_env_legacy.py`) stays byte-frozen for v0/v1.
- **Command term** — a term managed by the Isaac Lab CommandManager
  (`tasks/.../mdp/commands/`); metrics live on command terms.
- **Transition EWMA** — the recent active environment-step MPJPE health signal,
  with reset-step samples excluded and a default 200-control-step (about 4 s
  at 50 Hz) time constant. It lags policy changes by about 4 s and briefly
  mixes pre-resume and post-resume behavior. It is a training health signal,
  not a fixed-protocol evaluation result.
- **Manifest** (`motion_manifest.py`, `manifests/`) — the declared list of
  NPZ motions. Dataset caches (Zarr) are content-specific: latent and
  vanilla recipes use separate cache paths; never rely on an environment
  default for paper jobs.

## Task versioning

- Task IDs are `Isaac-Imitation-G1-vN` in
  `tasks/manager_based/imitation/config/g1/`. "The default" is always the
  highest N.
- A breaking change to the stable recipe registers `vN+1`; the old `vN`
  keeps its exact kwargs forever and stops being cited as the default.
- Layout: `config/g1/common/` shared parts, `imitation_g1_env_v0/v1/v2.py`
  releases, `variants/` standalone one-offs. Old module paths are shims; a
  layout contract test is the gate.
- `Isaac-Imitation-G1-v2` (default since 2026-08-01, retuned in place
  2026-08-04): rewards `G1V2TunedRewardsCfg`, DiffSR macro state
  `root_qpos` (380). Older v2 checkpoints need their original overrides;
  a wrong encoder pairing must fail loudly. Invoke the
  `g1-encoder-interface` skill before changing or pairing an encoder.
- `Isaac-Imitation-Vega-Wuji-v0` is the internal dexterous-manipulation
  surface for the Vega U plus two Wuji hands. Its recipe derives from the
  released CHORD task (NVIDIA `video_to_data`); symbol names are repo-local
  because the implementation diverges independently. ILTools owns its robot, wrist, fingertip, rigid-object,
  support-surface, and padded contact Reference. The task uses Newton MJWarp
  and RLOpt PPO only.
- **Vega/Wuji Reference-residual action** — the 59-value actor output in live
  actuator order. Each value is bounded with `tanh`, scaled by joint type,
  filtered with an exponential moving average, added to the action-aligned
  Reference joint position, and clipped to the live soft joint limits.
- **Vega/Wuji transition state** — the actor receives live and action-aligned
  desired `[q, qdot]`, plus live and desired per-object
  `[XYZ, WXYZ, linear-XYZ, angular-XYZ]`. The default one-object policy vector
  is 609 values.
- **Vega/Wuji asset contract** — the vendored robot-only MJCF must expose 59
  actuators, the `R_ee`/`L_ee` parents, the `r_mount`/`l_mount` runtime wrist
  bodies, and no table or cube scene bodies. `right_palm` and `left_palm` must
  be identity sites on `r_mount` and `l_mount`. Validate it before Isaac
  startup.

## Invariants

- The actor consumes exactly one command source; enforced on derived actor
  input keys, not by splitting observation groups.
- Actor command terms and matching critic entries hold the same values;
  the critic may add privileged state. Command-side expert noise stays
  disabled.
- Quaternion convention: ILTools files and the dexmanip tensor math use
  WXYZ. Isaac
  Lab 3.0 live asset state uses XYZW. Convert only at this boundary.
- Planner command publication is per-environment renewal. Global timestep
  modulo logic is invalid with asynchronous resets.
- Vega/Wuji Reference joint names must contain the MJCF actuator joint set.
  The environment reorders Reference columns to the MJCF actuator order and
  checks the imported articulation for the same set. Newton can expose a
  different live order, so the environment then reorders the tensors once to
  that live order before reset or control.
- Vega/Wuji Reference wrist poses are the world poses of the named MuJoCo palm
  sites. They must declare `right_palm` and `left_palm`. The action, command,
  observation, reward, and termination code use the identity-equivalent
  `r_mount` and `l_mount` Newton bodies. They never use the rotated
  `R_ee`/`L_ee` parent frames as wrists.
- All motions in one Vega/Wuji manifest must use one fixed root, object list,
  object asset list, positive object radii, support scene, hand-frame names,
  and contact-link layout. Objects must be single rigid bodies.
- A Vega/Wuji JSON Manifest must bind its data to the runtime robot MJCF with
  `model_sha256`. Training additionally requires typed ILTools
  `TrainingQualification` and `ScenePhysics` on every motion. Direct NPZ and
  inspection-only inputs are state-lock replay inputs, never training inputs.
- Training qualification v2 hash-binds URDF collision meshes in addition to
  their wrapper file. Legacy dependency-unaware v1 qualification remains
  inspectable but is rejected for training. Self-contained ASCII USDA is
  accepted; uninspectable/external-reference USD fails closed without pxr.
- The task-specific Manifest writer compiles the MJCF and rejects missing joint
  limits, out-of-range/non-finite qpos, implausible qvel, or qvel inconsistent
  with the 20 Hz position trajectory. Collision evidence may declare at most
  1 mm penetration tolerance.
- Reset synchronizes the robot and all objects to one Reference frame. Object
  twists are world-frame root-origin twists in ILTools and are converted to
  Isaac COM velocity only at the simulator write boundary. Evaluation requires
  virtual object control to be exactly zero from frame zero.
- Robot-support contact force is an explicit negative reward and hard safety
  termination. Whole-robot and object pose/twist tracking provide the positive
  ReconBody-style objective alongside the hand/contact terms.
- Startup material events write Wuji hand friction and typed object/support
  dynamic friction plus restitution directly into Newton's live shape arrays.
  Full-path replicated-shape matching and readback are fail-closed; this is the
  authoritative path when USD material binding is blocked by instancing.
- Isaac Lab's Newton contact sensor does not publish contact points. The task
  reads Newton 1.2.1's public post-step contact buffer and aggregates exact
  object-side points and forces by object and hand link. The contact sensor
  stays active because it enables Newton's optional force buffer.
- The Vega/Wuji asset contract is checked by
  `scripts/data/validate_vega_wuji_asset.py` and again by the environment
  before Isaac startup; a failing audit is not a valid Isaac input.

## Validation

```bash
pixi run -e isaaclab test-isaaclab
pixi run -e isaaclab smoke-ipmd
```

Contract-only and config tests that avoid Isaac Sim imports can run in the
default environment.
