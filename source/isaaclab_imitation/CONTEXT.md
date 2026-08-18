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

## Invariants

- The actor consumes exactly one command source; enforced on derived actor
  input keys, not by splitting observation groups.
- Actor command terms and matching critic entries hold the same values;
  the critic may add privileged state. Command-side expert noise stays
  disabled.
- Quaternion convention at the Isaac Lab 3.0 boundary: WXYZ inside Isaac
  Lab, XYZW at external boundaries; convert at the edge only.
- Planner command publication is per-environment renewal. Global timestep
  modulo logic is invalid with asynchronous resets.

## Validation

```bash
pixi run -e isaaclab test-isaaclab
pixi run -e isaaclab smoke-ipmd
```

Contract-only and config tests that avoid Isaac Sim imports can run in the
default environment.
