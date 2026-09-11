# Environment extension

isaaclab_imitation owns environments, command terms, robot/task configs,
reference data access, and shared environment contracts.

- Reference channel: dataset-backed rewards, terminations, resets, and metrics.
- Actor channel: exactly one command source. ExplicitCommandCfg reads reference
  commands; LatentCommandCfg reads agent latents; ChunkCommandCfg reads packets.
- Encoder view: windowed terms read by the skill encoder, not another channel.
- ExpertDataPlane: dataset load, reference caches, frame refresh, window
  sampling, and MPJPE. Construct first, finalize after scene creation.
- Contracts: env-free shared schemas; import torch without starting Isaac.
- Command term: Isaac CommandManager term; metrics live on these terms.
- Transition EWMA: moving training MPJPE health signal, excluding reset samples;
  not a fixed-protocol evaluation result.

The current environment implementation is imitation_rl_env_v2.py; the legacy
environment stays frozen for v0/v1. Task recipe versions are separate from the
environment implementation. Inspect config/g1/__init__.py for registered vN
recipes; breaking changes add a version and preserve prior kwargs.

Actor and matching critic command terms agree; the critic may add privileged
state. Keep command-side expert noise disabled. Convert quaternion order at
boundaries: Isaac Lab WXYZ, external XYZW. Publish planner commands per
environment, not by a global modulo.

Dataset caches are recipe-specific. Bind checkpoints to the exact encoder
view, anchor, cadence, and weights; old v2 checkpoints may require their
original full-body input and reward overrides. Keep the batched final_obs,
device-tensor logging, and prefetch semantics documented in AGENTS.md.
