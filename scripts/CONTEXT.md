# CONTEXT.md — `scripts/` (entrypoints)

Bounded context: command-line entrypoints. Scripts wire configs to the
library and the environments; they hold no shared implementation. Read the
repository root [`CONTEXT.md`](../CONTEXT.md) first.

## Layout language

- `rlopt/` — RLOpt train, test, playback, and closed-loop evaluation
  entrypoints (`train.py`, `play.py`, `eval_skill_commander_closed_loop.py`,
  `train_hl_skill_diffsr.py`, ...). `runtime_bootstrap.py` is the shared
  Isaac runtime bootstrap.
- `rsl_rl/`, `sb3/`, `skrl/` — alternative RL-framework entrypoints.
- `data/` — dataset preparation tools.
- `audit/` — standalone audit gates, e.g.
  `audit_g1_lafan1_body_frames.py` and `audit_bones_seed_phase5.py`.
- `viz/` — visualization and policy/reference comparison.
- `bench/` — benchmarks.
- `zero_agent.py`, `random_agent.py` — smoke-test runners.
- `list_envs.py` — prints registered task IDs.

## Rules

- Run everything from the repository root through Pixi:
  `pixi run -e isaaclab python scripts/rlopt/train.py ...` for Isaac
  workflows; plain `pixi run` for pure-Python tools.
- Preserve the existing Isaac Lab / Hydra CLI patterns. `--task` selects a
  registered `Isaac-Imitation-G1-vN` id; `env.` and `agent.` dotted
  overrides go through Hydra.
- Isaac entrypoints may re-assign `env_cfg` fields after Hydra parsing.
  Trust the run directory's recorded config (`summary.json`), not the
  launcher command line, when you audit a run.
- Whenever a script produces a video, print the video's absolute retained
  path to stdout.
- New shared logic goes to `source/imitation_experiments/` with a test; a
  script stays a thin wrapper.

## Validation

```bash
pixi run test-scripts
bash -n <script>.sh
```
