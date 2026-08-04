# scripts/

Standalone command-line tools, grouped by function. Run everything from the
repository root through Pixi (`pixi run python ...` or
`pixi run -e isaaclab python ...` for anything that boots Isaac Sim).

| Directory | Purpose |
| --- | --- |
| `data/` | Dataset preparation: CSV→NPZ conversion, BONES-SEED selection/packing/upload, LAFAN1 setup, manifest tools. |
| `audit/` | Data and cache audits that gate training runs (`audit_bones_seed_phase5.py`, `audit_g1_lafan1_body_frames.py`, ...). |
| `viz/` | Playback, rendering, and policy-vs-reference comparison tools. |
| `bench/` | Physics-backend, renderer, and MDP benchmarks plus dynamics diagnostics. |
| `rlopt/` | RLOpt train/eval/play entrypoints and the CU130 runtime bootstrap. |
| `rsl_rl/`, `sb3/`, `skrl/` | Alternative RL-framework train/play entrypoints. |

Top level keeps only workspace plumbing (`install_workspace.sh`,
`list_envs.py`) and the smoke-test agents (`zero_agent.py`,
`random_agent.py`).

Shared experiment *library* code does not belong here: put importable planner,
evaluation, audit, or provenance logic in `source/imitation_experiments/` with
a test, and call it from a thin script if a CLI is needed.
