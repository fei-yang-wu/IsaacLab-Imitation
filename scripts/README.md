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

## Sharpa floating-hand data tools

The Sharpa video-to-data port keeps its dataset tools in `data/` and its
parity and runtime gates in `audit/`:

| Script | Purpose |
| --- | --- |
| `data/convert_mano_sharpa_to_iltools.py` | Convert `ManoSharpaData` Parquet to ILTools v2 references; `--retarget` runs the Pink MANO-to-Sharpa solver (100 iterations per frame, the source default). Pass `--mano-to-robot-scale` (1.0 for arctic and synthbox, 1.2 for the other released datasets) and `--motion-speed 0.5` for the source-parity task. `--settle-contacts --object-mesh <obj>` presses the fingers onto the object in MuJoCo (DexMachina style, wrists held); a deliberate departure from the release, not used for the ARCTIC campaign. |
| `data/run_source_sharpa_retarget.py` | Run the upstream retarget script natively from an adjacent `video_to_data` checkout, with the visualization modules stubbed. Produces the reference the parity gate compares against. |
| `data/make_rigid_object_urdf.py` | Write a single-link rigid URDF for one object mesh, the documented rigid proxy for an articulated source object. |
| `data/inspect_sharpa_reference.py` | Hash-verify and summarize a Sharpa manifest; warns when a reference's Sharpa solution is a `placeholder` or has no IK record. |
| `data/make_sharpa_rigid_smoke_fixture.py` | Build a small rigid Sharpa reference for import and simulation smoke tests. |
| `data/convert_vega_wuji_npz_to_sharpa.py` | Adapt a fixed-base Vega-Wuji reference to the dual floating-hand Sharpa layout. |
| `bench/measure_task_memory.py` | Peak GPU and host memory of one training command, attributed to its process tree, for comparing task capacity at equal environment counts. |
| `audit/compare_sharpa_references.py` | Frame-level parity gate between two Sharpa references (wrist pose, finger joints, object pose, contacts) with threshold exit codes. |
| `audit/check_sharpa_backend_smoke.py` | One-minute local gate that the Sharpa task can run on a chosen physics backend: resets, steps with zero actions, and fails on a NaN reward, an excessive reset contact force, a termination while tracking the Reference, or one-step episodes. Run it before any cluster submission. |
| `audit/measure_sharpa_reference_penetration.py` | Rebuild a Sharpa reference in MuJoCo and report how deep the hands sit inside the object, per frame, plus frames that claim contact while the geometry is apart. Collision only, no Isaac. |
| `data/convert_sharpa_assets_to_usd.py` | Pre-convert the Sharpa hand URDFs and one rigid object URDF to USD, reusing the task's own spawn settings, so training runs kit-less under Newton. Forces `convexDecomposition` on object colliders (the release's collider; the importer's hull fills a container's cavity) and can author Newton per-shape contact stiffness with `--contact-stiffness KE KD` (off by default). |
| `viz/replay_sharpa_reference.py` | Replay a Sharpa reference with zero residual actions in Isaac Lab. |
| `viz/render_sharpa_retarget_comparison.py` | MuJoCo side-by-side video: source MANO skeleton and object (left) versus the retargeted Sharpa hands and object (right), synchronized by source time; needs EGL (`MUJOCO_GL=egl`, NVIDIA vendor file). |

## Mounted Vega U + Sharpa evaluation

`rlopt/evaluate_vega_sharpa.py` evaluates frozen PPO checkpoints and zero
residuals on identical starts with explicit object-assistance levels. It
records first-episode metrics before reset and saves Newton trajectories for
visual inspection. Use `--protocol object-task` for fixed-horizon final-box
success scoring, and `--reference-overlay` in the replay renderer to compare
the actual box with its Reference. The curve exporter is
`python -m imitation_experiments.audit.vega_sharpa_training`.
See [the local PPO protocol and results](../wiki/vega-sharpa-local-ppo.md).
