# imitation_experiments

The shared experiment library for the causal-interface study: everything the
campaign and paper launchers execute, packaged so it has one import root, one
test suite, and no `sys.path` tricks.

Installed editable in both Pixi environments; the CU130 container runtime
exposes it via `scripts/rlopt/runtime_python.sh`. Invoke entrypoints as
modules from the repository root:

```bash
pixi run python -m imitation_experiments.planner.train_chunked_transformer_planner --help
pixi run -e isaaclab python -m imitation_experiments.evaluation.eval_interface_planner_closed_loop --help
```

| Subpackage | Contents |
| --- | --- |
| `data` | Sample schema, balanced row budgeting, manifest writers, rollout-sample collection, merging. |
| `planner` | Planner training, command publication, publish scheduling, latency measurement. |
| `lowlevel` | Frozen tracker loading, checkpoint resolution and evaluation. |
| `evaluation` | Closed-loop/offline evaluation, metrics, comparison summaries. |
| `audit` | Qualification gates, artifact audits, submission validators (fail loudly by design). |
| `provenance` | Frozen paper-protocol metadata and run-provenance records. |
| `pipeline` | Multi-stage workflow orchestrators invoked by campaign/paper launchers. |
| `capacity` | Secondary planner-capacity scaling study. |

Rules:

- Keep module scope import-light: Isaac Sim imports belong inside entrypoint
  functions, never at module top level, so the default environment can import
  and test everything.
- Every module change ships with a test in `tests/`
  (`pixi run test-experiments`).
- Resolve repository paths through `imitation_experiments.paths`
  (`REPO_ROOT`, `PAPER_DIR`, `module_source_rel`), never `parents[N]`.
- Campaign shell launchers and `experiments/paper/` entrypoints call this
  package; the package never calls back into `experiments/`.
