# CONTEXT.md — `imitation_experiments` (experiment library)

Bounded context: the installable, tested experiment library. All shared
planner, evaluation, audit, data, and provenance Python lives here — never in
a campaign directory. Launchers invoke it with
`python -m imitation_experiments.<subpackage>.<module>`. Read the repository
root [`CONTEXT.md`](../../CONTEXT.md) first.

## Subpackages

- `data` — planner sample schemas and dataset preparation.
- `planner` — planner model definitions and training (flow-matching
  Transformer, chunked Transformer variants).
- `lowlevel` — low-level checkpoint evaluation and qualification.
- `evaluation` — closed-loop and M3 evaluation.
- `audit` — gate scripts that verify data, checkpoints, and protocol
  bindings; they fail loudly and write records.
- `provenance` — hash and binding records for datasets, checkpoints, and
  aggregates.
- `pipeline` — multi-stage orchestration (prepare, rollout, finetune,
  eval, summarize).
- `capacity` — the planner-scaling study.
- `paths.py` — `REPO_ROOT` and path helpers. Never mutate `sys.path` and
  never count `parents[N]` to find the repo root.

## Ubiquitous language

- **Planner sample** — one training row for the planner: keyed causal
  planner state plus targets (oracle latents or explicit packet labels).
- **Sample budget** — the exact positive sample count a training stage
  must use; identical across the two comparison rows.
- **Demonstration-only vs rollout-finetuned** — planner trained on oracle
  collections only, versus additionally fine-tuned on planner-driven
  rollouts. Report both, never merged.
- **Collection** — running a policy in the env to write planner samples.
  Oracle collection uses the frozen tracker with fresh reference commands;
  planner-driven collection uses the planner under an explicit goal.
- **Complete-trajectory collection** — the 2026-08-05 selected-ten unit:
  one full oracle trajectory per env, not a row budget.
- **Explicit goal** — the language/motion selection passed at deployment.
  Never derive it from trajectory rank, expert history, or the reference
  cursor. A goal/reference mismatch must fail immediately.
- **Audit record / preparation record** — the JSON a gate writes: input and
  output hashes, exact commands, dataset paths. Later gates require it.
- **Binding** — proof that a skill checkpoint's `skill_encoder_state_dict`
  is tensor-identical to the encoder inside a latent low-level checkpoint
  (`validate_latent_skill_checkpoint_binding.py`).
- **Planner latency** — wall time around the planner's root forward call at
  command publication only: CUDA-synchronized, one warmup excluded, no sim
  stepping, no low-level policy, no I/O.
- **Aggregate** — a paper-facing summary built only from complete, audited
  grids; aggregation scripts refuse overwrites and write
  `aggregation_manifest.json`.

## Invariants

- New shared experiment Python lands here with a test in
  `source/imitation_experiments/tests/`.
- Both comparison rows use the same planner backbone, stages, sample
  budget, optimizer budget, seed, evaluation starts, and low-level
  protocol.
- Paired results are reported latent-minus-explicit by goal within seed;
  goal rows are never pooled as independent seeds.
- Keep the package import-light: Isaac Sim imports belong inside entrypoint
  functions, never at module scope.

## Validation

```bash
pixi run test-experiments
```
