# Experiment library

imitation_experiments owns shared data, planner, lowlevel, evaluation, audit,
provenance, pipeline, capacity, and reporting code. Launch modules with
python -m imitation_experiments.<subpackage>.<module>. Keep imports light;
Isaac runtime imports belong inside entrypoints.

- Planner sample: causal history and explicit task input paired with labels.
- Sample budget: positive training rows, matched across the main comparison.
- Collection: policy interaction that writes samples. Oracle collection uses
  reference commands; planner collection requires an independent explicit goal.
- Complete-trajectory collection: whole trajectories, not a fixed row budget.
- Demonstration-only / rollout-finetuned: distinct planner training stages;
  report separately.
- Audit / preparation record: machine-readable evidence with input hashes and
  exact protocol.
- Aggregate: summary of complete audited cells with aggregation_manifest.json.
- Planner latency: synchronized planner forward time at publication, excluding
  warmup, simulator, tracker, and I/O.

Pair comparisons by goal within seed; do not count goals as independent seeds.
Bind encoder checkpoints before latent evaluation. Match backbone, budgets,
seeds, starts, and tracker protocol for the main comparison. Use paths.REPO_ROOT
and absolute package imports. Tests belong in source/imitation_experiments/tests/.
