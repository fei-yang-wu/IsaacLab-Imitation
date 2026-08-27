# hold5-curve-eval

Score the budget axis of the two hold-5 arms on ICE.

## Why this campaign exists

The interface design study measures budget as an axis: a checkpoint every 250M
frames, scored on `bones_milestone_testbed256_v1`, so a design conclusion that
only holds at one end of the curve is reported as budget-dependent. The 29 arms
of the frozen star are scored from the local mirror, because their ICE trees
were cleaned under the 300 GB quota. The hold-5 pair added on 2026-08-26 is
different: its checkpoints are on ICE, so scoring them there moves no bytes.

## What it runs

One Slurm job per arm. `scripts/rlopt/eval_checkpoint_tree.py` keeps a single
Isaac Sim start and swaps the policy weights across the eight milestones,
instead of paying a simulation start per checkpoint.

Measured locally on `hold1_seed1`, eight cells on the 256-clip board:

| path | per cell | eight cells |
|---|---|---|
| one process per checkpoint | 47.4 s | 6 min 19 s |
| `eval_checkpoint_tree.py` | 30.9 s | 4 min 07 s |

The rows agree within Isaac's own nondeterminism: mean success-rate difference
+0.0010 (largest 0.0156), MPJPE-L +0.06% mean (largest 1.3%), MPJPE-G -1.30%
mean (largest 6.7%). The differences are scattered in sign, so they are
run-to-run noise, not an offset. Isaac evaluation is not deterministic; a
tree-scored row is not expected to be bit-identical to a single-cell one.

`--trajectory_ranks` pins env i to a pure function of env id, so every cell
resets onto the same clips. The evaluator asserts that across cells and stops
if a later cell lands on a different population.

## Order of operations

The training chain must finish first. `preflight` requires the tracker tree and
the encoder file, so planning early fails loudly instead of queueing a job that
would find nothing.

```bash
pixi run python -m imitation_experiments.pipeline.cluster plan \
    --campaign experiments/campaigns/2026-08-27-hold5-curve-eval/campaign.yaml \
    --arm use_hold5 --seed 0
```

Then submit with the printed `PLAN_SHA`, and repeat for `ix_fsq64_hold5` --
**after the first job finishes**. On 2026-08-27 the two arms were submitted
eleven minutes apart and the second crashed inside Kit startup while the first
was running; re-run alone it passed. Both read the same
`CLUSTER_ISAAC_SIM_CACHE_DIR`. Until that is understood, serialize the jobs or
give each its own cache directory.

Measured on ICE: about 16 minutes per arm for eight cells, most of it Kit
startup and the resident reference-array load, which one job pays once.

## Reading the result

Rows land in `/data/eval/hold5_curve` as
`<arm>_seed0_milestone_f<frames>.json`. Pull them next to the local eval
directories and fold them into the curve table:

```bash
pixi run python -m imitation_experiments.reporting.curve_table \
    logs/interface_design_study_eval logs/hold5_curve_eval \
    --row milestone --csv logs/report/milestone_curve.csv
```

The hold axis then reads 10 / 5 / 1 at both code widths: `ctrl`, `use_hold5`,
`use_hold1` at 256-D, and `bn_sonic_fsq64`, `ix_fsq64_hold5`, `ix_fsq64_hold1`
at 64-D.

## Arms

| arm | code width | hold | its control |
|---|---|---|---|
| `use_hold5` | 256-D continuous | 5 | `ctrl` (hold 10), `use_hold1` (hold 1) |
| `ix_fsq64_hold5` | 64-D SONIC FSQ | 5 | `bn_sonic_fsq64` (hold 10), `ix_fsq64_hold1` (hold 1) |
