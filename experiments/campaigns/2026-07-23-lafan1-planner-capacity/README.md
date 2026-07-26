# LAFAN1 one-motion planner-capacity study (2026-07-23)

**Question.** Does the DiffSR **latent** high-level interface reduce planner-training
complexity relative to explicit **full-body-chunk** and **EE-chunk** interfaces,
on a single LAFAN1 motion (`walk1_subject1`)?

Two readouts (Study 1 of `wiki/ablation-experiment-plan.md`, restricted to one motion):

1. **Iso-performance** — smallest planner parameter count per interface that reaches a
   fixed target (survival = 1 AND oracle-normalized MPJPE ≤ threshold).
2. **Iso-parameter** — closed-loop MPJPE / survival at matched planner parameter counts.

## Fixed protocol

- Planner family: **flow matching** (fixed; no family sweep here).
- Sizes: `tiny / small / medium / large` (`MODEL_PRESETS` in
  `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/train_chunked_transformer_planner.py`).
- Planner seeds: `0 1 2`. Low-level trackers stay seed 0.
- Planner input: causal `10 × 93` achieved-robot history + task index (one motion → trivial).
- Publication: 5 Hz, held 10 steps (`planner_update_interval=10`), per-env renewal.
- Eval: start at reference frame 0, ~700 control steps; M3 survival (`base_too_low`
  only) + full-horizon no-termination MPJPE pass. Demonstration-only and
  rollout-finetuned reported **separately**.
- Metrics normalized by each interface's own converged frozen oracle
  (`converged + oracle-normalized MPJPE`, user decision 2026-07-23).

## Interfaces & frozen oracles

See `paths.env`. Converged seed-0 checkpoints pulled from ICE to
`logs/downloaded_checkpoints/`:

| Interface | Command @ 5 Hz | Oracle MPJPE / ep_len |
| --- | --- | --- |
| `latent_skill` (DiffSR deterministic) | 258-d held z + phase | 45.6 mm / 393 |
| `full_body_trajectory` (FB chunk) | full-body packet, held 10 | 34.1 mm / 454 |
| `ee_trajectory` (EE chunk) | EE packet, held 10 | 41.3 mm / 424 |

The original "main" latent LAFAN1 tracker (job `5525664`) was destroyed in the
2026-07-22 Slurm-TIMEOUT data-loss incident; the surviving latent-learning-ablation
`deterministic` arm is the protocol-matched substitute.

## Pipeline

1. `run_capacity_point.sh` (per size × seed): oracle demos → planner pretrain (demo-only)
   → eval → rollout collect → merge → finetune → eval, for all three interfaces.
2. `aggregate_one_motion_capacity_scaling.py` (3-interface) → per-size table + iso-perf minimums.
3. `aggregate_one_motion_capacity_seeds.py` (3-interface) → across-seed means + latent-minus-explicit pairs.

Oracle baselines (frame-0/700, terminations disabled) are generated once per interface
and reused across all size/seed cells.

## Run

```bash
# dry-run everything (prints every command, touches nothing)
DRY_RUN=1 experiments/campaigns/2026-07-23-lafan1-planner-capacity/run_sweep.sh

# shared oracle baselines only (6 Isaac runs, once)
experiments/campaigns/2026-07-23-lafan1-planner-capacity/prepare_oracle_baselines.sh

# one validation cell end-to-end (3 interfaces)
MODEL_SIZE=small PLANNER_SEED=1 \
  experiments/campaigns/2026-07-23-lafan1-planner-capacity/run_capacity_point.sh

# full sweep + aggregation -> STUDY_ROOT/capacity_seeds_summary
experiments/campaigns/2026-07-23-lafan1-planner-capacity/run_sweep.sh
```

Scripts: `paths.env` (resolved inputs), `prepare_oracle_baselines.sh`,
`run_capacity_point.sh` (one size×seed), `run_sweep.sh` (sizes×seeds + aggregate).
Aggregators: `experiments/campaigns/2026-07-23-lafan1-planner-capacity/interface_baselines/aggregate_one_motion_capacity_scaling.py`
(per seed, 3-interface) and `aggregate_one_motion_capacity_seeds.py` (across seeds).

## Run on ICE (PACE)

Runs inside the CU130 Newton runtime container via `run_capacity_entry.py`
(pixi is unavailable in-container → `ISAAC_PY=/isaac-sim/python.sh`). Profile
`docker/cluster/.env.ice_capacity`. Data = corrected 40-motion tree under `/data`
(`walk1_subject1` restricted via `--motion_name`); checkpoints staged into
`<ISAACLAB>/logs/downloaded_checkpoints/`. Newton solver args + `--assert-kitless`
are injected automatically (h100, compute-only). Three afterok-chained stages:

```bash
# 1) oracle baselines (single job, ~1h) — validates runtime+data+checkpoints
./docker/cluster/cluster_interface.sh -c ice_capacity job --stage oracle
#    -> scrape "Submitted batch job <ORACLE_ID>"

# 2) 12-task array (one per size x seed), depends on oracle
CLUSTER_SLURM_ARRAY=0-11 CLUSTER_SLURM_DEPENDENCY=afterok:<ORACLE_ID> \
  ./docker/cluster/cluster_interface.sh -c ice_capacity job --stage cell
#    -> scrape "Submitted batch job <ARRAY_ID>"

# 3) aggregation, depends on the whole array
CLUSTER_SLURM_DEPENDENCY=afterok:<ARRAY_ID> \
  ./docker/cluster/cluster_interface.sh -c ice_capacity job --stage aggregate
```

Array index → cell: `size = (tiny,small,medium,large)[idx%4]`,
`seed = (0,1,2)[idx//4]`. Outputs persist under
`<ISAACLAB>/logs/interface_baselines/lafan1_planner_capacity_20260723/` (logs bind),
shared across all stages. Each task fits the 16 h ICE cap.

**Not a paper result** — one motion. It is the pilot for the multi-motion Study 1 grid.
