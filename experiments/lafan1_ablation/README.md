# LAFAN1 Ablation

Small launcher for the LAFAN1 horizon ablation.

It runs four interfaces:

| ID | Meaning |
|----|---------|
| `latent_cont` | DiffSR continuous latent + latent LL |
| `latent_fsq` | Online FSQ latent + latent LL |
| `ee_chunk` | End-effector command chunk + vanilla LL |
| `wb_chunk` | Whole-body command chunk + vanilla LL |

The table is:

```text
W in {10, 5, 1} x {latent_cont, latent_fsq, ee_chunk, wb_chunk}
```

Each cell trains one shared low-level policy, then one planner per selected
trajectory.

## Protocol

- Full LL budget: `TOTAL_FRAMES=2000000000`
- DiffSR encoder budget: `SKILL_UPDATES=50000`
- Skill train/eval split: `train/eval` with `SKILL_EVAL_TRAJECTORY_FRACTION=0.1`
- Planner history: `STATE_HISTORY_STEPS=9` (`10 x 93` input)
- Full eval: `EVAL_STEPS=500`
- Full rollout collection: `PLANNER_ROWS_PER_TRAJECTORY=1000`
- Smoke: `RANKS=0`, `EVAL_STEPS=16`, `PLANNER_ROWS_PER_TRAJECTORY=2`

Cadence is matched by `W`:

- `latent_*`: hold the latent for `W` control steps.
- `*_chunk`: publish every `W` control steps and consume `W` slots once each.

Do not use every-step chunk replanning for the fair cells.

## Run

Dry run:

```bash
DRY_RUN=1 experiments/lafan1_ablation/submit_cluster_horizon.sh
```

W10 smoke:

```bash
CLUSTER_PROFILE=skynet_lafan1_ablation \
DRY_RUN=0 BUDGET=smoke RANKS=0 WINDOWS=10 \
INTERFACES="latent_cont latent_fsq ee_chunk wb_chunk" \
experiments/lafan1_ablation/submit_cluster_horizon.sh
```

## Outputs

```text
logs/lafan1_ablation/<run>/horizon_ablation/
  W10/<interface>/
    encoder/
    oracle_ll/
    trajectories/rank_<k>_<name>/
      planner_pretrain/
      planner_finetune/
      eval/{oracle,pretrained,finetuned}/summary.json
  tables/
    horizon_ablation_long.csv
    horizon_ablation_wide.csv
    horizon_ablation_mean_by_setting.csv
    horizon_ablation_mean_wide.csv
```

Use `horizon_ablation_mean_wide.csv` for the compact result table.
