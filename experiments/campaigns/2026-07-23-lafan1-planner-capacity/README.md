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

The interfaces form a **packet-size ladder** at a fixed 5 Hz publication rate. Each
has its **own** low-level controller trained natively on that command space — this
is not one tracker fed adapted commands, so nothing is reconstructed (a
`root_qpos` controller simply never receives joint velocities).

| Interface | Packet @ 5 Hz | Contents / frame | Frames | Qualified oracle | Training plateau |
| --- | --- | --- | --- | --- | --- |
| `full_body_trajectory` | 670 | 29 qpos + 29 qvel + root 9 | 5.000B | **23.8 mm** | 34.1 mm / 454 |
| `root_qpos` | 380 | 29 qpos + root 9 (no qvel) | 4.600B | **23.6 mm** | — |
| `ee_trajectory` | 360 | 4 EE poses, **no root** | 5.000B | **405.2 mm** | 41.3 mm / 424 |
| `latent_skill` (DiffSR det.) | 258 | z256 + phase | 4.525B | **30.5 mm** | 45.6 mm / 393 |
| `root_points5` | 240 | 5 keypoints ×3 + root 9 | 4.800B | **30.6 mm** | — |

"Root 9" is `expert_anchor_pos_b` (3) + `expert_anchor_ori_b` (rot6d, 6) — a pose,
**no velocity anywhere** in either reduced packet.

> **Use the qualified column, never the training plateau.** The training metric is
> measured under random starts with terminations active, so episodes end before
> drift accumulates. `ee_trajectory` reads a healthy 41.3 mm there while its true
> frame-0/700-step floor is 405.2 mm — a 10× gap under the same metric name. The
> same effect made full-body look like 68 mm under random-start M3 versus 309 mm
> full-horizon.

**`ee_trajectory` is the rootless control, not a failed run.** Its adapter was
verified correct (chunked streaming reproduced its own floor to −5.3 mm); the
interface itself is under-determined — 4 body poses in the *torso* frame never say
where the torso goes, and 4 poses do not determine 29 joints. Note it carries
**more** values than `root_points5` (360 vs 240) and is 13× worse: packet size does
not determine trackability, closing the kinematic chain does. It is disabled by
default in the paper config; enable it only to reproduce that measurement.

The original "main" latent LAFAN1 tracker (job `5525664`) was destroyed in the
2026-07-22 Slurm-TIMEOUT data-loss incident; the surviving latent-learning-ablation
`deterministic` arm is the protocol-matched substitute. Note it is ~475M frames
short of the explicit arms — a gap that runs *against* the latent claim, so the
result stands as a conservative lower bound.

### `enc380` — the content-controlled latent arm (in flight, 2026-07-28)

The ladder above confounds two axes. `root_qpos` (380 explicit) reaches 23.6 mm
and `latent_skill` (258) reaches 30.5 mm, but the latent encoder was fit on the
**full-body 670** packet — so "explicit vs latent" and "qpos+root content vs
qpos+qvel+root content" move together, and neither row alone attributes the gap.

`enc380` holds the content fixed and moves only the compression: the same 38
values per frame the `root_qpos` tracker consumes explicitly are fed through a
DiffSR skill encoder and published as the same 258-value latent command.

| | planner/oracle output | tracker input |
| --- | --- | --- |
| `root_qpos` | 380 explicit values | 380 |
| `enc380` | 380 → **frozen encoder** | z256 + phase = 258 |
| `latent_skill` | 670 → **frozen encoder** | z256 + phase = 258 |

`enc380` vs `root_qpos` isolates compression at fixed content; `enc380` vs
`latent_skill` isolates content at fixed compression.

Everything but the encoder's input width is copied from the frozen latent
oracle's recipe (deterministic continuous z256 + sin_cos phase, h10 hold,
encoder 1024/512/512, 50k pretrain updates at batch 8192, corrected 40-motion
tree), and the tracker geometry is the same H100 point as `root_qpos` /
`root_points5` (12,288 × 12, minibatch 18,432, lr 1e-3, 5B cap).

The env-side selector is `env.expert_macro_state_terms`; the window term
`expert_motion_qpos` makes the encoder's input byte-identical to the `root_qpos`
packet. Launcher: `submit_enc380_latent_low_level_ice.sh`, run as two stages —
`STAGE=pretrain` once, then `STAGE=train` per ~16 h segment. The split is
deliberate: the encoder is written to the shared `/data` bind rather than to
per-submission workspace logs, so a TIMEOUT-killed segment can never silently
re-pretrain a *different* encoder and resume a tracker into a latent space it was
never trained on. Nothing downstream would error if that happened.

Local gates that preceded the ICE push (both on `walk1_subject1`, both passed):
a 5,000-update encoder pretrain over the 380 macro state
(`logs/interface_baselines/enc380_root_qpos_seed0`, sample-recon L1 62.8 → 0.387,
z effective rank 135/256) and a 40-iteration frozen-encoder tracker smoke on
`Isaac-Imitation-G1-Latent-Strict-v0` (`logs/interface_baselines/enc380_tracker_smoke`).
Neither is a performance result.

## Reproducing the whole study

The paper-facing entrypoint drives the scripts in this directory and enforces the
gates around them. Prefer it over calling the shell scripts directly — it verifies
checkpoint hashes, refuses an unqualified interface, refuses a dirty study root,
and records `study_provenance.json` (resolved config + hashes + git commit) next
to the artifacts.

```bash
# everything: qualify -> oracle -> grid -> aggregate
pixi run python experiments/paper/run_interface_capacity_study.py

# one interface / one cell, for a smoke check
pixi run python experiments/paper/run_interface_capacity_study.py \
    grid.interfaces=[root_points5] grid.seeds=[0] grid.sizes=[tiny]

# reproduce the rootless-control measurement (expects UNUSABLE INTERFACE)
pixi run python experiments/paper/run_interface_capacity_study.py \
    stages=[qualify] grid.interfaces=[ee_trajectory] \
    interfaces.ee_trajectory.enabled=true
```

Every parameter — interfaces, checkpoints + hashes, protocol, planner budget,
grid, thresholds — lives in `experiments/paper/conf/interface_capacity.yaml`.
Nothing needs hand-editing before a run.

**Adding an interface** requires, in order: a command space in
`rlopt_ipmd_cfg.py`; an entry in `INTERFACE_TERMS`; a dispatch arm in
`run_capacity_point.sh` and `prepare_oracle_baselines.sh`; its own trained
low-level controller; a PASS from `qualify_interface.sh`; then an entry in the
paper config. The packet layout is *derived* from the command space in
`collect_interface_rollout_samples.py` and cross-checked against
`INTERFACE_TERMS` at startup, so a new interface cannot silently disagree with
itself — that mismatch is what the joint-order bug was.

## Gates

- `qualify_interface.sh` — replay floor vs 5 Hz oracle stream, plus an absolute
  floor ceiling (`FLOOR_MAX_MM`). Writes `qualification.json`; the paper
  entrypoint refuses to spend planner compute without a `PASS`. The ceiling
  exists because the gate originally *passed* `ee_trajectory` at a 405 mm floor —
  it only checked faithfulness-to-floor, not whether the floor was sane.
- `smoke_test_reduced_interface_streaming.py` — certifies that the phase-aligned
  held-packet slot equals the live unchunked reference at every hold phase, with
  and without asynchronous resets. Needs no trained policy, so it can run before
  any controller exists.

## Pipeline

1. `run_capacity_point.sh` (per size × seed): oracle demos → planner pretrain (demo-only)
   → eval → rollout collect → merge → finetune → eval, for each selected interface.
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
