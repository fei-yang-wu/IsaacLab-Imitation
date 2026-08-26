# 2026-08-23 — SONIC resets, 30B to 50B, harder failure focus

Continue `ln_hold1_sonicreset` from its 30B checkpoint for another 20B, with
ONE regime change by explicit user decision:
`env.command_interface.reference.selection.adaptive_uniform_ratio` moves
0.1 -> 0.05. That field is `SonicAdaptiveResetSampler(uniform_sampling_rate=...)`
in `ImitationLearningTools/iltools/datasets/reset_sampling.py`, and it sets the
UNIFORM share of the reset-start distribution over 50-frame bins:
`P(bin) = failure_prob * (1 - r) + (1/num_bins) * r`. So 0.05 puts 95% of the
mass on bins in proportion to their clipped failure rate (up from 90%) — more
reset starts at the trajectory points the tracker still fails.

State at 30B (canonical 4,096 board, `no_push`, one seed):

| row | SR | L | G |
|---|---:|---:|---:|
| `ln_hold1_sonicreset` @30B | 0.9707 | 21.75 | 154.64 |
| @20B | 0.9558 | 22.15 | 168.15 |
| released SONIC | 0.9937 | 28.65 | — |

## Axis warning

30B -> 50B is NOT a budget-only read: budget and the sharper reset focus move
together, by design. 20B -> 30B stays the clean budget axis. If the 50B row
must be decomposed, a 0.1-ratio control continuation would be needed.

## Protocol

- Resume: segment 12 loads the newest 30B-tree checkpoint
  (`cumulative_env_frames` carried; global `frame_cap=50000000000`).
- Everything else byte-identical to segments 9-11: frozen encoder from
  `/data/bottleneck_10b/cont_det_ln_hold1_seed0/encoder`, termination
  curriculum off, `selection=sonic`, `save_interval` 500M.
- Four chained segments (`afterany`); 20B at the chain's ~129k fps is ~43 h,
  so segments 12-14 carry it and segment 15 is insurance.
- W&B: same run `ln-hold1-sonicreset-s0-r1` (group `sonic-reset-20b`);
  `wandb_run_id` state file hand-seeded on ICE before submission. Shared-mode
  env block grandfathered from the chain's creation; never copy it into new
  runs. Read `env_frames`, not `_step`.

## Scoring

`2026-08-15-latent-bottleneck-10b/eval_scoreboard4096.sh` with an
`ln_hold1_sonicreset|50000021504-ish` row (file name carries the exact final
frames), plus milestone rows every 500M for the SR-curve shape.

## Status

- 2026-08-26 10:05: at 46.67B of 50B. `lowlevel14` (5590009) hit its walltime
  at 46.60B and `lowlevel15` (5590010) resumed from it. Node
  `atl1-1-03-017-16-0` carries five jobs, so the chain logs 43-50k fps against
  its usual ~129k; at that rate segment 15 reaches only ~49.1B and it was the
  last declared segment. Declared `lowlevel16` as second insurance and planned
  it (preflight all OK), but the user held the submission, so the chain still
  ends after segment 15. Segment 16's predecessor is a live job with no sibling
  stage in the plan, so it uses the control plane's new external dependency.
  Re-plan before submitting: the plan seals the working tree, which has moved
  since.

  ```bash
  pixi run python -m imitation_experiments.pipeline.cluster plan \
      --campaign experiments/campaigns/2026-08-23-sonic-reset-50b/campaign.yaml \
      --arm ln_hold1_sonicreset --seed 0 \
      --only-stage lowlevel16 --set vars.chain_after=job:5590010
  ```

  The 46.5B checkpoint scored 1.0000 SR / 19.44 mm L / 122.08 mm G on the new
  common eval subset `sonic_capability124_v1` — a progress read, not the 50B
  row. See `wiki/sonic-v1_1-subsets.md`.
- 2026-08-23 20:07: submitted, jobs 5588645-48 (segments 12-15). Run-id file
  seeded. Nothing measured. fsq64 stays at 30B pending its row.
