# 2026-08-27 — `diffntp_token_h1_ee_wide`: re-score and showcase clips

Score the arm's final checkpoint on the canonical 4,096-clip board, clean and
robust, and render policy-only clips from the same file. Everything runs on
ICE, where the tree already sits, so no checkpoint moves.

## The arm

Round 4 of `2026-08-22-pareto-stack`: generative next-TOKEN prediction
(`--jepa_ntp_head diff_token`), the clean one-variable swap of the hub's
conditional-mean MLP predictor for a DiffSR diffusion head at the same
`jepa_ntp_coeff` slot. Hold 1, 256-D deterministic LN encoder, `ee` + `wide`
rewards. Jobs 5591939-41.

It trained 2B frames in ONE segment and stopped there — the pareto-stack screen
budget. `model_step_2000289792.pt` is its last checkpoint AND its final one;
there is no longer chain to catch up with.

## What was already known

| row | board | clips | SR | MPJPE-L | MPJPE-G |
|---|---|---:|---:|---:|---:|
| clean | `bones_testbed4096_v1` | 4096 | 0.9121 | 24.44 | 86.29 |
| f2.0B | `bones_milestone_testbed256_v1` | 256 | 0.9258 | 24.11 | 84.20 |

Both are one seed, scored from the local mirror on 2026-08-26. The 256-clip row
is the last point of the budget curve, not a board of record: the canonical
metric set is the 4,096-clip board. `wiki/results-interface-ablations.md` §5.6
prints the 256-clip numbers in a table whose other sections quote 4,096-clip
rows.

The `robust` row (`bones_testbed4096_v1`, domain randomization on, pushes off)
was never scored for this arm.

## Stages

| stage | what | why |
|---|---|---|
| `clean` | final checkpoint, 4,096 clips, `--randomization none` | second sample of a non-deterministic evaluation |
| `robust` | same checkpoint and board, `--randomization no_push` | the missing row |
| `video` | three clips, PhysX + Kit RTX, `studio_light` / `hero_low` | showcase render |

The stages are chained `afterany` and therefore serialized. Two evaluation jobs
that started eleven minutes apart shared one Isaac Sim cache and the second died
inside Kit startup (ICE 2026-08-27, `hold5-curve-eval`); until that is
understood, evaluation jobs on this profile do not run concurrently.

Rows land in `/data/eval/diffntp_token_showcase` as
`diffntp_token_h1_ee_wide_seed0_<row>_f2000289792.json`; clips and their summary
JSON land in `/data/eval/diffntp_token_showcase/video`.

## The ranks

Surviving clips of the arm's own clean row, picked for low local error and
enough length to read as a clip:

| rank | motion | control steps | MPJPE-L | MPJPE-G |
|---:|---|---:|---:|---:|
| 2389 | `walk_forward_loop_001_A024_M` | 1041 | 12.52 | 45.73 |
| 72338 | `dancing_routine_1_003_A041_M` | 820 | 12.90 | 25.91 |
| 121035 | `reach_jump_R_002_A215` | 661 | 12.32 | 29.90 |

## One code change this campaign needed

`docker/cluster/run_singularity.sh` sent `scripts/viz/*.py` to the bare
`/isaac-sim/python.sh` branch, which has Kit but no torch, so a render job would
have died on `No module named 'torch'` seconds in. The rendering entrypoints are
in the evaluator class — same environment, same torch checkpoint, `AppLauncher`
at import — and now select the same interpreter.

The video stage is the FIRST RTX render submitted on ICE. The repo's standing
preference is to render locally, because a fresh Isaac Lab container costs more
per job than the render itself; this run is here because the user asked for the
whole sequence on the cluster.

## Status

- 2026-08-27 first pass, jobs 5593843-45. `robust` MEASURED:
  **0.9050 / 26.96 / 135.88** on 4,096 clips, one seed, 12 minutes. Against the
  clean row (0.9121 / 24.44 / 86.29) domain randomization costs 0.007 SR,
  2.5 mm local and 50 mm global.
  `clean` died at 560 ms in a Kit startup segfault (exit 139), the same
  signature as the `hold5-curve-eval` startup failures on this profile.
  `video` reached the render and died on `No module named 'torch'` -- the
  interpreter was right after the `run_singularity.sh` fix, but nothing had put
  the CU130 site-packages on the path, and `AppLauncher.__init__` imports Torch.
  **It exited 0 and Slurm recorded COMPLETED.** The evaluator branch has no
  workload success marker, so an output file is the only trustworthy signal.
- 2026-08-27 second pass, jobs 5593917-19, after
  `render_paper_policy_video.py` gained the `configure_cu130_bridge` call that
  `eval_checkpoint_tree.py` already made. `--rescore` was dropped from both
  evaluation stages, so `robust` now plans zero cells and returns before Isaac
  Sim starts; only `clean` and `video` do work.
