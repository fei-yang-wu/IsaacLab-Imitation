# BONES-SEED language10 oracle trajectory pretraining

This is the current local planner baseline for the ten tracking-qualified
BONES-SEED language goals. It replaces row budgets with complete trajectories.

Status: the real end-to-end smoke passed on 2026-08-05. Its oracle collection
completed all ten assigned motions at 100% SONIC SR and saved 519 planner
publications. The medium 20-update smoke then evaluated every explicit goal at
updates 10 and 20; both tiny checkpoints scored 0.4 SR, which is a wiring gate
rather than a convergence result.

The full seed-0 collection also completed on 2026-08-05: exactly 1,000
trajectories, 100 for every motion, with no discarded/incomplete episodes and
1,000/1,000 official SONIC successes. It recorded 513,700 control transitions
and wrote seven planner-sample shards. The complete run is under
`logs/bones_language10_oracle_pretrain_seed0`.

The planner was subsequently resumed with model and AdamW state preserved to a
20,000-update total. All 100 milestone evaluations are complete:

| update | SONIC SR | successful MPJPE-L | plateau? |
|---:|---:|---:|:---:|
| 2,000 | 0.295 | 47.79 mm | no |
| 4,000 | 0.293 | 53.39 mm | no |
| 6,000 | 0.307 | 52.60 mm | no |
| 8,000 | 0.306 | 49.08 mm | no |
| 10,000 | 0.339 | 46.68 mm | no |
| 12,000 | 0.305 | 44.09 mm | no |
| 14,000 | 0.307 | 48.98 mm | no |
| 16,000 | 0.312 | **41.96 mm** | no |
| 18,000 | 0.322 | 45.22 mm | no |
| 20,000 | **0.344** | 45.48 mm | no |

The curve never triggers the agreed plateau heuristic because it oscillates,
but more training is not a strong closed-loop win: 20k improves SR by only 0.5
percentage points over 10k. Held-out normalized RMSE improves from 0.1628 at
10k to a best 0.1485 at update 19.6k without a corresponding monotonic SR gain.
At 20k, fishing, lift-crate, and slow-arc walking are 1.00 SR; feeding birds is
0.36, stoop is 0.06, mosquito is 0.02, and four goals remain at zero. This is
still a closed-loop precision/covariate gap. Do not extend again or begin
DAgger implicitly; the receding-horizon comparison is the next controlled
decision point.

The required full-horizon visual diagnostic was also completed for the 10k
checkpoint. It rendered all ten motions to their own reference ends with every
early termination disabled, deterministic policy inference, startup/reset
randomization retained, and only the interval push removed. All ten remained
upright for the combined 5,137 control steps (fall-free survival 1.0). The
step-weighted full-horizon errors were 58.28 mm MPJPE-L, 0.322 m EE XYZ, and
300.10 mm MPJPE-G. The videos show a tracking/command-precision failure rather
than a balance failure: fishing remains visually close, whereas surrender does
not reproduce the raised-hands pose. The retained videos and per-motion
`metrics.json` files are under
`logs/bones_language10_oracle_pretrain_seed0/nonterminating_video/update_0010000_randomized_no_push`.

The collector launches 1,000 environments once: ten motions, 100 environments
per motion, all pinned to reference frame 0. The frozen oracle encoder supplies
the latent command and the low-level policy executes it until either an
official SONIC tracking termination or `reference_finished`. Policy actions are
deterministic. Startup and reset randomization remain enabled, while only the
push event is removed. SONIC foot-position XYZ and base-height terminations are
disabled.

Each accepted trajectory retains causal robot history, oracle latent target,
current expert and achieved 38-D `root_qpos`, and a masked 30-frame expert
`root_qpos` window. This is enough to retrain the latent planner and later test
direct root-qpos packets or temporal ensembles without recollecting simulation.
The ten-motion reference data is materialized as compact reference arrays,
not a Zarr replay; a direct evaluator check reproduces the known-good full
129k-array behavior, while the generic Zarr path does not for this checkpoint.
The ordered manifest SHA-256 is `60a5b7a5cf0056261d295f6ad02f70bbaf866409f69790932ad33d8ae736e7d1`;
the canonical 384-D `all-MiniLM-L6-v2` table SHA-256 is
`04624a22adba42f8db9acdc8c74f85ff985305c98ee9857f43b352c54048e0cd`.

The first planner is oracle-only: no planner-driven DAgger rows are mixed in.
The medium flow Transformer trains for 10,000 optimizer updates with a
trajectory-wise 80/20 train/validation split. Optimizer-free checkpoints at
2k, 4k, 6k, 8k, and 10k are each evaluated closed-loop on all ten explicit
language goals. `milestone_curve.md` reports SONIC SR and success-only MPJPE-L;
the plateau flag requires two consecutive 2k intervals with less than one
percentage point SR movement and less than 1 mm MPJPE-L movement.

From the repository root:

```bash
MODE=print experiments/campaigns/2026-08-05-bones-language10-oracle-pretrain/run.sh
MODE=smoke experiments/campaigns/2026-08-05-bones-language10-oracle-pretrain/run.sh
MODE=run experiments/campaigns/2026-08-05-bones-language10-oracle-pretrain/run.sh
MODE=video experiments/campaigns/2026-08-05-bones-language10-oracle-pretrain/run.sh
```

Resume or run one stage with `MODE=run RESUME=1 STAGE=collect|train|eval`. For
an optimizer-preserving extension, set the new total update target explicitly;
milestone numbering continues from the checkpoint rather than restarting:

```bash
MODE=run RESUME=1 STAGE=train NUM_UPDATES=20000 \
  experiments/campaigns/2026-08-05-bones-language10-oracle-pretrain/run.sh
MODE=run RESUME=1 STAGE=eval NUM_UPDATES=20000 \
  experiments/campaigns/2026-08-05-bones-language10-oracle-pretrain/run.sh
```

The default full output is `logs/bones_language10_oracle_pretrain_seed0`.
`MODE=video` reproduces the ten mandatory non-terminating randomized/no-push
comparison videos from the final checkpoint and prints every absolute path.
