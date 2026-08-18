# 2026-08-17 — canonical paper-facing metrics

## Purpose

Decide, once, what a paper-facing tracking number in this repo means, and
calibrate it against NVIDIA's released SONIC checkpoint so our figures can be
set beside the SONIC paper's without an apples-to-oranges comparison.

The protocol definition lives in `wiki/canonical-paper-metrics.md`. Read it
first. This directory holds only the launcher that regenerates the reference
rows.

## Question that started it

The SONIC paper advertises 22.3 mm MPJPE-L at 100% success. Our 4,096-clip
board scored the released SONIC checkpoint at 28.65 mm. Where does the
difference come from?

## Answer

Two separate things, in this order.

1. **Domain randomization.** Our board ran `no_push`, which keeps startup and
   reset randomization. Turning randomization fully off moves the released
   checkpoint from 28.66 mm to **25.90 mm** and success rate from 0.9934 to
   0.9946. Quality is randomization-sensitive; success is not. (The `no_push`
   row was measured twice, 2026-08-07 and 2026-08-17: 28.65 and 28.66 mm, so
   MPJPE-L repeats to about 0.01 mm on this board.)
2. **Motion population, which is the larger term.** The paper's 22.3 mm is its
   **123-clip hardware deployment set** scored in simulation, not a large
   benchmark; its large-set row is test-content 98.7% / 23.2 mm. Our block
   contains deep-crouch and ground clips no hardware set has. Restricting to
   hardware-plausible clips by a reference-only rule gives **21.92 mm at
   SR 1.0000** on 123 clips — the paper's headline, reproduced.

Neither retargeting nor the physics backend is needed to explain the gap.
(PhysX is 2 mm *worse* than Newton/MJWarp for this checkpoint, so a backend
argument moves the number the wrong way.)

## The falsification step

The rule's thresholds were chosen while looking at the canonical block, so the
first pool number was post-hoc. The frozen rule was then applied unchanged to
a disjoint block (ranks 20480-24575):

| board | canonical 12288-16383 | held-out 20480-24575 |
| --- | --- | --- |
| full 4,096 | 25.90 mm / SR 0.9946 | 25.86 mm / SR 0.9937 |
| deployable pool | 22.16 mm / SR 1.0000 (n=1869) | 21.92 mm / SR 0.9995 (n=1893) |
| deployable-123 | 21.92 mm / SR 1.0000 | 22.02 mm / SR 1.0000 |

The rule holds on data it was not written on.

## Run it

```bash
./experiments/campaigns/2026-08-17-paper-metric-canon/run_sonic_reference_rows.sh
./experiments/campaigns/2026-08-17-paper-metric-canon/run_sonic_reference_rows.sh --report
```

About 12 minutes per row on one RTX PRO 6000. Results land in
`logs/sonic_release_4096/`. The deployable-123 row is a subset of the clean
canonical run, not a fourth run.

## What this does not settle

- Whether our own trackers keep their ranking under the clean protocol. Every
  pre-2026-08-17 arm was scored under `no_push` and needs re-scoring on
  `paper_scoreboard4096_v1` before its quality number is paper-facing.
- Run-to-run Isaac noise on a single arm. Population noise between two 4,096
  blocks is under 0.1 mm, which is not the same quantity as seed noise.
