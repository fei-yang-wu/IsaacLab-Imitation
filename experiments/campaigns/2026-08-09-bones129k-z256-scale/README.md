# 2026-08-09 — continuous z256, scaled capacity

One low-level arm on ICE. One variable against the existing `old_z256` row of
the 4,096-motion scoreboard: **network capacity**.

| row | job | W&B run | encoder trunk | tracker cells |
|---|---|---|---|---|
| control | ICE `5567801` | `bones129k-ablation` / `reset80_diffsr` | `[1024, 512, 512]` mish + LayerNorm | `[1024, 1024, 512]` |
| arm | ICE `5573513` → `5573514` | `bones129k_z256_scaled_*_seed0` | `[2048, 1024, 512, 512]` SiLU, no LayerNorm | `[2048, 2048, 1024, 1024, 512, 512]` SiLU |

The control's frozen scoreboard row is SONIC SR **0.9058**, success-only
MPJPE-L **24.52 mm** at 5,000,134,656 frames.

## What "scaled" means here

The widths are not new. They are the exact geometry of campaign
`2026-08-06-bones129k-sonic-fsq-scale`, which measured them under an FSQ
bottleneck (`fsq64_sonic`, SR 0.8943 / 25.74 mm). This campaign moves that same
capacity onto the **continuous z256** bottleneck, so the capacity axis reads
against a control that differs in nothing else.

| | control | arm |
|---|---|---|
| encoder trunk | `[1024, 512, 512]`, mish, LayerNorm on | `[2048, 1024, 512, 512]`, SiLU, LayerNorm off |
| DiffSR feature / embed | 128 / 512 | 256 / 1024 |
| DiffSR `g` and `mu` heads | `[512]` | `[1024, 1024, 512]` |
| actor `num_cells` | `[1024, 1024, 512]` | `[2048, 2048, 1024, 1024, 512, 512]` |
| critic `num_cells` | `[1024, 1024, 512]` | `[2048, 2048, 1024, 1024, 512, 512]` |
| activation | tuned entry-point default | SiLU |

Unchanged, byte-for-byte from the control: root-qpos macro state (380 wide),
ten slots at stride 1, `robot` anchor frame, endpoint DiffSR objective,
deterministic z256 + sin/cos phase = 258-wide command held 10 control steps,
critic channels `[actor, reference]` (the critic still sees the latent),
`random80_adaptive20` resets, the reward weights, the 5M–30M termination
curriculum, 16,384 environments x 24 rollout steps, minibatch 294,912,
gamma 0.97, seed 0, 50 M-frame checkpoints under persistent `/data`, and the
10B frame cap.

**Deliberately not combined** with the two 2026-08-08 ingredients still in
flight — `expert_heading` anchor frame (ICE `5573234`) and critic without the
actor latent (ICE `5573413`). Folding either in would make this a combined arm
and destroy the capacity attribution.

## Gates that ran before submit

- Local smoke, `logs/bones129k_z256_scale_smoke/run1`: one real pretrain update
  plus one real PPO iteration.
- Encoder gate: the checkpoint records `macro_anchor_mode=robot`, stride 1,
  input width 380, trunk `2048x380`, DiffSR 256/1024, LayerNorm off. It fails
  loudly if the trunk comes back at the control's 1024, so a silent revert to
  the tuned geometry cannot burn a 10B allocation as a duplicate control.
- Tracker gate: the actor's first layer is 2,048 wide at input 351
  (93 + 258). Same refusal on the control's 1,024.
- Remote gates: reference-array identity (129,785 trajectories, 47,491,234
  frames, `bones_seed_sonic_full_129785@e714bbff`) and fresh output paths.

## Commands

From the repository root:

```bash
MODE=print experiments/campaigns/2026-08-09-bones129k-z256-scale/run.sh
MODE=smoke experiments/campaigns/2026-08-09-bones129k-z256-scale/run.sh
MODE=validate LOCAL_SMOKE_ROOT=logs/bones129k_z256_scale_smoke/run1 \
  experiments/campaigns/2026-08-09-bones129k-z256-scale/run.sh
MODE=submit LOCAL_SMOKE_ROOT=logs/bones129k_z256_scale_smoke/run1 \
  CONFIRM_SUBMIT=latent-capacity-scale \
  experiments/campaigns/2026-08-09-bones129k-z256-scale/run.sh
```

W&B project `g1-bones-seed`, group `latent-capacity-scale`.

## Reading the result

Score it on the frozen 4,096-motion scoreboard
(`experiments/campaigns/2026-08-08-bones129k-4096-scoreboard/`) at the
**5,000,134,656-frame** checkpoint, which the control also has. The scaled
tracker is slower per frame, so a 15:59:00 allocation may TIMEOUT before the
10B cap; that costs nothing because checkpoints land on persistent `/data`
every 50 M frames, but it does mean the matched-budget row is the 5B one, not
whatever the job reaches last.

Submission record with both job IDs and the source-contract hash:
`cluster_submission.json`.
