# 2026-08-24 — The paper headliner: `ema_h1_ee_wide` to 50B

Continue the promoted screen winner from 20B to 50B. This chain is intended to
REPLACE `cont_det_ln_hold1` / `ln_hold1_sonicreset` as the headline low-level
row, so it is budget-matched to the ln leader's 50B chain.

## The recipe and why each piece is in it

| ingredient | evidence (2B screen, matched clips unless noted) |
|---|---|
| DiffSR endpoint grounding | owns LOCAL tracking: -22% / -43% L in the two one-variable contrasts |
| EMA-lagged chunk prediction | owns GLOBAL drift: `dsrsig` -> hub 103.0 -> 78.0 mm G (-24%); stopgrad-only prediction gives just -8% |
| SIGReg | smallest term, +0.007 SR (inside noise), kept because it is free |
| hold 1 | -29% G vs hold 10 at the star; leads at 10B |
| `motion_ee_pos` 1.0 | wrist terminations are 78-96% of all failures; the matched reward was inert at 0.0 |
| `motion_global_anchor_pos_wide` 1.0 (std 0.5) | -18 to -29% G alone on both interfaces; designed-but-unscreened since 2026-08-04 |

Deliberately EXCLUDED, each with a measured reason: qvel frames (L gain 4.5%
is inside noise, G cost 20.5% is not), chunk triplets (refuted, both cells),
the online/LeJEPA target (0.8384 vs hub 0.9060), online dyn finetune (null at
2B), the asymmetric critic (null at 2x cost).

## Schedule

0 -> 20B is `2026-08-23-emastack-20b` (jobs 5590001-06): 10B under
`random80_adaptive20` with the termination curriculum, then `selection=sonic`
with the 0.5 -> 0.1 landing ramp, then pinned 0.1 — the ln leader's exact
history, so the 20B rows compare directly.

20B -> 50B is this campaign: `selection=sonic` with
`env.command_interface.reference.selection.adaptive_uniform_ratio=0.1` and
`adaptive_uniform_ratio_final=null` (static, no ramp) for every segment.
That field is `SonicAdaptiveResetSampler(uniform_sampling_rate=...)` in
`ImitationLearningTools/iltools/datasets/reset_sampling.py`: reset starts are
drawn from a distribution over 50-frame bins (`adaptive_bin_size`) where
10% of the mass is uniform across all bins and 90% is proportional to each
bin's failure rate, clipped at `failure_rate.mean() * 50.0` and normalized by
`bin_weights` so long clips do not dominate; the drawn frame is then shifted
back by a random lead-in of up to `adaptive_pre_failure_window=200` frames.
0.1 is the MEASURED value (0.9558 @20B, 0.9707 @30B on the ln arm). The
concurrent ln 50B chain runs 0.05 (5% uniform / 95% failure-weighted) for its
last 20B; that value has no scored row yet, so the headliner does not adopt
it. Revisit if the ln 50B row shows 0.05 wins.

Five chained segments (`afterany`); 30B at ~125k fps is ~67 h, so cont1-4
carry it and cont5 is insurance.

## Rows to beat (canonical 4,096 board, `no_push`, one seed)

| row | SR | L | G |
|---|---:|---:|---:|
| released SONIC | 0.9937 | 28.65 | — |
| `ln_hold1_sonicreset` @30B | 0.9707 | 21.75 | 154.64 |
| `ln_hold1_sonicreset` @20B | 0.9558 | 22.15 | 168.15 |

## Submission gate

DO NOT submit until the 20B chain reaches its cap — `cont1` resumes from
`/data/emastack_20b/ema_h1_ee_wide_seed0/tracker` and would otherwise continue
an unfinished run. Check with:

```bash
ssh ice 'ls ~/scratch/Research/IsaacLab/data/emastack_20b/ema_h1_ee_wide_seed0/tracker/*/models/ | sort -t_ -k3 -n | tail -1'
```

W&B: appends to the same run as the 20B chain (`ps20-emaeew-s0-1b9913`);
`<output_root>/wandb_run_id` was seeded with that resolved id on ICE before
planning, so the control plane cannot mint a new one and split the curve.

## Status

- 2026-08-24: campaign created, plan resolves, run-id seeded. NOT SUBMITTED —
  waiting on the 20B chain. Score the 20B row when it lands (it is the direct
  comparison to `ln_hold1_sonicreset` @20B) before continuing.
