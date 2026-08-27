# 2026-08-22 — Pareto stack: SR, MPJPE-L, and MPJPE-G together

Read `wiki/tracker-pareto-program.md` first; it is the plan of record this
campaign executes. Goal: find the recipe that improves all three canonical
metrics at once, from the measured levers — the online-dynamics finetune
(dyn) on the hold-1 interfaces, and the two UNSCREENED reward terms aimed at
the dominant failure modes (wrist terminations for SR, root drift for G).

W&B group: `pareto-stack-2b` (confirmed with the user 2026-08-22).
2B screen, byte-identical star protocol except where an arm says otherwise.
Single seed per arm except the declared s1 pair.

## Reference rows (2B screen, seed 0, `paper_testbed4096_v1` clean)

| row | SR | MPJPE-L | MPJPE-G |
|---|---:|---:|---:|
| `ctrl` (star hub, hold 10) | 0.9023 | 24.49 | 212.3 |
| `use_hold1` | 0.8921 | 26.44 | 150.4 |
| `jepa_ebm_hold1_256d` | 0.8918 | 27.50 | 142.0 |

Dyn evidence at 10B (canonical board): `cont_det_hold1_dyn` G -17.9% at level
SR/L; dyn saturates on top of failure-driven resets; dyn hurts fsq64 hold-10.

## Arms

Q1 — is the dyn G-effect real at the screen, and does it stack?

| arm | seed | changes vs parent | parents |
|---|---|---|---|
| `hold1` | 1 | none (use_hold1 re-declared for the seed axis) | `use_hold1` s0 |
| `hold1_dyn` | 0 | + dyn block | `use_hold1` s0 |
| `hold1_dyn_s1` | 1 | + dyn block, encoder from `hold1` seed 1 | `hold1` s1 |
| `jepa_h1_dyn` | 0 | + dyn block | `jepa_ebm_hold1_256d` s0 |

`jepa_h1_dyn` is the decisive cell: stack (sub-120 mm G) or saturate
(~140 mm, dyn dies at the screen).

Q2 — do the unscreened reward terms move SR (wrist) and G (root)?

| arm | reward delta | interface |
|---|---|---|
| `ctrl_ee1` / `jepa_h1_ee1` | `motion_ee_pos.weight=1.0` | hub / jepa hold-1 |
| `ctrl_wide1` / `jepa_h1_wide1` | `motion_global_anchor_pos_wide.weight=1.0` | hub / jepa hold-1 |
| `ctrl_ee_wide` / `jepa_h1_ee_wide` | both | hub / jepa hold-1 |
| `ctrl_bodyglobal` | `motion_body_pos_global.weight=0.5` | hub |
| `ctrl_asymcritic` | critic reads the FULL explicit reference set (`command_interface.reference.critic_components`), policy byte-identical | hub |

REWARD ARMS DO NOT READ AGAINST THE STAR TABLE: the reward set moves, so the
return column is incomparable. SR and both MPJPE columns are the readout.

Encoder reuse: reward and dyn arms pin `encoder_ckpt` to their parent's
pretrained file (preflight-verified on ICE), so encoder-initialization
variance is excluded from those comparisons by construction. Only `hold1`
(seed 1) pretrains an encoder; `hold1_dyn_s1` consumes it.

## Objective-mechanism square (added 2026-08-22, post-screen)

Five arms on the winning cell (hold-1 + `ee` 1.0 + `wide` 1.0), pretrain per
arm, decomposing `sigreg_ebm` = DiffSR + NTP + SIGReg with EMA targets:

| arm | DiffSR | NTP | SIGReg | target | question |
|---|---|---|---|---|---|
| `jepa_h1_ee_wide` (hub, screened) | y | y | y | EMA | 0.9060/26.98/103.6 |
| `endpoint_h1_ee_wide` | y | - | - | - | interface-matched objective control |
| `dsrsig_h1_ee_wide` | y | - | y | - | chunk-wise non-JEPA (`--jepa_ntp_coeff 0`, new knob) |
| `lejepa_h1_ee_wide` | - | y | y | online | true LeJEPA shape |
| `jepa_ol_h1_ee_wide` | y | y | y | online | does the EMA copy matter |
| `jepa_nosig_h1_ee_wide` | y | y | - | EMA | is SIGReg load-bearing |

Skipped with evidence: NTP+SIGReg+EMA without DiffSR was the star's
`obj_jepa_ntp` (0.8218/36.77 at hold-10). `--jepa_ntp_coeff` is a new RLOpt
config field (validated non-negative; 0 drops the predictor from the loss),
tests in `RLOpt/tests/test_hl_skill_recon_phase.py`.

## Mechanism-square results (2026-08-23, clean rows, one seed)

| arm | SR | L | G |
|---|---:|---:|---:|
| `jepa_h1_ee_wide` (hub) | 0.9060 | 26.98 | 103.6 |
| `jepa_nosig_h1_ee_wide` | 0.8992 | 27.07 | 107.5 |
| `dsrsig_h1_ee_wide` | 0.8877 | 27.83 | 124.6 |
| `endpoint_h1_ee_wide` | 0.8823 | 27.55 | 117.4 |
| `jepa_ol_h1_ee_wide` | 0.8655 | 28.29 | 128.6 |
| `lejepa_h1_ee_wide` | 0.7048 | 47.66 | 568.9 |

Attribution: EMA target is the second-biggest ingredient (-0.041 SR when
dropped, even with DiffSR+SIGReg intact); true LeJEPA (online, no EBM)
collapses; the NTP predictor is load-bearing (+0.024 SR, -12% G over plain
endpoint at matched interface+rewards — the screen's jepa-vs-ctrl gap was the
objective, not the hold confound); SIGReg is the smallest term (+0.007, at
noise). The trained composite sits at the optimum of its own square.

`ctrl_asymcritic` 0.8955/24.68/201.0 vs ctrl 0.9023/24.49/212.3: NULL on all
three at 2x training cost (51k vs 128k fps). Retired. (Its eval needs the
arm's `extra_args` env overrides — eval.sh now passes them through; the
strict ValueOperator restore fails otherwise.)

## Round 3 (2026-08-23): triplet, qvel, asymmetry-vs-lag

Motivated by the lejepa diagnosis (prediction shortcut: pred/copy 1e-4 at
z-rank 149 — co-adaptation, not collapse; only the EMA arm's predictor beats
copy, 0.145 at 62% retrieval). All on hold-1 + ee + wide, own pretrains:

| arm | delta vs `jepa_h1_ee_wide` | jobs |
|---|---|---|
| `trip_h1_ee_wide` | chunk TRIPLET (`--jepa_context_chunks 1`), first trained use | 5588649-51 |
| `qvel_h1_ee_wide` | full_body 67-wide frames (qpos+qvel), pair | 5588652-54 |
| `trip_qvel_h1_ee_wide` | triplet + qvel (Euler-Lagrange cell) | 5588675-77 |
| `lejepa_sg_h1_ee_wide` | JEPA + stop-grad target (SimSiam asymmetry, no lag, no DiffSR) | 5588655-57 |
| `sg_h1_ee_wide` | production recipe, stopgrad instead of EMA | 5588658-60 |

RLOpt gains (2026-08-23, tests in `test_hl_skill_recon_phase.py`, 178 pass):
`jepa_target_encoder_mode=stopgrad`, and `_reanchor_heading_frames` /
the jepa_ntp state gate generalized to the 67-wide full_body frame (joint
qpos+qvel prefix is frame-invariant; only the trailing 9 transform).

## Round 4 (2026-08-26): generative next-chunk prediction

The hub's prediction term `jepa_ntp_coeff * ||P(z1) - z2||^2` is a conditional
MEAN estimator; chunk futures are multimodal. `--jepa_ntp_head` (new RLOpt
flag) swaps the estimator for a second DiffSR diffusion head at the same
coefficient slot, leaving the endpoint term and SIGReg unchanged:

| arm | denoising target | width | jobs |
|---|---|---:|---|
| `diffntp_token_h1_ee_wide` | p(z_next \| s_t, z_t), EMA token | 256 | 5591939-41 |
| `diffntp_chunk_h1_ee_wide` | p(x_{t+H+1:t+2H} \| s_t, z_t), executed chunk's heading frame | 380 | 5591942-44 |
| `diffntp_chunkra_h1_ee_wide` | same, RE-ANCHORED onto s_{t+H}'s own frame (drift erased; isolates whether drift-in-target is load-bearing) | 380 | 5591947-49 |
| `diffntp_pair_h1_ee_wide` | p(s_{t+H}, z_next \| s_t, z_t) JOINTLY (38+256) | 294 | 5591950-52 |

The estimator question is mean-vs-generative only: the discriminative /
contrastive family is EXCLUDED by user decision (2026-08-26) — pure JEPA-style
formulations are known dead here and diffusion is the chosen generative form. Design decisions recorded: the chunk head's target keeps
the cross-chunk displacement (anchored at s_t, not at s_{t+H}, so drift stays
in the target); `diff_chunk` carries NO self-target — the EMA trick leaves
the objective entirely, making it the fully well-posed generative cell; both
heads condition on (s_t, z_t) where the mlp P saw z_t alone (a deliberate,
documented second delta). Deliberately skipped: the deterministic-MSE-on-raw-
chunk cell (a blurry mean chunk), per-frame autoregressive factorization
(breaks the chunk-atomic design at Hx cost), CVAE (out of paper scope),
flow matching (drop-in alternative parameterization of the same head, defer).

### Round 4 results (2026-08-26, clean rows, one seed)

Matched on the 3,521 clips all six arms survived (own -> matched):

| arm | SR | L | G | ee |
|---|---:|---|---|---:|
| `dsrsig` (no prediction) | 0.8877 | 28.07 -> 27.06 | 129.00 -> 120.82 | 382 |
| hub (mlp, conditional mean) | 0.9060 | 27.16 -> 25.62 | 104.72 -> 91.40 | 322 |
| `diffntp_token` | 0.9121 | 24.44 -> 23.43 | 86.29 -> 79.29 | 277 |
| `diffntp_chunk` (exec frame) | 0.9163 | **24.18 -> 22.87** | **85.74 -> 74.57** | 276 |
| `diffntp_chunkra` (next frame) | **0.9194** | 24.81 -> 23.37 | 103.31 -> 88.49 | 276 |
| `diffntp_pair` (s,z joint) | 0.9182 | 24.51 -> 23.06 | 96.22 -> 82.86 | 282 |

EVERY generative head beats the deterministic hub on ALL THREE metrics.
`diffntp_token` vs hub is the clean one-variable estimator swap (same target,
same horizon): +0.006 SR, -8.6% L, -13.2% G matched. The conditional-mean
limitation of the MSE predictor was real and costly.

ANCHOR VERDICT (`chunk` vs `chunkra`): keeping the cross-chunk displacement in
the target buys drift resistance — matched G 74.57 vs 88.49 (-16%) — while
erasing it buys slightly more SR (0.9194 vs 0.9163, inside noise). For a
planner-facing interface, `chunk` (executed frame) is the pick.

Wrist (`ee_body_pos`) failures fall ~14% across all four generative arms
(276-282 vs 322), which is where the SR gain originates.

CAVEAT: the diffusion heads condition on (s_t, z_t) while the hub's MLP P saw
z_t alone, so part of the gain may be richer conditioning rather than the
generative estimator. A z-only-conditioned diffusion control would isolate it.

`diffntp_chunk` DOMINATES the current promotion candidate on all three
metrics at matched frames; the running `emastack-20b` chain uses the mlp head.

## Promotion (2026-08-23): `2026-08-23-emastack-20b`

`ema_h1_ee_wide` (the winner under its post-terminology name) to 20B on the
leaders' exact schedule: 10B random80_adaptive20 + curriculum (std1-3), then
SONIC selection with the 0.5 -> 0.1 landing ramp in sonic1 and pinned 0.1
(sonic2-3). Encoder pinned to the screen winner's file. Jobs 5588664-70.
Reference row: `ln_hold1_sonicreset` @20B 0.9558/22.15/168.15.

## Complete objective decomposition (2026-08-23/24, clean rows, one seed)

All on hold-1 + ee 1.0 + wide 1.0, 2B matched frames. "pred" = the chunk
transition model (NTP); terminology per CONTEXT.md.

| arm | DiffSR | pred | SIGReg | target | SR | L | G |
|---|---|---|---|---|---:|---:|---:|
| `jepa_h1_ee_wide` (hub) | y | y | y | EMA | **0.9060** | **26.98** | **103.6** |
| `jepa_nosig_h1_ee_wide` | y | y | - | EMA | 0.8992 | 27.07 | 107.5 |
| `sg_h1_ee_wide` | y | y | y | stopgrad | 0.8972 | 27.18 | 123.2 |
| `dsrsig_h1_ee_wide` | y | - | y | - | 0.8877 | 27.83 | 124.6 |
| `endpoint_h1_ee_wide` | y | - | - | - | 0.8823 | 27.55 | 117.4 |
| `jepa_ol_h1_ee_wide` | y | y | y | online | 0.8655 | 28.29 | 128.6 |
| `lejepa_sg_h1_ee_wide` | - | y | y | stopgrad | 0.8384 | 33.75 | **104.8** |
| `lejepa_h1_ee_wide` | - | y | y | online | 0.7048 | 47.66 | 568.9 |

Two clean attributions:

1. **The stop-gradient is what makes prediction work at all.** Adding it to
   pure JEPA moves 0.7048 -> 0.8384 SR, 47.66 -> 33.75 L, and 568.9 -> 104.8 G
   (-82%). The 2026-08-23 shortcut diagnosis (pred/copy 1e-4 at rank 149) was
   the right cause: co-adaptation, not collapse. The EMA lag adds a further
   +0.009 SR on top of the stop-grad (hub vs `sg_h1_ee_wide`), so asymmetry is
   the bulk of the EMA trick's value and lag is the remainder.
2. **Global drift and local tracking have different sources**, but the
   success-only metrics must be matched first: MPJPE is averaged over
   surviving clips, and these arms survive different sets (hub 3711/4096,
   `lejepa_sg` 3434), so a weaker arm's average covers an easier subset. On
   the 2,768 clips ALL eight arms survived:

   | arm | L(own -> common) | G(own -> common) |
   |---|---|---|
   | hub | 27.16 -> 23.78 | 104.7 -> 77.95 |
   | `sg_h1_ee_wide` | 27.16 -> 24.22 | 125.0 -> 94.46 |
   | `dsrsig_h1_ee_wide` | 28.07 -> 25.15 | 129.0 -> 103.02 |
   | `jepa_ol_h1_ee_wide` | 28.45 -> 26.13 | 132.2 -> 112.52 |
   | `lejepa_sg_h1_ee_wide` | 33.88 -> 31.16 | 108.5 -> 91.87 |
   | `lejepa_h1_ee_wide` | 47.28 -> 46.20 | 550.9 -> 555.64 |

   DiffSR endpoint denoising owns LOCAL tracking: matched one-variable
   contrasts give -22% (stopgrad pair 31.16 -> 24.22) and -43% (online pair
   46.20 -> 26.13).
   The EMA-lagged prediction term owns GLOBAL drift: `dsrsig` -> hub is
   103.0 -> 77.95 (-24%), while `dsrsig` -> `sg` (stopgrad prediction) is only
   103.0 -> 94.5 (-8%, inside noise). Stop-grad prevents the pathological case
   (555.6 -> 91.87) rather than supplying the gain.
   CORRECTION (2026-08-24): an earlier version of this section claimed
   `lejepa_sg` matched the hub on G (104.8 vs 103.6). That was the selection
   artifact above; on matched clips the hub is 18% better.

Round-3 interface/context cells (same protocol):

Matched on the 3,198 clips all four arms survived (own -> matched):

| arm | SR | L | G | reading |
|---|---:|---|---|---|
| hub | 0.9060 | 27.16 -> 24.42 | 104.7 -> 84.09 | reference |
| `qvel_h1_ee_wide` | 0.9070 | 26.07 -> **23.33** | 124.9 -> 101.32 | best matched L (-4.5%); G +20% — a real L<->G trade, unresolved |
| `trip_h1_ee_wide` | 0.9004 | 27.53 -> 25.09 | 133.3 -> 113.38 | loses on all three; matched G gap +35% (wider than unmatched) |
| `trip_qvel_h1_ee_wide` | 0.7891 | 34.24 -> 33.79 | 136.3 -> 133.54 | clear failure: -0.117 SR, +38% L, +59% G |

TRIPLET AXIS REFUTED: two independent cells, both negative, mechanism
explained by the copy gate (`pred_over_copy` ~0.32-0.34 for both triplet arms
vs 0.145 for the hub — the predictor succeeds more, the interface degrades).
Do not spend further budget on `jepa_context_chunks=1` without a new reason.

Unifying result: predictor success is anti-correlated with tracker quality.
`pred_over_copy` 0.145 (hub, best) -> 0.317-0.343 (triplet arms) -> ~0 with a
shortcut (lejepa). An easier pretext task yields a worse command interface.

Convergence caveat: no 2B arm is converged. Training MPJPE-L slope over
1.5B->2.0B is still negative for every arm except the hub (flat), so this
screen ranks arms mid-flight.

## Pipeline

```bash
./smoke.sh        # local wiring qualification (exercises dyn + reward args)
./plan_all.sh     # resolve every plan offline at its declared submit_seed
./submit.sh <arm> <seed>   # plan one arm; prints the submit --confirm line
./mirror.sh       # pull checkpoints off ICE (per-arm submit_seed)
./eval.sh         # milestone + clean + robust rows
./eval.sh --report
```

SUBMISSION ORDER: submit `hold1_dyn_s1` only after `hold1` seed 1's pretrain
stage COMPLETES — its encoder path cannot exist before then and is therefore
not in `require_container_paths`. Every other arm submits in any order.

Evaluation reads every per-arm field (width, hold, phase, macro terms,
encoder file, encoder source) back out of `campaign.yaml`. Dyn arms score
with `--skill_encoder_source checkpoint`: their encoder diverges during RL
and lives inside the tracker checkpoint.

## Promotion (decided in advance)

The Pareto winner (all three metrics at-or-better vs its parent) goes to 10B
under the standard regime (row against `cont_det_ln_hold1` 0.9368/22.86/168.67),
then 20B under the SONIC-reset continuation recipe (row against
`ln_hold1_sonicreset` 0.9558/22.15/168.15). Q3 of the program — whether dyn
survives contact with SONIC resets — is decided there, not at the screen.

## Status

- 2026-08-22: campaign created. All 11 arms pass `./smoke.sh` (dyn ring and
  reward args exercised); all 11 plans resolve at their declared seeds.
- 2026-08-22 15:08: 10 arms SUBMITTED to ICE (one workspace archive, drift
  recorded): `hold1` s1 5587981-83 (pretrain + lowlevel1/2), `hold1_dyn` s0
  5587984-85, `jepa_h1_dyn` s0 5587986-87, `ctrl_ee1` 5587988-89,
  `ctrl_wide1` 5587990-91, `ctrl_ee_wide` 5587992-93, `ctrl_bodyglobal`
  5587994-95, `jepa_h1_ee1` 5587996-97, `jepa_h1_wide1` 5587998-99,
  `jepa_h1_ee_wide` 5588000-01. `hold1_dyn_s1` gate cleared the same
  day: pretrain 5587981 COMPLETED (encoder verified on ICE), submitted as
  5588019-20.
- 2026-08-22 21:04: five objective-mechanism arms added (see the square
  above), all smoke-passed, SUBMITTED to ICE seed 0:
  `endpoint_h1_ee_wide` 5588169-71, `dsrsig_h1_ee_wide` 5588172-74,
  `lejepa_h1_ee_wide` 5588175-77, `jepa_ol_h1_ee_wide` 5588178-80,
  `jepa_nosig_h1_ee_wide` 5588181-83 (pretrain -> lowlevel1 -> lowlevel2). RLOpt gains
  `jepa_ntp_coeff` (submodule edit, test added, 175 rlopt tests pass).
- 2026-08-22 16:31: arm 12 `ctrl_asymcritic` added by user decision — pure
  config on the v2 interface (`critic_components`; no code change, the
  tracking_env knob idea was superseded by the existing v2 knob). Smoked
  (`joint_qpos` dropped: mutually exclusive with `joint_qpos_qvel`),
  submitted as 5588028-29. All 12 arms in flight. Nothing measured.
