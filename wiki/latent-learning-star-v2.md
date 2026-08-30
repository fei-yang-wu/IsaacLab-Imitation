# Latent-Learning Star v2: the ablation rebased on `diffntp_chunk`

Design page for the second full command-interface star. The first star
([Interface Ablation Study](interface-ablation-study.md), 72 arms) measured
every axis against a hub whose encoder was trained to denoise one endpoint
state and whose code was held for 10 control steps. That hub is no longer the
best interface we have, so every one of its main effects is measured against
a control that four later campaigns beat.

This page rebases the whole star on `diffntp_chunk_h1_ee_wide`.

**Status: design. No v2 arm is submitted.** Rows marked MEASURED already exist
because they were trained inside `2026-08-22-pareto-stack` at the v2 hub's
cell; rows marked NEW are not trained.

---

## 1. Why rebase instead of extend

The v1 star answers "which single field should change from `ctrl`". Every
answer is conditional on `ctrl`, and `ctrl` has since been beaten on all three
canonical metrics by a margin far outside the noise band:

| hub | objective | hold | SR | MPJPE-L | MPJPE-G |
|---|---|---:|---:|---:|---:|
| v1 `ctrl` | endpoint DiffSR | 10 | 0.9023 | 24.49 | 212.3 |
| v2 `diffntp_chunk_h1_ee_wide` | generative next chunk | 1 | 0.9163 | 24.07 | **84.69** |

A 2.5x difference in global error is not a perturbation of the same operating
point. Three v1 findings are already known to be conditional on the v1 hub,
which is the direct evidence that the rest may be too:

- **The objective ordering inverts.** `obj_jepa_ntp` was the WORST converged
  objective on success rate at the v1 hub (0.8438). Its descendants at hold 1
  with the generative head are the best arms in the program.
- **The hold effect was measured on one objective only.** Global error fell
  29% from hold 10 to hold 1 under the endpoint objective. The v2 hub is
  already at hold 1, and whether a chunk-trained code can be held longer than
  an endpoint-trained one is unmeasured.
- **Width was a null and quantization was a cost** at a hub whose loss had no
  marginal regularizer. The v2 hub carries SIGReg, whose whole function is
  shaping the code marginal, so both verdicts are re-openable.

## 2. The v2 hub

| field | v1 `ctrl` | v2 hub | override that sets it |
|---|---|---|---|
| objective | endpoint DiffSR | DiffSR endpoint + generative next chunk + SIGReg | `--transition_objective jepa_ntp --jepa_loss sigreg_ebm --jepa_ntp_head diff_chunk` |
| latent mode | deterministic continuous | same | `--latent_mode deterministic` |
| code width | 256 (command 258) | same | `--z_dim 256` |
| encoder LayerNorm | on | on | `--encoder_layer_norm` |
| macro state | `root_qpos`, 380 values | same | `expert_macro_state_terms=[expert_motion_qpos,...]` |
| frame stride | 1 | same | `expert_macro_frame_stride=1` |
| horizon | 10 | same | `--horizon_steps 10` |
| anchor mode | `robot_heading` | same | `expert_macro_anchor_mode=robot_heading` |
| **hold** | **10** | **1** | `latent_steps_min/max=1`, `code_period=1` |
| phase | `sin_cos` from the hold clock | same (constant at hold 1) | `command_phase_mode=sin_cos` |
| command mode | publish `z` | same | `hl_skill_command_mode=z` |
| encoder during RL | frozen | frozen | `hl_skill_finetune_enabled=false` |
| **rewards** | base | **base + `motion_ee_pos` 1.0 + `motion_global_anchor_pos_wide` 1.0** | two weight overrides |

The hub's pretraining loss is
`DiffSR-denoise(s[t+10]) + diff_chunk(s[t+11..t+20] | s_t, z_t) + SIGReg(z)`.
Every target is data. The objective carries no self-target, so no EMA or
stop-gradient asymmetry appears in it at all.

**The reward pair is part of the hub, not an axis.** `motion_ee_pos` and
`motion_global_anchor_pos_wide` were screened in `2026-08-22-pareto-stack` and
are inert at weight 0.0 in the v1 hub. They are held fixed here so the star
measures latent learning, not reward shaping. Their own screen rows stay in the
pareto-stack README.

## 3. Axes and the arm census

Every arm changes exactly ONE field of section 2 unless its row says
otherwise. `2B` frames, seed 0, checkpoint every 250M frames.

Legend: **MEASURED** = trained and scored at this cell; **TRAINED** = trained,
no scored row; **NEW** = not trained.

### Axis A — what the encoder is trained to predict

The family spine. This is the headline table: predictive versus reconstruction
versus posterior, with the predictive family resolved into its estimator forms.

**A1. Predictive, generative (the hub's family).** The prediction term is a
diffusion head; the axis is what it generates.

| arm | target | override | status | SR | L | G |
|---|---|---|---|---:|---:|---:|
| `diffntp_chunk` (hub) | next chunk of data, executed frame | `--jepa_ntp_head diff_chunk` | MEASURED | 0.9163 | 24.07 | **84.69** |
| `diffntp_token` | next token, EMA target | `--jepa_ntp_head diff_token` | MEASURED | 0.9121 | 24.33 | 84.97 |
| `diffntp_pair` | joint (endpoint state, next token) | `--jepa_ntp_head diff_pair` | MEASURED | 0.9182 | 24.42 | 97.10 |
| `diffntp_chunkra` | next chunk, target re-anchored | `--jepa_ntp_chunk_anchor next` | MEASURED | 0.9194 | 24.74 | 106.45 |
| `diffntp_chunktok` | next chunk + additive EMA token term | `--jepa_token_pred_coeff 1.0` | MEASURED | 0.9189 | 23.55 | 88.92 |
| `diffntp_merged` | one head over `s[t+10..t+20]`, endpoint term dropped | `--jepa_ntp_chunk_span boundary_next --jepa_endpoint_coeff 0` | TRAINED | | | |
| `diffntp_merged64` | the merged head at 64-D | as above + `--z_dim 64` | MEASURED | 0.9207 | 24.54 | 91.12 |

**A2. Predictive, conditional mean.** The same composite with the diffusion
head replaced by an MLP that regresses the mean, plus its decomposition. This
is the v1 hub's descendant line.

| arm | change | override | status | SR | L | G |
|---|---|---|---|---:|---:|---:|
| `jepa_h1_ee_wide` | MLP predictor on the EMA token | `--jepa_ntp_head mlp` | MEASURED | 0.9060 | 26.99 | 103.61 |
| `endpoint_h1_ee_wide` | prediction term removed entirely | `--transition_objective endpoint` | MEASURED | 0.8823 | 27.55 | 117.42 |
| `dsrsig_h1_ee_wide` | predictor dropped, SIGReg kept | `--jepa_ntp_coeff 0` | MEASURED | 0.8877 | 27.83 | 124.60 |
| `jepa_nosig_h1_ee_wide` | SIGReg removed | `--jepa_sigreg_coeff 0` | MEASURED | 0.8992 | 27.07 | 107.52 |
| `sg_h1_ee_wide` | stop-gradient target | `--jepa_target_encoder_mode stopgrad` | MEASURED | 0.8972 | 27.18 | 123.24 |
| `jepa_ol_h1_ee_wide` | online target | `--jepa_target_encoder_mode online` | MEASURED | 0.8655 | 28.29 | 128.63 |
| `lejepa_ema_h1_ee_wide` | prediction + SIGReg only, EMA | `--jepa_loss sigreg` | MEASURED | 0.8591 | 31.39 | 117.75 |
| `lejepa_sg_h1_ee_wide` | prediction + SIGReg only, stop-grad | + `stopgrad` | MEASURED | 0.8384 | 33.75 | 104.78 |
| `lejepa_h1_ee_wide` | prediction + SIGReg only, online | + `online` | MEASURED | 0.7048 | 47.66 | 568.85 |
| `trip_h1_ee_wide` | predictor reads `cat(z[t-H], z[t])` | `--jepa_context_chunks 1` | MEASURED | 0.9004 | 27.35 | 127.28 |

**A3. Predictive, other targets.** The v1 star's remaining objectives, none of
which has ever been trained at hold 1 with a chunk-aware cell.

| arm | what it predicts | override | status |
|---|---|---|---|
| `v2_endpoint_delta` | the endpoint DELTA rather than the endpoint | `--transition_objective endpoint_delta` | NEW |
| `v2_state_occupancy` | successor-representation occupancy | `--transition_objective state_occupancy` | NEW |
| `v2_semimarkov` | semi-Markov factorization of the transition | `--transition_objective semimarkov_chain` | NEW |
| `v2_phi_bilinear` | hub target, legacy bilinear successor head | `--diffsr_phi_parameterization bilinear` | NEW |

**A4. Reconstruction.** The v1 star's largest global-error failure (2.0x the
hub) and its two follow-ups. The v2 question is whether the failure is a
property of autoencoding or of the v1 cell.

| arm | decode target | override | status |
|---|---|---|---|
| `v2_recon` | the exact 380-value input window | `--transition_objective reconstruction` | NEW |
| `v2_recon_endpoint` | the held-out endpoint | + `--reconstruction_target endpoint` | NEW |
| `v2_recon_full_window` | all ten future slots including the hidden endpoint | + `--reconstruction_target full_window` | NEW |

**A5. Contrastive.** `--jepa_loss infonce`. EXCLUDED by user decision
2026-08-26 (v1 row 0.8906 / 29.87 / 201.04). Listed so the family table can
say why the cell is empty; one arm reinstates it.

**A6. Posterior — the code learned during RL, no pretraining at all.** The
v1 route's nine arms were 2.1-2.5x worse globally than the v1 hub with no
escape. Rebased, the grid is quantizer x training signal at the v2 cell.

| arm | training signal | quantizer | status |
|---|---|---|---|
| `v2_post_recon_ae` | reconstruction only | continuous 256 | NEW |
| `v2_post_pg_ae` | policy gradient only | continuous 256 | NEW |
| `v2_post_pgrecon_ae` | both | continuous 256 | NEW |
| `v2_post_recon_fsq` | reconstruction only | FSQ | NEW |
| `v2_post_pg_fsq` | policy gradient only | FSQ | NEW |
| `v2_post_pgrecon_fsq` | both | FSQ | NEW |
| `v2_post_recon_vq` | reconstruction only | EMA VQ 512 | NEW |
| `v2_post_pg_vq` | policy gradient only | EMA VQ 512 | NEW |
| `v2_post_pgrecon_vq` | both | EMA VQ 512 | NEW |

These arms carry no pretrain stage and score with
`--skill_encoder_source checkpoint`. If the axis must be cut, keep the three
continuous cells: they are the family claim, and the quantizer sub-grid
repeats Axis B inside a route already known to lose.

### Axis B — the shape of the code: continuous versus discrete

Every cell is config-only: `LATENT_MODES` in `RLOpt/rlopt/agent/hl_skill_encoder.py`
carries all eight quantizers and no guard couples them to the objective, so
each composes with the hub's diffusion head unchanged.

**The v2 question is sharper than v1's.** Under the v1 hub a single-codebook
quantizer collapsed to success rate 0.0000. The v2 hub carries SIGReg, whose
function is to hold the code marginal near `N(0, I)` — an anti-collapse
mechanism the v1 hub did not have. Re-running the dead cells is therefore a
new measurement, not a repeat.

| arm | code | override | status | v1 row (old hub, 256-clip board) |
|---|---|---|---|---|
| hub | continuous 256 | — | MEASURED | 0.9102 / 23.44 / 199.87 |
| `v2_c64` | continuous 64 | `--z_dim 64`, command 66 | NEW | 0.9180 / 22.80 / 204.74 |
| `v2_c128` | continuous 128 | `--z_dim 128`, command 130 | NEW | 0.9102 / 23.39 / 205.03 |
| `v2_gaussian` | Gaussian posterior + KL | `--latent_mode gaussian` | NEW | 0.9023 / 27.68 / 212.18 |
| `v2_fsq64` | FSQ 64 coords x 32 levels (SONIC's space) | `--latent_mode sonic_fsq --sonic_fsq_levels 32 x64` | NEW | 0.9023 / 28.86 / 177.70 |
| `v2_fsq32` | FSQ 32 x 32 — width at fixed levels | `--sonic_fsq_levels 32 x32` | NEW | 0.8945 / 31.99 / 206.43 |
| `v2_fsq16` | FSQ 16 x 32 — the narrow end | `--sonic_fsq_levels 32 x16` | NEW | 0.6836 / 44.33 / 242.30 |
| `v2_fsq64_l8` | FSQ 64 x 8 — levels at fixed width | `--sonic_fsq_levels 8 x64` | NEW | 0.8672 / 33.82 / 193.73 |
| `v2_gumbel_multicat` | 64 groups x 32 categories, Gumbel | `--latent_mode gumbel_multicat` | NEW | 0.7031 / 46.45 / 340.05 |
| `v2_categorical` | the same 64 x 32, hard straight-through | `--latent_mode categorical` | NEW | 0.5703 / 54.77 / 689.24 |
| `v2_gumbel` | one Gumbel codebook, K=512 | `--latent_mode gumbel --gumbel_codebook_size 512` | NEW | 0.3750 / 68.45 / 1364.72 |
| `v2_vq_ema` | one EMA VQ codebook, K=512 | `--latent_mode vq --vq_codebook_size 512` | NEW | **0.0000** |
| `v2_no_ln` | encoder LayerNorm off | `--no_encoder_layer_norm` | NEW | 0.8945 / 22.48 / 222.26 |

`v2_no_ln` has a second consumer: the linear-closure program needs a convex
`z` domain and therefore needs the final LayerNorm gone
([linear-closure-problem-statement.md](linear-closure-problem-statement.md)).

`v2_gaussian` is the pretrained-stochastic cell of Axis A's family table and
the stochastic cell of Axis B. One arm, cited in both.

### Axis C — what the encoder reads, and how wide the window is

| arm | change | override | status | v1 row |
|---|---|---|---|---|
| hub | 10 consecutive frames, stride 1, `root_qpos` 380 | — | MEASURED | — |
| `v2_fullbody670` | adds 29 reference joint velocities (380 -> 670) | `expert_macro_state_terms=[expert_motion,...]` | NEW | 0.8984 / 26.92 / 225.79 |
| `v2_stride5` | 10 frames spaced 5 apart — SONIC's 0.9 s window | `expert_macro_frame_stride=5` | NEW | 0.6992 / 45.62 / 446.67 |
| `v2_window_full` | window includes the endpoint instead of hiding it | `--encoder_window_mode full` | NEW | 0.9102 / 23.20 / 224.25 |
| `v2_anchor_robot` | macro anchor in the live robot frame | `expert_macro_anchor_mode=robot` | NEW | 0.9062 / 24.64 / 253.86 |
| `v2_anchor_expert` | macro anchor in the expert's heading frame | `expert_macro_anchor_mode=expert_heading` | NEW | 0.8945 / 30.44 / 350.25 |
| `v2_h5` | horizon 5 | `--horizon_steps 5` | NEW | — |
| `v2_h20` | horizon 20 | `--horizon_steps 20` | NEW | — |

`qvel_h1_ee_wide` (MEASURED, 0.9070 / 26.02 / 121.54) is the full-body cell on
the CONDITIONAL-MEAN hub, not on the v2 hub; it belongs to Axis A2's line and
does not substitute for `v2_fullbody670`.

**CONFOUND ON RECORD.** `--horizon_steps` sets the encoder's input window AND
the chunk head's target span together, so `v2_h5` and `v2_h20` move two things
at once and are a joint window-size axis, not an input-size axis. The
input-only causal read is the `encoder_window_mode=suffixN` series already
running locally (the endpoint-collapse Tier B probe). A target-only read does
not exist and would need a new RLOpt knob.

### Axis D — how the code is published

The hub is at hold 1, so this axis runs upward. No arm here changes the
encoder: every one reuses the hub's `encoder/checkpoints/latest.pt` and skips
the pretrain stage, which also removes encoder-initialization variance.

| arm | change | override | status | v1 row |
|---|---|---|---|---|
| hub | hold 1 | — | MEASURED | 0.9180 / 25.76 / 140.94 (`use_hold1`) |
| `v2_hold5` | publish every 5 control steps | `latent_steps_min/max=5`, `code_period=5` | NEW | 0.9102 / 24.79 / 146.92 |
| `v2_hold10` | publish every 10 control steps | `=10` | NEW | 0.9102 / 23.44 / 199.87 (`ctrl`) |
| `v2_phase_none_h10` | hold 10 with the sin/cos slot clock dropped | `command_phase_mode=none` at hold 10 | NEW | 0.4141 / 64.82 / 1344.94 |
| `v2_live_phase` | hold 1 with a live clock instead of the hold clock | `command_phase_source=episode`, `period=10` | NEW | 0.9102 / 25.52 / 143.53 |

`v2_phase_none_h10` must sit at hold 10: at hold 1 the slot clock is constant
and dropping it is a null by construction (`hold1_live_phase` measured that
null at v1). Pairing it with `v2_hold10` keeps it one variable.

**Interaction probes** (the v1 star kept two; the v2 versions matter because
Axis B may move):

| arm | cell | status |
|---|---|---|
| `v2_ix_fsq64_hold5` | FSQ 64 x 32 at hold 5 | NEW |
| `v2_ix_fsq64_hold10` | FSQ 64 x 32 at hold 10 | NEW |

### Axis E — is the encoder frozen during RL

| arm | change | override | status |
|---|---|---|---|
| hub | frozen | `hl_skill_finetune_enabled=false` | MEASURED |
| `v2_dyn` | online achieved-ring dynamics finetune | the `dyn_block` | NEW |
| Axis A6 | no pretraining at all | posterior route | NEW |

The three rows are one ordered axis — frozen, finetuned, learned from scratch
in RL — and A6 is its far end. `jepa_h1_dyn` (0.8860 / 29.73 / 119.62) and
`hold1_dyn` (0.8850 / 27.76 / 181.37) are MEASURED but sit at the base-reward
cell, so neither is a v2 row.

### Axis F — loss composition inside the hub, and the seed

| arm | change | override | status |
|---|---|---|---|
| `diffntp_chunk_nosig` | SIGReg off on the hub | `--jepa_sigreg_coeff 0` | in `campaign.yaml`, NOT submitted |
| `diffntp_chunk_s1` | seed repeat of the hub | seed 1 | in `campaign.yaml`, NOT submitted |

`diffntp_chunk_nosig` is NOT a repeat of `jepa_nosig`: there SIGReg was pinning
the marginal that a self-target depended on, and the hub has no self-target.
It is also the control for every Axis B discrete cell, since the SIGReg
anti-collapse argument above is exactly what it tests.

## 4. Census and cost

| axis | MEASURED | TRAINED, unscored | NEW |
|---|---:|---:|---:|
| A1 generative heads | 6 | 1 | 0 |
| A2 conditional mean and its decomposition | 10 | 0 | 0 |
| A3 other predictive targets | 0 | 0 | 4 |
| A4 reconstruction | 0 | 0 | 3 |
| A5 contrastive | 0 | 0 | 1 (excluded) |
| A6 posterior route | 0 | 0 | 9 |
| B code shape | 0 | 0 | 12 |
| C encoder input and window | 0 | 0 | 7 |
| D publication cadence | 0 | 0 | 6 |
| E encoder frozen or not | 0 | 0 | 1 |
| F loss detail and seed | 0 | 0 | 2 |
| **total** | **16** | **1** | **45** |

62 rows, of which 45 need training. The v1 star was 72.

Stage cost: an arm that changes the encoder needs `pretrain -> lowlevel1 ->
lowlevel2`; an arm that changes only how the code is used needs the two
lowlevel segments; a posterior arm needs the two lowlevel segments and no
pretrain. Of the 45: 27 full chains, 9 posterior (lowlevel only), 7 lowlevel
only, 2 full chains already written. That is about 118 ICE segments of up to
15:59 each on one H200.

**Cut lines, cheapest first**, if the whole star is too large:

1. Drop the six quantizer cells of A6, keep the three continuous ones (-6).
2. Drop `v2_fsq16`, `v2_gumbel`, `v2_categorical` — the v1 rows are 0.2-0.9 SR
   failures and the SIGReg-rescue argument is testable on `v2_vq_ema` alone
   (-3).
3. Drop `v2_c128` — width is a two-point axis with 64 and 256 (-1).
4. Drop `v2_live_phase` and `v2_anchor_robot` — both v1 nulls (-2).

That floor is 33 new arms and keeps every family, code-shape, window, and
cadence claim.

## 5. Protocol

Identical to the v1 star ([Interface Ablation Study](interface-ablation-study.md)
sections 3 and 4) except where stated: `Isaac-Imitation-G1-v2`, Newton MJWarp,
the 129,785-clip BONES-SEED set resident in RAM, 16,384 envs x 24 rollout
steps, `gamma 0.97`, 2B frames, checkpoint every 250M, encoder pretrain 50,000
updates at batch 8,192.

**Boards.** Score every arm on `bones_testbed4096_v1` clean at 2B — the
canonical board, and what every v2 number on this page uses. Keep the
256-clip `bones_milestone_testbed256_v1` milestone series for the budget axis.
The two boards disagree by more than a rounding step (`diffntp_token` reads
0.9258 / 24.11 / 84.20 on 256 clips and 0.9121 / 24.33 / 84.97 on 4,096), so
never mix them in one table.

**Matched success sets.** MPJPE is success-only, so any table that compares
L or G across rows must recompute both on the rows' common success set, then
freeze and name that set. Adding a row changes every number in the table.

**Noise band.** 0.016 success rate, 1.3% MPJPE-L, 6.7% MPJPE-G, from the v1
re-scoring experiment. One seed per arm except the declared repeats; every
within-band ordering is unresolved.

## 6. Gates before submission

1. **`feet_acc` pin.** `G1SonicRewardsCfg` corrected `feet_acc` from -2.5e-7 to
   -2.5e-6 in place on 2026-08-28. Every MEASURED row above trained on the weak
   value. Every v2 arm must pin `env.rewards.feet_acc.weight=-2.5e-7` or the
   whole star must be re-trained at the corrected weight; the pin is far
   cheaper. Verify which side of the fix `diffntp_chunktok` and
   `diffntp_merged64` trained on before citing them tightly.
2. **The rot6d convention.** The data plane emits the interleaved 6-D rotation
   layout and RLOpt's `_rot6d_to_matrix` parses two concatenated columns
   (found 2026-08-29, unfixed). Any target that routes through
   `_reanchor_heading_frames` is a deterministically distorted view: that is
   `diffntp_chunkra`, and the EMA token target of every A2 arm and of
   `diffntp_token`, `diffntp_pair`, `diffntp_chunktok`. **The v2 hub is not
   affected** (context 0, executed anchor), and neither is any A3, A4, B, C, D
   or E arm that keeps the hub's target. Decide the convention before
   re-running the affected rows.
3. **Score `diffntp_merged`.** It trained (jobs 5594018-20) but only its
   encoder is mirrored locally; the tracker tree is on ICE. Until it has a row,
   `diffntp_merged64`'s success-rate lead cannot be attributed, because that
   arm differs from the hub by both the merged span and the 64-D width.
4. **Quantizer smoke.** Each Axis B mode composes with `diff_chunk` by
   inspection, not by execution. One short pretrain per quantizer family (FSQ,
   Gumbel, categorical, VQ, Gaussian) before the batch, checking finite loss
   and `z_effective_rank`, costs minutes and catches a dead cell before a
   GPU-day.

## 7. What this star cannot answer

- **Interactions.** A star measures main effects. The only interaction probes
  are the two `ix_fsq64_hold*` cells; everything else needs the
  `interface-combos` pattern.
- **Robustness.** Domain randomization off, no push. A robust row per leader is
  a separate pass.
- **Convergence.** No v1 arm was converged at 2B (MPJPE-L slope still negative
  for every arm except the hub), and no v2 arm will be either. The star ranks
  arms mid-flight, deliberately, so that no arm is promoted on the outcome.
- **Smoothness.** Excluded by user decision 2026-08-30; the smoothness program
  runs on its own campaigns.
