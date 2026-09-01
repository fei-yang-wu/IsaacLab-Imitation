# Latent-Learning Star v2: the ablation rebased on the merged 64-D hub

Design page for the second command-interface star. The first star
([Interface Ablation Study](interface-ablation-study.md), 72 arms) measured
every axis against a hub whose encoder was trained to denoise one endpoint
state and whose code was held for 10 control steps. Four later campaigns beat
that hub, so every one of its main effects is measured against a control that
is no longer competitive.

**Status: SUBMITTED 2026-08-31.** 38 of 41 arms are on ICE as 110 jobs
(`hub` 5600005-07, the rest 5600008-5600122); Group 5's three arms are held
until the hub's pretrain writes its encoder. Campaign
`experiments/campaigns/2026-08-30-latent-star-v2/`, W&B project
`g1-bs-ablation`, group `latent-star-v2`.

The regime moved at launch by user instruction: the SONIC reset ramp replaces
`random80_adaptive20` and the budget is 5B rather than 2B, which at a 200M
checkpoint interval gives 25 curve points. Section 2 below is the submitted
configuration.

Three things below were changed by what the smoke and the first scoring pass
found — the Group 2b triplet head, the Group 6 placement, and the
`diffntp_merged` row — and are marked where they appear.

---

## 1. Why rebase instead of extend

Three v1 findings are already known to be conditional on the v1 hub, which is
the direct evidence that the rest are too:

- **The objective ordering inverts.** `obj_jepa_ntp` was the worst converged
  objective on success rate at the v1 hub (0.8438). Its descendants at hold 1
  with a generative head are the best arms in the program.
- **The hold effect was measured on one objective.** Global error fell 29% from
  hold 10 to hold 1 under the endpoint objective; whether a chunk-trained code
  can be held longer is unmeasured.
- **Width was a null and quantization was a cost** under a loss with no
  marginal regularizer. The v2 hub carries SIGReg, whose function is shaping
  the code marginal, so both verdicts are re-openable.

## 2. The v2 hub (confirmed 2026-08-30)

`diffntp_merged64`-style: **one** diffusion head denoising `s[t+10..t+20]`,
continuous 64-D code, hold 1, phase channel kept.

| field | value | override |
|---|---|---|
| objective | DiffSR over the merged span + SIGReg | `--transition_objective jepa_ntp --jepa_loss sigreg_ebm --jepa_ntp_head diff_chunk --jepa_ntp_chunk_span boundary_next --jepa_endpoint_coeff 0` |
| latent mode | deterministic, continuous | `--latent_mode deterministic` |
| code width | **64** (command 66 = z + 2-wide sin/cos phase) | `--z_dim 64`, `command_interface.actor.dim=66` |
| encoder LayerNorm | on | `--encoder_layer_norm` |
| macro state | `root_qpos` frame, 380 values | `expert_macro_state_terms=[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b]` |
| frame stride | 1 | `expert_macro_frame_stride=1` |
| horizon (window) | 10 | `--horizon_steps 10` |
| anchor mode | `robot_heading` | `expert_macro_anchor_mode=robot_heading` |
| hold | 1 | `latent_steps_min/max=1`, `code_period=1` |
| phase | `sin_cos` from the hold clock | `command_phase_mode=sin_cos` |
| encoder during RL | frozen | `hl_skill_finetune_enabled=false` |
| rewards | `motion_ee_pos` 1.0, `motion_global_anchor_pos_wide` 1.0, **`action_rate_l2` -0.03** | three weight overrides |
| `feet_acc` | **-2.5e-6** (SONIC parity, in-tree since 2026-08-28) | recipe default |
| reset selection | SONIC, failure share ramped 0.8 -> 0.2 over the first 1B frames | `reference.selection=sonic`, `adaptive_uniform_ratio` 0.8 -> `_final` 0.2, `adaptive_ratio_ramp_frames=1000000000` |
| **environments** | **20,480** x 24 rollout steps = 491,520 frames per batch | `--num_envs 20480` |
| budget | **5B frames** = 10,173 iterations | `max_iterations` |
| **checkpoints** | every **200M** frames = 25 points over 5B | `agent.save_interval=200000000` |
| seed | 0 | |

**Why the merged head and not the two-head `diffntp_chunk`.** At 64-D and
hold 1 the two-head form has never trained: `leader64_h1_nophase` was cancelled
at 0.84B with episode length plateaued at 50-62 and MPJPE-L flat near 51 mm,
while its 256-D control was at 166 / 42.6 mm by 0.17B. Its encoder pretrain was
provably healthy (effective rank 47.9/64, endpoint triplet fully informative),
so the stall was in the tracker. The merged head at the same width and hold
works twice over: `diffntp_merged64_h1_ee_wide` 0.9207 / 24.54 / 91.12 at 2B,
and `merged64_pen_ramp_5b` 0.9543 / 23.67 / 89.33 at 5B with the same -0.03
penalty. The stalled arm also dropped the phase channel, so width and phase are
confounded in that one failure — which is why **the hub keeps the phase channel
(command 66)** and why `diffntp_chunk` (two-head) is a row of Group 2 rather
than the hub.

**Why 64-D.** It width-matches the discrete arms of Group 3 (FSQ 64, VQ 64,
multicat), so continuous-versus-discrete is a clean contrast rather than one
carrying a width confound.

**Environment count.** 20,480 is the measured-safe step up: 24,576 environments
OOM'd the H200 at the Newton graph launch (job 5580202).

## 3. Groups and arms

Each arm changes exactly one field of section 2 unless its row says otherwise.

### Group 1 — reconstruction and posterior (7 arms, all NEW)

The family contrast: is a predictive objective actually better than
autoencoding, and better than learning the code inside RL?

| arm | route | quantizer | training signal |
|---|---|---|---|
| `g1_recon_ae` | pretrained reconstruction | continuous 64 | reconstruction |
| `g1_recon_fsq` | pretrained reconstruction | FSQ | reconstruction |
| `g1_recon_vq` | pretrained reconstruction | VQ-VAE | reconstruction |
| `g1_post_ae` | posterior, learned in RL | continuous 64 | policy gradient |
| `g1_post_pgrecon_ae` | posterior, learned in RL | continuous 64 | policy gradient + reconstruction |
| `g1_post_pgrecon_fsq` | posterior, learned in RL | FSQ | policy gradient + reconstruction |
| `g1_post_pgrecon_vq` | posterior, learned in RL | VQ-VAE | policy gradient + reconstruction |

The four posterior arms carry no pretrain stage; their encoder lives inside the
tracker checkpoint and evaluation restores it with
`--skill_encoder_source checkpoint`.

v1 evidence, for expectation only: reconstruction matched the v1 hub on success
rate and was 2.0x worse globally; the nine posterior arms were 2.1-2.5x worse
globally with no arm escaping.

### Group 2 — what the latent factorization predicts (16 arms, all NEW)

**2a. The target.** What the head is asked to produce, at a fixed head form.

| arm | target | override | status |
|---|---|---|---|
| `g2_endpoint` | the immediate next state only, not the chunk | `--transition_objective endpoint` | NEW |
| `g2_twohead` | separate endpoint head + next-chunk head (the round-4 winner's form) | `--jepa_ntp_head diff_chunk` (default span, `jepa_endpoint_coeff` 1.0) | NEW |
| `g2_delta` | the endpoint DELTA | `--transition_objective endpoint_delta` | NEW |
| `g2_state_occupancy` | successor-representation occupancy | `--transition_objective state_occupancy` | NEW |
| `g2_semimarkov` | semi-Markov factorization | `--transition_objective semimarkov_chain` | NEW |
| `g2_phi_bilinear` | hub target, legacy bilinear successor head | `--diffsr_phi_parameterization bilinear` | NEW |

**2b. The predictor form, re-trained at the v2 hub (10 arms).**
These rows exist in the v1 regime but are four fields from this hub, so the
user's decision (2026-08-30) is to re-train them rather than cite them across
regimes. They resolve the estimator question — generative diffusion head
against a conditional mean, and what the conditional mean's asymmetry is worth.

| arm | change from the hub's loss | override | v1-regime row |
|---|---|---|---|
| `g2_mlp` | conditional-mean MLP predictor on the EMA token | `--jepa_ntp_head mlp` | 0.9060 / 26.99 / 103.61 |
| `g2_dsrsig` | predictor dropped, SIGReg kept | `--jepa_ntp_coeff 0` | 0.8877 / 27.83 / 124.60 |
| `g2_nosig` | SIGReg removed | `--jepa_sigreg_coeff 0` | 0.8992 / 27.07 / 107.52 |
| `g2_sg` | stop-gradient target | `--jepa_target_encoder_mode stopgrad` | 0.8972 / 27.18 / 123.24 |
| `g2_online` | online target | `--jepa_target_encoder_mode online` | 0.8655 / 28.29 / 128.63 |
| `g2_lejepa_ema` | prediction + SIGReg only, EMA target | `--jepa_loss sigreg` | 0.8591 / 31.39 / 117.75 |
| `g2_lejepa_sg` | prediction + SIGReg only, stop-grad | + `stopgrad` | 0.8384 / 33.75 / 104.78 |
| `g2_lejepa_online` | prediction + SIGReg only, online | + `online` | 0.7048 / 47.66 / 568.85 |
| `g2_trip` | predictor reads `cat(z[t-H], z[t])`, MLP head required | `--jepa_ntp_head mlp --jepa_context_chunks 1` | 0.9004 / 27.35 / 127.28 |
| `g2_token` | generative next-TOKEN instead of next-chunk | `--jepa_ntp_head diff_token` | 0.9121 / 24.33 / 84.97 |

The v1-regime column is context for what to expect, not a row of this table.
Several of these arms carry an EMA token target and therefore route through
the rot6d mismatch — see gate 3.

### Group 3 — encoder space design (6 arms, all NEW)

Config-only: `LATENT_MODES` in `RLOpt/rlopt/agent/hl_skill_encoder.py` carries
all eight quantizers and no guard couples them to the objective.

| arm | code | override |
|---|---|---|
| `g3_cont128` | continuous 128 | `--z_dim 128`, command 130 |
| `g3_cont256` | continuous 256 | `--z_dim 256`, command 258 |
| `g3_vq64` | VQ, 64-D | `--latent_mode vq` |
| `g3_fsq64` | FSQ 64 coords x 32 levels (SONIC's space) | `--latent_mode sonic_fsq --sonic_fsq_levels 32 x64` |
| `g3_multicat` | 64 groups x 32 categories, hard straight-through | `--latent_mode categorical` |
| `g3_multicat_gumbel` | the same 64 x 32 with a Gumbel-softmax relaxation | `--latent_mode gumbel_multicat` |

The v2 question is sharper than v1's, and the v1 evidence is more specific than
"discrete costs something". A single codebook collapsed to success rate 0.0000
when PRETRAINED and frozen (`bn_vq_ema`), but the same quantizer learned during
RL did not — `post_recon_vq` 0.8945, `post_pgrecon_vq` 0.8867, `post_pg_vq`
0.8828, the first being the best success rate of that nine-arm grid. What
failed was the frozen pretrained codebook, not vector quantization. The v1 hub
had no marginal regularizer; this one carries SIGReg, which holds the code
marginal near `N(0, I)` — precisely the missing anti-collapse mechanism. So
`g3_vq64` is a new measurement, and `g1_post_pgrecon_vq` is the working-route
partner it reads against.

### Group 4 — encoder input and window (7 arms, all NEW)

| arm | change | override |
|---|---|---|
| `g4_fullbody670` | adds 29 reference joint velocities (380 -> 670) | `expert_macro_state_terms=[expert_motion,...]` |
| `g4_stride5` | 10 frames spaced 5 apart — SONIC's 0.9 s window | `expert_macro_frame_stride=5` |
| `g4_window_full` | window includes the endpoint instead of hiding it | `--encoder_window_mode full` |
| `g4_anchor_robot` | macro anchor in the live robot frame | `expert_macro_anchor_mode=robot` |
| `g4_anchor_expert` | macro anchor in the expert's heading frame | `expert_macro_anchor_mode=expert_heading` |
| `g4_h5` | horizon 5 | `--horizon_steps 5` |
| `g4_h20` | horizon 20 | `--horizon_steps 20` |

**CONFOUND ON RECORD.** `--horizon_steps` sets the encoder's input window AND
the merged head's target span together, so `g4_h5` and `g4_h20` move two things
at once. The input-only causal read is the `encoder_window_mode=suffixN` series
running locally (endpoint-collapse Tier B). A target-only read needs a new
RLOpt knob and does not exist.

### Group 5 — publication cadence (3 arms, all NEW, no pretrain)

Every arm here reuses the hub's `encoder/checkpoints/latest.pt` and skips the
pretrain stage, which also removes encoder-initialization variance.

| arm | change | override |
|---|---|---|
| `g5_hold5` | publish every 5 control steps | `latent_steps_min/max=5`, `code_period=5` |
| `g5_hold10` | publish every 10 control steps | `=10` |
| `g5_phase_none_h10` | hold 10 with the sin/cos slot clock dropped | `command_phase_mode=none` at hold 10, command 64 |

`g5_phase_none_h10` must sit at hold 10 and is therefore paired with
`g5_hold10`, not with the hub: at hold 1 the slot clock is constant and
dropping it is a null by construction. This pairing also finally separates
width from phase in the `leader64_h1_nophase` failure.

### Group 6 — encoder frozen or adapted (1 arm, NEW, own pretrain)

| arm | change | override |
|---|---|---|
| `g6_dyn` | online achieved-ring dynamics finetune, on the MLP cell | `--jepa_ntp_head mlp` + the `dyn_block` |

**CORRECTED 2026-08-30 by the wiring smoke: this axis is NOT measured at the
hub.** RLOpt's online finetune requires `jepa_ntp_head='mlp'` and refuses every
diffusion head (`hl_skill_diffsr.py`, the "Online jepa_ntp finetuning supports
only the chunk-pair EMA recipe" guard), so dyn on the merged-head hub cannot
run without an RLOpt change. `g6_dyn` therefore sits on the conditional-mean
cell with `g2_mlp` as its control, and needs its own pretrain because no other
arm leaves a frozen MLP-head encoder behind. The ordered axis now reads
frozen-MLP (`g2_mlp`) → finetuned-MLP (`g6_dyn`) → learned-from-scratch-in-RL
(Group 1's posterior arms). Extending the finetune to the diffusion heads is
the work that would restore the hub reading.

## 4. Census, cost, and the results artifacts

| group | arms | pretrain needed |
|---|---:|---:|
| hub | 1 | yes |
| 1 reconstruction and posterior | 7 | 3 of 7 |
| 2a factorization targets | 6 | yes |
| 2b predictor form | 10 | yes |
| 3 encoder space | 6 | yes |
| 4 input and window | 7 | yes |
| 5 cadence | 3 | no |
| 6 encoder adaptation | 1 | yes |
| **total** | **41** | 34 |

34 full `pretrain -> lowlevel1 -> lowlevel2` chains plus 7 lowlevel-only pairs
is 116 ICE segments of up to 15:59 on one H200.

**Two artifacts, from the same runs.**

1. **The table.** Every arm's 2B row on `bones_testbed4096_v1` clean: success
   rate, success-only micro MPJPE-L, MPJPE-G. Per the standing directive each
   table also carries the same-board `sonic_v1_1` row.
2. **Convergence figures.** One line plot per metric, x-axis environment
   frames, one line per arm, a point every **200M** frames — 10 points per arm
   over the 2B budget. This is why the hub sets
   `agent.save_interval=200000000` rather than the v1 star's 250M.

The 200M grid and the v1 star's 250M grid coincide only at 1.0B and 2.0B, so a
figure that draws both families shares an axis, not sample points.

## 5. The retained v1-regime rows, and what they can and cannot support

The user's decision (2026-08-30) is to leave the 16 already-measured
`2026-08-22-pareto-stack` rows intact rather than re-train them, and to score
their intermediate checkpoints so they have convergence curves too.

**They are a second panel, not rows of the v2 table.** Against the v2 hub they
differ in four fields at once — 256-D instead of 64-D, 16,384 environments
instead of 20,480, `action_rate_l2` 0.0 instead of -0.03, and `feet_acc`
-2.5e-7 instead of -2.5e-6 — so no difference between a retained row and a v2
row is attributable to any one of them. Plot them as their own family with
their regime named in the caption; do not draw a v2 conclusion across the two
panels.

**Their curves are 8 points, not 20.** Those arms trained with
`agent.save_interval=250000000`, so eight checkpoints per arm is all that was
ever written; re-evaluation cannot manufacture the missing twelve. The two
grids coincide exactly at 500M, 1.0B, 1.5B and 2.0B.

**Consequence for Group 2.** The plan was to reuse the JEPA-family rows
(`jepa_h1_ee_wide` and its decomposition: `dsrsig`, `nosig`, `sg`, `ol`,
`lejepa` x3, `trip` — ten arms) rather than re-train them. Under the regime
change they are cross-regime and cannot carry a one-variable claim about the
predictor form. **Decision 2026-08-30: re-train all ten at the v2 hub** as
Group 2b, about 30 extra segments. The v1-regime originals stay as context.

**Curve backfill DONE 2026-08-30.** All 27 arms of this family now have
complete 8-point milestone series in `logs/report/milestone_curve.csv`
(620 rows). The three that were missing — `diffntp_chunktok`,
`diffntp_merged64`, `diffntp_merged` — were mirrored off ICE and scored
locally at 8/8 cells each, after the optimizer-restore fix that had been
blocking every pre-2026-08-30 checkpoint.

**`diffntp_merged` scored for the first time**, which was the gate on
attributing `merged64`'s lead:

| arm | width | SR | MPJPE-L | MPJPE-G |
|---|---|---:|---:|---:|
| `diffntp_merged64` | 64 | **0.9207** | **24.54** | **91.12** |
| `diffntp_merged` | 256 | 0.9004 | 26.47 | 115.76 |
| `diffntp_chunk` (two-head) | 256 | 0.9163 | 24.07 | **84.69** |

Two readings, and only one of them is safe.

**Safe: at 256-D the separate endpoint head earns its place.** Two-head
`diffntp_chunk` beats merged-256 on all three metrics (0.9163 / 24.07 /
84.69 against 0.9004 / 26.47 / 115.76), which is the predicted consequence
of folding the endpoint into 1 of 11 slots and dropping its grounding
pressure roughly tenfold.

**UNRESOLVED: whether the merged head prefers 64-D.** At the 2B checkpoint
merged64 beats merged-256 on all three, but merged-256 REGRESSED over its
last 500M on the milestone board — 0.9219 / 24.34 / 81.09 at 1.5B falling to
0.8945 / 25.14 / 93.34 at 2.0B — and its 1.5B point is level with merged64's
2B (0.9258 / 23.58 / 77.72). Per the standing rule that checkpoint variance
exceeds evaluation noise, this comparison needs the neighbouring checkpoints
reported, and it does not settle the width question for the merged objective.

## 6. Protocol

As the v1 star ([Interface Ablation Study](interface-ablation-study.md)
sections 3 and 4) except where section 2 above states otherwise:
`Isaac-Imitation-G1-v2`, Newton MJWarp, the 129,785-clip BONES-SEED set
resident in RAM, `gamma 0.97`, policy and value `[2048, 2048, 1024, 1024, 512,
512]` with SiLU, encoder pretrain 50,000 updates at batch 8,192.

**Boards.** Every v2 row is scored on `bones_testbed4096_v1` clean. The
256-clip `bones_milestone_testbed256_v1` board carries the convergence series.
The two disagree by more than a rounding step (`diffntp_token` reads
0.9258 / 24.11 / 84.20 on 256 clips and 0.9121 / 24.33 / 84.97 on 4,096), so
never mix them inside one table or one figure.

**Matched success sets.** MPJPE is success-only, so any table comparing L or G
across rows must recompute both on the rows' common success set, then freeze
and name it. Adding a row changes every number.

**Noise band.** 0.016 success rate, 1.3% MPJPE-L, 6.7% MPJPE-G, from the v1
re-scoring experiment. One seed per arm; every within-band ordering is
unresolved.

## 7. Gates before submission

1. **Hub qualification.** The hub cell — merged head, 64-D, hold 1, phase on,
   20,480 environments, `random80_adaptive20`, penalty on — has never been
   trained in exactly this combination. `diffntp_merged64` is the nearest
   measured relative and differs in environment count, reset schedule, penalty
   and `feet_acc`. Train the hub first and check its episode-length slope
   against `diffntp_merged64`'s curve before committing the other 40 arms.
   The curve is now in `logs/report/milestone_curve.csv` at eight points:
   0.8594 at 0.25B rising to 0.9258 at 2B.
2. **Quantizer smoke — RUN 2026-08-30.** 35 arms pass, 4 posterior arms are
   skipped by design, and the two PRETRAINED VQ arms fail: `g1_recon_vq` and
   `g3_vq64` report code perplexity exactly 1.0 with the code numerically
   constant (`z_dim_std_mean` 4e-9 and 6e-9), against healthy FSQ partners at
   23.1 and 26.8 and healthy categorical and Gumbel cells.

   **Vector quantization is not what fails; the frozen pretrained codebook
   is.** The v1 study measured both routes: `bn_vq_ema` (pretrained) scored
   0.0000, while `post_recon_vq` 0.8945, `post_pgrecon_vq` 0.8867 and
   `post_pg_vq` 0.8828 all trained normally with the code learned during RL —
   `post_recon_vq` is the best success rate in that nine-arm grid. Our two
   failing arms sit on the pretrained side; `g1_post_pgrecon_vq` sits on the
   working side and the smoke never touches it.

   The two therefore carry different value. `g1_recon_vq` is pretrained
   reconstruction with **no SIGReg** — `bn_vq_ema`'s configuration under a
   different decode target, predicted dead, worth keeping only as a declared
   negative control. `g3_vq64` is pretrained with **SIGReg active**, which is
   the unmeasured cell this whole axis exists for, since shaping the code
   marginal is exactly the mechanism a collapsing codebook lacks. A 4-update
   smoke leaves the codebook near its initialization and cannot test that.
3. **The rot6d convention.** The data plane emits the interleaved 6-D rotation
   layout while RLOpt's `_rot6d_to_matrix` parses two concatenated columns
   (found 2026-08-29, unfixed). Targets routed through
   `_reanchor_heading_frames` are distorted. **The hub is not affected** — the
   merged span uses the executed anchor at context 0 — and neither is any
   Group 1, 3, 4, 5 or 6 arm. It DOES affect most of Group 2b: `g2_mlp`,
   `g2_nosig`, `g2_sg`, `g2_online`, `g2_lejepa_*`, `g2_trip` and `g2_token`
   all carry an EMA or stop-grad token target that routes through the
   re-anchoring. Those eight arms should not be launched before the
   convention is decided, or the whole predictor-form comparison is run on a
   distorted target.
4. **Merged-head evaluation quirk.** With `jepa_endpoint_coeff=0` the endpoint
   eval triplet is inert by design (real ~ zero ~ shuffled ~ 39.6 for both
   merged arms). Do not read it as encoder collapse; use
   `z_effective_rank` and the chunk loss instead.

## 8. What this star cannot answer

- **Interactions.** A star measures main effects. Group 5's cadence arms cross
  no other axis; a width x hold probe would have to be added explicitly.
- **Robustness.** Domain randomization off, no push throughout.
- **Convergence.** No v1 arm was converged at 2B and no v2 arm will be. The
  star ranks arms mid-flight, deliberately, so that no arm is promoted on the
  outcome.
- **Smoothness.** Excluded from this star by user decision 2026-08-30, even
  though every arm now trains with the action-rate penalty; the smoothness
  program keeps its own campaigns.
