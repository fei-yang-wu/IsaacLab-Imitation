# 2026-08-13 — latent-bottleneck ablation (robot_heading, hold 1, 100M)

Eight arms, one variable each: the latent-learning method behind a 64-D
tracker command. Shared: `robot_heading` anchor, hold 1 (no phase channel),
encoder trunk 2048/1024/512/512 silu, SONIC-scale tracker, full BONES-SEED
(129,785 motions), 16384 envs, 100M frames, seed 0. `run.sh <arm> <stage>`;
oracle scoring via `eval_oracle.sh <arm>`.

## Oracle eval, number of record

30 compositionality motions x 5 episodes, M3 fall-only, Newton, 2000-step cap:

| arm | objective | bottleneck | MPJPE-L | fall-free |
| --- | --- | --- | ---: | ---: |
| cont_det | DiffSR endpoint | continuous | **79.40** | 0.687 |
| jepa_sigreg_ebm | NTP + DiffSR + SIGReg | continuous | 82.40 | 0.593 |
| jepa_ntp | NTP, InfoNCE energy | continuous | 84.67 | 0.673 |
| cont_det_ln | DiffSR endpoint | continuous + LN | 85.00 | **0.840** |
| fsq64 | DiffSR endpoint | FSQ 64x32 | 89.24 | 0.833 |
| jepa_pure | NTP + SIGReg (LeJEPA) | continuous | 93.03 | 0.707 |
| group_vq | DiffSR endpoint | 8x32 categorical | 93.63 | **0.860** |
| vq | DiffSR endpoint | EMA VQ, unbounded | — | diverged |

Reference points on this protocol: deployed fsq64 (4.5B frames, hold 10)
22.06 mm / 0.947; SONIC v1.1 25.41 mm / 1.000.

## Readings (single seed, 150 episodes/arm, 100M frames — preliminary)

1. **Two clean tiers on survival.** Survivors (>=0.83): the three DiffSR
   arms with a bounded or normalized latent (group_vq, cont_det_ln, fsq64).
   Fallers (<=0.71): every jepa-family arm and the bare continuous one. The
   signature is "quantized or LayerNormed"; the mechanism is NOT settled —
   SIGReg shapes the distribution as strongly as anything here and its arms
   still fall, so distribution shaping alone is not the cause.
2. **MPJPE and survival anti-correlate across arms** (finer command space =
   tighter, riskier), the same axis trade seen in the planner inference
   knobs. Pareto set: cont_det (precision), group_vq (survival),
   cont_det_ln (balance).
3. **The NTP/JEPA objectives bought nothing at this budget.** All three
   variants sit mid-to-worst on both axes; adding the DiffSR grounding to
   NTP (sigreg_ebm) produced the WORST survival of the ablation.
4. **LayerNorm is decisively not bad**: +15 points of survival over the
   identical LN-free arm at 5.7 mm better MPJPE than fsq64.
5. **Unbounded VQ is refuted** (codebook scale explosion, z_rms 0.03 -> 41,
   instant tracker collapse).
6. Training-tail metrics were a 1 mm band across all seven healthy arms —
   they predicted none of this separation. Oracle eval is the only signal.

vq divergence detail and all raw logs: `logs/bones129k_latent_quant_ablation/`.

## LayerNorm arm validity window (2026-08-15)

A 2026-08-14 edit to `run.sh` made `--encoder_layer_norm` unconditional for
every arm, which turned `fsq64_ln`/`cont_det_ln` into no-op duplicates of
their siblings. Restored to a per-arm axis on 2026-08-15.

- The table above and reading 4 come from pretrains launched from the
  2026-08-13 per-arm revision; they stand as recorded (single seed,
  preliminary).
- Any encoder pretrained from this script or from the
  `2026-08-14-latent-quant-ice-repeats` campaign between 2026-08-14 and
  2026-08-15 has LayerNorm regardless of arm name. Do not aggregate those
  seeds with post-fix seeds of the non-LN arms, and do not read an LN-vs-no-LN
  comparison out of that window.

## Training-signal analysis (2026-08-14)

Three hypotheses for the survival split, tested against measurements:

**Token information content — refuted.** DiffSR endpoint eval with real vs
shuffled tokens (`shuffled/real` loss ratio): cont_det 341x (falls),
cont_det_ln 306x (survives), fsq64 182x (survives), group_vq 6.1x (survives
best), jepa_sigreg_ebm 6.8x (falls worst). No correlation with survival in
either direction. (The ratio is 1.0 for jepa_ntp/jepa_pure only because their
DiffSR heads are untrained — no valid measurement exists for them.)

**Latent temporal smoothness — refuted.** Direct measurement (2,048 window
rows, 10 consecutive hold-1 tokens each, per-step relative jump
`||dz||/rms||z||`): group_vq is by far the JERKIEST stream (0.506, with 26.5%
of steps snapping to an identical code) yet survives best; cont_det is the
smoothest (0.146) and falls. Piecewise-constant-with-jumps beats smooth.

**Action smoothness — the surviving correlate.** The tail
`action_rate_l2` penalty orders the arms almost exactly by oracle survival:
group_vq -0.136 < cont_det_ln -0.147 ~ fsq64 -0.148 < cont_det -0.152 ~
jepa_pure -0.152 < jepa_ntp -0.155 < jepa_sigreg_ebm -0.161. Training
episode length gives the same ordering (45.2/42.1/41.1 for survivors vs
35.2-37.7 for the jepa arms). What predicts falling is not the command
stream's roughness but the roughness of the ACTIONS the tracker converged to
under that command distribution.

**Practical consequence:** training-time `action_rate_l2` and episode length
predicted the oracle survival ranking that `mpjpe_l_mm` and
`reference_finished` completely missed. Future sweeps can gate arms on these
two scalars without waiting for an eval.

**Scale-inflation observation (mechanism candidate for the quantized tier).**
Token norm on freshly sampled windows vs pretrain-batch expectation: fsq64
1.1x, group_vq 1.0x (lattice/codebook clamp off-distribution inputs back to
the training range) — while cont_det, cont_det_ln and the jepa arms inflate
1.7-1.9x, with the jepa family the largest in absolute scale (||z|| 11-16).
Bounded output = distribution-proof command. Does not explain cont_det_ln's
survival, which remains the open question. Caveat: the k>=1 chunks in this
measurement go through the offline re-anchoring path; a frame error there
would inflate continuous tokens while quantized ones hide it.

**Directions this suggests for the LN and JEPA families:**
- `jepa_fsq`: the jepa_pure objective with an FSQ lattice on the token output
  — keeps the NTP-learned geometry, adds the boundedness the survivor tier
  shares. One-arm test of whether JEPA's problem is merely its unbounded head.
- Tracker-boundary normalization of continuous tokens (the cont_det_ln
  result suggests normalizing at the interface, not just inside the encoder).
- Seed repeats of cont_det_ln vs fsq64 before promoting LN into the deployed
  recipe.

## Extended table: stride-5 and online-dynamics arms (2026-08-14)

Five arms added; same oracle protocol (30 motions x 5 episodes, M3 fall-only,
Newton, 2000-step cap). Stride arms declare `env.expert_macro_frame_stride=5`
at eval or the checkpoint guard refuses them; dyn arms are scored with the
encoder weights extracted from the tracker checkpoint's sampler state (the
ONLINE-finetuned encoder), not the frozen pretrain file.

| arm | fall-free | MPJPE-L |
| --- | ---: | ---: |
| **cont_det_ln_dyn** | **0.873** | 86.91 |
| **fsq64_dyn** | 0.860 | 88.17 |
| **fsq64_s5** | 0.860 | 90.13 |
| group_vq | 0.860 | 93.63 |
| cont_det_ln | 0.840 | 85.00 |
| fsq64 | 0.833 | 89.24 |
| jepa_pure_s5 | 0.780 | 88.32 |
| jepa_sigreg_ebm_s5 | 0.733 | 86.61 |
| jepa_pure | 0.707 | 93.03 |
| cont_det | 0.687 | 79.40 |
| jepa_ntp | 0.673 | 84.67 |
| jepa_sigreg_ebm | 0.593 | 82.40 |

### Three findings

1. **Online dynamics is the only intervention that improved BOTH axes.**
   Each dyn arm beats its frozen-encoder sibling on survival AND MPJPE:
   `cont_det_ln` 0.840/85.00 -> `cont_det_ln_dyn` 0.873/86.91 (survival, MPJPE
   ~flat) and `fsq64` 0.833/89.24 -> `fsq64_dyn` 0.860/88.17. During training
   the achieved-window DiffSR loss fell 5.49 -> 2.45 while the expert term
   stayed flat at ~0.27, so the encoder learned the robot's own transitions
   without losing the reference ones. The training-signal predictors
   (`action_rate_l2`, episode length) called this in advance.
2. **Stride 5 is rehabilitated.** `fsq64_s5` 0.860 vs `fsq64` 0.833, and
   stride lifted both jepa arms (+0.073, +0.140). Our standing "stride-5
   encoders collapse" conclusion (0.68 SR vs 0.90) was an artifact of the OLD
   recipe, not a property of the sparse window. SONIC's cadence works here.
3. **JEPA improves with stride but stays last.** Consistent with the
   unbounded-token diagnosis (scale inflation 1.7-1.9x vs 1.0x for quantized)
   rather than with the objective being wrong. The untested fix is an FSQ
   lattice on the jepa head.

All single seed; differences under ~15% are directional only. `vq` remains
refuted (codebook scale explosion).
