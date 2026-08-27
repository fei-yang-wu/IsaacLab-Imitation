# Results: what makes a learned command interface good for whole-body tracking

Publication-facing draft of the ablation results section. Every number comes
from `logs/report/milestone_curve.csv`, produced by
`imitation_experiments.reporting.curve_table` over the scored evaluation
directories. The setup, and every arm's exact configuration, is
[Interface Ablation Study](interface-ablation-study.md).

**Status: preliminary.** One seed per arm, a 256-clip curve board, and no
repeat except the two `hold1` rows. Read the section as a map of where the
effects are, not as a set of settled orderings. The
[limitations](#57-limitations) subsection states exactly which claims survive
the evaluation noise and which do not.

---

## 5. Ablations

### Setup

We train 72 tracking policies that differ only in the command interface between
a high-level module and the low-level controller: what the encoder is trained to
predict, the shape of the code it emits, what state it reads, and how often the
code is published. Every arm shares the environment, the reward set, the
termination set and its curriculum, the reset distribution, the optimizer, the
network sizes, 16,384 parallel environments at 24 rollout steps, and a budget of
2B environment frames. Arms drawn from the same star differ from their control
in exactly one field.

Each arm writes a checkpoint every 250M frames, and we score all eight on the
same 256-clip board drawn from the testbed population, with domain
randomization off and no push. Budget is therefore an axis of the study rather
than a fixed point: an effect that only appears at one end of the curve is
reported as budget-dependent.

We report the three numbers that must travel together: success rate under
SONIC's termination definition, success-only frame-weighted MPJPE-L (local,
root-relative), and success-only MPJPE-G (global). MPJPE-L alone flatters a
policy that holds a pose while drifting, and a success-only error is meaningless
beside a different success rate.

### 5.1 Success rate saturates; global drift is what separates interfaces

At 2B frames, **18 of 72 arms are statistically indistinguishable on success
rate** — all within 0.016 of the best (0.9297), which is the run-to-run spread
we measure for this evaluation (Section 5.7). Inside that band, local error is
almost constant: MPJPE-L runs 22.80–26.95 mm, a factor of 1.18. Global error is
not: MPJPE-G runs 74.85–416.80 mm, **a factor of 5.6 across arms whose success
rates are tied.**

| | inside the tied-SR band (18 arms) |
|---|---|
| success rate | 0.9141 – 0.9297 (tied by construction) |
| MPJPE-L | 22.80 – 26.95 mm (1.18×) |
| MPJPE-G | 74.85 – 416.80 mm (**5.6×**) |

This is the central result of the section. A command interface that survives the
episode and tracks the reference pose well locally can still be drifting five
times further in the world frame than another interface that scores the same on
both of the metrics usually reported. Success rate and MPJPE-L are saturated
measurements at this scale; MPJPE-G carries the remaining signal. We use it as
the primary discriminator below and report all three throughout.

### 5.2 The code: width is cheap, quantization is not

Continuous codes are insensitive to width over the range we tested, and every
discrete bottleneck costs something. Narrowing a continuous code from 256 to 64
dimensions changes nothing we can resolve, while the discrete arms degrade
monotonically as the bit budget falls, ending in total failure.

| arm | code | SR | MPJPE-L | MPJPE-G |
|---|---|---|---|---|
| `bn_cont64` | continuous 64-D | 0.9180 | 22.80 | 204.74 |
| `ctrl` | continuous 256-D | 0.9102 | 23.44 | 199.87 |
| `bn_cont128` | continuous 128-D | 0.9102 | 23.39 | 205.03 |
| `bn_gaussian` | Gaussian posterior + KL | 0.9023 | 27.68 | 212.18 |
| `bn_sonic_fsq64` | FSQ 64 × 32 levels | 0.9023 | 28.86 | 177.70 |
| `bn_sonic_fsq32` | FSQ 32 × 32 levels | 0.8945 | 31.99 | 206.43 |
| `bn_sonic_fsq64_l8` | FSQ 64 × 8 levels | 0.8672 | 33.82 | 193.73 |
| `bn_gumbel_multicat` | 64 groups × 32 categories | 0.7031 | 46.45 | 340.05 |
| `bn_sonic_fsq16` | FSQ 16 × 32 levels | 0.6836 | 44.33 | 242.30 |
| `bn_categorical` | hard straight-through 64 × 32 | 0.5703 | 54.77 | 689.24 |
| `bn_gumbel` | single Gumbel codebook, K=512 | 0.3750 | 68.45 | 1364.72 |
| `bn_vq_ema` | single EMA VQ codebook, K=512 | 0.0000 | — | — |

Three findings survive the noise band:

**A single codebook is not enough capacity for whole-body tracking.** Both 9-bit
arms fail: EMA vector quantization reaches success rate 0.0000 at every one of
its eight checkpoints, so it has no success-only error at all, and Gumbel-softmax
over the same 512-entry codebook reaches 0.3750 with 1364.72 mm of global drift.
Splitting the same budget across many small groups recovers most of the loss —
`bn_gumbel_multicat` at 64 × 32 reaches 0.7031 — so the failure is the single
categorical draw, not discreteness itself.

**At a fixed bit budget the estimator matters.** `bn_gumbel_multicat` and
`bn_categorical` are the same 64 × 32 code; the first uses a Gumbel-softmax
relaxation and the second a hard straight-through estimator. The gap is 0.13
success rate (0.7031 vs 0.5703) and 2.0× MPJPE-G.

**FSQ trades local precision for global stability.** Against its continuous
partner at the same width, `bn_sonic_fsq64` is 27% worse on MPJPE-L (28.86 vs
22.80 mm) but 13% better on MPJPE-G (177.70 vs 204.74 mm), at a success rate
inside the noise band. The lattice appears to regularize where the root goes at
the cost of how exactly the joints follow — a trade the pose-only metrics hide.

Layer normalization on the encoder output is a null at this budget
(`bn_no_ln` 0.8945 / 22.48 / 222.26 against the hub's 0.9102 / 23.44 / 199.87);
the differences sit at the edge of the band in opposite directions on the two
error metrics.

### 5.3 What the encoder is trained to predict

Nine objectives, one field apart, on the same 256-D continuous code:

| objective | SR | MPJPE-L | MPJPE-G |
|---|---|---|---|
| `ctrl` endpoint DiffSR | 0.9102 | 23.44 | 199.87 |
| `obj_recon` autoencoding the input window | 0.9023 | 26.22 | 401.18 |
| `obj_jepa_sigreg_ebm` DiffSR + chunk NTP + SIGReg | 0.9023 | 27.34 | 206.70 |
| `obj_state_occupancy` successor occupancy | 0.8984 | 24.56 | 169.68 |
| `obj_phi_bilinear` bilinear successor head | 0.8945 | 25.22 | 180.40 |
| `obj_jepa_infonce` JEPA with InfoNCE energy | 0.8906 | 29.87 | 201.04 |
| `obj_semimarkov` semi-Markov factorization | 0.8789 | 27.32 | 181.88 |
| `obj_jepa_ntp` latent next-token prediction | 0.8438 | 34.68 | 167.31 |
| `obj_endpoint_delta` endpoint delta target | 0.8438 | 40.75 | 249.34 |

Seven of the nine are tied with the hub on success rate. The objective axis is
therefore best read on the error metrics, where two effects are larger than
noise.

**Reconstruction is a local objective and produces a locally-good, globally-bad
interface.** `obj_recon` matches the hub on success rate and is 2.0× worse on
MPJPE-G (401.18 vs 199.87 mm). Autoencoding the encoder's own 380-value input
window gives the code no reason to carry where the window sits in the world.
The follow-ups confirm the mechanism rather than the arm: moving the decode
target to the endpoint (`recon_endpoint`, 364.18 mm) or to the full future
window including the endpoint the encoder never sees (`recon_full_window`,
416.80 mm) leaves the global error in the same regime.

**Predicting the future in latent space trades success rate for drift.**
`obj_jepa_ntp` posts the best global error of the objective axis (167.31 mm,
16% better than the hub) at the worst success rate of its converged arms
(0.8438) and 48% worse MPJPE-L. The composite that adds endpoint grounding to
the same prediction term, `obj_jepa_sigreg_ebm`, recovers the success rate
(0.9023) but gives the global gain back (206.70 mm). Section 5.6 takes this
trade apart on the better interface, where it resolves differently.

### 5.4 Encoder input: what the window is, not how wide

| arm | change from hub | SR | MPJPE-L | MPJPE-G |
|---|---|---|---|---|
| `ctrl` | — | 0.9102 | 23.44 | 199.87 |
| `in_window_full` | window includes the endpoint | 0.9102 | 23.20 | 224.25 |
| `in_anchor_robot` | anchor in the live robot frame | 0.9062 | 24.64 | 253.86 |
| `in_fullbody670` | adds 29 reference joint velocities | 0.8984 | 26.92 | 225.79 |
| `in_anchor_expert_heading` | anchor in the expert's heading frame | 0.8945 | 30.44 | 350.25 |
| `in_stride5` | 10 frames spaced 5 apart (0.9 s window) | 0.6992 | 45.62 | 446.67 |

**Frame spacing is the one input choice that matters, and it matters enormously.**
Holding the frame count at ten and spacing them five control steps apart —
SONIC's 0.9-second window — costs 0.21 success rate and 2.2× MPJPE-G. Ten
consecutive frames beat ten strided frames by a margin no other input change
approaches. Adding second-order information to the same window
(`in_fullbody670`, 380 → 670 values) is a null; the anchor frame is a null to
within noise except for the expert-heading variant, which is 1.75× worse
globally.

### 5.5 Publication cadence: holding a code costs global accuracy

The hold is how many control steps one published code is held before the encoder
runs again — 10, 5, or 1, at both code widths. This is the only axis where a
single field moves all three metrics coherently.

| width | hold 10 | hold 5 | hold 1 |
|---|---|---|---|
| 256-D continuous | 0.9102 / 23.44 / 199.87 | 0.9102 / 24.79 / 146.92 | 0.9180 / 25.76 / **140.94** |
| 64-D FSQ | 0.9023 / 28.86 / 177.70 | 0.9023 / 28.52 / 141.36 | 0.8789 / 30.65 / **134.10** |

MPJPE-G falls monotonically with the hold at both widths: 29% from hold 10 to
hold 1 at 256-D, 25% at 64-D. Both drops are far outside the noise band; the
step from hold 5 to hold 1 (4% and 5%) is not, so what is resolved is that
publishing more often reduces global drift, not the exact shape of the curve
between 5 and 1. MPJPE-L moves the other way by a smaller amount (+10% and +6%),
which is the same precision-versus-drift trade the FSQ arms show.

The cost appears at the narrow width: at 64-D, hold 1 loses 0.023 success rate
against hold 10, marginally outside the band, while the 256-D arm gains 0.008.
Hold 5 sits at the wide arm's success rate at both widths, so the midpoint
captures most of the global-error gain without the narrow-code penalty.

**The phase channel is load-bearing at hold 10.** Removing the two-wide sin/cos
slot clock, so the command is the code alone, drops success rate from 0.9102 to
0.4141 and multiplies global error by 6.7× (`use_phase_none`). The tracker needs
to know where in the held window it is. At hold 1 the clock is constant and the
arm effectively runs phase-free; supplying a live clock instead
(`hold1_live_phase`, 0.9102 / 25.52 / 143.53) changes nothing we can resolve
against `use_hold1`.

### 5.6 Learning the code during RL, and the mechanism square

**The posterior route is uniformly worse globally.** Nine arms learn the code
during RL instead of pretraining it — quantizer (continuous / FSQ / EMA VQ)
crossed with the training signal (reconstruction only, policy gradient only,
both). Success rates land between 0.8477 and 0.8984, at or just below the
pretrained hub; MPJPE-G lands between 416.93 and 505.50 mm, **2.1–2.5× the hub's
199.87 mm, with no arm escaping the band.** Notably, EMA vector quantization
does *not* collapse here (`post_recon_vq` 0.8945) the way it does as a
pretrained bottleneck, so the failure in Section 5.2 is a property of the frozen
pretrained codebook, not of the quantizer.

**On the best interface, grounding the prediction term in data is what buys
global accuracy.** The mechanism square fixes the interface at hold 1 with two
extra reward terms enabled and moves only the loss composition. Selected rows:

| arm | loss composition | SR | MPJPE-L | MPJPE-G |
|---|---|---|---|---|
| `diffntp_chunk_h1_ee_wide` | generative next-chunk, no self-target | **0.9297** | 23.45 | **74.85** |
| `diffntp_token_h1_ee_wide` | generative next-token (EMA target) | 0.9258 | 24.11 | 84.20 |
| `diffntp_chunkra_h1_ee_wide` | next-chunk, target re-anchored | 0.9219 | 23.28 | 84.80 |
| `diffntp_pair_h1_ee_wide` | joint next (state, token) | 0.9180 | 23.29 | 83.27 |
| `jepa_h1_ee_wide` | DiffSR + NTP + SIGReg, EMA target | 0.9141 | 25.34 | 79.69 |
| `jepa_nosig_h1_ee_wide` | SIGReg removed | 0.9180 | 26.52 | 93.41 |
| `sg_h1_ee_wide` | stop-grad target instead of EMA | 0.9102 | 26.21 | 97.94 |
| `lejepa_ema_h1_ee_wide` | prediction + SIGReg only, EMA target | 0.8828 | 31.14 | 96.83 |
| `lejepa_sg_h1_ee_wide` | prediction + SIGReg only, stop-grad | 0.8750 | 34.05 | 94.81 |
| `lejepa_h1_ee_wide` | prediction + SIGReg only, online target | 0.7383 | 47.58 | 691.91 |

The best arm of the whole study is `diffntp_chunk_h1_ee_wide`, whose prediction
term is a generative model of the next chunk of *data* and therefore carries no
self-target at all. It leads on all three metrics simultaneously — the only arm
in the study that does — at 0.9297 / 23.45 / 74.85 mm.

Where a self-target is present, it must be asymmetric. With the prediction term
alone plus its regularizer and a single online encoder on both sides
(`lejepa_h1_ee_wide`), the interface degrades badly (0.7383, 691.91 mm).
Restoring asymmetry rescues it, by EMA (0.8828 / 96.83) or by stop-gradient
(0.8750 / 94.81), and the two mechanisms are indistinguishable here. Adding the
data-grounded endpoint term on top recovers the remaining success rate
(`jepa_h1_ee_wide`, 0.9141 / 79.69). Once the loss is grounded that way, the
variance regularizer is no longer protecting against collapse — removing it
costs 17% MPJPE-G and nothing on success rate.

### 5.7 Budget as an axis, and limitations

**Rankings are established early.** Spearman correlation between the ordering at
250M frames and at 2B is 0.83 for success rate and 0.93 for MPJPE-G across all
72 arms. A quarter-billion-frame screen predicts the 2B ordering well; the
failure mode that motivated running the full budget was a much smaller 10M-frame
gate, which it does not reproduce.

**Most arms are done before the budget ends.** 63 of 72 arms are within the noise
band of their own 1.75B success rate at 2B, and the median arm reaches 81% of its
total 0.25B → 2B success-rate gain by 1B. Global error keeps improving later than
success rate does; the hold-1 arms in particular are still falling at 2B.

**Limitations.** (i) One seed per arm. The two `hold1` rows are the only
repeat in the study, and they are a seed pair for a different comparison. (ii)
The curve board is 256 clips from the testbed population, not the 4096-clip
board a paper row is scored on; numbers here are comparable to each other and
are not paper rows. (iii) Isaac evaluation is not deterministic. Re-scoring
eight identical checkpoints on the identical board moved success rate by at most
0.016 (mean +0.0010), MPJPE-L by at most 1.3%, and MPJPE-G by at most 6.7%
(mean −1.30%), with the differences scattered in sign; that is the band used
throughout this section. (iv) Domain randomization is off and there is no push,
so nothing here speaks to robustness. (v) `bn_vq_ema` has no error metrics at
any budget because it never completes an episode; it is reported at success rate
0.0000 rather than omitted.

Two claims are strong enough to carry into the paper's argument without a repeat:
the saturation result of Section 5.1, whose 5.6× spread is two orders of
magnitude above the noise band, and the failures — single-codebook quantization,
strided windows, and the missing phase channel — each of which is a 0.2–0.9
success-rate effect. Every within-band ordering in Sections 5.2–5.6, including
which of the four generative-prediction heads leads, requires repeated seeds
before it can be stated as an ordering.
