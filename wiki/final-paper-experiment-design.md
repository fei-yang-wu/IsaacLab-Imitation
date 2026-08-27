# Final Paper Experiment Design

Status: decisions locked with the user on 2026-08-17. This page is the
paper-facing contract for the final experiment sections. It supersedes the
two-row main-grid decision in
[causal-interface-paper-plan.md](causal-interface-paper-plan.md) where the two
pages disagree; that page remains the authority for provenance, gates, and the
frozen evaluation machinery it describes.

The paper has three result sections:

1. **Low-level tracker**: tracking performance against SONIC, plus a
   latent-space design ablation.
2. **Planner interface comparison**: one language-conditioned planner per
   interface on the same protocol.
3. **Skill composition**: owned by another collaborator. Not planned on this
   page. Do not start work on it this week.

## Headline lock

The headline system for the whole pipeline is:

- **Tracker: `fsq64_hold10`**, the 10.00B-frame FSQ arm of
  `experiments/campaigns/2026-08-15-latent-bottleneck-10b/`
  (64 FSQ coordinates x 32 levels, code held ten 50 Hz control steps,
  published at 5 Hz).
- **Planner: the GR00T-based language planner** with exponential temporal
  ensembling, in the recipe of the 2026-08-13 best result
  (46.95 mm MPJPE-L / 0.998 fall-free over 28 motions).

Lock procedure, required before any paper table cites the headline:

1. Record the tracker checkpoint path and SHA-256.
2. Run `validate_latent_skill_checkpoint_binding.py` and keep the binding
   record (encoder in the tracker checkpoint tensor-identical to the selected
   skill checkpoint).
3. **Parity re-run gate (open):** the 46.95 mm planner result used the older
   `fsq64_scaled28` tracker checkpoint. Re-run the planner evaluation against
   the locked 10B `fsq64_hold10` checkpoint and confirm the result holds
   before the lock is final.
4. Export the policy bundle for the locked pair.

Known open risk carried with the lock: `fsq64_hold10` has the largest
Isaac-to-MuJoCo gap in the 2026-08-17 sidecar screen (0.55 weighted success in
MuJoCo under sensor noise against 0.92 in Isaac). The 4,096-motion Isaac board
stays the deciding board, but do not make a sim-robustness claim for the
headline until that backend gap is explained.

## Approved claim boundary against SONIC

> **SUPERSEDED 2026-08-26 for the tracker row.** The live paper numbers are
> Tables A-C in
> `experiments/campaigns/2026-08-17-paper-metric-canon/README.md`: the
> `bones_testbed4096_v1` board, the `sonic_v1_1` checkpoint (not
> `sonic_release`), `ln_hold1_sonicreset` as the tracker, and matched
> intersections frozen as named artifacts. The table below is the 2026-08-17
> state — contiguous ranks 12288-16383, `fsq64_hold10`, `sonic_release` — and
> is kept only as history. Do not cite it in the paper.

On the frozen 4,096-motion scoreboard (2026-08-17, historical):

| Arm | Frames | SR | Success-only MPJPE-L | `ee_body_pos` failures |
| --- | ---: | ---: | ---: | ---: |
| released SONIC | - | 0.9937 | 28.65 mm | 26 |
| `fsq64_hold10` | 10.00B | 0.9197 | 24.93 mm | 260 |

- **Do not claim "our tracker beats SONIC."** SONIC is clearly better on
  falls (ten times fewer end-effector failures). We are 13% better on
  success-only MPJPE-L, but that number carries two qualifiers that must
  appear with it: it is near the unresolved evaluation-noise band, and
  success-only MPJPE across unequal success rates has a selection bias —
  the higher-SR arm averages over hard motions the lower-SR arm did not
  survive.
- **The defensible headline claim is pipeline-level parity:**
  `fsq64_hold10` + our planner reaches 46.95 mm / 0.998 against the released
  SONIC planner on its own tracker at 46.33 mm / 1.000; the 0.6 mm gap is
  inside evaluation noise. The tracker trades fall rate for precision.

## Section 1: low-level tracker

### Main results

1. **Two-axis main table** on `bones_testbed4096_v1` (frame-0, seed 0, mode
   actions, clean AND `no_push`, released-SONIC thresholds, `foot_pos_xyz`
   and `base_too_low` disabled). Superseded 2026-08-26: the population is the
   registry board, NOT contiguous ranks 12288-16383, and success-only columns
   reduce to the matched intersection frozen per table.
   Columns: frames, SR, success-only MPJPE-L, `ee_body_pos` failure count,
   and interface bandwidth (command width, values per second). Always report
   SR and MPJPE together; either alone is misleading against SONIC.
2. **One training-curve figure** on a fair setup (same environment geometry,
   same recipe; frames on the x-axis). One figure with all arms; the full
   per-arm curve grid goes to the appendix.
3. **Termination-cause attribution**: one small table or bar chart showing
   that `ee_body_pos` dominates failures in every arm. This answers the
   reviewer question "why do you fall."
4. Optional one-figure Pareto scatter (SR against success-only MPJPE-L) to
   present the SONIC trade-off visually.

### Latent-design ablation

A 2 x 2 grid plus two reference rows, **table primary, curves secondary**:

| Axis | Values |
| --- | --- |
| Latent space | continuous, discrete (FSQ) |
| Objective | spectral (DiffSR), reconstruction |

Reference rows: the explicit `root_qpos` interface (ceiling) and released
SONIC. All ablation arms use hold 10 and matched training recipe. Do not mix
the hold axis into this grid; hold length is a separate, already-resolved
axis (the 2026-08-15 nine-arm grid).

Frame counts are printed per row. The explicit row stays at its existing
7.60B checkpoint by explicit user decision (2026-08-17); no 10B re-run.
Because the frame mismatch favors the latent arms, any sentence of the form
"the latent interface reaches the explicit baseline" must carry the
"frames not matched" qualifier in the same sentence.

Report both the table metrics (SR, success-only MPJPE-L) and the curves; the
table carries the claim, the curve figure supports sample efficiency.

## Section 2: planner interface comparison

One language-conditioned planner per interface, same backbone, same data
budget, same optimizer budget, same evaluation starts, same language goals,
at least three seeds. Language conditioning provides motion variety; the
interface is the only main variable.

### Rows

| Row | Interface | Publication | Consumption | Status |
| --- | --- | --- | --- | --- |
| Explicit | 38-D `root_qpos`, single frame | 5 Hz | held ten 50 Hz steps | tracker exists (`root_qpos_explicit`) |
| Continuous latent | 256-D code | 5 Hz | held ten steps | tracker exists (`cont_det` family) |
| Discrete latent (headline) | FSQ 64x32 code | 5 Hz | held ten steps | locked headline pair |
| SONIC-style | FSQ per-step token chunk | 5 Hz chunk of ten tokens | one token per 50 Hz step | **conditional**: needs a new 10B hold-1 FSQ tracker; known dead-zone risk (64-D hold-1 collapsed in our 100M grid; released SONIC escapes it with its stride-5 future window and scale) |
| Explicit-encoder | latent from a reconstruction-only encoder over the explicit window | 5 Hz | held ten steps | definition to confirm before it enters the table |

Decisions folded into this table:

- **The direct 50 Hz ceiling row is dropped** (user decision, 2026-08-17).
- **The explicit row is the 38-D single-frame `root_qpos` command, not the
  670-value ten-frame packet.** The streamed-packet equivalence machinery is
  no longer on the main path; it remains available for appendix work.
- **Hold-1 does not disqualify an interface from planner service.** SONIC's
  VLA predicts a chunk of per-step tokens and the tracker consumes one per
  50 Hz step; a 5 Hz planner can publish such a chunk. Consumption cadence
  is part of the interface and is reported per row.
- EE/keypoint chunks, per-step token variants, and Future-CVAE stay
  appendix/diagnostic; do not enlarge the main table.

### Required columns and rules

- SR (SONIC termination definition, no push, domain randomization on) and
  success-only MPJPE-L.
- **Oracle-normalized score**: planner SR divided by the same interface's
  oracle SR on its own tracker. Each interface has its own tracker, so raw SR
  confounds tracker quality with interface learnability.
- Bandwidth (values per second, bits per second) and consumption cadence.
- Planner parameter count and measured latency.
- **One inference-time smoothing policy, stated explicitly.** Either
  temporal ensembling for every row where it is defined, or ensembling off
  everywhere with ensembling as a separate ablation. A silent asymmetry here
  invalidates the comparison.
- Single-seed differences inside the known evaluation noise (about 12-15%
  relative) are unresolved; use repeated seeds before claiming a difference.

## Section 3: skill composition

Owned by another collaborator. Placeholder only. Revisit after sections 1
and 2 are frozen.

## Open items

1. Run the parity re-run gate (planner against the locked 10B
   `fsq64_hold10`) and record the result here.
2. Decide whether to fund the SONIC-style hold-1 FSQ tracker run (one 10B
   run, dead-zone risk) or drop that row.
3. Confirm the exact definition of the explicit-encoder row before adding it
   to the table.
4. Explain or bound the `fsq64_hold10` MuJoCo backend gap before any
   robustness claim.
