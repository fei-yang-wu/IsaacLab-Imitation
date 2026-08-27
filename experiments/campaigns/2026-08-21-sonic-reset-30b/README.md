# 2026-08-21 — SONIC resets, 20B to 30B, chasing SONIC's success rate

Continue both SONIC-reset leaders from their 20B checkpoints for another 10B,
changing NOTHING but the budget. Goal set by the user: push the success rate
toward the released SONIC checkpoint's level.

| row | SR | succ MPJPE-L |
|---|---:|---:|
| released SONIC | 0.9937 | 28.65 mm |
| `ln_hold1_sonicreset` @20B | 0.9558 | 22.15 mm |
| `fsq64_hold10_sonicreset` @20B | 0.9468 | 24.57 mm |

Both arms were still gaining at 20B (10B -> 20B bought +0.019 and +0.027 SR).
Whether another 10B closes the remaining ~0.04 is exactly what this measures:
20B vs 30B is a clean budget axis (identical regime), and `save_interval`
500M keeps the milestone trail. If the curve is flat by ~25B, cancel the tail
segments rather than burning the insurance segment.

## Protocol

- Resume: segment 9 loads the newest checkpoint under
  `/data/sonic_reset_20b/<arm>_seed0/tracker`. Those checkpoints carry
  `cumulative_env_frames` (post-2026-08-16 RLOpt), so no
  `initial_frame_offset` anywhere; the global `frame_cap=30000000000` stops
  the chain at exactly 30B.
- Regime byte-identical to `2026-08-18-sonic-reset-20b` segments 6-8:
  `selection=sonic`, `adaptive_uniform_ratio` pinned 0.1 (the landing ramp
  completed long ago), termination curriculum off, frozen encoder from
  `/data/bottleneck_10b/<base>_seed0/encoder` (binding unchanged end to end).
- Three chained segments (`afterany`); the last 10B took ~22 h, so segment 11
  is insurance and normally exits at zero iterations.

## W&B

Appends to the SAME runs as the 20B chain: ids
`ln-hold1-sonicreset-s0-r1` / `fsq64-hold10-sonicreset-s0-r1`, group
`sonic-reset-20b`. That chain predates the per-chain run-id state file and
used its declared ids verbatim, so `<output_root>/wandb_run_id` was seeded
with the SAME verbatim ids before submission — without that the control plane
would mint a token-suffixed new id and split the curve. The runs were created
in shared mode and a chain never changes mode mid-flight; read `env_frames`
as the x-axis, never `_step`.

## Scoring

`2026-08-15-latent-bottleneck-10b/eval_scoreboard4096.sh` with new
`f30…` rows (add the arms' 30B lines to its `ARMS_TABLE`), plus the strat64
EC sidecar as the CPU screen. Read `wiki/canonical-paper-metrics.md` first.

## INCIDENT: fsq64 30B chain NaN-poisoned from ~24.2B (found 2026-08-23)

The fsq64 chain COMPLETED at 30B but its checkpoints carry non-finite policy
tensors (17/19, including the obs-normalizer running stats) from the 24.5B
save onward. Bisection over the saved trail: 24.0B clean, 24.5B poisoned —
onset inside segment 10 on 2026-08-22 ~13:00-14:40. The 30B row scored 0.0 SR
(every env dead in ~2 steps); every board number from this chain past 24.0B
is VOID. Training-side logs did not flag it: r_step/ep_len stayed plausible
and `pi_loss=nan` printed only in the final ~400 iterations — a monitoring
gap (no non-finite watchdog in the trainer).

Cause unresolved. Suspicions: (a) node `atl1-1-03-017-16-0` — the chain sat
on it for all three segments at roughly half the ln chain's throughput
(50-75k vs 129k fps); but pareto-stack jobs ran on the same node at full
speed with sane results; (b) an fsq64-specific numeric instability under the
sonic reset distribution at high SR. Distinguishable only by a rerun.

Last valid checkpoint: `model_step_24000331776` (24.0B, segment-10 tree,
mirrored locally). `ln_hold1_sonicreset` is unaffected (its 30B final is
finite and scored 0.9707; different node).

## Status

- 2026-08-23: fsq64 chain finished but is INVALID past 24.0B (see incident
  above). ln chain finished clean and scored 0.9707/21.75/154.64 at 30B.
- 2026-08-21: submitted, seed 0. `ln_hold1_sonicreset` jobs 5587505-5587507
  (segments 9-11), `fsq64_hold10_sonicreset` jobs 5587509-5587511. Run-id
  state files seeded on ICE before submission. Nothing measured.
