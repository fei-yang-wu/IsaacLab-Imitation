# 2026-09-12 -- the PoE family WITH the latent regularizers

Four arms, one seed each, 10B, W&B group `latent64-probe-10b`. Every PoE arm
submitted before this one turned SIGReg and the latent L2 off, so "product-of-
experts score" and "no regularizer" are confounded in all of them. These
restore the hub defaults and change nothing else.

| this arm | partner (reg off) | phi | experts | base | jobs |
|---|---|---|---:|---|---|
| `z64_poe_reg` | `z64_poe` | `z` | 64 | no | 5767907-10 |
| `z64_poe_base_reg` | `z64_poe_base` | `[1; z]` | 64 + 1 | yes | 5767916-19 |
| `poe_proj256_reg` | `poe_proj256` | `A z` | 256 | no | 5767922-27 |
| `poe_proj256_base_reg` | `poe_proj256_base` | `[1; A z]` | 255 + 1 | yes | 5767929-31 |

`poe_z256` deliberately has no partner here: it is the only arm whose command
changes (258 against 66), so a regularizer contrast on it would move two
things at once against the rest of the family.

## The one variable

`--jepa_sigreg_coeff` and `--reg_coeff` are simply not passed, so the pretrain
takes its defaults, 1.0 and 1e-3. Verified in the resolved batch scripts:
neither flag appears. Everything else is the width-ablation campaign verbatim
(merged `diff_chunk` head, `boundary_next`, endpoint coefficient 0,
pair-conditioned mu, past-five source, 50,000 updates at batch 8,192, then the
`z64_merged` 10B tracker stage).

## Why it is worth four arms

`z64_merged_noreg` against `z64_merged` put the regularizers at under 0.01
success rate on the concat head at matched frames, mean -0.0039 over 18
frames. That does not transfer to `phi = z`: with the identity phi and no
regularizer, nothing pins the scale of the latent against the scale of `E`,
and the regularizer-off PoE pretrain ended at a latent std of 0.17 against the
hub's 1.01. These arms measure what that costs.

Fresh pretrain per arm, so each pairing also carries encoder-init noise
(0.0064 SR / 0.46 mm / 10.8 mm, `2026-08-30-encoder-interface-500m`).

## Score

Wired into `2026-09-12-latent64-probe-live-eval`: four `*_curve` arms in
`2026-09-02-latest-eval` plus the launcher's and report's arm lists. The
shared 66-wide interface block applies unchanged; only the encoder file
differs per arm.

## Status

- 2026-09-12: submitted, eight ICE preflight checks each. 404 GB free on
  ice-shared at submission; 19 of my jobs were already queued. No local Isaac
  smoke (shared workstation); the pretrain stage is the smoke and `afterok`
  gates each chain.
