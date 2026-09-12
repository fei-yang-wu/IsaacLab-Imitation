# 2026-09-12 -- expert count against latent width in the PoE score

Three arms, one seed each, 10B, W&B group `latent64-probe-10b`. The question
the two running PoE arms cannot answer: in `<phi(z), E(s, s')>` both have
expert count equal to latent width, so a difference cannot be attributed to
either.

| arm | phi | z_dim | experts | base term | command | job |
|---|---|---:|---:|---|---:|---|
| `z64_poe` (running) | `z` | 64 | 64 | no | 66 | 5764382 |
| `z64_poe_base` (running) | `[1; z]` | 64 | 64 + base | yes | 66 | 5766787 |
| `poe_proj256` | `A z` | 64 | 256 | no | 66 | 5766921-23 |
| `poe_proj256_base` | `[1; A z]` | 64 | 255 + base | yes | 66 | 5766924-26 |
| `poe_z256` | `z` | 256 | 256 | no | 258 | 5766930-32 |

What each contrast moves:

- `poe_proj256` against `z64_poe`: expert count 64 to 256 at a FIXED 64-D
  command. The tracker interface is byte-identical, so only the encoder's
  expert bank widens.
- `poe_proj256_base` against `poe_proj256`: the base term alone, the same
  contrast `z64_poe_base` makes at 64 experts.
- `poe_z256` against `poe_proj256`: 256 experts either way, but the command
  itself becomes 256-D. This is the "just scale up z" arm, and it is the only
  one that moves the tracker interface (command 258 against 66).

`A` is bias-free in both projected arms. `linear_bias` carries its
command-independent term in the constant channel and `linear` deliberately
has none, so an affine `A z + b` would confound the two.

## Implementation

`RLOpt/rlopt/agent/ipmd/module.py`: `phi_parameterization="linear"`
(`phi = A z`, `A` a bias-free `Linear(z_dim, feature_dim)`) and
`"linear_bias"` (`phi = [1; A z]`, `A` into `feature_dim - 1`). Both sit
beside the user's `identity` / `identity_bias`. Nine tests in
`RLOpt/tests/test_ipmd_components.py`; 28 phi/PoE tests pass.
`--diffsr_phi_parameterization` gained the two choices.

## Held fixed

`z64_poe_base` verbatim: merged `diff_chunk` head, `boundary_next`, endpoint
coefficient 0, pair-conditioned mu, BOTH regularizers off
(`--jepa_sigreg_coeff 0 --reg_coeff 0`), past-five source
(`--source_history_steps 5 --source_anchor current`), 50,000 updates at batch
8,192, then the `z64_merged` 10B tracker stage (20,480 x 24,
`random80_adaptive20`, 5M-30M curriculum, ee + wide rewards,
`action_rate_l2` -0.03, full-batch entry point, checkpoint every 500M).
Fresh pretrain per arm, so every comparison also carries encoder-init noise
(0.0064 SR / 0.46 mm / 10.8 mm, `2026-08-30-encoder-interface-500m`).

Output on ice-shared, `latent64_probe_10b/<arm>_seed0/`.

## Score

`2026-09-12-latent64-probe-live-eval/submit_live_eval.sh` after adding the
arms to its list and to `2026-09-02-latest-eval` as `*_curve` arms. Note
`poe_z256` needs its own interface block there: command 258, latent_dim 258,
`code_latent_dim` 256.

## Status

- 2026-09-12: submitted. All three plans passed the eight ICE preflight
  checks; the resolved pretrain scripts carry the right phi, feature dim, and
  z dim, and the resolved tracker scripts carry command 66 / 66 / 258. No
  local Isaac smoke (shared workstation); the pretrain stage is the smoke and
  `afterok` gates each chain.
