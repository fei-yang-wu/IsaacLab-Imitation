# Variable-Window Skill Encoders

Evidence page for the variable-window program of 2026-09-05 to 2026-09-06:
one skill encoder that serves several reference-window lengths, six
mechanisms screened at 5B on the `combo` hub, and the decision to park the
line with the attention trunk (`vw_sequence`) named as the future extension.

**Status: PARKED 2026-09-06 (user decision).** The six arms are trained and
scored at one seed. No arm is promoted, nothing enters a paper table, and no
further job is planned. The code stays on branch `feat/variable-horizon`
(RLOpt `14c8367`, worktree `.claude/worktrees/varwin`), unmerged.

## 1. Question

The hub encoder reads one fixed 10-frame reference window `s[t+1..t+9]`
(the endpoint `s[t+10]` hidden) and its merged head denoises `s[t+10..t+20]`.
Prior evidence on window length, one seed each on `bones_testbed4096_v1`
clean: horizon 5 inside noise of horizon 10 (`g4_h5` 0.9158 / 25.93 / 96.84
against the hub 0.9280 / 24.42 / 98.98 at 2B), horizon 20 collapsed
(`g4_h20` 0.5669 at 2B), horizons 1 and 2 collapsed at the 500M screen
(0.7878, 0.7891 against 0.8765). The open question was whether ONE code
space can span windows from 2 to 15 frames without the long-window collapse,
and whether the window can be chosen at rollout.

## 2. Mechanism

RLOpt `hl_skill_diffsr.py` / `hl_skill_encoder.py`, all behind config
fields so the fixed-window path is byte-identical to every existing
checkpoint:

- `horizon_choices` (the horizon SET, maximum = `horizon_steps`) or
  `stride_choices` (a frame-stride set on a fixed frame count). Each pretrain
  row draws one member; the macro batch is sampled at the widest span and
  sliced per row.
- Trunks: `flat` (the hub MLP), `padded` (the same MLP over the window
  zero-masked past the drawn length plus a one-hot of the member),
  `sequence` (one token per frame, learned position and member embeddings,
  pre-LN transformer, key-padding mask past the drawn length, mean pool over
  valid tokens).
- `horizon_target_mode`: `follow` gives one merged diffusion head per
  member, denoising `s[t+h..t+2h]`; `fixed` keeps one head at
  `horizon_fixed_target_steps` for every drawn length.
- `horizon_code_layout`: `flat`, `nested` (head for member i reads the
  prefix `z[: z_dim * i / n]`), `block` (head i reads its own disjoint
  block); both need `horizon_input_mode=full` (the encoder always sees the
  maximum window and the member selects the head only).
- SIGReg is applied to the pooled code across members, so every member
  shares one marginal.
- Rollout: `agent.ipmd.hl_skill_live_horizon` = a member, `base` (maximum
  horizon, stride 1, or every block), `episode` (per environment, redrawn at
  reset) or `step` (redrawn at every renewal). The frozen sampler draws the
  member per environment, requests the maximum window, masks or strides it,
  and masks the code for the non-flat layouts. `hl_skill_horizon_steps` must
  equal the checkpoint's maximum horizon.

Tests: `RLOpt/tests/test_hl_skill_variable_window.py` (44).

## 3. Campaign

`experiments/campaigns/2026-09-05-variable-window-5b/` and
`2026-09-06-variable-window-5b-eval/`. Tracker recipe = `combo`
(`2026-09-01-latent64-probe-10b`) field for field: past-5 affine phi
encoder pretrain (merged head, 64-D, LayerNorm, `intermediate` window,
50,000 updates at batch 8,192), MLP actor with the ten-step proprio
history, full-batch entry point, `optim.weight_decay=1e-2`, linear critic
decay to 1e-5, `random80_adaptive20` resets with the 5M-30M termination
curriculum, 16,384 x 24 frames per batch, hold 1, sin_cos phase, 5B frames,
checkpoints every 200M. W&B project `g1-bs-vw`, group `variable-window-5b`.

| arm | pretrain change | live |
|---|---|---|
| `vw_padded` | horizon set {2,5,10,15}, padded MLP + one-hot, head per h | 10 |
| `vw_sequence` | same set, attention trunk (width 384, depth 4, heads 6, about the MLP's parameter count) | 10 |
| `vw_fixed_target` | same set on the input, ONE head at `s[t+10..t+20]` | 10 |
| `vw_stride` | stride set {1,2,3} on 10 frames (0.2 / 0.4 / 0.6 s), head per stride | 1 |
| `vw_nested` | full 15-frame input, prefix code 16 / 32 / 48 / 64 | base |
| `vw_block` | full 15-frame input, disjoint 16-dim block per h | base |

Each arm differs from `combo` in the encoder pretrain and the live policy it
implies, and in nothing else; every new pretrain also carries
encoder-initialization noise (0.0064 SR / 0.46 mm L / 10.8 mm G across three
replicates of one recipe, `2026-08-30-encoder-interface-500m`).

Cost: pretrain peak 90-92 GB at batch 8,192 on every arm (the affine phi
materializes `[rows, 1024, 256]` per head), so a 96 GB card holds the MLP
arms only; encoder checkpoints 8-11 GB (heads' AdamW state). Tracker wall
time at 5B: MLP arms 9:28-9:32, `vw_stride` 10:33, `vw_sequence` 13:22.

## 4. Results (one seed, 5B checkpoint, 4096 clean, DR off)

`bones_testbed4096_v1`, `--randomization none`, star-v2 curves evaluator
arguments verbatim. `clean` = the arm's train live policy; `live<m>` = the
same tracker and encoder with the frozen encoder at member m. Rows on
ice-shared `eval/variable_window_5b/` and in the worktree's
`logs/variable_window_5b_eval/`.

| arm | row | SR | MPJPE-L | MPJPE-G | acc | jerk |
|---|---|---:|---:|---:|---:|---:|
| combo (control, 5B checkpoint of the 10B run) | clean | 0.9009 | 24.27 | 101.24 | 4.82 | 212.2 |
| vw_padded | clean (10) | 0.8938 | 24.52 | 91.52 | 4.86 | 214.9 |
| vw_sequence | clean (10) | 0.9058 | 23.47 | 101.87 | 4.83 | 211.2 |
| vw_fixed_target | clean (10) | 0.8896 | 25.48 | 82.30 | 4.66 | 200.4 |
| vw_stride | clean (1) | 0.9001 | 24.50 | 117.24 | 4.79 | 208.1 |
| vw_nested | clean (base) | 0.8857 | 26.43 | 130.71 | 4.82 | 209.2 |
| vw_block | clean (base) | 0.8125 | 34.36 | 182.80 | 4.66 | 196.2 |

Live-policy matrix (SR / MPJPE-L / MPJPE-G):

| arm | live 2 | live 5 | live 10 | live 15 |
|---|---|---|---|---|
| vw_padded | 0.0530 / 132.2 / 815 | 0.4448 / 75.9 / 443 | clean | 0.0000 / - / - |
| vw_sequence | 0.6628 / 46.6 / 253 | 0.8257 / 35.9 / 150 | clean | 0.0654 / 135.6 / 1556 |
| vw_fixed_target | 0.6389 / 44.8 / 151 | 0.7949 / 35.0 / 96 | clean | 0.8828 / 28.2 / 112 |
| vw_nested | 0.4500 / 70.9 / 532 | 0.5649 / 54.9 / 399 | 0.6118 / 51.5 / 256 | clean |
| vw_block | 0.3706 / 80.1 / 460 | 0.2065 / 81.4 / 1071 | 0.2219 / 77.0 / 1402 | 0.1724 / 87.0 / 1123 |
| vw_stride | stride 2: 0.5930 / 47.1 / 171 | stride 3: 0.4075 / 65.4 / 252 | | |

Qualifications: single seed; one checkpoint per arm, no curve points scored;
`vw_padded` live 15 had zero successes of 4096, so its MPJPE is undefined;
`combo`'s 10B endpoint (0.9214 / 22.64 / 88.52) is a different frame count
and not the matched row. sonic_v1_1 on this board: 0.9888 / 26.73 / 187.7.
The clean rows of the four MLP-trunk arms and of `vw_sequence` sit within
about 0.01 SR of the control, inside the range the evaluation noise leaves
unresolved.

## 5. Decision and future extension

User decision 2026-09-06: the attention trunk is the way to read a
variable-length window (the padded MLP needs a one-hot and a fixed slot
layout; the code layouts and the stride set do not need it and did not
help), and `vw_sequence` is the arm to promote to a longer budget. For this
work the line is PARKED as a future extension; nothing here enters the
paper.

What a promotion would run, none of it started:

- `vw_sequence` at the headline budget under the `combo` chain (10B, then
  the 50B regime), against `combo` at the matched frame count.
- Seeds: the clean-row gap to the control is inside noise at one seed.
- The second-wave live-policy arms (`vw_padded_live_step` / `episode` / `15`
  / `5` in the 5B campaign, plannable now that the padded encoder exists),
  rerun on the sequence encoder: does a mixed-window rollout policy hold the
  clean row?
- Curve points at 200M for the six arms (25 checkpoints each exist).
- Throughput: the attention trunk costs about 40% of tracker fps at 16,384
  environments; a fused or cached token path would be needed before a 50B
  chain.

Ops notes worth keeping: a RUNNING stage's slurm log is silent until exit
(container stdout is block-buffered), so progress is read from
`checkpoints/best.pt.json` or W&B; one pretrain hung before its first update
on `atl1-1-03-013-26-0` and was resubmitted with the node excluded; the Kit
boot crash hit 3 of 25 evaluation jobs and a plain resubmit fixed each.
