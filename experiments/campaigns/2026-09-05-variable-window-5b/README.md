# 2026-09-05 -- Variable-window skill encoders on the `combo` hub (5B)

Six encoders, each pretrained on a SET of window sizes instead of the hub's
one fixed 10-frame window, each deployed under the `combo` tracker recipe at
5B frames (user, 2026-09-05). The question: can one skill code serve several lookahead
lengths, and does that change tracking against the fixed-window control?

## Control

`combo` from `2026-09-01-latent64-probe-10b` (W&B `l64p-combo-s0`), one
seed, `bones_testbed4096_v1` clean at 10B: 0.9214 SR / 22.64 MPJPE-L /
88.52 MPJPE-G. Recipe, copied field for field into `campaign.yaml`:

| field | value |
|---|---|
| encoder | past-5 affine phi (`--source_history_steps 5 --source_anchor current --diffsr_phi_parameterization affine`), merged head (`diff_chunk`, `boundary_next`, endpoint coeff 0), 64-D deterministic, LayerNorm, `intermediate` window, 50,000 updates at batch 8,192 |
| command | `z` + sin/cos phase, width 66, hold 1 |
| actor | MLP 2048/2048/1024/1024/512/512 silu, ten-step history on the five policy terms; critic single frame |
| optimizer | `rlopt_ipmd_tuned_fullbatch_cfg_entry_point`, `optim.weight_decay=1e-2`, critic lr linear to 1e-5 over the frame cap |
| resets | `random80_adaptive20`, termination curriculum 5M-30M |
| rewards | motion_ee_pos 1.0, motion_global_anchor_pos_wide 1.0, tracking_reward_points 4.0, action_rate_l2 -0.03 |
| scale | 16,384 x 24 = 393,216 frames per batch; here a 5B cap (combo ran 10B; read it at its 5B checkpoint), two chained 15:59 segments, checkpoints every 200M |

## Terms

- **Window** (`horizon_steps`, h): the encoder sees the reference frames
  `s[t+1..t+h-1]` (mode `intermediate` hides `s[t+h]`) and its merged head
  denoises `s[t+h..t+2h]`. The hub has h = 10.
- **Horizon set**: {2, 5, 10, 15}. In pretraining each row draws one member
  uniformly. `horizon_steps` is the maximum (15) and the checkpoint records
  the set.
- **Live policy** (`agent.ipmd.hl_skill_live_horizon`): which member the
  frozen encoder uses at rollout. A member value, `base` (maximum horizon,
  stride 1, or every block), `episode` (per environment, redrawn at reset),
  or `step` (redrawn at every code renewal, every step at hold 1).
- **Variant one-hot**: the padded MLP trunk receives the window zero-masked
  past the row's visible length plus a one-hot of the drawn member.

## Arms

| arm | pretrain change vs the control | live | W&B id |
|---|---|---|---|
| `vw_padded` | horizon set; padded MLP trunk with a one-hot; one merged head per h | 10 | `vw5b-padded-s0` |
| `vw_sequence` | horizon set; attention trunk over frame tokens (width 384, depth 4, heads 6), key-padded past h-1, mean-pooled | 10 | `vw5b-sequence-s0` |
| `vw_fixed_target` | horizon set on the input only; ONE head at `s[t+10..t+20]` for every drawn h | 10 | `vw5b-fixedtgt-s0` |
| `vw_stride` | stride set {1, 2, 3} on a 10-frame window (0.2 / 0.4 / 0.6 s), gathered in RLOpt from a stride-1 macro window; one head per stride | 1 | `vw5b-stride-s0` |
| `vw_nested` | full 15-frame input, one code; the head for h reads the prefix `z[:16 * i]` (16 / 32 / 48 / 64 dims) | base | `vw5b-nested-s0` |
| `vw_block` | full 15-frame input, one code; the head for h reads its own disjoint 16-dim block | base | `vw5b-block-s0` |

Each arm differs from `combo` in the encoder pretrain and in the live policy
that pretrain implies, and in nothing else. The horizon arms deploy at 10 so
the tracker's command comes from the same window as the control's;
`vw_stride` deploys at stride 1 for the same reason. `vw_nested` and
`vw_block` produce one code for every scale, so `base` (the full code) is
their deployable command. SIGReg is applied to the pooled code across the
set in every arm, so short-window and long-window codes share one marginal.

Every new pretrain carries encoder-initialization noise: 0.0064 SR /
0.46 mm MPJPE-L / 10.8 mm MPJPE-G across three replicates of one recipe
(`2026-08-30-encoder-interface-500m`).

### Second wave: the live-policy axis

`vw_padded_live_step`, `vw_padded_live_episode`, `vw_padded_live15`,
`vw_padded_live5` reuse `vw_padded`'s encoder and differ from `vw_padded` in
the live policy alone. They have no pretrain stage, so `plan` refuses them
until `vw_padded`'s pretrain has written `latest.pt` on the cluster.

### Evaluation matrix

Evaluate every tracker at each member of its set by overriding the live
policy on the evaluator command line, with the star-v2 curve evaluator
arguments otherwise verbatim:

```bash
agent.ipmd.hl_skill_live_horizon=5     # 2, 5, 10, 15 for horizon arms
agent.ipmd.hl_skill_live_horizon=2     # 1, 2, 3 for vw_stride
```

Report the train-policy x eval-policy table per arm with SR, MPJPE-L and
MPJPE-G, plus the `sonic_v1_1` row for the same board.

## What the code does

RLOpt (`rlopt/agent/hl_skill_diffsr.py`, `rlopt/agent/hl_skill_encoder.py`):

- `HighLevelSkillDiffSRConfig.horizon_choices`, `stride_choices`,
  `horizon_input_mode`, `horizon_encoder`, `horizon_target_mode`,
  `horizon_fixed_target_steps`, `horizon_code_layout`, `sequence_*`.
  Validation pins the merged chunk recipe and the layout rules.
- `HighLevelSkillEncoder` trunks `flat` (unchanged, byte-identical to every
  existing checkpoint), `padded`, `sequence`; `encode(..., lengths, variant)`.
- `HighLevelSkillDiffSRTrainer._jepa_variant_train_step`: one macro batch
  at the widest span the set needs, per-row draw, per-head loss weighted by
  the head's share of the batch, per-member losses logged as
  `train/jepa_ntp_loss/h5`; evaluation scores every member on its own batch
  (`train/jepa_ntp_loss_eval/h5`, `train/z_effective_rank/h5`) and reports
  the mean under the usual keys.
- `FrozenHighLevelSkillCommandSampler(live_horizon=...)` draws the live
  member per environment, requests the maximum window from the data plane,
  masks or strides it per environment, and masks the code for the nested and
  block layouts. `agent.ipmd.hl_skill_live_horizon` feeds it. Finetuning and
  the `phi` command refuse a variable-window checkpoint.

CLI flags on `scripts/rlopt/train_hl_skill_diffsr.py` mirror the config.
Tests: `RLOpt/tests/test_hl_skill_variable_window.py` (registered in
`pixi run test-rlopt`).

## Cost

- **GPU memory.** The affine phi materializes a `[rows, 1024, 256]` tensor
  per head, so the pretrain peak at batch 8,192 is 90-92 GB for every arm
  (`train/gpu_max_allocated_gb`, local smoke). A 96 GB card holds the MLP
  arms with about 2 GB to spare and does not hold `vw_sequence`; the ICE
  H200 (141 GB) holds all of them. The local smoke therefore runs
  `vw_sequence` at batch 4,096, a wiring check only.
- **Checkpoints.** One merged head per member: an encoder checkpoint is
  8-11 GB (`vw_fixed_target`, one head: 3.9 GB) because it carries the
  heads' AdamW state; `best.pt` and `latest.pt` are both written, so up to
  22 GB per arm. The output root is the shared 2 TB allocation.

## Local smoke (2026-09-05)

`smoke.sh` on the workstation, real data plane, 4 pretrain updates at batch
8,192 (`vw_sequence` 4,096), one 128-frame tracker iteration at each arm's
live policy: `vw_padded`, `vw_fixed_target`, `vw_stride`, `vw_nested`,
`vw_block` pass; `vw_sequence` at 8,192 dies on CUDA OOM at 91.8 GB on the
96 GB card, see above. Per-member eval keys (`train/jepa_ntp_loss_eval/h5`,
`train/z_effective_rank/h5`) appear in every log.

## Decision (2026-09-06)

PARKED by the user after the 5B rows (`2026-09-06-variable-window-5b-eval`).
`vw_sequence` is the named future extension for a longer budget. Record:
`wiki/variable-window-encoders.md`.

## Run

```bash
./experiments/campaigns/2026-09-05-variable-window-5b/smoke.sh   # local wiring check
./experiments/campaigns/2026-09-05-variable-window-5b/submit.sh vw_padded 0
# then the printed `submit --confirm <PLAN_SHA>` line
```

Outputs: `/storage/ice-shared/vip-vwt/scratch-fwu91/variable_window_5b/<arm>_seed<seed>/{encoder,tracker}`.
W&B project `g1-bs-vw`, group `variable-window-5b`.
