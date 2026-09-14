# 2026-09-14 expert-heading encoders, 10B fine-tune off the e5 hub (CANCELLED)

Cancelled 2026-09-14 10:35 EDT (user): the encoder-swap re-fit was far
behind the e5 hub at every row. Last Isaac rows (4096 clean board, single
seed): eh 74.0B SR 0.8933 / L 38.84 / G 291 (trend 0.67 / 0.82 / 0.86 /
0.89 at 71 / 72 / 73 / 74B), eh_ee 71.5B SR 0.4839 / L 61.99 / G 786.
Plant rows: eh 74.5B 29/43 clean, L 49.4. Jobs 5777401/5777402 (eh),
5778168/5778169 (eh_ee) and their pending evals cancelled. The encoders
(`/data/expert_heading_encoders_10b/<arm>_seed0/encoder/checkpoints/latest.pt`)
and the tracker trees stay on ICE; the code paths (50-wide compact cache,
restore flag, EC expert_heading reference check) stay in the repo.

Submitted 2026-09-13 23:17 EDT from a clean worktree of commit 2c66df2 (no
drift), partition coe-gpu:

| arm   | pretrain | finetune1 | finetune2 |
| ----- | -------- | --------- | --------- |
| eh    | 5777400  | 5777401   | 5777402   |
| eh_ee | 5778167  | 5778168   | 5778169   |

The first submission (jobs 5776810-5776816, 2026-09-13 21:58 EDT) died at once:
`scripts/rlopt/train_hl_skill_diffsr.py` passed `diffsr_mu_conditioning`
to `HighLevelSkillDiffSRConfig`, a field that exists only on RLOpt
`feat/poe-identity-phi`, not on the dev pointer a0add23. The local smokes had
run against the main tree's RLOpt working copy, which carried that field
uncommitted. Fix 2c66df2 passes the field only when RLOpt defines it; the
smokes were repeated from a clean worktree against a0add23 (both pretrains
and both 3-iteration fine-tunes, 0 errors) before the resubmission.

Two encoder-swap fine-tunes. Each arm first pretrains a p5_affine skill
encoder whose window is anchored to the window's own first frame
(`env.expert_macro_anchor_mode=expert_heading`), then fine-tunes the e5 hub
(`2026-09-13-e5-hardware-gap-10b`, 70,000,312,320 frames) onto that encoder
for 10B more frames.

## Why

The h1 arm of `2026-09-13-e5-hardware-gap-10b` could not start: the
p5_affine encoder records `macro_anchor_mode=robot_heading`, and the
binding guard (`_require_matching_macro_anchor_mode`) refuses an
`expert_heading` environment. Under `expert_heading` the encoder never reads
the robot's position, so hardware needs no localization and the latents can
be computed offline, which is SONIC's actor design.

The second arm adds the four end-effector positions (left and right
`ankle_roll_link`, left and right `wrist_yaw_link`, term `expert_ee_pos_b`,
in the window's anchor frame) to the encoder input, 50 values per frame
instead of 38 (500-wide window instead of 380). The latent then carries where
the feet and hands are in the reference, which is the part of the reference
the tracker drags on in the EC plant rehearsals.

The eh_ee pretrain of the resubmission (5777404) also died at once:
`env.data.macro_cache_device=cuda:0` (the compact GPU macro cache the
cluster data args select) accepted only the root_qpos and full_body term
sets. The local smokes had used a manifest, which never builds that cache.
Fix: the compact cache now also carries the command end-effector world
poses when the macro terms are root_qpos+ee (`expert_data_plane.py`,
`_ROOT_QPOS_EE_MACRO_TERMS`), verified by unit tests against the
replay-window terms and by a pretrain + 3-iteration fine-tune smoke on a
local reference-array store with `macro_cache_device=cuda:0` and RLOpt
a0add23. eh_ee was resubmitted a third time (01:19 EDT, 2026-09-14) from commit
adee385: pretrain 5778167, finetune1 5778168, finetune2 5778169. The eh
chain (2c66df2) was not touched: its pretrain finished in 1 h 32 min and
its finetune1 (5777401) started 2026-09-14 01:08 EDT.

## Arms

| arm   | macro state per frame                                   | width | anchor mode    |
| ----- | ------------------------------------------------------- | ----- | -------------- |
| eh    | `expert_motion_qpos, expert_anchor_pos_b, expert_anchor_ori_b`                   | 38 | expert_heading |
| eh_ee | `expert_motion_qpos, expert_anchor_pos_b, expert_anchor_ori_b, expert_ee_pos_b`  | 50 | expert_heading |

The two arms differ from e5 in the encoder (new weights, new anchor mode)
and, for `eh_ee`, in the encoder input. Everything else is e5's own
argument list: `rate05` action-rate penalty -0.5, energy -1e-4, frozen
normalizer (`agent.ppo.update_normalizers_after_rollout=false`),
`adaptive_uniform_ratio=0.2`, `adaptive_failure_rate_max_over_mean=50.0`.
The r1 reset values of the e5 campaign are not applied.

## Chain per arm

1. `pretrain`: `scripts/rlopt/train_hl_skill_diffsr.py`, the 2026-08-30
   p5_affine recipe verbatim (`--source_history_steps 5 --source_anchor
   current`, `jepa_ntp` / `sigreg_ebm` / `diff_chunk` / `boundary_next`,
   affine phi, z 64, 50,000 updates of 8,192), about 5 h on one H200.
   Output `<output_root>/encoder/checkpoints/latest.pt`.
2. `finetune1` (afterok pretrain): `scripts/rlopt/train.py` from the e5 hub
   file `model_step_70000312320.pt` with
   `agent.ipmd.hl_skill_checkpoint_path=<new encoder>` and
   `agent.ipmd.hl_skill_restore_from_checkpoint=false` (RLOpt a0add23).
   Without that flag `load_model` restores the tracker checkpoint's embedded
   encoder, which for `eh` silently replaces the new weights (same width)
   and for `eh_ee` fails on a 380-vs-500 size mismatch.
3. `finetune2` (afterany finetune1): resume from the arm's own tracker tree.

Frame cap 80,000,312,320 (e5's 70B plus 10B), `max_iterations` 203,452 at
16,384 envs x 24 steps. The tracker's latent input changes meaning under the
new encoder, so the first ~1B is a re-fit; read rows from 3B on.

W&B project `g1-bs-finetune`, group `expert-heading-encoders-10b`, run ids
`ehenc-eh-s0` and `ehenc-ehee-s0`. Output root
`/data/expert_heading_encoders_10b/<arm>_seed0` on ICE.

## Qualification

RLOpt a0add23 adds the 50-wide (`root_qpos` + ee) heading re-anchoring
layout to `hl_skill_diffsr.py` (the trailing 12 ee values rotate and
translate like the anchor position) and the restore flag. Local smokes on
2026-09-13: 20-update pretrains for both encoders and 3-iteration
fine-tunes from the e5 hub file, both arms, 0 errors.

## Commands

```bash
python -m imitation_experiments.pipeline.cluster plan \
  --campaign experiments/campaigns/2026-09-14-expert-heading-encoders-10b/campaign.yaml \
  --arm eh --seed 0
python -m imitation_experiments.pipeline.cluster submit --plan <plan dir> --confirm <PLAN_SHA>
```

Submit from a clean detached worktree of the committed SHA so the packed
workspace does not carry unrelated working-tree edits.

## Evaluation hookup

- Isaac board: `2026-09-02-latest-eval` arms `ehenc_eh_live` and
  `ehenc_eh_ee_live` (rows `ehenc_<arm>_seed0_clean_f<frames>.json`). Each
  scores with its own encoder file and `anchor_mode: expert_heading`;
  `ehenc_eh_ee_live` also sets the scorer's `macro_terms` var (50-wide
  `env.expert_macro_state_terms`). `./submit_live_eval.sh` relinks the
  newest checkpoint into `/data/expert_heading_encoders_10b_live` and
  submits from a clean detached worktree of HEAD, not from a `git stash`,
  because the scorer needs RLOpt a0add23.
- EC plant: `./ec_grade_echost.sh` grades `eh` on echost with the arm's
  own encoder (downloaded once to `logs/expert_heading_encoders_ec/encoders`),
  `--anchor expert_heading`, cuda inference, measured plant noise. Rows in
  `logs/expert_heading_encoders_ec/results_echost.tsv`, same columns as the
  e5 pass. `eh_ee` is refused: the EC native oracle builds the 38-per-frame
  window and has no `expert_ee_pos_b` terms yet.
