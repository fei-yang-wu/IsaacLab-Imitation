# 2026-08-18 — Joint velocities back into the leader's encoder input

Retrain the leading arm of `2026-08-15-latent-bottleneck-10b` with joint
velocities added back to the skill-encoder input. One arm, one variable, local
workstation (RTX PRO 6000), full 10B frame budget.

| arm | base row (frozen, 4,096 board) | the one variable |
|---|---|---|
| `cont_det_ln_hold1_fullbody` | `cont_det_ln_hold1`: 0.9368 SR / 22.86 mm at 10B | `expert_macro_state_terms`: `root_qpos` (38/frame, 380-wide encoder input) -> `full_body` (67/frame, 670-wide) |

The added channels are the 29 joint velocities. The actor command does not
change: 258-D (`z_dim` 256 + `sin_cos` phase), hold 1, stride 1,
`robot_heading` anchor. Encoder trunk, DiffSR heads, tuned low-level contract,
optimizer, seed, and data are byte-identical to the base campaign.

## Why

The DiffSR endpoint objective grounds the token in `p(s[t+H] | s[t], z)`. With
the `root_qpos` state, both boundary frames are position-only. The dynamics is
second order (Euler-Lagrange): the sufficient boundary state is position plus
velocity. With position-only boundaries, the same boundary position with a
different velocity has a different successor, so the density is forced to be
multimodal and `z` must carry the boundary velocity implicitly.

The prior datum for `root_qpos` (2026-08-04, LAFAN1, 500M, one seed) showed no
precision cost from dropping qvel, but it predates the BONES-SEED 129k data,
the tuned low-level contract, the 10B budget, and this arm's LN encoder. This
campaign re-measures the choice on the current leader.

## Protocol

- Pretrain: `train_hl_skill_diffsr.py`, deterministic z256, LayerNorm trunk
  2048/1024/512/512 silu, DiffSR feature 256 / embed 1024, 50,000 updates at
  batch 8,192, horizon 10, `intermediate` window — the base campaign's
  `pretrain_tail`/`pretrain_end` exactly, plus the `full_body` macro override.
- Low-level: IPMD tuned entry, 16,384 envs x 24 steps, 25,432 iterations =
  10,000,269,312 frames, minibatch 294,912, gamma 0.97, tuned rewards,
  termination curriculum 5M->30M, selection `random80_adaptive20`,
  Newton/MJWarp. Identical to the base `tuned_lowlevel` + `lowlevel_tail`.
- Local run, no walltime segmentation. `run.sh lowlevel` resumes from the
  tracker tree automatically; the budget continues through
  `cumulative_env_frames`.

```bash
./experiments/campaigns/2026-08-18-qvel-fullbody-leader/run.sh pretrain
./experiments/campaigns/2026-08-18-qvel-fullbody-leader/run.sh lowlevel
```

W&B: project `g1-bones-seed`, group `latent-bottleneck-10b` (the arm joins the
existing comparison table), runs `cont-det-ln-hold1-fullbody-pretrain-s0` and
`cont-det-ln-hold1-fullbody-s0`.

## Evaluation

Score on the frozen 4,096 board and `bones_testbed4096_v1` with the SAME two
overrides the g1-encoder-interface skill requires everywhere:

- `env.expert_macro_state_terms=[expert_motion,expert_anchor_pos_b,expert_anchor_ori_b]`
- `agent.ipmd.hl_skill_checkpoint_path=<this arm's encoder>`

`eval_scoreboard4096.sh` and `score_latent_arms_testbed.sh` assume the
`root_qpos` interface; add the override before scoring this arm, or the run
fails loudly with `hl/state shape mismatch: expected (N, 67)`.

Decision rule: compare against the frozen `cont_det_ln_hold1` row at matched
10B. One seed; treat a relative difference below ~15% as unresolved.

## Status

- 2026-08-18: campaign created. Encoder pretrained locally (`run.sh pretrain`,
  50k updates, input layer verified `(2048, 670)`).
- 2026-08-18: the local lowlevel attempt hit a Warp CUDA OOM at the first
  graph launch — the workstation GPU was shared with a running planner
  closed-loop eval (25.9 GB). Moved the lowlevel to ICE by user decision:
  one job with the full 10B budget, manual relaunch via
  `submit.sh --only-stage lowlevel_resume` until
  `cumulative_env_frames` reaches 10B. The local encoder was uploaded to
  `/data/qvel_fullbody_10b/.../encoder/checkpoints/latest.pt` and is the
  binding encoder for every segment.
- 2026-08-18 17:24: first submission RUNNING — ICE job 5580198 (H200,
  15:59:00). Expected ~6-7B frames in this walltime at the base campaign's
  ~100k fps; one resume should finish the budget.
- 2026-08-18 17:27: batch-scale arm at 24,576 envs failed twice and is
  retired. Job 5580199: W&B refuses tags over 64 chars and RLOpt's
  `logdir:` tag overflowed with the long run id (fixed with a per-arm
  `wandb_arm`). Job 5580202: real `Warp CUDA error 2: out of memory` at the
  Newton graph launch — 24,576 envs does not fit the H200.
- 2026-08-18 18:00: replacement `cont_det_ln_hold1_fullbody_env20k`
  submitted — ICE job 5580205. 20,480 envs (4096*5), frames/batch 491,520,
  20,346 iterations for the same 10B. Two variables against the
  base-campaign leader (qvel AND batch size); one variable (batch size)
  against `cont_det_ln_hold1_fullbody`. Same uploaded encoder file, own
  output tree, W&B run `cont-det-ln-fullbody-20k-s0`.

## Pipeline note (2026-08-18, post-launch)

`env.data.reference_prefetch_mode` moved `next` -> `next_and_reset` for all
future (re)submissions of this campaign, by explicit decision. This is a data-
plane change, not a training-value change: reset candidates are predicted from
the pre-physics SONIC distribution, so a reset draw sees sampler failure
weights that are one control step stale. Measured on the 8,192-env local
profile: env step 36.8 ms -> 34.5 ms (-6%), reference streaming otherwise
already fully overlapped (wait ~0.01 ms). The base campaign
(`2026-08-15-latent-bottleneck-10b`) ran `next`, so the "byte-identical
except macro terms" claim now carries this one additional pipeline delta.
`2026-08-18-sonic-reset-20b` stays pinned to `next` so its segments 6-8 match
segment 5 mid-chain.
