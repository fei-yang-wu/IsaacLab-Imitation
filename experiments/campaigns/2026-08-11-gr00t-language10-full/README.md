# 2026-08-11 — GR00T head, full-size training on the selected-ten language goals

Status: ACTIVE. Training arms launched on the local workstation
(MEL07876D, RTX PRO 6000 Blackwell 96 GB).

## Purpose

Train the verbatim GR00T N1.7 action head (warm-started from
`nvidia/GR00T-N1.7-3B`) as the language-conditioned planner for the ten
selected BONES-SEED motions, in six arms, then evaluate closed-loop.

Definitions used below:

- **chunk**: the head regresses the 30-frame expert `root_qpos` lookahead
  (`[30, 38]`). The tracker-side encoder turns each 10-frame slab into a
  latent at consume time.
- **latent (z256)**: the head regresses 3 consecutive published DiffSR
  latents (`[3, 256]`), each held 10 control steps. Slot k's target is the
  exact latent the oracle published at `t + 10k` (joined from the stored
  rows, never re-encoded from a shifted window — window frames are
  re-expressed against the query-time anchor, so shifting is invalid).
- **latent (fsq64)**: same 3-slot scheme, but the target is the FSQ
  encoder's PRE-quantization lattice-scaled value (`bound(z)/half_levels`,
  `[3, 64]`). Quantization to the lattice happens at the policy/runtime
  stage (SONIC convention). Recomputed offline from the stored encoder
  input windows; parity-gated so `quantize(recomputed) == stored z_target`.
- **rollout data**: state histories observed by the real robot during the
  oracle-policy collection (`oracle_rollout_state_history`).
- **mocap data**: state histories produced by kinematic reference replay
  (`env.replay_only=true`): the robot state is written from the reference
  each step, so the causal sensor pipeline reports expert kinematics. No
  policy-driven dynamics are involved. This is the "train on the mocap
  dataset" arm.

## Arms (6 = 3 targets x 2 data modes)

| arm | target | data | table |
| --- | --- | --- | --- |
| chunk_mocap | chunk | mocap | z256_mocap_table.pt |
| chunk_rollout | chunk | rollout | z256_rollout_table.pt |
| z256_mocap | latent 3x256 | mocap | z256_mocap_table.pt |
| z256_rollout | latent 3x256 | rollout | z256_rollout_table.pt |
| fsq64_mocap | latent 3x64 pre-quant | mocap | fsq64_mocap_table.pt |
| fsq64_rollout | latent 3x64 pre-quant | rollout | fsq64_rollout_table.pt |

Mocap arms run first (user priority), then rollout arms.

Training recipe: GR00T finetune defaults (AdamW fused, lr 1e-4, wd 1e-5,
cosine + warmup 0.05, batch 64, bf16 + tf32, grad clip 1.0, no EMA,
projectors + DiT together from step 0, state_dropout 0.2), 12,000 updates
(the requested 10-15k range), checkpoints every 1,000. Overfitting is
accepted; checkpoint selection happens at closed-loop evaluation.
W&B: project `g1-lafan1`, group `gr00t-language10-full`.

## Fixed inputs (all local on MEL07876D)

- z256 rollout collection (51,900 rows):
  `logs/bones_language10_oracle_pretrain_seed0/collection`
  (tracker `rollout24_gamma097_foot_disabled_eval/checkpoints/model_step_3500015616.pt`,
  encoder `rollout24_gamma097_foot_disabled_eval/encoder/latest.pt`).
- fsq64 rollout collection (51,836 rows):
  `logs/bones_language10_fsq64_planner_seed0/collection`
  (tracker `bones129k_sonic_fsq_scale_eval/4500357120/fsq64_sonic/model_step_4500357120.pt`,
  encoder `bones129k_sonic_fsq_scale_eval/encoders/fsq64_scaled.pt`).
- Mocap collections: produced by `collect_mocap_z256.sh` /
  `collect_mocap_fsq64.sh` — identical protocol plus `env.replay_only=true`,
  10 envs, 1 trajectory per motion (expert rows are deterministic; repeats
  add nothing).
- Goal features: `outputs/gr00t_language10/goal_features/goal_features.pt`
  (Cosmos-Reason2-2B, 10 goals, from the v1 manifest language sidecar).
- Warm-start trunk: `outputs/gr00t_language10/goal_features/action_head_trunk.pt`
  (nvidia/GR00T-N1.7-3B, filtered keep/fresh manifest).

## Verified semantics this campaign relies on

- Encoder input layout: `[state frame; 9 future frames frame-major]`
  (verified 8e-7 vs stored z256 targets; FSQ parity exact except 2/51,836
  borderline one-lattice-step CPU/GPU rounding rows).
- `encoder_input_packet_target` is the TERM-MAJOR packet (10x29|10x3|10x6)
  — not the encoder input layout.
- Window frames are query-time-anchored: consecutive rows' overlapping
  frames differ (median 0.03). Hence the row-join construction for slots.

## Workflow

```bash
# 1. Mocap collections (isaaclab env, ~minutes each)
./experiments/campaigns/2026-08-11-gr00t-language10-full/collect_mocap_z256.sh
./experiments/campaigns/2026-08-11-gr00t-language10-full/collect_mocap_fsq64.sh

# 2. Tables (default env)
./experiments/campaigns/2026-08-11-gr00t-language10-full/prepare_tables.sh

# 3. Train one arm (gr00t env)
./experiments/campaigns/2026-08-11-gr00t-language10-full/train_arm.sh z256_mocap
```

## Evaluation plan (next phase)

- Closed-loop tracker: `rollout24_gamma097` for chunk + z256 arms;
  `fsq64_sonic` for fsq64 arms (the "good fsq policy").
- FSQ consume-time snap happens in the runtime
  (`Embodied-Control` CommandContract `_snap_fsq`), never in the head.
- RTC comparison: each arm evaluated with RTC off (basic) and RTC on.
- EC (Embodied-Control) statistical eval after policy bundles for both
  trackers are exported on this host; synced closed-loop eval stays in
  IsaacLab-Imitation.

## Three-seed EC grid aggregate (2026-08-12)

Seeds 0-2, one 500-step episode per (goal, seed), MuJoCo rehearsal dynamics
— a deployment signal, not the number of record. 30 cells per variant.

| variant | basic | rtc |
| --- | --- | --- |
| chunk_mocap | 19/30 (379) | 19/30 (366) |
| chunk_rollout | 22/30 (426) | 18/30 (388) |
| z256_mocap | 17/30 (390) | 21/30 (392) |
| z256_rollout | 21/30 (453) | 12/30 (316) |
| fsq64_mocap | 30/30 (500) | 27/30 (494) |
| fsq64_rollout | 26/30 (481) | 30/30 (500) |

(cell = survived/30, mean steps in parentheses)

- fsq64 arms: 113/120 pooled vs 80/120 for chunk + z256 — large and
  consistent across seeds, but still confounded by the different tracker
  (`fsq64_sonic` vs `rollout24_gamma097`).
- The z256_rollout RTC drop replicates across seeds (21/30 -> 12/30); RTC
  effects on every other arm stay within +-4 cells.
- Mocap-only vs rollout training remains within noise in both directions.

## Tracker-confound resolution (2026-08-12, seeds 0-2)

The SAME chunk-head checkpoints, re-evaluated through the fsq64_sonic
tracker (`eval_ec_tracker_matched.yaml`; the chunk head is
interface-agnostic — its root_qpos frames feed whichever encoder the
bundle embeds):

| head | on rollout24 tracker | on fsq64_sonic tracker |
| --- | --- | --- |
| chunk_mocap | 19/30 | 30/30 |
| chunk_rollout | 22/30 | 26/30 |

Verdict: the survival gap belongs to the TRACKER, not to the FSQ command
interface or head training. With the tracker held fixed, the chunk head
matches the fsq64 arms (26-30/30).

The precision difference is ALSO tracker-dominated (seed 3, survivor-only,
one seed): chunk heads on fsq64_sonic score 71.3 / 86.6 mm MPJPE-L — the
same coarse range as the fsq64 heads on that tracker (73-85 mm), while the
same chunk heads on rollout24 scored 57-76 mm and z256 arms 48-58 mm.
Corrected campaign reading: rollout24_gamma097 is precise-but-fragile,
fsq64_sonic is robust-but-coarse, and at this data scale the head's
command interface moves both metrics far less than the tracker does.

## Seed-3 MPJPE pass (2026-08-12)

Same grid at seed 3 with `record_states` + post-hoc MPJPE-L (survivor
episodes only, FK through the MJCF, frame-0 anchor alignment). Table:
`logs/gr00t_language10_ec_eval/grid_v1_mpjpe/summary.md`.

Headline: the survival ranking INVERTS on tracking precision. fsq64 arms
survive most but track coarsest (MPJPE-L 73-85 mm); z256 arms track
tightest (48-58 mm) but fall more; chunk arms sit between (57-76 mm).
One seed; survivor-only averaging biases toward easy goals, so treat the
precision ordering as preliminary. Head latency: chunk 37 ms, latent 26 ms
(p50, RTX PRO 6000).

## First EC grid result (2026-08-12, PRELIMINARY)

One seed, one 500-step episode per goal, MuJoCo rehearsal dynamics, flow
sampling unpinned — a deployment signal, not a paper number. All six arms
at update 12,000. Full table: `logs/gr00t_language10_ec_eval/grid_v1/summary.md`.

| variant | survived | mean steps | head p50 ms |
| --- | --- | --- | --- |
| chunk_mocap basic / rtc | 6/10 / 7/10 | 359 / 387 | 37.2 |
| chunk_rollout basic / rtc | 7/10 / 6/10 | 417 / 385 | 37.2 |
| z256_mocap basic / rtc | 6/10 / 7/10 | 376 / 395 | 25.8 |
| z256_rollout basic / rtc | 7/10 / 3/10 | 460 / 304 | 25.8 |
| fsq64_mocap basic / rtc | 10/10 / 9/10 | 500 / 489 | 26.3 |
| fsq64_rollout basic / rtc | 9/10 / 10/10 | 486 / 500 | 26.3 |

Preliminary reading:

- The fsq64 arms survive 38/40 pooled cells vs 25/40 for the chunk + z256
  arms together. CONFOUND: the fsq arms run on a different tracker
  (`fsq64_sonic` 4.5B) than the chunk/z256 arms (`rollout24_gamma097`
  3.5B), so head quality and tracker robustness are not separated.
- Mocap-only training holds up: mocap arms are within one goal of their
  rollout counterparts everywhere (fsq64_mocap even 10/10 basic).
- The one large RTC effect (z256_rollout 7/10 -> 3/10) is one seed and
  below the known evaluation noise until repeated.

## Next phase (resume plan, not started)

1. Isaac number-of-record eval: load the trained head inside the isaaclab
   environment (verified import path: `gr00t.model` stub per
   `gr00t_head.ensure_gr00t_importable`), publish latents through the v2
   env's `hl_skill` sampler surface on the per-environment renewal
   schedule, evaluate with the M3 survival definition plus the
   non-terminating diagnostic pass. Entry point candidate: extend
   `scripts/rlopt/eval_skill_commander_closed_loop.py` with a
   `--gr00t_checkpoint` planner source (its `--planner_checkpoint` path
   loads the medium flow planner and does not fit the GR00T head).
2. Tracker re-pairing: export a bundle for a stronger z256 tracker
   generation (e.g. `bones129k_recent_ice/z256_scaled` 5750M — on the
   other workstation) and repeat the latent arms against it. The
   tracker-dominance finding predicts this closes most of the survival
   gap while keeping the ~48-58 mm precision.
3. Scale-up only after 1-2: the grid says tracker quality, not head/data
   scale, is the binding constraint at 10 goals.

## Notes

- The old `2026-08-10-gr00t-planner-local-debug/run.sh` uses the pre-Hydra
  trainer CLI and is historical; this campaign supersedes it.
- All paths live in `conf/*.yaml` — the trainer and preparation
  entrypoints contain no pinned paths.
