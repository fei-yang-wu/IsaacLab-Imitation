# 2026-08-23 — Promote the screen winner to 20B

Train `ema_h1_ee_wide` — the `2026-08-22-pareto-stack` screen winner under its
post-terminology name — to 20B on the same schedule `ln_hold1_sonicreset`
followed, so the 20B rows compare directly. This is the candidate to replace
`cont_det_ln_hold1` as the paper headline low-level row.

## Recipe

Encoder: pretrained by `interface-combos/jepa_ebm_hold1_256d`
(`--transition_objective jepa_ntp --jepa_loss sigreg_ebm
--jepa_target_encoder_mode ema --latent_mode deterministic --z_dim 256
--encoder_layer_norm`) = DiffSR endpoint denoising + EMA token-prediction +
SIGReg. Loaded through `agent.ipmd.hl_skill_checkpoint_path`, frozen by
`agent.ipmd.hl_skill_finetune_enabled=false`; this campaign trains only the
IPMD tracker.

Interface: hold 1, 258-D command (256 code + `sin_cos` phase), 380-wide
`root_qpos` macro state, `robot_heading` anchor, stride 1.

Rewards: `G1V2TunedRewardsCfg` plus `env.rewards.motion_ee_pos.weight=1.0`
(pelvis-relative wrist error, `std=0.1`) and
`env.rewards.motion_global_anchor_pos_wide.weight=1.0` (world-frame pelvis
error, `std=0.5`), with `action_rate_l2=0.0` and
`tracking_reward_points=4.0`.

Schedule: `std1`-`std3` train 0 -> 10B under `random80_adaptive20` with the
termination curriculum; `sonic1` switches to `selection=sonic` and ramps
`adaptive_uniform_ratio` 0.5 -> 0.1 over 2.5B frames; `sonic2`-`sonic3` pin
0.1 (`adaptive_uniform_ratio_final=null`) to 20B. The ramp keys off
`common_step_counter`, which restarts each segment, so it must complete inside
`sonic1` and later segments must pin — leaving the ramp on would re-sweep it.

## Row to beat

`ln_hold1_sonicreset` @20B: 0.9558 SR / 22.15 mm local / 168.15 mm global
(canonical 4,096 board, `no_push`, one seed).

## INCIDENT: NaN collapse at 6.9B, restarted from zero (2026-08-25)

The first attempt (jobs 5590001-06) trained cleanly to 6.89B — `r_step`
0.2545, `ep_len` 182.8, `pi_loss` -0.0074 — then went non-finite within 25
iterations, at 6.90B, with no precursor in any logged quantity. Training
continued at full speed for about 11B further frames (`std1` tail, all of
`std2`, 12 h of `sonic1`) producing only NaN; roughly 26 GPU-hours were lost
before evaluation would have caught it.

Bisected: `model_step_6500253696` is finite (0/19 non-finite policy tensors),
`model_step_7000031232` is not (17/19).

NOT the cause: the `nefc overflow - please increase njmax to 324` messages.
The first NaN is at log line 576007 and the first overflow at 576008, with all
four overflows after it — the policy went non-finite first, the robot then
entered a contact-rich pose, and that overflowed the constraint buffer. The
healthiest chain in the program (`ln_hold1_sonicreset` segment 9) logged 75
overflows with zero NaN. `njmax` stays at 320.

Rate: two events across about 96B frames of long-run training — this one and
`fsq64` at ~24.2B on 2026-08-23 — on different nodes, with different encoders,
holds, and reward sets. Roughly one per 50B frames. Treat it as a stochastic
hazard of long chains, not a property of a recipe. Root cause unresolved;
consistent with either a rare numerical edge case triggered by an unusual
state or silent hardware corruption.

Mitigation shipped the same day: `IPMD._abort_on_nonfinite` raises on the
first non-finite `train/step_reward_mean`, `episode/return`, or policy weight,
before any checkpoint is written. Disable with
`agent.abort_on_nonfinite=false`. Tests in `RLOpt/tests/test_ipmd_components.py`.

Restart decision (user, 2026-08-25): abandon the 6.5B checkpoint and retrain
from zero rather than resume. Resuming into the existing W&B run logged
nothing — that run's step counter had already advanced to 15.22B during the
NaN training, and `wandb.log(step=...)` refuses to move backwards, so every
metric was discarded. The old tree is preserved at
`<output_root>/tracker_abandoned_nan_20260825`, and
`<output_root>/wandb_run_id` was deleted so the control plane minted a fresh
id.

## Status

- 2026-08-25 20:38: RESTARTED from zero, jobs 5591478-83, W&B run
  `ps20-emaeew-s0-b46680`. Encoder binding unchanged. Nothing measured.
- 2026-08-25: first attempt (5590001-06) abandoned at 6.9B, see incident.
- 2026-08-23: campaign created, submitted as 5590001-06.
