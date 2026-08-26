# 2026-08-22 — reward estimation (IPMD direct IRL) — PARKED 2026-08-25

**Status: parked, future work needed.** Eight arms trained; the tracker is
unaffected by the reward-estimation stack (a clean null at 10B), and the
estimator itself saturates in every configuration tried. See
"PARKED — future work needed" below before reviving this.

Trains the IPMD reward estimator (direct reward estimation, not a
discriminator) alongside the tuned explicit root_qpos tracker, with the fixed
and normalized reward input introduced on 2026-08-22.

## What changed (the campaign's one variable)

The reward-estimation stack is opted in on the v2 surface:

- `env.enable_reward_input_observations=true` — enables the `reward_input`
  observation group. On v2 this is now `RewardInputUnitCfg`: every feature is
  normalized into [0, 1].
  - `expert_motion`: 29 joint positions, normalized per joint by the soft
    joint position limits (was 58-wide raw joint_pos+joint_vel).
  - `expert_anchor_pos_b`: relative root (anchor) position of the reference in
    the robot anchor frame, mapped from [-1 m, +1 m]; perfect tracking = 0.5.
  - `expert_anchor_ori_b`: relative root orientation as flattened rot6d,
    mapped from [-1, 1]; identity = (1, .5, .5, .5, 1, .5).
  - Policy side computed from the live robot; the expert side is served by
    the data plane through the same helpers
    (`isaaclab_imitation.envs.reward_input_normalization`) and the same
    pinned joint order.
- `agent.reward_estimation=true` — vanilla estimator coefficients
  (`reward_loss_coeff=1.0`, all regularizers 0.0), `reward_input_type="s"`,
  tanh output. PPO still trains on the task reward
  (`use_estimated_rewards_for_ppo=false`), so the tracker itself is the
  known explicit recipe.

The frozen v0/v1 surfaces keep the old raw `RewardInputCfg`
(`ImitationRLEnvLegacy` pairs it with its own expert-side cache).

## Protocol

- Task `Isaac-Imitation-G1-v2`, explicit single-frame 38-D root_qpos command
  (`[joint_qpos,root_pos,root_ori]`), tuned recipe, 16,384 envs x 24 steps,
  gamma 0.97, BONES-SEED full 129,785-clip reference set, 10B frame budget,
  walltime-segmented (full budget every segment; `cumulative_env_frames`
  carries the chain).
- W&B: project `g1-reward-estimation` (own project by user directive),
  group `irl-explicit-10b`, run id `irl-expl-s<seed>`.

## Local qualification (2026-08-22, workstation)

- 2-iteration smoke (16 envs, posterior latent arm) and 30-iteration run
  (1,024 envs, explicit arm): estimator updates end-to-end,
  `reward_diff` -0.92 → -2.0, `exp_r` → 1.0, no NaN, ~11.2k fps.
- The pure diff loss with zero regularizers saturates the tanh output early
  (expert at +1, policy at -1). This is the declared vanilla contract; if a
  non-saturated estimate is wanted, add `agent.ipmd.reward_logit_reg_coeff`
  or a grad penalty as a follow-up arm.
- Contract tests: `test_g1_task_layout_contract.py`,
  `test_g1_backend_joint_contract.py` — 16 passed.

## Submit

```bash
./experiments/campaigns/2026-08-22-reward-estimation/submit.sh 0   # plan seed 0
# then: pixi run python -m imitation_experiments.pipeline.cluster submit \
#     --plan <plan_dir> --confirm <PLAN_SHA>
```

## Results

4,096-motion scoreboard (sonic pass, ranks 12288-16383, frame-0 starts, mode
actions, no_push, Newton/MJWarp — `eval_scoreboard4096.sh`):

| row | frames | net | SR | MPJPE (mm) | survival |
| --- | --- | --- | --- | --- | --- |
| `irl_explicit_root_qpos` (this campaign) | 4.0B (mid-run) | scaled 6-layer | 0.9346 | 20.44 | 345.6 |
| `root_qpos_explicit` baseline (08-05, no IRL) | 7.6B | tuned [1024,1024,512] | 0.9358 | 20.11 | 346.4 |

Preliminary, one seed, frames NOT matched (4.0B vs 7.6B) and the network
differs (scaled vs tuned cells). Within those confounds the rows are level —
the differences (0.0012 SR, 0.33 mm) are far below evaluation noise, so at
minimum the reward-estimation stack does not hurt tracking. A frame-matched
comparison needs either the IRL checkpoint near 7.5B (lands as the run
passes it) or is impossible against this exact baseline below 7.6B (its
earlier checkpoints were not kept).

## Arms

| arm | tracker | estimator | ICE chain |
| --- | --- | --- | --- |
| `irl_explicit_root_qpos` | tuned explicit root_qpos | vanilla (no regularizer) | 5588194-96 |
| `irl_gp_explicit_root_qpos` | same | + R1 grad penalty 0.5 | 5588408-11 |
| `irl_gp_latent_ln_hold1` | headline `cont_det_ln_hold1` recipe, its exact pretrained encoder reused | + R1 grad penalty 0.5 | 5588412-15 |
| `irl_hp_explicit_root_qpos` | tuned explicit root_qpos | grad penalty 5.0 + logit reg 0.01 | 5588546-49 |
| `irl_hp_latent_ln_hold1` | headline recipe, same encoder reuse | grad penalty 5.0 + logit reg 0.01 | 5588550-53 |
| `irl_pair_explicit_root_qpos` | tuned explicit root_qpos | paired r(s, g) input + harsh regularizers | 5588661-66 |
| `irl_pair_latent_ln_hold1` | headline recipe, same encoder reuse | paired r(s, g) input + harsh regularizers | 5588671-74 |

The pair arms (2026-08-23) train the goal-conditioned estimator: the
reward_input gains `expert_desired_joint_pos` (the commanded reference
joints, 67-wide total), so the estimator learns a tracking metric r(s, g);
the expert minibatch pairs its own joints with themselves. Gated by
`agent.reward_estimation_pair_input` (default off — pre-pairing chains
resume cleanly). Expert information stays estimator-only: no actor or
critic reads the reward_input group; the actor's expert view remains the
command interface.

The grad penalty rides the declarative
`agent.reward_estimation_grad_penalty_coeff` field (a
`agent.reward_estimation_logit_reg_coeff` sibling exists, unused so far); a
direct `agent.ipmd.reward_grad_penalty_coeff` override would be zeroed by
the reward-estimation switch. `irl_gp_latent_ln_hold1` differs from the
headline 10B row only in the reward-estimation stack, and from
`irl_gp_explicit_root_qpos` only in the command interface.

## Matched 1.0B comparison (2026-08-23, one seed, preliminary)

All six rows: 4,096 board, sonic pass, model_step_1000341504 per arm; the
latent comparator is the real headline `cont_det_ln_hold1` segment-1 1B
checkpoint with the same encoder file. No non-IRL explicit checkpoint
exists below 7.6B.

| interface | estimator | SR | MPJPE (mm) | survival |
| --- | --- | --- | --- | --- |
| explicit | vanilla IRL | 0.9172 | 22.43 | 341.4 |
| explicit | GP 0.5 | 0.9189 | 22.05 | 342.1 |
| explicit | GP 5.0 + LR 0.01 | 0.9167 | 21.85 | 340.9 |
| latent ln_hold1 | none (headline) | 0.8943 | 28.36 | 336.9 |
| latent ln_hold1 | GP 0.5 | 0.8882 | 28.32 | 333.3 |
| latent ln_hold1 | GP 5.0 + LR 0.01 | 0.8887 | 27.81 | 334.8 |

Every within-family difference is inside evaluation noise: the
reward-estimation stack does not affect tracking at matched budget, on
either interface, at any regularizer strength tried.

## Final 10B row (vanilla arm, 2026-08-24)

`irl_explicit_root_qpos` completed its 10B budget (segment 2). 4,096 board,
sonic pass: **SR 0.9558 / MPJPE 17.66 mm / survival 350.6** — the strongest
explicit tracker row on this board to date; beats the frame-matched headline
latent 10B row (0.9368 / 23.61) and the 7.6B small-net explicit baseline
(0.9358 / 20.11; that comparison is confounded by network size and frames).
One seed.

## Full 10B table (2026-08-25, one seed)

4,096 board, sonic pass. The latent comparator is the real headline
`cont_det_ln_hold1` 10B row (same encoder file, same recipe, no IRL stack).

| interface | estimator | frames | SR | MPJPE (mm) | survival |
| --- | --- | --- | --- | --- | --- |
| explicit | vanilla IRL | 10B | 0.9558 | 17.66 | 350.6 |
| explicit | GP 0.5 | 10B | 0.9583 | 17.92 | 351.1 |
| explicit | GP 5.0 + LR 0.01 | 10B | **0.9604** | **17.47** | 351.9 |
| explicit | paired r(s, g) + harsh | 10B | 0.9578 | 17.64 | 351.5 |
| latent ln_hold1 | none (headline) | 10B | 0.9368 | 23.61 | 347.3 |
| latent ln_hold1 | GP 0.5 | 10B | 0.9373 | 23.65 | 347.3 |
| latent ln_hold1 | paired r(s, g) + harsh | 9.0B | 0.9377 | 24.23 | 347.1 |
| latent ln_hold1 | GP 5.0 + LR 0.01 | 8.5B, still training | - | - | - |

Reading: the IRL stack remains a NULL on tracking at 10B — every explicit
row sits at 0.956-0.960 SR / 17.5-17.9 mm and every latent row at
0.937 SR / 23.6-24.2 mm, all inside evaluation noise of their
no-reward-estimation counterpart. Reward estimation neither helps nor hurts
the tracker, on either interface, at any regularizer strength tried. That
is the expected result while PPO trains on the task reward
(`use_estimated_rewards_for_ppo=false`); the estimator is along for the
ride.

### Divergence: `irl_pair_latent_ln_hold1` (2026-08-25)

The paired latent arm trained healthily to ~9.33B frames (ep_len ~200) and
then blew up inside a single 25-iteration window: ep_len 164 at iter 16073,
then `pi_loss`, `reward_diff`, `exp_r` and `r_step` all NaN at iter 16099,
never recovering. Its 10B checkpoint is therefore dead (SR 0.000, survival
2.3 steps); the row above is its last pre-divergence checkpoint (9.0B),
which is healthy and level with the other latent rows. The eval itself is
sound - encoder binding reports 0.0 divergence.

Not yet attributed. The same pairing on the explicit interface reached 10B
cleanly, and GP 0.5 on the latent interface reached 10B cleanly, so the
suspect is the interaction of the goal-conditioned input with the harsh
(GP 5.0) penalty on the latent route: the grad penalty builds a
second-order graph through the estimator input, and the paired input makes
policy/expert trivially separable, so the estimator sits pinned at its
logit-reg ceiling (reward_diff -1.9548, exactly constant for thousands of
iterations before the blowup). A rerun would settle whether it is
reproducible or a one-off numeric event.

## PARKED — future work needed (2026-08-25)

User decision: the reward estimator in its current form is **not useful**,
and the reason is **saturation**. The campaign stops here; the tracker rows
stand as a completed null result.

What the campaign established:

- The stack is correct and safe. Normalized [0, 1] inputs, matched
  policy/expert feature maps, goal-conditioned pairing, and four regularizer
  settings all train end to end at 10B without disturbing the tracker
  (every row inside noise of its no-reward-estimation counterpart).
- The estimator itself is degenerate in every configuration tried. The pure
  diff objective (`E_pi[r] - E_exp[r]`) has no interior optimum: it is
  minimized by driving policy and expert outputs as far apart as the output
  activation allows. Without regularization it pins at the tanh rails
  (-1.0 / +1.0 exactly). With the R1 grad penalty it still rails, because
  input gradients vanish at the rails, which is exactly where the penalty
  needs force. With logit regularization it holds a soft ceiling instead
  (reward_diff -1.958, exp_r 0.977) — a *lower* rail, not a shaped reward.
  Goal-conditioned pairing makes it worse, not better: the expert side has
  identically zero tracking error, so separation becomes trivial.
- Consequence: `use_estimated_rewards_for_ppo=true` was never worth trying.
  A reward that is constant on each side carries no gradient for the policy.

What future work would have to change (none of this is implemented):

1. **A bounded objective with an interior optimum.** The diff loss needs
   either a Wasserstein-style constraint that actually binds (a gradient
   penalty evaluated on interpolates between policy and expert samples, as
   in WGAN-GP, rather than the current per-side R1), or replacement with a
   proper density-ratio / logistic objective whose optimum is a finite
   log-ratio rather than a saturating separation.
2. **Harder negatives.** While the tracker is imperfect, policy and expert
   state distributions barely overlap, so any estimator separates them
   trivially. Candidates: sample negatives from near-expert perturbations,
   or condition the estimator on the tracking error magnitude so it must
   rank *degrees* of failure instead of classifying two obvious classes.
3. **A calibration target.** Nothing currently asks the estimated reward to
   agree with anything measurable. Regressing it against the hand-designed
   tracking reward (or against MPJPE) on held-out states would turn
   "expert vs not" into a usable learned metric, and gives a diagnostic
   that reports saturation immediately instead of after 10B frames.
4. **A stability fix for the paired latent route** before reusing that
   combination — see the divergence note above.

The machinery stays in the tree and is off by default
(`agent.reward_estimation=false`, `env.enable_reward_input_observations`
false on `-G1-v2`), so nothing in the paper path is affected. Re-enabling is
a two-flag change plus whichever objective replaces the diff loss.

## Status

- 2026-08-22: campaign created; local qualification passed.
- 2026-08-23: submitted to ICE as chain 5588194 -> 5588195 -> 5588196;
  training healthy (~128k fps). Mid-run 4.0B checkpoint scored on the 4,096
  board (table above). Estimator output saturated at the tanh rails
  (reward_diff -2.0, exp_r 1.0) throughout — vanilla zero-regularizer
  contract; follow-up arm with logit-reg/grad-penalty needed for a
  non-degenerate estimate.
- 2026-08-23 (later): grad-penalty knob added; `irl_gp_explicit_root_qpos`
  and `irl_gp_latent_ln_hold1` smoked locally (5 iters, reward_diff
  -0.31/-0.32 vs -0.92 vanilla at the same point — the penalty visibly slows
  rail saturation) and submitted as chains 5588408-11 / 5588412-15.
- 2026-08-24: ICE user quota hit 300/300 GB and killed every checkpoint
  save: gp chains died at the 6.0B save (last good 5.5B), hp at 4.5B (last
  good 4.0B), pair-explicit at 3.0B (last good 2.5B); follow-on segments
  crashed instantly on `Disk quota exceeded`. Corrupt checkpoint files
  deleted; `interface_design_study` outputs purged from ICE per user
  approval (75G freed); all five dead chains resumed as
  lowlevel3+lowlevel_resume pairs (jobs 5588914/16/18/20/22). The vanilla
  arm finished 10B before the quota filled.
- 2026-08-25 (later): feature PARKED by user decision — saturation makes the
  current estimator useless. The queued `irl_hp_latent_ln_hold1` segments
  (5591490 -> 5591491) are left to run out; no further submissions.
- 2026-08-25: six of seven arms reached the full 10B budget and were scored
  (table above). `irl_hp_latent_ln_hold1` ran out of chained segments at
  8.5B and was continued as jobs 5591490 -> 5591491.
  `irl_pair_latent_ln_hold1` diverged at ~9.33B (see above); scored at 9.0B.
- 2026-08-23 (evening): both gp arms MEASURED railed by ~1B frames
  (reward_diff -2.0, exp_r 1.0) — R1 has no force at the rails. Harsh
  arms submitted per user directive (kept the gp arms running): grad
  penalty 5.0 + logit reg 0.01, chains 5588546-49 (explicit) and
  5588550-53 (latent headline). Local 8-iter smoke: reward_diff -0.048,
  exp_r 0.28, no NaN.
