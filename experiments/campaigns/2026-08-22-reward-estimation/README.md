# 2026-08-22 — reward estimation (IPMD direct IRL)

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

## Status

- 2026-08-22: campaign created; local qualification passed.
- 2026-08-23: submitted to ICE as chain 5588194 -> 5588195 -> 5588196;
  training healthy (~128k fps). Mid-run 4.0B checkpoint scored on the 4,096
  board (table above). Estimator output saturated at the tanh rails
  (reward_diff -2.0, exp_r 1.0) throughout — vanilla zero-regularizer
  contract; follow-up arm with logit-reg/grad-penalty needed for a
  non-degenerate estimate.
