#!/usr/bin/env bash
# Arm table for the 2026-08-02 RLOpt hyperparameter screen.
#
# Sourced by both launchers -- `run_hp_screen_local.sh` (workstation, sequential,
# CSV metrics) and `submit_rlopt_hp_search_ice.sh` (ICE H100s, one Slurm job per
# arm, W&B metrics). It lives in its own file precisely so the two cannot drift:
# a screen whose arms differ between where it was designed and where it was run
# is not a screen.
#
# This file only defines variables. It must stay side-effect free: no `set -e`,
# no output, no work.
#
# Each entry is: name | description | space-separated overrides.
#
# Overrides are Hydra key=value pairs, with one exception: the pseudo-override
# `ROLLOUT_STEPS=N` is consumed by the launcher rather than passed to Hydra, and
# rebatches that arm. Both launchers hold the frame budget fixed by scaling the
# iteration count against it, so a rebatched arm still sees the same 50M frames.
#
# Sizing note that the arms are built around: optimizer steps per frame is
# `epochs / mini_batch_size`. It does not involve `frames_per_batch` at all.
# That identity is why the cluster r12-vs-r24 comparison confounded two axes --
# r24 doubled the rollout *and* doubled the minibatch, halving its steps per
# frame from 271 to 136 per million.

# 12288 x 12 with mini_batch_size 18432 is the cluster recipe exactly: 8
# minibatches x 5 epochs = 40 optimizer steps per 147,456 frames = 271 per
# million. Every arm holds that unless it is the knob under test.
HP_SCREEN_BASE_MINIBATCH=18432
HP_SCREEN_DOUBLE_UPDATES_MINIBATCH=9216

# Widened critic for the round-3 capacity arm: (768,512,256) -> this, roughly
# 0.95M -> 2.1M parameters. Assigned here rather than inside the array literal,
# where bash would read it as one more arm spec and never define it.
HP_SCREEN_BIG_CRITIC="[1024,1024,512]"

# Round-4 exploration constants, from the SONIC release contract.
#
# THE CAP AND THE INIT ARE NOT SEPARABLE. `clip_log_std` clamps log_std on every
# forward (rlopt/models/gaussian_policy.py), and torch.clamp passes zero gradient
# outside its range -- so a run that caps at log(0.5) while initializing log_std
# at 0.0 (sigma 1.0, our current default) puts the parameter permanently outside
# the clamp: it receives no gradient, never moves, and sigma is pinned at exactly
# 0.5 for the whole run while the logs look unremarkable. SONIC avoids this by
# initializing at log(0.05), well inside [log(0.001), log(0.5)].
#
# Every clipped arm below therefore also carries the low init. There is no
# "cap with the current init" cell because that cell is broken by construction,
# not because it was not worth testing.
HP_SCREEN_LOG_STD_INIT_005="-2.995732273553991"   # log(0.05)
HP_SCREEN_LOG_STD_MAX_05="-0.6931471805599453"    # log(0.5)
HP_SCREEN_LOG_STD_MIN_0001="-6.907755278982137"   # log(0.001)

# Round 5. Round 4 established that log(0.05) is not merely a small init, it is a
# trap at this learning rate: Adam moves log_std by about the LR per step, the
# adaptive LR geomean is 2.8e-5, and climbing from sigma 0.05 to the 0.36 a12
# actually operates at needs ln(0.36/0.05)=1.97 in log space -- roughly 70,000
# updates, about 260M frames, before learning could even begin. Every round-4 arm
# sat at episode length 5-8 for its whole block, and c5 (which changes only this
# knob from a12) reproduced the 2026-07-20 release-contract failure exactly.
#
# So initialize just INSIDE the cap rather than far below it: sigma 0.45 against
# a 0.5 ceiling. Gradient still flows (the frozen-parameter hazard needs init
# strictly outside the clamp), and the policy starts where a12 already works.
HP_SCREEN_LOG_STD_INIT_045="-0.7985076962177716"  # log(0.45), inside [log(0.001), log(0.5)]

HP_SCREEN_ARM_SPECS=(
"b0_baseline|Cluster recipe verbatim: adaptive LR per minibatch, entropy 0.005, 5 epochs, 8 minibatches (40 updates/iter, 271 per M frames).|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH}"
"a1_updates_2x|Optimizer steps per frame doubled: 16 minibatches, 80 updates/iter (542 per M frames). Tests the mechanism the r12-vs-r24 gap implicates.|agent.loss.mini_batch_size=${HP_SCREEN_DOUBLE_UPDATES_MINIBATCH}"
"a2_kl_per_iteration|Adaptive-KL rule fires once per iteration on the mean KL instead of once per minibatch.|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.optim.kl_adapt_step=iteration"
"a3_lr_fixed_1e4|Actor LR pinned at 1e-4 (min_lr=max_lr) -- roughly 3x the reference run's adaptive geomean.|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.ipmd.actor_learning_rate=1.0e-4 agent.optim.min_lr=1.0e-4 agent.optim.max_lr=1.0e-4"
"a4_lr_fixed_3e5|Actor LR pinned at 3e-5, the reference run's adaptive geomean. Isolates whether the LR *noise* hurts at an unchanged average LR.|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.ipmd.actor_learning_rate=3.0e-5 agent.optim.min_lr=3.0e-5 agent.optim.max_lr=3.0e-5"
"a5_entropy_1e3|Entropy coefficient 0.005 -> 0.001.|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.ppo.entropy_coeff=0.001"
"a6_entropy_0|Entropy bonus off. Bounds how much of the action-rate penalty and the MPJPE floor is exploration noise.|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.ppo.entropy_coeff=0.0"
"a7_epochs_3|5 epochs -> 3: sample reuse 5 -> 3, and 163 updates per M frames.|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.loss.epochs=3"
"a8_r24_matched|24-step rollout at an update budget matched to b0 (16 minibatches, 271 per M frames). The cell the cluster r12-vs-r24 comparison never ran: freshness isolated from update density.|ROLLOUT_STEPS=24 agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH}"
"a9_epochs_10|5 epochs -> 10: 542 updates per M frames, the same count as a1 but reached by reusing each sample twice as often rather than by halving the minibatch.|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.loss.epochs=10"
# --- Round 2, added after the first round returned -----------------------------
# a2 (KL per iteration) and a6 (entropy off) each beat b0 at a matched 50M
# frames, on different metrics -- a2 on episode length (144.0 vs 126.3), a6 on
# return (10.52 vs 8.85) -- and a2 costs nothing (train 0.325s vs b0's 0.329s).
# a11 combines them; a12 adds observation normalization on top; a10 isolates
# normalization so a12's result stays attributable.
#
# Observation normalization is not new code: RLOpt already builds
# RunningMeanStdCatInputs behind `normalize_input`, and the G1 config already
# lists the latent command in `normalize_input_exclude_keys`, so z and its
# sin/cos phase stay raw. Normalizing z with running statistics would make the
# planner -> tracker interface non-stationary, which is a correctness problem
# for the equivalence certificate rather than a tuning question.
"a10_obs_norm|Running observation normalization on actor and critic, latent command excluded (already declared in normalize_input_exclude_keys). Isolates normalization from the a11 winners.|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.policy.normalize_input=true agent.value_function.normalize_input=true"
"a11_kl_iter_entropy0|The two confirmed round-1 winners together: adaptive-KL per iteration plus no entropy bonus.|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.optim.kl_adapt_step=iteration agent.ppo.entropy_coeff=0.0"
"a12_kl_iter_entropy0_obsnorm|a11 plus observation normalization: the full stack. Attributable only because a10 runs normalization alone.|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.optim.kl_adapt_step=iteration agent.ppo.entropy_coeff=0.0 agent.policy.normalize_input=true agent.value_function.normalize_input=true"
# a13 is a2 with the phase vector removed, so it is read against a2 rather than
# against b0. Published width drops 258 -> 256: z alone, no sin/cos.
#
# The phase is appended by the command publisher, not produced by the encoder,
# so the same frozen det-SR checkpoint applies unchanged and the arm stays
# comparable. What it removes is the tracker's explicit signal for where it sits
# inside the 10-step latent hold (code_period=10) -- with the phase gone the
# tracker has to infer that from proprioception. That makes this a question
# about the interface rather than the optimizer: it is the one arm here that
# changes what the low level is told, and it lowers the planner's published
# bandwidth, which is a paper-facing quantity.
#
# PHASE_MODE is a pseudo-override; the launcher derives the published width from
# it so the declared and actual sizes cannot disagree.
"a13_kl_iter_nophase|a2 (adaptive-KL per iteration) with the sin/cos phase dropped from the published latent command: 258 -> 256, z only. Read against a2, not b0.|PHASE_MODE=none agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.optim.kl_adapt_step=iteration"
# a14 gives the critic the reference end-effector positions. Read against a2.
#
# 90% of episodes at 50M end on foot_pos_xyz (76%) or ee_body_pos (14%), and
# both predicates compare an achieved EE position against the *reference* EE
# position, each expressed relative to its own anchor. The critic already sees
# the achieved side -- `body_pos` spans G1_TRACKED_BODY_NAMES, which includes
# both ankle_roll links -- but on the reference side it gets only
# `expert_motion` (joint qpos/qvel). Recovering a reference foot position from
# 29 joint angles means learning forward kinematics, and that quantity decides
# when the episode, and therefore the return, stops.
#
# `ee_pos` adds `expert_ee_pos_b`: 4 bodies x 3 = 12 dims on a 544-dim critic,
# taking it to 556. Critic-only, so the actor contract, the published bandwidth
# and the equivalence certificate are all untouched -- exactly the privileged
# critic state AGENTS.md allows. Note explained variance is already 0.977, so
# this is a hypothesis about *where* the residual sits (at the termination
# boundary, where advantage error matters most), not a safe bet.
"a14_kl_iter_critic_ee|a2 plus the reference EE positions in the critic (critic 544 -> 556): lets the critic represent the predicate behind 90% of terminations instead of learning FK for it. Read against a2, not b0.|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.optim.kl_adapt_step=iteration env.command_interface.reference.critic_components=[joint_qpos_qvel,root_pos,root_ori,ee_pos]"
# --- Round 3 ------------------------------------------------------------------
# Round 2 made observation normalization the single largest effect in the screen
# (a10 alone: ep_len 189.8 vs b0's 126.3, +50%), and a12 the best on return
# (14.76). But round 2 also showed stacking is NOT monotone: a11
# (kl_iter + entropy0) scored 128.7, *below* either a2 (144.0) or a6 (134.9)
# alone. So a stack cannot be assumed to inherit its parts' gains.
#
# a16 is the cell round 2 never ran -- entropy0 + obs norm without kl_iter --
# and it exists so a15 is attributable. Without it, a15 differs from a12 by two
# factors at once (kl_iter removed, critic widened), which is the same
# confound that made the r12-vs-r24 cluster comparison unreadable.
"a16_entropy0_obsnorm|Entropy off plus observation normalization, without kl_iter: the missing round-2 cell, and the baseline a15 is read against.|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.ppo.entropy_coeff=0.0 agent.policy.normalize_input=true agent.value_function.normalize_input=true"
"a15_entropy0_obsnorm_bigcritic|a16 plus a wider critic (768,512,256 -> 1024,1024,512; ~0.95M -> ~2.1M params). Tests whether the critic is capacity-limited once its inputs are normalized.|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.ppo.entropy_coeff=0.0 agent.policy.normalize_input=true agent.value_function.normalize_input=true agent.value_function.num_cells=${HP_SCREEN_BIG_CRITIC}"
# a17 is the wider critic on the arm that is actually winning. a12 (wandb
# 8n48jbvw) leads the screen on both return (14.76) and return per minute
# (0.522 vs b0's 0.286), and it carries kl_iter -- so a12 + wider critic is the
# one-factor step from the champion, whereas a15 reaches the same width from
# the no-kl_iter side. Running both, with a16 as their shared control, is what
# makes "does the critic want capacity" separable from "does kl_iter help here".
"a17_a12_bigcritic|a12 (the current best: kl_iter + entropy0 + obs norm) plus the wider critic. The one-factor step from the leading arm.|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.optim.kl_adapt_step=iteration agent.ppo.entropy_coeff=0.0 agent.policy.normalize_input=true agent.value_function.normalize_input=true agent.value_function.num_cells=${HP_SCREEN_BIG_CRITIC}"
# --- Round 4: exploration control ----------------------------------------------
# a12 controls sigma by deleting the entropy bonus, which works (sigma reached
# 0.406 at 50M without collapsing) but has no floor and no guarantee over a 5B
# run. SONIC solves the same problem from the other side: keep a *larger* bonus
# (0.01) and bound sigma from above instead, sigma in [0.001, 0.5] starting at
# 0.05. That is the safer shape, so it is worth knowing whether it also scores.
#
# Every arm shares a12's base -- kl_iter + obs norm -- so each is one factor from
# the reigning champion. The critic stays at a12's width deliberately: folding in
# the unvalidated wider critic would confound this comparison, and a15/a16/a17
# settle that axis separately.
#
# The chain is single-factor end to end:
#   a12 -> c5 (init 1.0 -> 0.05) -> c3 (add the cap) -> c2 (add bonus 0.005)
#       -> c1 (bonus 0.005 -> 0.01, = SONIC's exploration contract exactly)
# with c4 = c2 without the cap, isolating the cap from the low init.
"c5_lowinit_ent0|a12 with sigma initialized at 0.05 instead of 1.0, bonus still off and no cap. Isolates the initial noise scale, which a12 never varied.|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.optim.kl_adapt_step=iteration agent.policy.normalize_input=true agent.value_function.normalize_input=true agent.ppo.entropy_coeff=0.0 agent.ppo.log_std_init=${HP_SCREEN_LOG_STD_INIT_005}"
"c3_clip_ent0|c5 plus the sigma cap at 0.5 (floor 0.001). Bonus still off: tests the cap alone.|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.optim.kl_adapt_step=iteration agent.policy.normalize_input=true agent.value_function.normalize_input=true agent.ppo.entropy_coeff=0.0 agent.ppo.log_std_init=${HP_SCREEN_LOG_STD_INIT_005} agent.ppo.clip_log_std=true agent.ppo.log_std_max=${HP_SCREEN_LOG_STD_MAX_05} agent.ppo.log_std_min=${HP_SCREEN_LOG_STD_MIN_0001}"
"c2_clip_ent005|c3 with the entropy bonus back at 0.005. The bounded-exploration answer to a12's deleted bonus.|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.optim.kl_adapt_step=iteration agent.policy.normalize_input=true agent.value_function.normalize_input=true agent.ppo.entropy_coeff=0.005 agent.ppo.log_std_init=${HP_SCREEN_LOG_STD_INIT_005} agent.ppo.clip_log_std=true agent.ppo.log_std_max=${HP_SCREEN_LOG_STD_MAX_05} agent.ppo.log_std_min=${HP_SCREEN_LOG_STD_MIN_0001}"
"c1_sonic_explore|SONIC's exploration contract exactly: bonus 0.01, sigma in [0.001, 0.5] from 0.05, on a12's base.|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.optim.kl_adapt_step=iteration agent.policy.normalize_input=true agent.value_function.normalize_input=true agent.ppo.entropy_coeff=0.01 agent.ppo.log_std_init=${HP_SCREEN_LOG_STD_INIT_005} agent.ppo.clip_log_std=true agent.ppo.log_std_max=${HP_SCREEN_LOG_STD_MAX_05} agent.ppo.log_std_min=${HP_SCREEN_LOG_STD_MIN_0001}"
"c4_lowinit_ent005|c2 without the cap: bonus 0.005 and sigma from 0.05, unbounded. Separates the cap from the low init.|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.optim.kl_adapt_step=iteration agent.policy.normalize_input=true agent.value_function.normalize_input=true agent.ppo.entropy_coeff=0.005 agent.ppo.log_std_init=${HP_SCREEN_LOG_STD_INIT_005}"
# --- Round 5: bounded exploration, initialized where the policy already works ---
# a12 controls sigma by deleting the entropy bonus. That works at 50M but leaves
# no floor: nothing stops sigma collapsing later in a 5B run, and log_std_min is
# effectively unbounded at -7. These two arms bound sigma into [0.001, 0.5]
# instead, which is the durable shape -- the FLOOR is the part that matters, and
# the 0.5 ceiling is nearly inert since a12's sigma settles around 0.36.
#
# d1 asks what bounding costs against a12 with the bonus still off; d2 restores
# the bonus, which is only safe to do once sigma has a ceiling.
"d1_bounded_ent0|a12 with sigma bounded to [0.001, 0.5] from an init of 0.45, bonus still off. Costs of bounding, measured against a12.|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.optim.kl_adapt_step=iteration agent.policy.normalize_input=true agent.value_function.normalize_input=true agent.ppo.entropy_coeff=0.0 agent.ppo.log_std_init=${HP_SCREEN_LOG_STD_INIT_045} agent.ppo.clip_log_std=true agent.ppo.log_std_max=${HP_SCREEN_LOG_STD_MAX_05} agent.ppo.log_std_min=${HP_SCREEN_LOG_STD_MIN_0001}"
"d2_bounded_ent005|d1 with the entropy bonus restored at 0.005: keep the bonus, bound sigma. The collapse-proof candidate for a 5B run.|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.optim.kl_adapt_step=iteration agent.policy.normalize_input=true agent.value_function.normalize_input=true agent.ppo.entropy_coeff=0.005 agent.ppo.log_std_init=${HP_SCREEN_LOG_STD_INIT_045} agent.ppo.clip_log_std=true agent.ppo.log_std_max=${HP_SCREEN_LOG_STD_MAX_05} agent.ppo.log_std_min=${HP_SCREEN_LOG_STD_MIN_0001}"
# --- Round 6: the remaining SONIC release components, decomposed ----------------
# The release contract was tested once as a bundle of eleven simultaneous changes
# (2026-07-20), went flat, and was reverted the next day. Round 4 pinned that
# failure on one of its knobs -- log_std_init=log(0.05) alone reproduces it -- so
# the remaining components have never been judged on their own merits. Each of
# these is one release knob on a12's base.
#
# b5 is the one with prior evidence: its start thresholds were measured as the
# fastest local learner (episode length 25.9 vs 14.6 over 50M, migration wiki),
# and it is the only arm here that touches the environment rather than the
# optimizer -- 76% of episodes at 50M end on foot_pos_xyz against 1.9% reaching
# time_out, so the policy never sees three quarters of a motion window.
"b5_term_curriculum|Re-enable the SONIC termination-threshold anneal (loose -> strict over 50M-500M). The only arm targeting the termination-limited episodes.|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.optim.kl_adapt_step=iteration agent.ppo.entropy_coeff=0.0 agent.policy.normalize_input=true agent.value_function.normalize_input=true env.enable_termination_curriculum=true"
"b2_adv_global|Advantage normalized once per rollout instead of per minibatch.|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.optim.kl_adapt_step=iteration agent.ppo.entropy_coeff=0.0 agent.policy.normalize_input=true agent.value_function.normalize_input=true agent.ppo.normalize_advantage_global=true"
"b4_silu|SiLU activations on both networks, sizes unchanged. Isolates the release contract's activation from its 15x-larger layers, which 50M frames could not fill anyway.|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.optim.kl_adapt_step=iteration agent.ppo.entropy_coeff=0.0 agent.policy.normalize_input=true agent.value_function.normalize_input=true agent.policy.activation_fn=silu agent.value_function.activation_fn=silu"
# Expected near-neutral, run to close the question rather than on a hypothesis:
# Adam's update is scale-invariant in the gradient (m/sqrt(v)), so uniformly
# clipping every gradient mostly cancels and only outlier spikes are removed.
# Measured grad_norm on the reference run was ~1.9, so 0.1 clips essentially
# every update.
"b3_gradclip|Gradient-norm clip 1.0 -> 0.1, the release value.|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.optim.kl_adapt_step=iteration agent.ppo.entropy_coeff=0.0 agent.policy.normalize_input=true agent.value_function.normalize_input=true agent.optim.max_grad_norm=0.1"
# Also expected near-neutral: the adaptive rule's geomean is already 2.9e-5,
# essentially the release's 2e-5 starting point, and the observed LR never
# exceeded 5.8e-5, so the 2e-4 ceiling would not have bound either.
"b6_actor_lr_split|Actor lr 2e-5 with the adaptive band narrowed to [1e-5, 2e-4], critic held at 1e-3.|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.optim.kl_adapt_step=iteration agent.ppo.entropy_coeff=0.0 agent.policy.normalize_input=true agent.value_function.normalize_input=true agent.ipmd.actor_learning_rate=2.0e-5 agent.ipmd.critic_learning_rate=1.0e-3 agent.optim.min_lr=1.0e-5 agent.optim.max_lr=2.0e-4"
# --- Round 7: discount, and capacity ------------------------------------------
# The architecture arms are worth running NOW in a way they were not before.
# Every previous test of the release contract's 2048x6 networks was trapped by
# log_std_init=log(0.05) (round 4 attributed the whole release failure to that
# one knob), so the big architecture has never actually been judged on a base
# that learns. e3 is its first fair test.
#
# What these arms can and cannot answer: the motivation is a motion library
# scaling toward 5000 clips, and a 40-motion 50M screen cannot measure capacity
# demand at 5000. It can measure whether extra capacity COSTS anything at the
# current scale -- a risk check before committing the bigger net to a long run,
# not evidence that it will pay later.
"e1_gamma097|Discount 0.99 -> 0.97: horizon 100 -> 33 control steps (2.0s -> 0.67s at 50 Hz). Shorter credit assignment for a dense tracking reward.|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.optim.kl_adapt_step=iteration agent.ppo.entropy_coeff=0.0 agent.policy.normalize_input=true agent.value_function.normalize_input=true agent.loss.gamma=0.97"
"e2_widenet|Both networks widened to [1024,1024,512] at unchanged depth. Width alone, separable from e3's depth and activation.|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.optim.kl_adapt_step=iteration agent.ppo.entropy_coeff=0.0 agent.policy.normalize_input=true agent.value_function.normalize_input=true agent.policy.num_cells=[1024,1024,512] agent.value_function.num_cells=[1024,1024,512]"
"e3_sonicnet|The release architecture, first fair test: [2048,2048,1024,1024,512,512] SiLU on both networks, on a base that learns.|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.optim.kl_adapt_step=iteration agent.ppo.entropy_coeff=0.0 agent.policy.normalize_input=true agent.value_function.normalize_input=true agent.policy.num_cells=[2048,2048,1024,1024,512,512] agent.value_function.num_cells=[2048,2048,1024,1024,512,512] agent.policy.activation_fn=silu agent.value_function.activation_fn=silu"
# --- Round 8: privileged teacher / distilled student ---------------------------
# IPMD_L2T trains a teacher on the exact ordered CRITIC inputs (privileged: the
# reference channel plus body_pos/body_ori/base velocities) and distills it
# online into a student restricted to the deployable actor observations. The G1
# config mirrors the student architecture from the teacher, so this arm changes
# the algorithm and nothing else about the recipe.
#
# ALGO is a pseudo-override: the entry point derives the agent config from the
# algorithm name (rlopt_ipmd_l2t_cfg_entry_point), so it must not go to Hydra.
#
# Read against a12. The deployable policy here is the STUDENT, so a teacher that
# scores well on privileged inputs is not the result -- the student is.
"f1_ipmd_l2t|IPMD_L2T: privileged teacher on the critic contract, online-distilled into a student on actor observations. a12 base, algorithm swapped.|ALGO=IPMD_L2T agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.optim.kl_adapt_step=iteration agent.ppo.entropy_coeff=0.0 agent.policy.normalize_input=true agent.value_function.normalize_input=true"
# --- Round 8: 100M budget, stacking the round-6/7 winners -----------------------
# Budget moves 50M -> 100M so longer-horizon behaviour is observable; 100M arms
# are NOT comparable to the 50M ones and the aggregator's geometry gate enforces
# that, so g0 is the new reference at this budget.
#
# The termination curriculum (b5) is promoted into the base for every arm: it was
# the largest single effect in the campaign, 0.892 ret/min against a12's 0.495
# (1.80x) and 11.51 len/min against 5.96 (1.93x), while finishing 25% faster
# because longer episodes mean fewer resets.
#
# Second tier from round 6/7, all of which beat a12 on both rates: silu (0.528 /
# 6.49), gradclip 0.1 (0.516 / 6.27), advantage-global (0.517 / 6.17). And on
# len/min specifically the larger nets already led -- e2 6.13, e3 6.08 against
# a12's 5.96, with e3 posting the highest episode length of any 50M arm (188.3)
# on a clearly lower return. That is the signature of an undertrained large net,
# which is exactly what a doubled budget should resolve.
"g0_base100m|a12 + termination curriculum, at 100M. The reference every other 100M arm is read against.|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.optim.kl_adapt_step=iteration agent.ppo.entropy_coeff=0.0 agent.policy.normalize_input=true agent.value_function.normalize_input=true env.enable_termination_curriculum=true"
"g1_silu|g0 + SiLU on both networks (b4 was the best non-curriculum single knob).|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.optim.kl_adapt_step=iteration agent.ppo.entropy_coeff=0.0 agent.policy.normalize_input=true agent.value_function.normalize_input=true env.enable_termination_curriculum=true agent.policy.activation_fn=silu agent.value_function.activation_fn=silu"
"g2_widenet|g0 + [1024,1024,512] on both networks: e2 led len/min at 50M.|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.optim.kl_adapt_step=iteration agent.ppo.entropy_coeff=0.0 agent.policy.normalize_input=true agent.value_function.normalize_input=true env.enable_termination_curriculum=true agent.policy.num_cells=[1024,1024,512] agent.value_function.num_cells=[1024,1024,512]"
"g3_sonicnet|g0 + the release architecture (2048x6 SiLU): e3 had the highest episode length of any 50M arm, with return still climbing.|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.optim.kl_adapt_step=iteration agent.ppo.entropy_coeff=0.0 agent.policy.normalize_input=true agent.value_function.normalize_input=true env.enable_termination_curriculum=true agent.policy.num_cells=[2048,2048,1024,1024,512,512] agent.value_function.num_cells=[2048,2048,1024,1024,512,512] agent.policy.activation_fn=silu agent.value_function.activation_fn=silu"
"g4_advglobal_gradclip|g0 + the two mild round-6 wins together: global advantage normalization and grad clip 0.1.|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.optim.kl_adapt_step=iteration agent.ppo.entropy_coeff=0.0 agent.policy.normalize_input=true agent.value_function.normalize_input=true env.enable_termination_curriculum=true agent.ppo.normalize_advantage_global=true agent.optim.max_grad_norm=0.1"
# --- Round 8b: architecture on an HONEST base (no curriculum) ------------------
# MPJPE exposed a confound in the round-6 result. The termination curriculum wins
# ret/min 1.80x and len/min 1.93x, but its MPJPE is 99.18 mm against a12's 71.59
# -- 39% WORSE. Looser thresholds let the robot drift further from the reference
# before terminating, so episodes lengthen and return accumulates while tracking
# quality falls. Both goal metrics reward that mechanically, so neither can be
# read without MPJPE beside it.
#
# b4_silu is the opposite and is the genuine win: episode length 209.3 (vs 173.3)
# AND the best MPJPE of any arm, 66.76, at unchanged per-step reward. So SiLU
# joins the base and the curriculum does not.
#
# These read against h1, which is b4 at the 100M budget.
"h1_silu100m|a12 + SiLU at 100M, no curriculum. The honest reference for the architecture arms.|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.optim.kl_adapt_step=iteration agent.ppo.entropy_coeff=0.0 agent.policy.normalize_input=true agent.value_function.normalize_input=true agent.policy.activation_fn=silu agent.value_function.activation_fn=silu"
"h2_silu_widenet|h1 + [1024,1024,512] on both networks.|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.optim.kl_adapt_step=iteration agent.ppo.entropy_coeff=0.0 agent.policy.normalize_input=true agent.value_function.normalize_input=true agent.policy.activation_fn=silu agent.value_function.activation_fn=silu agent.policy.num_cells=[1024,1024,512] agent.value_function.num_cells=[1024,1024,512]"
"h3_silu_sonicnet|h1 + the release architecture (2048x6). Capacity on a base whose episode length is not inflated by looser terminations.|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.optim.kl_adapt_step=iteration agent.ppo.entropy_coeff=0.0 agent.policy.normalize_input=true agent.value_function.normalize_input=true agent.policy.activation_fn=silu agent.value_function.activation_fn=silu agent.policy.num_cells=[2048,2048,1024,1024,512,512] agent.value_function.num_cells=[2048,2048,1024,1024,512,512]"
# --- Round 9: on h2, the honest 100M champion -----------------------------------
# At 100M the capacity question flipped. At 50M wider nets were neutral-to-
# negative; at 100M h2 (silu + [1024,1024,512]) beats h1 on all three -- ret/min
# 0.566 vs 0.555, len/min 6.36 vs 5.83, MPJPE 64.39 vs 65.70. h3 (2048x6) has the
# best tracking of any arm (62.52 mm) but costs 47 min against h2's 41, so its
# rates lag while its per-frame quality leads.
#
# The curriculum arms are excluded from the base: at matched 100M they run
# 80-90 mm MPJPE against the h-arms' 62-66, so their 1.6x rate advantage is still
# the relaxed-test artifact.
#
# i1 is the one that decides whether residual/LayerNorm machinery is worth
# building: it adds DEPTH at h2's width, so h3's tracking edge can be attributed
# to depth or to width. If depth alone helps, deeper is worth pursuing and
# gradient flow becomes the next constraint; if it does not, h3's edge is width
# and there is nothing for residual connections to fix.
"i1_deeper|h2 plus one more layer at the same width ([1024,1024,1024,512]): isolates depth from width in h3's tracking edge.|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.optim.kl_adapt_step=iteration agent.ppo.entropy_coeff=0.0 agent.policy.normalize_input=true agent.value_function.normalize_input=true agent.policy.activation_fn=silu agent.value_function.activation_fn=silu agent.policy.num_cells=[1024,1024,1024,512] agent.value_function.num_cells=[1024,1024,1024,512]"
"i2_desired_kl02|h2 with the adaptive-KL target doubled to 0.02, widening the dead band to [0.01, 0.04]. Never varied in any run in this project.|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.optim.kl_adapt_step=iteration agent.ppo.entropy_coeff=0.0 agent.policy.normalize_input=true agent.value_function.normalize_input=true agent.policy.activation_fn=silu agent.value_function.activation_fn=silu agent.policy.num_cells=[1024,1024,512] agent.value_function.num_cells=[1024,1024,512] agent.optim.desired_kl=0.02"
"i3_gae_lambda098|h2 with GAE lambda 0.95 -> 0.98: less advantage bias, more variance.|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.optim.kl_adapt_step=iteration agent.ppo.entropy_coeff=0.0 agent.policy.normalize_input=true agent.value_function.normalize_input=true agent.policy.activation_fn=silu agent.value_function.activation_fn=silu agent.policy.num_cells=[1024,1024,512] agent.value_function.num_cells=[1024,1024,512] agent.ppo.gae_lambda=0.98"
"i4_action_rate_half|h2 with the action-rate penalty halved (-0.1 -> -0.05). It is the largest-magnitude reward term and was 28% of positive reward mass early.|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.optim.kl_adapt_step=iteration agent.ppo.entropy_coeff=0.0 agent.policy.normalize_input=true agent.value_function.normalize_input=true agent.policy.activation_fn=silu agent.value_function.activation_fn=silu agent.policy.num_cells=[1024,1024,512] agent.value_function.num_cells=[1024,1024,512] env.rewards.action_rate_l2.weight=-0.05"
"i5_epochs8|h2 with 5 -> 8 epochs. Sample reuse was only ever tuned on the pre-normalization base, where interactions differed.|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.optim.kl_adapt_step=iteration agent.ppo.entropy_coeff=0.0 agent.policy.normalize_input=true agent.value_function.normalize_input=true agent.policy.activation_fn=silu agent.value_function.activation_fn=silu agent.policy.num_cells=[1024,1024,512] agent.value_function.num_cells=[1024,1024,512] agent.loss.epochs=8"
# --- Round 10: stacking the round-9 winners -------------------------------------
# Two round-9 arms beat h2 on ALL THREE axes (both rates and MPJPE), which is the
# bar for a genuine win here: desired_kl 0.02 (0.762 / 7.43 / 63.38) and the
# halved action-rate penalty (0.697 / 7.58 / 63.23). Both are in j0's base.
#
# Rejected from round 9: gae_lambda 0.98 (worse on every axis), epochs 8 (better
# MPJPE but 45 min against 41, so both rates fall), and depth. i1_deeper scored
# 0.561 against h2's 0.566 -- depth at fixed width buys nothing, so h3's tracking
# edge is width, and there is no gradient-flow problem for residual connections
# or LayerNorm to solve. That line of work is closed.
#
# j1 is the honest curriculum test the 500M window could not give. Compressing
# the anneal to 10M-60M means it COMPLETES inside the budget, so the scored tail
# (last 20M frames) is measured at strict thresholds. If the curriculum is a
# genuine accelerator, j1 keeps its speed advantage with MPJPE in the 62-64 band;
# if it only ever relaxed the test, MPJPE returns to the pack and the rates fall
# back with it.
"j0_kl02_actrate|h2 + both round-9 winners: desired_kl 0.02 and action-rate penalty halved.|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.optim.kl_adapt_step=iteration agent.ppo.entropy_coeff=0.0 agent.policy.normalize_input=true agent.value_function.normalize_input=true agent.policy.activation_fn=silu agent.value_function.activation_fn=silu agent.policy.num_cells=[1024,1024,512] agent.value_function.num_cells=[1024,1024,512] agent.optim.desired_kl=0.02 env.rewards.action_rate_l2.weight=-0.05"
"j1_curr_compressed|j0 + the termination anneal compressed to 10M-60M so it finishes inside the budget and the scored tail runs at strict thresholds.|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.optim.kl_adapt_step=iteration agent.ppo.entropy_coeff=0.0 agent.policy.normalize_input=true agent.value_function.normalize_input=true agent.policy.activation_fn=silu agent.value_function.activation_fn=silu agent.policy.num_cells=[1024,1024,512] agent.value_function.num_cells=[1024,1024,512] agent.optim.desired_kl=0.02 env.rewards.action_rate_l2.weight=-0.05 env.enable_termination_curriculum=true env.termination_curriculum_start_frames=10000000 env.termination_curriculum_end_frames=60000000"
"j2_actrate_quarter|j0 with the action-rate penalty at -0.025: pushes the winning direction further.|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.optim.kl_adapt_step=iteration agent.ppo.entropy_coeff=0.0 agent.policy.normalize_input=true agent.value_function.normalize_input=true agent.policy.activation_fn=silu agent.value_function.activation_fn=silu agent.policy.num_cells=[1024,1024,512] agent.value_function.num_cells=[1024,1024,512] agent.optim.desired_kl=0.02 env.rewards.action_rate_l2.weight=-0.025"
"j3_desired_kl04|j0 with desired_kl 0.04: pushes the other winning direction further.|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.optim.kl_adapt_step=iteration agent.ppo.entropy_coeff=0.0 agent.policy.normalize_input=true agent.value_function.normalize_input=true agent.policy.activation_fn=silu agent.value_function.activation_fn=silu agent.policy.num_cells=[1024,1024,512] agent.value_function.num_cells=[1024,1024,512] agent.optim.desired_kl=0.04 env.rewards.action_rate_l2.weight=-0.05"
"j4_kl02_actrate_sonicnet|j0 with the 2048x6 architecture, which had the best MPJPE of any arm (62.52) on the weaker base.|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.optim.kl_adapt_step=iteration agent.ppo.entropy_coeff=0.0 agent.policy.normalize_input=true agent.value_function.normalize_input=true agent.policy.activation_fn=silu agent.value_function.activation_fn=silu agent.policy.num_cells=[2048,2048,1024,1024,512,512] agent.value_function.num_cells=[2048,2048,1024,1024,512,512] agent.optim.desired_kl=0.02 env.rewards.action_rate_l2.weight=-0.05"
# --- Round 11: the curriculum is real once the anneal completes -----------------
# j1 settled the question the earlier curriculum arms could not. With the anneal
# compressed to 10M-60M it FINISHES inside the budget, so the scored tail runs at
# strict thresholds -- and MPJPE came back at 63.87, inside the honest 62-64
# band, with ret/min 1.071 and len/min 10.17 in 28 minutes. The 89 mm readings
# from b5/g0-g4 were an anneal still 89% incomplete at 100M being scored under
# loose thresholds, not a real loss of tracking quality.
#
# Also promoted: action-rate at -0.025 (j2: 0.956 / 8.93 / 62.33, the best MPJPE
# of round 10 bar j0). Rejected: the 2048x6 architecture, which loses on the
# stronger base (j4 0.689) as it did on the weaker one.
#
# k3/k4 vary only WHERE the anneal ends, which is now the highest-leverage free
# parameter: it trades how long the policy trains under loose thresholds against
# how much strict-threshold training it gets before scoring.
"k0_combined|Everything that has won: silu, [1024,1024,512], kl_iter, entropy 0, obs norm, desired_kl 0.02, action-rate -0.025, anneal 10M-60M.|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.optim.kl_adapt_step=iteration agent.ppo.entropy_coeff=0.0 agent.policy.normalize_input=true agent.value_function.normalize_input=true agent.policy.activation_fn=silu agent.value_function.activation_fn=silu agent.policy.num_cells=[1024,1024,512] agent.value_function.num_cells=[1024,1024,512] agent.optim.desired_kl=0.02 env.rewards.action_rate_l2.weight=-0.025 env.enable_termination_curriculum=true env.termination_curriculum_start_frames=10000000 env.termination_curriculum_end_frames=60000000"
"k1_kl04|k0 with desired_kl 0.04 (j3 beat j0 on rates at 0.02 -> 0.04).|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.optim.kl_adapt_step=iteration agent.ppo.entropy_coeff=0.0 agent.policy.normalize_input=true agent.value_function.normalize_input=true agent.policy.activation_fn=silu agent.value_function.activation_fn=silu agent.policy.num_cells=[1024,1024,512] agent.value_function.num_cells=[1024,1024,512] agent.optim.desired_kl=0.04 env.rewards.action_rate_l2.weight=-0.025 env.enable_termination_curriculum=true env.termination_curriculum_start_frames=10000000 env.termination_curriculum_end_frames=60000000"
"k2_actrate_zero|k0 with the action-rate penalty removed entirely. Halving then quartering both won; this finds the end of that direction.|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.optim.kl_adapt_step=iteration agent.ppo.entropy_coeff=0.0 agent.policy.normalize_input=true agent.value_function.normalize_input=true agent.policy.activation_fn=silu agent.value_function.activation_fn=silu agent.policy.num_cells=[1024,1024,512] agent.value_function.num_cells=[1024,1024,512] agent.optim.desired_kl=0.02 env.rewards.action_rate_l2.weight=0.0 env.enable_termination_curriculum=true env.termination_curriculum_start_frames=10000000 env.termination_curriculum_end_frames=60000000"
"k3_curr_early|k0 with the anneal at 5M-30M: reaches strict thresholds sooner, so 70M of the budget is scored strict rather than 40M.|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.optim.kl_adapt_step=iteration agent.ppo.entropy_coeff=0.0 agent.policy.normalize_input=true agent.value_function.normalize_input=true agent.policy.activation_fn=silu agent.value_function.activation_fn=silu agent.policy.num_cells=[1024,1024,512] agent.value_function.num_cells=[1024,1024,512] agent.optim.desired_kl=0.02 env.rewards.action_rate_l2.weight=-0.025 env.enable_termination_curriculum=true env.termination_curriculum_start_frames=5000000 env.termination_curriculum_end_frames=30000000"
"k4_curr_late|k0 with the anneal at 10M-100M, finishing exactly at the budget end: maximum time under loose thresholds, and the strictest test of whether that is the mechanism.|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.optim.kl_adapt_step=iteration agent.ppo.entropy_coeff=0.0 agent.policy.normalize_input=true agent.value_function.normalize_input=true agent.policy.activation_fn=silu agent.value_function.activation_fn=silu agent.policy.num_cells=[1024,1024,512] agent.value_function.num_cells=[1024,1024,512] agent.optim.desired_kl=0.02 env.rewards.action_rate_l2.weight=-0.025 env.enable_termination_curriculum=true env.termination_curriculum_start_frames=10000000 env.termination_curriculum_end_frames=100000000"
# --- Round 12: combine the round-11 winners, and re-test on the new base --------
# k2 (action-rate removed) led on both rates at 1.237 / 11.33 with MPJPE 61.69;
# k3 (anneal 5M-30M) had the best MPJPE of the campaign at 60.83. m0 is both.
#
# k3 vs k4 also confirmed the curriculum mechanism outright: ending the anneal at
# 30M rather than 100M buys 60.83 mm against 65.25, because more of the budget
# then trains under strict thresholds. Earlier completion is better, which m1
# pushes to its limit.
#
# desired_kl is peaked at 0.02 (k1 at 0.04 was worse on both rates and MPJPE).
#
# m2/m3/m4 re-test three knobs that were only ever measured on the ORIGINAL base,
# before normalization, SiLU, the wider net and the curriculum. Interactions have
# already flipped a result once in this campaign -- kl_iter hurt without
# normalization and helped with it -- so a stale rejection is not evidence.
"m0_best|k2 + k3: action-rate removed and the anneal completed by 30M.|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.optim.kl_adapt_step=iteration agent.ppo.entropy_coeff=0.0 agent.policy.normalize_input=true agent.value_function.normalize_input=true agent.policy.activation_fn=silu agent.value_function.activation_fn=silu agent.policy.num_cells=[1024,1024,512] agent.value_function.num_cells=[1024,1024,512] agent.optim.desired_kl=0.02 env.rewards.action_rate_l2.weight=0.0 env.enable_termination_curriculum=true env.termination_curriculum_start_frames=5000000 env.termination_curriculum_end_frames=30000000"
"m1_curr_earliest|m0 with the anneal at 2M-15M, the earliest completion tested.|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.optim.kl_adapt_step=iteration agent.ppo.entropy_coeff=0.0 agent.policy.normalize_input=true agent.value_function.normalize_input=true agent.policy.activation_fn=silu agent.value_function.activation_fn=silu agent.policy.num_cells=[1024,1024,512] agent.value_function.num_cells=[1024,1024,512] agent.optim.desired_kl=0.02 env.rewards.action_rate_l2.weight=0.0 env.enable_termination_curriculum=true env.termination_curriculum_start_frames=2000000 env.termination_curriculum_end_frames=15000000"
"m2_entropy_1e3|m0 with entropy 0.001. The bonus was zeroed on the pre-normalization base; sigma dynamics differ now.|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.optim.kl_adapt_step=iteration agent.ppo.entropy_coeff=0.001 agent.policy.normalize_input=true agent.value_function.normalize_input=true agent.policy.activation_fn=silu agent.value_function.activation_fn=silu agent.policy.num_cells=[1024,1024,512] agent.value_function.num_cells=[1024,1024,512] agent.optim.desired_kl=0.02 env.rewards.action_rate_l2.weight=0.0 env.enable_termination_curriculum=true env.termination_curriculum_start_frames=5000000 env.termination_curriculum_end_frames=30000000"
"m3_updates_2x|m0 with mini_batch 9216 (542 updates per M frames). Doubling updates lost on the original base; the base has changed.|agent.loss.mini_batch_size=${HP_SCREEN_DOUBLE_UPDATES_MINIBATCH} agent.optim.kl_adapt_step=iteration agent.ppo.entropy_coeff=0.0 agent.policy.normalize_input=true agent.value_function.normalize_input=true agent.policy.activation_fn=silu agent.value_function.activation_fn=silu agent.policy.num_cells=[1024,1024,512] agent.value_function.num_cells=[1024,1024,512] agent.optim.desired_kl=0.02 env.rewards.action_rate_l2.weight=0.0 env.enable_termination_curriculum=true env.termination_curriculum_start_frames=5000000 env.termination_curriculum_end_frames=30000000"
"m4_gamma097|m0 with discount 0.97. It was mixed on the original base (return up, length down); episodes are now twice as long.|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.optim.kl_adapt_step=iteration agent.ppo.entropy_coeff=0.0 agent.policy.normalize_input=true agent.value_function.normalize_input=true agent.policy.activation_fn=silu agent.value_function.activation_fn=silu agent.policy.num_cells=[1024,1024,512] agent.value_function.num_cells=[1024,1024,512] agent.optim.desired_kl=0.02 env.rewards.action_rate_l2.weight=0.0 env.enable_termination_curriculum=true env.termination_curriculum_start_frames=5000000 env.termination_curriculum_end_frames=30000000 agent.loss.gamma=0.97"
# --- Round 13: is the curriculum window budget-relative or absolute? ------------
# The one thing standing between this recipe and a 5B run. At 100M the window is
# well mapped -- 2M-15M (m1), 5M-30M (m0/k3, best), 10M-60M (j1/k0), 10M-100M
# (k4, worst MPJPE) -- but every one of those shares a budget, so the optimum
# cannot be told apart from a fraction of it. "5M-30M" is simultaneously
# "5%-30% of budget", and those extrapolate to very different settings at 5B:
# 5M-30M absolute, or 250M-1.5B scaled.
#
# Run both at 200M, where the two hypotheses separate for the first time. If
# they tie, the window is simply insensitive in this range -- also worth knowing,
# because it would mean the knob needs no retuning at scale.
"n1_window_absolute|Champion at 200M, anneal held at 5M-30M (absolute frames).|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.optim.kl_adapt_step=iteration agent.ppo.entropy_coeff=0.0 agent.policy.normalize_input=true agent.value_function.normalize_input=true agent.policy.activation_fn=silu agent.value_function.activation_fn=silu agent.policy.num_cells=[1024,1024,512] agent.value_function.num_cells=[1024,1024,512] agent.optim.desired_kl=0.02 env.rewards.action_rate_l2.weight=0.0 agent.loss.gamma=0.97 env.enable_termination_curriculum=true env.termination_curriculum_start_frames=5000000 env.termination_curriculum_end_frames=30000000"
"n2_window_scaled|Champion at 200M, anneal scaled to 10M-60M -- the same 5%-30% of budget that won at 100M.|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.optim.kl_adapt_step=iteration agent.ppo.entropy_coeff=0.0 agent.policy.normalize_input=true agent.value_function.normalize_input=true agent.policy.activation_fn=silu agent.value_function.activation_fn=silu agent.policy.num_cells=[1024,1024,512] agent.value_function.num_cells=[1024,1024,512] agent.optim.desired_kl=0.02 env.rewards.action_rate_l2.weight=0.0 agent.loss.gamma=0.97 env.enable_termination_curriculum=true env.termination_curriculum_start_frames=10000000 env.termination_curriculum_end_frames=60000000"
# --- Round 14: axes never touched on any base ----------------------------------
# Everything so far has moved the optimizer, the architecture, the discount, one
# reward weight and the terminations. These five are untested anywhere in the
# campaign, and two of them are only now worth asking:
#
#   p2  lr_adaptation_factor was 1.5 back when the rule fired per MINIBATCH and
#       a random walk swamped it. Now that it fires once per iteration the step
#       size is a real knob rather than noise amplitude.
#   p3  the r24-at-matched-update-density cell. a8 was meant to answer it, was
#       cancelled mid-run, and the r12-vs-r24 confound is still formally open --
#       on a base four rounds stronger than the one it was designed against.
#
# p4 leans on a measurement rather than a guess: tracking_reward_points at
# weight 2.0 is the single largest positive term (0.357 of ~1.21 total positive
# mass early), so it is where extra weight has the most leverage.
"p1_clip_eps03|Champion + PPO clip 0.2 -> 0.3. Never varied in this project.|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.optim.kl_adapt_step=iteration agent.ppo.entropy_coeff=0.0 agent.policy.normalize_input=true agent.value_function.normalize_input=true agent.policy.activation_fn=silu agent.value_function.activation_fn=silu agent.policy.num_cells=[1024,1024,512] agent.value_function.num_cells=[1024,1024,512] agent.optim.desired_kl=0.02 env.rewards.action_rate_l2.weight=0.0 agent.loss.gamma=0.97 env.enable_termination_curriculum=true env.termination_curriculum_start_frames=5000000 env.termination_curriculum_end_frames=30000000 agent.ppo.clip_epsilon=0.3"
"p2_lr_factor12|Champion + lr_adaptation_factor 1.5 -> 1.2: a gentler LR step, meaningful only now the rule is per-iteration.|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.optim.kl_adapt_step=iteration agent.ppo.entropy_coeff=0.0 agent.policy.normalize_input=true agent.value_function.normalize_input=true agent.policy.activation_fn=silu agent.value_function.activation_fn=silu agent.policy.num_cells=[1024,1024,512] agent.value_function.num_cells=[1024,1024,512] agent.optim.desired_kl=0.02 env.rewards.action_rate_l2.weight=0.0 agent.loss.gamma=0.97 env.enable_termination_curriculum=true env.termination_curriculum_start_frames=5000000 env.termination_curriculum_end_frames=30000000 agent.optim.lr_adaptation_factor=1.2"
"p3_r24_matched|Champion at 24-step rollouts, mini_batch HELD at 18432 so update density stays 271 per M frames (24 steps x 12288 = 294912 per iteration / 18432 = 16 minibatches x 5 epochs = 80 updates). Scaling the minibatch to 36864 instead would give 8 minibatches, 40 updates, 136 per M frames -- exactly the confound that made the original cluster r12-vs-r24 comparison unreadable.|ROLLOUT_STEPS=24 agent.loss.mini_batch_size=18432 agent.optim.kl_adapt_step=iteration agent.ppo.entropy_coeff=0.0 agent.policy.normalize_input=true agent.value_function.normalize_input=true agent.policy.activation_fn=silu agent.value_function.activation_fn=silu agent.policy.num_cells=[1024,1024,512] agent.value_function.num_cells=[1024,1024,512] agent.optim.desired_kl=0.02 env.rewards.action_rate_l2.weight=0.0 agent.loss.gamma=0.97 env.enable_termination_curriculum=true env.termination_curriculum_start_frames=5000000 env.termination_curriculum_end_frames=30000000"
"p4_tracking_points_2x|Champion + tracking_reward_points 2.0 -> 4.0, the largest positive reward term.|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.optim.kl_adapt_step=iteration agent.ppo.entropy_coeff=0.0 agent.policy.normalize_input=true agent.value_function.normalize_input=true agent.policy.activation_fn=silu agent.value_function.activation_fn=silu agent.policy.num_cells=[1024,1024,512] agent.value_function.num_cells=[1024,1024,512] agent.optim.desired_kl=0.02 env.rewards.action_rate_l2.weight=0.0 agent.loss.gamma=0.97 env.enable_termination_curriculum=true env.termination_curriculum_start_frames=5000000 env.termination_curriculum_end_frames=30000000 env.rewards.tracking_reward_points.weight=4.0"
"p5_critic_coeff05|Champion + critic_coeff 1.0 -> 0.5: less value-loss pull on shared optimisation.|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.optim.kl_adapt_step=iteration agent.ppo.entropy_coeff=0.0 agent.policy.normalize_input=true agent.value_function.normalize_input=true agent.policy.activation_fn=silu agent.value_function.activation_fn=silu agent.policy.num_cells=[1024,1024,512] agent.value_function.num_cells=[1024,1024,512] agent.optim.desired_kl=0.02 env.rewards.action_rate_l2.weight=0.0 agent.loss.gamma=0.97 env.enable_termination_curriculum=true env.termination_curriculum_start_frames=5000000 env.termination_curriculum_end_frames=30000000 agent.ppo.critic_coeff=0.5"
# --- Round 15: more environments, since the H100 is barely loaded --------------
# Measured on the champion at 12288 envs: 45.2 GB of 81.6 GB VRAM (52.8%) and
# 13.2% mean GPU utilization, 39% peak. Neither memory nor compute is the
# binding constraint, so environments are the obvious thing to buy -- and they
# buy wall-clock directly, which is the metric.
#
# All three hold the 99,975,168-frame budget exactly (only some env x rollout
# products divide it) and hold updates-per-frame at 271, since that is
# epochs/mini_batch_size and does not involve the batch size at all.
#
# What DOES change with a bigger batch is how often the LR controller fires: it
# now runs once per ITERATION, so q1 at 339 iterations gets half the adaptations
# of the 678 the champion gets. q2 exists to separate that from the env count --
# it doubles environments while HOLDING batch, iterations and LR steps fixed, by
# halving the rollout to 6. If q2 wins and q1 does not, the cost was the halved
# LR adaptation, not the parallelism.
#
# q2/q3 shorten the rollout, which truncates the GAE horizon (lambda 0.95,
# gamma 0.97) to 6 and 9 steps. That is a real cost and the reason q3 exists at
# a milder 1.33x.
"q1_envs24k_r12|24576 envs x 12 = 2x envs and 2x batch, 339 iterations. Most parallelism, fewest LR adaptations.|ENVS=24576 agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.optim.kl_adapt_step=iteration agent.ppo.entropy_coeff=0.0 agent.policy.normalize_input=true agent.value_function.normalize_input=true agent.policy.activation_fn=silu agent.value_function.activation_fn=silu agent.policy.num_cells=[1024,1024,512] agent.value_function.num_cells=[1024,1024,512] agent.optim.desired_kl=0.02 env.rewards.action_rate_l2.weight=0.0 agent.loss.gamma=0.97 env.rewards.tracking_reward_points.weight=4.0 env.enable_termination_curriculum=true env.termination_curriculum_start_frames=5000000 env.termination_curriculum_end_frames=30000000"
"q2_envs24k_r6|24576 envs x 6 = 2x envs at UNCHANGED batch, iterations and LR steps. Isolates parallelism from batch size.|ENVS=24576 ROLLOUT_STEPS=6 agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.optim.kl_adapt_step=iteration agent.ppo.entropy_coeff=0.0 agent.policy.normalize_input=true agent.value_function.normalize_input=true agent.policy.activation_fn=silu agent.value_function.activation_fn=silu agent.policy.num_cells=[1024,1024,512] agent.value_function.num_cells=[1024,1024,512] agent.optim.desired_kl=0.02 env.rewards.action_rate_l2.weight=0.0 agent.loss.gamma=0.97 env.rewards.tracking_reward_points.weight=4.0 env.enable_termination_curriculum=true env.termination_curriculum_start_frames=5000000 env.termination_curriculum_end_frames=30000000"
"q3_envs16k_r9|16384 envs x 9 = 1.33x envs at unchanged batch and iterations. The conservative step, and the least GAE truncation.|ENVS=16384 ROLLOUT_STEPS=9 agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.optim.kl_adapt_step=iteration agent.ppo.entropy_coeff=0.0 agent.policy.normalize_input=true agent.value_function.normalize_input=true agent.policy.activation_fn=silu agent.value_function.activation_fn=silu agent.policy.num_cells=[1024,1024,512] agent.value_function.num_cells=[1024,1024,512] agent.optim.desired_kl=0.02 env.rewards.action_rate_l2.weight=0.0 agent.loss.gamma=0.97 env.rewards.tracking_reward_points.weight=4.0 env.enable_termination_curriculum=true env.termination_curriculum_start_frames=5000000 env.termination_curriculum_end_frames=30000000"
# --- Round 16: wall-clock budget, and the SONIC action-rate penalty restored ---
# Screening is now budgeted in wall-clock rather than frames: the objective is
# per-minute progress, so the honest question is how far each arm gets in the
# same time. Both arms run the same non-binding iteration cap and are scored by
# interpolating at a matched training-minute mark.
#
# r1 restores action_rate_l2 to SONIC's -0.1, which the screen had driven to 0.
# THE RETURNS OF r0 AND r1 ARE NOT COMPARABLE -- r1 adds a negative term r0 does
# not have, so r1 is charged for behaviour r0 gets for free. Compare them on
# MPJPE and episode length, which no reward weight can move. That is the same
# rule that applies to tracking_reward_points at 4.0 in both arms, and the same
# reason the termination-curriculum arm had to be judged on MPJPE.
#
# The question r1 settles: dropping the penalty was measured as a win, but only
# ever on 50M/100M frame-bound screens where the smoothness it buys may not have
# had time to matter. It is also the term SONIC ships, and a 5B run keeping it
# would be the safer default if it costs nothing.
"r0_champion_wallclock|Champion under the wall-clock protocol: the reference r1 is read against.|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.optim.kl_adapt_step=iteration agent.ppo.entropy_coeff=0.0 agent.policy.normalize_input=true agent.value_function.normalize_input=true agent.policy.activation_fn=silu agent.value_function.activation_fn=silu agent.policy.num_cells=[1024,1024,512] agent.value_function.num_cells=[1024,1024,512] agent.optim.desired_kl=0.02 agent.loss.gamma=0.97 env.rewards.tracking_reward_points.weight=4.0 env.enable_termination_curriculum=true env.termination_curriculum_start_frames=5000000 env.termination_curriculum_end_frames=30000000 env.rewards.action_rate_l2.weight=0.0"
"r1_actrate_sonic|Champion with SONIC's action_rate_l2 = -0.1 restored. Judge on MPJPE and episode length; return is not comparable to r0.|agent.loss.mini_batch_size=${HP_SCREEN_BASE_MINIBATCH} agent.optim.kl_adapt_step=iteration agent.ppo.entropy_coeff=0.0 agent.policy.normalize_input=true agent.value_function.normalize_input=true agent.policy.activation_fn=silu agent.value_function.activation_fn=silu agent.policy.num_cells=[1024,1024,512] agent.value_function.num_cells=[1024,1024,512] agent.optim.desired_kl=0.02 agent.loss.gamma=0.97 env.rewards.tracking_reward_points.weight=4.0 env.enable_termination_curriculum=true env.termination_curriculum_start_frames=5000000 env.termination_curriculum_end_frames=30000000 env.rewards.action_rate_l2.weight=-0.1"
)











HP_SCREEN_ALL_ARM_NAMES=()
for _hp_screen_spec in "${HP_SCREEN_ARM_SPECS[@]}"; do
    HP_SCREEN_ALL_ARM_NAMES+=("${_hp_screen_spec%%|*}")
done
unset _hp_screen_spec
