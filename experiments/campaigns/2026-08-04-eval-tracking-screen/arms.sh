# Arms for the 2026-08-04 eval-time tracking screen. Sourced by the launcher.
#
# GOAL: lower EVAL-TIME MPJPE and EE error. Arms are scored by evaluating the
# 500M checkpoint under the tracking-fidelity protocol (`--randomization none`,
# MODE actions), NOT by the training curve.
#
# WHY THESE ARMS. At the measured operating point of the 1.9B checkpoint
# (MPJPE-L 20.2 mm, EE 53 mm, DR off) the exponential reward kernels are
# saturated exactly where we want precision:
#
#   term                       std   error    reward   d(w*r)/d(err)
#   motion_body_pos           0.30  20.2 mm   0.9955        -0.45
#   motion_body_lin_vel       1.00       -    0.9608        -0.38
#   motion_body_ang_vel       3.14       -    0.9750        -0.10
#   tracking_reward_points    0.10  53.0 mm   0.7511       -32.15
#   motion_foot_pos           0.10  53.0 mm   0.7511       -16.07
#
# `motion_body_pos` is the term whose error IS MPJPE, and it supplies 72x less
# gradient than `tracking_reward_points`. Below ~20 mm it is flat: the policy is
# paid almost nothing for improving. Sharpening its kernel is therefore the most
# direct lever on the metric we are trying to move, and it costs no new term.
#
# The velocity terms are saturated for the same reason; sharpening them is the
# same intervention on the derivative channel, which feeds position tracking.
#
# CONTROL IS FREE: job 5561149 (`lafan1_v2_foot_reward_5b`) is the current
# default and passes 500M. Do not submit a control.
#
# Format: name|description|overrides

EVAL_SCREEN_ARM_SPECS=(
"s1_bodypos_std010|motion_body_pos kernel 0.30 -> 0.10, matching the std the two terms that still have gradient already use. Reward at 20 mm falls 0.9955 -> 0.96, gradient 9x.|env.rewards.motion_body_pos.params.std=0.1"
"s2_bodypos_std005|motion_body_pos kernel 0.30 -> 0.05. Reward at 20 mm falls to 0.85, gradient 29x. Brackets s1 from below; if precision is kernel-limited this should beat it, and if it is too sharp early training will destabilise.|env.rewards.motion_body_pos.params.std=0.05"
"s3_vel_sharp|Velocity kernels sharpened, lin 1.0 -> 0.3 and ang 3.14 -> 1.0, both ~96-98% saturated today. Velocity error is the derivative channel of position tracking, so this tests whether position precision is limited by an unrewarded velocity mismatch.|env.rewards.motion_body_lin_vel.params.std=0.3 env.rewards.motion_body_ang_vel.params.std=1.0"
"s4_bodypos_std010_w2|s1 plus double weight on the same term, 1.0 -> 2.0. Separates 'the kernel was flat' from 'the term was underweighted' -- if s1 wins and this does not add, the shape mattered and the scale did not.|env.rewards.motion_body_pos.params.std=0.1 env.rewards.motion_body_pos.weight=2.0"
# --- Round 2: EE tracking is a DRIFT problem, not a wrist problem -----------
#
# `ee_pos_error_m` in the evaluator is WORLD-frame (`actual_pos - ref_pos`, no
# root subtraction). Decomposing the DR-off operating point:
#
#   root drift (world)       54.7 mm
#   EE error (world)         53.5 mm   <- essentially all root drift
#   MPJPE-L (root-relative)  20.2 mm   <- the actual pose error
#
# So EE tracking is limited by global drift, and a wrist-specific reward is a
# second-order fix. The term that controls root position is
# `motion_global_anchor_pos`.
#
# CORRECTION, measured rather than assumed. The saturation figures above and in
# the round-1 block were computed at the EVALUATION operating point, which is
# the wrong place to judge a training gradient: training runs with domain
# randomization and exploration noise, so its errors are much larger. Inverting
# the control run's actual `Episode_Reward` at 260M -- IsaacLab logs
# `weight * mean(kernel) * ep_len/500`, so the kernel value is recoverable --
# gives the real training operating point:
#
#   term                      kernel  implied err  gradient
#   motion_body_pos            0.970       0.052     -1.13
#   motion_global_anchor_ori   0.933       0.106     -0.62
#   tracking_reward_points     0.870       0.037    -25.94
#   motion_body_lin_vel        0.866       0.379     -0.66
#   motion_foot_pos            0.849       0.040    -13.73
#   motion_body_ori            0.767       0.206     -1.97
#   motion_body_ang_vel        0.692       1.905     -0.27
#   motion_global_anchor_pos   0.599       0.215     -1.43
#
# `motion_body_pos` is confirmed as the most saturated term at 0.970, so s1/s2/
# s4 stand -- though the gradient deficit against `tracking_reward_points` is
# ~23x, not the 72x the eval-time estimate suggested.
#
# `motion_global_anchor_pos` is the LEAST saturated term at 0.599, so the
# "96.7% saturated" claim behind s5 is WRONG and s5 is only weakly motivated;
# it is kept as the sharpen-only control for s6. s6 survives on a different and
# independent argument: root drift is ~100% of world-frame EE error, and this
# term carries the lowest weight of any tracking term (0.5) with 18x less
# gradient than `tracking_reward_points`. The dominant error source is the
# least incentivised one, whether or not its kernel is saturated.
#
# s7 is kept anyway because it isolates the wrist contribution: if drift is the
# whole story it should do nothing, and that is worth knowing rather than
# assuming.
# --- Round 4: the trend has not turned, so find where it does ----------------
#
# At matched 500M against the control, sharpening motion_body_pos is monotone
# and large on the goal metric:
#
#   std 0.30 (control)  MPJPE 22.03  EE 0.0785  survival 444.6
#   std 0.10 (s1)       MPJPE 19.61  EE 0.0791  survival 442.1
#   std 0.05 (s2)       MPJPE 17.89  EE 0.0722  survival 439.1
#
# -18.8% MPJPE and -8.0% EE at std 0.05, well outside the ~2% seed spread, with
# a small survival cost (444.6 -> 439.1). Full-horizon MPJPE barely moves
# (43.51 -> 43.10), which is what the failure-dominance finding predicts: that
# pass is governed by the clips that fail, not by precision on the ones that do
# not.
#
# s10 continues the sweep. A kernel this narrow eventually stops helping -- the
# reward goes flat again below the achievable error, and the survival cost keeps
# accruing -- so the point is to find the turn rather than to assume 0.05 is it.
"s10_bodypos_std0025|motion_body_pos kernel 0.05 -> 0.025, continuing a monotone sweep that has not yet turned. Watch survival and the full-horizon pass as well as strict MPJPE: the sharper arms trade a little survival for precision.|env.rewards.motion_body_pos.params.std=0.025"
# --- Round 5: shape and scale are partly interchangeable, so combine them ----
#
#   arm                    MPJPE   EE      surv    full-horizon
#   control (0.30, w1)     22.03  0.0785  444.6      43.51
#   s1      (0.10, w1)     19.61  0.0791  442.1      43.42
#   s2      (0.05, w1)     17.89  0.0722  439.1      43.10
#   s4      (0.10, w2)     18.17  0.0820  442.6      39.81
#
# s4 is the answer to what s1-vs-s4 was built to ask: doubling the WEIGHT gets
# nearly the same strict MPJPE as halving the STD, so the two are partly
# interchangeable there. But they differ elsewhere -- s4 is the only arm that
# meaningfully improves the full-horizon pass (-8.5%) and it costs less
# survival, while s2 wins strict MPJPE and EE.
#
# s11 takes both. If shape and scale act on different failure modes it should
# beat each; if they are redundant it will land between them, which is equally
# worth knowing before either is promoted to a 5B default.
"s11_bodypos_std005_w2|motion_body_pos std 0.05 AND weight 2.0 -- s2's kernel with s4's scale. s2 wins strict MPJPE and EE, s4 wins full-horizon and survival; this tests whether the two are additive.|env.rewards.motion_body_pos.params.std=0.05 env.rewards.motion_body_pos.weight=2.0"
# --- Round 6: MPJPE and EE are won by DIFFERENT arms, so combine them --------
#
#   arm                  MPJPE     EE     surv   full-horizon
#   control              22.03  0.0785   444.6      43.51
#   s2  (body 0.05)      17.89  0.0722   439.1      43.10   best MPJPE  -18.8%
#   s4  (body 0.10 w2)   18.17  0.0820   442.6      39.81   best full-horizon
#   s7  (wrist reward)   19.11  0.0781   445.5      40.60   best survival
#   s6  (anchor 0.10 w2) 24.98  0.0599   443.8      43.35   best EE     -23.7%
#   s5  (anchor 0.10)    23.47  0.0821   444.2      44.18
#   s3  (velocity)       23.03  0.0842   442.4      46.34
#   s8  (foot allowance) 22.37  0.0762   443.4      44.44   neutral, as predicted
#
# The two goal metrics are won by different arms with OPPOSING trade-offs. s2
# sharpens the root-relative body term and wins MPJPE while EE gains little;
# s6 upweights the global root term and wins EE by 23.7% while MPJPE gets
# worse. That is exactly the decomposition: MPJPE-L is root-RELATIVE so the
# body kernel owns it, and world-frame EE is mostly root drift so the anchor
# term owns that.
#
# s12 takes one from each. If they are orthogonal -- and the decomposition says
# they should be, since they act on different components of the error -- it
# should win both at once.
#
# s8 landing neutral (+1.5%) confirms the pre-run eval-time prediction that
# foot_pos_xyz is a tripwire rather than the cause.
"s12_body005_anchor_w2|s2's body kernel (0.05) with s6's anchor term (std 0.10, weight 2.0). MPJPE-L is root-relative and owned by the body kernel; world-frame EE is mostly root drift and owned by the anchor term. Tests whether the two best arms are orthogonal.|env.rewards.motion_body_pos.params.std=0.05 env.rewards.motion_global_anchor_pos.params.std=0.1 env.rewards.motion_global_anchor_pos.weight=2.0"
"s13_body005_wrist|s2's body kernel with s7's wrist reward, the arm with the best survival and second-best full-horizon. Tests whether precision and survival gains stack.|env.rewards.motion_body_pos.params.std=0.05 env.rewards.motion_ee_pos.weight=2.0"
# --- Round 7: root ORIENTATION is the untouched half of the EE budget -------
#
#   run                EE(w)   root_pos  root_ori  MPJPE-L
#   control           0.0785    0.0707    0.0554    22.03
#   s6 anchor-pos w2  0.0599    0.0404    0.0625    24.98   EE -23.7%
#   s7 wrist reward   0.0781    0.0732    0.0588    19.11   EE  -0.5%
#   s11 best MPJPE    0.0777    0.0715    0.0603    17.90
#
# EE is world-frame and tracks root POSITION almost 1:1 -- s6's whole EE gain
# came from cutting drift 43%, and `EE - root_pos` is a near-constant 8-20 mm
# everywhere. s7 rewards the wrists directly and moved EE 0.5%, so the wrists
# are not mistracking; a wrist-specific reward is refuted.
#
# What NO arm has touched is root ORIENTATION: 0.055-0.063 rad in every single
# run. With a ~0.6 m mean lever arm to the EE bodies that is ~35 mm of EE error
# sitting untouched, and once s6 drives position drift to 40 mm it is nearly
# half the remaining budget.
#
# `motion_global_anchor_ori` is in exactly the state `motion_global_anchor_pos`
# was in before s6 -- kernel 0.933, weight 0.5, the second-most saturated and
# lowest-weighted term. s14 applies the recipe that worked there.
"s14_anchor_ori_w2|motion_global_anchor_ori std 0.40 -> 0.15 and weight 0.5 -> 2.0, the same treatment that made s6 the best EE arm, applied to the orientation term no arm has moved. Targets the ~35 mm orientation lever in the EE budget.|env.rewards.motion_global_anchor_ori.params.std=0.15 env.rewards.motion_global_anchor_ori.weight=2.0"
"s15_ee_stack|s11 best-MPJPE kernel plus BOTH anchor terms sharpened and upweighted -- position (s6) and orientation (s14). The full stack against the EE budget, if the two anchor components are independent.|env.rewards.motion_body_pos.params.std=0.05 env.rewards.motion_body_pos.weight=2.0 env.rewards.motion_global_anchor_pos.params.std=0.1 env.rewards.motion_global_anchor_pos.weight=2.0 env.rewards.motion_global_anchor_ori.params.std=0.15 env.rewards.motion_global_anchor_ori.weight=2.0"
# --- Round 8: give the BODIES a world-frame target, not just the root -------
#
#   arm                          MPJPE-G   EE-G    root   root_ori
#   control                       0.0758  0.0785  0.0707   0.0554
#   s6  (anchor-pos w2)           0.0535  0.0599  0.0404   0.0625
#   s12 (body + anchor-pos)       0.0494  0.0540  0.0411   0.0663
#   s15 (+ anchor-ori)            0.0439  0.0475  0.0385   0.0273   -42.1% / -39.5%
#
# s15 is the best arm and the first to move root_ori at all -- 0.055-0.066 rad
# in every other run, 0.0273 here. That confirms the lever-arm reading: the
# orientation term was worth ~33 mm of EE error and nothing had touched it.
#
# But s15 still anchors the world through the ROOT ALONE. Every body-level
# position reward remains rerooted and therefore drift-blind: roughly 8.0 of
# position-reward weight cannot see the world against 1.0 that can. s16 gives
# the bodies themselves a world-frame target using the new
# `reference_global_body_position_error_exp`.
"s16_global_bodies|s15 plus world-frame targets on the bodies, feet and wrists. Every body-level position reward today is rerooted and drift-blind; this is the untested continuation of the logic behind every gain so far.|env.rewards.motion_body_pos.params.std=0.05 env.rewards.motion_body_pos.weight=2.0 env.rewards.motion_global_anchor_pos.params.std=0.1 env.rewards.motion_global_anchor_pos.weight=2.0 env.rewards.motion_global_anchor_ori.params.std=0.15 env.rewards.motion_global_anchor_ori.weight=2.0 env.rewards.motion_body_pos_global.weight=2.0 env.rewards.motion_foot_pos_global.weight=1.0 env.rewards.motion_ee_pos_global.weight=1.0"
"s5_anchor_sharp|motion_global_anchor_pos kernel 0.30 -> 0.10. Targets root drift, which is ~100% of world-frame EE error and the larger half of global MPJPE. 96.7% saturated today.|env.rewards.motion_global_anchor_pos.params.std=0.1"
"s6_anchor_sharp_w2|s5 plus weight 0.5 -> 2.0, so the term controlling the dominant error source stops being the lowest-weighted tracking term. Gradient 0.59 -> 16.2, i.e. into the band where tracking_reward_points already operates.|env.rewards.motion_global_anchor_pos.params.std=0.1 env.rewards.motion_global_anchor_pos.weight=2.0"
# --- Round 3: the failures are ONE MOTION CLASS, and one missing allowance ---
#
# Per-environment survival on the pre-screen checkpoint, DR off, 10 envs:
#
#   fallAndGetUp1_subject1  290      dance2_subject1  500
#   fallAndGetUp2_subject3  328      dance2_subject2  500
#   fallAndGetUp2_subject2  348      dance2_subject3  500
#   fallAndGetUp1_subject5  382      dance2_subject4  500
#   fallAndGetUp1_subject4  403      dance2_subject5  500
#
# EVERY fall-and-get-up clip fails; EVERY dance clip survives the full horizon.
# Causes: foot_pos_xyz x4, anchor_ori x2. And the per-step curve is flat at
# ~11 mm for 300 steps before diverging, so this is not gradual drift -- it is
# these clips reaching their hard phase.
#
# There is a precise reason foot_pos_xyz is the one that fires. It is the only
# termination in the config without the crouching allowance: both
# `bad_anchor_pos_z_adaptive` and `bad_reference_body_pos_z_adaptive` relax to
# `down_threshold` when the REFERENCE root drops below `root_height_threshold`,
# and a fall-and-recover reference spends its hard phase exactly there. So the
# strict 0.2 m horizontal bar was being enforced precisely where the other two
# terms had already decided it should not be.
#
# s8 extends that existing pattern to the one term that lacked it. This is the
# most directly motivated arm in the screen: it targets the measured cause of
# the dominant failure, and it changes nothing while the reference is upright.
"s8_foot_allowance|Give foot_pos_xyz the crouching allowance the other two position terms already have: relax to 0.6 m when the reference root is below 0.5 m. Targets the fall-and-get-up failures that account for every non-surviving clip. No effect while the reference is upright.|env.terminations.foot_pos_xyz.params.down_threshold=0.6"
# s9 covers the regime s8 cannot. The failing clips are the DYNAMIC ones --
# jumps, runs, sprints, fights, falls -- while all 8 dance clips and 11 of 12
# walks survive. s8's allowance keys on a LOW reference root, so it fires for
# the falls and never for the airborne cases, whose root is high. During a
# flight phase the foot's horizontal position is not correctable at that
# instant and self-corrects on landing.
#
# Applied per body on the reference foot's own height: one foot is typically
# airborne while the other is planted, so a per-environment test would relax
# the stance foot too.
#
# HOLD until round 1 reports. This is a targeted follow-up, not a speculative
# addition to an already-contended queue.
"s9_foot_swing_allowance|foot_pos_xyz relaxes to 0.5 m for a foot whose REFERENCE is above 0.15 m, i.e. in flight, plus s8's low-root allowance. Targets the jump/run/sprint failures that s8 cannot reach.|env.terminations.foot_pos_xyz.params.down_threshold=0.6 env.terminations.foot_pos_xyz.params.swing_threshold=0.5"
"s7_ee_reward|Enable the inert wrist term motion_ee_pos at 2.0 -- same geometry as motion_foot_pos, on the hands. The wrists have no horizontal termination and only their 2-of-5 share of tracking_reward_points. Expected to be second-order if drift dominates.|env.rewards.motion_ee_pos.weight=2.0"
# --- Round 9: the observation space, which no arm has touched -----------------
#
# Every arm so far moved a reward std or weight. The seed repeat showed the
# full-horizon pass is dominated by WHICH CLIPS FALL -- and survival is the
# low-variance metric (438.7 vs 441.2 across s15's two seeds, under 1%). So the
# lever with room in it is survival, and the failures are concentrated:
# fallandgetup 0-1/6, fight 1/5, jump 1/3, run 1/4, against dance 8/8 and
# walk 11/12.
#
# Recovering from a stumble is the case where a single frame is least
# sufficient: the policy cannot tell a fall it is arresting from one it is
# starting. SONIC ships 10-step proprioceptive histories for this.
#
# PRIOR EVIDENCE, AND WHY IT IS NOT DECISIVE. The 2026-07-21 ablation
# (5525687 against 5525664) concluded histories "buy little at our scale", and
# `ImitationG1SonicNoHistorySurfaceEnvCfg` still carries that docstring. But it
# ran on the pre-v2 `Isaac-Imitation-G1-Latent-v0` surface, before the v2
# command interface and every reward change in this campaign, and it was a
# SINGLE SEED -- the same basis this campaign just showed cannot resolve a
# difference on the pass that matters.
#
# Verified before submission: the override is not a silent no-op. The actor
# observation goes 418 -> 1256 (projected_gravity 3->30, joint_pos_rel 29->290).
# Applied to policy AND critic, matching SONIC's recipe.
#
# NOTE: this arm changes the observation width, so evaluation must reproduce the
# override or the checkpoint will not load. See ARM_EVAL_EXTRA in the scorer.
"s17_history10|s15's rewards plus SONIC's 10-step proprioceptive histories on the actor and critic. The first arm to move the observation space rather than a reward coefficient, aimed at the fall-recovery clips where one frame cannot distinguish arresting a fall from starting one.|env.rewards.motion_body_pos.params.std=0.05 env.rewards.motion_body_pos.weight=2.0 env.rewards.motion_global_anchor_pos.params.std=0.1 env.rewards.motion_global_anchor_pos.weight=2.0 env.rewards.motion_global_anchor_ori.params.std=0.15 env.rewards.motion_global_anchor_ori.weight=2.0 env.observations.policy.projected_gravity.history_length=10 env.observations.policy.base_ang_vel.history_length=10 env.observations.policy.joint_pos_rel.history_length=10 env.observations.policy.joint_vel_rel.history_length=10 env.observations.policy.last_action.history_length=10 env.observations.critic.base_lin_vel.history_length=10 env.observations.critic.base_ang_vel.history_length=10 env.observations.critic.joint_pos_rel.history_length=10 env.observations.critic.joint_vel_rel.history_length=10 env.observations.critic.last_action.history_length=10"
)

# Arms whose training overrides change the OBSERVATION SPACE or the command
# interface must have those overrides reproduced at evaluation, or the actor is
# rebuilt at the wrong width and the checkpoint fails to restore. Reward-only
# arms need nothing here.
declare -A EVAL_SCREEN_ARM_EVAL_EXTRA=(
  ["s17_history10"]="env.observations.policy.projected_gravity.history_length=10 env.observations.policy.base_ang_vel.history_length=10 env.observations.policy.joint_pos_rel.history_length=10 env.observations.policy.joint_vel_rel.history_length=10 env.observations.policy.last_action.history_length=10 env.observations.critic.base_lin_vel.history_length=10 env.observations.critic.base_ang_vel.history_length=10 env.observations.critic.joint_pos_rel.history_length=10 env.observations.critic.joint_vel_rel.history_length=10 env.observations.critic.last_action.history_length=10"
)

EVAL_SCREEN_ALL_ARM_NAMES=()
for _spec in "${EVAL_SCREEN_ARM_SPECS[@]}"; do
    EVAL_SCREEN_ALL_ARM_NAMES+=("${_spec%%|*}")
done
