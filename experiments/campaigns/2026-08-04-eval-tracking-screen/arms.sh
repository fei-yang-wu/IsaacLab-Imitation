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
)

EVAL_SCREEN_ALL_ARM_NAMES=()
for _spec in "${EVAL_SCREEN_ARM_SPECS[@]}"; do
    EVAL_SCREEN_ALL_ARM_NAMES+=("${_spec%%|*}")
done
