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
)

EVAL_SCREEN_ALL_ARM_NAMES=()
for _spec in "${EVAL_SCREEN_ARM_SPECS[@]}"; do
    EVAL_SCREEN_ALL_ARM_NAMES+=("${_spec%%|*}")
done
