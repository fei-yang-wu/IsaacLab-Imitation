#!/usr/bin/env bash
# Arm definitions for the BONES-SEED 129k ablation screen.
#
# Every arm is ONE delta from `control`, which reproduces the local 10B run's
# recipe exactly. An arm that needs two overrides to express one idea is fine;
# an arm that changes two ideas is not, because the screen cannot attribute it.
#
# The baseline every arm is measured against, from the local run scored on the
# fixed protocol (frame 0, --randomization none, MODE, 1024 envs, 500 steps):
#
#   0.35B  25.8 mm MPJPE  0.606 success  222.6 survival
#   2.00B  23.9           0.668          235.7
#   4.03B  23.4           0.692          241.7
#
# That curve was still descending at 4.03B, at roughly 0.25 mm per B frames, so
# an arm scored at 500M has to beat the control AT 500M -- not the 4B number --
# and has to clear that drift to mean anything.

# arm -> extra Hydra overrides, space separated. Empty for the control.
declare -A ABLATION_ARM_OVERRIDES=(
    [control]=""

    # A1. Backend. DROPPED FROM THE 2026-08-05 SCREEN -- see below. Kept because
    # the question is still live, not because the arm ran.
    #
    # A policy-free oracle probe has PhysX tracking the reference ~3x better
    # than Newton (joint MAE 0.0327 vs 0.0975 rad) with stock MuJoCo agreeing
    # with PhysX, so every Newton-trained checkpoint may be fitted to the
    # outlier. Expect ~0.6x Newton throughput; the budget is in frames.
    #
    # WHY IT IS DROPPED. Two attempts died at ~50 s: 5567113 on gpu:h200:1 and
    # 5567121 on gpu:h100:1, the configuration the 2026-08-03 5B run proved.
    # The GPU is NOT the cause and neither is the GPU policy guard -- the H100
    # log shows `PhysX GPU policy accepted: NVIDIA H100 80GB HBM3` followed by
    # `AppLauncher initialization complete`, so Kit starts on Hopper. The
    # process then exits AFTER config parsing and BEFORE env construction, with
    # status 0 and no traceback, so only the missing workload success marker
    # reports it. The `/isaac-sim/kit/data` write errors in the log are
    # non-fatal noise; Kit continued past them.
    #
    # BEFORE RESUBMITTING: reproduce interactively at --num_envs 64 and find the
    # exit path. Do not spend another 16 h slot on a blind retry.
    [physx]="physics=physx"

    # A2. Reset curriculum aggressiveness. The local run's training metrics were
    # flat while fixed-protocol MPJPE fell 25.8 -> 23.4 mm: the sampler was
    # hardening the task about as fast as the policy improved. `sonic` uses a
    # 200-frame pre-failure credit window against a 287-frame MEDIAN clip, and a
    # 200x max-over-mean cap where the `default` preset uses 50.
    [reset_window50]="env.command_interface.reference.selection.adaptive_pre_failure_window=50"
    [reset_cap50]="env.command_interface.reference.selection.adaptive_failure_rate_max_over_mean=50.0"

    # A3. Exploration floor. 0.008 constant, never annealed, while the tuned
    # recipe that won the HP search on LAFAN1 used 0. On 129k motions a
    # permanent floor is a plausible reason MPJPE parks instead of tightening.
    [entropy0]="agent.ipmd.entropy_coeff=0.0"

    # A4. Encoder adaptation. Frozen from a 50k-update pretrain and never
    # adapted to what the controller actually needs. hl_skill_lr=3e-5 is already
    # configured and currently unused.
    #
    # ANSWERED AT 2B, NEGATIVE: every metric 16-26% worse than control, survival
    # -20%, success -37%. Uniform degradation, not a trade-off. Do not rerun at
    # this learning rate without a reason that explains the sign.
    #
    # The result is not a pairing artifact. RLOpt DOES persist the finetuned
    # encoder -- inside `hl_skill_command_sampler_state_dict`, not as a
    # top-level key, so grepping the IPMD agent for `skill_encoder` finds
    # nothing. Stripping that block from the same checkpoint drops it to
    # survival 43.4 / success 0.003 against 195.4 / 0.456 intact, which both
    # proves the binding works and shows how hard the policy co-adapts.
    [encoder_finetune]="agent.ipmd.hl_skill_finetune_enabled=true agent.ipmd.hl_skill_lr=3e-05"

    # A5. Discount. gamma 0.97 at 50 Hz is a 0.67 s effective horizon. The two
    # dominant failures are ee_body_pos (0.162) and foot_pos_xyz (0.145), both
    # recovery-shaped, and recovery takes longer than that.
    [gamma099]="agent.loss.gamma=0.99"

    # A7. COMPOSITION -- deliberately two deltas, added 2026-08-05 after the 1B
    # scores separated the screen into exactly two real effects that trade
    # against each other:
    #
    #   rollout24  local MPJPE -9.5%, root ori -13.8%, surv +6.6%, global flat
    #   gamma099   global -24.6%, at the cost of local +27.4% and root ori +30.6%
    #
    # Every other arm left local MPJPE within +-1% of control. The question this
    # arm exists to answer is whether the two compose or whether the gamma
    # damage to local tracking dominates. It breaks the one-delta rule on
    # purpose, and it is only interpretable BECAUSE both singles were measured
    # first -- do not read it without them.
    [gamma099_rollout24]="agent.loss.gamma=0.99"
)

# Arms that change the collector geometry rather than a Hydra value. Rollout is
# not a plain override: frames_per_batch and the segment arithmetic both derive
# from it.
declare -A ABLATION_ARM_ROLLOUT=(
    # Pairs with the gamma override above; see A7.
    [gamma099_rollout24]=24

    # A6. Rollout length, on the axis nobody measured. The 2026-08-02 screen
    # compared 3/4/6/12 and reported "unchanged MPJPE" -- but its MPJPE column
    # is `mpjpe_mm`, which the aggregator resolves to `mpjpe_l_mm`, i.e.
    # ROOT-RELATIVE only. Global tracking was never in that table, 24 was never
    # tested, and the eval-tracking screen's own conclusion was that
    # accumulating root drift dominates eval-time failure. Report mpjpe_g and
    # anchor_pos_err for this arm, not just mpjpe_l.
    [rollout24]=24
)

# `physx` is deliberately absent: it is defined above but does not run by
# default after failing twice on 2026-08-05. Ask for it explicitly once the exit
# path is understood: ARMS="physx" ./submit_bones129k_ablation_ice.sh
ABLATION_ALL_ARM_NAMES=(
    reset_window50
    reset_cap50
    entropy0
    encoder_finetune
    gamma099
    rollout24
    gamma099_rollout24
)
