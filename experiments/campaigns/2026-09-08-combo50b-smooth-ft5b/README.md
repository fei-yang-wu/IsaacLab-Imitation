# Combo50B smoothness fine-tuning, +5B per arm

Three user-authorized single-seed H200 continuations of the exact final
model_step_50000166912.pt. No new encoder training; retain p5_affine z64+phase,
ten-step actor history, fullbatch/3epochs, EE/wide rewards, Newton physics,
16,384 environments, 24-step rollout and final reset mix (20% uniform).

| Arm | Sole smoothness change |
| --- | --- |
| action01 | action_rate_l2 -0.03 -> -0.1 |
| antishake4 | anti_shake_ang_vel -0.005 -> -0.02; action-rate stays -0.03 |
| ema08 | EMAJointPositionAction alpha=0.8; rewards unchanged |

Resume checkpoint cumulative_env_frames verified locally: 50,000,166,912.
Requested cap 55,000,166,912; rollout rounding reaches 55,000,301,568:
5,000,134,656 additional frames per arm. Each uses one H200, 160G RAM,
16 CPUs, up to two 15:59 walltime segments with afterany continuation.
Second segment resumes the arm's output tree and carries the same total cap;
it adds no budget when the first segment completes. Checkpoints every 0.5B
under /data/combo50b_smooth_ft5b/<arm>_seed0/tracker. Independent W&B runs
in g1-bs-pareto / combo50b-smooth-ft5b.

Preserve loaded optimizer state: actor/log_std LR 1.7341529915832615e-5;
critic LR 1e-5. Set critic_lr_schedule=constant so changing the total cap
cannot reheat the finished 50B linear critic schedule. Other optimizer behavior
is inherited, including actor adaptation. No reset curriculum restart.

EMA uses the existing action-manager term, trained through and repeated at
evaluation/runtime with alpha=0.8. It filters scaled joint-position targets;
alpha weights the NEW target. Its state resets between episodes.

All three frozen plans passed remote dataset/checkpoint/storage/container
preflight. Plans under logs/cluster_control/combo50b-smooth-ft5b/.
Evaluate final/checkpoint rows on bones_testbed4096_v1 and frozen
sonic_capability124_v1, clean plus robust, carrying each arm's environment
contract. Compare SR, MPJPE-L/G, jerk and action delta together; use matched
completed-clip intersections for smoothness and keep single-seed caveats.
No extra control arm or evaluation job submitted as part of this request.

## Submission verified

| Arm | First segment | Resume fallback |
| --- | --- | --- |
| action01 | 5738566 | 5738568 |
| antishake4 | 5738572 | 5738573 |
| ema08 | 5738575 | 5738576 |

All three use the same uploaded working-tree snapshot:
8d0eb0ba4b073877e50d55b2e8f8b80d2dce20a2bbcbc1c191a94644c01f854d.
Plan SHA prefixes: action01 8e144b27; antishake4 d7d2dbdb; ema08 edab19c1.
Submission succeeded for all six jobs; fallback dependencies are afterany.
