# 2026-09-14 ankle-limit fine-tunes off c1 (SUBMITTED)

Submitted 2026-09-14 06:15 EDT from a clean worktree of commit 496ec2f (no
drift), partition coe-gpu: c1l1 5778947 / 5778948, c1l2 5778950 / 5778951
(finetune1 / finetune2 afterany). Smokes from the committed state: c1l2
3-iteration fine-tune from the c1 76.0B file on a local reference-array
store (cluster data args, RLOpt a0add23), and a 64-env evaluate_checkpoint
run reporting the three joint-limit metrics, 0 errors.

Two fine-tunes off the c1 arm of `2026-09-13-e5-hardware-gap-10b` at its
76,000,002,048 checkpoint, 10B each (cap 86,000,002,048), one added reward
term per arm. c1 keeps training to 80B without the term, so c1 vs c1l* at
matched 80B is a one-variable comparison.

## Why

Every non-fall failure of the e5-hub arms on the MuJoCo plant is the writer's
joint-limit guard on ankle pitch (11-16 of 43 motions per run; falls 0-1;
SONIC v1.1 1 of 43). The policy holds a PD target 1.7-2.3 rad past the limit
while the foot is already at its stop; the actuator shoves about 50 Nm into
the stop; the measured joint passes the soft limit by more than the 0.1 rad
guard margin. SONIC also commands targets past the limit but backs off as the
joint arrives (at the stop its target is at most 0.45 rad beyond). Same
motions fail at the same reference frame for every arm and checkpoint
(toe-off moments of exercise_3, jump_around, macarena, lunges, one-leg
jumps), and the anchor error is the same as SONIC's, so this is not the live
anchor estimate. Details: memory `plant-failures-are-ankle-guard`,
classifier `logs/e5_hardware_gap_ec/classify_faults.py`.

## Arms

| arm  | added override                               | what it charges |
| ---- | -------------------------------------------- | --------------- |
| c1l1 | `env.rewards.joint_limit.weight=-50.0`        | measured position past the soft limit (existing term, 5x); the "louder, later" control |
| c1l2 | `env.rewards.joint_limit_push.weight=-0.02`   | applied PD torque still pushing an offending joint outward (Nm), only while the joint is past its soft limit; zero inside the range whatever the target |

`joint_limit_push` is new (`mdp/rewards.py`, registered at weight 0.0 in
`G1RewardsCfg`, so every other arm is byte-identical). Both arms carry c1's
overrides verbatim (delay 0-1 sub-step, PD gains 0.9-1.1, anchor-jitter DR,
adaptive reset 0.1 / 200, rate05, energy, frozen normalizer).

## Evaluation

The Isaac evaluator now reports three reference-free joint-limit metrics per
step, also on the success-only row: `joint_limit_excess_rad` (max over joints
of position past the soft limit), `joint_limit_guard_frac` (fraction of steps
any joint is past the soft limit by more than the plant writer's 0.1 rad
guard margin) and `joint_limit_push_nm` (the reward term's quantity). The
plant grade keeps counting the guard directly.

Live rows: latest-eval arms `ankl_c1l1_live` / `ankl_c1l2_live`
(`./submit_live_eval.sh`), plant rows through `./ec_grade_echost.sh` into
`logs/ankle_limit_ec/results_echost.tsv` (same columns as the e5 pass).
