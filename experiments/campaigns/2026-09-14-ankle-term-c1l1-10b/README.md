# 2026-09-14 termination fine-tunes off c1l1 (SUBMITTED)

Submitted 2026-09-14 13:50 EDT from a clean worktree of commit c966537 (no
drift), partition coe-gpu: c1l1t1 5780776 / 5780777, c1l1j1 5780779 /
5780780 (finetune1 / finetune2 afterany). Smokes from the committed state
(3-iteration fine-tunes of both arms from the c1l1 79.0B file on a local
reference-array store with the cluster data args and RLOpt a0add23): 0
errors, `joint_near_limit` present with fraction 0.5 in the resolved env.

Two fine-tunes off the c1l1 arm of `2026-09-14-ankle-limit-c1-10b` at its
79,000,240,128 checkpoint, 10B each (cap 89,000,240,128), one added
termination per arm. c1l1 keeps training to 86B without them, so c1l1 vs
c1l1t1 / c1l1j1 at matched frames is a one-variable comparison.

## Arms

| arm    | added override                                          | what ends the episode |
| ------ | ------------------------------------------------------- | --------------------- |
| c1l1t1 | `env.terminations.foot_pos_xyz.params.threshold=0.1`     | pelvis-relative foot position error above 0.1 m (was 0.2) |
| c1l1j1 | `env.terminations.joint_near_limit.params.fraction=0.5`  | a MEASURED joint past its soft limit by half the soft-to-hard gap (ankle pitch: above 0.489 rad against soft 0.454, hard 0.5236); the PD target is free to exceed the limit |

`joint_pos_near_hard_limit` is new (`mdp/terminations.py`), registered in
`G1SonicTerminationsCfg` as `joint_near_limit` with `fraction: None`
(disabled, all False), so every other arm and the evaluation board are
byte-identical. Both arms carry c1's overrides and c1l1's
`joint_limit=-50` verbatim.

## Why

The plant's non-fall failures are measured ankle joints driven into their
stop (memory `plant-failures-are-ankle-guard`). c1l1 (the measured-excess
reward at -50) is the first arm to move the board's `joint_limit_guard_frac`
(0.0011 vs c1's 0.0024) and `joint_limit_push_nm` (0.02 vs 0.06) and sits at
the top of the plant clean range (32-33/43), so the user chose it as the hub.
A termination on the measured excursion ends the episode where the plant
writer would fault; the foot termination tightens the swing-foot bar the
plant rehearsals drag on.

## Evaluation

Board rows `ankt_c1l1t1` / `ankt_c1l1j1` through `./submit_live_eval.sh`
(latest-eval arms `ankt_c1l1t1_live` / `ankt_c1l1j1_live`; the board keeps
the frozen SONIC termination set, foot_pos_xyz disabled and joint_near_limit
disabled, so SR stays comparable); plant rows through `./ec_grade_echost.sh`
into `logs/ankle_term_ec/results_echost.tsv`.
