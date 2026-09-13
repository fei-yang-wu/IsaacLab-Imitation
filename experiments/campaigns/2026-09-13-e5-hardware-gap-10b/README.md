# 2026-09-13 -- hardware-gap fine-tunes off the e5 hub (PREPARED, NOT SUBMITTED)

Hub: `e5` (rate05 + `energy_consumption` -1e-4) at 70,000,312,320 frames,
`/data/energy_torque_10b/e5_seed0/tracker/2026-09-11_14-52-15_wandb-smooth-e5-s0-58c8d0/models/model_step_70000312320.pt`.
Each arm continues it for exactly 10B (cap 80,000,312,320), one variable, in
two chained 15:59 segments. W&B project `g1-bs-finetune`, group `e5-hardware-gap-10b`.

| arm | one variable | override |
|---|---|---|
| d1 | actuation delay 0-1 sub-steps (0-5 ms) per env, resampled at reset | `env.actions.joint_pos.delay_substeps_max=1` |
| d2 | actuation delay 0-2 sub-steps (0-10 ms) | `env.actions.joint_pos.delay_substeps_max=2` |
| g1 | PD gain scale 0.9-1.1 per env, fixed for the run | `env.events.randomize_actuator_gains.params.{stiffness,damping}_distribution_params=[0.9,1.1]` |
| a1 | anchor-jitter DR on the live window anchor (odometry envelope) | `env.expert_macro_anchor_walk_std=0.002 …_walk_max=0.1 …_jitter=0.005 …_jump_prob=0.02 …_jump=0.03` |
| r1 | adaptive reset at SONIC's values | `…selection.adaptive_uniform_ratio=0.1 …adaptive_failure_rate_max_over_mean=200` |
| h1 | expert_heading window (anchor-blind) | `env.expert_macro_anchor_mode=expert_heading` |
| c1 | d1 + g1 + a1 + r1 | the four lists spelled out |

Every arm carries e5's own arguments verbatim otherwise, plus
`agent.ppo.update_normalizers_after_rollout=false`, which is real since RLOpt
298f172 (e5 itself trained with a live normalizer; see the correction in
`2026-09-10-energy-torque-10b/README.md`).

Mechanisms added for this campaign (top-level commit of 2026-09-13):

- `EMAJointPositionActionCfg.delay_substeps_{min,max}`: for the first `d`
  physics sub-steps of a control step the PD loop tracks the previous target,
  as the robot's 500 Hz writer does while the new 50 Hz target is computed
  and put on the wire (3-5 ms measured on hardware, 2026-09-13).
- `G1SonicEventCfg.randomize_actuator_gains`: identity (1.0, 1.0) by default,
  startup mode, so the frozen protocol is unchanged until a campaign widens it.
- `expert_macro_anchor_{walk_std,walk_max,jitter,jump_prob,jump}` on the env
  cfg: applied to the LIVE window anchor in `robot` / `robot_heading` modes
  only; the evaluator zeroes them with the other observation corruption.

Not arms, and why: SONIC's exploration clamp (e5's per-joint std is already
0.043-0.092, the clamp [0.001, 0.5] is a no-op); `agent.sonic_release_optimizer=true`
(inert through Hydra: applied in `__post_init__` before overrides land);
trained-in EMA (the bundle / EC decode do not apply it yet).

Validated: offline plans for d1 / c1 / h1 resolve the intended overrides;
two-iteration local smokes of the c1 and h1 override sets on Newton pass.

Eval: `2026-09-02-latest-eval` carries `e5gap_<arm>_live` arms (tree root
`/data/e5_hardware_gap_10b_live`; the h1 scorer builds the expert_heading
window through `vars.anchor_mode`); `submit_live_eval.sh` and
`ec_grade_live.sh` here mirror the energy campaign's. The EC grade runs the
CUDA provider (default) with the live anchor; for h1 it passes
`--anchor expert_heading` because the exported bundle records the encoder
checkpoint's anchor mode, not the tracker's. The a1 / c1 rows want a plant
grade under replayed hardware anchor jitter, which the plant does not
produce on its own (not built yet).

Submit (when asked), per arm:

```bash
python -m imitation_experiments.pipeline.cluster plan --campaign experiments/campaigns/2026-09-13-e5-hardware-gap-10b/campaign.yaml --arm d1 --seed 0
python -m imitation_experiments.pipeline.cluster submit --plan-sha <PLAN_SHA>
```
