# 2026-09-06 -- the explicit end-effector interface on the `combo-50b` hub

The explicit interface (the reference's current frame published to the actor
directly, no encoder, no code) in its end-effector form, the command family
VLA systems emit, trained under the `combo-50b` recipe so it reads against
the hub at matched cumulative frames.

## Control

`combo` in `2026-09-03-combo-50b` (W&B `c50b-combo-s0`, project
`g1-bs-pareto`): the stacked 64-D recipe as a 50B chain. This campaign is
that yaml with one block swapped, the actor command. Everything else is
verbatim: release-size MLP actor with the ten-step proprio history,
full-batch / 3-epoch entry point, `optim.weight_decay=1e-2`, linear critic
decay to 1e-5, the two-stage sonic reset ramp (0.8 -> 0.5 in segment 1, 0.5
-> 0.2 in segment 2, pinned 0.2 after), termination curriculum 5M-30M,
rewards, 16,384 x 24 frames per batch, seven chained 15:59 segments at the
full cap. Budget (user, 2026-09-06): a 10B chain, three segments (the hub's first two ramp stages plus a slack segment), checkpoints every 500M, read against `combo-50b` at its 10B checkpoint.

## Arms

| arm | actor command | width |
|---|---|---:|
| `ee_explicit` | `ee_pos` (4 bodies x 3) + `ee_ori` (4 bodies x rot6d) + `root_pos` (3) + `root_ori` (6) | 45 |
| `root_qpos_explicit` | `joint_qpos` (29) + `root_pos` (3) + `root_ori` (6) | 38 |
| `combo` (control, not here) | 64-D skill code + sin/cos phase | 66 |

EE bodies: `left_ankle_roll_link`, `right_ankle_roll_link`,
`left_wrist_yaw_link`, `right_wrist_yaw_link`, expressed in the
`torso_link` frame (`G1_EE_BODY_NAMES`, `G1_OBS_ANCHOR_BODY_NAME`). The
root terms are the pelvis pose in the same frame. Every command is the
current reference frame (window 0/0), the contract the explicit trackers
have always trained on.

Both arms keep the critic on the full-body trio
(`env.command_interface.reference.critic_components=[joint_qpos_qvel,root_pos,root_ori]`),
which is what the latent hub's critic reads by default, so only the ACTOR's
command moves. `root_qpos_explicit` re-does the 2026-08-05 explicit row
(`root_qpos_explicit`, 0.9358 / 19.21 mm at 7.6B on the old recipe) under
this recipe, so EE-vs-qpos is one variable inside the campaign and each arm
is one variable against `combo`.

Interface overrides (per arm `components` / `command_space`):

```
env.command_interface.actor=explicit
env.command_interface.actor.components=[ee_pos,ee_ori,root_pos,root_ori]
env.command_interface.reference.critic_components=[joint_qpos_qvel,root_pos,root_ori]
agent.ipmd.use_latent_command=false
agent.command_space=ee
agent.command_components=[ee_pos,ee_ori,root_pos,root_ori]
agent.ipmd.command_source=random
```

No `hl_skill_*` or `latent_learning` lines. The `expert_macro_state_terms`
data override stays so the data plane is byte-identical to the hub's.

## Read

Score at the 500M checkpoints on `bones_testbed4096_v1` clean (the
star-v2 evaluator arguments) against `combo` at the same frame count; the
existing `combo` rows in `latest_eval` cover 9B-16B. Explicit arms evaluate
without `--skill_encoder_source` and with the same interface overrides.

## Run

```bash
./experiments/campaigns/2026-09-06-ee-explicit-10b/submit.sh ee_explicit 0
./experiments/campaigns/2026-09-06-ee-explicit-10b/submit.sh root_qpos_explicit 0
# then each printed `submit --confirm <PLAN_SHA>` line
```

Outputs: `/storage/ice-shared/vip-vwt/scratch-fwu91/ee_explicit_10b/<arm>_seed0/tracker`.
W&B project `g1-bs-pareto` (with the hub), group `ee-explicit-10b`.

## Live check and final evaluation (2026-09-07 17:11 UTC)

Root-qpos explicit completed 10,000,269,312 frames: segment 1 `5714553`
timed out, segment 2 `5714554` completed the budget, and segment 3 `5714555`
completed its no-op resume. The final checkpoint is
`/storage/ice-shared/vip-vwt/scratch-fwu91/ee_explicit_10b/root_qpos_explicit_seed0/tracker/2026-09-07_09-20-27_wandb-ee10b-rootqpos-s0-c5454a/models/model_step_10000269312.pt`.
The log explicitly records `frames=10000269312/10000269312`.

EE explicit segment 2 `5714551` is running; its latest sampled log records
9,960,554,496 frames. Segment 1 `5714550` timed out and continuation
`5714552` waits on segment 2. No final EE result is claimed.

Evaluation `5721164` is submitted and pending resources. `eval_campaign.yaml`
scores the pinned root-qpos final checkpoint on `bones_testbed4096_v1`,
clean (`randomization=none`), seed 0, deterministic actions, sequential
frame-0 starts, 10,000-step cap, and SONIC termination thresholds.
It preserves the full-batch agent entrypoint, explicit actor/critic components,
ten-frame actor history, reward weights, and network sizes. One H200,
16 CPUs, 160G RAM, two-hour limit; all nine preflight checks passed.
SR and success-only micro MPJPE-L/G must be reported together after checking
full completion; this remains a single-seed result. Smoothness fields are
collected by the existing evaluator.

Output directory: `/storage/ice-shared/vip-vwt/scratch-fwu91/ee_explicit_10b_eval`.
Plan SHA: `92efeced27a6b75db0225d15a1e72abcb19bbb317a898168b9be95c4b534785c`.
Submitted workspace SHA: `6d86e416fda619c0b4a6e347c1c7cfe731bc4a6db616e1048b73b534a86db616`.
Training campaign files were recovered from the original submitted workspace
`84a8b5b698a711f976a35fbc2188192ae5d46a0ff5dd8e214db06dc1b5ba0a9d`.

## Verified final root-qpos result (2026-09-07)

Evaluation `5721164` completed successfully in 7m53s. At 10,000,269,312
training frames, seed 0, canonical clean 4,096-motion board:
**SR 0.9563 (3,917/4,096), success-only micro MPJPE-L 19.75 mm,
MPJPE-G 303.80 mm**. Canonical summarizer smoothness statistics:
body jerk 265.9 m/s³ and action delta L2 1.056; tracking acceleration
distance 5.79 m/s². These are single-seed measurements.

Validation passed: exact ordered TESTBED4096_RANKS, done_rate=1,
time_out_rate=0, all_envs_done at 1,457 steps, deterministic mode actions,
and no startup/reset/push randomization. Values use the canonical
summarize_paper_boards per-episode reduction. Raw artifact:
`logs/ee_explicit_10b_eval/root_qpos_explicit_seed0_clean_f10000269312.json`
(local mirror; original remains in the remote output directory above).

EE explicit also reached 10,000,269,312 frames; `5714551` and no-op resume
`5714552` are COMPLETED. No EE evaluation was submitted in this task.

## EE final evaluation submitted (2026-09-07 17:55 UTC)

Submitted ICE job `5721417` for EE explicit at 10,000,269,312 frames.
Checkpoint: `/storage/ice-shared/vip-vwt/scratch-fwu91/ee_explicit_10b/ee_explicit_seed0/tracker/2026-09-07_09-19-49_wandb-ee10b-ee-s0-b22942/models/model_step_10000269312.pt`.
The `ee_explicit` arm in `eval_campaign.yaml` uses the same canonical clean
4,096-motion protocol as root-qpos, with the trained EE components
`[ee_pos,ee_ori,root_pos,root_ori]` and `command_space=ee` on actor and agent.
All nine preflight checks passed; frozen-plan EE component checks passed.
Resources: one H200, 16 CPUs, 160G RAM, two-hour limit.
Plan SHA: `3aae025d17169857f997219d2092ba5351f2e9812a3b2f9de43ac11abf9a558b`.
Workspace SHA: `9096dfb8c290454dc1cb228124abfa2d1ed879b8b9f8297f86b07c65659a8b0a`.
Expected output: `/storage/ice-shared/vip-vwt/scratch-fwu91/ee_explicit_10b_eval/ee_explicit_seed0_clean_f10000269312.json`.
No result is claimed until completion and metric validation.

## Matched 2B evaluation submitted (2026-09-07 17:57 UTC)

Both arms retain `model_step_2000289792.pt`: 2,000,289,792 frames,
seed 0. Submitted root-qpos job `5721421` and EE job `5721423` via
`root_qpos_explicit_2b` and `ee_explicit_2b` in `eval_campaign.yaml`.
Both use the identical clean 4,096-motion protocol and resources as the
10B evaluations. Only each arm's trained interface and checkpoint differ.
All preflight and frozen-plan interface checks passed. Each pinned tree
under `/storage/ice-shared/vip-vwt/scratch-fwu91/ee_explicit_2b_eval_pinned`
contains only its 2B checkpoint, preventing selection of a later milestone.
Outputs share the evaluation directory above, with filenames
`<arm>_seed0_clean_f2000289792.json`. Results are pending validation.

Root-qpos plan SHA: `b73753436466981debeb99779ddbf9c0d5de7455c5379329bca93495901894ff`.
EE plan SHA: `d09e789a111303cfff25b92120aaab320c37266cc2162e8b7f56162b031e508d`.
Shared submitted workspace SHA: `8b11a276eb6278f3fc190776d116b995a4097f0dfdc056f931e004290def3c15`.

## Completed 2B/10B evaluations (verified 2026-09-08 UTC)

All four jobs completed; canonical ranks, full episode completion, zero
timeouts and clean deterministic settings validated. Single seed;
success-only micro MPJPE-L/G from the canonical summarizer:

| Arm | Frames (approx.) | SR | MPJPE-L mm | MPJPE-G mm |
| --- | ---: | ---: | ---: | ---: |
| root_qpos_explicit | 2B | 0.9070 | 23.65 | 399.84 |
| ee_explicit | 2B | 0.8762 | 34.85 | 362.03 |
| root_qpos_explicit | 10B | 0.9563 | 19.75 | 303.80 |
| ee_explicit | 10B | 0.9407 | 27.85 | 173.65 |

Raw mirrors: `logs/ee_explicit_10b_eval/`. Jobs: root-qpos 2B `5721421`,
EE 2B `5721423`, root-qpos 10B `5721164`, EE 10B `5721417`.
