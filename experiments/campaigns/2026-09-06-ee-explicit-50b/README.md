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
full 50B cap, 1B checkpoints.

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

Score at the hub's 1B checkpoints on `bones_testbed4096_v1` clean (the
star-v2 evaluator arguments) against `combo` at the same frame count; the
existing `combo` rows in `latest_eval` cover 9B-16B. Explicit arms evaluate
without `--skill_encoder_source` and with the same interface overrides.

## Run

```bash
./experiments/campaigns/2026-09-06-ee-explicit-50b/submit.sh ee_explicit 0
./experiments/campaigns/2026-09-06-ee-explicit-50b/submit.sh root_qpos_explicit 0
# then each printed `submit --confirm <PLAN_SHA>` line
```

Outputs: `/storage/ice-shared/vip-vwt/scratch-fwu91/ee_explicit_50b/<arm>_seed0/tracker`.
W&B project `g1-bs-pareto` (with the hub), group `ee-explicit-50b`.
