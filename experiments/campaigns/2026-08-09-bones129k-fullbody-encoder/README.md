# BONES-129k — what the skill encoder reads

One encoder pretrain plus one dependent low-level job, one variable against the
running ICE job `5573413` (`expert_heading_critic_no_latent`).

```
env.expert_macro_state_terms
  control: [expert_motion_qpos, expert_anchor_pos_b, expert_anchor_ori_b]   380
  arm:     [expert_motion,      expert_anchor_pos_b, expert_anchor_ori_b]   670
```

`expert_motion` is the `joint_qpos_qvel` component: the same reference joint
positions the control reads, **plus reference joint velocities**. Nothing else
changes — same expert-heading frame, same stride 1, same ten slots, same z256
command held 10 control steps, same `critic_channels=[reference]`, same
tuned-entry-point tracker capacity, same 16,384 x 24 geometry, and the
control's own pretrain geometry (encoder `[1024, 512, 512]` mish with layer
norm, DiffSR feature 128 / embed 512, `[512]` heads, 50,000 updates, batch
8192) passed explicitly.

## Why velocity

The released SONIC checkpoint we benchmark against (SR 0.9937 against our
0.9062) feeds its tokenizer reference `joint_pos(29) + joint_vel(29)` plus a 6D
root-orientation difference per window frame. Our v2 default dropped to
positions only on 2026-08-04. Velocity is therefore the concrete encoder-input
difference between the two recipes.

It is **not** 14-body keypoint positions. That 480-value spec belongs to the
paper's `sonic_bones_seed.yaml` experiment config, not to the released model —
a distinction that matters because the released model is what the 0.9937 row
was measured from.

Stride stays 1 by explicit choice on 2026-08-09, so this arm does not confound
input content with window span.

## Shape safety

Unlike the anchor-mode and stride axes, this axis **does** change the encoder
input width (380 -> 670), so a mispairing fails on shape. The pretrain still
records the anchor mode and stride into the skill checkpoint, and the low level
still checks both on load.

The local smoke asserts the pretrained encoder's first-layer width is 670 and
that the tracker's actor (351) and critic (286) are unchanged — the encoder
reads more, the published command does not grow.

## Running it

```bash
MODE=print ./experiments/campaigns/2026-08-09-bones129k-fullbody-encoder/run.sh
MODE=smoke ./experiments/campaigns/2026-08-09-bones129k-fullbody-encoder/run.sh
MODE=validate LOCAL_SMOKE_ROOT=<smoke-dir> ./experiments/campaigns/2026-08-09-bones129k-fullbody-encoder/run.sh
MODE=submit LOCAL_SMOKE_ROOT=<smoke-dir> CONFIRM_SUBMIT=encoder-input \
  ./experiments/campaigns/2026-08-09-bones129k-fullbody-encoder/run.sh
```

W&B: project `g1-bones-seed`, group `encoder-input`. 10B frame cap, checkpoints
every 50M under persistent `/data`; ICE ends an allocation at 16 h, so a
TIMEOUT is expected and loses no training.
