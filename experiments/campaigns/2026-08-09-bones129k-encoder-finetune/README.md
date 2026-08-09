# BONES-129k — online skill-encoder finetuning

One low-level arm, one variable against the running ICE job `5573413`
(`expert_heading_critic_no_latent`): the frozen DiffSR skill encoder is allowed
to move during controller training.

```
agent.ipmd.hl_skill_finetune_enabled=true
```

Everything else — the expert-heading macro frame, stride 1, z256 command held
10 control steps, `critic_channels=[reference]`, tuned-entry-point tracker
capacity, 16,384 x 24 geometry, rewards, resets, curriculum — is the control's
contract, and the arm starts from the **same encoder file** the control loads,
matched by SHA-256 `be6d533f…`.

## Why this arm

Every latent arm this project has run freezes the encoder after pretraining.
The released SONIC checkpoint does not: its tokenizer sits inside the actor
backbone and takes the PPO gradient natively, next to a reconstruction loss.
That is the largest remaining structural difference between our recipe and the
checkpoint that scores 0.9937 SR against our 0.9062.

Unfreezing was previously unattractive for a specific reason: under the `robot`
anchor convention the frozen encoder was already queried off-distribution at
rollout time, so an online gradient would have been chasing a moving target.
The expert-heading frame removes that mismatch by construction, which is why
this arm sits on top of `5573413` rather than on the older robot-frame recipe.

## What "finetune" means here

The update is not raw policy gradient on the encoder. It is

```
pg_coeff              * second-pass PPO actor objective, gradient through the encoder
+ offline_diffsr_coeff * the original offline DiffSR loss
+ anchor_coeff         * distance to the FROZEN checkpoint encoder's output
```

so the offline objective and the anchor term act as a trust region around the
pretrained encoder. Coefficients are passed explicitly at their current library
defaults — `lr 3e-5`, `pg 0.05`, `diffsr 1.0`, `anchor 0.01`, grad-clip 1.0,
offline batch 8192, one high-level update per PPO minibatch update — so a later
default change cannot move this arm. The loaded DiffSR transition model itself
stays frozen (`hl_skill_train_diffsr=false`): the axis is "the command function
adapts", not "the pretrain restarts online".

Note for downstream work: this arm's controller checkpoint will **not** embed
the pretrain encoder unchanged. Any later planner binding must use this run's
own encoder, not `be6d533f…`.

## The gate

Turning the flag on must actually move the encoder. The local smoke requires
both: at least one logged high-level update, and encoder weights that differ
from the starting checkpoint afterwards. A run that logs zero updates would be
indistinguishable from the frozen control in W&B.

## Running it

```bash
MODE=print ./experiments/campaigns/2026-08-09-bones129k-encoder-finetune/run.sh
MODE=smoke ./experiments/campaigns/2026-08-09-bones129k-encoder-finetune/run.sh
MODE=validate LOCAL_SMOKE_ROOT=<smoke-dir> ./experiments/campaigns/2026-08-09-bones129k-encoder-finetune/run.sh
MODE=submit LOCAL_SMOKE_ROOT=<smoke-dir> CONFIRM_SUBMIT=encoder-finetune \
  ./experiments/campaigns/2026-08-09-bones129k-encoder-finetune/run.sh
```

W&B: project `g1-bones-seed`, group `encoder-finetune`. 10B frame cap,
checkpoints every 50M under persistent `/data`; ICE ends an allocation at 16 h,
so a TIMEOUT is expected and loses no training.
