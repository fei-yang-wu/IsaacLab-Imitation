# BONES-129k critic-command ablation

This is a one-variable comparison against W&B run `r09s1pc7` / ICE job
`5567801`. The actor keeps the frozen root-qpos DiffSR z256 plus phase command.
The baseline critic reads both the 258-D actor latent and the noise-free
reference channel; this ablation sets
`env.command_interface.critic_channels=[reference]`, removing only the actor
latent from the critic.

To isolate that change, submission reuses the baseline's immutable workspace
archive, SHA-256
`e20e93be390a9985df0472893f20ce2b68050dd12a89366743a9dfc66f951d05`.
All environment, encoder, PPO, reset-sampling, reward, and network settings are
otherwise copied from the baseline. The run targets 5B frames on one H200 in
`coe-gpu` and logs to `g1-bones-seed/bones129k-ablation`.

```bash
MODE=print experiments/campaigns/2026-08-06-bones129k-critic-no-latent/run.sh
MODE=validate experiments/campaigns/2026-08-06-bones129k-critic-no-latent/run.sh
MODE=submit CONFIRM_SUBMIT=critic-no-latent \
  experiments/campaigns/2026-08-06-bones129k-critic-no-latent/run.sh
```

## Submission

Attempt `5571182` failed in one second before opening output because the manual
immutable-archive path initially omitted the shared-SIF runtime flag. Retry
`5571183` restores the complete `ice_runtime` environment and is running on an
H200 in `coe-gpu`. Its resolved model keeps the actor input at 351 values and
reduces the critic input from 544 to 286 values; the printed critic keys contain
the reference command and privileged state but no `latent_command`.

W&B: [run `mp85ex1f`](https://wandb.ai/feiyangwu-georgia-institute-of-technology/g1-bones-seed/runs/mp85ex1f).
Startup was verified through 20.05M frames at about 44.4k fps with no errors.
