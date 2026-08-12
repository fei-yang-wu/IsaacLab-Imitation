# BONES-129k FSQ action-rate continuation

This campaign resumes the latest checkpoint from the combined FSQ tracker
(`2026-08-08-bones129k-fsq-anchor-critic`, ICE job `5573503`). The source job
reached `5,750,390,784` environment frames before its 16-hour ICE walltime.

The continuation keeps the same FSQ64 command, stride-1 expert-heading macro
window, scaled tracker, reference-only critic channel, 16,384 environments,
24-step rollout, 294,912 minibatch, gamma `0.97`, and 250M checkpoint cadence.
It changes only:

```text
env.rewards.action_rate_l2.weight: 0.0 -> -0.1
```

The requested segment has a 10B-frame cap. ICE walltime may stop one segment
before that cap; checkpoints persist under `/data`.

## Run

```bash
MODE=print experiments/campaigns/2026-08-10-bones129k-fsq-action-rate-continuation/run.sh
MODE=validate experiments/campaigns/2026-08-10-bones129k-fsq-action-rate-continuation/run.sh
MODE=submit CONFIRM_SUBMIT=fsq-action-rate-continuation \
  experiments/campaigns/2026-08-10-bones129k-fsq-action-rate-continuation/run.sh
```
