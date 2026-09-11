Monitor the SONIC-sized combo H200 campaign every hour and return a concise
status report to this chat on EVERY run, including when jobs remain queued.
Run from /home/fwu91/Documents/SL/IsaacLab-Imitation. Use the cluster-job-submission
skill and the repository control plane. Current replacement submission:
/home/fwu91/Documents/SL/IsaacLab-Imitation/logs/cluster_control/combo-sonic-mlp-50b-memr1/combo-sonic-mlp-50b-memr-sonic_mlp-s0-20260908-204108-3b5b9975

Check status, relevant active/failed job logs, cumulative frame counts and
latest checkpoint progress. Report timestamp, active job and stage, scheduler
state, frames, available loss/teacher-rate metrics, throughput and defensible
ETA. Keep DAgger progress (400097280 frames) separate from PPO progress
(50000166912 frames). Do not infer progress merely from an existing filename
or a queued dependent job. Flag OOMs, failures, stalled progress and impossible
dependencies. Preserve the 200M teacher-only,200–400M exponential handoff,
supervised-through400M, then full50B PPO protocol.

Append reports to logs/combo_actor_audit_20260908/hourly_monitor.md. Monitor
only: do not cancel, resubmit, change budgets/configs, or submit evaluations.
Stop the schedule after verified full 50B completion and provide a final
checkpoint/report location. On failure continue hourly reporting unless the
user stops the schedule. If a newer repair replaces this chain, verify the
campaign record and follow that replacement instead.
