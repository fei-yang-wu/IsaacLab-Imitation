#!/usr/bin/env bash
A=5533378
B='$HOME/scratch/Research/IsaacLab/isaaclab/logs/interface_baselines/lafan1_planner_capacity_20260723'
for i in $(seq 1 70); do
  S=$(timeout 90 ssh -o BatchMode=yes ice "sacct -j $A --format=State%12 -n -P 2>/dev/null | grep -vE 'batch|extern' | cut -d'|' -f1 | sort | uniq -c | tr '\n' ' '")
  R=$(timeout 90 ssh -o BatchMode=yes ice "ls -d $B/scaling/seed*/*/full_body_trajectory/planner_rollout_collection/rollout_training_samples/sample_step_000000.pt 2>/dev/null | wc -l")
  E=$(timeout 90 ssh -o BatchMode=yes ice "ls -d $B/scaling/seed*/*/full_body_trajectory/eval_*_10starts/summary.json 2>/dev/null | wc -l")
  echo "[$i] $S| rollout-samples=$R/12  FB-evals=$E/24"
  Q=$(timeout 60 ssh -o BatchMode=yes ice "squeue -u \$USER -h | wc -l")
  [ "$Q" = "0" ] && { echo "=== queue empty ==="; exit 0; }
  sleep 300
done
