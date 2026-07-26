#!/usr/bin/env bash
B='$HOME/scratch/Research/IsaacLab/isaaclab/logs/interface_baselines/lafan1_planner_capacity_20260723'
for i in $(seq 1 40); do
  E=$(timeout 90 ssh -o BatchMode=yes ice "ls -d $B/scaling/seed*/*/full_body_trajectory/eval_*_10starts/summary.json 2>/dev/null | wc -l")
  Q=$(timeout 60 ssh -o BatchMode=yes ice "squeue -u \$USER -h | wc -l")
  echo "[$i] FB evals=$E/24  queued=$Q"
  if [ "$E" = "24" ]; then echo "=== FB side complete ==="; exit 0; fi
  if [ "$Q" = "0" ]; then echo "=== queue empty, FB evals=$E/24 ==="; exit 0; fi
  sleep 240
done
