#!/usr/bin/env bash
A=5533378
for i in $(seq 1 60); do
  S=$(timeout 90 ssh -o BatchMode=yes ice "sacct -j $A --format=JobID%18,State%12 -n -P 2>/dev/null | grep -vE '\.batch|\.extern'")
  DONE=$(echo "$S" | grep -c COMPLETED); FAIL=$(echo "$S" | grep -c FAILED)
  ROWS=$(timeout 90 ssh -o BatchMode=yes ice "ls -d \$HOME/scratch/Research/IsaacLab/isaaclab/logs/interface_baselines/lafan1_planner_capacity_20260723/scaling/seed*/*/full_body_trajectory/planner_rollout_collection/rollout_training_samples/sample_step_000000.pt 2>/dev/null | wc -l")
  EV=$(timeout 90 ssh -o BatchMode=yes ice "ls -d \$HOME/scratch/Research/IsaacLab/isaaclab/logs/interface_baselines/lafan1_planner_capacity_20260723/scaling/seed*/*/full_body_trajectory/eval_*_10starts/summary.json 2>/dev/null | wc -l")
  echo "[$i] completed=$DONE failed=$FAIL | FB rollout-sample files=$ROWS/12 | FB evals=$EV/24"
  if [ "$FAIL" -gt 0 ]; then
    T=$(echo "$S" | grep FAILED | head -1 | cut -d'|' -f1 | tr -d ' ')
    echo "=== FAILED $T ==="
    timeout 90 ssh -o BatchMode=yes ice "L=\$(ls -t \$HOME/scratch/Research/IsaacLab/isaaclab_2026072*/logs/slurm/*${T}.log 2>/dev/null | head -1); echo \$L; grep -nE 'RuntimeError|ValueError|missing artifact' \$L | tail -3"
    exit 1
  fi
  Q=$(timeout 60 ssh -o BatchMode=yes ice "squeue -u \$USER -h | wc -l")
  [ "$Q" = "0" ] && { echo "=== all cells finished ==="; exit 0; }
  sleep 240
done
