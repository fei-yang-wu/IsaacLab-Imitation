#!/usr/bin/env bash
B='$HOME/scratch/Research/IsaacLab/isaaclab/logs/interface_baselines/lafan1_planner_capacity_20260723'
for i in $(seq 1 50); do
  C=$(timeout 90 ssh -o BatchMode=yes ice "ls -d $B/finetune_method_b/seed*/*/*/oracle_aggregation/rollout_training_samples/sample_step_000000.pt 2>/dev/null | wc -l")
  E=$(timeout 90 ssh -o BatchMode=yes ice "ls -d $B/finetune_method_b/seed*/*/*/eval_finetuned_b/summary.json 2>/dev/null | wc -l")
  S=$(timeout 90 ssh -o BatchMode=yes ice "sacct -j 5533558 --format=State%12 -n -P 2>/dev/null | grep -vE 'batch|extern' | cut -d'|' -f1 | sort | uniq -c | tr '\n' ' '")
  echo "[$i] methodB collect=$C/3 eval=$E/3 | $S"
  Q=$(timeout 60 ssh -o BatchMode=yes ice "squeue -u \$USER -h | wc -l")
  [ "$Q" = "0" ] && { echo "=== done: collect=$C/3 eval=$E/3 ==="; exit 0; }
  sleep 240
done
