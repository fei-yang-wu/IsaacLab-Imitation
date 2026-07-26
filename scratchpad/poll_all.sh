#!/usr/bin/env bash
B='$HOME/scratch/Research/IsaacLab/isaaclab/logs/interface_baselines/lafan1_planner_capacity_20260723'
for i in $(seq 1 60); do
  E=$(timeout 90 ssh -o BatchMode=yes ice "ls -d $B/scaling/seed*/*/full_body_trajectory/eval_*_10starts/summary.json 2>/dev/null | wc -l")
  MB=$(timeout 90 ssh -o BatchMode=yes ice "ls -d $B/finetune_method_b/seed*/*/*/eval_finetuned_b/summary.json 2>/dev/null | wc -l")
  MBC=$(timeout 90 ssh -o BatchMode=yes ice "ls -d $B/finetune_method_b/seed*/*/*/oracle_aggregation/rollout_training_samples/sample_step_000000.pt 2>/dev/null | wc -l")
  ST=$(timeout 90 ssh -o BatchMode=yes ice "sacct -j 5533471 --format=State%12 -n -P 2>/dev/null | grep -vE 'batch|extern' | cut -d'|' -f1 | sort | uniq -c | tr '\n' ' '")
  echo "[$i] rebuild FB-evals=$E/24 | methodB: collect=$MBC/3 eval=$MB/3 | $ST"
  Q=$(timeout 60 ssh -o BatchMode=yes ice "squeue -u \$USER -h | wc -l")
  if [ "$Q" = "0" ]; then echo "=== queue empty: FB-evals=$E/24 methodB-eval=$MB/3 ==="; exit 0; fi
  sleep 300
done
