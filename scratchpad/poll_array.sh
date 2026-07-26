#!/usr/bin/env bash
AID=5531711
for i in $(seq 1 40); do
  # summary of array task states
  STATES=$(ssh -o BatchMode=yes ice "sacct -j $AID --format=JobID,State -n -P 2>/dev/null | grep -E '^${AID}_[0-9]+\|' | cut -d'|' -f2 | sort | uniq -c | tr '\n' ' '")
  echo "[$i] tasks: $STATES"
  # cell 0 detail
  C0=$(ssh -o BatchMode=yes ice "sacct -j ${AID}_0 --format=State,Elapsed -n -P 2>/dev/null | head -1")
  echo "     cell0: $C0"
  # any cell completed or failed?
  DONE=$(echo "$STATES" | grep -oE '[0-9]+ (COMPLETED|FAILED)' | head -1)
  if [ -n "$DONE" ]; then
    echo "=== a task reached terminal: $DONE — checking cell0 outputs + log ==="
    ssh -o BatchMode=yes ice "find /home/hice1/fwu91/scratch/Research/IsaacLab/isaaclab/logs/interface_baselines/lafan1_planner_capacity_20260723/scaling/seed0/tiny -path '*eval_*_10starts/summary.json' 2>/dev/null | sed 's#.*/scaling/##'"
    L=$(ssh -o BatchMode=yes ice "find \$HOME/scratch/Research/IsaacLab -name '*${AID}_0*.log' 2>/dev/null | head -1")
    ssh -o BatchMode=yes ice "grep -nE 'Error|Traceback|\[PASS\]|Completed .* seed|MPJPE|missing artifact' $L 2>/dev/null | tail -12"
    # only exit once a FULL array picture (all terminal) OR cell0 terminal
    C0S=$(echo "$C0" | cut -d'|' -f1)
    case "$C0S" in COMPLETED|FAILED) exit 0;; esac
  fi
  sleep 180
done
echo "poll cap; tasks: $STATES"
