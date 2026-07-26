#!/usr/bin/env bash
AID=5531806
for i in $(seq 1 60); do
  ST=$(ssh -o BatchMode=yes ice "sacct -j $AID --format=State,Elapsed -n -P 2>/dev/null | head -1")
  echo "[$i] $ST"
  S=$(echo "$ST" | cut -d'|' -f1 | tr -d ' ')
  # peek at progress markers in the log
  L=$(ssh -o BatchMode=yes ice "find \$HOME/scratch/Research/IsaacLab -name '*${AID}*.log' 2>/dev/null | head -1")
  if [ -n "$L" ]; then
    ssh -o BatchMode=yes ice "grep -nE 'failed to parse CPython|Traceback|planner train|\[eval\]|full-horizon|rollout|merge|finetune|\[PASS\]|missing artifact|Saved|MPJPE' $L 2>/dev/null | tail -5"
  fi
  case "$S" in
    COMPLETED) echo "=== CELL0 COMPLETED ==="; exit 0;;
    FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL) echo "=== CELL0 $S ==="; ssh -o BatchMode=yes ice "grep -nE 'Error|Traceback|ValueError|missing artifact' $L 2>/dev/null | tail -15"; exit 1;;
  esac
  sleep 180
done
echo "poll cap"
