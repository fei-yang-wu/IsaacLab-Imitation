#!/usr/bin/env bash
JOB=5529476
for i in $(seq 1 30); do
  read -r STATE NODE < <(ssh -o BatchMode=yes ice "sacct -j $JOB --format=State,NodeList -n -P 2>/dev/null | head -1" | tr '|' ' ')
  ELAPSED=$(ssh -o BatchMode=yes ice "sacct -j $JOB --format=Elapsed -n -P 2>/dev/null | head -1")
  echo "[$i] state=$STATE node=$NODE elapsed=$ELAPSED"
  case "$STATE" in
    FAILED|COMPLETED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY)
      echo "=== TERMINAL: $STATE ==="
      L=$(ssh -o BatchMode=yes ice "find \$HOME/scratch/Research/IsaacLab -name '*${JOB}*.log' 2>/dev/null | head -1")
      ssh -o BatchMode=yes ice "tail -35 $L 2>/dev/null"
      exit 0 ;;
    RUNNING)
      # once it's been running a while, Isaac init likely passed -> peek log then keep going a bit
      SECS=$(echo "$ELAPSED" | awk -F: '{n=NF; s=$n; if(n>1)s+=$(n-1)*60; if(n>2)s+=$(n-2)*3600; print s}')
      if [ "${SECS:-0}" -gt 780 ]; then
        echo "=== RUNNING >13min: peeking log for Isaac progress ==="
        L=$(ssh -o BatchMode=yes ice "find \$HOME/scratch/Research/IsaacLab -name '*${JOB}*.log' 2>/dev/null | head -1")
        ssh -o BatchMode=yes ice "tail -20 $L 2>/dev/null"
        exit 0
      fi ;;
  esac
  sleep 90
done
echo "=== poll cap reached; job still $STATE ==="
