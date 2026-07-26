#!/usr/bin/env bash
JOB=5529479
for i in $(seq 1 30); do
  STATE=$(ssh -o BatchMode=yes ice "sacct -j $JOB --format=State -n -P 2>/dev/null | head -1")
  ELAPSED=$(ssh -o BatchMode=yes ice "sacct -j $JOB --format=Elapsed -n -P 2>/dev/null | head -1")
  echo "[$i] state=$STATE elapsed=$ELAPSED"
  case "$STATE" in
    FAILED|COMPLETED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY)
      echo "=== TERMINAL: $STATE ==="
      L=$(ssh -o BatchMode=yes ice "find \$HOME/scratch/Research/IsaacLab -name '*${JOB}*.log' 2>/dev/null | head -1")
      ssh -o BatchMode=yes ice "grep -nE 'Error|Traceback|No module|undefined symbol|Exception|\[PASS\]|summary.json|Isaac Sim|Simulation App|Startup' $L 2>/dev/null | grep -viE 'device handle|No devices|No running' | tail -25"
      exit 0 ;;
    RUNNING)
      SECS=$(echo "$ELAPSED" | awk -F: '{n=NF;s=$n; if(n>1)s+=$(n-1)*60; if(n>2)s+=$(n-2)*3600; print s}')
      if [ "${SECS:-0}" -gt 900 ]; then
        echo "=== RUNNING >15min: Isaac init likely OK, peeking ==="
        L=$(ssh -o BatchMode=yes ice "find \$HOME/scratch/Research/IsaacLab -name '*${JOB}*.log' 2>/dev/null | head -1")
        ssh -o BatchMode=yes ice "tail -15 $L 2>/dev/null"
        exit 0
      fi ;;
  esac
  sleep 90
done
echo "=== poll cap; still $STATE ==="
