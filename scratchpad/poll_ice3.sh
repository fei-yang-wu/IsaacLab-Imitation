#!/usr/bin/env bash
JOB=5531679
for i in $(seq 1 40); do
  STATE=$(ssh -o BatchMode=yes ice "sacct -j $JOB --format=State -n -P 2>/dev/null | head -1")
  EL=$(ssh -o BatchMode=yes ice "sacct -j $JOB --format=Elapsed -n -P 2>/dev/null | head -1")
  echo "[$i] state=$STATE elapsed=$EL"
  case "$STATE" in
    FAILED|COMPLETED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY)
      echo "=== TERMINAL: $STATE ==="
      L=$(ssh -o BatchMode=yes ice "find \$HOME/scratch/Research/IsaacLab -name '*${JOB}*.log' 2>/dev/null | head -1")
      echo "LOG=$L"
      ssh -o BatchMode=yes ice "grep -nE 'Error|Traceback|platform|ModuleNotFound|ncclDev|MPJPE|PASS|Oracle baselines prepared|summary.json|assert-kitless|Injected' $L 2>/dev/null | tail -25"
      exit 0 ;;
    RUNNING)
      SECS=$(echo "$EL" | awk -F: '{n=NF;s=$n;if(n>1)s+=$(n-1)*60;if(n>2)s+=$(n-2)*3600;print s}')
      if [ "${SECS:-0}" -gt 1500 ]; then
        echo "=== RUNNING >25min: peeking log ==="
        L=$(ssh -o BatchMode=yes ice "find \$HOME/scratch/Research/IsaacLab -name '*${JOB}*.log' 2>/dev/null | head -1")
        ssh -o BatchMode=yes ice "grep -nE 'MPJPE|summary.json|Rollout steps|Error|Traceback' $L 2>/dev/null | tail -12"
        exit 0
      fi ;;
  esac
  sleep 120
done
echo "poll cap; still $STATE"
