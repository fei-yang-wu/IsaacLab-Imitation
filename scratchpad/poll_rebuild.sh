#!/usr/bin/env bash
A=5533358
for i in $(seq 1 60); do
  S=$(timeout 90 ssh -o BatchMode=yes ice "sacct -j $A --format=State%12 -n -P 2>/dev/null | grep -vE 'batch|extern' | sort | uniq -c | tr '\n' ' '")
  Q=$(timeout 60 ssh -o BatchMode=yes ice "squeue -u \$USER -h 2>/dev/null | wc -l")
  echo "[$i] $S | queued/running: $Q"
  # surface an early failure with its cause rather than waiting silently
  F=$(timeout 90 ssh -o BatchMode=yes ice "sacct -j $A --format=JobID%20,State%12 -n -P 2>/dev/null | grep -vE '\.batch|\.extern' | grep FAILED | head -1 | cut -d'|' -f1")
  if [ -n "$F" ]; then
    echo "=== FAILED: $F ==="
    L=$(timeout 90 ssh -o BatchMode=yes ice "ls -t \$HOME/scratch/Research/IsaacLab/isaaclab_2026072*/logs/slurm/*${F##*_}*.log 2>/dev/null | head -1")
    timeout 90 ssh -o BatchMode=yes ice "grep -nE 'RuntimeError|ValueError|Traceback|missing artifact' $L 2>/dev/null | tail -4"
    exit 1
  fi
  [ "$Q" = "0" ] && { echo "=== all cells done ==="; exit 0; }
  sleep 240
done
