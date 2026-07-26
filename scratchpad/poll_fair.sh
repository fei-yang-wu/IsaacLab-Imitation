#!/usr/bin/env bash
for i in $(seq 1 30); do
  S=$(timeout 90 ssh -o BatchMode=yes ice "sacct -j 5532443,5532444 --format=JobID,State -n -P 2>/dev/null | grep -vE 'batch|extern' | cut -d'|' -f2 | sort | uniq -c | tr '\n' ' '")
  Q=$(timeout 60 ssh -o BatchMode=yes ice "squeue -u \$USER -h 2>/dev/null | wc -l")
  echo "[$i] states: $S | in-queue: $Q"
  if [ "$Q" = "0" ]; then echo "=== all jobs left the queue ==="; exit 0; fi
  sleep 240
done
