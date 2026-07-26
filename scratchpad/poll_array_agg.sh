#!/usr/bin/env bash
ARR=5531880; AGG=5531891
for i in $(seq 1 60); do
  S=$(ssh -o BatchMode=yes ice "sacct -j $ARR --format=JobID,State -n -P 2>/dev/null | grep -E '^${ARR}_[0-9]+\|' | cut -d'|' -f2 | sort | uniq -c | tr '\n' ' '")
  AG=$(ssh -o BatchMode=yes ice "sacct -j $AGG --format=State -n -P 2>/dev/null | head -1")
  echo "[$i] array: $S | agg: $AG"
  NONTERM=$(echo "$S" | grep -oE '[0-9]+ (PENDING|RUNNING)' | head -1)
  AGS=$(echo "$AG" | tr -d ' ')
  case "$AGS" in
    COMPLETED) echo "=== AGGREGATE COMPLETED ==="; exit 0;;
    FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY) echo "=== AGGREGATE $AGS ==="; exit 2;;
  esac
  if [ -z "$NONTERM" ] && [ -n "$S" ]; then
    echo "=== array all terminal; agg not yet done ($AG) ==="
  fi
  sleep 300
done
echo "poll cap; array: $S agg: $AG"
