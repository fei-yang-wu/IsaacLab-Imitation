#!/usr/bin/env bash
# Wait until the local GPU is free (glove-hot3d job done), then run the oracle
# baselines locally as the validation step. Polls every 5 min, up to ~10h.
cd /mnt/hsstorage/fwu91/Projects/SL/IsaacLab-Imitation
free_streak=0
for i in $(seq 1 120); do
  read -r USED UTIL < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ',')
  echo "[$(date +%H:%M)] check $i: mem_used=${USED}MiB util=${UTIL}%"
  if [ "${USED:-99999}" -lt 8000 ] && [ "${UTIL:-100}" -lt 30 ]; then
    free_streak=$((free_streak+1))
  else
    free_streak=0
  fi
  if [ "$free_streak" -ge 2 ]; then
    echo "=== GPU FREE (2 consecutive) -> launching oracle baselines locally ==="
    DRY_RUN=0 bash experiments/campaigns/2026-07-23-lafan1-planner-capacity/prepare_oracle_baselines.sh \
      > scratchpad/local_oracle_prepare.log 2>&1
    echo "=== prepare_oracle_baselines exit=$? (see scratchpad/local_oracle_prepare.log) ==="
    echo "--- oracle summaries present? ---"
    find logs/interface_baselines/lafan1_planner_capacity_20260723/oracle_baselines -name summary.json 2>/dev/null
    exit 0
  fi
  sleep 300
done
echo "=== GPU still busy after ~10h; not launched ==="
