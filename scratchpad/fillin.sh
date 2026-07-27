#!/usr/bin/env bash
set -uo pipefail
cd /mnt/hsstorage/fwu91/Projects/SL/IsaacLab-Imitation
SP=/tmp/claude-1169732/-mnt-hsstorage-fwu91-Projects-SL-IsaacLab-Imitation/8340342d-5101-4ea9-8e2f-402c72185d28/scratchpad
for spec in "medium 5" "large 3" "large 5"; do
  set -- $spec; size=$1; seed=$2
  DEMO_ONLY=1 INTERFACES="full_body_trajectory" \
  MODEL_SIZE=$size PLANNER_SEED=$seed \
  STUDY_ROOT=logs/interface_baselines/lafan1_planner_capacity_20260723_REBUILD \
  ORACLE_ROOT=logs/interface_baselines/lafan1_fillin/oracle_baselines \
  RENDER_VIDEO=0 \
    experiments/campaigns/2026-07-23-lafan1-planner-capacity/run_capacity_point.sh \
    > "$SP/fillin_${size}_s${seed}.log" 2>&1
  echo "  ${size}/seed${seed} exit=$?"
done
