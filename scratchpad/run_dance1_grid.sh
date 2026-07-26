#!/usr/bin/env bash
set -uo pipefail
cd /mnt/hsstorage/fwu91/Projects/SL/IsaacLab-Imitation
for seed in 0 1 2; do
  for size in tiny small medium large; do
    MOTION_NAME=dance1_subject1 \
    MANIFEST=data/lafan1/manifests/g1_lafan1_manifest.json \
    LATENT_DATASET_PATH=data/lafan1/zarr/latent_dance1_subject1_corrected_8e95d557 \
    STUDY_ROOT=logs/interface_baselines/lafan1_planner_capacity_dance1 \
    INTERFACES="latent_skill full_body_trajectory" \
    RENDER_VIDEO=0 DEMO_ONLY=1 MODEL_SIZE=$size PLANNER_SEED=$seed \
      experiments/campaigns/2026-07-23-lafan1-planner-capacity/run_capacity_point.sh \
      > /tmp/claude-1169732/-mnt-hsstorage-fwu91-Projects-SL-IsaacLab-Imitation/8340342d-5101-4ea9-8e2f-402c72185d28/scratchpad/dance1_${size}_s${seed}.log 2>&1
    echo "  seed=$seed size=$size exit=$?"
  done
done
echo "=== dance1 grid done: $(find logs/interface_baselines/lafan1_planner_capacity_dance1/scaling -path '*eval_pretrained_10starts/summary.json' | wc -l)/24 evals ==="
