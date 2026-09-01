#!/usr/bin/env bash
# Figure clips: one rendered episode per selected motion, camera tracking the
# robot, for the teaser and the results row of the paper.
#
#   ./render_clips.sh                 # the four default motions
#   RANKS_TO_RENDER="13 29" ./render_clips.sh
#   ARM=fsq64_10b ./render_clips.sh
#
# One Isaac process per motion, one environment per process, and the recording
# camera follows that robot (`--video_track_env 0`), because the Isaac Lab video
# recorder camera is otherwise static in world space and a walking robot leaves
# the frame. The deployment config is the headline one: consume all 30 slots
# before re-planning, no temporal ensembling, domain randomization off.
set -euo pipefail
CAMPAIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ARM="${ARM:-ln_hold1_10b}"
# Manifest ranks in data/bones_seed_language30_compositionality_v1:
#   0  picking up an object from the ground while walking
#   13 opening a door from inside, walking out, closing it
#   17 lifting a crate and walking forward
#   29 slow walk forward
RANKS_TO_RENDER="${RANKS_TO_RENDER:-0 13 17 29}"
ALL_RANKS="${ALL_RANKS:-$(seq 0 29)}"

# Camera position offset from the robot root, in world axes (m), and the height
# above the root that the camera aims at. Scale CAM_OFFSET for a wider shot.
CAM_OFFSET="${CAM_OFFSET:-2.8 2.8 1.4}"
CAM_LOOK_HEIGHT="${CAM_LOOK_HEIGHT:-0.0}"
CAM_WIDTH="${CAM_WIDTH:-1920}"
CAM_HEIGHT="${CAM_HEIGHT:-1080}"

for rank in ${RANKS_TO_RENDER}; do
    exclude=""
    for other in ${ALL_RANKS}; do
        [ "${other}" = "${rank}" ] || exclude="${exclude} ${other}"
    done
    echo "=== rendering rank ${rank} (${ARM}) ==="
    EXCLUDE_RANKS="${exclude# }" \
    EPISODES_PER_GOAL=1 \
    VIDEO=1 \
    ENSEMBLE="${ENSEMBLE:-none}" \
    FORCE_CONSUME_SLOTS="${FORCE_CONSUME_SLOTS:-30}" \
    MAX_STEPS="${MAX_STEPS:-600}" \
    LABEL_SUFFIX="clip_rank${rank}" \
    "${CAMPAIGN_DIR}/eval.sh" "${ARM}" \
        --video_track_env 0 \
        --video_track_offset ${CAM_OFFSET} \
        --video_track_height "${CAM_LOOK_HEIGHT}" \
        "env.video_recorder.window_width=${CAM_WIDTH}" \
        "env.video_recorder.window_height=${CAM_HEIGHT}"
done
