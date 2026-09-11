#!/usr/bin/env bash
# Stack the before/after renders side by side, label the columns, and join
# every clip into one video. Pure ffmpeg; run after render.sh.
#
#   ./compose.sh
#   OUTPUT_ROOT=... LEFT_LABEL="..." RIGHT_LABEL="..." ./compose.sh
set -euo pipefail
CAMPAIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${CAMPAIGN_DIR}" rev-parse --show-toplevel)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/logs/action_rate_side_by_side}"
BEFORE_DIR="${OUTPUT_ROOT}/before/videos"
AFTER_DIR="${OUTPUT_ROOT}/after/videos"
OUT_DIR="${OUTPUT_ROOT}/side_by_side"
LEFT_LABEL="${LEFT_LABEL:-before: combo-50b (action-rate weight -0.03)}"
RIGHT_LABEL="${RIGHT_LABEL:-after: +10B finetune, action-rate weight -0.5}"
FONT="${FONT:-${HOME}/.local/share/fonts/IBMPlexSans/IBMPlexSans-Medium.ttf}"
[ -f "${FONT}" ] || FONT="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
mkdir -p "${OUT_DIR}"

label() {  # $1 text -> drawtext filter with a soft plate behind it
    local text="${1//:/\\:}"
    printf "drawtext=fontfile='%s':text='%s':fontcolor=0x0B2A5B:fontsize=30:box=1:boxcolor=white@0.72:boxborderw=12:x=24:y=24" "${FONT}" "${text}"
}

list=()
shopt -s nullglob
for left in "${BEFORE_DIR}"/rank-*.mp4; do
    name="$(basename "${left}")"
    right="${AFTER_DIR}/${name}"
    [ -f "${right}" ] || { echo "[skip] no after video for ${name}" >&2; continue; }
    out="${OUT_DIR}/${name%.mp4}_side_by_side.mp4"
    motion="${name#rank-*-}"; motion="${motion%.mp4}"
    # The same rank runs the same number of control steps in both columns
    # (terminations are off), so no padding is needed; shortest= guards the
    # odd frame anyway.
    ffmpeg -y -loglevel error -i "${left}" -i "${right}" -filter_complex \
        "[0:v]$(label "${LEFT_LABEL}")[l];[1:v]$(label "${RIGHT_LABEL}")[r];[l][r]hstack=inputs=2:shortest=1,drawtext=fontfile='${FONT}':text='${motion//_/ }':fontcolor=0x4A5B78:fontsize=26:x=(w-text_w)/2:y=h-th-20[v]" \
        -map "[v]" -c:v libx264 -crf 18 -pix_fmt yuv420p -movflags +faststart "${out}"
    echo "wrote ${out}"
    list+=("${out}")
done
[ "${#list[@]}" -gt 0 ] || { echo "[FATAL] nothing composed" >&2; exit 1; }

concat_list="${OUT_DIR}/concat.txt"
: > "${concat_list}"
for f in "${list[@]}"; do printf "file '%s'\n" "${f}" >> "${concat_list}"; done
ffmpeg -y -loglevel error -f concat -safe 0 -i "${concat_list}" -c copy "${OUT_DIR}/all_clips_side_by_side.mp4"
echo "wrote ${OUT_DIR}/all_clips_side_by_side.mp4 (${#list[@]} clips)"
