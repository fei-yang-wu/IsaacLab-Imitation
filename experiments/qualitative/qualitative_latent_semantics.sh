#!/usr/bin/env bash
set -euo pipefail

# Analysis 7: cluster the latent space, then WATCH each cluster to name it.
#
# Three stages:
#   1 encode   N motions x M windows -> latents.npz            (Isaac)
#   2 cluster  KMeans, pick the motions to show per cluster    (no Isaac)
#   3 gallery  ONE VIDEO PER CLUSTER, its member motions side by side (Isaac)
#
#   DRY_RUN=1 bash experiments/qualitative/qualitative_latent_semantics.sh
#   SMOKE=1   bash experiments/qualitative/qualitative_latent_semantics.sh
#   bash experiments/qualitative/qualitative_latent_semantics.sh
#
# The point is stage 3. Each clip puts MEMBERS_PER_CLUSTER robots in one scene,
# each replaying a different motion from the same cluster, all at once. If the
# cluster means something, robots drawn from different motions visibly do the
# same thing and you name it from what you watched. If they do unrelated things,
# that cluster has no topic -- a result, not a knob to tune.
#
# The robots are NOT tracking here. Every robot is driven straight onto its
# reference pose each step, so you are watching the motion data the encoder
# read, not a controller's imitation of it.
#
# A window is 10 frames = 0.2 s. Shown alone that is a blink, and from a shot
# wide enough to hold eight robots it is a few pixels of change that nobody can
# name. So a clip surrounds the window with CONTEXT_FRAMES of reference either
# side (0.5 s, giving 1.2 s total), holds each frame SLOWDOWN steps, and repeats
# the span LOOPS times. Those context frames are NOT what the latent encodes --
# they are there so the action is recognisable at all, while the CLUSTERING
# still uses the window alone. CONTEXT_FRAMES=0 shows only the encoded window.
# Provenance records `shows_only_encoded_frames` either way.
#
# Common knobs:
#
#   NUM_MOTIONS=4000               motions drawn from the 129,785
#   WINDOWS_PER_MOTION=8           windows per motion, spread over the whole clip
#   K_CLUSTERS=24                  clusters, and therefore videos
#   MEMBERS_PER_CLUSTER=8          motions shown per cluster (one window each)
#   MEMBER_SELECTION=farthest      spread members across the cluster instead of
#                                  the most typical ones (default centroid)
#   MIN_LOCAL_STEP=0               keep the near-identical standing-start
#                                  windows the default (50 = first 1 s) drops
#   MIN_ROOT_SPEED=0               with MIN_LIMB_SPEED, the static-window
#   MIN_LIMB_SPEED=0               gate: keep a window when root speed >=
#                                  MIN_ROOT_SPEED OR root-relative top-5
#                                  body speed >= MIN_LIMB_SPEED; drop only
#                                  windows static on both counts. The OR
#                                  keeps in-place dances/kicks that a single
#                                  whole-body mean would drop while it passed
#                                  slow walking. Defaults 0.4 / 0.6 m/s keep
#                                  roughly the most dynamic 40%; 0 0 keeps
#                                  every window
#   EXCLUDE_MOTION_REGEX=""        keep idle-family clips; the default 'idle'
#                                  removes every window of every motion whose
#                                  name matches (case-insensitive)
#   TSNE_ROWS=0                    skip the standalone tsne_scatter.png
#   LOOPS=2 SLOWDOWN=2             playback: 2 replays, half speed
#   CONTEXT_FRAMES=0               show ONLY the 0.2 s the encoder saw
#   FILMSTRIP_MEMBERS=0            no still strips; 3 most typical members
#                                  otherwise each get a 6-frame close-up
#                                  strip under <gallery>/filmstrips/
#   CLUSTERS=0,3,7                 render only these clusters
#   SKIP_ENCODE=1                  reuse an existing encode run
#   SKIP_CLUSTER=1                 reuse an existing clustering, re-render only
#   SEED=1                         change the motion draw
#   CUDA_VISIBLE_DEVICES=1,3       two render-capable GPUs; video needs both
#   OVERWRITE=1                    replace existing output directories

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/qualitative_env.sh"

# --- mode constants ---------------------------------------------------------
MODE_NAME="latent_semantics"
NUM_MOTIONS="${NUM_MOTIONS:-4000}"
WINDOWS_PER_MOTION="${WINDOWS_PER_MOTION:-8}"
ENCODE_ENVS="${ENCODE_ENVS:-512}"
K_CLUSTERS="${K_CLUSTERS:-24}"
MEMBERS_PER_CLUSTER="${MEMBERS_PER_CLUSTER:-8}"
MEMBER_SELECTION="${MEMBER_SELECTION:-centroid}"
# BONES-SEED clips open in a neutral standing pose, so windows from the first
# second are near-identical across motions and drag every cluster toward one
# shared standing blob. Drop them by default; MIN_LOCAL_STEP=0 keeps them.
MIN_LOCAL_STEP="${MIN_LOCAL_STEP:-50}"
# Static/idle windows carry no dynamics for a cluster to be about: a window
# below BOTH speed gates (root translation, root-relative limb motion) is
# dropped, and idle-family clips are excluded by name entirely.
# MIN_ROOT_SPEED=0 MIN_LIMB_SPEED=0 and EXCLUDE_MOTION_REGEX="" restore the
# unfiltered behaviour.
MIN_ROOT_SPEED="${MIN_ROOT_SPEED:-0.4}"
MIN_LIMB_SPEED="${MIN_LIMB_SPEED:-0.6}"
EXCLUDE_MOTION_REGEX="${EXCLUDE_MOTION_REGEX-idle}"
TSNE_ROWS="${TSNE_ROWS:-6000}"
LOOPS="${LOOPS:-2}"
SLOWDOWN="${SLOWDOWN:-2}"
CONTEXT_FRAMES="${CONTEXT_FRAMES:-25}"
FILMSTRIP_MEMBERS="${FILMSTRIP_MEMBERS:-3}"
FILMSTRIP_FRAMES="${FILMSTRIP_FRAMES:-6}"
FILMSTRIP_PX="${FILMSTRIP_PX:-300}"
CLUSTERS="${CLUSTERS:-}"
LANGUAGE_JSON="${LANGUAGE_JSON:-}"
RANKS="${RANKS:-}"
MOTIONS="${MOTIONS:-}"
VIDEO="${VIDEO:-1}"
ENV_SPACING="${ENV_SPACING:-}"
NJMAX="${NJMAX:-320}"
NCONMAX="${NCONMAX:-40}"
ENCODE_DIR="${ENCODE_DIR:-${OUTPUT_ROOT}/${MODE_NAME}_encode}"
CLUSTER_DIR="${CLUSTER_DIR:-${OUTPUT_ROOT}/${MODE_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${MODE_NAME}_gallery}"
SKIP_ENCODE="${SKIP_ENCODE:-0}"
SKIP_CLUSTER="${SKIP_CLUSTER:-0}"

if [[ "${SMOKE}" == "1" ]]; then
    NUM_MOTIONS=40
    WINDOWS_PER_MOTION=4
    ENCODE_ENVS=32
    K_CLUSTERS=3
    MEMBERS_PER_CLUSTER=4
    LOOPS=2
    ENCODE_DIR="${OUTPUT_ROOT}-smoke/${MODE_NAME}_encode"
    CLUSTER_DIR="${OUTPUT_ROOT}-smoke/${MODE_NAME}"
    OUTPUT_DIR="${OUTPUT_ROOT}-smoke/${MODE_NAME}_gallery"
    OVERWRITE=1
fi

[[ "${VIDEO}" == "1" ]] && qualitative_require_render_gpus
qualitative_check_data
qualitative_check_encoder
# Recorded for provenance and binding only; no policy is stepped in any stage.
qualitative_resolve_policy
ablate_base_overrides

encode_cmd=(
    pixi run -e isaaclab python experiments/qualitative/src/qualitative_latent_semantics.py
    --task "${TASK_NAME}" --headless --device "${DEVICE}"
    --encoder_checkpoint "${ENCODER_CKPT}"
    --policy_checkpoint "${POLICY_CKPT}"
    --reference_arrays_dir "${REFERENCE_ARRAYS_DIR}"
    --output_dir "${ENCODE_DIR}"
    --num_motions "${NUM_MOTIONS}"
    --windows_per_motion "${WINDOWS_PER_MOTION}"
    --num_envs "${ENCODE_ENVS}"
    --seed "${SEED}"
)
[[ -n "${RANKS}" ]] && encode_cmd+=(--ranks "${RANKS}")
[[ -n "${MOTIONS}" ]] && encode_cmd+=(--motions "${MOTIONS}")
[[ "${OVERWRITE}" == "1" ]] && encode_cmd+=(--overwrite)
encode_cmd+=("${BASE_OVERRIDES[@]}")

# Default environment on purpose: stage 2 is sklearn only, and the isaaclab
# environment is expensive to enter for nothing.
cluster_cmd=(
    pixi run python experiments/qualitative/src/qualitative_latent_semantics_cluster.py
    --run_dir "${ENCODE_DIR}"
    --output_dir "${CLUSTER_DIR}"
    --k "${K_CLUSTERS}"
    --members "${MEMBERS_PER_CLUSTER}"
    --member_selection "${MEMBER_SELECTION}"
    --tsne_rows "${TSNE_ROWS}"
    --min_local_step "${MIN_LOCAL_STEP}"
    --min_root_speed "${MIN_ROOT_SPEED}"
    --min_limb_speed "${MIN_LIMB_SPEED}"
    --seed "${SEED}"
)
[[ -n "${EXCLUDE_MOTION_REGEX}" ]] && cluster_cmd+=(--exclude_motion_regex "${EXCLUDE_MOTION_REGEX}")
[[ -n "${LANGUAGE_JSON}" ]] && cluster_cmd+=(--language_json "${LANGUAGE_JSON}")
[[ "${OVERWRITE}" == "1" ]] && cluster_cmd+=(--overwrite)

gallery_cmd=(
    pixi run -e isaaclab python experiments/qualitative/src/qualitative_latent_semantics_gallery.py
    --task "${TASK_NAME}" --headless --device "${DEVICE}"
    --clusters_json "${CLUSTER_DIR}/clusters.json"
    --encoder_checkpoint "${ENCODER_CKPT}"
    --reference_arrays_dir "${REFERENCE_ARRAYS_DIR}"
    --output_dir "${OUTPUT_DIR}"
    --loops "${LOOPS}"
    --slowdown "${SLOWDOWN}"
    --context_frames "${CONTEXT_FRAMES}"
    --filmstrip_members "${FILMSTRIP_MEMBERS}"
    --filmstrip_frames "${FILMSTRIP_FRAMES}"
    --filmstrip_px "${FILMSTRIP_PX}"
    --seed "${SEED}"
    --njmax "${NJMAX}" --nconmax "${NCONMAX}"
)
[[ -n "${CLUSTERS}" ]] && gallery_cmd+=(--clusters "${CLUSTERS}")
[[ -n "${ENV_SPACING}" ]] && gallery_cmd+=(--env_spacing "${ENV_SPACING}")
[[ "${VIDEO}" == "1" ]] && gallery_cmd+=(--video)
[[ "${OVERWRITE}" == "1" ]] && gallery_cmd+=(--overwrite)
gallery_cmd+=("${BASE_OVERRIDES[@]}")

echo "[PLAN] mode        : ${MODE_NAME} (analysis 7)"
echo "[PLAN] code space  : ${ABLATE_CODE_SPACE_DESC}"
echo "[PLAN] encoder     : ${ENCODER_CKPT}"
echo "[PLAN] encoder sha : ${ENCODER_SHA256}"
echo "[PLAN] motions     : ${NUM_MOTIONS} (seed ${SEED}) x ${WINDOWS_PER_MOTION} windows"
echo "[PLAN] clustering  : KMeans k=${K_CLUSTERS} (fixed; silhouette is diagnostic only)"
echo "[PLAN] window gate : min_local_step ${MIN_LOCAL_STEP} (drops standing-start windows; 0 keeps all)"
echo "[PLAN] dynamics    : keep if root>=${MIN_ROOT_SPEED} OR limb>=${MIN_LIMB_SPEED} m/s (0 0 keeps static windows); exclude_motion_regex '${EXCLUDE_MOTION_REGEX}'"
echo "[PLAN] gallery     : ${K_CLUSTERS} videos, ${MEMBERS_PER_CLUSTER} motions each (${MEMBER_SELECTION})"
echo "[PLAN] playback    : ${LOOPS} loops, ${SLOWDOWN}x slower, context ${CONTEXT_FRAMES} frames"
echo "[PLAN] filmstrips  : ${FILMSTRIP_MEMBERS} members/cluster x ${FILMSTRIP_FRAMES} frames @ ${FILMSTRIP_PX}px (0 members disables)"
echo "[PLAN] GPU         : CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} -> ${DEVICE}"
echo "[PLAN] encode      : ${ENCODE_DIR}"
echo "[PLAN] clusters    : ${CLUSTER_DIR}"
echo "[PLAN] output      : ${OUTPUT_DIR}"

if [[ "${SKIP_ENCODE}" != "1" ]]; then
    qualitative_run "${encode_cmd[@]}"
    qualitative_require_output "${ENCODE_DIR}"
else
    echo "[PLAN] SKIP_ENCODE=1; reusing ${ENCODE_DIR}"
fi

if [[ "${SKIP_CLUSTER}" != "1" ]]; then
    qualitative_run "${cluster_cmd[@]}"
    qualitative_require_output "${CLUSTER_DIR}"
else
    echo "[PLAN] SKIP_CLUSTER=1; reusing ${CLUSTER_DIR}"
fi

qualitative_run "${gallery_cmd[@]}"
qualitative_require_output "${OUTPUT_DIR}"
