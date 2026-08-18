#!/usr/bin/env bash
# Shared arm table for the 10B-tracker planner round. Sourced by collect.sh,
# train.sh and eval.sh so one edit changes every stage.
#
# Both trackers come from `2026-08-15-latent-bottleneck-10b` at 10.0B frames,
# mirrored from ICE under logs/bottleneck_10b_mirror/. Both were trained with
# the SONIC v1.1 `robot_heading` macro frame, which is NOT the environment
# default: the override is mandatory and width-invisible if forgotten.
#
# HOLD is the publication cadence the tracker was TRAINED at, so it is also
# the cadence the planner must emit at. It fixes the head's target layout:
#
#   fsq64_10b     hold 10 -> 3 latents x 64 dims per call   (SONIC token space)
#   ln_hold1_10b  hold  1 -> 30 latents x 256 dims per call (per-step stream)
#
# At hold 1 the `sin_cos` phase channel is the constant (0, 1) — `code_period`
# 1 makes phase (period - steps)/period = 0 on every step — which is exactly
# what the tracker saw in training. The 2026-08-13 hold-1 refutation (85.77 mm)
# does not apply here: that arm drove a hold-10-trained tracker whose phase
# channel swept 0 -> 0.9 in training and sat pinned at slot 0 under the planner.
set -euo pipefail

MIRROR_ROOT="${MIRROR_ROOT:-${REPO_ROOT}/logs/bottleneck_10b_mirror}"

arm_config() {
    case "${1}" in
        fsq64_10b)
            TRACKER="${MIRROR_ROOT}/fsq64_hold10_seed0/tracker/f10000269312/models/model_step_10000269312.pt"
            ENCODER="${MIRROR_ROOT}/fsq64_hold10_seed0/encoder/checkpoints/latest.pt"
            ACTOR_DIM=66; CODE_DIM=64; HOLD=10; CODE_PERIOD=10; NJMAX=320
            SLOTS=30            # every-step rows; the join builds 3 x hold 10
            CONSUME_SLOTS=1     # publish one latent per tick, held 10 steps
            WITH_ROOT_QPOS=1    # fsq_prequant re-encodes from the lookahead
            ;;
        ln_hold1_10b)
            TRACKER="${MIRROR_ROOT}/cont_det_ln_hold1_seed0/tracker/f10000269312/models/model_step_10000269312.pt"
            ENCODER="${MIRROR_ROOT}/cont_det_ln_hold1_seed0/encoder/checkpoints/latest.pt"
            ACTOR_DIM=258; CODE_DIM=256; HOLD=1; CODE_PERIOD=1; NJMAX=320
            SLOTS=30            # 30 consecutive per-step latents per call
            CONSUME_SLOTS=10    # consume 10, re-plan, ensemble the overlap
            WITH_ROOT_QPOS=0    # target is the stored per-step z
            ;;
        *)
            echo "arm must be fsq64_10b|ln_hold1_10b, got ${1}" >&2
            return 1
            ;;
    esac
    # Both arms carry the scaled tracker geometry and the heading frame.
    ARM_CFG=(
        'agent.policy.num_cells=[2048,2048,1024,1024,512,512]'
        'agent.value_function.num_cells=[2048,2048,1024,1024,512,512]'
        agent.policy.activation_fn=silu
        agent.value_function.activation_fn=silu
        env.expert_macro_anchor_mode=robot_heading
    )
    for required in "${TRACKER}" "${ENCODER}"; do
        [ -f "${required}" ] || { echo "missing tracker input: ${required}" >&2; return 1; }
    done
}

# The 2026-08-17 motion set (`bones_seed_language30_v2`): the 2026-08-13
# thirty, minus `walk_big_dog_ff_225_stop` (tracker-limited on both 10B
# trackers) and `looking_in_the_mirror` (least discriminating, and Object
# Interaction was over-represented), plus `walk_ff_loop_360` ("walk
# backwards") and `reaching_up` ("reaching up"). Ranks are this dataset's
# own 0..29 and do NOT match the v1 dataset's.
DATA_ROOT="${REPO_ROOT}/data/bones_seed_language30_v2"
MANIFEST_LANG="${DATA_ROOT}/manifests/g1_bones_seed_language30_v2_manifest_language.json"
REFERENCE_ARRAYS="${DATA_ROOT}/reference_arrays/root_qpos_v1"
PERSIST_ID="bones_seed_language30_v2@7a6d5c49"
GOAL_FEATURES="${REPO_ROOT}/outputs/planner_10b/goal_features/goal_features.pt"
LANGUAGE_EMBEDDINGS="${DATA_ROOT}/language/g1_bones_seed_language30_v2_minilm_goal_embeddings.pt"
RUNTIME_BODY_NAMES='env.data.runtime_cache_body_names=[pelvis,left_hip_roll_link,left_knee_link,left_ankle_roll_link,right_hip_roll_link,right_knee_link,right_ankle_roll_link,torso_link,left_shoulder_roll_link,left_elbow_link,left_wrist_yaw_link,right_shoulder_roll_link,right_elbow_link,right_wrist_yaw_link]'

# Every motion in this dataset is trainable on both trackers: the one that
# was not (`walk_big_dog_ff_225_stop`) is already absent from it, so nothing
# is excluded at run time. EXCLUDE_RANKS stays as an escape hatch.
EXCLUDE_RANKS="${EXCLUDE_RANKS:-}"

load_motions() {
    readarray -t ALL_MOTIONS < <(
        "${REPO_ROOT}/.pixi/envs/default/bin/python" - "${MANIFEST_LANG}" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1]))
motions = manifest["motions"] if "motions" in manifest else manifest
print("\n".join(m["name"] for m in motions))
PY
    )
    MOTIONS=(); RANKS=()
    for index in "${!ALL_MOTIONS[@]}"; do
        skip=0
        for dropped in ${EXCLUDE_RANKS}; do
            [ "${index}" = "${dropped}" ] && skip=1
        done
        # Ranks stay the ORIGINAL manifest positions: they index the reference
        # arrays, so renumbering after a drop would score different motions.
        [ "${skip}" = "0" ] && { MOTIONS+=("${ALL_MOTIONS[$index]}"); RANKS+=("${index}"); }
    done
    NUM_GOALS="${#MOTIONS[@]}"
}
