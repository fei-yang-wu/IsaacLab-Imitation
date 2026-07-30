#!/usr/bin/env bash
set -euo pipefail

# Interface-design ablation: how much of the explicit command packet is actually
# needed? Extends the latent-vs-full-body comparison with reduced explicit
# interfaces, all evaluated under the identical frozen protocol.
#
# VARIANTS (per-frame widths; packets are 10 frames)
#   full_body      root 9 + qpos 29 + qvel 29 = 67  -> 670   (existing baseline)
#   root_qpos       root 9 + qpos 29             = 38  -> 380   Heracles 38D
#   root_qpos_sel   root 9 + selected qpos k     = 9+k -> 10(9+k)
#   root_points5    root 9 + 5 keypoints x 3     = 24  -> 240   HuMI-style
#   ee_chunk        4 EE x (3 pos + 6 rot6d)     = 36  -> 360   ABANDONED (see below)
#
# G1_EE_BODY_NAMES is {left,right}_ankle_roll_link + {left,right}_wrist_yaw_link
# -- feet and wrists, NO pelvis/root. HuMI's ablation reports EE-only loses
# whole-body intent and that adding pelvis substantially improves it, so
# ee_chunk is the configuration HuMI calls deficient; root_points5 (4 EE +
# pelvis) is the HuMI-style variant. Label them accordingly in the paper.
#
# EACH INTERFACE GETS ITS OWN LOW-LEVEL CONTROLLER. This is a full study, not an
# adaptation of predicted actions onto the existing full-body tracker, so the
# controller is RETRAINED to consume each command space natively. Consequently
# there is NO channel reconstruction: a root+qpos controller simply never
# receives joint velocities, so qvel is absent by design rather than
# finite-differenced. This matches how the existing rows were built -- latent,
# full-body and EE each already have their own 5B-step controller.
#
# COST: a new controller is ~1B environment frames / two-day walltime (AGENTS.md
# default), which dominates the ablation budget. Variants whose controller
# already exists are therefore dramatically cheaper and should run first.
#
# STATUS: this launcher is complete; the target-spec support for the reduced
# variants is NOT. Each unimplemented variant fails fast with what is missing
# rather than silently running the full-body packet.
#
# Usage:
#   DRY_RUN=1 VARIANT=root_qpos experiments/campaigns/2026-07-23-lafan1-planner-capacity/run_interface_ablation.sh
#   VARIANT=root_qpos MOTION_NAME=walk1_subject1 SIZES="tiny small medium large" SEEDS="0 1 2" ...

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || (cd "${SCRIPT_DIR}/../../.." && pwd))"
cd "${REPO_ROOT}"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/paths.env"

VARIANT="${VARIANT:?set VARIANT=ee_chunk|root_qpos|root_qpos_sel|root_points5}"
SIZES="${SIZES:-tiny small medium large}"
SEEDS="${SEEDS:-0 1 2}"
DEVICE="${DEVICE:-cuda:0}"
DRY_RUN="${DRY_RUN:-0}"
EVAL_STEPS="${EVAL_STEPS:-700}"
FH_ENVS="${FH_ENVS:-4}"
DEMO_ONLY="${DEMO_ONLY:-1}"
STUDY_ROOT="${STUDY_ROOT:-logs/interface_baselines/lafan1_interface_ablation_${VARIANT}}"

# Per-variant contract. TARGET_DIM is the planner's output width; RECONSTRUCT
# names how the publisher rebuilds the tracker's full 67-per-frame command.
case "${VARIANT}" in
    ee_chunk)
        # ABANDONED 2026-07-27. The adapter was built and VERIFIED (chunked
        # streaming reproduces the unchunked floor to -5.3mm), but the interface
        # itself is unusable: the released EE controller cannot track walk1 even
        # with a perfect expert command every control step -- replay floor
        # 405.2mm, root error 2.05m, joint RMSE 0.93 rad (full-body oracle is
        # 23.8mm). The packet carries 4 body poses in the TORSO frame and
        # nothing else, so it never specifies where the torso should be and 4
        # poses do not determine 29 joints. This is HuMI's "EE-only loses
        # whole-body intent" in its starkest form. root_points5 is the same idea
        # WITH the root added, which is what HuMI reports rescues it.
        echo "[ERROR] ee_chunk is abandoned: interface is under-determined (replay floor 405mm)." >&2
        echo "        See qualify_interface.sh; use root_points5 instead." >&2
        exit 2 ;;
    root_qpos)
        # Command space BUILT and certified 2026-07-27: streamed slots reproduce
        # the unchunked reference bit-exactly on the joint half and to ~1e-6 on
        # the root half, at all 10 hold phases (smoke_test_reduced_interface_streaming.py).
        PER_FRAME=38
        LOW_LEVEL="${ROOT_QPOS_LOW_LEVEL_CHECKPOINT:-}"
        CONTROLLER="NEEDS TRAINING (~1B frames, 2-day walltime)"
        NEEDS="the low-level controller for command_space='root_qpos' (the command space itself is built and certified)" ;;
    root_qpos_sel)
        SELECTED_JOINTS="${SELECTED_JOINTS:?set SELECTED_JOINTS, e.g. the 12 leg joints}"
        PER_FRAME=$((9 + $(awk -F, '{print NF}' <<<"${SELECTED_JOINTS}")))
        LOW_LEVEL="${ROOT_QPOS_SEL_LOW_LEVEL_CHECKPOINT:-}"
        CONTROLLER="NEEDS TRAINING (~1B frames, 2-day walltime)"
        NEEDS="command space 'root_qpos_sel' + a PRINCIPLED joint-selection rule; then train the controller" ;;
    root_points5)
        # Command space BUILT and certified 2026-07-27 alongside root_qpos.
        # Keypoints are pelvis + the four EE bodies, positions only, expressed in
        # the same anchor frame as the root half so one transform re-expresses
        # the whole packet.
        PER_FRAME=24
        LOW_LEVEL="${ROOT_POINTS5_LOW_LEVEL_CHECKPOINT:-}"
        CONTROLLER="NEEDS TRAINING (~1B frames, 2-day walltime)"
        NEEDS="the low-level controller for command_space='root_points5' (the command space itself is built and certified)" ;;
    *) echo "[ERROR] unknown VARIANT=${VARIANT}" >&2; exit 2 ;;
esac
# ------------------------------------------------------- horizon / overlap --
# Second ablation axis. The tracker runs at 50 Hz and the planner publishes every
# PUBLISH_INTERVAL control steps, so:
#
#   WINDOW_STEPS == PUBLISH_INTERVAL  -> chunks are NON-overlapping (the current
#       H=10 @ 5 Hz design). Each control step is covered by exactly one
#       prediction, so per-step temporal ensembling is structurally unavailable.
#   WINDOW_STEPS >  PUBLISH_INTERVAL  -> receding horizon with OVERLAP. Each step
#       is covered by ceil(WINDOW/INTERVAL) predictions from different publish
#       times, which is what makes smoothing possible at a fixed planner rate.
#
# Published practice is split, which is why this is worth measuring rather than
# assuming: SafeFlow generates T_fut=8 at 6.25 Hz into a 50 Hz tracker -- exactly
# non-overlapping, like the current design. Heracles predicts a 0.2 s window but
# replans every 2 control steps (25 Hz), a 5x overlap. HuMI trains a 48-step
# horizon while publishing at 5 Hz, also heavily overlapping.
#
# SMOOTHING applies only when there IS overlap. ACT-style temporal ensembling
# assumes the policy runs at the control rate, which a 5 Hz planner cannot do
# (measured planner latency is 25-32 ms against a 20 ms budget at 50 Hz), so the
# adapted forms are:
#   none      consume the newest chunk slot-by-slot (current behaviour)
#   blend     exponentially weight the overlapping predictions for this step
#   crossfade linearly fade the outgoing chunk into the incoming one over a few
#             steps -- cheapest, targets chunk-boundary discontinuity only
PUBLISH_INTERVAL="${PUBLISH_INTERVAL:-10}"
COMMAND_FUTURE_STEPS="${COMMAND_FUTURE_STEPS:-9}"
SMOOTHING="${SMOOTHING:-none}"
case "${SMOOTHING}" in none|blend|crossfade) ;; *)
    echo "[ERROR] SMOOTHING must be none|blend|crossfade" >&2; exit 2 ;; esac

WINDOW_STEPS=$(( ${COMMAND_PAST_STEPS:-0} + 1 + COMMAND_FUTURE_STEPS ))
TARGET_DIM=$(( PER_FRAME * WINDOW_STEPS ))
if (( WINDOW_STEPS < PUBLISH_INTERVAL )); then
    echo "[ERROR] WINDOW_STEPS=${WINDOW_STEPS} < PUBLISH_INTERVAL=${PUBLISH_INTERVAL}: some held control step would have no command slot." >&2
    exit 2
fi
OVERLAP=$(( (WINDOW_STEPS + PUBLISH_INTERVAL - 1) / PUBLISH_INTERVAL ))
if (( OVERLAP == 1 )) && [[ "${SMOOTHING}" != "none" ]]; then
    echo "[ERROR] SMOOTHING=${SMOOTHING} needs overlapping chunks, but WINDOW_STEPS=${WINDOW_STEPS} == PUBLISH_INTERVAL=${PUBLISH_INTERVAL} gives exactly one prediction per step. Raise COMMAND_FUTURE_STEPS." >&2
    exit 2
fi

cat <<INFO
[ablation] variant=${VARIANT}
[ablation]   per-frame=${PER_FRAME}  window=${WINDOW_STEPS}  target_dim=${TARGET_DIM}  (full_body=670)
[ablation]   low-level controller: ${CONTROLLER}
[ablation]   horizon=${WINDOW_STEPS} publish_every=${PUBLISH_INTERVAL} -> overlap=${OVERLAP}x  smoothing=${SMOOTHING}
[ablation]   motion=${MOTION_NAME}  sizes='${SIZES}'  seeds='${SEEDS}'  demo_only=${DEMO_ONLY}
[ablation]   study_root=${STUDY_ROOT}
INFO

# Fail fast rather than silently running the full-body packet under an ablation
# label. A variant is READY only once its command space exists, its own low-level
# controller is trained, and that controller's oracle has passed
# qualify_interface.sh on this interface.
#
# READY as of 2026-07-28 (qualify_interface.sh, walk1_subject1, frame-0/700-step):
#   root_qpos     380  oracle 23.6 mm  (full-body reference 23.8 mm)
#   root_points5  240  oracle 30.6 mm  (latent reference     30.5 mm)
# Chunking loss was +0.0/-0.0 mm on both, so planner error on these interfaces is
# attributable to the planner.
READY_VARIANTS="${READY_VARIANTS:-root_qpos root_points5}"
variant_is_ready=0
for _ready in ${READY_VARIANTS}; do
    [[ "${VARIANT}" == "${_ready}" ]] && variant_is_ready=1
done

if [[ "${variant_is_ready}" != "1" && "${ALLOW_UNIMPLEMENTED:-0}" != "1" ]]; then
    cat >&2 <<ERR
[ERROR] VARIANT=${VARIANT} is not implemented yet. Missing: ${NEEDS}.

  The launcher, dims and protocol are settled; what remains is the planner
  target spec (and, for ee_chunk, an env adapter). Running now would train a
  670-value full-body planner and mislabel it as '${VARIANT}'.

  Required before any planner compute, per the campaign's own protocol:
    1. add the '${VARIANT}' command space + planner target spec
       (plus overlap/smoothing support if OVERLAP>1)
    2. TRAIN the low-level controller for this command space, unless it already
       exists (~1B frames / two-day walltime -- this dominates the budget)
    3. qualify that controller's oracle ON THIS INTERFACE (frame 0, ${EVAL_STEPS}
       steps, no terminations) -- a planner study on an interface its own
       controller cannot follow measures nothing
    4. only then run the capacity grid

  Set ALLOW_UNIMPLEMENTED=1 to print the planned command matrix without running.
ERR
    exit 3
fi

# ---------------------------------------------------------------- oracle ----
# Qualification gate: drive the frozen tracker with EXPERT commands through this
# reduced interface. If the oracle cannot track, the interface is unusable and
# planner results on it would be meaningless.
ORACLE_DIR="${STUDY_ROOT}/oracle_baselines/${VARIANT}"
echo "[ablation] STEP 1 oracle metrics + demonstrations -> ${ORACLE_DIR}"
# The grid's planner-pretrain stage reads
# ${ORACLE_DIR}/oracle_demonstrations/rollout_training_samples, so the demos must
# actually be collected here -- creating the directory is not enough.
if [[ -s "${ORACLE_DIR}/oracle_demonstrations/rollout_training_samples/sample_step_000000.pt" ]]; then
    echo "[ablation]   demonstrations already present; skipping collection"
elif [[ "${DRY_RUN}" == "1" ]]; then
    echo "[ablation]   would run prepare_oracle_baselines.sh INTERFACES=${VARIANT}"
else
    mkdir -p "${ORACLE_DIR}"
    INTERFACES="${VARIANT}" OUTPUT_ROOT="${STUDY_ROOT}/oracle_baselines" \
    MOTION_NAME="${MOTION_NAME}" MANIFEST="${MANIFEST}" \
        "${SCRIPT_DIR}/prepare_oracle_baselines.sh"
fi

# ------------------------------------------------------------------ grid ----
echo "[ablation] STEP 2 capacity grid (one cell per size x seed)"
for seed in ${SEEDS}; do
    for size in ${SIZES}; do
        echo "[ablation]   cell size=${size} seed=${seed}"
        [[ "${DRY_RUN}" == "1" ]] && continue
        DEMO_ONLY="${DEMO_ONLY}" MODEL_SIZE="${size}" PLANNER_SEED="${seed}" \
        STUDY_ROOT="${STUDY_ROOT}" INTERFACES="${VARIANT}" \
        MOTION_NAME="${MOTION_NAME}" MANIFEST="${MANIFEST}" \
            "${SCRIPT_DIR}/run_capacity_point.sh"
    done
done

echo "[PASS] interface ablation ${VARIANT} complete under ${STUDY_ROOT}"
