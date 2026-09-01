#!/usr/bin/env bash
set -euo pipefail

# Stage 1 of 2: pretrain the scaled DETERMINISTIC-64 root_qpos DiffSR skill
# encoder on the full 129,785-motion BONES-SEED reference arrays.
#
#   DRY_RUN=1 bash experiments/qualitative/pretrain_skill_encoder_deter64.sh
#   bash experiments/qualitative/pretrain_skill_encoder_deter64.sh
#
# This is the UNQUANTIZED arm of this directory's two-arm ablation. It is
# pretrain_skill_encoder_fsq64.sh with exactly one training delta:
#
#     --latent_mode sonic_fsq   ->   --latent_mode deterministic
#
# plus the --reg_coeff that only matters once the quantizer is gone. Horizon,
# macro state, trunk widths, DiffSR head, budget, seed, and dataset are
# byte-identical to that file, so QUANTIZATION is the only axis between the two
# encoders. That is the whole point: a difference in the qualitative results is
# then attributable to the bottleneck and to nothing else.
#
# The same delta is what separates the `det64` and `fsq64` arms of
# experiments/campaigns/2026-08-07-bones129k-latent-mode-stride5/run.sh, whose
# `append_latent_mode` is the contract this copies. That campaign runs at macro
# frame stride 5; this file stays at stride 1 (the default, unset here) to match
# the fsq64 arm it is compared against. Do not mix the two.
#
# Terms used here:
#   DiffSR         - the successor-representation skill objective in
#                    RLOpt/rlopt/agent/hl_skill_diffsr.py, pretrained offline
#                    and then frozen for the tracker.
#   deterministic  - no bottleneck at all: the encoder trunk output IS z, all 64
#                    continuous values of it. DeterministicSkillEncoder._latent
#                    returns `(raw, raw.pow(2).mean(), {})`, so the only thing
#                    limiting z is the --reg_coeff L2 penalty on that second
#                    term. There is no codebook, no lattice, and therefore NO
#                    DISCRETE CODE: nothing downstream can name, edit, or count
#                    a code here, and the qualitative analyses omit their
#                    `category_*` columns rather than fabricating one.
#   endpoint       - the transition objective: predict the window endpoint, not
#                    the per-step occupancy.
#
# Self-contained by design: every constant is declared here, so an ablation is
# a single-file edit. Stage 2 is
# experiments/qualitative/train_lowlevel_deter64.sh and declares its own copy.
# Anything the tracker also reads -- HORIZON_STEPS, Z_DIM, LATENT_MODE, the
# macro state terms, the dataset identity -- must change in both, or stage 2
# refuses the encoder on its width assert.
#
# Acceptance gate before stage 2: watch train/loss_real_z_eval against
# train/loss_zero_z_eval and train/loss_shuffled_z_eval in the run's
# metrics.jsonl. Accept checkpoints/latest.pt only once held-out real-z loss has
# flattened while staying clearly below both controls and the latent has not
# collapsed.

# -f, not -x: train.py is mode 644 and is invoked as `python scripts/rlopt/train.py`.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
while [[ ! -f "${REPO_ROOT}/scripts/rlopt/train.py" ]]; do
    [[ "${REPO_ROOT}" == "/" ]] && { echo "[FATAL] repository root not found" >&2; exit 2; }
    REPO_ROOT="$(dirname "${REPO_ROOT}")"
done
cd "${REPO_ROOT}"

fail() { echo "[FATAL] $*" >&2; exit 1; }

DRY_RUN="${DRY_RUN:-0}"

# --- protocol (campaign values) ---------------------------------------------
TASK_NAME="${TASK_NAME:-Isaac-Imitation-G1-v2}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda:0}"
PHYSICS="${PHYSICS:-newton_mjwarp}"

# `expert_motion_qpos` is what makes the macro state root+qpos: 29 joint
# positions + root position + 6D root orientation = 38 per frame. The encoder
# input is therefore 38 x HORIZON_STEPS = 380, and stage 2 asserts that width.
MACRO_STATE_TERMS_OVERRIDE=env.expert_macro_state_terms=[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b]
MACRO_STATE_VALUES_PER_FRAME=38

HORIZON_STEPS="${HORIZON_STEPS:-10}"
Z_DIM="${Z_DIM:-64}"
ENCODER_INPUT_WIDTH=$((HORIZON_STEPS * MACRO_STATE_VALUES_PER_FRAME))
# The tracker publishes z_dim + a 2-value sin_cos phase = 66 -- the SAME width
# as the fsq64 arm, which is what makes the two trackers comparable. Printed
# here only so stage 1 and stage 2 can be read against each other.
LATENT_COMMAND_DIM=$((Z_DIM + 2))

# --- latent bottleneck (campaign values) ------------------------------------
LATENT_MODE="${LATENT_MODE:-deterministic}"
ENCODER_WINDOW_MODE="${ENCODER_WINDOW_MODE:-intermediate}"
TRANSITION_OBJECTIVE="${TRANSITION_OBJECTIVE:-endpoint}"
# L2 weight on z. The parser default, and what the stride-5 campaign's det64 arm
# ran with -- but it is named explicitly here because in this arm it is the ONLY
# thing bounding the latent. The FSQ arm's z lies in [-1, 1] by construction; a
# deterministic z is whatever the trunk emits, so this coefficient sets the scale
# the tracker is commanded in and the scale every later distance is measured in.
REG_COEFF="${REG_COEFF:-1.0e-3}"
ENCODER_HIDDEN_DIMS=(2048 1024 512 512)
DIFFSR_FEATURE_DIM="${DIFFSR_FEATURE_DIM:-256}"
DIFFSR_EMBED_DIM="${DIFFSR_EMBED_DIM:-1024}"
DIFFSR_G_HIDDEN_DIMS=(1024 1024 512)
DIFFSR_MU_HIDDEN_DIMS=(1024 1024 512)

# --- encoder budget (campaign values) ---------------------------------------
# Isaac starts only to build the environment the sampler reads; the encoder
# trains offline from the macro cache, so this is not a rollout width.
PRETRAIN_NUM_ENVS="${PRETRAIN_NUM_ENVS:-16}"
PRETRAIN_UPDATES="${PRETRAIN_UPDATES:-50000}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-8192}"
PRETRAIN_LOG_INTERVAL="${PRETRAIN_LOG_INTERVAL:-1000}"
PRETRAIN_EVAL_BATCHES="${PRETRAIN_EVAL_BATCHES:-4}"

# --- reference data ---------------------------------------------------------
# The campaign read /data/bones_seed_ref_arrays/... inside the ICE container.
# Same content, same persist id, local path.
REFERENCE_ARRAYS_DIR="${REFERENCE_ARRAYS_DIR:-${REPO_ROOT}/data/g1_bones_seed_sonic_129k_50hz_refarrays}"
REFERENCE_ARRAYS_RESIDENT="${REFERENCE_ARRAYS_RESIDENT:-true}"
REFERENCE_ARRAYS_WARM_WORKERS="${REFERENCE_ARRAYS_WARM_WORKERS:-16}"
REFERENCE_PREFETCH_MODE="${REFERENCE_PREFETCH_MODE:-off}"
PERSIST_ID="${PERSIST_ID:-bones_seed_sonic_full_129785@e714bbff}"
EXPECTED_MOTIONS="${EXPECTED_MOTIONS:-129785}"
EXPECTED_TRANSITIONS="${EXPECTED_TRANSITIONS:-47491234}"
# Matches the v2 reference channel's anchor. The arrays bake it in and loading
# refuses a directory built for a body set that does not contain it.
ANCHOR_BODY="${ANCHOR_BODY:-pelvis}"

# Provenance only: the arrays carry their own identity gate, so training never
# reads this file. When present it must be the manifest the arrays were built
# from.
MANIFEST_PATH="${MANIFEST_PATH:-${REPO_ROOT}/data/bones_seed_sonic_129k_50hz/g1_bones_seed_sonic_full_manifest.json}"
EXPECTED_MANIFEST_SHA256="${EXPECTED_MANIFEST_SHA256:-eb0ad052afe72fb6228f4be9d52132c9cb9a52ac9c561751e49e6ca31346e688}"

# Order is column position in the arrays, not a set. Do not sort or edit.
RUNTIME_BODY_NAMES=(
    pelvis
    left_hip_roll_link left_knee_link left_ankle_roll_link
    right_hip_roll_link right_knee_link right_ankle_roll_link
    torso_link
    left_shoulder_roll_link left_elbow_link left_wrist_yaw_link
    right_shoulder_roll_link right_elbow_link right_wrist_yaw_link
)
RUNTIME_BODY_NAMES_OVERRIDE="env.data.runtime_cache_body_names=[$(IFS=,; echo "${RUNTIME_BODY_NAMES[*]}")]"

# --- output and logging -----------------------------------------------------
# The campaign's W&B destination. Confirm or override the group before a real
# launch; do not silently mix an ablation into the campaign's group.
WANDB_PROJECT="${WANDB_PROJECT:-g1-bones-seed}"
WANDB_GROUP="${WANDB_GROUP:-skill-encoding-ablation}"
WANDB_TAGS="${WANDB_TAGS:-bones-seed,129785,v2,root-qpos,deterministic64,scaled-pretrain,${TRANSITION_OBJECTIVE}}"
LOGGER_BACKEND="${LOGGER_BACKEND:-wandb}"

# The tag carries the ablated values so variants land in separate directories.
# Stage 2 derives the same default path from its own copy of these constants.
ABLATE_LOG_ROOT="${ABLATE_LOG_ROOT:-${REPO_ROOT}/logs/ablate_latent}"
ENCODER_RUN_TAG="${ENCODER_RUN_TAG:-bones129k_encoder_${LATENT_MODE}_h${HORIZON_STEPS}_z${Z_DIM}_seed${SEED}}"
ENCODER_DIR="${ENCODER_DIR:-${ABLATE_LOG_ROOT}/encoder/${ENCODER_RUN_TAG}}"

# --- data gate --------------------------------------------------------------
if [[ -f "${MANIFEST_PATH}" ]]; then
    actual_manifest_sha="$(sha256sum "${MANIFEST_PATH}" | awk '{print $1}')"
    [[ "${actual_manifest_sha}" == "${EXPECTED_MANIFEST_SHA256}" ]] \
        || fail "manifest SHA mismatch: expected=${EXPECTED_MANIFEST_SHA256} actual=${actual_manifest_sha}"
    echo "[PASS] manifest provenance: ${MANIFEST_PATH}"
else
    echo "[NOTE] manifest absent (${MANIFEST_PATH}); relying on the arrays' own identity gate."
fi

[[ -d "${REFERENCE_ARRAYS_DIR}" ]] || fail "reference arrays not found: ${REFERENCE_ARRAYS_DIR}
  Fetch them with
    pixi run python -m imitation_experiments.data.publish_reference_arrays fetch \\
      --repo_id GeorgiaTech/g1_bones_seed_sonic_129k_50hz_refarrays \\
      --dest_dir ${REFERENCE_ARRAYS_DIR} --persist_id ${PERSIST_ID} \\
      --expected_motions ${EXPECTED_MOTIONS} --expected_transitions ${EXPECTED_TRANSITIONS}
  or build them from the NPZ tree with
    pixi run python -m imitation_experiments.data.build_reference_arrays --help"

# Size, sidecar, persist id, body list, motion and transition counts. Seconds,
# no rebuild. --manifest is required by the parser but unread without
# --verify_load, which is a build-time check against the source NPZs.
pixi run python -m imitation_experiments.data.build_reference_arrays \
    --manifest "${MANIFEST_PATH}" \
    --output_dir "${REFERENCE_ARRAYS_DIR}" \
    --persist_id "${PERSIST_ID}" \
    --anchor_body "${ANCHOR_BODY}" \
    --body_names "${RUNTIME_BODY_NAMES[@]}" \
    --expected_motions "${EXPECTED_MOTIONS}" \
    --expected_transitions "${EXPECTED_TRANSITIONS}" \
    --validate_only

export TERM="${TERM:-xterm}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export WANDB_TAGS

# --- command ----------------------------------------------------------------
# Pretraining samples encoder windows from the macro cache only; it does not
# need the 14-body live-tracking cache the tracker builds in stage 2.
pretrain_cmd=(
    pixi run -e isaaclab python scripts/rlopt/train_hl_skill_diffsr.py
    --task "${TASK_NAME}" --num_envs "${PRETRAIN_NUM_ENVS}"
    --seed "${SEED}" --device "${DEVICE}" --headless --assert-kitless
    --output_dir "${ENCODER_DIR}"
    --logger_backend "${LOGGER_BACKEND}"
    --wandb_project "${WANDB_PROJECT}"
    --wandb_group "${WANDB_GROUP}"
    --wandb_run_name "${ENCODER_RUN_TAG}"
    "physics=${PHYSICS}"
    env.data.manifest=null
    "env.data.reference_arrays_dir=${REFERENCE_ARRAYS_DIR}"
    "env.data.persist_id=${PERSIST_ID}"
    "env.data.reference_arrays_resident=${REFERENCE_ARRAYS_RESIDENT}"
    "env.data.reference_arrays_warm_workers=${REFERENCE_ARRAYS_WARM_WORKERS}"
    env.data.runtime_cache_device=cpu
    "env.data.reference_prefetch_mode=${REFERENCE_PREFETCH_MODE}"
    "env.data.macro_cache_device=${DEVICE}"
    "${RUNTIME_BODY_NAMES_OVERRIDE}"
    "${MACRO_STATE_TERMS_OVERRIDE}"
    # --- append_pretrain_contract, verbatim ---
    --horizon_steps "${HORIZON_STEPS}"
    --encoder_window_mode "${ENCODER_WINDOW_MODE}"
    --transition_objective "${TRANSITION_OBJECTIVE}"
    --z_dim "${Z_DIM}" --latent_mode "${LATENT_MODE}"
    --reg_coeff "${REG_COEFF}"
    --encoder_hidden_dims "${ENCODER_HIDDEN_DIMS[@]}"
    --encoder_activation silu --no_encoder_layer_norm
    --diffsr_feature_dim "${DIFFSR_FEATURE_DIM}"
    --diffsr_embed_dim "${DIFFSR_EMBED_DIM}"
    --diffsr_g_hidden_dims "${DIFFSR_G_HIDDEN_DIMS[@]}"
    --diffsr_mu_hidden_dims "${DIFFSR_MU_HIDDEN_DIMS[@]}"
    --batch_size "${PRETRAIN_BATCH_SIZE}"
    --num_updates "${PRETRAIN_UPDATES}"
    --log_interval "${PRETRAIN_LOG_INTERVAL}"
    --eval_batches "${PRETRAIN_EVAL_BATCHES}"
    --reconstruction_eval --window_probe_eval
    --window_probe_train_batches 8 --window_probe_eval_batches 4
)

echo "[PLAN] stage       : 1 of 2 -- skill encoder"
echo "[PLAN] contract    : fsq64 pretrain with --latent_mode deterministic (unquantized arm)"
echo "[PLAN] data        : ${EXPECTED_MOTIONS} motions / ${EXPECTED_TRANSITIONS} transitions"
echo "[PLAN] source      : reference arrays (mapped) ${REFERENCE_ARRAYS_DIR}"
echo "[PLAN] macro state : root+qpos, ${MACRO_STATE_VALUES_PER_FRAME}/frame, ${ENCODER_INPUT_WIDTH}D h${HORIZON_STEPS} window"
echo "[PLAN] latent      : ${LATENT_MODE} (continuous, no code), z${Z_DIM} -> ${LATENT_COMMAND_DIM} published by stage 2 with the sin_cos phase"
echo "[PLAN] z bound     : --reg_coeff ${REG_COEFF} only; z is otherwise unbounded"
echo "[PLAN] objective   : ${TRANSITION_OBJECTIVE}, encoder [${ENCODER_HIDDEN_DIMS[*]}] silu, no layer norm"
echo "[PLAN] diffsr      : feature ${DIFFSR_FEATURE_DIM}, embed ${DIFFSR_EMBED_DIM}, g [${DIFFSR_G_HIDDEN_DIMS[*]}], mu [${DIFFSR_MU_HIDDEN_DIMS[*]}]"
echo "[PLAN] budget      : ${PRETRAIN_UPDATES} updates x ${PRETRAIN_BATCH_SIZE} = $((PRETRAIN_UPDATES * PRETRAIN_BATCH_SIZE)) windows"
echo "[PLAN] output      : ${ENCODER_DIR}"
echo "[PLAN] W&B         : ${WANDB_PROJECT} / ${WANDB_GROUP} / ${WANDB_TAGS}"
printf '  '; printf '%q ' "${pretrain_cmd[@]}"; printf '\n'

if [[ "${DRY_RUN}" == "1" ]]; then
    exit 0
fi

mkdir -p "${ENCODER_DIR}"
exec "${pretrain_cmd[@]}"
