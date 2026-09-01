#!/usr/bin/env bash
# Shared setup for the qualitative-analysis launchers in this directory.
#
# Sourced, never executed. Self-contained: this directory owns its own copy of
# the launcher contract (data gate, encoder width assert, policy resolver, Hydra
# override block, GPU rules, artifact check) and its own copy of the Python
# entrypoints.
#
# TWO ARMS live here, selected by LATENT_ARM. Both publish a 66-wide command
# (64 code values + a 2-value sin_cos phase) from the same 380-wide root_qpos
# macro window, at horizon 10. QUANTIZATION is the only axis between them.
#
#   LATENT_ARM=fsq64   (default) `sonic_fsq`: 64 finite-scalar-quantization
#       coordinates at 32 ordered levels each. Coordinate g owns exactly ONE
#       value of the 64-value z, so `code_dim` is 1, against the multicat
#       encoder's 64 groups x 128 nominal categories over 4 values each. Two
#       consequences for reading the results:
#         * A per-coordinate edit moves 1 of 64 latent values, not 4 of 256.
#         * Level ids are ORDERED lattice positions, so +-1 level is a small
#           move and an edit across the lattice is a large one. Multicat
#           category ids are nominal and carry no such ordering.
#
#   LATENT_ARM=deter64  `deterministic`: the encoder trunk output IS z. No
#       quantizer, no codebook, no lattice, and therefore NO DISCRETE CODE --
#       nothing to name, edit, or count. Three consequences:
#         * z is unbounded. Only the pretrain `--reg_coeff` L2 penalty limits
#           it, so its per-dimension scale is a property of the trained encoder
#           rather than of the code space, and any distance measured in it
#           inherits that scale. Every entrypoint records the spread.
#         * The `category_*`/`level` output columns are ABSENT, not zero-filled.
#           The `latent_*` columns carry the full record for both arms.
#         * qualitative_ncoord_intervention.sh REFUSES this arm at encoder load.
#           A per-coordinate level edit has no continuous analogue and none is
#           invented here.
#
# Every value below is environment-overridable, so an ablation is one exported
# variable rather than an edit.

fail() { echo "[FATAL] $*" >&2; exit 1; }

# -f, not -x: train.py is mode 644 and is invoked as `python scripts/rlopt/train.py`.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
while [[ ! -f "${REPO_ROOT}/scripts/rlopt/train.py" ]]; do
    [[ "${REPO_ROOT}" == "/" ]] && { echo "[FATAL] repository root not found" >&2; exit 2; }
    REPO_ROOT="$(dirname "${REPO_ROOT}")"
done
cd "${REPO_ROOT}"

DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
OVERWRITE="${OVERWRITE:-0}"

# --- which code space -------------------------------------------------------
# Sets only what differs between arms. Defaulting to fsq64 keeps every existing
# command line resolving exactly as it did before this arm was added.
LATENT_ARM="${LATENT_ARM:-fsq64}"
case "${LATENT_ARM}" in
    fsq64)
        ABLATE_LATENT_MODE=sonic_fsq
        ABLATE_ARM_SLUG=sonic-fsq64
        ABLATE_IS_DISCRETE=1
        ;;
    deter64)
        ABLATE_LATENT_MODE=deterministic
        ABLATE_ARM_SLUG=deterministic64
        ABLATE_IS_DISCRETE=0
        ;;
    *) fail "LATENT_ARM must be fsq64 or deter64; got ${LATENT_ARM}" ;;
esac

# --- which tracker ----------------------------------------------------------
# The two 2026-08-06-bones129k-sonic-fsq-scale arms, identical except for
# actor/critic capacity. Only `tuned` has local checkpoints today; `sonic` is
# accepted so the same launchers work the moment that arm produces one.
TRACKER_ARM="${TRACKER_ARM:-tuned}"
case "${TRACKER_ARM}" in
    tuned) ABLATE_TRACKER_CELLS="[1024,1024,512]" ;;
    sonic) ABLATE_TRACKER_CELLS="[2048,2048,1024,1024,512,512]" ;;
    *) fail "TRACKER_ARM must be tuned or sonic; got ${TRACKER_ARM}" ;;
esac

# --- protocol ---------------------------------------------------------------
TASK_NAME="${TASK_NAME:-Isaac-Imitation-G1-v2}"
AGENT_ENTRY_POINT="${AGENT_ENTRY_POINT:-rlopt_ipmd_tuned_cfg_entry_point}"
PHYSICS="${PHYSICS:-newton_mjwarp}"
SEED="${SEED:-0}"

# This box enumerates 8 GPUs through NVML but only 0-6 are actually usable, and
# Isaac Lab's AppLauncher picks its device before Kit narrows the visible set --
# leaving it unpinned makes torch assert `device=7, num_gpus=7` during startup.
# Pin one usable GPU here and address it as cuda:0 inside the process.
# Not every GPU here can drive the RTX renderer. Measured directly: any visible
# list with >= 2 of GPUs 0-3 renders (0,2 / 0,3 / 1,2 / 1,2,3 / 0..6); every list
# with fewer fails with "Skipping NVIDIA graphics device" and a segfault
# (0,6 / 2,6 / 3,6 / 5,6 / 4,5 / a single index) -- even when all of them are
# completely idle. GPUs 4-6 can pad a list but cannot satisfy the renderer.
RENDER_CAPABLE_GPUS="${RENDER_CAPABLE_GPUS:-0 1 2 3}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES
DEVICE="${DEVICE:-cuda:0}"

# Must match the encoder: root+qpos is 29 joint positions + root position + 6D
# root orientation = 38 per frame, so the encoder input is 38 x HORIZON_STEPS.
MACRO_STATE_TERMS_OVERRIDE=env.expert_macro_state_terms=[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b]
MACRO_STATE_VALUES_PER_FRAME=38
HORIZON_STEPS="${HORIZON_STEPS:-10}"
Z_DIM="${Z_DIM:-64}"
EXPECTED_LATENT_MODE="${EXPECTED_LATENT_MODE:-${ABLATE_LATENT_MODE}}"
LATENT_HOLD_STEPS="${LATENT_HOLD_STEPS:-${HORIZON_STEPS}}"
ENCODER_INPUT_WIDTH=$((HORIZON_STEPS * MACRO_STATE_VALUES_PER_FRAME))
# The published command is z_dim + the 2-value sin_cos phase.
LATENT_COMMAND_DIM=$((Z_DIM + 2))

# One-line description of the selected code space, for the [PLAN] blocks. Set
# here, not in each launcher, so no launcher can print "32 ordered levels" over
# a continuous latent.
if [[ -z "${ABLATE_CODE_SPACE_DESC:-}" ]]; then
    if [[ "${ABLATE_IS_DISCRETE}" == "1" ]]; then
        ABLATE_CODE_SPACE_DESC="${ABLATE_LATENT_MODE}, ${Z_DIM} coordinates x 32 ordered levels"
    else
        ABLATE_CODE_SPACE_DESC="${ABLATE_LATENT_MODE}, ${Z_DIM} continuous values (no code)"
    fi
fi

# --- checkpoints ------------------------------------------------------------
BONES129K_LOG_ROOT="${BONES129K_LOG_ROOT:-${REPO_ROOT}/logs/ablate_latent}"
# Mirrors the default output path of this directory's stage-1 script for the
# selected arm, so a trained encoder is found without passing ENCODER_CKPT.
ENCODER_RUN_TAG="${ENCODER_RUN_TAG:-bones129k_encoder_${ABLATE_LATENT_MODE}_h${HORIZON_STEPS}_z${Z_DIM}_seed0}"
# latest.pt, not best.pt: both trackers' command.txt pin latest.pt, and the
# binding gate in the Python entrypoints requires tensor-identical weights.
ENCODER_CKPT="${ENCODER_CKPT:-${BONES129K_LOG_ROOT}/encoder/${ENCODER_RUN_TAG}/checkpoints/latest.pt}"
LOWLEVEL_RUN_TAG="${LOWLEVEL_RUN_TAG:-bones129k_scaled_${LATENT_ARM}_${TRACKER_ARM}_tracker_5000m_seed0}"
LOWLEVEL_RUN_DIR="${LOWLEVEL_RUN_DIR:-${BONES129K_LOG_ROOT}/lowlevel/${LOWLEVEL_RUN_TAG}}"
# Empty means "resolve the newest model_step_<N>.pt under LOWLEVEL_RUN_DIR".
POLICY_CKPT="${POLICY_CKPT:-}"

# --- reference data ---------------------------------------------------------
REFERENCE_ARRAYS_DIR="${REFERENCE_ARRAYS_DIR:-${REPO_ROOT}/data/g1_bones_seed_sonic_129k_50hz_refarrays}"
REFERENCE_ARRAYS_WARM_WORKERS="${REFERENCE_ARRAYS_WARM_WORKERS:-8}"
PERSIST_ID="${PERSIST_ID:-bones_seed_sonic_full_129785@e714bbff}"
EXPECTED_MOTIONS="${EXPECTED_MOTIONS:-129785}"
EXPECTED_TRANSITIONS="${EXPECTED_TRANSITIONS:-47491234}"
ANCHOR_BODY="${ANCHOR_BODY:-pelvis}"
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

# --- output -----------------------------------------------------------------
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/qualitative_analysis/bones129k-${ABLATE_ARM_SLUG}/${TRACKER_ARM}}"

export TERM="${TERM:-xterm}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"

# --- gates ------------------------------------------------------------------
# Refuse an analysis that needs a discrete code, BEFORE anything launches. The
# Python entrypoint refuses too, but only after Isaac has started, which is a
# few minutes of startup to learn something knowable here.
qualitative_require_discrete_code() {
    [[ "${ABLATE_IS_DISCRETE}" == "1" ]] && return 0
    # $1 names what THIS analysis does with the code, so the refusal says why
    # the arm cannot answer it rather than describing some other mode.
    local what="${1:-edits one ${ABLATE_LATENT_MODE} code position at a time}"
    fail "this analysis ${what},
  but LATENT_ARM=${LATENT_ARM} has no discrete code: z is ${Z_DIM} continuous
  values straight out of the encoder trunk. There is no level to step, no
  category to swap and no alphabet to draw from, and no continuous analogue is
  defined here.
  Run it with LATENT_ARM=fsq64, or use one of the analyses that reads the
  latent itself: qualitative_reference_rollout.sh,
  qualitative_motion_switch_grid.sh, qualitative_latent_semantics.sh."
}

qualitative_check_data() {
    if [[ -f "${MANIFEST_PATH}" ]]; then
        local actual
        actual="$(sha256sum "${MANIFEST_PATH}" | awk '{print $1}')"
        [[ "${actual}" == "${EXPECTED_MANIFEST_SHA256}" ]] \
            || fail "manifest SHA mismatch: expected=${EXPECTED_MANIFEST_SHA256} actual=${actual}"
        echo "[PASS] manifest provenance: ${MANIFEST_PATH}"
    else
        echo "[NOTE] manifest absent (${MANIFEST_PATH}); relying on the arrays' own identity gate."
    fi

    [[ -d "${REFERENCE_ARRAYS_DIR}" ]] || fail "reference arrays not found: ${REFERENCE_ARRAYS_DIR}
  Fetch them with
    pixi run python -m imitation_experiments.data.publish_reference_arrays fetch \\
      --repo_id GeorgiaTech/g1_bones_seed_sonic_129k_50hz_refarrays \\
      --dest_dir ${REFERENCE_ARRAYS_DIR} --persist_id ${PERSIST_ID} \\
      --expected_motions ${EXPECTED_MOTIONS} --expected_transitions ${EXPECTED_TRANSITIONS}"

    # Size, sidecar, persist id, body list, motion and transition counts.
    # Seconds, no rebuild.
    pixi run python -m imitation_experiments.data.build_reference_arrays \
        --manifest "${MANIFEST_PATH}" \
        --output_dir "${REFERENCE_ARRAYS_DIR}" \
        --persist_id "${PERSIST_ID}" \
        --anchor_body "${ANCHOR_BODY}" \
        --body_names "${RUNTIME_BODY_NAMES[@]}" \
        --expected_motions "${EXPECTED_MOTIONS}" \
        --expected_transitions "${EXPECTED_TRANSITIONS}" \
        --validate_only
}

qualitative_check_encoder() {
    [[ -s "${ENCODER_CKPT}" ]] || fail "encoder checkpoint not found: ${ENCODER_CKPT}"

    # A z_dim, window-mode, or macro-state mismatch is silent until the command
    # space is already wrong, so assert the encoder's input width instead. Read
    # the trunk's first Linear by NAME: the quantized latent modes prepend a
    # `codebook` or quantizer parameter, so taking the first entry reads that
    # instead and reports a bogus mismatch.
    pixi run python -c '
import sys, torch
checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
expected = int(sys.argv[2])
state_dict = checkpoint["skill_encoder_state_dict"]
trunk = [
    (k, v) for k, v in state_dict.items()
    if k.startswith("net.") and k.endswith(".weight") and v.ndim == 2
]
if not trunk:
    raise SystemExit(
        "no trunk Linear weight (net.<i>.weight) in skill_encoder_state_dict; "
        f"keys: {list(state_dict)[:8]}"
    )
key, value = trunk[0]
if int(value.shape[1]) != expected:
    raise SystemExit(
        f"root_qpos encoder width mismatch: {key} {tuple(value.shape)}, expected {expected}"
    )
mode = checkpoint["config"]["latent_mode"]
expected_mode = sys.argv[3]
if mode != expected_mode:
    raise SystemExit(f"expected a {expected_mode} encoder, got {mode!r}")
if mode == "deterministic":
    # The trunk output IS the command: no codebook, no lattice. Assert the
    # absence of both, so a checkpoint whose config and weights disagree is
    # caught here rather than after Isaac has started.
    stray = [k for k in state_dict if "codebook" in k or "fsq" in k]
    if stray:
        raise SystemExit(
            f"deterministic checkpoint carries quantizer weights {stray}; its "
            "config and its weights disagree"
        )
    # The trunk must emit z itself. A last layer wider or narrower than z_dim
    # would mean a projection this arm does not have.
    z_dim = int(checkpoint["config"]["z_dim"])
    out_key, out_value = trunk[-1]
    if int(out_value.shape[0]) != z_dim:
        raise SystemExit(
            f"deterministic trunk emits {int(out_value.shape[0])} values but "
            f"z_dim is {z_dim}: {out_key} {tuple(out_value.shape)}"
        )
    print(
        f"[PASS] root_qpos deterministic encoder: {key} {tuple(value.shape)}, "
        f"continuous z_dim {z_dim} (no codebook, no lattice)"
    )
elif mode == "sonic_fsq":
    # No codebook: the lattice is the code space, and the published command is
    # the quantizer output, so z_dim must equal the coordinate count.
    levels = checkpoint["config"].get("sonic_fsq_levels")
    if not levels:
        raise SystemExit("sonic_fsq checkpoint carries no sonic_fsq_levels")
    z_dim = int(checkpoint["config"]["z_dim"])
    if len(set(int(v) for v in levels)) != 1:
        raise SystemExit(f"non-uniform FSQ levels: {sorted(set(levels))}")
    if z_dim != len(levels):
        raise SystemExit(f"sonic_fsq z_dim {z_dim} != {len(levels)} coordinates")
    print(
        f"[PASS] root_qpos sonic_fsq encoder: {key} {tuple(value.shape)}, "
        f"{len(levels)} coordinates x {int(levels[0])} levels"
    )
else:
    codebook = state_dict["codebook"]
    print(
        f"[PASS] root_qpos {mode} encoder: {key} {tuple(value.shape)}, "
        f"codebook {tuple(codebook.shape)}"
    )
' "${ENCODER_CKPT}" "${ENCODER_INPUT_WIDTH}" "${EXPECTED_LATENT_MODE}"
    ENCODER_SHA256="$(sha256sum "${ENCODER_CKPT}" | awk '{print $1}')"
    echo "[PASS] encoder sha256: ${ENCODER_SHA256}"
}

qualitative_resolve_policy() {
    if [[ -z "${POLICY_CKPT}" ]]; then
        [[ -d "${LOWLEVEL_RUN_DIR}" ]] || fail "low-level run directory not found: ${LOWLEVEL_RUN_DIR}"
        POLICY_CKPT="$(pixi run python -c '
import sys
sys.path.insert(0, "'"${SCRIPT_DIR}/src"'")
import qualitative_common as qc
print(qc.resolve_latest_policy_checkpoint(sys.argv[1]))
' "${LOWLEVEL_RUN_DIR}" | tail -1)"
    fi
    [[ -s "${POLICY_CKPT}" ]] || fail "low-level checkpoint not found: ${POLICY_CKPT}"
    POLICY_SHA256="$(sha256sum "${POLICY_CKPT}" | awk '{print $1}')"
    echo "[PASS] policy checkpoint: ${POLICY_CKPT}"
    echo "[PASS] policy sha256: ${POLICY_SHA256}"
}

# The Hydra override block every mode shares. Reference arrays only, root_qpos
# macro state, and the z_dim+2-wide actor command.
qualitative_base_overrides() {
    BASE_OVERRIDES=(
        "physics=${PHYSICS}"
        "${MACRO_STATE_TERMS_OVERRIDE}"
        env.data.manifest=null
        "env.data.reference_arrays_dir=${REFERENCE_ARRAYS_DIR}"
        "env.data.reference_arrays_warm_workers=${REFERENCE_ARRAYS_WARM_WORKERS}"
        env.data.runtime_cache_device=cpu
        "${RUNTIME_BODY_NAMES_OVERRIDE}"
        "env.data.macro_cache_device=${DEVICE}"
        "env.data.persist_id=${PERSIST_ID}"
        "env.command_interface.actor.dim=${LATENT_COMMAND_DIM}"
        "agent.ipmd.latent_dim=${LATENT_COMMAND_DIM}"
        "agent.ipmd.latent_learning.code_latent_dim=${Z_DIM}"
        agent.ipmd.latent_learning.command_phase_mode=sin_cos
        "agent.ipmd.latent_learning.code_period=${LATENT_HOLD_STEPS}"
        "agent.ipmd.latent_steps_min=${LATENT_HOLD_STEPS}"
        "agent.ipmd.latent_steps_max=${LATENT_HOLD_STEPS}"
    )
}

# --- tracker geometry -------------------------------------------------------
# Passed explicitly for both arms. The `sonic` arm's widths are not the entry
# point's default, so a strict restore fails without them; stating the `tuned`
# arm's widths too keeps the two launches symmetric and self-documenting.
ablate_base_overrides() {
    qualitative_base_overrides
    BASE_OVERRIDES+=(
        "agent.policy.num_cells=${ABLATE_TRACKER_CELLS}"
        agent.policy.activation_fn=silu
        "agent.value_function.num_cells=${ABLATE_TRACKER_CELLS}"
        agent.value_function.activation_fn=silu
    )
}

# --- GPU selection ----------------------------------------------------------
# Choose the GPUs to expose. Rank by UTILIZATION first, then memory: a GPU at
# 100% util with little memory allocated is still unusable here -- CUDA context
# creation on it times out, Omniverse then finds no matching CUDA device for any
# adapter, and Kit segfaults. Ranking by memory alone picks exactly those GPUs.
# Always returns at least MIN_GPUS entries because rendering needs >= 2.
qualitative_pick_gpus() {
    local min_gpus="${1:-2}" exclude="${2:-7}"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
        --format=csv,noheader,nounits \
    | awk -F', ' -v excl="${exclude}" '$1 != excl {printf "%s %s %s\n", $3, $2, $1}' \
    | sort -n -k1,1 -k2,2 \
    | awk '{print $3}' \
    | head -n "${min_gpus}" \
    | sort -n \
    | paste -sd,
}

# Least-utilized GPUs from the render-capable set. Use this for anything with
# VIDEO=1; qualitative_pick_gpus is fine for headless runs.
qualitative_pick_render_gpus() {
    local want="${1:-2}" list=""
    for idx in ${RENDER_CAPABLE_GPUS}; do list+="${idx}|"; done
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
        --format=csv,noheader,nounits \
    | awk -F', ' -v ok="|${list}" 'index(ok, "|"$1"|") {printf "%s %s %s\n", $3, $2, $1}' \
    | sort -n -k1,1 -k2,2 | awk '{print $3}' | head -n "${want}" | sort -n | paste -sd,
}

# Omniverse enumerates devices differently from CUDA. With exactly ONE visible
# GPU it cannot match its graphics device to a CUDA device, logs "Skipping NVIDIA
# graphics device", and segfaults during Kit startup. Rendering therefore needs
# at least two visible GPUs -- the extra one is only there to keep enumeration
# consistent and is not used for compute.
qualitative_require_render_gpus() {
    local count renderable=0 idx
    count="$(awk -F, '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")"
    for idx in ${CUDA_VISIBLE_DEVICES//,/ }; do
        case " ${RENDER_CAPABLE_GPUS} " in *" ${idx} "*) renderable=$((renderable + 1));; esac
    done
    if [[ "${count}" -lt 2 || "${renderable}" -lt 2 ]]; then
        fail "VIDEO=1 needs at least two RENDER-CAPABLE GPUs visible, but
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} has ${count} visible of which
  ${renderable} are render-capable (capable set: ${RENDER_CAPABLE_GPUS}).
  Otherwise Omniverse logs 'Skipping NVIDIA graphics device' and Kit segfaults
  a few seconds into startup -- idle GPUs outside the capable set do not help.
  Pick from the capable set, or run with VIDEO=0."
    fi
    echo "[PASS] ${count} visible GPUs (${renderable} render-capable): ${CUDA_VISIBLE_DEVICES}"
}

# --- launch helpers ---------------------------------------------------------
qualitative_run() {
    printf '  '; printf '%q ' "$@"; printf '\n'
    if [[ "${DRY_RUN}" == "1" ]]; then
        # Validate the entrypoint's argument parser. Printing a command is not
        # evidence it can run: a duplicate option definition raises at parser
        # construction and would otherwise only surface on the real launch.
        # `--help` exits inside argparse, before AppLauncher, so this is seconds
        # and starts no simulator.
        local entry=""
        for arg in "$@"; do
            case "${arg}" in *.py) entry="${arg}"; break;; esac
        done
        if [[ -n "${entry}" ]]; then
            # A duplicate option raises argparse.ArgumentError while the parser
            # is being BUILT, so it shows up as a traceback. Exit status is not
            # the signal here: --help falls through to "required arguments"
            # and exits non-zero even when the parser is perfectly healthy.
            local probe
            probe="$(pixi run -e isaaclab python "${entry}" --help 2>&1 || true)"
            if grep -qE "Traceback|ArgumentError|conflicting option" <<< "${probe}"; then
                echo "${probe}" | tail -5 >&2
                fail "entrypoint argument parser is broken: ${entry}
  Reproduce with: pixi run -e isaaclab python ${entry} --help"
            fi
            echo "[PASS] entrypoint arguments parse: $(basename "${entry}")"
        fi
        echo "[PLAN] DRY_RUN=1; launching nothing."
        return 0
    fi
    "$@"
}

# Do NOT trust the exit status of an Isaac entrypoint. `SimulationApp`'s
# shutdown path forces exit 0 even after an unhandled exception: the traceback
# is printed, "SimulationApp.close() was not called explicitly" follows, and the
# process reports success. A crashed run therefore looks fine to any caller, and
# a batch driver will happily march on. Verify the artifacts instead -- every
# mode writes provenance.json only after its real work has completed.
qualitative_require_output() {
    local directory="$1"
    if [[ "${DRY_RUN}" == "1" ]]; then
        return 0
    fi
    if [[ ! -s "${directory}/provenance.json" ]]; then
        local nested
        nested="$(find "${directory}" -mindepth 2 -maxdepth 2 -name provenance.json 2>/dev/null | head -1)"
        [[ -n "${nested}" ]] || fail "run produced no provenance.json under ${directory}
  The entrypoint exited without completing. Isaac's SimulationApp masks a
  crash as exit 0, so read the run log for the real traceback."
    fi
    echo "[PASS] artifacts present under ${directory}"
}
