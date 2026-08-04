#!/usr/bin/env bash
set -euo pipefail

# enc380: the LATENT arm built from the 380-value root_qpos packet.
#
# WHY THIS ARM EXISTS
#   The packet-size ladder currently confounds two things. `root_qpos` (380
#   explicit values) tracks at 23.6 mm while `latent_skill` (258 values) tracks
#   at 30.5 mm -- but the latent encoder was trained on the *full-body* 670
#   packet, so "explicit vs latent" and "qpos+root content vs qpos+qvel+root
#   content" move together. enc380 holds the CONTENT fixed and moves only the
#   compression: the same 29 qpos + 9 root frame the `root_qpos` tracker
#   consumes explicitly is instead fed through a DiffSR skill encoder and
#   published as a 258-value latent command.
#
#       root_qpos      planner/oracle -> 380 explicit values -> rq tracker
#       enc380         planner/oracle -> 380 -> [encoder] -> z258 -> tracker
#       latent_skill   planner/oracle -> 670 -> [encoder] -> z258 -> tracker
#
#   enc380 vs root_qpos isolates compression at fixed content; enc380 vs
#   latent_skill isolates content at fixed compression.
#
# WHAT MAKES IT MATCHED
#   Everything except the encoder's input width is copied from the frozen
#   latent oracle `lafan1_latent_deterministic_5b_seed0` (see its command.txt
#   under logs/downloaded_checkpoints/): deterministic continuous z256 + sin_cos
#   phase = 258, h10 hold, encoder hidden dims 1024/512/512, 50k pretrain
#   updates at batch 8192, and the corrected 40-motion LAFAN1 tree. The tracker
#   geometry (12288 x 12, minibatch 18432, lr 1e-3, 5B cap) is the same H100
#   point the `root_qpos` and `root_points5` arms ran at.
#
#   TASK ID: `Isaac-Imitation-G1-Latent-Strict-v0`, deliberately NOT
#   `Isaac-Imitation-G1-Latent-v0`. On 2026-07-27 that default id was re-pointed
#   at the new Stable surface; every latent row this arm must be comparable to
#   ran on the strict surface, which now lives only behind the explicit id.
#
# TWO STAGES, ON PURPOSE
#   stage 1 `pretrain`  --pretrain-only, ~50k updates (about 20 min of compute).
#                       Writes the encoder to the SHARED /data bind, not to the
#                       per-submission workspace logs, so every later training
#                       segment loads the byte-identical encoder.
#   stage 2 `train`     --skip-pretrain against that exact checkpoint, chained
#                       across ~16h segments until FRAME_CAP.
#
#   Folding the pretrain into segment 1 (the way the latent-ablation launcher
#   does) is what this split exists to avoid: an ICE TIMEOUT wipes node-local
#   output before the log sync, so a resumed segment could silently re-pretrain
#   a DIFFERENT encoder and continue a tracker into a latent space it was never
#   trained on. Nothing downstream would error. Keep the stages separate.
#
# Usage:
#   DRY_RUN=1 experiments/campaigns/2026-07-23-lafan1-planner-capacity/submit_enc380_latent_low_level_ice.sh
#   DRY_RUN=0 STAGE=pretrain ...      # once
#   DRY_RUN=0 STAGE=train    ...      # re-run for each next segment

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
while [ ! -x "${REPO_ROOT}/docker/cluster/cluster_interface.sh" ]; do
    if [ "${REPO_ROOT}" = "/" ]; then
        echo "[ERROR] Could not locate the repository root above ${SCRIPT_DIR}." >&2
        exit 2
    fi
    REPO_ROOT="$(dirname "${REPO_ROOT}")"
done
cd "${REPO_ROOT}"

DRY_RUN="${DRY_RUN:-1}"
STAGE="${STAGE:-auto}"
SEED="${SEED:-0}"
FRAME_CAP="${FRAME_CAP:-5000000000}"
TRAIN_NUM_ENVS=12288
ROLLOUT_STEPS=12
MINIBATCH_SIZE=18432
NJMAX=320
NCONMAX=40
FRAMES_PER_BATCH=$((TRAIN_NUM_ENVS * ROLLOUT_STEPS))
SAVE_INTERVAL="${SAVE_INTERVAL:-100000000}"

if [[ ! "${FRAME_CAP}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] FRAME_CAP must be a positive integer; got '${FRAME_CAP}'." >&2
    exit 2
fi

# Matched to the frozen latent oracle's encoder recipe.
Z_DIM=256
HORIZON_STEPS=10
PRETRAIN_UPDATES="${PRETRAIN_UPDATES:-50000}"
PRETRAIN_BATCH_SIZE=8192
PRETRAIN_NUM_ENVS=16
# 29 joint positions + anchor pos 3 + anchor rot6d 6 = 38/frame -> 380 over the
# 10-frame window. Byte-identical to the root_qpos packet, which is the point.
MACRO_TERMS='[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b]'

# Per-segment iteration cap so a segment finishes BEFORE the wall instead of
# being TIMEOUT-killed. ICE TIMEOUT is a hard SIGKILL: the final save never
# happens and everything since the last save_interval boundary is lost
# (segment 1 of the 2026-07-27 reduced-interface arms lost ~100M frames each).
#
# 76000, NOT the 80000 the explicit reduced-interface launcher uses. This arm is
# a LATENT one: every publish runs the frozen encoder forward and builds the
# macro state, which the explicit arms never pay for. Measured on job 5546958
# (h100, 12288 x 12): 76.9-77.6k fps sustained, 76.8k averaged over the first
# 8h55m including boot, against the 80.6-86.7k the explicit arms hit at the same
# geometry. Sizing that segment at 80000 overran the wall by ~15 min. Do not
# copy the explicit arms' number back over this one.
SEGMENT_FPS="${SEGMENT_FPS:-76000}"
SEGMENT_WALL_S="${SEGMENT_WALL_S:-57540}"      # 15:59:00
SEGMENT_STARTUP_S="${SEGMENT_STARTUP_S:-900}"  # Isaac boot + data load
SEGMENT_TAIL_S="${SEGMENT_TAIL_S:-600}"        # final save + log sync
SEGMENT_MAX_ITERATIONS=$((
    (SEGMENT_WALL_S - SEGMENT_STARTUP_S - SEGMENT_TAIL_S) * SEGMENT_FPS / FRAMES_PER_BATCH
))

TASK_NAME="${TASK_NAME:-Isaac-Imitation-G1-Latent-Strict-v0}"
RUN_TAG="${RUN_TAG:-lafan1_enc380_rootqpos_h10_z${Z_DIM}_seed${SEED}}"
MANIFEST_PATH="${MANIFEST_PATH:-/data/lafan1_corrected_8e95d557/manifests/g1_lafan1_manifest.json}"
DATASET_PATH="${DATASET_PATH:-/data/lafan1_corrected_8e95d557/g1_hl_diffsr}"
EXPECTED_MANIFEST_SHA256="${EXPECTED_MANIFEST_SHA256:-d972c37c41dadbb68c30fc456a9dc9c1bd6d30ed0b7aa9d34b1797472c945db8}"
EXPECTED_NPZ_COUNT="${EXPECTED_NPZ_COUNT:-40}"
REMOTE_PROJECT_ROOT="${REMOTE_PROJECT_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/isaaclab}"
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/data}"
REMOTE_ICE_BASE="$(dirname "${REMOTE_PROJECT_ROOT}")"
WANDB_PROJECT="${WANDB_PROJECT:-g1-lafan1-strict}"
WANDB_GROUP="${WANDB_GROUP:-enc380-interface-e12288-5b-resumable}"

# The encoder lives on the shared /data bind, NOT in per-submission workspace
# logs, because every training segment runs in a different workspace.
ENCODER_DIR_CONTAINER="/data/enc380_store/${RUN_TAG}/skill_encoder"
ENCODER_CKPT_CONTAINER="${ENCODER_DIR_CONTAINER}/checkpoints/latest.pt"
ENCODER_CKPT_REMOTE="${REMOTE_DATA_ROOT}/enc380_store/${RUN_TAG}/skill_encoder/checkpoints/latest.pt"
RESUME_CKPT_CONTAINER="/data/resume_store/${RUN_TAG}/model_resume.pt"

case "${DRY_RUN}" in
    1|true|TRUE|yes|YES|on|ON) is_dry_run=1 ;;
    0|false|FALSE|no|NO|off|OFF) is_dry_run=0 ;;
    *) echo "[ERROR] DRY_RUN must be a boolean; got '${DRY_RUN}'." >&2; exit 2 ;;
esac

# Captured once, before any stage exports its own default. Stages must not read
# CLUSTER_SLURM_TIME_LIMIT with `:-`, or a dry run that prints both stages would
# leak the pretrain stage's 2h wall into the training segment's iteration cap.
TIME_LIMIT_OVERRIDE="${CLUSTER_SLURM_TIME_LIMIT:-}"
PRETRAIN_TIME_LIMIT="${TIME_LIMIT_OVERRIDE:-02:00:00}"
TRAIN_TIME_LIMIT="${TIME_LIMIT_OVERRIDE:-15:59:00}"

ssh_ice() {
    ssh -o BatchMode=yes -o ConnectTimeout=10 ice "$@"
}

check_data_gate() {
    local actual_sha remote_npz_count
    actual_sha="$(ssh_ice "sha256sum '${REMOTE_DATA_ROOT}/lafan1_corrected_8e95d557/manifests/g1_lafan1_manifest.json'" | awk '{print $1}')"
    remote_npz_count="$(ssh_ice "find '${REMOTE_DATA_ROOT}/lafan1_corrected_8e95d557' -type f -name '*.npz' | wc -l")"
    if [[ "${actual_sha}" != "${EXPECTED_MANIFEST_SHA256}" || "${remote_npz_count}" != "${EXPECTED_NPZ_COUNT}" ]]; then
        echo "[ERROR] ICE corrected-LAFAN1 data gate failed: sha=${actual_sha}, npz=${remote_npz_count}." >&2
        exit 2
    fi
    echo "[PASS] ICE corrected-LAFAN1 manifest and NPZ count match the frozen protocol."
}

encoder_exists_remote() {
    ssh_ice "test -f '${ENCODER_CKPT_REMOTE}'"
}

common_cluster_env() {
    export CLUSTER_LOGIN="${CLUSTER_LOGIN:-login-ice.pace.gatech.edu}"
    export CLUSTER_SLURM_SUBMIT_SCRIPT="${CLUSTER_SLURM_SUBMIT_SCRIPT:-pace}"
    export CLUSTER_PYTHON_EXECUTABLE="scripts/rlopt/train_hl_skill_pipeline.py"
    export CLUSTER_APPEND_DEFAULT_G1_MANIFEST=0
    export CLUSTER_G1_MANIFEST_REFRESH_POLICY="${CLUSTER_G1_MANIFEST_REFRESH_POLICY:-auto}"
    export CLUSTER_SLURM_PARTITION="${CLUSTER_SLURM_PARTITION:-ice-gpu}"
    export CLUSTER_SLURM_QOS="${CLUSTER_SLURM_QOS:-coe-ice}"
    export CLUSTER_SLURM_GPU_GRES="${CLUSTER_SLURM_GPU_GRES:-gpu:h100:1}"
    export CLUSTER_SLURM_CPUS_PER_TASK=16
    export CLUSTER_SLURM_MEM=96G
    export CLUSTER_GIT_SYNC_FIRST="${CLUSTER_GIT_SYNC_FIRST:-0}"
    export CLUSTER_G1_USD_PATH=repo
}

run_or_print() {
    printf "[CMD] "
    printf "%q " "$@"
    printf "\n"
    if [[ "${is_dry_run}" == "1" ]]; then
        echo "[INFO] DRY_RUN=${DRY_RUN}; not contacting the cluster."
        return 0
    fi
    "$@"
}

submit_pretrain() {
    if [[ "${is_dry_run}" == "0" ]] && encoder_exists_remote; then
        echo "[ERROR] An encoder already exists at ${ENCODER_CKPT_REMOTE}." >&2
        echo "[ERROR] Re-running the pretrain would overwrite the encoder that live" >&2
        echo "[ERROR] tracker segments are frozen against. Delete it deliberately," >&2
        echo "[ERROR] or use a different RUN_TAG, if a fresh encoder is intended." >&2
        exit 2
    fi

    common_cluster_env
    export CLUSTER_SLURM_TIME_LIMIT="${PRETRAIN_TIME_LIMIT}"
    export CLUSTER_SLURM_JOB_NAME_PREFIX="lafan1-enc380-pretrain"

    local cmd=(./docker/cluster/cluster_interface.sh -c ice_runtime job
        --task "${TASK_NAME}"
        --seed "${SEED}"
        --headless
        --assert-kitless
        --app-arg=--kit_args=--/app/extensions/fsWatcherEnabled=false
        --manifest-path "${MANIFEST_PATH}"
        --dataset-path "${DATASET_PATH}"
        --pretrain-only
        --pretrain-output-dir "${ENCODER_DIR_CONTAINER}"
        --pretrain-num-envs "${PRETRAIN_NUM_ENVS}"
        --pretrain-updates "${PRETRAIN_UPDATES}"
        --pretrain-batch-size "${PRETRAIN_BATCH_SIZE}"
        --horizon-steps "${HORIZON_STEPS}"
        --z-dim "${Z_DIM}"
        --latent-mode deterministic
        --encoder-hidden-dims 1024 512 512
        --phase-mode sin_cos
        --latent-hold-steps "${HORIZON_STEPS}"
        --exp-name "${RUN_TAG}"
        --pretrain-override physics=newton_mjwarp
        # MUST stay false: /data/.../g1_hl_diffsr is shared with every other
        # LAFAN1 arm. A refresh=true job rebuilds it underneath them (2026-07-26:
        # four of seven groupvq arms died and the cache truncated to 56 KB).
        --pretrain-override env.refresh_zarr_dataset=false
        --pretrain-override "env.expert_macro_state_terms=${MACRO_TERMS}"
    )

    echo "[INFO] Stage pretrain: run_tag='${RUN_TAG}' updates='${PRETRAIN_UPDATES}' macro_terms='${MACRO_TERMS}'"
    echo "[INFO] Encoder will be written to (container) ${ENCODER_CKPT_CONTAINER}"
    echo "[INFO]                            (login)     ${ENCODER_CKPT_REMOTE}"
    run_or_print "${cmd[@]}"
}

submit_train_segment() {
    if [[ "${is_dry_run}" == "0" ]] && ! encoder_exists_remote; then
        echo "[ERROR] No enc380 encoder at ${ENCODER_CKPT_REMOTE}." >&2
        echo "[ERROR] Run STAGE=pretrain first and let it finish." >&2
        exit 2
    fi

    # --- Resume detection + checkpoint staging ---
    # Adapted from submit_reduced_interface_low_level_ice.sh. The durable copy
    # must live under the /data bind: ICE TIMEOUT wipes node-local $TMPDIR
    # before the log sync runs.
    local cumulative_frames=0
    local latest_checkpoint=""
    if [[ "${is_dry_run}" == "0" ]]; then
        local resume_state
        resume_state="$(ssh -o BatchMode=yes -o ConnectTimeout=10 ice bash -s -- \
            "${REMOTE_ICE_BASE}" "${TASK_NAME}" "${RUN_TAG}" "${REMOTE_DATA_ROOT}" <<'REMOTE_EOF'
set -uo pipefail
ice_base="$1"
task_name="$2"
run_tag="$3"
data_root="$4"
resume_store="${data_root}/resume_store/${run_tag}"
state_file="${resume_store}/resume_state.tsv"

cumulative=0
last_counted=""
if [[ -f "${state_file}" ]]; then
    IFS=$'\t' read -r cumulative last_counted < "${state_file}"
fi

# Scan every per-submission workspace; keep only run dirs whose recorded
# command carries THIS run's exp_name.
run_dirs="$(grep -ls -- "agent.logger.exp_name=${run_tag}" \
    "${ice_base}"/isaaclab*/logs/rlopt/ipmd/"${task_name}"/*/command.txt 2>/dev/null \
    | xargs -r -n1 dirname)"
latest=""
if [[ -n "${run_dirs}" ]]; then
    latest="$(find ${run_dirs} -name 'model_step_*.pt' -printf '%T@\t%p\n' 2>/dev/null \
        | sort -n -k1,1 | tail -1 | cut -f2-)"
fi

if [[ -n "${latest}" && "${latest}" != "${last_counted}" ]]; then
    segment_dir="$(dirname "${latest}")"
    segment_frames="$(find "${segment_dir}" -name 'model_step_*.pt' \
        | sed -E 's#.*model_step_([0-9]+)\.pt#\1#' | sort -n | tail -1)"
    cumulative=$((cumulative + segment_frames))
    mkdir -p "${resume_store}"
    printf '%s\t%s\n' "${cumulative}" "${latest}" > "${state_file}"
fi

if [[ -n "${latest}" ]]; then
    mkdir -p "${resume_store}"
    cp -f "${latest}" "${resume_store}/model_resume.pt"
fi

printf '%s\t%s\n' "${cumulative}" "${latest}"
REMOTE_EOF
        )"
        cumulative_frames="$(printf '%s' "${resume_state}" | cut -f1)"
        latest_checkpoint="$(printf '%s' "${resume_state}" | cut -f2-)"
        cumulative_frames="${cumulative_frames:-0}"
    fi

    local max_iterations
    local checkpoint_args=()
    if [[ -n "${latest_checkpoint}" ]]; then
        local remaining_frames=$((FRAME_CAP - cumulative_frames))
        if (( remaining_frames <= 0 )); then
            echo "[INFO] ${RUN_TAG} already reached FRAME_CAP=${FRAME_CAP} (cumulative ${cumulative_frames} frames). Not submitting."
            return 0
        fi
        max_iterations=$(( (remaining_frames + FRAMES_PER_BATCH - 1) / FRAMES_PER_BATCH ))
        echo "[INFO] Resuming ${RUN_TAG} from ${latest_checkpoint} (${cumulative_frames}/${FRAME_CAP} cumulative frames done; ${max_iterations} iterations remaining)."
        checkpoint_args=(--train-checkpoint "${RESUME_CKPT_CONTAINER}")
    else
        max_iterations=$(( (FRAME_CAP + FRAMES_PER_BATCH - 1) / FRAMES_PER_BATCH ))
        echo "[INFO] No tracker checkpoint found for ${RUN_TAG}; submitting a fresh first segment (${max_iterations} iterations for ${FRAME_CAP} frames)."
    fi

    if (( max_iterations > SEGMENT_MAX_ITERATIONS )); then
        echo "[INFO] Capping this segment at ${SEGMENT_MAX_ITERATIONS} iterations (~$((SEGMENT_MAX_ITERATIONS * FRAMES_PER_BATCH)) frames) so it finishes before the ${TRAIN_TIME_LIMIT} wall; ${max_iterations} were remaining. Re-run this script for the next segment."
        max_iterations="${SEGMENT_MAX_ITERATIONS}"
    fi

    common_cluster_env
    export CLUSTER_SLURM_TIME_LIMIT="${TRAIN_TIME_LIMIT}"
    export CLUSTER_SLURM_JOB_NAME_PREFIX="lafan1-enc380-5b-resume"

    local cmd=(./docker/cluster/cluster_interface.sh -c ice_runtime job
        --task "${TASK_NAME}"
        --seed "${SEED}"
        --headless
        --assert-kitless
        --app-arg=--kit_args=--/app/extensions/fsWatcherEnabled=false
        --manifest-path "${MANIFEST_PATH}"
        --dataset-path "${DATASET_PATH}"
        --skip-pretrain
        --pretrained-checkpoint "${ENCODER_CKPT_CONTAINER}"
        --horizon-steps "${HORIZON_STEPS}"
        --phase-mode sin_cos
        --latent-hold-steps "${HORIZON_STEPS}"
        --train-num-envs "${TRAIN_NUM_ENVS}"
        --train-max-iterations "${max_iterations}"
        --no-train-video
        --save-interval "${SAVE_INTERVAL}"
        --logger-backend wandb
        --wandb-project "${WANDB_PROJECT}"
        --wandb-group "${WANDB_GROUP}"
        --exp-name "${RUN_TAG}"
    )
    if (( ${#checkpoint_args[@]} > 0 )); then
        cmd+=("${checkpoint_args[@]}")
    fi
    cmd+=(
        --train-override physics=newton_mjwarp
        --train-override "agent.collector.frames_per_batch=${ROLLOUT_STEPS}"
        --train-override "agent.loss.mini_batch_size=${MINIBATCH_SIZE}"
        --train-override agent.ipmd.actor_learning_rate=1.0e-3
        --train-override agent.ipmd.critic_learning_rate=1.0e-3
        --train-override agent.optim.max_lr=1.0e-3
        --train-override "env.sim.physics.solver_cfg.njmax=${NJMAX}"
        --train-override "env.sim.physics.solver_cfg.nconmax=${NCONMAX}"
        --train-override env.refresh_zarr_dataset=false
        # The tracker's environment must build the SAME 380-wide macro state the
        # encoder was fit on. Without this the env would hand a frozen 380-input
        # encoder the default 670-wide full-body frame.
        --train-override "env.expert_macro_state_terms=${MACRO_TERMS}"
    )

    echo "[INFO] Stage train: task='${TASK_NAME}' run_tag='${RUN_TAG}' max_iterations='${max_iterations}' frames_per_batch='${FRAMES_PER_BATCH}' save_interval='${SAVE_INTERVAL}'"
    echo "[INFO] Frozen encoder: ${ENCODER_CKPT_CONTAINER}"
    run_or_print "${cmd[@]}"
}

if [[ "${is_dry_run}" == "0" ]]; then
    check_data_gate
fi

resolved_stage="${STAGE}"
if [[ "${resolved_stage}" == "auto" ]]; then
    if [[ "${is_dry_run}" == "1" ]]; then
        echo "[INFO] STAGE=auto under DRY_RUN prints BOTH stages; a real run picks one."
        resolved_stage="both"
    elif encoder_exists_remote; then
        resolved_stage="train"
    else
        resolved_stage="pretrain"
    fi
fi

case "${resolved_stage}" in
    pretrain) submit_pretrain ;;
    train) submit_train_segment ;;
    both)
        submit_pretrain
        echo
        submit_train_segment
        ;;
    *)
        echo "[ERROR] STAGE must be auto, pretrain, or train; got '${STAGE}'." >&2
        exit 2
        ;;
esac
