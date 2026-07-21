#!/usr/bin/env bash
set -euo pipefail

# Resumable BONES-SEED-91 job at a 5B-frame cap, chained across multiple
# Slurm segments because every ICE GPU partition (ice-gpu, ice-bw-gpu,
# coc-gpu, coe-gpu, pace-gpu) hard-caps walltime at 16-18h (scontrol/sinfo
# confirmed 2026-07-21; not a QoS-configurable limit).
#
# TASK is Isaac-Imitation-G1-Latent-v0, which since the 2026-07-21 revert
# resolves to the strict/legacy-optimizer surface (the SONIC release surface
# is opt-in only via Isaac-Imitation-G1-Latent-Sonic-v0; the old
# Latent-Strict-v0 alias was removed on 2026-07-21 -- Latent-v0 always
# denotes the latest preferred surface). W&B run bn931wny
# (g1-lafan1-strict/ice3-l1-novideo) is the actual "L1" submission this run
# is meant to match: this surface + the legacy/local optimizer contract
# (512/256/128 ELU MLPs, lr 1e-3), which reached episode/length=244 and
# episode/return=13.1 -- far better than what the new SONIC release-optimizer
# contract has produced so far in the concurrent VRAM ablation. Latent-v0's
# kwargs route to the legacy-style G1ImitationLatentRLOptIPMDConfig (not the
# Sonic one), so selecting this task is sufficient; no separate
# policy-contract flag needed.
#
# Each invocation of this script submits exactly ONE segment: it inspects the
# central log tree (logs/rlopt/ipmd/<TASK>/<timestamp>/models/ -- the same
# location every other job on this task uses; no agent.logger.log_dir
# override) for the latest low-level checkpoint, and if one exists, resumes
# from it (--train-checkpoint) with
# --max_iterations reduced by the frames already trained (RLOpt's
# save_model/load_model restores weights + optimizer state but NOT the
# frame/iteration counter, so the remaining-iteration math must happen here).
# Re-invoke this script after each segment ends (success, crash, or walltime
# cutoff) until it reports FRAME_CAP reached and refuses to submit further.
#
# Uses the 91/100-motion SONIC-exclusion-filtered manifest at the scaled
# config validated by the VRAM ablation (2026-07-21): 12288 envs x 12 rollout
# steps (v2, the largest arm that fits one H100; 16384 envs and 12288x24 both
# OOM), minibatch = frames_per_batch / 8 = 18432, and njmax=320/nconmax=40
# (fixed per-step contact budget with headroom above the 288/32 that measured
# zero overflow on BONES-SEED; njmax does NOT scale with env count).
#
# RUN_TAG rotated on 2026-07-21 after the Newton joint-order fix
# (fix/migration, merged as PR #24 / 900c66c): every checkpoint trained
# before the fix encodes a Newton-specific joint permutation and is invalid,
# so the old nj288_nc32 tag's tree (segment 1 = cancelled job 5524342) must
# never be resumed. The fresh tag also retrains the skill encoder under the
# pinned causal planner frame.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

SEED="${SEED:-0}"
DRY_RUN="${DRY_RUN:-1}"
FRAME_CAP=5000000000
TRAIN_NUM_ENVS=12288
ROLLOUT_STEPS=12
MINIBATCH_SIZE=18432
NJMAX=320
NCONMAX=40
FRAMES_PER_BATCH=$((TRAIN_NUM_ENVS * ROLLOUT_STEPS))
RUN_TAG="${RUN_TAG:-bones_seed_91_strict_h25_z256_5b_seed${SEED}_20260721_jointfix_e12288_r12_nj320_nc40}"
MANIFEST_PATH="${MANIFEST_PATH:-/data/bones_seed_100/manifests/g1_bones_seed_100_sonic_filtered_manifest.json}"
DATASET_PATH="${DATASET_PATH:-/data/bones_seed_100/g1_hl_diffsr_sonic_filtered}"
TASK_NAME="Isaac-Imitation-G1-Latent-v0"
PRETRAIN_OUTPUT_DIR="logs/bones_seed_sonic/${RUN_TAG}/skill_encoder_h25_z256"
# Central log location every job on this task shares (no per-run log_dir
# override): logs/rlopt/ipmd/<TASK>/<timestamp>/models/. Resume detection
# below picks the newest checkpoint by mtime, which only correctly identifies
# "our" latest segment if no unrelated job trains this exact task
# concurrently -- unlike the old RUN_TAG-scoped log_dir, this does not
# self-isolate. Keep that in mind if another Latent-v0 job is ever run in
# parallel with this one.
CENTRAL_LOG_DIR="logs/rlopt/ipmd/${TASK_NAME}"
EXPECTED_MANIFEST_SHA256="${EXPECTED_MANIFEST_SHA256:-8d48750177efb3e9118c5d0ca14b69d62abedff16eb8c00585920a34bd87ee8d}"
EXPECTED_NPZ_COUNT="${EXPECTED_NPZ_COUNT:-100}"
REMOTE_PROJECT_ROOT="${REMOTE_PROJECT_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/isaaclab}"
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/data}"

local_manifest="${REPO_ROOT}/data/bones_seed_100/manifests/g1_bones_seed_100_sonic_filtered_manifest.json"
actual_local_sha="$(sha256sum "${local_manifest}" | awk '{print $1}')"
if [[ "${actual_local_sha}" != "${EXPECTED_MANIFEST_SHA256}" ]]; then
    echo "[ERROR] Local SONIC-filtered BONES-SEED-100 manifest hash mismatch." >&2
    exit 2
fi

case "${DRY_RUN}" in
    1|true|TRUE|yes|YES|on|ON) is_dry_run=1 ;;
    0|false|FALSE|no|NO|off|OFF) is_dry_run=0 ;;
    *)
        echo "[ERROR] DRY_RUN must be a boolean; got '${DRY_RUN}'." >&2
        exit 2
        ;;
esac

if [[ "${is_dry_run}" == "0" ]]; then
    actual_remote_sha="$(ssh -o BatchMode=yes -o ConnectTimeout=10 ice "sha256sum '${REMOTE_DATA_ROOT}/bones_seed_100/manifests/g1_bones_seed_100_sonic_filtered_manifest.json'" | awk '{print $1}')"
    remote_npz_count="$(ssh -o BatchMode=yes -o ConnectTimeout=10 ice "find '${REMOTE_DATA_ROOT}/bones_seed_100/npz/g1' -type f -name '*.npz' | wc -l")"
    if [[ "${actual_remote_sha}" != "${EXPECTED_MANIFEST_SHA256}" || "${remote_npz_count}" != "${EXPECTED_NPZ_COUNT}" ]]; then
        echo "[ERROR] ICE BONES-SEED-100 SONIC-filtered data gate failed: sha=${actual_remote_sha}, npz=${remote_npz_count}." >&2
        exit 2
    fi
fi

# --- Resume detection ---
# RLOpt's load_model restores weights + optimizer state but NOT the
# frame/iteration counter (base_class.py IterationMetadata.frames_processed
# defaults to 0 on every fresh `agent.train()` call), so each segment's own
# checkpoint filenames (model_step_<N>.pt) are LOCAL to that segment, not a
# global cumulative total. This script therefore tracks the true cumulative
# frame count itself in a state file (STATE_FILE), keyed by which checkpoint
# was last counted, and only credits a segment's contribution once:
#   1. Find the checkpoint file with the newest mtime anywhere under
#      TRAIN_LOG_DIR (the most recently finished segment's final save).
#   2. If it differs from the state file's last-counted path, that segment's
#      own contribution is the MAX step number among files in ITS OWN
#      directory (not a global max across segments), added to the running
#      cumulative total.
#   3. remaining = FRAME_CAP - cumulative; if <= 0, the job is done.
STATE_FILE="${REMOTE_PROJECT_ROOT}/logs/bones_seed_sonic/${RUN_TAG}/resume_state.tsv"
cumulative_frames=0
latest_checkpoint=""
if [[ "${is_dry_run}" == "0" ]]; then
    resume_state="$(ssh -o BatchMode=yes -o ConnectTimeout=10 ice bash -s -- \
        "${REMOTE_PROJECT_ROOT}/${CENTRAL_LOG_DIR}" "${STATE_FILE}" <<'REMOTE_EOF'
set -uo pipefail
train_log_dir="$1"
state_file="$2"

cumulative=0
last_counted=""
if [[ -f "${state_file}" ]]; then
    IFS=$'\t' read -r cumulative last_counted < "${state_file}"
fi

latest="$(find "${train_log_dir}" -name 'model_step_*.pt' -printf '%T@\t%p\n' 2>/dev/null \
    | sort -n -k1,1 | tail -1 | cut -f2-)"

if [[ -n "${latest}" && "${latest}" != "${last_counted}" ]]; then
    segment_dir="$(dirname "${latest}")"
    segment_frames="$(find "${segment_dir}" -name 'model_step_*.pt' \
        | sed -E 's#.*model_step_([0-9]+)\.pt#\1#' | sort -n | tail -1)"
    cumulative=$((cumulative + segment_frames))
    printf '%s\t%s\n' "${cumulative}" "${latest}" > "${state_file}"
    last_counted="${latest}"
fi

printf '%s\t%s\n' "${cumulative}" "${last_counted}"
REMOTE_EOF
    )"
    cumulative_frames="$(printf '%s' "${resume_state}" | cut -f1)"
    latest_checkpoint="$(printf '%s' "${resume_state}" | cut -f2-)"
    cumulative_frames="${cumulative_frames:-0}"
fi

if [[ -n "${latest_checkpoint}" ]]; then
    remaining_frames=$((FRAME_CAP - cumulative_frames))
    if (( remaining_frames <= 0 )); then
        echo "[INFO] ${RUN_TAG} already reached FRAME_CAP=${FRAME_CAP} (cumulative ${cumulative_frames} frames). Not submitting."
        exit 0
    fi
    max_iterations=$(( (remaining_frames + FRAMES_PER_BATCH - 1) / FRAMES_PER_BATCH ))
    skill_checkpoint="${PRETRAIN_OUTPUT_DIR}/checkpoints/latest.pt"
    echo "[INFO] Resuming ${RUN_TAG} from ${latest_checkpoint} (${cumulative_frames}/${FRAME_CAP} cumulative frames done; ${max_iterations} iterations remaining this segment)."
    extra_args=(
        --assert-kitless
        --skip-pretrain
        --pretrained-checkpoint "${skill_checkpoint}"
        --train-checkpoint "${latest_checkpoint}"
        --train-override "physics=newton_mjwarp"
        --train-override "env.sim.physics.solver_cfg.njmax=${NJMAX}"
        --train-override "env.sim.physics.solver_cfg.nconmax=${NCONMAX}"
        --train-override "env.refresh_zarr_dataset=false"
    )
else
    max_iterations=$(( (FRAME_CAP + FRAMES_PER_BATCH - 1) / FRAMES_PER_BATCH ))
    echo "[INFO] No existing checkpoint found for ${RUN_TAG}; submitting a fresh first segment (${max_iterations} iterations for ${FRAME_CAP} frames)."
    if [[ "${is_dry_run}" == "0" ]]; then
        remote_pretrain_output="${REMOTE_PROJECT_ROOT}/${PRETRAIN_OUTPUT_DIR}"
        if ssh -o BatchMode=yes -o ConnectTimeout=10 ice "test -e '${remote_pretrain_output}'"; then
            echo "[ERROR] Refusing to reuse existing ICE output with no resume checkpoint found: ${remote_pretrain_output}" >&2
            exit 2
        fi
    fi
    extra_args=(
        --assert-kitless
        --pretrain-output-dir "${PRETRAIN_OUTPUT_DIR}"
        --pretrain-override physics=newton_mjwarp
        --pretrain-override env.refresh_zarr_dataset=true
        --train-override physics=newton_mjwarp
        --train-override "env.sim.physics.solver_cfg.njmax=${NJMAX}"
        --train-override "env.sim.physics.solver_cfg.nconmax=${NCONMAX}"
        --train-override env.refresh_zarr_dataset=false
    )
fi
printf -v extra_args_string '%q ' "${extra_args[@]}"

export TASK="${TASK_NAME}"
export FRAME_CAP
export TRAIN_NUM_ENVS
export ROLLOUT_STEPS
export MINIBATCH_SIZE
export MAX_ITERATIONS="${max_iterations}"
export PRETRAIN_NUM_ENVS=16
export PRETRAIN_UPDATES=5000
export PRETRAIN_BATCH_SIZE=8192
export HORIZON_STEPS=25
export TRAIN_VIDEO=0
export SAVE_INTERVAL=100000000
export MANIFEST_PATH
export DATASET_PATH
export WANDB_PROJECT="${WANDB_PROJECT:-g1-bones-seed-100-sonic-latent-ice}"
export WANDB_GROUP="${WANDB_GROUP:-strict-e12288-5b-resumable-jointfix}"
export EXP_NAME="${EXP_NAME:-${RUN_TAG}_oracle_low_level}"
export CLUSTER_CONFIG=ice_runtime
export CLUSTER_SLURM_TIME_LIMIT=15:59:00
export CLUSTER_SLURM_PARTITION=ice-gpu
export CLUSTER_SLURM_QOS=coe-ice
export CLUSTER_SLURM_GPU_GRES=gpu:h100:1
export CLUSTER_SLURM_CPUS_PER_TASK=16
export CLUSTER_SLURM_MEM=96G
export CLUSTER_SLURM_JOB_NAME_PREFIX=bones-sonic-5b-resume
export CLUSTER_G1_USD_PATH=repo
export EXTRA_PIPELINE_ARGS="${extra_args_string}"
export DRY_RUN

exec "${REPO_ROOT}/experiments/submit_hl_skill_pipeline_pace_2b.sh"
