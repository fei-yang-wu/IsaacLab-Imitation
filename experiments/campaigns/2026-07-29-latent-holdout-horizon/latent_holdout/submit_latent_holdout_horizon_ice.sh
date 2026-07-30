#!/usr/bin/env bash
set -euo pipefail

# Latent hold-out horizon ablation on corrected LAFAN1 (ICE).
#
# Question: with the latent representation held EXACTLY fixed, how much does the
# command *interface* matter? Every arm is frozen against one shared h10 DiffSR
# encoder checkpoint; the only thing that moves is how many 50 Hz control steps
# one published latent is held for.
#
#   hold=10  control  1 latent per 200 ms  (the 5525664 run; curve only, see README)
#   hold=5   arm      2 latents per 200 ms
#   hold=1   arm     10 latents per 200 ms
#
# This is the GR00T-style "predict H, execute k" axis: the encoder always
# summarizes a 10-step future window (`hl_skill_horizon_steps=10`, which is
# checkpoint-bound and therefore identical across arms), but only the first
# `hold` steps of that window are consumed before a fresh latent is published.
# At deployment the planner emits a chunk of 10/hold latents per 200 ms, so the
# publication rate stays 5 Hz and only the output bandwidth moves.
#
# The phase clock must move with the hold. `phase_period` is fed from
# `latent_learning.code_period` (RLOpt/rlopt/agent/ipmd/ipmd.py:1248,1315) and
# the sampler computes phase as (phase_period - latent_steps)/phase_period
# (hl_skill_diffsr.py:1078). Setting code_period=hold keeps the control's
# semantics exactly -- "fraction of my current command's hold elapsed", sweeping
# 0 -> (hold-1)/hold. Leaving code_period at 10 while hold=5 would instead emit
# 0.5..0.9 and silently desynchronize the clock. Command width stays 258
# (256 code + 2 phase) for every arm, so the observation contract never moves.
#
# Expectation setting: the 2026-07-22 controlled isolation
# (wiki/ablation-experiment-plan.md:331-347) ran hold=1 against a frozen h10
# encoder and it collapsed (ep_len 2.76 vs 46.38 for hold=10 at 30M frames).
# That repro also switched phase-mode to none, so it confounded hold with
# command width; this campaign removes that confound. A collapse here is a
# legitimate, citable negative result, not a bug to tune away.
#
# Default DRY_RUN=1. One 16h segment per arm, no resume chain (user-decided
# 2026-07-29): whatever frames the segment reaches is the result.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

DRY_RUN="${DRY_RUN:-1}"
ARMS="${ARMS:-5 1}"
SEED="${SEED:-0}"

# --- Geometry: identical to the 5525664 control (e12288_r12_nj320_nc40) -------
TRAIN_NUM_ENVS="${TRAIN_NUM_ENVS:-12288}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-12}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-18432}"
NJMAX="${NJMAX:-320}"
NCONMAX="${NCONMAX:-40}"
FRAME_CAP="${FRAME_CAP:-5000000000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100000000}"
FRAMES_PER_BATCH=$((TRAIN_NUM_ENVS * ROLLOUT_STEPS))

# ICE TIMEOUT is a hard SIGKILL: the final save never runs and everything since
# the last save_interval boundary is lost. Size the segment to exit cleanly just
# under the wall instead. Latent arms carry the macro-state encode cost and
# sustain ~76-77k fps at this geometry (measured on job 5546958); the explicit
# arms' 80k+ does NOT apply here and sizing at 80000 overran the wall by ~15min.
SEGMENT_FPS="${SEGMENT_FPS:-76000}"
SEGMENT_WALL_S="${SEGMENT_WALL_S:-57540}"      # 15:59:00
SEGMENT_STARTUP_S="${SEGMENT_STARTUP_S:-900}"  # Isaac boot + data load
SEGMENT_TAIL_S="${SEGMENT_TAIL_S:-600}"        # final save + log sync
SEGMENT_MAX_ITERATIONS=$((
    (SEGMENT_WALL_S - SEGMENT_STARTUP_S - SEGMENT_TAIL_S) * SEGMENT_FPS / FRAMES_PER_BATCH
))

# --- Frozen protocol surface -------------------------------------------------
# After the 2026-07-27 task repoint, `Isaac-Imitation-G1-Latent-v0` resolves to
# the Stable/SONIC reset-sampling surface. The control, its encoder, Study B and
# Study C all ran on the plain strict surface, which is now reachable only under
# this explicit id. Using the bare v0 id here would confound hold with the
# reset-sampling change.
TASK_NAME="${TASK_NAME:-Isaac-Imitation-G1-Latent-Strict-v0}"

MANIFEST_PATH="${MANIFEST_PATH:-/data/lafan1_corrected_8e95d557/manifests/g1_lafan1_manifest.json}"
DATASET_PATH="${DATASET_PATH:-/data/lafan1_corrected_8e95d557/g1_hl_diffsr}"
EXPECTED_MANIFEST_SHA256="${EXPECTED_MANIFEST_SHA256:-d972c37c41dadbb68c30fc456a9dc9c1bd6d30ed0b7aa9d34b1797472c945db8}"
EXPECTED_NPZ_COUNT="${EXPECTED_NPZ_COUNT:-40}"

REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/data}"

# The one asset every arm shares. This is the surviving encoder from the 5B h10
# control run (5525664); its low-level policy checkpoints were lost to a TIMEOUT
# but the encoder persisted on the /data bind. Reusing this exact file is what
# makes the ablation single-variable: same latent space, different interface.
CONTROL_RUN_TAG="lafan1_strict_h10_z256_5b_seed0_20260721_jointfix_nocur_e12288_r12_nj320_nc40"
ENCODER_CKPT_CONTAINER="${ENCODER_CKPT_CONTAINER:-/data/pretrain_store/${CONTROL_RUN_TAG}/checkpoints/latest.pt}"
ENCODER_CKPT_REMOTE="${REMOTE_DATA_ROOT}/pretrain_store/${CONTROL_RUN_TAG}/checkpoints/latest.pt"

HORIZON_STEPS="${HORIZON_STEPS:-10}"   # encoder window; checkpoint-bound, never moves
Z_DIM="${Z_DIM:-256}"
LATENT_COMMAND_DIM=$((Z_DIM + 2))      # + sin/cos phase

WANDB_PROJECT="${WANDB_PROJECT:-g1-lafan1-strict}"
WANDB_GROUP="${WANDB_GROUP:-holdout-horizon-h10enc-e12288-16h}"

case "${DRY_RUN}" in
    1|true|TRUE|yes|YES|on|ON) is_dry_run=1 ;;
    0|false|FALSE|no|NO|off|OFF) is_dry_run=0 ;;
    *) echo "[ERROR] DRY_RUN must be a boolean; got '${DRY_RUN}'." >&2; exit 2 ;;
esac

ssh_ice() {
    ssh -o BatchMode=yes -o ConnectTimeout=10 ice "$@"
}

check_gates() {
    local actual_sha remote_npz_count
    actual_sha="$(ssh_ice "sha256sum '${REMOTE_DATA_ROOT}/lafan1_corrected_8e95d557/manifests/g1_lafan1_manifest.json'" | awk '{print $1}')"
    remote_npz_count="$(ssh_ice "find '${REMOTE_DATA_ROOT}/lafan1_corrected_8e95d557' -type f -name '*.npz' | wc -l")"
    if [[ "${actual_sha}" != "${EXPECTED_MANIFEST_SHA256}" || "${remote_npz_count}" != "${EXPECTED_NPZ_COUNT}" ]]; then
        echo "[ERROR] ICE corrected-LAFAN1 data gate failed: sha=${actual_sha}, npz=${remote_npz_count}." >&2
        exit 2
    fi
    echo "[PASS] corrected-LAFAN1 manifest sha and NPZ count match the frozen protocol."

    if ! ssh_ice "test -f '${ENCODER_CKPT_REMOTE}'"; then
        echo "[ERROR] Shared encoder missing at ${ENCODER_CKPT_REMOTE}." >&2
        echo "[ERROR] Every arm must be frozen against this exact file; refusing to submit." >&2
        exit 2
    fi
    # Record the encoder identity so the arms are provably frozen against one
    # latent space. Both arms must report the same digest in their job logs.
    ENCODER_SHA256="$(ssh_ice "sha256sum '${ENCODER_CKPT_REMOTE}'" | awk '{print $1}')"
    echo "[PASS] shared h10 encoder present: ${ENCODER_CKPT_REMOTE}"
    echo "[INFO] shared encoder sha256=${ENCODER_SHA256}"
}

submit_arm() {
    local hold="$1"

    if (( hold < 1 || hold > HORIZON_STEPS )); then
        echo "[ERROR] hold=${hold} must satisfy 1 <= hold <= HORIZON_STEPS=${HORIZON_STEPS}." >&2
        exit 2
    fi

    local run_tag="lafan1_holdout${hold}_h${HORIZON_STEPS}enc_z${Z_DIM}_seed${SEED}_e${TRAIN_NUM_ENVS}_r${ROLLOUT_STEPS}_nj${NJMAX}_nc${NCONMAX}"
    local log_dir="/data/holdout_store/${run_tag}/rlopt_train"

    local max_iterations=$(( (FRAME_CAP + FRAMES_PER_BATCH - 1) / FRAMES_PER_BATCH ))
    if (( max_iterations > SEGMENT_MAX_ITERATIONS )); then
        max_iterations="${SEGMENT_MAX_ITERATIONS}"
    fi

    export CLUSTER_LOGIN="${CLUSTER_LOGIN:-login-ice.pace.gatech.edu}"
    export CLUSTER_SLURM_SUBMIT_SCRIPT="${CLUSTER_SLURM_SUBMIT_SCRIPT:-pace}"
    export CLUSTER_PYTHON_EXECUTABLE="scripts/rlopt/train.py"
    export CLUSTER_APPEND_DEFAULT_G1_MANIFEST=0
    export CLUSTER_G1_MANIFEST_REFRESH_POLICY="${CLUSTER_G1_MANIFEST_REFRESH_POLICY:-auto}"
    export CLUSTER_SLURM_TIME_LIMIT="${CLUSTER_SLURM_TIME_LIMIT:-15:59:00}"
    export CLUSTER_SLURM_PARTITION="${CLUSTER_SLURM_PARTITION:-ice-gpu}"
    export CLUSTER_SLURM_QOS="${CLUSTER_SLURM_QOS:-coe-ice}"
    export CLUSTER_SLURM_GPU_GRES="${CLUSTER_SLURM_GPU_GRES:-gpu:h100:1}"
    export CLUSTER_SLURM_CPUS_PER_TASK=16
    export CLUSTER_SLURM_MEM=96G
    export CLUSTER_SLURM_JOB_NAME_PREFIX="lafan1-holdout${hold}"
    export CLUSTER_SLURM_EXCLUDE="${EXCLUDE_NODES:-atl1-1-03-010-15-0}"
    export CLUSTER_GIT_SYNC_FIRST="${CLUSTER_GIT_SYNC_FIRST:-0}"
    export CLUSTER_G1_USD_PATH=repo

    local cmd=(./docker/cluster/cluster_interface.sh -c ice_runtime job
        --task "${TASK_NAME}"
        --num_envs "${TRAIN_NUM_ENVS}"
        --headless
        --assert-kitless
        --algo IPMD
        --seed "${SEED}"
        --max_iterations "${max_iterations}"
        --kit_args=--/app/extensions/fsWatcherEnabled=false
        physics=newton_mjwarp
        "env.sim.physics.solver_cfg.njmax=${NJMAX}"
        "env.sim.physics.solver_cfg.nconmax=${NCONMAX}"
        "env.lafan1_manifest_path=${MANIFEST_PATH}"
        "env.dataset_path=${DATASET_PATH}"
        # MUST stay false: /data/.../g1_hl_diffsr is shared with every other
        # LAFAN1 arm, and a refresh=true job rebuilds it underneath them.
        env.refresh_zarr_dataset=false
        "env.latent_command_dim=${LATENT_COMMAND_DIM}"
        "agent.ipmd.latent_dim=${LATENT_COMMAND_DIM}"
        agent.ipmd.command_source=hl_skill
        "agent.ipmd.hl_skill_checkpoint_path=${ENCODER_CKPT_CONTAINER}"
        # Encoder window is checkpoint-bound and identical for every arm. This
        # is the assertion that makes the latent space shared, not merely similar.
        "agent.ipmd.hl_skill_horizon_steps=${HORIZON_STEPS}"
        agent.ipmd.hl_skill_command_mode=z
        # ---- the only axis that moves ----
        "agent.ipmd.latent_steps_min=${hold}"
        "agent.ipmd.latent_steps_max=${hold}"
        "agent.ipmd.latent_learning.code_period=${hold}"
        # ----------------------------------
        agent.ipmd.latent_learning.command_phase_mode=sin_cos
        "agent.ipmd.latent_learning.code_latent_dim=${Z_DIM}"
        # Encoder stays frozen: this is a command-interface ablation, not a
        # representation-learning one.
        agent.ipmd.hl_skill_finetune_enabled=false
        agent.ipmd.hl_skill_pg_coeff=0.05
        agent.ipmd.hl_skill_anchor_coeff=0.01
        agent.ipmd.hl_skill_offline_diffsr_coeff=1.0
        agent.ipmd.hl_skill_lr=3e-05
        agent.ipmd.reward_loss_coeff=0.0
        agent.ipmd.reward_l2_coeff=0.0
        agent.ipmd.reward_grad_penalty_coeff=0.0
        agent.ipmd.reward_logit_reg_coeff=0.0
        agent.ipmd.reward_param_weight_decay_coeff=0.0
        "agent.collector.frames_per_batch=${ROLLOUT_STEPS}"
        "agent.loss.mini_batch_size=${MINIBATCH_SIZE}"
        "agent.save_interval=${SAVE_INTERVAL}"
        agent.logger.backend=wandb
        agent.logger.video=false
        "agent.logger.project_name=${WANDB_PROJECT}"
        "agent.logger.group_name=${WANDB_GROUP}"
        "agent.logger.exp_name=${run_tag}"
        # Checkpoints go to the /data bind, never the per-submission workspace:
        # a TIMEOUT wipes node-local output before any log sync runs.
        "agent.logger.log_dir=${log_dir}"
    )

    echo
    echo "[INFO] arm hold=${hold}: task='${TASK_NAME}' run_tag='${run_tag}'"
    echo "[INFO]   max_iterations=${max_iterations} (~$((max_iterations * FRAMES_PER_BATCH)) frames of the ${FRAME_CAP} cap)"
    echo "[INFO]   encoder=${ENCODER_CKPT_CONTAINER}"
    echo "[INFO]   checkpoints -> ${log_dir}"
    printf "[CMD] "
    printf "%q " "${cmd[@]}"
    printf "\n"

    if [[ "${is_dry_run}" == "1" ]]; then
        echo "[INFO] DRY_RUN=${DRY_RUN}; not contacting the cluster."
        return 0
    fi
    "${cmd[@]}"
}

if [[ "${is_dry_run}" == "0" ]]; then
    check_gates
else
    echo "[INFO] DRY_RUN=${DRY_RUN}; skipping remote data/encoder gates."
fi

echo "[INFO] segment sizing: ${FRAMES_PER_BATCH} frames/iter, cap ${SEGMENT_MAX_ITERATIONS} iters (~$((SEGMENT_MAX_ITERATIONS * FRAMES_PER_BATCH)) frames per 16h segment)."
for hold in ${ARMS}; do
    submit_arm "${hold}"
done

if [[ "${is_dry_run}" == "1" ]]; then
    echo
    echo "[INFO] Nothing was submitted. Re-run with DRY_RUN=0 to submit."
fi
