#!/usr/bin/env bash
set -euo pipefail

# RLOpt hyperparameter screen on ICE: one H100 Slurm job per arm, all submitted
# at once so the ten arms run concurrently instead of taking 5-6 h sequentially
# on the workstation.
#
# The arms, the frame budget and the geometry are identical to the local screen
# (`run_hp_screen_local.sh`) -- both source `arms.sh`. What differs is only what
# has to differ on the cluster:
#
#   * /data-bound corrected-LAFAN1 manifest and Zarr cache, gated by sha256;
#   * the v2 campaign's det-SR encoder from /data/pretrain_store, so the screen
#     conditions on the same frozen encoder as the 5B run it is meant to improve;
#   * newton_mjwarp physics with the campaign's njmax/nconmax, so b0 really is
#     the cluster recipe verbatim rather than a PhysX lookalike;
#   * W&B rather than the CSV backend, because ten concurrent jobs are only
#     watchable if their curves land somewhere shared while they run.
#
# Every arm is its own `cluster_interface.sh job` submission. That is the proven
# path -- it is what the running 5B job 5558165 was submitted with -- and each
# submission repacks and uploads the ~680 MB workspace archive, so expect the
# submission loop itself to take a few minutes per arm. The jobs queue and run in
# parallel regardless.
#
# DRY_RUN=1 by default. Nothing reaches the cluster without DRY_RUN=0.
#
# Usage, from anywhere:
#   ./experiments/campaigns/2026-08-02-rlopt-hp-search/submit_rlopt_hp_search_ice.sh
#     -> dry run: gates are skipped, every arm's exact command is printed.
#   DRY_RUN=0 ./experiments/campaigns/2026-08-02-rlopt-hp-search/submit_rlopt_hp_search_ice.sh
#     -> submits all ten arms.
#   DRY_RUN=0 ARMS="b0_baseline a2_kl_per_iteration" ...
#     -> submits a subset.

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
case "${DRY_RUN}" in
    1|true|TRUE|yes|YES|on|ON) is_dry_run=1 ;;
    0|false|FALSE|no|NO|off|OFF) is_dry_run=0 ;;
    *) echo "[ERROR] DRY_RUN must be a boolean; got '${DRY_RUN}'." >&2; exit 2 ;;
esac

fail() { echo "[FATAL] $*" >&2; exit 1; }

SEED="${SEED:-0}"

# --- Fixed geometry: identical for every arm, and checked by the aggregator ---
NUM_ENVS="${NUM_ENVS:-12288}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-12}"
# Screening is budgeted in WALL-CLOCK, not frames (2026-08-03). The objective is
# return and episode length per minute, so the honest comparison is "how far does
# each arm get in the same time", not "how fast does each reach a fixed frame
# count" -- and a fixed frame budget actively distorts it, because a faster
# configuration is charged nothing for its speed until the rate is computed.
#
# Implementation: MAX_ITERATIONS is deliberately non-binding and the Slurm wall
# ends the run. Metrics stream to W&B continuously, so everything up to the kill
# is retained; a screen needs no final checkpoint. Arms are then scored by
# interpolating each curve at the same training-minute mark.
#
# Set MAX_ITERATIONS explicitly to go back to a fixed-frame screen.
MAX_ITERATIONS="${MAX_ITERATIONS:-2000}"  # non-binding: ~295M frames, no arm reaches it
# Must be reachable inside the wall, or the arm produces no checkpoint to evaluate.
SAVE_INTERVAL="${SAVE_INTERVAL:-50000000}"
FRAMES_PER_BATCH=$((NUM_ENVS * ROLLOUT_STEPS))
TOTAL_FRAMES=$((MAX_ITERATIONS * FRAMES_PER_BATCH))

# --- Task + frozen latent recipe ---------------------------------------------
TASK_NAME="${TASK_NAME:-Isaac-Imitation-G1-v2}"
HORIZON_STEPS="${HORIZON_STEPS:-10}"
Z_DIM="${Z_DIM:-256}"
LATENT_COMMAND_DIM=$((Z_DIM + 2))
LATENT_HOLD_STEPS="${LATENT_HOLD_STEPS:-10}"
NJMAX="${NJMAX:-320}"
NCONMAX="${NCONMAX:-40}"

# --- Corrected LAFAN1 on the /data bind --------------------------------------
MANIFEST_PATH="${MANIFEST_PATH:-/data/lafan1_corrected_8e95d557/manifests/g1_lafan1_manifest.json}"
DATASET_PATH="${DATASET_PATH:-/data/lafan1_corrected_8e95d557/g1_hl_diffsr}"
EXPECTED_MANIFEST_SHA256="${EXPECTED_MANIFEST_SHA256:-d972c37c41dadbb68c30fc456a9dc9c1bd6d30ed0b7aa9d34b1797472c945db8}"
EXPECTED_NPZ_COUNT="${EXPECTED_NPZ_COUNT:-40}"
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/data}"

# The encoder the v2 5B campaign actually trained and is running on, not the
# locally downloaded one: the screen is only informative about that run if it
# conditions on that run's frozen encoder.
#
# No geometry in the tag. The encoder is fixed by the latent recipe alone
# (horizon, z, mode, seed) and is pretrained at 16 envs -- it knows nothing about
# the rollout geometry. The v2 launcher separated these identities on 2026-08-02
# precisely because folding geometry in made a rollout-steps change point at a
# non-existent encoder; this screen varies rollout steps in a8, so it would have
# walked straight into that.
ENCODER_TAG="${ENCODER_TAG:-lafan1_v2_det_sr_h${HORIZON_STEPS}_z${Z_DIM}_seed${SEED}}"
ENCODER_CKPT_CONTAINER="${ENCODER_CKPT_CONTAINER:-/data/pretrain_store/${ENCODER_TAG}/checkpoints/latest.pt}"
ENCODER_CKPT_REMOTE="${REMOTE_DATA_ROOT}/pretrain_store/${ENCODER_TAG}/checkpoints/latest.pt"

SCREEN_TAG="${SCREEN_TAG:-rlopt_hp_screen_100m_20260802}"
SCREEN_ROOT_CONTAINER="/data/${SCREEN_TAG}"
SUBMISSION_RECORD="${SUBMISSION_RECORD:-${SCRIPT_DIR}/cluster_submission.json}"

# --- W&B ---------------------------------------------------------------------
# Shared training project, functional group name (confirmed 2026-08-02).
WANDB_PROJECT="${WANDB_PROJECT:-g1-lafan1}"
WANDB_GROUP="${WANDB_GROUP:-rlopt-hparam-search}"

# 50M frames is ~12 min of compute at the 72k fps this geometry was measured at
# on an H100 (job 5558142), but early iterations run far slower because episodes
# are short and resets dominate -- the v2 run only reached 75k fps around frame
# 21M, having started near 13k. Two hours is a wide margin over the worst case
# and still schedules quickly; the full 15:59 wall would queue behind everything.
# ~30 min of training plus Isaac startup and a margin.
CLUSTER_SLURM_TIME_LIMIT_DEFAULT="0:50:00"

# Nodes Slurm advertises as healthy but whose GPU is unusable; carried over from
# the v2 campaign launcher.
EXCLUDE_NODES="${EXCLUDE_NODES:-atl1-1-03-010-15-0,atl1-1-03-013-13-0}"

# shellcheck source=./arms.sh
source "${SCRIPT_DIR}/arms.sh"
ARMS="${ARMS:-${HP_SCREEN_ALL_ARM_NAMES[*]}}"

for name in ${ARMS}; do
    found=0
    for known in "${HP_SCREEN_ALL_ARM_NAMES[@]}"; do
        [[ "${name}" == "${known}" ]] && found=1 && break
    done
    [[ "${found}" == 1 ]] || fail "unknown arm '${name}'. Known: ${HP_SCREEN_ALL_ARM_NAMES[*]}"
done

ssh_ice() {
    ssh -o BatchMode=yes -o ConnectTimeout=10 ice "$@"
}

check_data_gate() {
    local actual_sha remote_npz_count
    actual_sha="$(ssh_ice "sha256sum '${REMOTE_DATA_ROOT}/lafan1_corrected_8e95d557/manifests/g1_lafan1_manifest.json'" | awk '{print $1}')"
    remote_npz_count="$(ssh_ice "find '${REMOTE_DATA_ROOT}/lafan1_corrected_8e95d557' -type f -name '*.npz' | wc -l")"
    if [[ "${actual_sha}" != "${EXPECTED_MANIFEST_SHA256}" || "${remote_npz_count}" != "${EXPECTED_NPZ_COUNT}" ]]; then
        fail "ICE corrected-LAFAN1 data gate failed: sha=${actual_sha}, npz=${remote_npz_count}."
    fi
    echo "[PASS] corrected-LAFAN1 manifest sha and NPZ count match the frozen protocol."
}

check_encoder_gate() {
    local bytes
    bytes="$(ssh_ice "if [ -s '${ENCODER_CKPT_REMOTE}' ]; then stat -c %s '${ENCODER_CKPT_REMOTE}'; else echo 0; fi")"
    if (( bytes < 1000000 )); then
        fail "skill encoder missing or truncated (${bytes} bytes): ${ENCODER_CKPT_REMOTE}"
    fi
    echo "[PASS] skill encoder present (${bytes} bytes): ${ENCODER_CKPT_REMOTE}"
}

# The screen only means anything if `agent.optim.kl_adapt_step` exists in the
# RLOpt that ships. It is a local, uncommitted change; the ice_runtime profile
# uses CLUSTER_ARCHIVE_SYNC=1, which tars the workspace (RLOpt included, .git
# excluded), so the working tree is what runs. Check the working tree directly
# rather than trusting that.
check_rlopt_gate() {
    grep -q "kl_adapt_step" "${REPO_ROOT}/RLOpt/rlopt/config_base.py" \
        || fail "RLOpt/rlopt/config_base.py has no kl_adapt_step; arm a2 would fail on an unknown Hydra key."
    grep -q "kl_adapt_step" "${REPO_ROOT}/RLOpt/rlopt/base_class.py" \
        || fail "RLOpt/rlopt/base_class.py does not implement kl_adapt_step routing."
    echo "[PASS] RLOpt working tree carries the kl_adapt_step routing that arm a2 needs."
}

export_cluster_env() {
    export CLUSTER_LOGIN="${CLUSTER_LOGIN:-login-ice.pace.gatech.edu}"
    export CLUSTER_SLURM_SUBMIT_SCRIPT="${CLUSTER_SLURM_SUBMIT_SCRIPT:-pace}"
    export CLUSTER_APPEND_DEFAULT_G1_MANIFEST=0
    export CLUSTER_G1_MANIFEST_REFRESH_POLICY="${CLUSTER_G1_MANIFEST_REFRESH_POLICY:-auto}"
    export CLUSTER_SLURM_TIME_LIMIT="${CLUSTER_SLURM_TIME_LIMIT:-${CLUSTER_SLURM_TIME_LIMIT_DEFAULT}}"
    export CLUSTER_SLURM_PARTITION="${CLUSTER_SLURM_PARTITION:-ice-gpu}"
    export CLUSTER_SLURM_QOS="${CLUSTER_SLURM_QOS:-coe-ice}"
    export CLUSTER_SLURM_GPU_GRES="${CLUSTER_SLURM_GPU_GRES:-gpu:h100:1}"
    export CLUSTER_SLURM_CPUS_PER_TASK="${CLUSTER_SLURM_CPUS_PER_TASK:-16}"
    export CLUSTER_SLURM_MEM="${CLUSTER_SLURM_MEM:-96G}"
    export CLUSTER_SLURM_EXCLUDE="${EXCLUDE_NODES}"
    export CLUSTER_GIT_SYNC_FIRST="${CLUSTER_GIT_SYNC_FIRST:-0}"
    export CLUSTER_G1_USD_PATH=repo
    export CLUSTER_PYTHON_EXECUTABLE="scripts/rlopt/train.py"
}

echo "[INFO] screen tag  : ${SCREEN_TAG}"
echo "[INFO] geometry    : ${NUM_ENVS} envs x ${ROLLOUT_STEPS} steps = ${FRAMES_PER_BATCH} frames/iter"
echo "[INFO] budget      : ${MAX_ITERATIONS} iterations = ${TOTAL_FRAMES} frames per arm"
echo "[INFO] encoder     : ${ENCODER_CKPT_CONTAINER}"
echo "[INFO] wandb       : project=${WANDB_PROJECT} group=${WANDB_GROUP}"
echo "[INFO] wall        : ${CLUSTER_SLURM_TIME_LIMIT:-${CLUSTER_SLURM_TIME_LIMIT_DEFAULT}} per arm, gpu:h100:1"
echo "[INFO] arms        : ${ARMS}"
echo "[INFO] dry run     : ${DRY_RUN}"
echo

if [[ "${is_dry_run}" == "0" ]]; then
    check_rlopt_gate
    check_data_gate
    check_encoder_gate
else
    check_rlopt_gate
    echo "[INFO] DRY_RUN=${DRY_RUN}; skipping remote data/encoder gates."
fi
echo

submitted_names=()
submitted_ids=()

submit_arm() {
    local name="$1" description="$2" overrides="$3"

    # A ROLLOUT_STEPS=N pseudo-override rebatches this arm; the frame budget is
    # held by scaling the iteration count against it, so the arm still sees the
    # same 50M frames. That is what makes it comparable at all.
    #
    # PHASE_MODE=none|sin_cos is the second pseudo-override, and it is a
    # pseudo-override for two reasons. The phase mode already appears in the base
    # command, so appending it would put the same Hydra key on the line twice;
    # and it determines the published command width, which appears twice more
    # (the environment's actor channel and the agent's latent_dim).
    #
    # The width is therefore *derived* here rather than passed alongside. An arm
    # that set the mode without the width -- or set them inconsistently -- would
    # publish a command whose declared and actual sizes disagree, which is
    # exactly the class of mismatch the v2 command interface was introduced to
    # make unrepresentable. Deriving keeps that guarantee.
    # ALGO=<name> is the third pseudo-override. It is not a Hydra key at all --
    # it selects the RLOpt algorithm, and the training entry point derives the
    # agent config entry point from it (`rlopt_<algo>_cfg_entry_point`), so an
    # arm that changed the algorithm without it would run the new algorithm
    # against the default agent config.
    # ENVS=N is the fourth pseudo-override. Like ROLLOUT_STEPS it changes the
    # frames-per-iteration, so the iteration count is rederived from it to hold
    # the frame budget; passing it as a Hydra key would change the geometry
    # without changing the budget and silently break comparability.
    local arm_rollout="${ROLLOUT_STEPS}"
    local arm_envs="${NUM_ENVS}"
    local arm_phase_mode="sin_cos"
    local arm_algo="IPMD"
    local hydra_overrides=""
    for token in ${overrides}; do
        if [[ "${token}" == ENVS=* ]]; then
            arm_envs="${token#ENVS=}"
        elif [[ "${token}" == ROLLOUT_STEPS=* ]]; then
            arm_rollout="${token#ROLLOUT_STEPS=}"
        elif [[ "${token}" == PHASE_MODE=* ]]; then
            arm_phase_mode="${token#PHASE_MODE=}"
        elif [[ "${token}" == ALGO=* ]]; then
            arm_algo="${token#ALGO=}"
        else
            hydra_overrides+="${token} "
        fi
    done
    overrides="${hydra_overrides% }"

    local arm_command_dim
    case "${arm_phase_mode}" in
        sin_cos) arm_command_dim=$((Z_DIM + 2)) ;;
        none)    arm_command_dim=$((Z_DIM)) ;;
        *) fail "${name}: PHASE_MODE must be 'sin_cos' or 'none'; got '${arm_phase_mode}'" ;;
    esac
    local arm_frames_per_batch=$((arm_envs * arm_rollout))
    if (( TOTAL_FRAMES % arm_frames_per_batch != 0 )); then
        fail "${name}: ${TOTAL_FRAMES} frames is not a whole number of ${arm_frames_per_batch}-frame iterations"
    fi
    local arm_iterations=$((TOTAL_FRAMES / arm_frames_per_batch))

    export_cluster_env
    export CLUSTER_SLURM_JOB_NAME_PREFIX="hpscreen-${name//_/-}"
    # wandb reads WANDB_TAGS itself; run_singularity.sh carries it across the
    # container boundary as CLUSTER_WANDB_TAGS.
    export CLUSTER_WANDB_TAGS="sweep,lafan1,v2,det-sr,${name}"

    local -a cmd=(./docker/cluster/cluster_interface.sh -c ice_runtime job
        --task "${TASK_NAME}"
        --num_envs "${arm_envs}"
        --headless
        --assert-kitless
        --algo "${arm_algo}"
        --seed "${SEED}"
        --max_iterations "${arm_iterations}"
        --kit_args=--/app/extensions/fsWatcherEnabled=false
        # Every metric every iteration: 340 rows is nothing, and the whole point
        # is the shape of the early curve.
        --log_interval "${arm_frames_per_batch}"
        physics=newton_mjwarp
        "env.sim.physics.solver_cfg.njmax=${NJMAX}"
        "env.sim.physics.solver_cfg.nconmax=${NCONMAX}"
        "env.data.manifest=${MANIFEST_PATH}"
        "env.data.cache_dir=${DATASET_PATH}"
        # MUST stay false: the /data cache is shared with every other LAFAN1 arm
        # and a refresh=true job rebuilds it underneath them -- including the
        # running 5B job.
        env.data.cache_refresh=false
        "env.command_interface.actor.dim=${arm_command_dim}"
        "agent.ipmd.latent_dim=${arm_command_dim}"
        agent.ipmd.command_source=hl_skill
        "agent.ipmd.hl_skill_checkpoint_path=${ENCODER_CKPT_CONTAINER}"
        "agent.ipmd.hl_skill_horizon_steps=${HORIZON_STEPS}"
        agent.ipmd.hl_skill_command_mode=z
        "agent.ipmd.latent_steps_min=${LATENT_HOLD_STEPS}"
        "agent.ipmd.latent_steps_max=${LATENT_HOLD_STEPS}"
        "agent.ipmd.latent_learning.code_period=${LATENT_HOLD_STEPS}"
        "agent.ipmd.latent_learning.command_phase_mode=${arm_phase_mode}"
        "agent.ipmd.latent_learning.code_latent_dim=${Z_DIM}"
        agent.ipmd.hl_skill_finetune_enabled=false
        agent.ipmd.hl_skill_pg_coeff=0.05
        agent.ipmd.hl_skill_anchor_coeff=0.01
        agent.ipmd.hl_skill_offline_diffsr_coeff=1.0
        agent.ipmd.hl_skill_lr=3e-05
        "agent.collector.frames_per_batch=${arm_rollout}"
        # A screen does not need checkpoints, but one at the end is cheap and
        # lets a winning arm be continued rather than re-run from scratch.
        # NOT TOTAL_FRAMES. Under the wall-clock protocol the iteration cap is
        # deliberately non-binding, so an arm never reaches TOTAL_FRAMES and would
        # save nothing at all -- leaving it unevaluable after the fact. A fixed
        # interval well inside what any arm reaches guarantees a checkpoint, and
        # the Slurm wall killing the job mid-interval only costs the last partial
        # one.
        "agent.save_interval=${SAVE_INTERVAL}"
        agent.logger.backend=wandb
        agent.logger.video=false
        "agent.logger.project_name=${WANDB_PROJECT}"
        "agent.logger.group_name=${WANDB_GROUP}"
        "agent.logger.exp_name=${SCREEN_TAG}_${name}"
        # /data, never the per-submission workspace: an ICE TIMEOUT is a hard
        # SIGKILL that wipes node-local output before any log sync runs.
        "agent.logger.log_dir=${SCREEN_ROOT_CONTAINER}/${name}"
    )
    # shellcheck disable=SC2206
    cmd+=(${overrides})

    echo "=== ${name} ==="
    echo "    ${description}"
    echo "    overrides : ${overrides}"
    echo "    geometry  : ${arm_envs} x ${arm_rollout} = ${arm_frames_per_batch}/iter, ${arm_iterations} iters = ${TOTAL_FRAMES} frames"

    if [[ "${is_dry_run}" != "0" ]]; then
        printf '    [CMD] '; printf '%q ' "${cmd[@]}"; printf '\n\n'
        return 0
    fi

    local out status=0
    out="$("${cmd[@]}" 2>&1)" || status=$?
    printf '%s\n' "${out}" | sed 's/^/    | /'
    if [[ "${status}" != "0" ]]; then
        echo "    [ERROR] submission failed for ${name} (exit ${status})"
        submitted_names+=("${name}")
        submitted_ids+=("SUBMIT_FAILED")
        echo
        return 0
    fi

    local job_id
    job_id="$(printf '%s\n' "${out}" | grep -oE 'Submitted batch job [0-9]+' | tail -1 | awk '{print $NF}')"
    if [[ -z "${job_id}" ]]; then
        echo "    [WARN] submitted but could not parse a job id for ${name}"
        job_id="UNKNOWN"
    else
        echo "    [OK] ${name} -> job ${job_id}"
    fi
    submitted_names+=("${name}")
    submitted_ids+=("${job_id}")
    echo
}

for spec in "${HP_SCREEN_ARM_SPECS[@]}"; do
    name="${spec%%|*}"
    rest="${spec#*|}"
    description="${rest%%|*}"
    overrides="${rest#*|}"
    for requested in ${ARMS}; do
        if [[ "${requested}" == "${name}" ]]; then
            submit_arm "${name}" "${description}" "${overrides}"
            break
        fi
    done
done

if [[ "${is_dry_run}" != "0" ]]; then
    echo "[INFO] Nothing was submitted. Re-run with DRY_RUN=0 to submit."
    exit 0
fi

# One record per screen, so the arm -> job id mapping survives the terminal.
#
# Merged, not overwritten. Arms are routinely submitted in more than one batch --
# one to prove the path works, then the rest -- and a record that silently
# dropped the first batch would misreport which jobs a screen consists of.
HP_ARM_NAMES="${submitted_names[*]}" \
HP_ARM_IDS="${submitted_ids[*]}" \
HP_RECORD="${SUBMISSION_RECORD}" \
HP_META="$(printf '%s\n' \
    "screen_tag=${SCREEN_TAG}" \
    "task=${TASK_NAME}" \
    "wandb_project=${WANDB_PROJECT}" \
    "wandb_group=${WANDB_GROUP}" \
    "num_envs=${NUM_ENVS}" \
    "rollout_steps=${ROLLOUT_STEPS}" \
    "max_iterations=${MAX_ITERATIONS}" \
    "total_frames=${TOTAL_FRAMES}" \
    "seed=${SEED}" \
    "encoder_checkpoint=${ENCODER_CKPT_CONTAINER}" \
    "manifest_sha256=${EXPECTED_MANIFEST_SHA256}" \
    "log_root=${SCREEN_ROOT_CONTAINER}" \
    "workspace_git_sha=$(git -C "${REPO_ROOT}" rev-parse HEAD)" \
    "workspace_dirty=$( [ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ] && echo true || echo false )" \
    "rlopt_git_sha=$(git -C "${REPO_ROOT}/RLOpt" rev-parse HEAD)" \
    "rlopt_dirty=$( [ -n "$(git -C "${REPO_ROOT}/RLOpt" status --porcelain)" ] && echo true || echo false )" \
)" \
python3 - <<'PY'
import json, os, datetime, pathlib

record_path = pathlib.Path(os.environ["HP_RECORD"])
meta = dict(
    line.split("=", 1)
    for line in os.environ["HP_META"].splitlines()
    if "=" in line
)
names = os.environ["HP_ARM_NAMES"].split()
ids = os.environ["HP_ARM_IDS"].split()

record = {}
if record_path.exists():
    try:
        record = json.loads(record_path.read_text())
    except json.JSONDecodeError:
        # A corrupt record is worth keeping for inspection rather than silently
        # replacing: it is the only local trace of an earlier submission.
        record_path.rename(record_path.with_suffix(".json.corrupt"))
        record = {}

for key in ("num_envs", "rollout_steps", "max_iterations", "total_frames", "seed"):
    meta[key] = int(meta[key])
for key in ("workspace_dirty", "rlopt_dirty"):
    meta[key] = meta[key] == "true"

geometry = {k: meta.pop(k) for k in
            ("num_envs", "rollout_steps", "max_iterations", "total_frames", "seed")}
wandb = {"project": meta.pop("wandb_project"), "group": meta.pop("wandb_group")}

# Per-arm, not global: the budget changed mid-campaign (50M -> 100M), and a
# single geometry block would retroactively relabel every earlier arm with the
# current budget. A provenance record that misstates what an arm ran is worse
# than no record.
arms = record.get("arms", {})
for name, job in zip(names, ids):
    arms[name] = {
        "job": job,
        "total_frames": geometry["total_frames"],
        "num_envs": geometry["num_envs"],
        "rollout_steps": geometry["rollout_steps"],
        "seed": geometry["seed"],
    }

record.update(meta)
record["geometry"] = geometry
record["wandb"] = wandb
record["arms"] = arms
record["last_submitted_at"] = datetime.datetime.now().astimezone().isoformat()
record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
PY

echo "[INFO] submitted ${#submitted_names[@]} arms; record -> ${SUBMISSION_RECORD}"
echo "[INFO] watch:  ssh ice \"squeue -u \\\$USER -o '%.10i %.28j %.8T %.10M %R'\""
echo "[INFO] wandb:  https://wandb.ai/<entity>/${WANDB_PROJECT}/groups/${WANDB_GROUP}"
