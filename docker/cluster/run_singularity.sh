#!/usr/bin/env bash

set -e

echo "(run_singularity.py): Called on compute node from current isaaclab directory $1 with container profile $2 and arguments ${@:3}"

#==
# Helper functions
#==

CLUSTER_ENV_OVERRIDES=()

capture_cluster_env_overrides() {
    local name
    local value

    CLUSTER_ENV_OVERRIDES=()
    while IFS='=' read -r name value; do
        case "$name" in
            CLUSTER_*)
                CLUSTER_ENV_OVERRIDES+=("$name=$value")
                ;;
        esac
    done < <(env)
}

restore_cluster_env_overrides() {
    local assignment

    for assignment in "${CLUSTER_ENV_OVERRIDES[@]}"; do
        export "$assignment"
    done
}

prefix_home_if_relative() {
    local home_dir="$1"
    local raw_path="$2"

    if [ -z "$raw_path" ]; then
        echo ""
        return
    fi
    case "$raw_path" in
        /*)
            echo "$raw_path"
            ;;
        *)
            echo "$home_dir/$raw_path"
            ;;
    esac
}

resolve_rlopt_backend() {
    local explicit_backend="${CLUSTER_SIM_BACKEND:-auto}"
    local token=""
    local token_backend=""

    case "$explicit_backend" in
        auto|physx|newton) ;;
        *)
            echo "[ERROR] CLUSTER_SIM_BACKEND must be auto, physx, or newton; got '$explicit_backend'." >&2
            return 1
            ;;
    esac

    for token in "$@"; do
        case "$token" in
            physics=physx)
                [ -z "$token_backend" ] || [ "$token_backend" = "physx" ] || {
                    echo "[ERROR] Conflicting physics backend arguments." >&2
                    return 1
                }
                token_backend="physx"
                ;;
            physics=newton*|presets=*newton*|--assert-kitless)
                [ -z "$token_backend" ] || [ "$token_backend" = "newton" ] || {
                    echo "[ERROR] Conflicting physics backend arguments." >&2
                    return 1
                }
                token_backend="newton"
                ;;
        esac
    done

    if [ "$explicit_backend" != "auto" ]; then
        if [ -n "$token_backend" ] && [ "$token_backend" != "$explicit_backend" ]; then
            echo "[ERROR] CLUSTER_SIM_BACKEND=$explicit_backend conflicts with CLI backend $token_backend." >&2
            return 1
        fi
        echo "$explicit_backend"
    elif [ -n "$token_backend" ]; then
        echo "$token_backend"
    else
        echo "physx"
    fi
}

setup_directories() {
    # Check and create directories
    for dir in \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/kit" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/ov" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/pip" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/glcache" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/computecache" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/triton" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/torchinductor" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/home" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/logs" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/data" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/documents" \
        "${CLUSTER_DATA_DIR}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            echo "Created directory: $dir"
        fi
    done
}

load_secret_from_file() {
    local secret_file="$1"
    if [ ! -f "$secret_file" ]; then
        return 1
    fi
    tr -d '\r\n' < "$secret_file"
}

prefix_home_if_relative() {
    local home_dir="$1"
    local path="$2"

    if [ -z "$path" ] || [[ "$path" = /* ]]; then
        echo "$path"
    else
        echo "${home_dir%/}/$path"
    fi
}

build_g1_preflight_cmd() {
    local data_root="$1"
    local manifest_path="$2"
    local expected_motion_count="$3"
    local repo_id="$4"
    local repo_revision="$5"
    local force_download="$6"
    local manifest_refresh_policy="$7"
    local quoted_data_root=""
    local quoted_manifest_path=""
    local quoted_expected_motion_count=""
    local quoted_repo_id=""
    local quoted_repo_revision=""
    local quoted_force_download=""
    local quoted_manifest_refresh_policy=""

    printf -v quoted_data_root '%q' "$data_root"
    printf -v quoted_manifest_path '%q' "$manifest_path"
    printf -v quoted_expected_motion_count '%q' "$expected_motion_count"
    printf -v quoted_repo_id '%q' "$repo_id"
    printf -v quoted_repo_revision '%q' "$repo_revision"
    printf -v quoted_force_download '%q' "$force_download"
    printf -v quoted_manifest_refresh_policy '%q' "$manifest_refresh_policy"

    cat <<EOF
cluster_g1_data_root=${quoted_data_root}
cluster_g1_manifest_path=${quoted_manifest_path}
cluster_g1_expected_motion_count=${quoted_expected_motion_count}
cluster_g1_repo_id=${quoted_repo_id}
cluster_g1_repo_revision=${quoted_repo_revision}
cluster_g1_force_download=${quoted_force_download}
cluster_g1_manifest_refresh_policy=${quoted_manifest_refresh_policy}
cluster_g1_npz_dir="\${cluster_g1_data_root}/npz/g1"
cluster_g1_npz_count=0

if [ -d "\${cluster_g1_npz_dir}" ]; then
    cluster_g1_npz_count=\$(find "\${cluster_g1_npz_dir}" -type f -name '*.npz' | wc -l | tr -d '[:space:]')
fi

echo "[INFO] Checking G1 dataset under '\${cluster_g1_data_root}' (npz_count=\${cluster_g1_npz_count}, expected=\${cluster_g1_expected_motion_count})"

if [ "\${cluster_g1_force_download}" = "1" ]; then
    echo "[INFO] Force-refreshing G1 dataset from Hugging Face repo '\${cluster_g1_repo_id}' at revision '\${cluster_g1_repo_revision}'."
    /isaac-sim/python.sh scripts/setup_g1_lafan1_npz_dataset.py \\
        --data_root "\${cluster_g1_data_root}" \\
        --repo_id "\${cluster_g1_repo_id}" \\
        --revision "\${cluster_g1_repo_revision}" \\
        --force-download
elif [ "\${cluster_g1_npz_count}" -lt "\${cluster_g1_expected_motion_count}" ]; then
    echo "[INFO] G1 dataset incomplete. Downloading from Hugging Face repo '\${cluster_g1_repo_id}' at revision '\${cluster_g1_repo_revision}'."
    /isaac-sim/python.sh scripts/setup_g1_lafan1_npz_dataset.py \\
        --data_root "\${cluster_g1_data_root}" \\
        --repo_id "\${cluster_g1_repo_id}" \\
        --revision "\${cluster_g1_repo_revision}"
fi

if [ ! -d "\${cluster_g1_npz_dir}" ]; then
    echo "[ERROR] Missing G1 NPZ directory after setup: \${cluster_g1_npz_dir}" >&2
    exit 1
fi

mkdir -p "\$(dirname "\${cluster_g1_manifest_path}")"
cluster_g1_refresh_manifest=0
case "\${cluster_g1_manifest_refresh_policy}" in
    always)
        echo "[INFO] G1 manifest refresh policy is 'always'; generating '\${cluster_g1_manifest_path}'."
        cluster_g1_refresh_manifest=1
        ;;
    never)
        echo "[INFO] G1 manifest refresh policy is 'never'; leaving '\${cluster_g1_manifest_path}' untouched."
        ;;
    auto)
        if [ ! -f "\${cluster_g1_manifest_path}" ]; then
            echo "[INFO] G1 manifest missing; generating '\${cluster_g1_manifest_path}'."
            cluster_g1_refresh_manifest=1
        elif find "\${cluster_g1_npz_dir}" -type f -name '*.npz' -newer "\${cluster_g1_manifest_path}" -print -quit | grep -q .; then
            echo "[INFO] G1 manifest is older than the NPZ tree; regenerating '\${cluster_g1_manifest_path}'."
            cluster_g1_refresh_manifest=1
        else
            echo "[INFO] Reusing existing G1 manifest: '\${cluster_g1_manifest_path}'."
        fi
        ;;
    *)
        echo "[ERROR] Unsupported CLUSTER_G1_MANIFEST_REFRESH_POLICY='\${cluster_g1_manifest_refresh_policy}'. Use one of: auto, never, always." >&2
        exit 1
        ;;
esac

if [ "\${cluster_g1_manifest_refresh_policy}" = "never" ] && [ ! -f "\${cluster_g1_manifest_path}" ]; then
    echo "[ERROR] CLUSTER_G1_MANIFEST_REFRESH_POLICY=never but manifest does not exist: \${cluster_g1_manifest_path}" >&2
    exit 1
fi

if [ "\${cluster_g1_refresh_manifest}" = "1" ]; then
    /isaac-sim/python.sh scripts/write_lafan1_npz_manifest.py \\
        --npz_dir "\${cluster_g1_npz_dir}" \\
        --manifest_path "\${cluster_g1_manifest_path}" \\
        --recursive
fi

cluster_g1_npz_count=\$(find "\${cluster_g1_npz_dir}" -type f -name '*.npz' | wc -l | tr -d '[:space:]')
if [ "\${cluster_g1_npz_count}" -lt "\${cluster_g1_expected_motion_count}" ]; then
    echo "[ERROR] G1 dataset is still incomplete after setup: found \${cluster_g1_npz_count} motions, expected at least \${cluster_g1_expected_motion_count}." >&2
    exit 1
fi

if [ ! -f "\${cluster_g1_manifest_path}" ]; then
    echo "[ERROR] G1 manifest was not generated: \${cluster_g1_manifest_path}" >&2
    exit 1
fi

echo "[INFO] G1 dataset ready: manifest='\${cluster_g1_manifest_path}', motions=\${cluster_g1_npz_count}"
EOF
}

sync_project_logs_back() {
    local tmp_project_logs=""

    if [ "${PROJECT_LOGS_SYNCED:-0}" = "1" ]; then
        return
    fi
    if [ -z "${dir_name:-}" ]; then
        echo "[INFO] No submitted workspace name yet; skipping per-job project log sync."
        return
    fi

    tmp_project_logs="${TMPDIR}/${dir_name}/logs"

    if [ ! -d "$tmp_project_logs" ]; then
        echo "[INFO] No per-job project logs found to sync back: $tmp_project_logs"
        return
    fi
    mkdir -p "$CLUSTER_ISAACLAB_DIR/logs"
    echo "[INFO] Syncing per-job project logs back to permanent workspace: $tmp_project_logs -> $CLUSTER_ISAACLAB_DIR/logs"
    rsync -a "$tmp_project_logs/" "$CLUSTER_ISAACLAB_DIR/logs/"
    PROJECT_LOGS_SYNCED=1
}

seed_shared_project_logs_from_submission() {
    local submitted_project_logs=""

    if [ -z "${dir_name:-}" ]; then
        echo "[INFO] No submitted workspace name yet; skipping shared project log seeding."
        return
    fi

    submitted_project_logs="${TMPDIR}/${dir_name}/logs"
    if [ ! -d "$submitted_project_logs" ]; then
        return
    fi

    mkdir -p "$CLUSTER_ISAACLAB_DIR/logs"
    echo "[INFO] Seeding shared project logs from submitted workspace: $submitted_project_logs -> $CLUSTER_ISAACLAB_DIR/logs"
    rsync -a "$submitted_project_logs/" "$CLUSTER_ISAACLAB_DIR/logs/"
}

capture_requested_env_var() {
    local var_name="$1"
    local marker_name="REQUESTED_${var_name}_IS_SET"
    local value_name="REQUESTED_${var_name}"

    if [ "${!var_name+x}" ]; then
        printf -v "$marker_name" '%s' "1"
        printf -v "$value_name" '%s' "${!var_name}"
    else
        printf -v "$marker_name" '%s' ""
        printf -v "$value_name" '%s' ""
    fi
}

restore_requested_env_var() {
    local var_name="$1"
    local marker_name="REQUESTED_${var_name}_IS_SET"
    local value_name="REQUESTED_${var_name}"

    if [ -n "${!marker_name:-}" ]; then
        printf -v "$var_name" '%s' "${!value_name}"
    fi
}

capture_cluster_env_overrides() {
    local var_name

    for var_name in \
        CLUSTER_ISAAC_SIM_CACHE_DIR \
        CLUSTER_ISAACLAB_DIR \
        CLUSTER_PROJECT_LOGS_DIR \
        CLUSTER_SIF_PATH \
        CLUSTER_DATA_DIR \
        CLUSTER_HF_TOKEN_FILE \
        CLUSTER_WANDB_API_KEY_FILE \
        CLUSTER_CONTAINER_HOME \
        CLUSTER_PYTHON_EXECUTABLE \
        CLUSTER_SIM_BACKEND \
        CLUSTER_USE_OVERLAY \
        CLUSTER_CU130_RUNTIME_ROOT \
        CLUSTER_G1_USD_PATH \
        CLUSTER_AUTO_SETUP_G1_DATA \
        CLUSTER_G1_EXPECTED_MOTION_COUNT \
        CLUSTER_G1_DATA_ROOT \
        CLUSTER_G1_REPO_ID \
        CLUSTER_G1_MANIFEST_PATH \
        CLUSTER_G1_MANIFEST_REFRESH_POLICY \
        CLUSTER_SKIP_CACHE_COPY \
        CLUSTER_OVERLAY_SIZE_MB \
        CLUSTER_JOB_TMPDIR_ROOT \
        CLUSTER_REMOVE_JOB_TMPDIR_AFTER_JOB \
        CLUSTER_USE_SHARED_SIF \
        CLUSTER_SHARED_SIF_PATH \
        CLUSTER_ALLOW_TORCH_COMPILE_DEBUG \
        CLUSTER_USE_XVFB \
        CLUSTER_EXTRA_PYTHONPATH_REL \
        REMOVE_CODE_COPY_AFTER_JOB \
        REMOVE_OVERLAY_AFTER_JOB; do
        capture_requested_env_var "$var_name"
    done
}

restore_cluster_env_overrides() {
    local var_name

    for var_name in \
        CLUSTER_ISAAC_SIM_CACHE_DIR \
        CLUSTER_ISAACLAB_DIR \
        CLUSTER_PROJECT_LOGS_DIR \
        CLUSTER_SIF_PATH \
        CLUSTER_DATA_DIR \
        CLUSTER_HF_TOKEN_FILE \
        CLUSTER_WANDB_API_KEY_FILE \
        CLUSTER_CONTAINER_HOME \
        CLUSTER_PYTHON_EXECUTABLE \
        CLUSTER_SIM_BACKEND \
        CLUSTER_USE_OVERLAY \
        CLUSTER_CU130_RUNTIME_ROOT \
        CLUSTER_G1_USD_PATH \
        CLUSTER_AUTO_SETUP_G1_DATA \
        CLUSTER_G1_EXPECTED_MOTION_COUNT \
        CLUSTER_G1_DATA_ROOT \
        CLUSTER_G1_REPO_ID \
        CLUSTER_G1_MANIFEST_PATH \
        CLUSTER_G1_MANIFEST_REFRESH_POLICY \
        CLUSTER_SKIP_CACHE_COPY \
        CLUSTER_OVERLAY_SIZE_MB \
        CLUSTER_JOB_TMPDIR_ROOT \
        CLUSTER_REMOVE_JOB_TMPDIR_AFTER_JOB \
        CLUSTER_USE_SHARED_SIF \
        CLUSTER_SHARED_SIF_PATH \
        CLUSTER_ALLOW_TORCH_COMPILE_DEBUG \
        CLUSTER_USE_XVFB \
        CLUSTER_EXTRA_PYTHONPATH_REL \
        REMOVE_CODE_COPY_AFTER_JOB \
        REMOVE_OVERLAY_AFTER_JOB; do
        restore_requested_env_var "$var_name"
    done
}


#==
# Main
#==


# get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# load variables to set the Isaac Lab path on the cluster
capture_cluster_env_overrides
source $SCRIPT_DIR/.env.cluster
restore_cluster_env_overrides
source $SCRIPT_DIR/../.env.base

base_tmpdir="${CLUSTER_JOB_TMPDIR_ROOT:-${TMPDIR:-/tmp}}"
base_tmpdir="$(prefix_home_if_relative "$HOME" "$base_tmpdir")"
mkdir -p "$base_tmpdir"
job_tmpdir="${base_tmpdir%/}/isaaclab-${SLURM_JOB_ID:-$$}"
mkdir -p "$job_tmpdir"
xvfb_pid=""
cleanup_job_tmpdir() {
    local status=$?
    set +e
    if [ -n "${xvfb_pid:-}" ]; then
        kill "$xvfb_pid" 2>/dev/null || true
        wait "$xvfb_pid" 2>/dev/null || true
    fi
    sync_project_logs_back
    if [ "${CLUSTER_REMOVE_JOB_TMPDIR_AFTER_JOB:-1}" = "1" ] && [ -n "${job_tmpdir:-}" ]; then
        rm -rf "$job_tmpdir" || true
    fi
    exit "$status"
}
trap cleanup_job_tmpdir EXIT
export TMPDIR="$job_tmpdir"
echo "[INFO] Using per-job TMPDIR: $TMPDIR"

# Paths in .env.cluster are relative to $HOME; prepend it to make them absolute.
CLUSTER_ISAAC_SIM_CACHE_DIR="$(prefix_home_if_relative "$HOME" "$CLUSTER_ISAAC_SIM_CACHE_DIR")"
CLUSTER_ISAACLAB_DIR="$(prefix_home_if_relative "$HOME" "$CLUSTER_ISAACLAB_DIR")"
# Direct invocations that bypass cluster_interface.sh still get a durable
# per-workspace logs directory.  Normal submissions pass the stable base
# workspace's logs directory so every run retains the usual centralized tree.
CLUSTER_PROJECT_LOGS_DIR="${CLUSTER_PROJECT_LOGS_DIR:-${CLUSTER_ISAACLAB_DIR}/logs}"
CLUSTER_PROJECT_LOGS_DIR="$(prefix_home_if_relative "$HOME" "$CLUSTER_PROJECT_LOGS_DIR")"
CLUSTER_SIF_PATH="$(prefix_home_if_relative "$HOME" "$CLUSTER_SIF_PATH")"
CLUSTER_DATA_DIR="$(prefix_home_if_relative "$HOME" "$CLUSTER_DATA_DIR")"
CLUSTER_HF_TOKEN_FILE="$(prefix_home_if_relative "$HOME" "${CLUSTER_HF_TOKEN_FILE:-}")"
CLUSTER_WANDB_API_KEY_FILE="$(prefix_home_if_relative "$HOME" "${CLUSTER_WANDB_API_KEY_FILE:-}")"
[ -n "${CLUSTER_G1_MANIFEST_PATH:-}" ] && CLUSTER_G1_MANIFEST_PATH="$(prefix_home_if_relative "$HOME" "$CLUSTER_G1_MANIFEST_PATH")"
[ -n "${CLUSTER_G1_DATA_ROOT:-}" ] && CLUSTER_G1_DATA_ROOT="$(prefix_home_if_relative "$HOME" "$CLUSTER_G1_DATA_ROOT")"

# Runtime home inside singularity container.
# Defaults to /home/$USER so Isaac Sim writes to a user path, but the path is
# backed by scratch via bind mount below.
container_home="${CLUSTER_CONTAINER_HOME:-/home/${USER}}"
container_triton_cache_dir="${container_home}/.cache/triton"
container_torchinductor_cache_dir="${container_home}/.cache/torchinductor"
allow_torch_compile_debug="${CLUSTER_ALLOW_TORCH_COMPILE_DEBUG:-0}"
auto_setup_g1_data="${CLUSTER_AUTO_SETUP_G1_DATA:-1}"
cluster_g1_expected_motion_count="${CLUSTER_G1_EXPECTED_MOTION_COUNT:-40}"
cluster_g1_data_root="${CLUSTER_G1_DATA_ROOT:-${CLUSTER_DATA_DIR}/lafan1}"
cluster_g1_repo_id="${CLUSTER_G1_REPO_ID:-GeorgiaTech/g1_lafan1_50hz}"
cluster_g1_repo_revision="${CLUSTER_G1_REPO_REVISION:-main}"
cluster_g1_force_download="${CLUSTER_G1_FORCE_DOWNLOAD:-0}"
cluster_g1_manifest_path="${CLUSTER_G1_MANIFEST_PATH:-${cluster_g1_data_root}/manifests/g1_lafan1_manifest.json}"
cluster_g1_manifest_refresh_policy="${CLUSTER_G1_MANIFEST_REFRESH_POLICY:-auto}"
cluster_hf_token="${CLUSTER_HF_TOKEN:-}"
cluster_wandb_api_key="${CLUSTER_WANDB_API_KEY:-${WANDB_API_KEY:-}}"

if [ -z "${cluster_hf_token}" ] && [ -n "${CLUSTER_HF_TOKEN_FILE:-}" ]; then
    if [ -f "${CLUSTER_HF_TOKEN_FILE}" ]; then
        cluster_hf_token="$(load_secret_from_file "${CLUSTER_HF_TOKEN_FILE}")"
    else
        echo "[WARNING] CLUSTER_HF_TOKEN_FILE is set but the file does not exist: ${CLUSTER_HF_TOKEN_FILE}"
    fi
fi

if [ -n "${cluster_hf_token}" ]; then
    export SINGULARITYENV_HF_TOKEN="${cluster_hf_token}"
    export SINGULARITYENV_HUGGINGFACE_HUB_TOKEN="${cluster_hf_token}"
    export APPTAINERENV_HF_TOKEN="${cluster_hf_token}"
    export APPTAINERENV_HUGGINGFACE_HUB_TOKEN="${cluster_hf_token}"
    echo "[INFO] Loaded Hugging Face token for container runtime."
else
    echo "[INFO] No Hugging Face token configured for container runtime."
fi

if [ -z "${cluster_wandb_api_key}" ] && [ -n "${CLUSTER_WANDB_API_KEY_FILE:-}" ]; then
    if [ -f "${CLUSTER_WANDB_API_KEY_FILE}" ]; then
        cluster_wandb_api_key="$(load_secret_from_file "${CLUSTER_WANDB_API_KEY_FILE}")"
    else
        echo "[WARNING] CLUSTER_WANDB_API_KEY_FILE is set but the file does not exist: ${CLUSTER_WANDB_API_KEY_FILE}"
    fi
fi

if [ -n "${cluster_wandb_api_key}" ]; then
    export SINGULARITYENV_WANDB_API_KEY="${cluster_wandb_api_key}"
    export APPTAINERENV_WANDB_API_KEY="${cluster_wandb_api_key}"
    echo "[INFO] Loaded W&B API key for container runtime."
else
    echo "[INFO] No W&B API key configured for container runtime."
fi

# `--containall` isolates the workload environment, so Slurm's array index is
# not inherited automatically. Staged array workflows use it to select an
# explicit goal; forward it deliberately across the container boundary.
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    export SINGULARITYENV_SLURM_ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID}"
    export APPTAINERENV_SLURM_ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID}"
fi

# Construct PYTHONPATH entries from synced repos.
# NOTE: We intentionally avoid "IsaacLab/source" because it makes "isaaclab" a namespace package
# (no __file__), which can break wandb/pydantic introspection in torchrl logging.
extra_pythonpath_rel="${CLUSTER_EXTRA_PYTHONPATH_REL:-IsaacLab/source/isaaclab:IsaacLab/source/isaaclab_tasks:IsaacLab/source/isaaclab_assets:IsaacLab/source/isaaclab_rl:IsaacLab/source/isaaclab_mimic:source/isaaclab_imitation:RLOpt:ImitationLearningTools}"
container_pythonpath_prefix=""
IFS=':' read -ra extra_pythonpath_items <<< "$extra_pythonpath_rel"
for rel_path in "${extra_pythonpath_items[@]}"; do
    if [ -n "$rel_path" ]; then
        container_pythonpath_prefix="${container_pythonpath_prefix}/workspace/isaaclab/project/${rel_path}:"
    fi
done
container_pythonpath="${container_pythonpath_prefix}\${PYTHONPATH}"

# make sure that all directories exists in cache directory
setup_directories
# copy all cache files unless the caller requests a lightweight startup.  Short
# eval/export jobs do not need to spend minutes copying the full Isaac Sim cache.
tmp_isaac_sim_cache_dir="$TMPDIR/$(basename "$CLUSTER_ISAAC_SIM_CACHE_DIR")"
if [ "${CLUSTER_SKIP_CACHE_COPY:-0}" = "1" ]; then
    echo "[INFO] Skipping Isaac Sim cache copy via CLUSTER_SKIP_CACHE_COPY=1"
    for dir in \
        "$tmp_isaac_sim_cache_dir/cache/kit" \
        "$tmp_isaac_sim_cache_dir/cache/ov" \
        "$tmp_isaac_sim_cache_dir/cache/pip" \
        "$tmp_isaac_sim_cache_dir/cache/glcache" \
        "$tmp_isaac_sim_cache_dir/cache/computecache" \
        "$tmp_isaac_sim_cache_dir/cache/triton" \
        "$tmp_isaac_sim_cache_dir/cache/torchinductor" \
        "$tmp_isaac_sim_cache_dir/home" \
        "$tmp_isaac_sim_cache_dir/logs" \
        "$tmp_isaac_sim_cache_dir/data" \
        "$tmp_isaac_sim_cache_dir/documents"; do
        mkdir -p "$dir"
    done
else
    echo "[INFO] Copying Isaac Sim cache to per-job TMPDIR."
    cp -r $CLUSTER_ISAAC_SIM_CACHE_DIR $TMPDIR
fi

# make sure logs directory exists (in the permanent isaaclab directory)
mkdir -p "$CLUSTER_ISAACLAB_DIR/logs"
touch "$CLUSTER_ISAACLAB_DIR/logs/.keep"

# copy the temporary isaaclab directory with the latest changes to the compute node
echo "[INFO] Copying submitted workspace into per-job TMPDIR."
cp -r $1 $TMPDIR
# Get the directory name
dir_name=$(basename "$1")
seed_shared_project_logs_from_submission
project_logs_bind_args=()
if [ -n "${CLUSTER_PROJECT_LOGS_DIR:-}" ]; then
    mkdir -p "$CLUSTER_PROJECT_LOGS_DIR" "$TMPDIR/$dir_name/logs"
    project_logs_bind_args=(-B "$CLUSTER_PROJECT_LOGS_DIR:/workspace/isaaclab/project/logs:rw")
    echo "[INFO] Binding persistent project logs: $CLUSTER_PROJECT_LOGS_DIR -> /workspace/isaaclab/project/logs"
fi

container_image="$TMPDIR/$2.sif"
if [ "${CLUSTER_USE_SHARED_SIF:-0}" = "1" ]; then
    container_image="${CLUSTER_SHARED_SIF_PATH:-${CLUSTER_SIF_PATH}/$2.sif}"
    container_image="$(prefix_home_if_relative "$HOME" "$container_image")"
    if [ ! -e "$container_image" ]; then
        mkdir -p "$(dirname "$container_image")"
        lock_file="${container_image}.lock"
        (
            flock 9
            if [ ! -e "$container_image" ]; then
                tmp_container_image="${container_image}.tmp.${SLURM_JOB_ID:-$$}"
                rm -rf "$tmp_container_image"
                mkdir -p "$tmp_container_image"
                tar -xf "$CLUSTER_SIF_PATH/$2.tar" -C "$tmp_container_image"
                mv "$tmp_container_image/$2.sif" "$container_image"
                rm -rf "$tmp_container_image"
            fi
        ) 9>"$lock_file"
    fi
    echo "[INFO] Using shared container image: $container_image"
else
    # copy container to the compute node
    echo "[INFO] Extracting container image into per-job TMPDIR."
    tar -xf "$CLUSTER_SIF_PATH/$2.tar" -C "$TMPDIR"
fi

# The CU130 runtime is immutable. Normal jobs bind writable caches, data, logs,
# home, /tmp, and the submitted source tree without mutating the SIF root.
container_overlay_args=()
overlay_path="${CLUSTER_ISAACLAB_DIR}/${dir_name}.img"
if [ "${CLUSTER_USE_OVERLAY:-0}" = "1" ]; then
    overlay_size_mb="${CLUSTER_OVERLAY_SIZE_MB:-20240}"
    echo "[INFO] Creating optional Apptainer overlay: size=${overlay_size_mb}MB"
    apptainer overlay create --size "$overlay_size_mb" "$overlay_path"
    container_overlay_args=(--overlay "$overlay_path")
else
    echo "[INFO] Running the SIF read-only without an overlay."
fi

# execute command in singularity container
# NOTE: ISAACLAB_PATH is normally set in `isaaclab.sh` but we directly call the isaac-sim python because we sync the entire
# Isaac Lab directory to the compute node and remote the symbolic link to isaac-sim
container_tmpdir="$TMPDIR/container-tmp"
mkdir -p "$container_tmpdir"
container_display_args=()
case "${CLUSTER_USE_XVFB:-0}" in
    1)
        if ! command -v Xvfb >/dev/null 2>&1; then
            echo "[ERROR] CLUSTER_USE_XVFB=1 but Xvfb is unavailable on the compute node." >&2
            exit 1
        fi
        mkdir -p "$container_tmpdir/.X11-unix"
        display_seed="${SLURM_JOB_ID:-$$}"
        display_num=$((100 + display_seed % 800))
        while [ -e "/tmp/.X11-unix/X${display_num}" ]; do
            display_num=$((display_num + 1))
        done
        Xvfb ":${display_num}" -screen 0 1280x720x24 -nolisten tcp -ac > "$job_tmpdir/xvfb.log" 2>&1 &
        xvfb_pid=$!
        for _ in $(seq 1 50); do
            [ -S "/tmp/.X11-unix/X${display_num}" ] && break
            sleep 0.1
        done
        if [ ! -S "/tmp/.X11-unix/X${display_num}" ]; then
            echo "[ERROR] Xvfb did not create display :${display_num}." >&2
            exit 1
        fi
        export APPTAINERENV_DISPLAY=":${display_num}"
        export SINGULARITYENV_DISPLAY=":${display_num}"
        container_display_args=(-B "/tmp/.X11-unix:/tmp/.X11-unix:rw")
        echo "[INFO] Xvfb display ready: DISPLAY=:${display_num}"
        ;;
    0) ;;
    *)
        echo "[ERROR] CLUSTER_USE_XVFB must be 0 or 1." >&2
        exit 1
        ;;
esac
preflight_cmd=""
if [ "${auto_setup_g1_data}" = "1" ]; then
    preflight_cmd="$(build_g1_preflight_cmd "${cluster_g1_data_root}" "${cluster_g1_manifest_path}" "${cluster_g1_expected_motion_count}" "${cluster_g1_repo_id}" "${cluster_g1_repo_revision}" "${cluster_g1_force_download}" "${cluster_g1_manifest_refresh_policy}")"
fi
runtime_root="${CLUSTER_CU130_RUNTIME_ROOT:-/opt/isaaclab-imitation-runtime}"
# Default: no USD override; the job uses the repo-packaged official Unitree
# USD (assets/unitree/g1_description, git-lfs), which
# resolve_unitree_g1_29dof_usd_path() picks up as the packaged default. Set
# CLUSTER_G1_USD_PATH to an absolute path only for explicit asset experiments.
g1_usd_path="${CLUSTER_G1_USD_PATH:-repo}"
g1_usd_env_exports="export ISAACLAB_IMITATION_UNITREE_USD_CACHE_ROOT=$(dirname "${g1_usd_path}") && export ISAACLAB_IMITATION_UNITREE_USD_PATH=${g1_usd_path} && "
if [ "${g1_usd_path}" = "repo" ] || [ "${g1_usd_path}" = "none" ]; then
    g1_usd_env_exports=""
fi
rlopt_backend=""
rlopt_pipeline=0
if [ "${CLUSTER_PYTHON_EXECUTABLE}" = "scripts/rlopt/train_hl_skill_pipeline.py" ]; then
    rlopt_pipeline=1
fi
if [ "${CLUSTER_PYTHON_EXECUTABLE}" = "scripts/rlopt/train.py" ] || [ "$rlopt_pipeline" = "1" ]; then
    rlopt_backend="$(resolve_rlopt_backend "${@:3}")"
fi

if [ "$rlopt_backend" = "newton" ]; then
    printf -v workload_args '%q ' "${CLUSTER_PYTHON_EXECUTABLE}" "${@:3}" --assert-kitless
    workload_cmd='runtime_python=""; for candidate in "${ISAACLAB_CU130_RUNTIME_ROOT}/bin/python" /opt/isaaclab-imitation-runtime-spec/.pixi/envs/container-runtime/bin/python; do if [ -x "$candidate" ]; then runtime_python="$candidate"; break; fi; done; if [ -z "$runtime_python" ]; then echo "[ERROR] CU130 runtime Python not found." >&2; exit 1; fi; exec "$runtime_python" '"${workload_args}"
elif [ "$rlopt_backend" = "physx" ]; then
    if [ "$rlopt_pipeline" = "1" ]; then
        printf -v workload_args '%q ' "${CLUSTER_PYTHON_EXECUTABLE}" "${@:3}"
    else
        printf -v workload_args '%q ' scripts/rlopt/train_physx.py "${@:3}"
    fi
    workload_cmd='runtime_site=""; for candidate in "${ISAACLAB_CU130_RUNTIME_ROOT}"/lib/python*/site-packages /opt/isaaclab-imitation-runtime-spec/.pixi/envs/container-runtime/lib/python*/site-packages; do if [ -d "$candidate/torch" ]; then runtime_site="$candidate"; break; fi; done; if [ -z "$runtime_site" ]; then echo "[ERROR] CU130 runtime site-packages not found." >&2; exit 1; fi; export ISAACLAB_CU130_SITE_PACKAGES="$runtime_site"; runtime_nvidia_libs="$(find "$runtime_site/nvidia" -mindepth 2 -maxdepth 3 -type d -name lib -print 2>/dev/null | paste -sd: -)"; if [ -n "$runtime_nvidia_libs" ]; then export LD_LIBRARY_PATH="$runtime_nvidia_libs:${LD_LIBRARY_PATH:-}"; fi; runtime_nccl="$runtime_site/nvidia/nccl/lib/libnccl.so.2"; if [ ! -f "$runtime_nccl" ]; then echo "[ERROR] CU130 runtime NCCL not found: $runtime_nccl" >&2; exit 1; fi; export LD_PRELOAD="$runtime_nccl${LD_PRELOAD:+:$LD_PRELOAD}"; success_marker="${TMPDIR}/rlopt-physx-success"; rm -f "$success_marker"; export ISAACLAB_WORKLOAD_SUCCESS_MARKER="$success_marker"; /isaac-sim/python.sh '"${workload_args}"'; python_status=$?; if [ "$python_status" -ne 0 ]; then exit "$python_status"; fi; if [ ! -f "$success_marker" ]; then echo "[ERROR] PhysX process returned without its workload success marker." >&2; exit 1; fi'
else
    printf -v workload_cmd '%q ' /isaac-sim/python.sh "${CLUSTER_PYTHON_EXECUTABLE}" "${@:3}"
fi
container_entry_cmd="export ACCEPT_EULA=${ACCEPT_EULA:-Y} && export PRIVACY_CONSENT=${PRIVACY_CONSENT:-Y} && export OMNI_KIT_ACCEPT_EULA=YES && export HOME=${container_home} && export TMPDIR=/tmp && export XDG_CACHE_HOME=${container_home}/.cache && export XDG_DATA_HOME=${container_home}/.local/share && export ISAACLAB_WORKSPACE_PATH=/workspace/isaaclab/project && export ISAACLAB_PATH=/workspace/isaaclab/project/IsaacLab && export ISAACSIM_PATH=/isaac-sim && export ISAACLAB_DATA_DIR=/data && export CLUSTER_DATA_DIR=${CLUSTER_DATA_DIR} && export PYTHONPATH=${container_pythonpath} && export ISAACLAB_SPLIT_RUNTIME=1 && export ISAACLAB_REQUIRE_CU130_RUNTIME=1 && export ISAACLAB_REQUIRE_GPU_IDENTIFICATION=1 && export ISAACLAB_CU130_RUNTIME_ROOT=${runtime_root} && ${g1_usd_env_exports}export TRITON_CACHE_DIR=${container_triton_cache_dir} && export TORCHINDUCTOR_CACHE_DIR=${container_torchinductor_cache_dir} && export RL_WARNINGS=${RL_WARNINGS:-False} && if [ \"${allow_torch_compile_debug}\" != \"1\" ]; then unset TORCH_LOGS; export TORCHDYNAMO_VERBOSE=0; export TORCH_COMPILE_DEBUG=0; fi && cd /workspace/isaaclab/project"
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    printf -v quoted_slurm_array_task_id '%q' "${SLURM_ARRAY_TASK_ID}"
    container_entry_cmd="export SLURM_ARRAY_TASK_ID=${quoted_slurm_array_task_id} && ${container_entry_cmd}"
fi
if [ -n "${preflight_cmd}" ]; then
    container_entry_cmd="${container_entry_cmd} && ${preflight_cmd}"
fi
container_entry_cmd="${container_entry_cmd} && ${workload_cmd}"
if command -v apptainer >/dev/null 2>&1; then
    container_runtime=apptainer
elif command -v singularity >/dev/null 2>&1; then
    container_runtime=singularity
else
    echo "[ERROR] Neither apptainer nor singularity is available on the compute node." >&2
    exit 1
fi
echo "[INFO] Container runtime: $container_runtime"
set +e
"$container_runtime" exec \
    -B "$container_tmpdir:/tmp:rw" \
    "${container_display_args[@]}" \
    -B "$tmp_isaac_sim_cache_dir/cache/kit:${DOCKER_ISAACSIM_ROOT_PATH}/kit/cache:rw" \
    -B "$tmp_isaac_sim_cache_dir/cache/ov:${DOCKER_USER_HOME}/.cache/ov:rw" \
    -B "$tmp_isaac_sim_cache_dir/cache/pip:${DOCKER_USER_HOME}/.cache/pip:rw" \
    -B "$tmp_isaac_sim_cache_dir/cache/glcache:${DOCKER_USER_HOME}/.cache/nvidia/GLCache:rw" \
    -B "$tmp_isaac_sim_cache_dir/cache/computecache:${DOCKER_USER_HOME}/.nv/ComputeCache:rw" \
    -B ${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/triton:${container_triton_cache_dir}:rw \
    -B ${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/torchinductor:${container_torchinductor_cache_dir}:rw \
    -B "$tmp_isaac_sim_cache_dir/logs:${DOCKER_USER_HOME}/.nvidia-omniverse/logs:rw" \
    -B "$tmp_isaac_sim_cache_dir/data:${DOCKER_USER_HOME}/.local/share/ov/data:rw" \
    -B "$tmp_isaac_sim_cache_dir/documents:${DOCKER_USER_HOME}/Documents:rw" \
    -B "$tmp_isaac_sim_cache_dir/home:${container_home}:rw" \
    -B "$TMPDIR/$dir_name:/workspace/isaaclab/project:rw" \
    "${project_logs_bind_args[@]}" \
    -B "${CLUSTER_DATA_DIR}:/data:rw" \
    -B "${CLUSTER_DATA_DIR}:${CLUSTER_DATA_DIR}:rw" \
    "${container_overlay_args[@]}" \
    --nv --containall "$container_image" \
    bash -c "$container_entry_cmd"
workload_status=$?
set -e

sync_project_logs_back || true

# copy resulting cache files back to host
if [ "${CLUSTER_SKIP_CACHE_COPY:-0}" = "1" ]; then
    echo "[INFO] Skipping Isaac Sim cache rsync back via CLUSTER_SKIP_CACHE_COPY=1"
else
    rsync -azPv "$tmp_isaac_sim_cache_dir/" "$CLUSTER_ISAAC_SIM_CACHE_DIR/"
fi

# if defined, remove the temporary isaaclab directory pushed when the job was submitted
if $REMOVE_CODE_COPY_AFTER_JOB; then
    rm -rf $1
fi

# remove the temporary image file
if $REMOVE_OVERLAY_AFTER_JOB && [ "${CLUSTER_USE_OVERLAY:-0}" = "1" ]; then
    rm -f "$overlay_path"
fi

echo "(run_singularity.py): Return"
exit "$workload_status"
