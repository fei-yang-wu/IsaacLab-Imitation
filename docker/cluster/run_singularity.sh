#!/usr/bin/env bash

set -e

echo "(run_singularity.py): Called on compute node from current isaaclab directory $1 with container profile $2 and arguments ${@:3}"

#==
# Helper functions
#==

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

build_g1_preflight_cmd() {
    local data_root="$1"
    local manifest_path="$2"
    local expected_motion_count="$3"
    local repo_id="$4"
    local manifest_refresh_policy="$5"
    local quoted_data_root=""
    local quoted_manifest_path=""
    local quoted_expected_motion_count=""
    local quoted_repo_id=""
    local quoted_manifest_refresh_policy=""

    printf -v quoted_data_root '%q' "$data_root"
    printf -v quoted_manifest_path '%q' "$manifest_path"
    printf -v quoted_expected_motion_count '%q' "$expected_motion_count"
    printf -v quoted_repo_id '%q' "$repo_id"
    printf -v quoted_manifest_refresh_policy '%q' "$manifest_refresh_policy"

    cat <<EOF
cluster_g1_data_root=${quoted_data_root}
cluster_g1_manifest_path=${quoted_manifest_path}
cluster_g1_expected_motion_count=${quoted_expected_motion_count}
cluster_g1_repo_id=${quoted_repo_id}
cluster_g1_manifest_refresh_policy=${quoted_manifest_refresh_policy}
cluster_g1_npz_dir="\${cluster_g1_data_root}/npz/g1"
cluster_g1_npz_count=0

if [ -d "\${cluster_g1_npz_dir}" ]; then
    cluster_g1_npz_count=\$(find "\${cluster_g1_npz_dir}" -type f -name '*.npz' | wc -l | tr -d '[:space:]')
fi

echo "[INFO] Checking G1 dataset under '\${cluster_g1_data_root}' (npz_count=\${cluster_g1_npz_count}, expected=\${cluster_g1_expected_motion_count})"

if [ "\${cluster_g1_npz_count}" -lt "\${cluster_g1_expected_motion_count}" ]; then
    echo "[INFO] G1 dataset incomplete. Downloading from Hugging Face repo '\${cluster_g1_repo_id}'."
    /isaac-sim/python.sh scripts/setup_g1_lafan1_npz_dataset.py \\
        --data_root "\${cluster_g1_data_root}" \\
        --repo_id "\${cluster_g1_repo_id}"
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


#==
# Main
#==


# get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# load variables to set the Isaac Lab path on the cluster
requested_python_executable="${CLUSTER_PYTHON_EXECUTABLE:-}"
source $SCRIPT_DIR/.env.cluster
if [ -n "$requested_python_executable" ]; then
    CLUSTER_PYTHON_EXECUTABLE="$requested_python_executable"
fi
source $SCRIPT_DIR/../.env.base

base_tmpdir="${TMPDIR:-/tmp}"
job_tmpdir="${base_tmpdir%/}/isaaclab-${SLURM_JOB_ID:-$$}"
mkdir -p "$job_tmpdir"
export TMPDIR="$job_tmpdir"
echo "[INFO] Using per-job TMPDIR: $TMPDIR"

# Paths in .env.cluster are relative to $HOME; prepend it to make them absolute.
CLUSTER_ISAAC_SIM_CACHE_DIR="$HOME/$CLUSTER_ISAAC_SIM_CACHE_DIR"
CLUSTER_ISAACLAB_DIR="$HOME/$CLUSTER_ISAACLAB_DIR"
CLUSTER_SIF_PATH="$HOME/$CLUSTER_SIF_PATH"
CLUSTER_DATA_DIR="$HOME/$CLUSTER_DATA_DIR"
CLUSTER_HF_TOKEN_FILE="$HOME/$CLUSTER_HF_TOKEN_FILE"
CLUSTER_WANDB_API_KEY_FILE="$HOME/$CLUSTER_WANDB_API_KEY_FILE"
[ -n "${CLUSTER_G1_MANIFEST_PATH:-}" ] && CLUSTER_G1_MANIFEST_PATH="$HOME/$CLUSTER_G1_MANIFEST_PATH"
[ -n "${CLUSTER_G1_DATA_ROOT:-}" ] && CLUSTER_G1_DATA_ROOT="$HOME/$CLUSTER_G1_DATA_ROOT"

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
    cp -r $CLUSTER_ISAAC_SIM_CACHE_DIR $TMPDIR
fi

# make sure logs directory exists (in the permanent isaaclab directory)
mkdir -p "$CLUSTER_ISAACLAB_DIR/logs"
touch "$CLUSTER_ISAACLAB_DIR/logs/.keep"

# copy the temporary isaaclab directory with the latest changes to the compute node
cp -r $1 $TMPDIR
# Get the directory name
dir_name=$(basename "$1")

# copy container to the compute node
tar -xf "$CLUSTER_SIF_PATH/$2.tar" -C "$TMPDIR"

# create a persistant overlay using apptainer with fakeroot
overlay_size_mb="${CLUSTER_OVERLAY_SIZE_MB:-20240}"
echo "[INFO] Creating Apptainer overlay: size=${overlay_size_mb}MB"
apptainer overlay create --size "$overlay_size_mb" $CLUSTER_ISAACLAB_DIR/$dir_name.img

# execute command in singularity container
# NOTE: ISAACLAB_PATH is normally set in `isaaclab.sh` but we directly call the isaac-sim python because we sync the entire
# Isaac Lab directory to the compute node and remote the symbolic link to isaac-sim
preflight_cmd=""
if [ "${auto_setup_g1_data}" = "1" ]; then
    preflight_cmd="$(build_g1_preflight_cmd "${cluster_g1_data_root}" "${cluster_g1_manifest_path}" "${cluster_g1_expected_motion_count}" "${cluster_g1_repo_id}" "${cluster_g1_manifest_refresh_policy}")"
fi
printf -v workload_cmd '%q ' /isaac-sim/python.sh "${CLUSTER_PYTHON_EXECUTABLE}" "${@:3}"
container_entry_cmd="export ACCEPT_EULA=${ACCEPT_EULA:-Y} && export PRIVACY_CONSENT=${PRIVACY_CONSENT:-Y} && export OMNI_KIT_ACCEPT_EULA=YES && export HOME=${container_home} && export XDG_CACHE_HOME=${container_home}/.cache && export XDG_DATA_HOME=${container_home}/.local/share && export ISAACLAB_WORKSPACE_PATH=/workspace/isaaclab/project && export ISAACLAB_PATH=/workspace/isaaclab/project/IsaacLab && export ISAACSIM_PATH=/workspace/isaaclab/project/IsaacLab/_isaac_sim && export ISAACLAB_DATA_DIR=/data && export CLUSTER_DATA_DIR=${CLUSTER_DATA_DIR} && export PYTHONPATH=${container_pythonpath} && export TRITON_CACHE_DIR=${container_triton_cache_dir} && export TORCHINDUCTOR_CACHE_DIR=${container_torchinductor_cache_dir} && export RL_WARNINGS=${RL_WARNINGS:-False} && if [ \"${allow_torch_compile_debug}\" != \"1\" ]; then unset TORCH_LOGS; export TORCHDYNAMO_VERBOSE=0; export TORCH_COMPILE_DEBUG=0; fi && cd /workspace/isaaclab/project"
if [ -n "${preflight_cmd}" ]; then
    container_entry_cmd="${container_entry_cmd} && ${preflight_cmd}"
fi
container_entry_cmd="${container_entry_cmd} && ${workload_cmd}"
singularity exec \
    -B $TMPDIR/docker-isaac-sim/cache/kit:${DOCKER_ISAACSIM_ROOT_PATH}/kit/cache:rw \
    -B $TMPDIR/docker-isaac-sim/cache/ov:${DOCKER_USER_HOME}/.cache/ov:rw \
    -B $TMPDIR/docker-isaac-sim/cache/pip:${DOCKER_USER_HOME}/.cache/pip:rw \
    -B $TMPDIR/docker-isaac-sim/cache/glcache:${DOCKER_USER_HOME}/.cache/nvidia/GLCache:rw \
    -B $TMPDIR/docker-isaac-sim/cache/computecache:${DOCKER_USER_HOME}/.nv/ComputeCache:rw \
    -B ${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/triton:${container_triton_cache_dir}:rw \
    -B ${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/torchinductor:${container_torchinductor_cache_dir}:rw \
    -B $TMPDIR/docker-isaac-sim/logs:${DOCKER_USER_HOME}/.nvidia-omniverse/logs:rw \
    -B $TMPDIR/docker-isaac-sim/data:${DOCKER_USER_HOME}/.local/share/ov/data:rw \
    -B $TMPDIR/docker-isaac-sim/documents:${DOCKER_USER_HOME}/Documents:rw \
    -B $TMPDIR/docker-isaac-sim/home:${container_home}:rw \
    -B $TMPDIR/$dir_name:/workspace/isaaclab/project:rw \
    -B $CLUSTER_ISAACLAB_DIR/logs:/workspace/isaaclab/project/logs:rw \
    -B ${CLUSTER_DATA_DIR}:/data:rw \
    -B ${CLUSTER_DATA_DIR}:${CLUSTER_DATA_DIR}:rw \
    --overlay $CLUSTER_ISAACLAB_DIR/$dir_name.img \
    --nv --containall $TMPDIR/$2.sif \
    bash -c "$container_entry_cmd"

# copy resulting cache files back to host
if [ "${CLUSTER_SKIP_CACHE_COPY:-0}" = "1" ]; then
    echo "[INFO] Skipping Isaac Sim cache rsync back via CLUSTER_SKIP_CACHE_COPY=1"
else
    rsync -azPv $TMPDIR/docker-isaac-sim $CLUSTER_ISAAC_SIM_CACHE_DIR/..
fi

# if defined, remove the temporary isaaclab directory pushed when the job was submitted
if $REMOVE_CODE_COPY_AFTER_JOB; then
    rm -rf $1
fi

# remove the temporary image file
if $REMOVE_OVERLAY_AFTER_JOB; then
    rm -f $CLUSTER_ISAACLAB_DIR/$dir_name.img
fi

echo "(run_singularity.py): Return"
