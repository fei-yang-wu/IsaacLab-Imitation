#!/usr/bin/env bash

#==
# Configurations
#==

# Exits if error occurs
set -e

# get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# Resolved "<local_path>:<remote_subdir>" sync specs used by the current job submission.
SYNC_EXTRA_REPO_SPECS=""
CLUSTER_ISAACLAB_BASE_DIR=""
CLUSTER_SYNC_LATEST_LINK=""
CLUSTER_PREVIOUS_SYNC_DIR=""
REPO_SYNC_HEAD_SHA=""
REPO_SYNC_ORIGIN_URL=""
REPO_SYNC_BRANCH=""
REPO_SYNC_REMOTE_NAME=""
REPO_SYNC_REMOTE_BRANCH=""
REPO_SYNC_UPSTREAM_REF=""
REPO_SYNC_AHEAD_COUNT=""
REPO_SYNC_BEHIND_COUNT=""
REPO_SYNC_CAN_FETCH_HEAD=""
REPO_SYNC_REASON=""

#==
# Functions
#==
# Function to display warnings in red
display_warning() {
    echo -e "\033[31mWARNING: $1\033[0m"
}

resolve_local_tmp_dir() {
    local candidate=""

    for candidate in "${TMPDIR:-}" "/tmp"; do
        if [ -z "$candidate" ]; then
            continue
        fi
        if mkdir -p "$candidate" >/dev/null 2>&1; then
            realpath "$candidate" 2>/dev/null || echo "$candidate"
            return 0
        fi
    done

    candidate="$SCRIPT_DIR"
    if mkdir -p "$candidate" >/dev/null 2>&1; then
        display_warning "Could not prepare TMPDIR='${TMPDIR:-<unset>}'; using '$candidate' for local temporary files."
        realpath "$candidate" 2>/dev/null || echo "$candidate"
        return 0
    fi

    echo "[ERROR] Failed to resolve a writable local temporary directory." >&2
    exit 1
}

normalize_git_remote_to_https() {
    local remote_url="$1"

    if [ -z "$remote_url" ]; then
        echo "$remote_url"
        return
    fi

    case "$remote_url" in
        git@*:* )
            local host_part="${remote_url#git@}"
            local host="${host_part%%:*}"
            local repo_path="${host_part#*:}"
            echo "https://$host/$repo_path"
            ;;
        ssh://git@*/* )
            local without_prefix="${remote_url#ssh://git@}"
            local host="${without_prefix%%/*}"
            local repo_path="${without_prefix#*/}"
            echo "https://$host/$repo_path"
            ;;
        * )
            echo "$remote_url"
            ;;
    esac
}

# Helper function to compare version numbers
version_gte() {
    # Returns 0 if the first version is greater than or equal to the second, otherwise 1
    [ "$(printf '%s\n' "$1" "$2" | sort -V | head -n 1)" == "$2" ]
}

# Function to check docker versions
check_docker_version() {
    # check if docker is installed
    if ! command -v docker &> /dev/null; then
        echo "[Error] Docker is not installed! Please check the 'Docker Guide' for instruction." >&2;
        exit 1
    fi
    # Retrieve Docker version
    docker_version=$(docker --version | awk '{ print $3 }')
    apptainer_version=$(apptainer --version | awk '{ print $3 }')

    # Check if Docker version is exactly 24.0.7 or Apptainer version is exactly 1.2.5
    if [ "$docker_version" = "24.0.7" ] && [ "$apptainer_version" = "1.2.5" ]; then
        echo "[INFO]: Docker version ${docker_version} and Apptainer version ${apptainer_version} are tested and compatible."

    # Check if Docker version is >= 27.0.0 and Apptainer version is >= 1.3.4
    elif version_gte "$docker_version" "27.0.0" && version_gte "$apptainer_version" "1.3.4"; then
        echo "[INFO]: Docker version ${docker_version} and Apptainer version ${apptainer_version} are tested and compatible."

    # Else, display a warning for non-tested versions
    else
        display_warning "Docker version ${docker_version} and Apptainer version ${apptainer_version} are non-tested versions. There could be issues, please try to update them. More info: https://isaac-sim.github.io/IsaacLab/source/deployment/cluster.html"
    fi
}

# Checks if a docker image exists, otherwise prints warning and exists
check_image_exists() {
    image_name="$1"
    if ! docker image inspect $image_name &> /dev/null; then
        echo "[Error] The '$image_name' image does not exist!" >&2;
        echo "[Error] You might be able to build it with /IsaacLab/docker/container.py." >&2;
        exit 1
    fi
}

# Check if the singularity image exists on the remote host, otherwise print warning and exit
check_singularity_image_exists() {
    image_name="$1"
    if ! ssh "$CLUSTER_LOGIN" "[ -f $CLUSTER_SIF_PATH/$image_name.tar ]"; then
        echo "[Error] The '$image_name' image does not exist on the remote host $CLUSTER_LOGIN!" >&2;
        exit 1
    fi
}

sync_tree_to_cluster() {
    local src_path="$1"
    local dst_path="$2"
    local label="$3"
    local remote_subdir="${4:-.}"
    local link_dest=""
    local -a rsync_cmd

    if [ ! -d "$src_path" ]; then
        display_warning "Skipping sync for '$label': local path not found: $src_path"
        return
    fi

    if [ -n "${CLUSTER_PREVIOUS_SYNC_DIR:-}" ]; then
        if [ "$remote_subdir" = "." ]; then
            link_dest="$CLUSTER_PREVIOUS_SYNC_DIR"
        else
            link_dest="$CLUSTER_PREVIOUS_SYNC_DIR/$remote_subdir"
        fi
    fi

    echo "[INFO] Syncing $label from '$src_path' -> '$CLUSTER_LOGIN:$dst_path'"
    ssh "$CLUSTER_LOGIN" "mkdir -p '$dst_path'"
    rsync_cmd=(
        rsync -rh
        --exclude="*.git*"
        --exclude="wandb"
        --filter=':- .dockerignore'
    )

    if [ -n "$link_dest" ] && ssh "$CLUSTER_LOGIN" "[ -d '$link_dest' ]"; then
        echo "[INFO]   Using incremental sync base: '$link_dest'"
        rsync_cmd+=("--link-dest=$link_dest")
    fi

    rsync_cmd+=(
        "$src_path/"
        "$CLUSTER_LOGIN:$dst_path/"
    )
    "${rsync_cmd[@]}"
}

prepare_repo_git_sync_metadata() {
    local repo_path="$1"
    local upstream_ref=""
    local remote_ref_containing_head=""
    local remote_head_ref=""
    local ahead_behind_counts=""

    REPO_SYNC_HEAD_SHA=""
    REPO_SYNC_ORIGIN_URL=""
    REPO_SYNC_BRANCH=""
    REPO_SYNC_REMOTE_NAME=""
    REPO_SYNC_REMOTE_BRANCH=""
    REPO_SYNC_UPSTREAM_REF=""
    REPO_SYNC_AHEAD_COUNT="0"
    REPO_SYNC_BEHIND_COUNT="0"
    REPO_SYNC_CAN_FETCH_HEAD="0"
    REPO_SYNC_REASON="not_git_repo"

    if ! git -C "$repo_path" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        return 1
    fi

    REPO_SYNC_HEAD_SHA="$(git -C "$repo_path" rev-parse HEAD 2>/dev/null || true)"
    if [ -z "$REPO_SYNC_HEAD_SHA" ]; then
        REPO_SYNC_REASON="missing_head_sha"
        return 1
    fi

    REPO_SYNC_BRANCH="$(git -C "$repo_path" symbolic-ref --quiet --short HEAD 2>/dev/null || true)"
    upstream_ref="$(git -C "$repo_path" rev-parse --abbrev-ref --symbolic-full-name @{u} 2>/dev/null || true)"
    if [ -n "$upstream_ref" ] && [ "${upstream_ref#*/}" != "$upstream_ref" ]; then
        REPO_SYNC_REMOTE_NAME="${upstream_ref%%/*}"
        REPO_SYNC_REMOTE_BRANCH="${upstream_ref#*/}"
        REPO_SYNC_UPSTREAM_REF="$upstream_ref"
    else
        REPO_SYNC_REMOTE_NAME="origin"
        remote_ref_containing_head="$(git -C "$repo_path" for-each-ref --format='%(refname:short)' --contains "$REPO_SYNC_HEAD_SHA" "refs/remotes/${REPO_SYNC_REMOTE_NAME}/*" | grep -v '/HEAD$' | head -n 1)"
        if [ -n "$remote_ref_containing_head" ]; then
            REPO_SYNC_UPSTREAM_REF="$remote_ref_containing_head"
            REPO_SYNC_REMOTE_BRANCH="${remote_ref_containing_head#*/}"
        else
            remote_head_ref="$(git -C "$repo_path" symbolic-ref --quiet --short "refs/remotes/${REPO_SYNC_REMOTE_NAME}/HEAD" 2>/dev/null || true)"
            if [ -n "$remote_head_ref" ]; then
                REPO_SYNC_UPSTREAM_REF="$remote_head_ref"
                REPO_SYNC_REMOTE_BRANCH="${remote_head_ref#*/}"
            else
                REPO_SYNC_REMOTE_BRANCH="$REPO_SYNC_BRANCH"
            fi
        fi
    fi

    REPO_SYNC_ORIGIN_URL="$(git -C "$repo_path" remote get-url "${REPO_SYNC_REMOTE_NAME:-origin}" 2>/dev/null || true)"
    if [ -z "$REPO_SYNC_ORIGIN_URL" ]; then
        REPO_SYNC_REMOTE_NAME="origin"
        REPO_SYNC_REMOTE_BRANCH="${REPO_SYNC_REMOTE_BRANCH:-$REPO_SYNC_BRANCH}"
        REPO_SYNC_ORIGIN_URL="$(git -C "$repo_path" remote get-url origin 2>/dev/null || true)"
    fi
    if [ -z "$REPO_SYNC_ORIGIN_URL" ]; then
        REPO_SYNC_REASON="missing_origin_remote"
        return 1
    fi
    REPO_SYNC_ORIGIN_URL="$(normalize_git_remote_to_https "$REPO_SYNC_ORIGIN_URL")"

    if [ -n "$REPO_SYNC_UPSTREAM_REF" ]; then
        ahead_behind_counts="$(git -C "$repo_path" rev-list --left-right --count "${REPO_SYNC_UPSTREAM_REF}...HEAD" 2>/dev/null || true)"
        if [ -n "$ahead_behind_counts" ]; then
            REPO_SYNC_BEHIND_COUNT="$(echo "$ahead_behind_counts" | awk '{print $1}')"
            REPO_SYNC_AHEAD_COUNT="$(echo "$ahead_behind_counts" | awk '{print $2}')"
        fi
        if git -C "$repo_path" merge-base --is-ancestor "$REPO_SYNC_HEAD_SHA" "$REPO_SYNC_UPSTREAM_REF" >/dev/null 2>&1; then
            REPO_SYNC_CAN_FETCH_HEAD="1"
        fi
    fi

    REPO_SYNC_REASON="dirty_worktree"
    if [ -n "$(git -C "$repo_path" status --porcelain 2>/dev/null)" ]; then
        return 1
    fi

    REPO_SYNC_REASON="clean_git_repo"
    return 0
}

sync_repo_from_git_to_cluster() {
    local dst_path="$1"
    local label="$2"
    local origin_url="$3"
    local head_sha="$4"
    local branch_name="$5"
    local remote_branch="$6"

    echo "[INFO] Syncing $label via git checkout: commit='$head_sha' branch='${branch_name:-detached}' remote_branch='${remote_branch:-N/A}' origin='$origin_url'"
    ssh "$CLUSTER_LOGIN" bash -s -- "$dst_path" "$origin_url" "$head_sha" "$branch_name" "$remote_branch" <<'EOF'
set -e
target_dir="$1"
origin_url="$2"
head_sha="$3"
branch_name="$4"
remote_branch="$5"

if [ -z "$target_dir" ] || [ "$target_dir" = "/" ]; then
    echo "[ERROR] Invalid target dir for git sync: '$target_dir'" >&2
    exit 1
fi

mkdir -p "$(dirname "$target_dir")"

if [ -d "$target_dir/.git" ]; then
    current_origin="$(git -C "$target_dir" remote get-url origin 2>/dev/null || true)"
    if [ "$current_origin" != "$origin_url" ]; then
        rm -rf "$target_dir"
    fi
elif [ -e "$target_dir" ]; then
    rm -rf "$target_dir"
fi

if [ ! -d "$target_dir/.git" ]; then
    git clone --no-checkout "$origin_url" "$target_dir"
fi

if [ -n "$remote_branch" ]; then
    if ! git -C "$target_dir" fetch --depth 1 origin "refs/heads/$remote_branch:refs/remotes/origin/$remote_branch"; then
        git -C "$target_dir" fetch origin "$remote_branch" || true
    fi
fi

if [ -n "$branch_name" ] && [ -n "$remote_branch" ] \
    && git -C "$target_dir" show-ref --verify --quiet "refs/remotes/origin/$remote_branch"; then
    git -C "$target_dir" checkout -B "$branch_name" "refs/remotes/origin/$remote_branch"
elif [ -n "$branch_name" ]; then
    git -C "$target_dir" checkout -B "$branch_name"
fi

if [ -n "$head_sha" ]; then
    if ! git -C "$target_dir" fetch --depth 1 origin "$head_sha"; then
        git -C "$target_dir" fetch origin "$head_sha"
    fi
    if [ -n "$branch_name" ]; then
        git -C "$target_dir" checkout -B "$branch_name" "$head_sha"
    else
        git -C "$target_dir" checkout -f "$head_sha"
    fi
fi

git -C "$target_dir" reset --hard >/dev/null 2>&1 || true
git -C "$target_dir" clean -fdx >/dev/null 2>&1 || true

if [ -f "$target_dir/.gitmodules" ]; then
    git -C "$target_dir" submodule sync --recursive || true
    if ! git -C "$target_dir" submodule update --init --recursive --depth 1; then
        git -C "$target_dir" submodule update --init --recursive
    fi
    git -C "$target_dir" submodule foreach --recursive 'git reset --hard >/dev/null 2>&1 || true; git clean -fdx >/dev/null 2>&1 || true' >/dev/null 2>&1 || true
fi
EOF
}

apply_local_git_diff_to_cluster() {
    local src_path="$1"
    local dst_path="$2"
    local label="$3"
    local base_ref="${4:-HEAD}"
    local has_tracked_changes=0
    local has_untracked_changes=0
    local first_untracked_file=""

    if ! git -C "$src_path" diff --quiet "$base_ref" --; then
        has_tracked_changes=1
    fi

    first_untracked_file="$(git -C "$src_path" ls-files --others --exclude-standard | head -n 1)"
    if [ -n "$first_untracked_file" ]; then
        has_untracked_changes=1
    fi

    if [ "$has_tracked_changes" -eq 0 ] && [ "$has_untracked_changes" -eq 0 ]; then
        echo "[INFO] No local delta to apply for '$label' after git checkout."
        return 0
    fi

    if [ "$has_tracked_changes" -eq 1 ]; then
        echo "[INFO] Applying tracked git delta for '$label' on cluster (base_ref='$base_ref')."
        git -C "$src_path" diff --binary "$base_ref" -- \
            | ssh "$CLUSTER_LOGIN" "cd '$dst_path' && git apply --binary --3way --whitespace=nowarn --check"
        git -C "$src_path" diff --binary "$base_ref" -- \
            | ssh "$CLUSTER_LOGIN" "cd '$dst_path' && git apply --binary --3way --whitespace=nowarn"
    fi

    if [ "$has_untracked_changes" -eq 1 ]; then
        echo "[INFO] Syncing untracked files for '$label' via tar stream."
        git -C "$src_path" ls-files --others --exclude-standard -z \
            | tar -C "$src_path" --null --files-from=- --create --file=- \
            | ssh "$CLUSTER_LOGIN" "mkdir -p '$dst_path' && tar -xf - -C '$dst_path'"
    fi

    return 0
}

sync_repo_prefer_git_then_rsync() {
    local src_path="$1"
    local dst_path="$2"
    local label="$3"
    local remote_subdir="${4:-.}"
    local ahead_count="${REPO_SYNC_AHEAD_COUNT:-0}"
    local behind_count="${REPO_SYNC_BEHIND_COUNT:-0}"

    if is_truthy "${CLUSTER_GIT_SYNC_FIRST:-1}"; then
        if prepare_repo_git_sync_metadata "$src_path"; then
            ahead_count="${REPO_SYNC_AHEAD_COUNT:-0}"
            behind_count="${REPO_SYNC_BEHIND_COUNT:-0}"

            if [ "$ahead_count" -gt 0 ] && [ -n "$REPO_SYNC_UPSTREAM_REF" ]; then
                echo "[INFO] '$label' is ahead of remote base '$REPO_SYNC_UPSTREAM_REF' (ahead=${ahead_count}, behind=${behind_count}); cloning remote branch then applying local revision delta."
                if sync_repo_from_git_to_cluster "$dst_path" "$label" "$REPO_SYNC_ORIGIN_URL" "" "$REPO_SYNC_BRANCH" "$REPO_SYNC_REMOTE_BRANCH" \
                    && apply_local_git_diff_to_cluster "$src_path" "$dst_path" "$label" "$REPO_SYNC_UPSTREAM_REF"; then
                    return
                fi
                display_warning "Git branch+delta sync failed for '$label'; falling back to rsync from local workspace."
            elif [ "${REPO_SYNC_CAN_FETCH_HEAD:-0}" = "1" ]; then
                if sync_repo_from_git_to_cluster "$dst_path" "$label" "$REPO_SYNC_ORIGIN_URL" "$REPO_SYNC_HEAD_SHA" "$REPO_SYNC_BRANCH" "$REPO_SYNC_REMOTE_BRANCH"; then
                    return
                fi
                display_warning "Git sync failed for '$label'; falling back to rsync from local workspace."
            elif [ -n "$REPO_SYNC_UPSTREAM_REF" ]; then
                echo "[INFO] '$label' is using remote base '$REPO_SYNC_UPSTREAM_REF' (ahead=${ahead_count}, behind=${behind_count}); cloning remote branch then applying local revision delta."
                if sync_repo_from_git_to_cluster "$dst_path" "$label" "$REPO_SYNC_ORIGIN_URL" "" "$REPO_SYNC_BRANCH" "$REPO_SYNC_REMOTE_BRANCH" \
                    && apply_local_git_diff_to_cluster "$src_path" "$dst_path" "$label" "$REPO_SYNC_UPSTREAM_REF"; then
                    return
                fi
                display_warning "Git branch+delta sync failed for '$label'; falling back to rsync from local workspace."
            else
                display_warning "Unable to determine a remote base for '$label'; falling back to rsync from local workspace."
            fi
        else
            case "$REPO_SYNC_REASON" in
                dirty_worktree)
                    if [ -n "$REPO_SYNC_HEAD_SHA" ] && [ -n "$REPO_SYNC_ORIGIN_URL" ]; then
                        if [ "${REPO_SYNC_CAN_FETCH_HEAD:-0}" = "1" ]; then
                            echo "[INFO] '$label' has local changes but HEAD is reachable from remote; cloning exact commit then applying worktree delta."
                            if sync_repo_from_git_to_cluster "$dst_path" "$label" "$REPO_SYNC_ORIGIN_URL" "$REPO_SYNC_HEAD_SHA" "$REPO_SYNC_BRANCH" "$REPO_SYNC_REMOTE_BRANCH" \
                                && apply_local_git_diff_to_cluster "$src_path" "$dst_path" "$label" "HEAD"; then
                                return
                            fi
                            display_warning "Git checkout+worktree apply failed for '$label'; falling back to rsync."
                        elif [ -n "${REPO_SYNC_UPSTREAM_REF:-}" ]; then
                            echo "[INFO] '$label' has local changes ahead of remote base '$REPO_SYNC_UPSTREAM_REF'; cloning remote branch then applying the combined local delta."
                            if sync_repo_from_git_to_cluster "$dst_path" "$label" "$REPO_SYNC_ORIGIN_URL" "" "$REPO_SYNC_BRANCH" "$REPO_SYNC_REMOTE_BRANCH" \
                                && apply_local_git_diff_to_cluster "$src_path" "$dst_path" "$label" "$REPO_SYNC_UPSTREAM_REF"; then
                                return
                            fi
                            display_warning "Git branch+delta sync failed for '$label'; falling back to rsync."
                        else
                            echo "[INFO] '$label' is dirty but no upstream base could be determined; using rsync."
                        fi
                    else
                        echo "[INFO] '$label' is dirty but missing git metadata; using rsync."
                    fi
                    ;;
                missing_origin_remote)
                    echo "[INFO] '$label' has no origin remote; using rsync."
                    ;;
                missing_head_sha)
                    echo "[INFO] '$label' is missing a resolvable HEAD commit; using rsync."
                    ;;
                *)
                    echo "[INFO] '$label' is not a clean git repo; using rsync."
                    ;;
            esac
        fi
    else
        echo "[INFO] Git-first sync disabled via CLUSTER_GIT_SYNC_FIRST='${CLUSTER_GIT_SYNC_FIRST:-0}'. Using rsync for '$label'."
    fi

    sync_tree_to_cluster "$src_path" "$dst_path" "$label" "$remote_subdir"
}

dir_has_entries() {
    local dir_path="$1"
    if [ ! -d "$dir_path" ]; then
        return 1
    fi
    [ -n "$(find "$dir_path" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]
}

is_truthy() {
    case "${1:-}" in
        1|true|TRUE|yes|YES|on|ON)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

has_arg_with_prefix() {
    local prefix="$1"
    shift

    for arg in "$@"; do
        case "$arg" in
            "${prefix}"*)
                return 0
                ;;
        esac
    done

    return 1
}

init_incremental_sync_state() {
    CLUSTER_SYNC_LATEST_LINK="${CLUSTER_ISAACLAB_BASE_DIR}_latest"
    CLUSTER_PREVIOUS_SYNC_DIR=""

    if ! is_truthy "${CLUSTER_INCREMENTAL_SYNC:-1}"; then
        echo "[INFO] Incremental sync disabled via CLUSTER_INCREMENTAL_SYNC='${CLUSTER_INCREMENTAL_SYNC:-0}'."
        return
    fi

    if ssh "$CLUSTER_LOGIN" "[ -d '$CLUSTER_SYNC_LATEST_LINK' ]"; then
        CLUSTER_PREVIOUS_SYNC_DIR="$CLUSTER_SYNC_LATEST_LINK"
        echo "[INFO] Incremental sync enabled (base: '$CLUSTER_PREVIOUS_SYNC_DIR')."
    else
        echo "[INFO] Incremental sync enabled, but no previous snapshot found. Doing full sync."
    fi
}

update_latest_sync_link() {
    if ! is_truthy "${CLUSTER_INCREMENTAL_SYNC:-1}"; then
        return
    fi
    ssh "$CLUSTER_LOGIN" "ln -sfn '$CLUSTER_ISAACLAB_DIR' '$CLUSTER_SYNC_LATEST_LINK'"
    echo "[INFO] Updated latest sync link: '$CLUSTER_LOGIN:$CLUSTER_SYNC_LATEST_LINK' -> '$CLUSTER_ISAACLAB_DIR'"
}

resolve_repo_sync_path() {
    local repo_label="$1"
    local workspace_path="$2"
    local override_var_name="$3"
    local override_path="${!override_var_name:-}"
    local resolved_path

    if [ -n "$override_path" ]; then
        if [ ! -d "$override_path" ]; then
            echo "[ERROR] $override_var_name is set but path does not exist: '$override_path'" >&2
            exit 1
        fi
        resolved_path="$(realpath "$override_path")"
        echo "[INFO] Using $repo_label from $override_var_name: '$resolved_path'" >&2
        echo "$resolved_path"
        return
    fi

    if dir_has_entries "$workspace_path"; then
        resolved_path="$(realpath "$workspace_path")"
        echo "$resolved_path"
        return
    fi

    resolved_path="$(realpath "$workspace_path" 2>/dev/null || echo "$workspace_path")"
    echo "$resolved_path"
}

build_default_sync_specs() {
    local local_workspace_root="$1"
    local local_specs=""
    local resolved_path=""

    if [ -n "${CLUSTER_ISAACLAB_LOCAL_PATH:-}" ]; then
        resolved_path="$(resolve_repo_sync_path "IsaacLab" "$local_workspace_root/IsaacLab" "CLUSTER_ISAACLAB_LOCAL_PATH")"
        local_specs="$local_specs $resolved_path:IsaacLab"
    fi

    if [ -n "${CLUSTER_RLOPT_LOCAL_PATH:-}" ]; then
        resolved_path="$(resolve_repo_sync_path "RLOpt" "$local_workspace_root/RLOpt" "CLUSTER_RLOPT_LOCAL_PATH")"
        local_specs="$local_specs $resolved_path:RLOpt"
    fi

    if [ -n "${CLUSTER_IMITATION_TOOLS_LOCAL_PATH:-}" ]; then
        resolved_path="$(resolve_repo_sync_path "ImitationLearningTools" "$local_workspace_root/ImitationLearningTools" "CLUSTER_IMITATION_TOOLS_LOCAL_PATH")"
        local_specs="$local_specs $resolved_path:ImitationLearningTools"
    fi

    echo "${local_specs# }"
}

append_repo_manifest_entry() {
    local manifest_file="$1"
    local repo_name="$2"
    local local_path="$3"
    local remote_subdir="$4"
    local resolved_local_path
    local head_sha
    local branch
    local state
    local changed_files

    resolved_local_path="$(realpath "$local_path" 2>/dev/null || echo "$local_path")"

    if git -C "$resolved_local_path" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        head_sha="$(git -C "$resolved_local_path" rev-parse HEAD 2>/dev/null || echo "N/A")"
        branch="$(git -C "$resolved_local_path" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "N/A")"
        changed_files="$(git -C "$resolved_local_path" status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
        if [ "${changed_files:-0}" -gt 0 ]; then
            state="dirty"
        else
            state="clean"
        fi
    else
        head_sha="N/A"
        branch="N/A"
        state="not_git_repo"
        changed_files="N/A"
    fi

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$repo_name" \
        "$remote_subdir" \
        "$resolved_local_path" \
        "$head_sha" \
        "$branch" \
        "$state" \
        "$changed_files" >> "$manifest_file"

    echo "[INFO] Repo snapshot: name='$repo_name' remote_subdir='$remote_subdir' sha='$head_sha' branch='$branch' state='$state' changed_files='$changed_files' local_path='$resolved_local_path'"
}

record_repo_sync_manifest() {
    local local_workspace_root
    local manifest_local_file
    local manifest_remote_file
    local local_path
    local remote_subdir
    local local_tmp_dir

    local_workspace_root="$(realpath "$SCRIPT_DIR/../..")"
    local_tmp_dir="$(resolve_local_tmp_dir)"
    manifest_local_file="$(mktemp "${local_tmp_dir}/isaaclab_cluster_repo_manifest.XXXXXX")"
    manifest_remote_file="$CLUSTER_ISAACLAB_DIR/repo_sync_manifest.tsv"

    {
        echo "# Cluster repo sync manifest"
        echo "generated_at_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
        echo "submission_host=$(hostname)"
        echo "cluster_login=$CLUSTER_LOGIN"
        echo "cluster_workspace=$CLUSTER_ISAACLAB_DIR"
        echo
        printf "repo_name\tremote_subdir\tlocal_path\thead_sha\tbranch\tstate\tchanged_files\n"
    } > "$manifest_local_file"

    append_repo_manifest_entry "$manifest_local_file" "IsaacLab-Imitation" "$local_workspace_root" "."

    for spec in $SYNC_EXTRA_REPO_SPECS; do
        local_path="${spec%%:*}"
        remote_subdir="${spec#*:}"
        if [ -z "$local_path" ] || [ -z "$remote_subdir" ]; then
            continue
        fi
        append_repo_manifest_entry "$manifest_local_file" "$remote_subdir" "$local_path" "$remote_subdir"
    done

    ssh "$CLUSTER_LOGIN" "cat > '$manifest_remote_file'" < "$manifest_local_file"
    echo "[INFO] Saved repo sync manifest to '$CLUSTER_LOGIN:$manifest_remote_file'"
    rm -f "$manifest_local_file"
}

submit_job() {
    local -a job_args=("$@")
    local default_manifest_path=""
    local remote_job_cmd=""

    echo "[INFO] Arguments passed to job script ${job_args[*]}"

    case $CLUSTER_JOB_SCHEDULER in
        "SLURM")
            job_script_file=submit_job_slurm.sh
            ;;
        "PBS")
            job_script_file=submit_job_pbs.sh
            ;;
        *)
            echo "[ERROR] Unsupported job scheduler specified: '$CLUSTER_JOB_SCHEDULER'. Supported options are: ['SLURM', 'PBS']"
            exit 1
            ;;
    esac

    if is_truthy "${CLUSTER_APPEND_DEFAULT_G1_MANIFEST:-1}"; then
        default_manifest_path="${CLUSTER_G1_MANIFEST_PATH:-${CLUSTER_G1_DATA_ROOT:-${CLUSTER_DATA_DIR}/lafan1}/manifests/g1_lafan1_manifest.json}"
        if has_arg_with_prefix "env.lafan1_manifest_path=" "${job_args[@]}"; then
            echo "[INFO] Job already specifies env.lafan1_manifest_path; leaving it unchanged."
        else
            job_args+=("env.lafan1_manifest_path=${default_manifest_path}")
            echo "[INFO] Appended default G1 manifest override: env.lafan1_manifest_path=${default_manifest_path}"
        fi
    else
        echo "[INFO] Default G1 manifest override disabled via CLUSTER_APPEND_DEFAULT_G1_MANIFEST='${CLUSTER_APPEND_DEFAULT_G1_MANIFEST:-0}'."
    fi

    printf -v remote_job_cmd '%q ' \
        bash -l "$CLUSTER_ISAACLAB_DIR/docker/cluster/$job_script_file" \
        "$CLUSTER_ISAACLAB_DIR" \
        "isaac-lab-$profile" \
        "${job_args[@]}"

    ssh "$CLUSTER_LOGIN" "cd '$CLUSTER_ISAACLAB_DIR' && ${remote_job_cmd}"
}

sync_extra_repos() {
    local local_workspace_root
    local local_specs
    local local_path
    local remote_subdir

    local_workspace_root="$(realpath "$SCRIPT_DIR/../..")"

    if [ -n "${CLUSTER_EXTRA_SYNC_SPECS:-}" ]; then
        local_specs="$CLUSTER_EXTRA_SYNC_SPECS"
        echo "[INFO] Using CLUSTER_EXTRA_SYNC_SPECS for additional repo sync."
    else
        local_specs="$(build_default_sync_specs "$local_workspace_root")"
    fi
    SYNC_EXTRA_REPO_SPECS="$local_specs"

    if [ -z "$local_specs" ]; then
        echo "[INFO] No extra repo overlays requested; using submodule state from IsaacLab-Imitation."
        return
    fi

    for spec in $local_specs; do
        local_path="${spec%%:*}"
        remote_subdir="${spec#*:}"
        if [ -z "$local_path" ] || [ -z "$remote_subdir" ]; then
            display_warning "Ignoring invalid CLUSTER_EXTRA_SYNC_SPECS entry: '$spec'"
            continue
        fi
        local_path="$(realpath "$local_path" 2>/dev/null || echo "$local_path")"
        sync_repo_prefer_git_then_rsync "$local_path" "$CLUSTER_ISAACLAB_DIR/$remote_subdir" "$remote_subdir" "$remote_subdir"
    done
}

#==
# Main
#==

#!/bin/bash

help() {
    echo -e "\nusage: $(basename "$0") [-h] <command> [<profile>] [<job_args>...] -- Utility for interfacing between IsaacLab and compute clusters."
    echo -e "\noptions:"
    echo -e "  -h              Display this help message."
    echo -e "\ncommands:"
    echo -e "  push [<profile>]              Push the docker image to the cluster."
    echo -e "  job [<profile>] [<job_args>]  Submit a job to the cluster."
    echo -e "\nwhere:"
    echo -e "  <profile>  is the optional container profile specification. Defaults to 'base'."
    echo -e "  <job_args> are optional arguments specific to the job command."
    echo -e "\n" >&2
}

# Parse options
while getopts ":h" opt; do
    case ${opt} in
        h )
            help
            exit 0
            ;;
        \? )
            echo "Invalid option: -$OPTARG" >&2
            help
            exit 1
            ;;
    esac
done
shift $((OPTIND -1))

# Check for command
if [ $# -lt 1 ]; then
    echo "Error: Command is required." >&2
    help
    exit 1
fi

command=$1
shift
profile="base"

case $command in
    push)
        if [ $# -gt 1 ]; then
            echo "Error: Too many arguments for push command." >&2
            help
            exit 1
        fi
        [ $# -eq 1 ] && profile=$1
        echo "Executing push command"
        [ -n "$profile" ] && echo "Using profile: $profile"
        if ! command -v apptainer &> /dev/null; then
            echo "[INFO] Exiting because apptainer was not installed"
            echo "[INFO] You may follow the installation procedure from here: https://apptainer.org/docs/admin/main/installation.html#install-ubuntu-packages"
            exit
        fi
        # Check if Docker image exists
        check_image_exists isaac-lab-$profile:latest
        # Check docker and apptainer version
        check_docker_version
        # source env file to get cluster login and path information
        source $SCRIPT_DIR/.env.cluster
        # Prepend remote $HOME to relative cluster paths.
        CLUSTER_REMOTE_HOME=$(ssh "$CLUSTER_LOGIN" 'echo $HOME')
        CLUSTER_SIF_PATH="$CLUSTER_REMOTE_HOME/$CLUSTER_SIF_PATH"
        # make sure exports directory exists
        mkdir -p /$SCRIPT_DIR/exports
        # clear old exports for selected profile
        rm -rf /$SCRIPT_DIR/exports/isaac-lab-$profile*
        # create singularity image
        # NOTE: we create the singularity image as non-root user to allow for more flexibility. If this causes
        # issues, remove the --fakeroot flag and open an issue on the IsaacLab repository.
        cd /$SCRIPT_DIR/exports
        APPTAINER_NOHTTPS=1 apptainer build --sandbox isaac-lab-$profile.sif docker-daemon://isaac-lab-$profile:latest
        # tar image (faster to send single file as opposed to directory with many files)
        tar -cvf /$SCRIPT_DIR/exports/isaac-lab-$profile.tar isaac-lab-$profile.sif
        # make sure target directory exists
        ssh $CLUSTER_LOGIN "mkdir -p $CLUSTER_SIF_PATH"
        # send image to cluster
        scp $SCRIPT_DIR/exports/isaac-lab-$profile.tar $CLUSTER_LOGIN:$CLUSTER_SIF_PATH/isaac-lab-$profile.tar
        ;;
    job)
        local_workspace_root="$(realpath "$SCRIPT_DIR/../..")"
        if [ $# -ge 1 ]; then
            passed_profile=$1
            if [ -f "$SCRIPT_DIR/../.env.$passed_profile" ]; then
                profile=$passed_profile
                shift
            fi
        fi
        job_args="$@"
        echo "[INFO] Executing job command"
        [ -n "$profile" ] && echo -e "\tUsing profile: $profile"
        [ -n "$job_args" ] && echo -e "\tJob arguments: $job_args"
        source $SCRIPT_DIR/.env.cluster
        # Prepend remote $HOME to relative cluster paths.
        CLUSTER_REMOTE_HOME=$(ssh "$CLUSTER_LOGIN" 'echo $HOME')
        CLUSTER_ISAAC_SIM_CACHE_DIR="$CLUSTER_REMOTE_HOME/$CLUSTER_ISAAC_SIM_CACHE_DIR"
        CLUSTER_ISAACLAB_DIR="$CLUSTER_REMOTE_HOME/$CLUSTER_ISAACLAB_DIR"
        CLUSTER_SIF_PATH="$CLUSTER_REMOTE_HOME/$CLUSTER_SIF_PATH"
        CLUSTER_DATA_DIR="$CLUSTER_REMOTE_HOME/$CLUSTER_DATA_DIR"
        CLUSTER_HF_TOKEN_FILE="$CLUSTER_REMOTE_HOME/$CLUSTER_HF_TOKEN_FILE"
        CLUSTER_WANDB_API_KEY_FILE="$CLUSTER_REMOTE_HOME/$CLUSTER_WANDB_API_KEY_FILE"
        [ -n "${CLUSTER_G1_MANIFEST_PATH:-}" ] && CLUSTER_G1_MANIFEST_PATH="$CLUSTER_REMOTE_HOME/$CLUSTER_G1_MANIFEST_PATH"
        [ -n "${CLUSTER_G1_DATA_ROOT:-}" ] && CLUSTER_G1_DATA_ROOT="$CLUSTER_REMOTE_HOME/$CLUSTER_G1_DATA_ROOT"
        CLUSTER_ISAACLAB_BASE_DIR="$CLUSTER_ISAACLAB_DIR"
        # Get current date and time
        current_datetime=$(date +"%Y%m%d_%H%M%S")
        # Append current date and time to CLUSTER_ISAACLAB_DIR
        CLUSTER_ISAACLAB_DIR="${CLUSTER_ISAACLAB_BASE_DIR}_${current_datetime}"
        init_incremental_sync_state
        # Check if singularity image exists on the remote host
        check_singularity_image_exists isaac-lab-$profile
        # make sure target directory exists
        ssh $CLUSTER_LOGIN "mkdir -p $CLUSTER_ISAACLAB_DIR"
        # Sync Isaac Lab imitation code
        echo "[INFO] Syncing IsaacLab-Imitation code..."
        sync_repo_prefer_git_then_rsync "$local_workspace_root" "$CLUSTER_ISAACLAB_DIR" "IsaacLab-Imitation" "."
        # Sync optional extra repos only when explicitly requested via overrides/specs.
        sync_extra_repos
        # Record exact repo SHAs and dirty state used in this submission.
        record_repo_sync_manifest
        # Refresh latest snapshot pointer used by incremental sync on future submissions.
        update_latest_sync_link
        # execute job script
        echo "[INFO] Executing job script..."
        # check whether the second argument is a profile or a job argument
        submit_job $job_args
        ;;
    *)
        echo "Error: Invalid command: $command" >&2
        help
        exit 1
        ;;
esac
