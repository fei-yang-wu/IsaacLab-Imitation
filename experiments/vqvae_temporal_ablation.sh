#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
VARIANT="${2:-all}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
RLOPT_VQVAE_DIR="${RLOPT_VQVAE_DIR:-${WORKSPACE_ROOT}/RLOpt-vqvae}"

TASK="Isaac-Imitation-G1-Latent-VQVAE-v0"
MANIFEST="./data/unitree/manifests/g1_unitree_dance102_manifest.json"
DATASET="/tmp/iltools_g1_lafan1_tracking_g1_unitree_dance102_manifest_6d26546fd54a"
CONDA_ENV="${CONDA_ENV:-SL}"

VARIANTS=(
    vqvae_p10_d64
    vqvae_p10_phase_d66
    vqvae_p10_identity_d64
    vqvae_p5_d64
    vqvae_p10_fsq8_d64
)

COMMON_OVERRIDES=(
    "env.lafan1_manifest_path=${MANIFEST}"
    "env.dataset_path=${DATASET}"
    "env.refresh_zarr_dataset=False"
)

usage() {
    cat <<EOF
Usage: $0 <local|cluster|print> <variant|all>

Variants:
  ${VARIANTS[*]}
EOF
}

variant_overrides() {
    local variant="$1"
    OVERRIDES=()
    case "${variant}" in
        vqvae_p10_d64)
            OVERRIDES=(
                "agent.ipmd.latent_steps_min=10"
                "agent.ipmd.latent_steps_max=10"
                "agent.ipmd.latent_learning.code_period=10"
                "agent.ipmd.latent_learning.quantizer=fsq"
                "agent.ipmd.latent_learning.command_phase_mode=none"
                "agent.ipmd.latent_dim=64"
                "agent.ipmd.latent_learning.code_latent_dim=64"
                "env.latent_command_dim=64"
            )
            ;;
        vqvae_p10_phase_d66)
            OVERRIDES=(
                "agent.ipmd.latent_steps_min=10"
                "agent.ipmd.latent_steps_max=10"
                "agent.ipmd.latent_learning.code_period=10"
                "agent.ipmd.latent_learning.quantizer=fsq"
                "agent.ipmd.latent_learning.command_phase_mode=sin_cos"
                "agent.ipmd.latent_dim=66"
                "agent.ipmd.latent_learning.code_latent_dim=64"
                "env.latent_command_dim=66"
            )
            ;;
        vqvae_p10_identity_d64)
            OVERRIDES=(
                "agent.ipmd.latent_steps_min=10"
                "agent.ipmd.latent_steps_max=10"
                "agent.ipmd.latent_learning.code_period=10"
                "agent.ipmd.latent_learning.quantizer=identity"
                "agent.ipmd.latent_learning.command_phase_mode=none"
                "agent.ipmd.latent_dim=64"
                "agent.ipmd.latent_learning.code_latent_dim=64"
                "env.latent_command_dim=64"
            )
            ;;
        vqvae_p5_d64)
            OVERRIDES=(
                "agent.ipmd.latent_steps_min=5"
                "agent.ipmd.latent_steps_max=5"
                "agent.ipmd.latent_learning.code_period=5"
                "agent.ipmd.latent_learning.quantizer=fsq"
                "agent.ipmd.latent_learning.command_phase_mode=none"
                "agent.ipmd.latent_dim=64"
                "agent.ipmd.latent_learning.code_latent_dim=64"
                "env.latent_command_dim=64"
            )
            ;;
        vqvae_p10_fsq8_d64)
            OVERRIDES=(
                "agent.ipmd.latent_steps_min=10"
                "agent.ipmd.latent_steps_max=10"
                "agent.ipmd.latent_learning.code_period=10"
                "agent.ipmd.latent_learning.quantizer=fsq"
                "agent.ipmd.latent_learning.fsq_levels=[8,8,8,8,5,5,5,5]"
                "agent.ipmd.latent_learning.command_phase_mode=none"
                "agent.ipmd.latent_dim=64"
                "agent.ipmd.latent_learning.code_latent_dim=64"
                "env.latent_command_dim=64"
            )
            ;;
        *)
            echo "[ERROR] Unknown variant '${variant}'." >&2
            usage >&2
            exit 2
            ;;
    esac
}

run_local() {
    local variant="$1"
    local local_num_envs="${LOCAL_NUM_ENVS:-1024}"
    local save_interval=$((local_num_envs * 24))
    (
        cd "${REPO_ROOT}"
        TERM=xterm PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 TORCHDYNAMO_DISABLE=1 \
            conda run -n "${CONDA_ENV}" python scripts/rlopt/train.py \
            --task "${TASK}" \
            --num_envs "${local_num_envs}" \
            --headless \
            --video \
            --video_length "${LOCAL_VIDEO_LENGTH:-24}" \
            --video_interval "${LOCAL_VIDEO_INTERVAL:-1000000}" \
            --algo IPMD \
            --max_iterations "${LOCAL_MAX_ITERATIONS:-1}" \
            --log_interval "${LOCAL_LOG_INTERVAL:-1000}" \
            --kit_args=--/app/extensions/fsWatcherEnabled=false \
            "${COMMON_OVERRIDES[@]}" \
            "agent.logger.exp_name=${variant}_smoke_${local_num_envs}" \
            "agent.save_interval=${save_interval}" \
            "${OVERRIDES[@]}"
    )
}

run_cluster() {
    local variant="$1"
    CLUSTER_EXTRA_SYNC_SPECS="${CLUSTER_EXTRA_SYNC_SPECS:-${RLOPT_VQVAE_DIR}:RLOpt}" \
        "${REPO_ROOT}/docker/cluster/cluster_interface.sh" job \
        --task "${TASK}" \
        --num_envs "${CLUSTER_NUM_ENVS:-4096}" \
        --headless \
        --video \
        --algo IPMD \
        --kit_args=--/app/extensions/fsWatcherEnabled=false \
        "${COMMON_OVERRIDES[@]}" \
        "agent.logger.exp_name=${variant}" \
        "${OVERRIDES[@]}"
}

print_variant() {
    local variant="$1"
    printf '%s\n' "${variant}"
    printf '  %s\n' "${COMMON_OVERRIDES[@]}" "${OVERRIDES[@]}"
}

run_one() {
    local variant="$1"
    variant_overrides "${variant}"
    case "${MODE}" in
        local)
            run_local "${variant}"
            ;;
        cluster)
            run_cluster "${variant}"
            ;;
        print)
            print_variant "${variant}"
            ;;
        *)
            usage >&2
            exit 2
            ;;
    esac
}

if [[ -z "${MODE}" ]]; then
    usage >&2
    exit 2
fi

if [[ "${VARIANT}" == "all" ]]; then
    for variant in "${VARIANTS[@]}"; do
        run_one "${variant}"
    done
else
    run_one "${VARIANT}"
fi
