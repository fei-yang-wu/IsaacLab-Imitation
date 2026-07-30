#!/usr/bin/env bash
set -euo pipefail

# Fixed local evaluation matching the 2026-07-29 Stable 500M diagnostic.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
while [ ! -x "${REPO_ROOT}/docker/cluster/cluster_interface.sh" ]; do
    if [ "${REPO_ROOT}" = "/" ]; then
        echo "[ERROR] Could not locate repository root above ${SCRIPT_DIR}." >&2
        exit 2
    fi
    REPO_ROOT="$(dirname "${REPO_ROOT}")"
done
cd "${REPO_ROOT}"

CHECKPOINT="${CHECKPOINT:-}"
SKILL_CHECKPOINT="${SKILL_CHECKPOINT:-logs/downloaded_checkpoints/lafan1_stable_vs_strict_500m_20260729/skill_encoder/latest.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/interface_baselines/lafan1_stable_5b_converged_20260729}"
MANIFEST_PATH="${MANIFEST_PATH:-data/lafan1/manifests/g1_lafan1_manifest.json}"
DATASET_PATH="${DATASET_PATH:-/tmp/iltools_g1_lafan1_tracking_corrected_8029acbce33a}"
EXPECTED_MANIFEST_SHA256="d972c37c41dadbb68c30fc456a9dc9c1bd6d30ed0b7aa9d34b1797472c945db8"
EXPECTED_ENCODER_SHA256="5c84ff7261c5a3aca732e370ca39f889d68a5d39fb498fa9fde72c653eb264ea"

if [ -z "${CHECKPOINT}" ] || [ ! -s "${CHECKPOINT}" ]; then
    echo "[ERROR] Set CHECKPOINT to a non-empty local Stable policy checkpoint." >&2
    exit 2
fi
if [ ! -s "${SKILL_CHECKPOINT}" ]; then
    echo "[ERROR] Skill checkpoint missing: ${SKILL_CHECKPOINT}" >&2
    exit 2
fi
if [ ! -d "${DATASET_PATH}" ]; then
    echo "[ERROR] Corrected LAFAN1 cache missing: ${DATASET_PATH}" >&2
    exit 2
fi
manifest_sha="$(sha256sum "${MANIFEST_PATH}" | awk '{print $1}')"
encoder_sha="$(sha256sum "${SKILL_CHECKPOINT}" | awk '{print $1}')"
if [ "${manifest_sha}" != "${EXPECTED_MANIFEST_SHA256}" ]; then
    echo "[ERROR] Manifest hash mismatch: ${manifest_sha}" >&2
    exit 2
fi
if [ "${encoder_sha}" != "${EXPECTED_ENCODER_SHA256}" ]; then
    echo "[ERROR] Encoder hash mismatch: ${encoder_sha}" >&2
    exit 2
fi
if [ -e "${OUTPUT_ROOT}" ]; then
    echo "[ERROR] Refusing existing OUTPUT_ROOT: ${OUTPUT_ROOT}" >&2
    exit 2
fi

frame_label="$(basename "${CHECKPOINT}" .pt)"
strict_dir="${OUTPUT_ROOT}/strict_terminations"
full_dir="${OUTPUT_ROOT}/full_horizon_deterministic"
common=(
    --headless
    --assert-kitless
    --device cuda:0
    --task Isaac-Imitation-G1-Latent-v0
    --algorithm IPMD
    --checkpoint "${CHECKPOINT}"
    --skill_checkpoint "${SKILL_CHECKPOINT}"
    --max_steps 1000
    --seed 0
    --metric_interval 1
    --keep_time_out
    --extend_episode_length_for_max_steps
    --disable_reward_clipping
    --kit_args=--/app/extensions/fsWatcherEnabled=false
    agent.logger.backend=
    agent.ipmd.command_source=hl_skill
    "agent.ipmd.hl_skill_checkpoint_path=${SKILL_CHECKPOINT}"
    agent.ipmd.hl_skill_finetune_enabled=false
    "env.lafan1_manifest_path=${MANIFEST_PATH}"
    "env.dataset_path=${DATASET_PATH}"
    env.refresh_zarr_dataset=false
    env.reset_schedule=sequential
    env.wrap_steps=false
    env.observations.policy.enable_corruption=false
    env.latent_command_dim=258
    agent.ipmd.latent_dim=258
    agent.ipmd.hl_skill_horizon_steps=10
    agent.ipmd.hl_skill_command_mode=z
    agent.ipmd.latent_steps_min=10
    agent.ipmd.latent_steps_max=10
    agent.ipmd.latent_learning.command_phase_mode=sin_cos
    agent.ipmd.latent_learning.code_latent_dim=256
    agent.ipmd.latent_learning.code_period=10
    agent.ipmd.reward_loss_coeff=0.0
    agent.ipmd.reward_l2_coeff=0.0
    agent.ipmd.reward_grad_penalty_coeff=0.0
    agent.ipmd.reward_logit_reg_coeff=0.0
    agent.ipmd.reward_param_weight_decay_coeff=0.0
    physics=newton_mjwarp
    env.sim.physics.solver_cfg.njmax=320
    env.sim.physics.solver_cfg.nconmax=40
)

mkdir -p "${OUTPUT_ROOT}"
{
    printf "checkpoint="
    realpath "${CHECKPOINT}"
    printf "checkpoint_sha256="
    sha256sum "${CHECKPOINT}" | awk '{print $1}'
    printf "skill_checkpoint="
    realpath "${SKILL_CHECKPOINT}"
    printf "skill_checkpoint_sha256=%s\n" "${encoder_sha}"
    printf "manifest_sha256=%s\n" "${manifest_sha}"
} > "${OUTPUT_ROOT}/evaluation_provenance.txt"

pixi run -e isaaclab python scripts/rlopt/eval_skill_commander_closed_loop.py \
    "${common[@]}" \
    --num_envs 100 \
    --keep_early_terminations \
    --output_dir "${strict_dir}" \
    --label "stable_5b_${frame_label}_strict_terminations"

pixi run -e isaaclab python scripts/rlopt/eval_skill_commander_closed_loop.py \
    "${common[@]}" \
    --num_envs 40 \
    --video \
    --video_length 1000 \
    --deterministic_tracking \
    --output_dir "${full_dir}" \
    --label "stable_5b_${frame_label}_full_horizon_deterministic"

echo "[RESULT] strict summary: $(realpath "${strict_dir}/summary.json")"
echo "[RESULT] full-horizon summary: $(realpath "${full_dir}/summary.json")"
echo "[RESULT] retained video: $(realpath "${full_dir}/videos/play/rl-video-step-0.mp4")"
