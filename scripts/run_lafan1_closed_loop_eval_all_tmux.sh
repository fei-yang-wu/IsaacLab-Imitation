#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

SESSION="${SESSION:-lafan1-closed-loop-eval}"
TASK="${TASK:-Isaac-Imitation-G1-Latent-v0}"
ALGORITHM="${ALGORITHM:-IPMD}"
CHECKPOINT="${CHECKPOINT:-logs/rlopt/ipmd/Isaac-Imitation-G1-Latent-v0/2026-06-11_23-21-31/models/model_step_4600037376.pt}"
MANIFEST="${MANIFEST:-/mnt/hsstorage/fwu91/Projects/SL/IsaacLab-Imitation/data/lafan1/manifests/g1_lafan1_manifest.json}"
DATASET_PATH="${DATASET_PATH:-/mnt/hsstorage/fwu91/Projects/SL/IsaacLab-Imitation/data/lafan1/g1_hl_diffsr}"
SKILL_COMMANDER_CHECKPOINT="${SKILL_COMMANDER_CHECKPOINT:-logs/language_skill_generator/2026-06-17_matching_hl_skill_20260611_161049/checkpoints/latest.pt}"
LANGUAGE_EMBEDDINGS="${LANGUAGE_EMBEDDINGS:-/mnt/hsstorage/fwu91/Projects/SL/IsaacLab-Imitation/data/lafan1/language/g1_lafan1_name_embeddings.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/rlopt_eval/compare_policy_reference_all/${TASK}/2026-06-18_body_markers_framed_all_lafan1_10s}"
RANKS="${RANKS:-all}"
VIDEO_SECONDS="${VIDEO_SECONDS:-10}"
POLICY_START_STEP="${POLICY_START_STEP:-0}"
REFERENCE_VISUALIZATION="${REFERENCE_VISUALIZATION:-body_markers}"

if [[ "${OUTPUT_ROOT}" != /* ]]; then
  OUTPUT_ROOT="${REPO_ROOT}/${OUTPUT_ROOT}"
fi
mkdir -p "${OUTPUT_ROOT}"
LOG_FILE="${LOG_FILE:-${OUTPUT_ROOT}/tmux_eval.log}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session '${SESSION}' already exists. Attach with: tmux attach -t ${SESSION}" >&2
  exit 1
fi

cmd=(
  env OMNI_KIT_ACCEPT_EULA=YES
  pixi run -e isaaclab python scripts/compare_policy_reference_all.py
  --task "${TASK}"
  --algorithm "${ALGORITHM}"
  --checkpoint "${CHECKPOINT}"
  --manifest "${MANIFEST}"
  --ranks "${RANKS}"
  --video_seconds "${VIDEO_SECONDS}"
  --policy_start_step "${POLICY_START_STEP}"
  --output_root "${OUTPUT_ROOT}"
  --continue_on_error
  --resume
  --reference_visualization "${REFERENCE_VISUALIZATION}"
  --kit_args=--/app/extensions/fsWatcherEnabled=false
  env.lafan1_manifest_path="${MANIFEST}"
  env.dataset_path="${DATASET_PATH}"
  env.latent_command_dim=386
  agent.ipmd.command_source=skill_commander
  agent.ipmd.skill_commander_checkpoint_path="${SKILL_COMMANDER_CHECKPOINT}"
  agent.ipmd.skill_commander_embeddings_path="${LANGUAGE_EMBEDDINGS}"
  agent.ipmd.skill_commander_use_achieved_state=true
  agent.ipmd.latent_dim=386
  agent.ipmd.latent_steps_min=25
  agent.ipmd.latent_steps_max=25
  agent.ipmd.hl_skill_horizon_steps=25
  agent.ipmd.hl_skill_command_mode=z_phi
  agent.ipmd.latent_learning.command_phase_mode=sin_cos
  agent.ipmd.latent_learning.code_latent_dim=384
  agent.ipmd.latent_learning.code_period=25
)

printf -v quoted_cmd '%q ' "${cmd[@]}"
printf -v quoted_repo '%q' "${REPO_ROOT}"
printf -v quoted_log '%q' "${LOG_FILE}"

{
  echo "# Started: $(date -Is)"
  echo "# Session: ${SESSION}"
  echo "# Output: ${OUTPUT_ROOT}"
  echo "${quoted_cmd}"
} >> "${LOG_FILE}"

tmux new-session -d -s "${SESSION}" "cd ${quoted_repo} && ${quoted_cmd} 2>&1 | tee -a ${quoted_log}"

echo "Started tmux session: ${SESSION}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Log file: ${LOG_FILE}"
echo "Attach: tmux attach -t ${SESSION}"
echo "Tail: tail -f ${LOG_FILE}"
