#!/usr/bin/env bash
# Oracle eval of one ablation arm's tracker on the 30 compositionality motions.
#
# The number of record for the ablation: same protocol as every planner-thread
# row (M3 fall-only, Newton, 2000-step cap, metric_interval 10), encoder-driven
# oracle commands at HOLD 1 — the interface the arms were trained on. The
# environment must declare the arms' anchor frame (`robot_heading`) or the
# checkpoint guard refuses the pairing.
#
# Usage: eval_oracle.sh <arm>
set -euo pipefail
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

ARM="${1:?usage: eval_oracle.sh <arm>}"
ROOT="${REPO_ROOT}/logs/bones129k_latent_quant_ablation/${ARM}"
# Online-dynamics arms train the encoder DURING RL; the finetuned weights are
# saved inside the tracker checkpoint's sampler state, not in the pretrain
# encoder file, and the eval entrypoint does not restore that state. BASE_ARM
# names the pretrain whose config/architecture the finetuned weights share;
# the block below extracts them into a standalone encoder checkpoint so the
# eval pairs the tracker with the encoder it actually trained against.
ENCODER="${BASE_ARM:+${REPO_ROOT}/logs/bones129k_latent_quant_ablation/${BASE_ARM}/encoder/checkpoints/latest.pt}"
ENCODER="${ENCODER:-${ROOT}/encoder/checkpoints/latest.pt}"
TRACKER=$(find "${ROOT}/tracker" -name "model_step_100*.pt" | head -1)
[ -f "${ENCODER}" ] || { echo "missing encoder: ${ENCODER}" >&2; exit 1; }
[ -n "${TRACKER}" ] || { echo "missing 100M tracker under ${ROOT}" >&2; exit 1; }
if [ -n "${BASE_ARM:-}" ]; then
    EXTRACTED="${ROOT}/encoder_finetuned.pt"
    "${REPO_ROOT}/.pixi/envs/default/bin/python" - "$TRACKER" "$ENCODER" "$EXTRACTED" <<'EX'
import sys, torch
tracker, base, out = sys.argv[1:4]
t = torch.load(tracker, map_location="cpu", weights_only=False)
state = t.get("hl_skill_command_sampler_state_dict")
assert state is not None, "tracker checkpoint has no sampler state: not a dyn run?"
b = torch.load(base, map_location="cpu", weights_only=False)
enc = {k.split("skill_encoder.", 1)[1]: v for k, v in state.items()
       if k.startswith("skill_encoder.")} or state.get("skill_encoder_state_dict")
assert enc, f"no encoder weights in sampler state; keys={list(state)[:8]}"
b["skill_encoder_state_dict"] = enc
torch.save(b, out)
print(f"[EXTRACT] finetuned encoder -> {out}")
EX
    ENCODER="${EXTRACTED}"
fi

DATA_ROOT="${REPO_ROOT}/data/bones_seed_language30_compositionality_v1"
MANIFEST_LANG="${DATA_ROOT}/manifests/g1_bones_seed_language30_compositionality_v1_manifest_language.json"
readarray -t MOTIONS < <(
    "${REPO_ROOT}/.pixi/envs/default/bin/python" - "${MANIFEST_LANG}" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1]))
motions = manifest["motions"] if "motions" in manifest else manifest
print("\n".join(m["name"] for m in motions))
PY
)
RANKS=($(seq 0 $((${#MOTIONS[@]} - 1))))
EPISODES_PER_GOAL="${EPISODES_PER_GOAL:-5}"
NUM_ENVS=$(( ${#MOTIONS[@]} * EPISODES_PER_GOAL ))

OUTPUT_DIR="${REPO_ROOT}/logs/bones129k_latent_quant_ablation/oracle_eval/${ARM}"
if [ -e "${OUTPUT_DIR}" ]; then
    echo "Refusing to overwrite existing ${OUTPUT_DIR}" >&2
    exit 1
fi

pixi run -e isaaclab python scripts/rlopt/eval_skill_commander_closed_loop.py \
    --headless --task Isaac-Imitation-G1-v2 --algorithm IPMD \
    --agent_entry_point rlopt_ipmd_tuned_cfg_entry_point \
    --checkpoint "${TRACKER}" --skill_checkpoint "${ENCODER}" \
    --language_embeddings "${DATA_ROOT}/language/g1_bones_seed_language30_compositionality_v1_minilm_goal_embeddings.pt" \
    --state_history_steps 9 --output_dir "${OUTPUT_DIR}" --label "quant_${ARM}_oracle" \
    --num_envs "${NUM_ENVS}" --max_steps "${MAX_STEPS:-2000}" --seed "${SEED:-0}" \
    --metric_interval 10 \
    --motion_names "${MOTIONS[@]}" --trajectory_ranks "${RANKS[@]}" \
    --disable_push_event --disable_reward_clipping --assert-kitless \
    --disable_tracking_terminations --fall_only_success \
    physics=newton_mjwarp env.data.manifest=null env.data.cache_dir=null \
    env.data.reference_arrays_dir="${DATA_ROOT}/reference_arrays/root_qpos_v1" \
    env.data.persist_id=bones_seed_language30_compositionality_v1@f31fd755 \
    env.data.persist_dir=null env.data.macro_cache_device=cuda:0 \
    env.data.wrap_steps=false \
    'env.data.runtime_cache_body_names=[pelvis,left_hip_roll_link,left_knee_link,left_ankle_roll_link,right_hip_roll_link,right_knee_link,right_ankle_roll_link,torso_link,left_shoulder_roll_link,left_elbow_link,left_wrist_yaw_link,right_shoulder_roll_link,right_elbow_link,right_wrist_yaw_link]' \
    env.command_interface.actor=latent \
    env.command_interface.actor.dim=64 \
    env.command_interface.encoder=single \
    'env.expert_macro_state_terms=[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b]' \
    "env.expert_macro_frame_stride=${STRIDE:-1}" \
    env.expert_macro_anchor_mode=robot_heading \
    agent.logger.backend= agent.ipmd.command_source=hl_skill \
    agent.ipmd.hl_skill_checkpoint_path="${ENCODER}" \
    agent.ipmd.hl_skill_finetune_enabled=false \
    agent.ipmd.latent_dim=64 \
    agent.ipmd.latent_steps_min=1 agent.ipmd.latent_steps_max=1 \
    agent.ipmd.hl_skill_horizon_steps=10 \
    agent.ipmd.hl_skill_command_mode=z \
    agent.ipmd.latent_learning.command_phase_mode=none \
    agent.ipmd.latent_learning.code_latent_dim=64 \
    agent.ipmd.latent_learning.code_period=1 \
    'agent.policy.num_cells=[2048,2048,1024,1024,512,512]' \
    'agent.value_function.num_cells=[2048,2048,1024,1024,512,512]' \
    agent.policy.activation_fn=silu agent.value_function.activation_fn=silu \
    env.sim.physics.solver_cfg.njmax=320 \
    env.sim.physics.solver_cfg.nconmax=200

echo "retained: ${OUTPUT_DIR}"
